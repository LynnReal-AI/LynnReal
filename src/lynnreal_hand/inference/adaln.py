"""Specialize inference AdaLN projections to the selected sampling schedule.

AdaLN depends on time conditions, independently of scene tokens and CFG. Once
the schedule is fixed, retain its exact modulation outputs instead of keeping
every block's large learned projection on the device. Tables can be loaded from
a compact package or computed from a full model at initialization.
"""
from __future__ import annotations

import struct

import torch
from torch import nn

from .sampling import shift_sigma


def _key(times, targets=None):
    if isinstance(times, torch.Tensor):
        times = times.detach().float().cpu().tolist()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().float().cpu().tolist()
    values = tuple(struct.pack('<f', float(t)) for t in times)
    if targets is None:
        return values, None
    target_values = tuple(struct.pack('<f', float(t)) for t in targets)
    if len(values) != len(target_values):
        raise ValueError('AdaLN time and target-time rows must match')
    return values, target_values


def schedule_rows(config, device):
    """Include sampling transitions and the exact conditioning boundary rows."""
    sampling = config['sampling']
    if config['stage'] == 'stage1':
        sigmas = shift_sigma(torch.linspace(1., 0., sampling['schedule_points'], device=device),
                             sampling.get('shift', 3.))
        return [(torch.tensor([1. - float(sigma), .999], dtype=torch.float32)
                 .unique(sorted=True).tolist(), None) for sigma in sigmas[:-1]]
    if config['stage'] != 'stage3':
        raise ValueError('AdaLN specialization supports Base and Fast inference')
    sigmas = shift_sigma(torch.linspace(1., 0., sampling['steps'] + 1, device=device),
                         sampling.get('shift', 3.)).tolist()
    return [([.999, 1.], [.999, 1.]), ([1.], [1.])] + [
        ([1. - start], [1. - end]) for start, end in zip(sigmas[:-1], sigmas[1:])]


class ScheduledModulation(nn.Module):
    """Tables for one block, selected before model execution."""

    def __init__(self, tables, *, persistent=False):
        super().__init__()
        for index, table in enumerate(tables):
            self.register_buffer(f'table_{index}', table, persistent=persistent)
        self.active_index = -1

    def forward(self, temb):
        if self.active_index < 0:
            raise RuntimeError('Select the scheduled AdaLN rows before inference')
        table = getattr(self, f'table_{self.active_index}')
        if table.shape[0] != temb.shape[0] * 3:
            raise ValueError('The active AdaLN table does not match the time rows')
        return table.chunk(6, dim=-1)


def install_packaged_adaln(model, specification):
    """Construct table buffers before strict loading, including meta loading."""
    if specification.get('schema') != 'lynnreal-hand.adaln-schedule.v1':
        raise ValueError('Unsupported packaged AdaLN schedule')
    rows = specification.get('rows', [])
    if not rows:
        raise ValueError('A packaged AdaLN schedule must contain time conditions')
    indices = {}
    for index, row in enumerate(rows):
        times, targets = row['timestep'], row.get('r_timestep')
        key = _key(times, targets)
        if not times or key in indices:
            raise ValueError('Invalid or repeated packaged AdaLN time conditions')
        indices[key] = index
    modules = []
    for block in model.transformer_blocks:
        original = block.adaln_proj
        tables = [torch.empty((len(row['timestep']) * 3, original.hidden_size * 6),
                              device=original.linear.weight.device, dtype=torch.bfloat16)
                  for row in rows]
        replacement = ScheduledModulation(tables, persistent=True)
        block.adaln_proj = replacement
        modules.append(replacement)
    model._scheduled_adaln = dict(indices=indices, modules=modules)


def validate_packaged_schedule(model, config, device):
    """Reject sampler changes that require AdaLN values absent from the package."""
    expected = {_key(t, r) for t, r in schedule_rows(config, device)}
    available = set(model._scheduled_adaln['indices'])
    if expected != available:
        raise ValueError('The inference sampling schedule differs from the compact model package')
    tables = [buffer for module in model._scheduled_adaln['modules'] for buffer in module.buffers()]
    return dict(schedule_entries=len(available), modulation_table_bytes=sum(
        table.numel()*table.element_size() for table in tables), packaged=True)


@torch.inference_mode()
def specialize_adaln(model, config, device):
    """Precompute on the execution device, then release block projections.

    This reduces resident parameter memory after initialization. Initial model
    loading still uses the complete transformer; this is not a lower peak-memory
    loader or a claim of unchanged CUDA numerical results on an untested device.
    """
    if hasattr(model, '_scheduled_adaln'):
        raise ValueError('AdaLN has already been specialized')
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('AdaLN specialization requires a frozen inference model')
    rows, embeddings, indices = [], [], {}
    parameter_dtype = lambda module: next(module.parameters()).dtype
    for times, targets in schedule_rows(config, device):
        key = _key(times, targets)
        if key in indices:
            continue
        indices[key] = len(rows)
        rows.append((times, targets))
        timestep = torch.tensor(times, dtype=torch.float32, device=device)
        target = (torch.tensor(targets, dtype=torch.float32, device=device)
                  if targets is not None else None)
        embeddings.append(model._time_condition(timestep, target, parameter_dtype=parameter_dtype))
    projections, original_bytes, table_bytes = [], 0, 0
    for block in model.transformer_blocks:
        projection = block.adaln_proj
        original_bytes += sum(p.numel() * p.element_size() for p in projection.parameters())
        tables = [torch.cat(projection(embedding), dim=-1).contiguous() for embedding in embeddings]
        table_bytes += sum(table.numel() * table.element_size() for table in tables)
        replacement = ScheduledModulation(tables)
        block.adaln_proj = replacement
        projections.append(replacement)
    # Keep only runtime lookup state, without registering the modules twice.
    model._scheduled_adaln = dict(indices=indices, modules=projections)
    return dict(schedule_entries=len(rows), original_projection_bytes=original_bytes,
                modulation_table_bytes=table_bytes, resident_bytes_removed=original_bytes-table_bytes)


def select_adaln(model, times, targets=None):
    """Select known time rows outside compiled forwards and CUDA captures.

    CUDA capture callers supply Python row lists, avoiding device-to-host reads.
    Unknown schedules fail explicitly and must be specialized again.
    """
    state = getattr(model, '_scheduled_adaln', None)
    if state is None:
        return
    key = _key(times, targets)
    if key not in state['indices']:
        raise ValueError('Time conditions are outside the specialized AdaLN schedule')
    index = state['indices'][key]
    for module in state['modules']:
        module.active_index = index
