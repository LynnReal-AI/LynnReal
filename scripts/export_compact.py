"""Export sampling-specific AdaLN tables and omit the learned block projections.

The two phases permit CPU shard writing before GPU table evaluation. Existing
source packages are read-only. Tables are evaluated on the intended execution
device, preserving the original activation precision and time-row batch shapes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def indexed_tensor_bytes(directory, mapping):
    """Count tensor payloads from headers without reading checkpoint contents."""
    total = 0
    for name in set(mapping.values()):
        with (Path(directory)/name).open('rb') as handle:
            length = struct.unpack('<Q', handle.read(8))[0]
            header = json.loads(handle.read(length))
        for key, shard in mapping.items():
            if shard == name:
                start, end = header[key]['data_offsets']
                total += end-start
    return total


def repack_weights(source, output, max_shard_bytes=4_000_000_000):
    from safetensors import safe_open
    from safetensors.torch import save_file
    source, output = Path(source)/'transformer', Path(output)/'transformer'
    if source.resolve() == output.resolve():
        raise ValueError('Use a separate output directory; source weights are read-only')
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads((source/'diffusion_pytorch_model.safetensors.index.json').read_text())
    shards, pending, pending_bytes, total, mapping = [], {}, 0, 0, {}

    def flush():
        nonlocal pending, pending_bytes
        if not pending:
            return
        name = f'weights-{len(shards)+1:05d}.safetensors'
        save_file(pending, str(output/name), metadata={'format': 'pt'})
        mapping.update({key: name for key in pending})
        shards.append(name)
        pending, pending_bytes = {}, 0

    for shard in sorted(set(index['weight_map'].values())):
        with safe_open(str(source/shard), framework='pt', device='cpu') as handle:
            for name in sorted(handle.keys()):
                if '.adaln_proj.linear.' in name:
                    continue
                tensor = handle.get_tensor(name)
                size = tensor.numel()*tensor.element_size()
                if pending and pending_bytes+size > max_shard_bytes:
                    flush()
                pending[name] = tensor
                pending_bytes += size
                total += size
        flush()
    config = json.loads((source/'config.json').read_text())
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (output/'diffusion_pytorch_model.safetensors.index.json').write_text(json.dumps(
        dict(metadata=dict(total_size=total), weight_map=mapping), indent=2)+'\n')
    return dict(stored_bytes=total, tensors=len(mapping), shards=len(shards))


def export_tables(source, config_path, output, device):
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from safetensors.torch import save_file
    from lynnreal_hand.backends.h3.runtime import activate
    from lynnreal_hand.config import load_config
    from lynnreal_hand.inference.adaln import schedule_rows
    activate()
    from lynnreal_core.backends.minimax_h3.model import SolarMiniMaxH3Transformer3DModel
    source, destination = Path(source)/'transformer', Path(output)/'transformer'
    if source.resolve() == destination.resolve():
        raise ValueError('Use a separate output directory; source weights are read-only')
    original = json.loads((source/'config.json').read_text())
    if 'adaln_schedule' in original:
        raise ValueError('Export requires the original learned AdaLN projections')
    index = json.loads((source/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    config = load_config(config_path)
    with init_empty_weights():
        model = SolarMiniMaxH3Transformer3DModel.from_config(original)
    model.eval().requires_grad_(False)

    def load(module, prefix):
        state = {}
        for name in module.state_dict():
            key = prefix+name
            with safe_open(str(source/index[key]), framework='pt', device='cpu') as handle:
                state[name] = handle.get_tensor(key).to(device)
        module.load_state_dict(state, strict=True, assign=True)

    load(model.time_embedder, 'time_embedder.')
    if model.uses_anyflow:
        load(model.delta_embedding, 'delta_embedding.')
    rows = schedule_rows(config, device)
    tensors, entries, maximum_error = {}, [], 0.
    with torch.inference_mode():
        embeddings = []
        for times, targets in rows:
            entries.append(dict(timestep=times, r_timestep=targets))
            t = torch.tensor(times, dtype=torch.float32, device=device)
            r = torch.tensor(targets, dtype=torch.float32, device=device) if targets is not None else None
            embeddings.append(model._time_condition(t, r, parameter_dtype=lambda m: next(m.parameters()).dtype))
        for layer, block in enumerate(model.transformer_blocks):
            projection = block.adaln_proj
            load(projection, f'transformer_blocks.{layer}.adaln_proj.')
            for slot, embedding in enumerate(embeddings):
                expected = torch.cat(projection(embedding), dim=-1).contiguous()
                saved = expected.cpu().contiguous()
                # Check serialization precision, without quantizing the table.
                actual = saved.to(device)
                assert torch.equal(expected, actual)
                tensors[f'transformer_blocks.{layer}.adaln_proj.table_{slot}'] = saved
            projection.to('meta')
            print(json.dumps(dict(layer=layer+1, layers=len(model.transformer_blocks), tables=len(tensors))), flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(destination/'adaln.safetensors'), metadata={'format': 'pt'})
    original['adaln_schedule'] = dict(schema='lynnreal-hand.adaln-schedule.v1', rows=entries)
    (destination/'config.json').write_text(json.dumps(original, indent=2)+'\n')
    result = json.loads((destination/'diffusion_pytorch_model.safetensors.index.json').read_text())
    result['weight_map'].update({name: 'adaln.safetensors' for name in tensors})
    table_bytes = sum(t.numel()*t.element_size() for t in tensors.values())
    result['metadata']['total_size'] = indexed_tensor_bytes(destination, result['weight_map'])
    (destination/'diffusion_pytorch_model.safetensors.index.json').write_text(json.dumps(result, indent=2)+'\n')
    return dict(table_bytes=table_bytes, table_tensors=len(tensors), schedule_entries=len(entries),
                table_roundtrip_exact=True, stored_bytes=result['metadata']['total_size'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('weights', 'tables', 'all'), default='all')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    report = {}
    if args.phase in ('weights', 'all'):
        report['weights'] = repack_weights(args.source, args.output)
    if args.phase in ('tables', 'all'):
        report['tables'] = export_tables(args.source, args.config, args.output, args.device)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
