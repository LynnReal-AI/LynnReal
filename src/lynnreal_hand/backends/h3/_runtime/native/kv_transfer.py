"""Bounded exact KV copies; no changes to attention math or detach semantics."""
from concurrent.futures import ThreadPoolExecutor
import time
import torch


class ArchiveWriter:
    """Overlap D2H and pageable CPU archival using two pinned staging slots."""
    def __init__(self, device, slots=2):
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='duo-kv-archive')
        self.slots = [dict(buffers=[], future=None) for _ in range(slots)]
        self.index = 0
        self.closed = False
        self.wait_seconds = 0.
        self.flush_seconds = 0.
        self.staging_peak_bytes = 0

    @staticmethod
    def _drain(event, destinations, staging):
        event.synchronize()
        for destination, source in zip(destinations, staging):
            destination.copy_(source)

    def submit(self, pair):
        assert not self.closed and all(x.is_contiguous() for x in pair)
        slot = self.slots[self.index % len(self.slots)]
        self.index += 1
        if slot['future'] is not None:
            start = time.monotonic()
            slot['future'].result()
            self.wait_seconds += time.monotonic() - start
            slot['future'] = None
        if len(slot['buffers']) != len(pair) or any(
            a.dtype != b.dtype or a.numel() < b.numel()
            for a, b in zip(slot['buffers'], pair)
        ):
            slot['buffers'] = [torch.empty(x.numel(), dtype=x.dtype, device='cpu', pin_memory=True) for x in pair]
        staging = [a[:b.numel()].view(b.shape) for a, b in zip(slot['buffers'], pair)]
        destinations = tuple(torch.empty_like(x, device='cpu', memory_format=torch.contiguous_format) for x in pair)
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            for source, target in zip(pair, staging):
                target.copy_(source.detach(), non_blocking=True)
                source.record_stream(self.stream)
            event = torch.cuda.Event()
            event.record(self.stream)
        slot['future'] = self.executor.submit(self._drain, event, destinations, staging)
        self.staging_peak_bytes = max(self.staging_peak_bytes, sum(x.numel()*x.element_size() for s in self.slots for x in s['buffers']))
        return destinations

    def finish(self):
        if self.closed:
            return
        start = time.monotonic()
        try:
            for slot in self.slots:
                if slot['future'] is not None:
                    slot['future'].result()
            self.stream.synchronize()
        finally:
            self.executor.shutdown(wait=True)
            self.slots.clear()
            self.closed = True
            self.flush_seconds += time.monotonic() - start


class ReplayPrefetch:
    """Assemble raw CPU KV and transfer the next layer while this layer computes."""
    def __init__(self, cache, index, device, layers=50):
        self.cache, self.index, self.device, self.layers = cache, index, device, layers
        self.stream = torch.cuda.Stream(device=device)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='duo-kv-prefetch')
        self.pending = {}
        self.slots = [dict(buffers=None, done=None) for _ in range(2)]
        self.closed = False
        self.wait_seconds = 0.
        self.staging_peak_bytes = 0
        self.schedule(0)

    def _prepare(self, layer):
        slot = self.slots[layer % 2]
        if slot['done'] is not None:
            slot['done'].synchronize()  # DMA must finish before reusing pinned input.
        pk, pv = self.cache.prefix[layer]
        hk, hv = self.cache.hand_archive[layer][self.index]
        history = [self.cache.archive[layer][i] for i in range(max(0, self.index-5), self.index)]
        keys, values = [pk, hk, *[x[0] for x in history]], [pv, hv, *[x[1] for x in history]]
        shape = (keys[0].shape[0], sum(x.shape[1] for x in keys), *keys[0].shape[2:])
        if slot['buffers'] is None or slot['buffers'][0].shape != shape:
            slot['buffers'] = tuple(torch.empty(shape, dtype=keys[0].dtype, device='cpu', pin_memory=True) for _ in range(2))
        for source_list, target in zip((keys, values), slot['buffers']):
            offset = 0
            for source in source_list:
                assert source.device.type == 'cpu'
                target[:, offset:offset+source.shape[1]].copy_(source)
                offset += source.shape[1]
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            pair = tuple(x.to(self.device, non_blocking=True) for x in slot['buffers'])
            done = torch.cuda.Event()
            done.record(self.stream)
        slot['done'] = done
        self.staging_peak_bytes = max(self.staging_peak_bytes, sum(x.numel()*x.element_size() for s in self.slots if s['buffers'] is not None for x in s['buffers']))
        return pair, done

    def schedule(self, layer):
        if layer < self.layers and layer not in self.pending:
            self.pending[layer] = self.executor.submit(self._prepare, layer)

    def get(self, layer):
        assert not self.closed
        self.schedule(layer)
        start = time.monotonic()
        pair, done = self.pending.pop(layer).result()
        self.wait_seconds += time.monotonic() - start
        current = torch.cuda.current_stream(self.device)
        current.wait_event(done)
        for value in pair:
            value.record_stream(current)
        self.schedule(layer+1)
        return pair

    def finish(self):
        if self.closed:
            return
        try:
            for future in self.pending.values():
                future.result()
            self.stream.synchronize()
        finally:
            self.executor.shutdown(wait=True)
            self.pending.clear()
            self.slots.clear()
            self.closed = True
