"""Exact packed-KV, fullgraph and CUDA-graph sampler ported from the measured path.

The iterator delivers a current chunk before computing its clean history commit.
No experiment paths, schedulers, ground-truth history or approximation modules.
"""
from dataclasses import dataclass, replace
import gc
import time
import numpy as np
import torch
from lynnreal_core.backends.minimax_h3.layout import patchify_video, unpatchify_video
from inference_core import layout,student_layout,local_hand_positions,window_positions
from reference_kv import Control,transformer_blocks
from kv import RawCache
from lynnreal_hand.media import read_video


def read_rgb(path,count,height,width):
    return read_video(path,frames=count,height=height,width=width)


def build_runner(model,device,eager_model_forward,collect_gate):
    import deploy_helpers as helpers
    active_variant='compiled_graph'
    def progress(*args,**kwargs):pass
    def dump(name,value):collect_gate(name,value)
    @dataclass
    class NativeControl(Control):
        frames: int = 5
        span: int = 37

        def video_positions(self, slots):
            full = super().video_positions(slots)
            return full[:(slots - 1) * 5 * self.rows + self.frames * self.rows]

    class Core:
        def __init__(self):
            self.inputs = {}
            self.time_rows = {}

        def forward_rows(self, control, video, prompt, positions, tags, text_indices, t, r):
            t, r = float(t), float(r)
            mode, n = control.mode, len(positions)
            model._deploy_time_key = ('prefix' if mode == 'prefix' else 'rows', t, r)
            control.cache._capture_noisy_raw = mode == 'noisy' and r == 1.
            key = (mode, n, t, r)
            if key not in self.inputs:
                prefix = mode == 'prefix'
                times = [[.999, .999], [1., 1.]] if prefix else [[1., 1.] if mode in ('hand', 'commit') else [t, r]]
                self.time_rows[key] = ([row[0] for row in times], [row[1] for row in times])
                times = torch.tensor(times, device=device)
                inverse = torch.zeros(n, device=device, dtype=torch.long)
                if prefix:
                    inverse[:control.text_rows] = 1
                vi = torch.arange(control.text_rows if prefix else 0, n, device=device)
                self.inputs[key] = dict(timestep=times[:, 0], r_timestep=times[:, 1], timestep_indices=inverse,
                    token_tags=torch.as_tensor(tags, device=device), video_indices=vi, audio_indices=vi[:0],
                    text_indices=torch.as_tensor(text_indices, device=device, dtype=torch.long))
            from .adaln import select_adaln
            select_adaln(model, *self.time_rows[key])
            output, _ = model(hidden_states=video, encoder_hidden_states=prompt,
                audio_hidden_states=video.new_empty((1, 0, 32)), position_ids=positions,
                attention_mask=None, fused_prope=True, prope_kwargs={'duo_rawkv': control},
                prope_token_indices=self.inputs[key]['video_indices'], prope_frame_ids=None,
                cam_viewmats=control.views, cam_K=control.intrinsics,
                packed_sequence_parallel=False, return_dict=False, **self.inputs[key])
            return output

    def run_chunks(case, batch, *, encoder, decoder, warmup_rollouts=0, repeats=1, on_record=None):
        benchmark = warmup_rollouts > 0
        trials, graph_enabled = (warmup_rollouts + repeats if benchmark else 1), True
        if repeats < 1 or warmup_rollouts < 0:
            raise ValueError("Invalid benchmark rollout counts")
        core = Core()
        doc = student_layout(layout(batch, device, 'HAND', latent_frames=case['latent_frames']))
        rows = doc.rows_per_frame
        latent_height, latent_width = batch.target_latents.shape[-2:]
        height, width = latent_height*16, latent_width*16
        assert (height, width) in ((544, 960), (768, 1344))
        assert rows == latent_height*latent_width//4
        output_height = 540 if height == 544 else height
        prefix = doc.positions
        hand_positions = local_hand_positions(doc, 5)
        prompt = batch.prompt_embeds[None].to(device, torch.bfloat16)
        empty = prompt[:, :0]
        views, intrinsics = batch.camera_viewmats[None].to(device).float(), batch.camera_K[None].to(device).float()
        store = RawCache(archive_enabled=False, archive_async=False, window=2)
        base = NativeControl('prefix', store, 0, prefix, hand_positions, views, intrinsics,
                             model.rope, doc.text_rows, rows, frames=5, span=case['latent_frames'])
        # CPU geometry oracle checks the complete span, including native2 tail.
        for i in range((case['latent_frames'] + 4) // 5):
            length = min(5, case['latent_frames'] - i * 5)
            c = replace(base, index=i, frames=length)
            assert torch.equal(c.video_positions(min(i, 1) + 1)[-length*rows:], window_positions(doc, i, i, 2))
        graphs = {}
        pool = torch.cuda.graph_pool_handle() if graph_enabled else None
        geometry_for = transformer_blocks(model)[0][2].attn.processor.geometry_for
        sigma_grid = torch.linspace(1., 0., 4, device=device)
        sigmas = (3 * sigma_grid / (1 + 2 * sigma_grid)).tolist()
        raw_hand = read_rgb(case['original_hand'], case['frames'], height=height, width=width) if encoder else None
        graph_gates = []
        try:
            for trial in range(trials):
                warm = trial < warmup_rollouts if benchmark else graph_enabled and trial == 0
                if decoder is not None and trial == 1:
                    torch.cuda.reset_peak_memory_stats()
                progress('trial_start', case_id=case['case_id'], variant=active_variant, trial=trial, warmup=warm)
                for item in store.packed.values():
                    item['index'] = -1
                store.begin_chunk(0)
                gen = torch.Generator(device=device).manual_seed(case['seed'])
                anchor = patchify_video(batch.anchor_latents[None].to(device).float())
                anchor = .999 * anchor + .001 * torch.randn(anchor.shape, device=device, generator=gen)
                torch.cuda.synchronize()
                request_start = previous = time.perf_counter()
                core.forward_rows(base, anchor, prompt, prefix, doc.tags, range(doc.text_rows), 1., 1.)
                count = (case['latent_frames'] + 4) // 5
                for i in range(count):
                    length = min(5, case['latent_frames'] - i * 5)
                    store.begin_chunk(i)
                    c = replace(base, mode='hand', index=i, frames=length)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    views[:, i*5:i*5+length].copy_(batch.camera_viewmats[None, i*5:i*5+length], non_blocking=True)
                    intrinsics[:, i*5:i*5+length].copy_(batch.camera_K[None, i*5:i*5+length], non_blocking=True)
                    if encoder:
                        pixels = torch.from_numpy(raw_hand[i*17:i*17+(17 if length == 5 else 5)]).permute(0, 3, 1, 2)[None]
                        encoded = encoder(pixels.to(device, non_blocking=True).float().div_(255)).to(torch.bfloat16)
                        assert tuple(encoded.shape) == (1, 24, length, latent_height, latent_width)
                        hand = patchify_video(encoded.float())
                    else:
                        hand = patchify_video(batch.hand_latents[:, i*5:i*5+length][None].to(device).float())
                    hp = hand_positions[:length*rows]
                    tags = np.zeros(length*rows, np.int64)
                    core.forward_rows(c, hand, empty, hp, tags, [], 1., 1.)
                    current = torch.randn((1, length*rows, 96), device=device, generator=gen)
                    c = replace(c, mode='noisy')
                    positions = c.video_positions(min(i, 1) + 1)[-length*rows:]
                    geometry = geometry_for(c, 'rgb', torch.bfloat16)
                    bucket = (min(i, 1), length)
                    if (*bucket, 0) in graphs:
                        stable = graphs[(*bucket, 0)][3]
                        for key, value in stable.items():
                            incoming = geometry[key]
                            if isinstance(value, torch.Tensor):
                                value.copy_(incoming)
                            elif isinstance(value, tuple):
                                for a, b in zip(value, incoming):
                                    a.copy_(b)
                    for j in range(3):
                        t, r = 1-sigmas[j], 1-sigmas[j+1]
                        key = (*bucket, j)
                        if graph_enabled:
                            if key not in graphs:
                                assert warm, 'A new shape appeared in formal timing'
                                static = current.clone()
                                if not getattr(model, '_lynn_time_banks_ready', False):
                                    dispatch = model.forward
                                    model.forward = eager_model_forward
                                    try:
                                        for bank in range(3):
                                            core.forward_rows(c, static, empty, positions, tags, [],
                                                              1-sigmas[bank], 1-sigmas[bank+1])
                                    finally:
                                        model.forward = dispatch
                                    model._lynn_time_banks_ready = True
                                expected = core.forward_rows(c, static, empty, positions, tags, [], t, r).clone()
                                if active_variant == 'compiled_graph':
                                    original_dispatch = model.forward
                                    model.forward = eager_model_forward
                                    try:
                                        eager_output = core.forward_rows(c, static, empty, positions, tags, [], t, r).clone()
                                    finally:
                                        model.forward = original_dispatch
                                    compile_error = helpers.relative(expected, eager_output)
                                    graph_gates.append(dict(bucket=bucket, map=j, compile_eager_relative_L2=compile_error,
                                                            passed=compile_error <= .005))
                                    progress('DiT_shape_compile_gate', case_id=case['case_id'], bucket=bucket, map=j, relative_L2=compile_error, passed=compile_error <= .005)
                                    if compile_error > .005:
                                        raise RuntimeError(f'Complete DiT compile/eager parity failed: {compile_error}')
                                side = torch.cuda.Stream()
                                side.wait_stream(torch.cuda.current_stream())
                                with torch.cuda.stream(side):
                                    core.forward_rows(c, static, empty, positions, tags, [], t, r)
                                torch.cuda.current_stream().wait_stream(side)
                                torch.cuda.synchronize()
                                graph = torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph, pool=pool):
                                    output = core.forward_rows(c, static, empty, positions, tags, [], t, r)
                                graph.replay()
                                torch.cuda.synchronize()
                                error = helpers.relative(output, expected)
                                if error > .005:
                                    raise RuntimeError(f'Graph parity failed: {error}')
                                graph_gates.append(dict(bucket=bucket, map=j, relative_L2=error, passed=True))
                                graphs[key] = (graph, static, output, geometry, positions, c, empty)
                            graph, static, output, *_ = graphs[key]
                            static.copy_(current)
                            graph.replay()
                            velocity = output
                        else:
                            velocity = core.forward_rows(c, current, empty, positions, tags, [], t, r)
                        current = current + (sigmas[j] - sigmas[j+1]) * velocity.float()
                    z = unpatchify_video(current, length, doc.height, doc.width).contiguous().float()
                    if decoder:
                        delivered = decoder(z)
                        assert delivered.shape == (17 if length == 5 else 5, output_height, width, 3)
                    torch.cuda.synchronize()
                    ready = time.perf_counter()
                    if not benchmark:
                        yield delivered.clone(), z.cpu()
                    if i < count - 1:
                        core.forward_rows(replace(c, mode='commit'), current, empty, positions, tags, [], 1., 1.)
                        torch.cuda.synchronize()
                    committed = time.perf_counter()
                    if benchmark:
                        record = dict(case_id=case['case_id'], trial=trial, warmup=warm,
                            chunk=i, latent_frames=length, RGB_frames=17 if length == 5 else 5,
                            mtp_ms=1000*(ready-start), output_interval_ms=1000*(ready-previous),
                            CPU_RGB_ready_ms=1000*(ready-request_start),
                            commit_after_RGB_ms=1000*(committed-ready), commit_after_RGB=True)
                        previous = ready
                        # Match the source benchmark: CPU-ready endpoint precedes
                        # commit; owned result copies and journal I/O follow it.
                        owned = (delivered.clone(), z.cpu()) if not warm else None
                        if on_record is not None:
                            on_record(record)
                        if owned is not None:
                            yield owned
        finally:
            dump(f'gates/{active_variant}_{case["case_id"]}_graphs.json', {'passed': all(x['passed'] for x in graph_gates), 'checks': graph_gates})
            store.clear()
            del graphs, store, core
            gc.collect()
            torch.cuda.empty_cache()

    return run_chunks
