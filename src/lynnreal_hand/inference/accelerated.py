"""Single-Hopper deployment with fused W8A8 projections and all-block compilation.

Compilation and graph capture happen on first use. Numerical gates compare each
new DiT shape against the same fused eager model; failures abort generation.
"""
from pathlib import Path
import torch


class AcceleratedFastPipeline:
    @torch.inference_mode()
    def __init__(self, config, device=None):
        from ..backends.h3.runtime import initialize
        from ..models.h3 import build_model
        from ..models.codecs import TinyHandEncoder, LynnRealDecoder
        self.config = config
        self.device = device or initialize()
        self.gates = {}
        self.model, self.adaln_memory = build_model(config, self.device, role='student')
        import deploy_helpers as helpers
        from reference_kv import transformer_blocks
        from precision_boundary import install_before_KV, install_edge_linears
        from kv import install as install_kv
        from full_fusion import install as install_fusion
        from condition_policy import install as install_condition, configure
        helpers.write = self._gate
        install_before_KV(self.model)
        install_kv(self.model)
        self.time_counters = helpers.time_cache(self.model)
        install_fusion(self.model, transformer_blocks, helpers.fuse_qkv, self._gate)
        install_condition(self.model, transformer_blocks, self._gate)
        configure(name='accepted_exact_tail', exact_tail=True, cache_only_head=True)
        from ultra_fast_vae.stable_vae_ops import OpaqueLinear
        self.model.proj_in = OpaqueLinear(self.model.proj_in)
        self.model.proj_out = OpaqueLinear(self.model.proj_out)
        install_edge_linears(self.model)
        from native_kernel_ops import install as install_native
        self._gate('native_operator_boundaries', install_native(self.model))
        # These are intentional eager boundaries around the original kernels.
        # Register them before lowering to avoid expensive diagnostic IR printing.
        from torch._inductor.lowering import make_fallback
        for namespace, names in {
            'fast540_native': ['rms', 'rms_quant', 'residual', 'swiglu',
                               'swiglu_quant', 'residual_norm_quant', 'projection'],
            'fast540_precision': ['qk_norm', 'camera_project', 'output_norm'],
            'lynnreal_vae': ['linear_half'],
        }.items():
            for name in names:
                make_fallback(getattr(getattr(torch.ops, namespace), name).default,
                              warn=False)
        eager = self.model.forward
        compiled = torch.compile(eager, fullgraph=True, dynamic=False,
            options={'triton.cudagraphs': False, 'emulate_precision_casts': True})

        def dispatch(*args, **kwargs):
            control = kwargs['prope_kwargs']['duo_rawkv']
            return (compiled if control.mode == 'noisy' else eager)(*args, **kwargs)

        self.model.forward = dispatch
        self.encoder = TinyHandEncoder(config['hand_encoder'], self.device)
        self.compiled_encoders = {}
        self.decoder = LynnRealDecoder(config['decoder'], self.device, compiled=True,
            height=config['video']['height'], width=config['video']['width'])
        # Decoder initialization configures Dynamo too. Apply the combined
        # DiT/encoder/decoder budget last so new scene KV objects have room.
        torch._dynamo.config.recompile_limit = 128
        torch._dynamo.config.accumulated_recompile_limit = 256
        from .packed_graph import build_runner
        self.runner = build_runner(self.model, self.device, eager, self._gate)

    def _gate(self, name, value):
        self.gates[name] = value

    def _encode(self, pixels):
        from deploy_helpers import gated_compile
        count = pixels.shape[1]
        if count not in self.compiled_encoders:
            self.compiled_encoders[count] = gated_compile(
                self.encoder, (pixels,), f'hand_encoder_{count}', .005, graphs=True)
        return self.compiled_encoders[count](pixels)

    def stream(self, batch, *, seed=42):
        source = getattr(batch, 'source_hand_video', None)
        if not source or not Path(source).is_file():
            raise FileNotFoundError('Accelerated inference requires the original hand video; '
                                    'keep source_hand_video relative to the conditioning cache')
        case = dict(case_id=batch.sample_id, frames=batch.pixel_frames,
                    latent_frames=batch.target_latents.shape[1],
                    original_hand=source, seed=seed)
        # A generator must enter the context while advancing, not just when made.
        with torch.inference_mode():
            yield from self.runner(case, batch, encoder=self._encode,
                                   decoder=self.decoder.decode)

    def generate(self, batch, *, seed=42):
        frames, latents = zip(*self.stream(batch, seed=seed))
        return torch.cat(frames), torch.cat(latents, 2)


    def benchmark(self, batch, *, seed=42, warmup_rollouts=1, repeats=3, on_record=None):
        """Yield formal RGB/latent chunks after one retained-graph warm session.

        The timing endpoint is synchronized CPU RGB-ready, before clean commit.
        Benchmark delivery/copy and journaling follow commit, matching the paper
        protocol. Use stream() for delivery-before-commit application streaming.
        """
        if warmup_rollouts < 1:
            raise ValueError('At least one full warmup rollout is required')
        source = getattr(batch, 'source_hand_video', None)
        if not source or not Path(source).is_file():
            raise FileNotFoundError('Benchmark requires the original hand video')
        case = dict(case_id=batch.sample_id, frames=batch.pixel_frames,
                    latent_frames=batch.target_latents.shape[1], original_hand=source, seed=seed)
        with torch.inference_mode():
            yield from self.runner(case, batch, encoder=self._encode,
                decoder=self.decoder.decode, warmup_rollouts=warmup_rollouts,
                repeats=repeats, on_record=on_record)
