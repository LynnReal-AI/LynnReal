"""Preserve native reduction/rounding boundaries within the full DiT graph."""
import torch
from torch import nn

NORMS = []
OUTPUT_NORMS = []
PROJECT = None


@torch.library.custom_op('fast540_precision::qk_norm', mutates_args=())
def qk_norm(x: torch.Tensor, index: int) -> torch.Tensor:
    return NORMS[index](x).contiguous()


@qk_norm.register_fake
def _(x, index):
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


@torch.library.custom_op('fast540_precision::camera_project', mutates_args=())
def camera_project(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return PROJECT(x, matrix=matrix).contiguous()


@camera_project.register_fake
def _(x, matrix):
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


@torch.library.custom_op('fast540_precision::output_norm', mutates_args=())
def output_norm(x: torch.Tensor, temb: torch.Tensor, indices: torch.Tensor, index: int) -> torch.Tensor:
    # The model's next operation casts this result to its FP32 output head.
    return OUTPUT_NORMS[index](x, temb, indices).float().contiguous()


@output_norm.register_fake
def _(x, temb, indices, index):
    return torch.empty(x.shape, dtype=torch.float32, device=x.device)


class NativeQKNorm(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.index = len(NORMS)
        NORMS.append(original)

    def forward(self, x):
        return qk_norm(x, self.index)


class NativeOutputNorm(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.index = len(OUTPUT_NORMS)
        OUTPUT_NORMS.append(original)

    def forward(self, x, temb, indices):
        return output_norm(x, temb, indices, self.index)


def install_before_KV(model):
    global PROJECT
    import lynnreal_core.backends.minimax_h3.torch_prope as prope
    if PROJECT is None:
        PROJECT = prope._project
        prope._project = lambda features, *, matrix: camera_project(features, matrix)
    for block in model.transformer_blocks:
        block.attn.norm_q = NativeQKNorm(block.attn.norm_q)
        block.attn.norm_k = NativeQKNorm(block.attn.norm_k)
    model.norm_out = NativeOutputNorm(model.norm_out)


def install_edge_linears(model):
    from ultra_fast_vae.stable_vae_ops import OpaqueLinear
    for index in [0, 49]:
        block = model.transformer_blocks[index]
        for owner, attr in [(block.attn, 'to_qkv_deploy'), (block.attn.to_out, '0'),
                            (block.ff.net[0], 'proj'), (block.ff.net, '2')]:
            setattr(owner, attr, OpaqueLinear(getattr(owner, attr)))
