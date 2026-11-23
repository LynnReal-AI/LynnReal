"""H3 RoPE fusion preserving the eager BF16 rounding order.

Forward and backward only fuse the released elementwise rotation, keeping
the BF16 rounding after EACH multiplication. No approximation, in-place
write, quantized activation, new RoPE layout or gradient truncation. Tail
channels pass through unchanged; cos/sin must be frozen. No CUDA import or
context is created until the explicitly selected CUDA path is called.
"""
import torch
from torch.autograd.function import once_differentiable


def reference_backward(gradient, cos, sin):
    """CPU-testable derivative using the same eager BF16 operation order."""
    width = cos.shape[-1]
    cos, sin = cos.to(gradient.dtype), sin.to(gradient.dtype)
    if cos.ndim == 2:
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    else:
        cos, sin = cos[:, :, None, :], sin[:, :, None, :]
    rotating, tail = gradient[..., :width], gradient[..., width:]
    direct = rotating * cos
    sine = rotating * sin
    left, right = sine.chunk(2, dim=-1)
    rotated = torch.cat((right, -left), dim=-1)
    return torch.cat((direct + rotated, tail), dim=-1).contiguous()


def eligible(hidden, cos, sin):
    return (hidden.device.type == 'cuda' and hidden.dtype == torch.bfloat16
            and hidden.ndim == 4 and hidden.shape[0] == 1
            and hidden.numel() < 2**31
            and hidden.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()
            and hidden.device == cos.device == sin.device
            and cos.shape == sin.shape and cos.ndim in (2, 3)
            and (cos.ndim == 2 or cos.shape[0] == 1)
            and cos.shape[-2] == hidden.shape[1]
            and 0 < cos.shape[-1] <= hidden.shape[-1] and cos.shape[-1] % 2 == 0
            and cos.dtype in (torch.float32, torch.bfloat16)
            and sin.dtype == cos.dtype and not cos.requires_grad and not sin.requires_grad)


class _Rope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, cos, sin):
        from .rotary_kernels import apply
        ctx.save_for_backward(cos, sin)
        return apply(hidden, cos, sin, backward=False)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        from .rotary_kernels import apply
        cos, sin = ctx.saved_tensors
        return apply(gradient.contiguous(), cos, sin, backward=True), None, None


def fused_rotary(hidden, cos, sin):
    if not eligible(hidden, cos, sin):
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
        return _apply_rotary_emb(hidden, cos, sin)
    return _Rope.apply(hidden, cos, sin)
