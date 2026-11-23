"""Exact RoPE with 64-bit address arithmetic for large tensors.

Tensor precision and rotary operation order match the 32-bit implementation.
"""
import torch
from torch.autograd.function import once_differentiable


def eligible(hidden, cos, sin):
    return (hidden.device.type == 'cuda' and hidden.dtype == torch.bfloat16
            and hidden.ndim == 4 and hidden.shape[0] == 1
            and 0 < hidden.numel() < 2**40
            and hidden.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()
            and hidden.device == cos.device == sin.device
            and cos.shape == sin.shape and cos.ndim in (2, 3)
            and (cos.ndim == 2 or cos.shape[0] == 1)
            and cos.shape[-2] == hidden.shape[1]
            and 0 < cos.shape[-1] <= hidden.shape[-1] and cos.shape[-1] % 2 == 0
            and cos.dtype in (torch.float32, torch.bfloat16)
            and sin.dtype == cos.dtype and not cos.requires_grad and not sin.requires_grad)


class _WideRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, cos, sin):
        from .rotary_wide_kernels import apply
        ctx.save_for_backward(cos, sin)
        return apply(hidden, cos, sin, backward=False)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        from .rotary_wide_kernels import apply
        cos, sin = ctx.saved_tensors
        return apply(gradient.contiguous(), cos, sin, backward=True), None, None


def fused_rotary(hidden, cos, sin):
    if not eligible(hidden, cos, sin):
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
        return _apply_rotary_emb(hidden, cos, sin)
    return _WideRope.apply(hidden, cos, sin)
