"""Dispatch exact RoPE to 32-bit or 64-bit indexing based on tensor size."""


def fused_rotary(hidden, cos, sin):
    if hidden.numel() < 2**31:
        from .rotary_ops import fused_rotary as rotate
    else:
        from .rotary_wide_ops import fused_rotary as rotate
    return rotate(hidden, cos, sin)
