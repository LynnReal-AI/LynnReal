"""BF16 pointwise fusion with eager rounding boundaries and chain-rule gradients.

CPU tensors and unsupported layouts use the original elementwise expressions.
"""
import torch
from torch.autograd.function import once_differentiable


def eager(name, *values):
    if name == 'modulate':
        hidden, scale, shift = values
        return hidden * (1.0 + scale) + shift
    if name == 'gated_residual':
        residual, gate, value = values
        return residual + gate * value
    if name == 'swiglu':
        hidden, gate = values[0].chunk(2, dim=-1)
        return hidden * torch.nn.functional.silu(gate)
    raise ValueError('unknown pointwise region')


def reference_backward(name, gradient, *values):
    if name == 'modulate':
        hidden, scale, shift = values
        grads = gradient * (1.0 + scale), gradient * hidden, gradient
    elif name == 'gated_residual':
        residual, gate, value = values
        grads = gradient, gradient * value, gradient * gate
    elif name == 'swiglu':
        hidden, gate = values[0].chunk(2, dim=-1)
        return (torch.cat((gradient * torch.nn.functional.silu(gate),
                          torch.ops.aten.silu_backward(gradient * hidden, gate)), dim=-1),)
    else:
        raise ValueError('unknown pointwise region')
    # Only a leading singleton broadcast is eligible. There is no reduction.
    return tuple(g.reshape_as(v) for g, v in zip(grads, values))


def eligible(name, values):
    if name not in ('modulate', 'gated_residual', 'swiglu'):
        raise ValueError('unknown pointwise region')
    if len(values) != (1 if name == 'swiglu' else 3):
        return False
    first = values[0]
    if not (first.device.type == 'cuda' and first.dtype == torch.bfloat16
            and first.is_contiguous() and 0 < first.numel() < 2**63
            and first.ndim in (2, 3)):
        return False
    if name == 'swiglu':
        return first.shape[-1] % 2 == 0
    return all(v.device == first.device and v.dtype == first.dtype and v.is_contiguous()
               and v.numel() == first.numel()
               and tuple(v.shape) in (tuple(first.shape), tuple(first.shape[1:]) if first.shape[0] == 1 else ())
               for v in values)


class _Exact(torch.autograd.Function):
    @staticmethod
    def forward(ctx, name, *values):
        from .pointwise_kernels import forward
        ctx.name = name
        ctx.shapes = tuple(v.shape for v in values)
        # Match the original graph's necessary inputs only. Saving shift or
        # residual unnecessarily would undo part of the long-sequence HBM
        # benefit. Their derivatives are identity and need no tensor values.
        saved = values if name == 'swiglu' else values[:2] if name == 'modulate' else values[1:]
        ctx.save_for_backward(*saved)
        return forward(name, values)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        from .pointwise_kernels import backward
        gradient = gradient.contiguous()
        values = ctx.saved_tensors
        if ctx.name == 'modulate':
            values = (*values, gradient.reshape(ctx.shapes[2]))
        elif ctx.name == 'gated_residual':
            values = (gradient.reshape(ctx.shapes[0]), *values)
        return (None, *backward(ctx.name, gradient, values))


def fused(name, *values):
    if not eligible(name, values):
        return eager(name, *values)
    return _Exact.apply(name, *values)


def install_pointwise(transformer, *, implementation=fused):
    """Bind only this H3 instance; no global patch or parameter replacement."""
    from diffusers.models.activations import SwiGLU
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
    selected = []
    for name, module in transformer.named_modules():
        if type(module) in (MiniMaxH3TransformerBlock, SwiGLU):
            if hasattr(module, '_h3_pointwise_kernel'):
                raise RuntimeError('pointwise fusion already installed: ' + name)
            if type(module) is SwiGLU and (type(module.activation) is not torch.nn.SiLU or module.activation.inplace):
                raise RuntimeError('only the unchanged original SiLU is supported')
            selected.append((name, module))
    if not selected:
        raise RuntimeError('no H3 pointwise modules found')
    for _,module in selected:
        module._h3_pointwise_kernel = implementation
    return [name for name,_ in selected]
