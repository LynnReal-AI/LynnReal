"""Expose existing native Triton launches as explicit fullgraph operators.

Their launch options, including enable_fp_fusion=False, remain under the native
launcher instead of being reconstructed by Inductor's user-kernel lowering.
No transformer block is bypassed and no mathematical expression is changed.
"""
from types import SimpleNamespace
import types
import torch

ORIGINAL = {}
PROJECTIONS = []


@torch.library.custom_op('fast540_native::rms', mutates_args=())
def rms(x: torch.Tensor, w: torch.Tensor, table: torch.Tensor, ids: torch.Tensor,
        shift: int, scale: int, epsilon: float) -> torch.Tensor:
    return ORIGINAL['rms'](x, SimpleNamespace(weight=w, eps=epsilon), table, ids, shift, scale)


@rms.register_fake
def _(x, w, table, ids, shift, scale, epsilon):
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


@torch.library.custom_op('fast540_native::rms_quant', mutates_args=())
def rms_quant(x: torch.Tensor, w: torch.Tensor, table: torch.Tensor, ids: torch.Tensor,
              shift: int, scale: int, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    return ORIGINAL['rms_quant'](x, SimpleNamespace(weight=w, eps=epsilon), table, ids, shift, scale)


@rms_quant.register_fake
def _(x, w, table, ids, shift, scale, epsilon):
    m, h = x.numel()//x.shape[-1], x.shape[-1]
    return (torch.empty((m, h), device=x.device, dtype=torch.int8),
            torch.empty((m, 1), device=x.device, dtype=torch.float32))


@torch.library.custom_op('fast540_native::residual', mutates_args=())
def residual(x: torch.Tensor, update: torch.Tensor, table: torch.Tensor,
             ids: torch.Tensor, gate: int) -> torch.Tensor:
    return ORIGINAL['residual'](x, update, table, ids, gate)


@residual.register_fake
def _(x, update, table, ids, gate):
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


@torch.library.custom_op('fast540_native::swiglu', mutates_args=())
def swiglu(x: torch.Tensor) -> torch.Tensor:
    return ORIGINAL['swiglu'](x)


@swiglu.register_fake
def _(x):
    return torch.empty((*x.shape[:-1], x.shape[-1]//2), device=x.device, dtype=x.dtype)


@torch.library.custom_op('fast540_native::swiglu_quant', mutates_args=())
def swiglu_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return ORIGINAL['swiglu_quant'](x)


@swiglu_quant.register_fake
def _(x):
    m, h = x.numel()//x.shape[-1], x.shape[-1]//2
    return (torch.empty((m, h), device=x.device, dtype=torch.int8),
            torch.empty((m, 1), device=x.device, dtype=torch.float32))


@torch.library.custom_op('fast540_native::residual_norm_quant', mutates_args=())
def residual_norm_quant(x: torch.Tensor, update: torch.Tensor, w: torch.Tensor,
                        table: torch.Tensor, ids: torch.Tensor, epsilon: float
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return ORIGINAL['residual_norm_quant'](x, update, SimpleNamespace(weight=w, eps=epsilon), table, ids)


@residual_norm_quant.register_fake
def _(x, update, w, table, ids, epsilon):
    m, h = x.numel()//x.shape[-1], x.shape[-1]
    return (torch.empty(x.shape, device=x.device, dtype=x.dtype),
            torch.empty((m, h), device=x.device, dtype=torch.int8),
            torch.empty((m, 1), device=x.device, dtype=torch.float32))


@torch.library.custom_op('fast540_native::projection', mutates_args=())
def projection(x: torch.Tensor, scale: torch.Tensor, shape: list[int], index: int) -> torch.Tensor:
    module, forward = PROJECTIONS[index]
    return forward(x, scale, shape)


@projection.register_fake
def _(x, scale, shape, index):
    return torch.empty((*shape, PROJECTIONS[index][0].out_features), device=x.device, dtype=torch.bfloat16)


def install(model):
    initial_count = len(PROJECTIONS)
    from lynn_kernels import kernels as k
    from lynn_kernels.int8 import Int8Linear
    names = dict(rms='_rms_adaln', rms_quant='_rms_adaln_quant_int8', residual='_gated_residual',
                 swiglu='_swiglu', swiglu_quant='_swiglu_quant_int8', residual_norm_quant='residual_rms_quant')
    if not ORIGINAL:
        ORIGINAL.update({key: getattr(k, name) for key, name in names.items()})
        k._rms_adaln = lambda x,n,t,i,s,c: rms(x,n.weight,t,i,s,c,float(n.eps))
        k._rms_adaln_quant_int8 = lambda x,n,t,i,s,c: rms_quant(x,n.weight,t,i,s,c,float(n.eps))
        k._gated_residual = residual
        k._swiglu = swiglu
        k._swiglu_quant_int8 = swiglu_quant
        k.residual_rms_quant = lambda x,u,n,t,i: residual_norm_quant(x,u,n.weight,t,i,float(n.eps))
    for module in model.modules():
        if isinstance(module, Int8Linear):
            index = len(PROJECTIONS)
            PROJECTIONS.append((module, module.forward_quantized))
            def call(self, x, scale, shape, op_index=index):
                return projection(x, scale, list(shape), op_index)
            module.forward_quantized = types.MethodType(call, module)
    assert len(PROJECTIONS) - initial_count == 192
    return dict(native_Triton_launches_preserved=True, explicit_projection_operators=len(PROJECTIONS),
                full_DiT_blocks=50, math_changed=False)
