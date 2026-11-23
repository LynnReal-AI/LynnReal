"""Experimental out-of-place pointwise kernels with explicit BF16 casts.

SiLU follows PyTorch v2.7.1 ActivationSiluKernel.cu, including the separately
rounded upstream multiplication before SiLU backward. No fast exp/div math.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from .rotary_kernels import _round_bf16


@triton.jit
def _divide_preserve_subnormal(a, b):
    # tl.div_rn emits div.rn.ftz.f32 in this Triton build. Native CUDA SiLU
    # preserves these subnormal BF16 inputs; explicit PTX must omit .ftz.
    return tl.inline_asm_elementwise('div.rn.f32 $0, $1, $2;', constraints='=f,f,f',
                                    args=[a,b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _elementwise(A, B, C, DY, OUT, DA, DB, DC, COUNT,
                 KIND: tl.constexpr, BACKWARD: tl.constexpr, BLOCK: tl.constexpr):
    # Address arithmetic only: cast BEFORE multiplying. Native HD reference
    # batches can exceed 2**31 elements; falling back to eager then allocates
    # several extra full-sequence intermediates and caused step201's OOM.
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = i < COUNT
    if not BACKWARD or KIND == 0:
        a = tl.load(A+i, valid, 0).to(tl.float32)
    b = tl.load(B+i, valid, 0).to(tl.float32)
    if BACKWARD:
        dy = tl.load(DY+i, valid, 0).to(tl.float32)
        if KIND == 0:
            da, db, dc = dy * _round_bf16(1.0+b), dy*a, dy
        else:
            c = tl.load(C+i, valid, 0).to(tl.float32)
            da, db, dc = dy, dy*c, dy*b
        if KIND == 0:
            tl.store(DA+i, da, valid)
        tl.store(DB+i, db, valid)
        if KIND == 1:
            tl.store(DC+i, dc, valid)
    else:
        c = tl.load(C+i, valid, 0).to(tl.float32)
        if KIND == 0:
            out = _round_bf16(a * _round_bf16(1.0+b)) + c
        else:
            out = a + _round_bf16(b*c)
        tl.store(OUT+i, out, valid)


@triton.jit
def _swiglu(X, DY, Y, DX, COUNT, WIDTH: tl.constexpr,
            BACKWARD: tl.constexpr, BLOCK: tl.constexpr):
    # The interleaved input address can exceed int32 even when COUNT does not.
    i = tl.program_id(0).to(tl.int64)*BLOCK + tl.arange(0, BLOCK)
    valid = i < COUNT
    source = (i // WIDTH)*(WIDTH*2) + i % WIDTH
    hidden = tl.load(X+source, valid, 0).to(tl.float32)
    gate = tl.load(X+source+WIDTH, valid, 0).to(tl.float32)
    denom = 1.0 + libdevice.exp(-gate)
    silu = _round_bf16(_divide_preserve_subnormal(gate, denom))
    if BACKWARD:
        dy = tl.load(DY+i, valid, 0).to(tl.float32)
        upstream = _round_bf16(dy*hidden)
        sigmoid = _divide_preserve_subnormal(1.0, denom)
        # CUDA's native SiLU derivative contracts just this FP32 add/mul.
        derivative = tl.fma(gate, 1.0-sigmoid, 1.0)
        gate_gradient = (upstream*sigmoid)*derivative
        tl.store(DX+source, _round_bf16(dy*silu), valid)
        tl.store(DX+source+WIDTH, _round_bf16(gate_gradient), valid)
    else:
        tl.store(Y+i, _round_bf16(hidden*silu), valid)


def forward(name, values):
    if name == 'swiglu':
        x = values[0]
        shape = (*x.shape[:-1], x.shape[-1]//2)
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        count = out.numel()
        _swiglu[(triton.cdiv(count, 512),)](x, out, out, x, count, shape[-1], False, 512,
                                         num_warps=4, enable_fp_fusion=False)
    else:
        out = torch.empty_like(values[0])
        count = out.numel()
        _elementwise[(triton.cdiv(count, 1024),)](
            *values, out, out, out, out, out, count, int(name=='gated_residual'), False, 1024,
            num_warps=4, enable_fp_fusion=False)
    return out


def backward(name, gradient, values):
    if name == 'swiglu':
        x = values[0]
        dx = torch.empty_like(x)
        count = gradient.numel()
        _swiglu[(triton.cdiv(count, 512),)](
            x, gradient, gradient, dx, count, x.shape[-1]//2, True, 512,
            num_warps=4, enable_fp_fusion=False)
        return (dx,)
    # Identity branches return a view of the incoming gradient. There is
    # no write into it and no gradient quantization or accumulation here.
    identity = 2 if name == 'modulate' else 0
    grads = tuple(gradient.reshape_as(x) if i == identity else torch.empty_like(x)
                  for i,x in enumerate(values))
    count = gradient.numel()
    _elementwise[(triton.cdiv(count, 1024),)](
        *values, gradient, gradient, *grads, count, int(name=='gated_residual'), True, 1024,
        num_warps=4, enable_fp_fusion=False)
    return grads
