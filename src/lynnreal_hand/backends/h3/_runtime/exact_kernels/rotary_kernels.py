"""One elementwise Triton kernel per H3 RoPE direction (no reductions)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _round_bf16(value):
    # Explicit PTX conversion makes the eager BF16 boundary observable to
    # the compiler, including between the two multiplies and final add.
    rounded = tl.inline_asm_elementwise(
        'cvt.rn.bf16.f32 $0, $1;', constraints='=h,f', args=[value],
        dtype=tl.bfloat16, is_pure=True, pack=1)
    return rounded.to(tl.float32)


@triton.jit
def _rope(X, C, S, Y, COUNT, HEADS: tl.constexpr,
          DIM: tl.constexpr, ROTARY: tl.constexpr, BACKWARD: tl.constexpr,
          BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = index % DIM
    row = index // (HEADS * DIM)
    rotate = channel < ROTARY
    partner_channel = tl.where(channel < ROTARY // 2,
                               channel + ROTARY // 2, channel - ROTARY // 2)
    partner = index + partner_channel - channel
    x = tl.load(X + index, index < COUNT, other=0).to(tl.float32)
    other = tl.load(X + partner, (index < COUNT) & rotate, other=0).to(tl.float32)
    cosine = tl.load(C + row * ROTARY + channel, (index < COUNT) & rotate, other=0).to(tl.float32)
    sine_channel = partner_channel if BACKWARD else channel
    sine = tl.load(S + row * ROTARY + sine_channel, (index < COUNT) & rotate, other=0).to(tl.float32)
    cosine, sine = _round_bf16(cosine), _round_bf16(sine)
    a, b = _round_bf16(x * cosine), _round_bf16(other * sine)
    # Forward (-second, first); backward (second_gradient, -first_gradient).
    sign = tl.where(channel < ROTARY // 2, 1.0 if BACKWARD else -1.0,
                    -1.0 if BACKWARD else 1.0)
    result = tl.where(rotate, a + sign * b, x)
    tl.store(Y + index, result, index < COUNT)


def apply(hidden, cos, sin, *, backward):
    output = torch.empty_like(hidden)
    count = hidden.numel()
    # COUNT is runtime, not constexpr: native video lengths reuse the SAME
    # compiled kernel instead of compiling forward/backward for every clip.
    _rope[(triton.cdiv(count, 1024),)](
        hidden, cos, sin, output, count, hidden.shape[2], hidden.shape[3],
        cos.shape[-1], backward, 1024, num_warps=4, enable_fp_fusion=False)
    return output
