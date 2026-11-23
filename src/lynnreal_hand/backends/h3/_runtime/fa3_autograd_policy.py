from __future__ import annotations

import json
import os

import torch


def install():
    from diffusers.models import attention_dispatch as dispatch

    if getattr(dispatch._wrapped_flash_attn_3, "lynnreal_core_native_fa3_autograd", False):
        return
    original = dispatch._wrapped_flash_attn_3

    def differentiable_fa3(q, k, v, softmax_scale=None, causal=False, qv=None, q_descale=None, k_descale=None, v_descale=None, attention_chunk=0, softcap=0.0, num_splits=1, pack_gqa=None, deterministic=False, sm_margin=0):
        result = dispatch.flash_attn_3_func(q=q, k=k, v=v, softmax_scale=softmax_scale, causal=causal, qv=qv, q_descale=q_descale, k_descale=k_descale, v_descale=v_descale, window_size=(-1, -1), attention_chunk=attention_chunk, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa, deterministic=deterministic, sm_margin=sm_margin, return_attn_probs=True)
        output, lse, *_ = result
        return output, lse.permute(0, 2, 1)

    differentiable_fa3.lynnreal_core_native_fa3_autograd = True
    differentiable_fa3.original_custom_op = original
    dispatch._wrapped_flash_attn_3 = differentiable_fa3


@torch.inference_mode(False)
@torch.enable_grad()
def verify_gpu(device):
    from diffusers.models import attention_dispatch as dispatch

    install()
    torch.cuda.set_device(device)
    values = [torch.randn((1, 17, 2, 128), device=device, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    output, _ = dispatch._wrapped_flash_attn_3(*values)
    with torch.no_grad():
        reference, _ = dispatch._wrapped_flash_attn_3.original_custom_op(*values)
    difference = float((output.detach().float() - reference.float()).abs().max())
    if difference != 0 or not torch.isfinite(output).all():
        raise RuntimeError(f"native FA3/custom-op forward parity gate failed: {difference}")
    output.float().square().mean().backward()
    if any(value.grad is None or not torch.isfinite(value.grad).all() for value in values):
        raise RuntimeError("native FA3 autograd gradient gate failed")
    print(json.dumps({"event": "fa3_native_autograd_gpu_gate", "rank": os.environ.get("RANK"), "forward_max_abs_difference": difference, "qkv_gradient_finite": True, "backend": "_flash_3", "fallback": False}), flush=True)
