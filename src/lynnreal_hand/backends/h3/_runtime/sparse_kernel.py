"""Exact local HAND softmax partition, fused with the dense RGB partition.

Visibility is intentionally changed; algebraically matches that sparse mask.
Inputs already include native MM-RoPE and camera PRoPE. No temporal mixing.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _local_merge(Q, K, V, MASK, NEIGHBORS, DENSE, LSE, OUT,
                 N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                 DEG: tl.constexpr, BK: tl.constexpr, BQ: tl.constexpr):
    r = tl.program_id(0) * BQ + tl.arange(0, BQ)
    token, head = r // H, r % H
    d = tl.arange(0, D)
    base = tl.load(DENSE + r[:, None] * D + d[None, :], r[:, None] < N * H, 0).to(tl.float32)
    j = tl.arange(0, BK)
    ids = tl.load(NEIGHBORS + token[:, None] * DEG + j[None, :],
                  (token[:, None] < N) & (j[None, :] < DEG), -1)
    valid = (ids >= 0) & (j[None, :] < DEG)
    foreground = tl.load(MASK + tl.maximum(ids, 0), valid, 0).to(tl.int1) & valid
    has = tl.sum(foreground.to(tl.int32), 1) > 0
    if tl.sum(has.to(tl.int32), 0) > 0:
        q = tl.load(Q + r[:, None] * D + d[None, :], r[:, None] < N * H, 0).to(tl.float32)
        offsets = (tl.maximum(ids, 0)[:, :, None] * H + head[:, None, None]) * D + d[None, None, :]
        k = tl.load(K + offsets, foreground[:, :, None], 0).to(tl.float32)
        logits = tl.sum(q[:, None, :] * k, 2) * (D ** -0.5)
        logits = tl.where(foreground, logits, float('-inf'))
        maximum = tl.where(has, tl.max(logits, 1), 0.)
        p = tl.exp(logits - maximum[:, None])
        den = tl.sum(p, 1)
        v = tl.load(V + offsets, foreground[:, :, None], 0).to(tl.float32)
        local = tl.sum(p[:, :, None] * v, 1) / tl.maximum(den[:, None], 1e-20)
        lh = tl.where(has, maximum + tl.log(tl.maximum(den, 1e-20)), float('-inf'))
        ld = tl.load(LSE + head * N + token, token < N, 0)
        wh = tl.sigmoid(lh - ld)
        base = base.to(tl.float32) * (1. - wh[:, None]) + local * wh[:, None]
    tl.store(OUT + r[:, None] * D + d[None, :], base, r[:, None] < N * H)


def local_merge(q, hk, hv, mask, neighbors, dense, lse):
    assert q.shape[0] == 1 and q.shape[-1] == 128
    n, heads, d = q.shape[1:]
    deg = neighbors.shape[1]
    bq = 16 if deg == 1 else 4 if deg <= 9 else 2
    out = torch.empty_like(q)
    _local_merge[(triton.cdiv(n * heads, bq),)](
        q, hk, hv, mask, neighbors, dense, lse.contiguous(), out,
        n, heads, d, deg, triton.next_power_of_2(deg), bq, num_warps=4)
    return out


def merge_partitions(a, la, b, lb):
    weight = torch.sigmoid(lb - la).transpose(1, 2).unsqueeze(-1)
    return (a.float() * (1 - weight) + b.float() * weight).to(a.dtype)
