"""Validate sparse partition algebra and frame isolation against FP32 math."""
import json
from pathlib import Path
import torch
from flash_attn_interface import flash_attn_func
from sparse_kernel import local_merge
from sparse_hand import neighbor_table, hand_attention, configure

E = Path(__file__).resolve().parent


def relative(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8))


@torch.inference_mode()
def main():
    torch.manual_seed(271)
    frames, height, width, heads, dim = 5, 3, 4, 4, 128
    n = frames * height * width
    q = torch.randn(1, n, heads, dim, device='cuda', dtype=torch.bfloat16)
    hk = torch.randn_like(q); hv = torch.randn_like(q)
    dk = torch.randn(1, 11, heads, dim, device='cuda', dtype=torch.bfloat16)
    dv = torch.randn_like(dk)
    dense, dl = flash_attn_func(q, dk, dv, return_attn_probs=True)
    fg = torch.rand(n, device='cuda') > .5
    rows = []
    for radius in (0, 1, 2):
        neighbors = neighbor_table(frames, height, width, radius, q.device)
        sparse = local_merge(q, hk, hv, fg, neighbors, dense, dl)
        keys = torch.cat((dk, hk), 1); values = torch.cat((dv, hv), 1)
        allowed = torch.zeros(n, n, dtype=torch.int32, device='cuda')
        safe = neighbors.clamp_min(0).long()
        allowed.scatter_add_(1, safe, ((neighbors >= 0) & fg[safe]).to(torch.int32))
        allowed = allowed > 0
        mask = torch.cat((torch.ones(n, 11, dtype=torch.bool, device='cuda'), allowed), 1)
        logits = torch.einsum('bqhd,bkhd->bhqk', q.float(), keys.float()) / dim ** .5
        logits.masked_fill_(~mask[None, None], float('-inf'))
        ref = torch.einsum('bhqk,bkhd->bqhd', logits.softmax(-1), values.float())
        err = relative(sparse, ref)
        assert err < .006, (radius, err)
        # Perturb every other HAND frame: first-frame queries must be bitexact.
        altered = hv.clone(); altered[:, height*width:] += 7
        other = local_merge(q, hk, altered, fg, neighbors, dense, dl)
        assert torch.equal(sparse[:, :height*width], other[:, :height*width])
        empty = local_merge(q, hk, hv, torch.zeros_like(fg), neighbors, dense, dl)
        assert torch.equal(empty, dense)
        rows.append(dict(radius=radius,relative_L2_vs_FP32_sparse_math=err,
                         other_HAND_frame_cannot_affect_first_frame=True,
                         all_black_HAND_exactly_dense_partition=True))
    # Packed HAND self-attention must have the same five disjoint frame groups.
    configure(pack=True, hand_frame=True)
    indices = torch.tensor([0, 2, 12, 15, 17, 24, 36, 37, 48, 49, 50], device='cuda')
    cu = torch.tensor([0, 2, 5, 6, 8, 11], device='cuda', dtype=torch.int32)
    cache = type('Cache', (), dict(hand_cu=cu,hand_valid_count=len(indices)))()
    qp, kp, vp = [a.index_select(1, indices) for a in (q, hk, hv)]
    actual = hand_attention(qp, kp, vp, cache, height*width)
    expected = []
    for a, b in zip(cu.cpu().tolist()[:-1], cu.cpu().tolist()[1:]):
        score = torch.einsum('bqhd,bkhd->bhqk', qp[:,a:b].float(),kp[:,a:b].float()) / dim ** .5
        expected.append(torch.einsum('bhqk,bkhd->bqhd',score.softmax(-1),vp[:,a:b].float()))
    error = relative(actual, torch.cat(expected,1));assert error < .006,error
    padded = [torch.cat((a,torch.randn(1,5,heads,dim,device=a.device,dtype=a.dtype)),1) for a in(qp,kp,vp)]
    padded_out=hand_attention(*padded,cache,height*width)
    assert torch.equal(actual,padded_out[:,:len(indices)])
    assert torch.count_nonzero(padded_out[:,len(indices):]).item()==0
    value = dict(passed=True,bucket_padding_valid_rows_bitexact=True,padding_rows_zero=True,local_partitions=rows,packed_HAND_self_relative_L2=error,
                 scope='algebra matches the explicit sparse mask; mask itself changes model visibility')
    (E/'SPARSE_KERNEL_GATE.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value),flush=True)


if __name__ == '__main__':
    main()
