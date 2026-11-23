"""Current RGB controls define sparse HAND routes; no generated RGB or Q/K routing."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
from sparse_kernel import local_merge, merge_partitions

SETTINGS = dict(name='dense', rgb='dense', pack=False, hand_frame=False, radius=1,
                mask_margin=1, temporal_mask='groups', threshold=24.)


def configure(**kwargs):
    SETTINGS.update(name='dense', rgb='dense', pack=False, hand_frame=False, radius=1,
                    mask_margin=1, temporal_mask='groups', threshold=24.)
    SETTINGS.update(kwargs)


def neighbor_table(frames, height, width, radius, device):
    t, y, x = np.meshgrid(np.arange(frames), np.arange(height), np.arange(width), indexing='ij')
    columns = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yy, xx = y + dy, x + dx
            valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
            columns.append(np.where(valid, t * height * width + yy * width + xx, -1).reshape(-1))
    return torch.as_tensor(np.stack(columns, 1), device=device, dtype=torch.int32)


def prepare(rgb, cache, rows):
    if SETTINGS['rgb'] == 'dense' and not SETTINGS['pack']:
        return
    assert rgb.shape == (1, 17, 3, 768, 1344) and rows == 1008
    # Native 17RGB ->5latent groups [0], [1:5], [5:9], [9:13], [13:17].
    # A spatial margin covers encoder receptive field beyond visible raster pixels.
    mask = F.max_pool2d((rgb[0].amax(1, keepdim=True) * 255 > SETTINGS['threshold']).float(), 32, 32)
    groups = [(0, 1), (1, 5), (5, 9), (9, 13), (13, 17)]
    if SETTINGS['temporal_mask'] == 'union':
        masks = mask.amax(0, keepdim=True).expand(5, -1, -1, -1)
    elif SETTINGS['temporal_mask'] == 'prefix':
        masks = torch.stack([mask[:end].amax(0) for _, end in groups])
    else:
        masks = torch.stack([mask[start:end].amax(0) for start, end in groups])
    margin = SETTINGS['mask_margin']
    if margin:
        masks = F.max_pool2d(masks, 2 * margin + 1, 1, margin)
    flat = masks.flatten().bool()
    if not hasattr(cache, 'hand_mask'):
        cache.hand_mask = torch.empty_like(flat)
        cache.hand_neighbors = neighbor_table(5, 24, 42, SETTINGS['radius'], rgb.device)
    cache.hand_mask.copy_(flat)
    indices = torch.nonzero(flat).flatten()
    # Keep one computational sentinel in empty frames; it is never visible to RGB.
    counts = masks.flatten(1).sum(1).to(torch.int32)
    sentinels = torch.nonzero(counts == 0).flatten() * rows
    indices = torch.sort(torch.cat((indices, sentinels))).values
    cache.hand_real_indices = indices
    cache.hand_valid_count = len(indices)
    bucket = next(n for n in (1024,1536,2048,3072,4096,5120) if n >= len(indices))
    cache.hand_token_indices = F.pad(indices, (0, bucket-len(indices)), value=0)
    cache.hand_compute_count = bucket
    cache.hand_fg_indices = torch.nonzero(flat).flatten()
    counts = torch.bincount(indices // rows, minlength=5).to(torch.int32)
    cache.hand_cu = torch.cat((counts.new_zeros(1), counts.cumsum(0).to(torch.int32)))
    cache.hand_pack_count = len(indices)
    cache.hand_kept_fraction = len(indices) / (5 * rows)


def subset_geometry(g, indices):
    return dict(qrope=tuple(x.index_select(0, indices) for x in g['qrope']),
                krope=tuple(x.index_select(0, indices) for x in g['krope']),
                qm=g['qm'].index_select(1, indices), km=g['km'].index_select(1, indices),
                output=g['output'].index_select(1, indices),
                qmask=g['qmask'].index_select(0, indices), kmask=g['kmask'].index_select(0, indices))


def hand_attention(q, k, v, cache, rows):
    if SETTINGS['hand_frame']:
        if SETTINGS['pack']:
            valid = cache.hand_valid_count
            out = flash_attn_varlen_func(q[0,:valid].contiguous(), k[0,:valid].contiguous(), v[0,:valid].contiguous(),
                                         cache.hand_cu, cache.hand_cu, rows, rows, causal=False)
            return F.pad(out, (0,0,0,0,0,q.shape[1]-valid))[None]
        _, _, heads, dim = q.shape
        return flash_attn_func(q.reshape(5, rows, heads, dim), k.reshape(5, rows, heads, dim),
                              v.reshape(5, rows, heads, dim), causal=False).reshape_as(q)
    return flash_attn_func(q.contiguous(), k.contiguous(), v.contiguous(), causal=False)


def rgb_attention(q, k, v, control):
    p = control.text_rows + control.rows
    t = 5 * control.rows
    cache = control.cache
    if SETTINGS['rgb'] == 'foreground':
        keep = p + cache.hand_fg_indices
        keys = torch.cat((k[:, :p], k.index_select(1, keep), k[:, p+t:]), 1)
        values = torch.cat((v[:, :p], v.index_select(1, keep), v[:, p+t:]), 1)
        return flash_attn_func(q.contiguous(), keys, values, causal=False)
    keys = torch.cat((k[:, :p], k[:, p+t:]), 1)
    values = torch.cat((v[:, :p], v[:, p+t:]), 1)
    dense, lse = flash_attn_func(q.contiguous(), keys, values, causal=False, return_attn_probs=True)
    if SETTINGS['rgb'] == 'local':
        return local_merge(q.contiguous(), k[:, p:p+t], v[:, p:p+t], cache.hand_mask,
                           cache.hand_neighbors, dense, lse)
    if SETTINGS['rgb'] == 'frame':
        # Diagnostic variable-length attention; graph capture remains disabled here.
        hidx = cache.hand_real_indices
        hk = k[:, p:p+t].index_select(1, hidx)[0]
        hv = v[:, p:p+t].index_select(1, hidx)[0]
        cuq = torch.arange(6, device=q.device, dtype=torch.int32) * control.rows
        out, hand_lse = flash_attn_varlen_func(q[0].contiguous(), hk, hv, cuq, cache.hand_cu,
                                             control.rows, control.rows, causal=False,
                                             return_attn_probs=True)
        return merge_partitions(dense, lse, out[None], hand_lse[None] if hand_lse.ndim == 2 else hand_lse)
    raise ValueError(SETTINGS)
