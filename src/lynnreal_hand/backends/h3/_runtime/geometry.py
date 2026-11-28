"""Per-sample HAND/i2v/ref2v geometry with W2 raw-K re-encoding.

Bindings are changed only in this new process; frozen historical files are intact.
"""
from dataclasses import replace
import numpy as np
import torch
import inference_core as core

_window = core.window_positions


def layout(batch, device, phase, with_hand=True, latent_frames=37, *, target_rope_gap=None):
    from h3_components.layout import build_condition_layout, H3VisualObservationSpec
    frames = int(batch.target_latents.shape[1])
    h,w = batch.target_latents.shape[-2:]
    task = getattr(batch,'task','i2v_cam_hand')
    anchor = task in ('i2v','i2v_cam','i2v_cam_hand')
    hand = with_hand and getattr(batch,'hand_latents',None) is not None
    camera = task in ('i2v_cam_hand','i2v_cam')
    key=(str(device),phase,with_hand,latent_frames,task,target_rope_gap)
    if not hasattr(batch,'_inference_layout_cache'): batch._inference_layout_cache={}
    if key in batch._inference_layout_cache: return batch._inference_layout_cache[key]
    assert frames >= latent_frames and h%2 == w%2 == 0
    source=build_condition_layout(batch.text_token_tags.cpu().numpy(),frames,h,w,0,
        keyframe_anchors=('first',) if anchor else (),
        observations=(H3VisualObservationSpec(frames,h,w,tuple(range(frames))),) if hand else ())
    L=len(source.text_indices); rows=h*w//4; A=rows if anchor else 0
    positions=source.position_ids.copy(); cond=source.condition_video_indices
    if hand:
        spans=np.array([(5./3.)*(1,4,4,4,4)[i%5] for i in range(frames)],np.float64)
        gap=spans.sum() if target_rope_gap is None else target_rope_gap
        times=L+gap+np.concatenate(([0.],spans[:-1].cumsum()))
        positions[source.target_video_indices,0]=np.repeat(times,rows)
    if anchor: positions[cond[:A],0]=positions[source.target_video_indices[:A],0]
    prefix=np.concatenate((source.text_indices,cond[:A]))
    target=positions[source.target_video_indices[:latent_frames*rows]]
    if hand:
        hp=positions[cond[A:A+latent_frames*rows]]
    else:
        # A virtual grid lets the same local-clock code run without adding HAND
        # tokens or a fictitious HAND span to any attention sequence.
        hp=target.copy(); hp[:,0]-=17*(5/3)
    doc=core.Layout(torch.as_tensor(positions[prefix],device=device),
        torch.as_tensor(source.token_tags[prefix],device=device),L,A,hand,camera,
        torch.as_tensor(target,device=device),torch.as_tensor(hp,device=device),h,w)
    doc.window_chunks=2
    batch._inference_layout_cache[key]=doc
    return doc


def student_layout(doc):
    positions=doc.positions.clone()
    if doc.anchor_rows:
        positions[doc.text_rows:,0]=(doc.hand_positions[0,0] if doc.hand else doc.target_positions[0,0])
    result=replace(doc,positions=positions)
    result.window_chunks=getattr(doc,'window_chunks',2)
    return result


def window_positions(doc,index,physical_chunk,window=None):
    return _window(doc,index,physical_chunk,window or getattr(doc,'window_chunks',2))


def install_geometry():
    core.layout=layout
    core.student_layout=student_layout
    core.window_positions=window_positions
