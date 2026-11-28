"""Base inference packing, joint CFG and output unpacking."""
from dataclasses import dataclass
import torch
from lynnreal_core.backends.minimax_h3.layout import patchify_video,unpatchify_video
from lynnreal_hand.inference.base_windows import ROPE_GAP
from geometry import layout

@dataclass
class Packed:
    inputs: dict
    row_times: torch.Tensor
    text_rows: int
    anchor_rows: int
    hand_rows: int
    frames: int
    height: int
    width: int

def finish_times(inputs, row_times):
    inputs = dict(inputs)
    unique, inverse = torch.unique(row_times, sorted=True, return_inverse=True)
    inputs.update(timestep=unique, timestep_indices=inverse)
    return inputs

def pack(batch, noisy, t, *, device, latent_frames, seed, history_frames=0):
    doc = layout(batch, device, 'HAND', with_hand=True, latent_frames=latent_frames,
                 target_rope_gap=ROPE_GAP)
    per_frame = doc.rows_per_frame
    rows = latent_frames * per_frame
    prompt = batch.prompt_embeds[None].to(device, torch.bfloat16)
    generator = torch.Generator(device=device).manual_seed(seed)
    chunks, positions, tags, times = [], [doc.positions], [doc.tags], []
    # Stage0.5 text rows follow the noisy target time, unlike a causal prefix.
    times.append(torch.full((doc.text_rows,), float(t), device=device))
    if doc.anchor_rows:
        anchor = batch.anchor_latents[None].to(device, torch.float32)
        augmented = .999 * anchor + .001 * torch.randn(anchor.shape, device=device, generator=generator)
        chunks.append(patchify_video(augmented))
        times.append(torch.full((doc.anchor_rows,), .999, device=device))
    if doc.hand:
        hand = batch.hand_latents[:, :latent_frames][None].to(device, torch.float32)
        augmented = .999 * hand + .001 * torch.randn(hand.shape, device=device, generator=generator)
        chunks.append(patchify_video(augmented))
        positions.append(doc.hand_positions)
        tags.append(torch.zeros(rows, device=device, dtype=torch.long))
        times.append(torch.full((rows,), .999, device=device))
    chunks.append(patchify_video(noisy))
    positions.append(doc.target_positions)
    tags.append(torch.zeros(rows, device=device, dtype=torch.long))
    target_times = torch.full((rows,), float(t), device=device)
    target_times[:history_frames*per_frame] = .999
    times.append(target_times)
    row_times = torch.cat(times)
    video_indices = torch.arange(doc.text_rows, row_times.numel(), device=device)
    frame_ids = torch.cat(([torch.zeros(doc.anchor_rows, device=device, dtype=torch.long)] if doc.anchor_rows else []) +
                          [torch.arange(latent_frames, device=device).repeat_interleave(per_frame)] * (1 + int(doc.hand)))
    inputs = dict(hidden_states=torch.cat(chunks, 1), encoder_hidden_states=prompt,
                  audio_hidden_states=noisy.new_empty((1, 0, 32)), token_tags=torch.cat(tags),
                  position_ids=torch.cat(positions), video_indices=video_indices,
                  audio_indices=video_indices[:0], text_indices=torch.arange(doc.text_rows, device=device),
                  fused_prope=doc.camera, prope_token_indices=video_indices if doc.camera else None,
                  prope_frame_ids=frame_ids if doc.camera else None,
                  cam_viewmats=batch.camera_viewmats[:latent_frames][None].to(device, torch.float32) if doc.camera else None,
                  cam_K=batch.camera_K[:latent_frames][None].to(device, torch.float32) if doc.camera else None,
                  stage0p5_sequence_parallel=False, return_dict=False)
    return Packed(finish_times(inputs, row_times), row_times, doc.text_rows, doc.anchor_rows,
                  rows if doc.hand else 0, latent_frames, doc.height, doc.width)

def captionless_without_hand(full):
    """Equivalent to source remove_hand then captionless_prompt_layout.

    Retained media keep the HAND clock offset. Removing caption rows shifts
    that entire media clock only by the removed Qwen-prefix length.
    """
    inputs = full.inputs
    device = full.row_times.device
    text_tags = inputs['token_tags'][:full.text_rows]
    visual = torch.nonzero(text_tags == 0).reshape(-1)
    if not visual.numel():
        raise ValueError('Image-conditioned joint CFG requires Qwen visual rows')
    new_text = int(visual[-1]) + 1
    delta = new_text - full.text_rows
    keep = torch.ones(full.row_times.numel(), device=device, dtype=torch.bool)
    if full.hand_rows:
        start = full.text_rows + full.anchor_rows
        keep[start:start + full.hand_rows] = False
    video_keep = keep[inputs['video_indices']]
    retained = torch.cat((torch.arange(new_text, device=device),
                          torch.nonzero(keep[full.text_rows:]).reshape(-1) + full.text_rows))
    positions = inputs['position_ids'][retained].clone()
    positions[new_text:, 0] += delta
    positions[:new_text] = 0
    positions[:new_text, 0] = torch.arange(new_text, device=device, dtype=positions.dtype)
    video_indices = torch.arange(new_text, retained.numel(), device=device)
    negative = dict(inputs)
    negative.update(hidden_states=inputs['hidden_states'][:, video_keep],
                    encoder_hidden_states=inputs['encoder_hidden_states'][:, :new_text],
                    position_ids=positions, token_tags=inputs['token_tags'][retained],
                    text_indices=torch.arange(new_text, device=device), video_indices=video_indices,
                    audio_indices=video_indices[:0],
                    prope_token_indices=video_indices if inputs['fused_prope'] else None,
                    prope_frame_ids=inputs['prope_frame_ids'][video_keep] if inputs['fused_prope'] else None)
    row_times = full.row_times[retained]
    return Packed(finish_times(negative, row_times), row_times, new_text, full.anchor_rows, 0,
                  full.frames, full.height, full.width)

def predict(model, packed):
    from lynnreal_hand.inference.adaln import select_adaln
    select_adaln(model, packed.inputs['timestep'])
    output, _ = model(**packed.inputs)
    rows = packed.frames * packed.height * packed.width // 4
    return unpatchify_video(output[:, -rows:], packed.frames, packed.height, packed.width).float()
