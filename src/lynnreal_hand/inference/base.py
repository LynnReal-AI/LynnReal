"""Bidirectional Base generation on the native H3 timeline."""
from types import SimpleNamespace

import torch

from ..data import native_length
from .base_windows import CHUNK_FRAMES, HISTORY_LATENTS, WINDOW_FRAMES, WINDOW_LATENTS, window_spans
from .sampling import shift_sigma


class BasePipeline:
    def __init__(self,config,device=None):
        from ..backends.h3.runtime import initialize
        from ..models.h3 import build_model
        self.config=config;self.device=device or initialize()
        self.model,self.adaln_memory=build_model(config,self.device,role='base')

    @torch.no_grad()
    def generate_latents(self,batch,*,seed=42):
        if batch.pixel_frames <= WINDOW_FRAMES:
            return self._sample(batch, seed=seed)
        return self._sample_windows(batch, seed=seed).to(self.device)

    def _sample(self, batch, *, seed, history=None):
        from base_conditioning import captionless_without_hand, pack, predict

        self.model.to(self.device)
        device = self.device
        shape = (1, *batch.target_latents.shape)
        generator = torch.Generator(device=device).manual_seed(seed)
        z = torch.randn(shape, device=device, generator=generator)
        known = 0
        if history is not None:
            known = history.shape[2]
            history = history.to(device=device, dtype=z.dtype)
            z[:, :, :known] = .999 * history + .001 * z[:, :, :known]

        options = self.config['sampling']
        schedule = shift_sigma(torch.linspace(1, 0, options['schedule_points'], device=device),
                               options.get('shift', 3.0))
        scale = options.get('guidance_scale', 3.0)
        for sigma, next_sigma in zip(schedule[:-1], schedule[1:]):
            packed = pack(batch, z, 1 - float(sigma), device=device, latent_frames=shape[2],
                          seed=seed, history_frames=known)
            velocity = predict(self.model, packed)
            if scale != 1:
                negative = predict(self.model, captionless_without_hand(packed))
                velocity = negative + scale * (velocity - negative)
            z[:, :, known:] += (sigma - next_sigma) * velocity[:, :, known:].float()
        if history is not None:
            z[:, :, :known] = history
        return z

    def _sample_windows(self, batch, *, seed):
        from ..models.codecs import decode_native, encode_native, native_encoder
        from h3_components.loader import load_conditioners
        from lynnreal_core.backends.minimax_h3.official_codec import encode_joint_prompt_condition

        foundation = self.config['conditioning']['checkpoint_path']
        spans = window_spans(batch.pixel_frames)
        full_hand = batch.hand_latents
        tail = getattr(batch, 'base_hand_tail', None)
        if tail is not None:
            full_hand = torch.cat((full_hand, tail), dim=1)

        pieces = []
        previous = None
        vae = text = None
        for start, end in spans:
            offset = start // CHUNK_FRAMES * HISTORY_LATENTS
            if previous is None:
                prompt, tags, anchor = batch.prompt_embeds, batch.text_token_tags, batch.anchor_latents
                history = None
            else:
                self.model.to('cpu')
                torch.cuda.empty_cache()
                if vae is None:
                    vae = native_encoder(foundation, self.device)
                else:
                    vae.to(self.device)
                first = (start - previous_start) // CHUNK_FRAMES * HISTORY_LATENTS
                before = max(0, first - HISTORY_LATENTS)
                after = first + 2 * HISTORY_LATENTS
                rgb = decode_native(vae, previous[:, :, before:after], self.device)
                image = rgb[(first - before) // HISTORY_LATENTS * CHUNK_FRAMES].numpy()
                anchor = encode_native(vae, image[None], self.device, seed=seed)
                vae.to('cpu')
                torch.cuda.empty_cache()
                if text is None:
                    text = load_conditioners(dict(checkpoint_path=foundation, torch_dtype='bfloat16',
                                                   local_files_only=True), device=self.device, qwen=True,
                                             video_vae=False, audio_vae=False, schedulers=False)
                else:
                    text.text_encoder.to(self.device)
                prompt, tags = encode_joint_prompt_condition(
                    image, batch.caption, processor=text.processor, tokenizer=text.tokenizer,
                    text_encoder=text.text_encoder, device=self.device)
                text.text_encoder.to('cpu')
                torch.cuda.empty_cache()
                history = pieces[-1][:, :, -HISTORY_LATENTS:]

            window = _window_batch(batch, full_hand, offset, prompt, tags, anchor)
            previous = self._sample(window, seed=seed, history=history).cpu()
            previous_start = start
            keep = native_length(end - start, 'stage1')
            pieces.append(previous[:, :, HISTORY_LATENTS if history is not None else 0:keep].clone())

        joined = torch.cat(pieces, dim=2)
        if joined.shape[2] != batch.target_latents.shape[1]:
            raise ValueError('Sliding windows do not cover the source latent timeline')
        return joined

    @torch.no_grad()
    def generate(self,batch,*,seed=42):
        from ..models.codecs import native_encoder,decode_native
        z=self.generate_latents(batch,seed=seed)
        self.model.to('cpu');torch.cuda.empty_cache()
        vae=native_encoder(self.config['conditioning']['checkpoint_path'],self.device)
        images=decode_native(vae,z,self.device,output_height=self.config['video']['height'])
        return images[:batch.pixel_frames],z.cpu()


def _window_batch(batch, hand, offset, prompt, tags, anchor):
    """Slice phase-aligned conditions and express camera poses in the local frame."""
    stop = offset + WINDOW_LATENTS
    hand_idx = torch.arange(offset, stop, device=hand.device).clamp_max(hand.shape[1] - 1)
    hand = hand.index_select(1, hand_idx)
    view_idx = torch.arange(offset, stop, device=batch.camera_viewmats.device)
    view_idx.clamp_max_(batch.camera_viewmats.shape[0] - 1)
    views = batch.camera_viewmats.index_select(0, view_idx).float()
    views = views @ torch.linalg.inv(views[0])
    views[0] = torch.eye(4, device=views.device, dtype=views.dtype)
    intrinsics = batch.camera_K.index_select(0, view_idx.to(batch.camera_K.device))
    return SimpleNamespace(
        task=batch.task, prompt_embeds=prompt, text_token_tags=tags, anchor_latents=anchor,
        hand_latents=hand, target_latents=hand, camera_viewmats=views, camera_K=intrinsics)
