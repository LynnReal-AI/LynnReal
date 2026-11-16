"""Build conditioning caches from replaceable image/video/camera examples."""
from pathlib import Path
import gc
import os
import numpy as np
import torch
from PIL import Image, ImageOps
from .config import canvas
from .data import load_example, save_cache, NATIVE_LENGTHS
from .media import read_video


@torch.no_grad()
def prepare_example(config, example_path, output, *, device='cuda:0'):
    from .backends.h3.runtime import activate
    from .models.codecs import native_encoder, encode_native, TinyHandEncoder
    activate(); device=torch.device(device)
    from h3_components.loader import load_conditioners
    from lynnreal_core.backends.minimax_h3.official_codec import encode_joint_prompt_condition
    sample=load_example(example_path,verify_media=True); height,width=canvas(config)
    frames=sample['frames']; length=NATIVE_LENGTHS[frames]
    foundation=config['conditioning']['checkpoint_path']
    modules=load_conditioners(dict(checkpoint_path=foundation,torch_dtype='bfloat16',local_files_only=True),
        device=device,qwen=True,video_vae=False,audio_vae=False,schedulers=False)
    image=ImageOps.fit(Image.open(sample['image']).convert('RGB'),(1344,768),Image.Resampling.LANCZOS)
    image=image.resize((width,height),Image.Resampling.LANCZOS)
    prompt,tags=encode_joint_prompt_condition(image,sample['caption'],processor=modules.processor,
        tokenizer=modules.tokenizer,text_encoder=modules.text_encoder,device=device)
    del modules;gc.collect();torch.cuda.empty_cache()
    vae=native_encoder(foundation,device)
    first=np.asarray(image)[None].copy()
    anchor=encode_native(vae,first,device,seed=sample.get('seed',42))
    hand_pixels=read_video(sample['hand_video'],frames=frames,height=height,width=width)
    native_hand=encode_native(vae,hand_pixels,device,seed=42) if config['stage']=='stage1' else None
    del vae;gc.collect();torch.cuda.empty_cache()
    if config['stage']=='stage3':
        encoder=TinyHandEncoder(config['hand_encoder'],device);pieces=[]
        for start in range(0,frames,17):
            x=torch.from_numpy(hand_pixels[start:start+17]).permute(0,3,1,2)[None].to(device).float()/255
            pieces.append(encoder(x)[0].cpu().bfloat16())
        hand=torch.cat(pieces,dim=1).contiguous();del encoder
    else:hand=native_hand
    if tuple(hand.shape)!=(24,length,height//16,width//16):raise ValueError('Native hand encoder geometry differs')
    camera=np.load(sample['camera'],allow_pickle=False)
    poses=camera['camera_viewmats'];intrinsics=camera['camera_K']
    if poses.shape!=(length,4,4) or intrinsics.shape!=(length,3,3):
        raise ValueError('camera.npz must contain latent-aligned camera_viewmats[L,4,4] and camera_K[L,3,3]')
    tensors=dict(prompt_embeds=prompt,text_token_tags=tags,anchor_latents=anchor,hand_latents=hand,
        camera_viewmats=torch.from_numpy(poses).float(),camera_K=torch.from_numpy(intrinsics).float(),
        source_frame_indices=torch.arange(frames))
    output=Path(output).resolve()
    metadata=dict(sample_id=sample['id'],caption=sample['caption'],pixel_frames=frames,fps=sample['fps'],
        target_shape=list(hand.shape),task='i2v_cam_hand',variant=config['stage'],
        source_hand_video=os.path.relpath(sample['hand_video'],output),
        source_video=os.path.relpath(sample['target_video'],output) if sample.get('target_video') else None,
        source_example=os.path.relpath(Path(example_path).resolve(),output),
        geometry='native H3, latent-frame-aligned physical camera poses',inference_only=True)
    return save_cache(tensors,metadata,output)
