"""Native H3 temporal geometry, including the generated two-latent tail."""
import torch


def pixel_frames(latent_frames):
    whole,tail=divmod(latent_frames,5)
    if latent_frames<1:raise ValueError('A native H3 video must have at least one latent')
    return whole*17+(4*tail-3 if tail else 0)


@torch.no_grad()
def decode(vae,latent,device):
    mean=torch.tensor(vae.config.latents_mean,device=device,dtype=torch.float32).view(1,-1,1,1,1)
    std=torch.tensor(vae.config.latents_std,device=device,dtype=torch.float32).view(1,-1,1,1,1)
    with torch.autocast('cuda',dtype=torch.float16):
        x=vae.decode(latent.float()*std+mean,return_dict=False)[0]
    mean=torch.tensor([.485,.456,.406],device=device,dtype=torch.float32).view(1,3,1,1,1)
    std=torch.tensor([.229,.224,.225],device=device,dtype=torch.float32).view(1,3,1,1,1)
    x=(x.float()*std+mean).clamp(0,1)
    if x.shape!=(1,3,pixel_frames(latent.shape[2]),latent.shape[-2]*16,latent.shape[-1]*16) or not bool(x.isfinite().all()):
        raise ValueError('native H3 decode shape/value differs')
    return x
