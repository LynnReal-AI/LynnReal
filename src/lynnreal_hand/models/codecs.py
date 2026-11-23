"""Native H3 conditioning, optimized hand encoding and LynnReal-VAE decoding."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class TinyHandEncoder(nn.Module):
    def __init__(self, directory, device):
        super().__init__()
        from safetensors.torch import load_file
        from .tiny_hand import TAEHV, MemBlock, SuperMemBlock
        model = TAEHV(checkpoint_path=None, arch_name='taeh3')
        state = load_file(str(Path(directory)/'encoder.safetensors'),device='cpu')
        model.encoder.load_state_dict(state, strict=True)
        self.network = model.encoder
        self.preprocess = model.preprocess_input_frames
        self.memory_types = (MemBlock,SuperMemBlock)
        self.to(device).eval().requires_grad_(False)
        self.to(memory_format=torch.channels_last)

    def forward(self, frames):
        if frames.shape[1] not in (17,5): raise ValueError('Hand encoder takes 17 or 5 genuine frames')
        x = self.preprocess(F.pad(frames,(0,0,0,0,0,0,3,0)))
        batch=x.shape[0]; x=x.flatten(0,1)
        with torch.autocast('cuda',dtype=torch.float16):
            for layer in self.network:
                if isinstance(layer,self.memory_types):
                    f=x.reshape(batch,-1,*x.shape[1:])
                    x=layer(x,torch.cat((f[:,:1]*0,f[:,:-1]),1).flatten(0,1))
                else: x=layer(x)
        return x.reshape(batch,-1,*x.shape[1:]).transpose(1,2).float()


def native_encoder(foundation, device):
    from ..backends.h3.runtime import activate, configure_native_vae
    activate()
    from diffusers import AutoencoderKLMiniMaxH3
    vae=AutoencoderKLMiniMaxH3.from_pretrained(str(foundation),subfolder='vae',
        torch_dtype=torch.bfloat16,local_files_only=True).eval().requires_grad_(False).to(device)
    return configure_native_vae(vae)


@torch.no_grad()
def encode_native(vae, pixels, device, *, seed=42):
    from lynnreal_core.backends.minimax_h3.official_codec import OfficialH3Codec
    codec=OfficialH3Codec.__new__(OfficialH3Codec);codec.video_vae=vae;codec.device=device
    x=torch.from_numpy(pixels).permute(3,0,1,2)[None].to(device)
    with torch.autocast('cuda',dtype=torch.float16):
        return codec._video_latents(x,seed=seed)[0]


class LynnRealDecoder:
    def __init__(self, directory, device, *, compiled=True, height=540, width=960):
        from ..backends.h3.runtime import activate
        activate()
        if (height, width) not in ((540, 960), (768, 1344)):
            raise ValueError('Unsupported Fast decoder resolution')
        self.compiled=compiled
        self.height=height
        canvas_height=544 if height==540 else height
        if compiled:
            if height==540:
                from codec540 import Decoder540
                self.impl=Decoder540(Path(directory),device=device)
            else:
                from ultra_fast_vae.inference import Decoder
                self.impl=Decoder(Path(directory),device=device)
            self.impl.warmup(torch.zeros((1,24,5,canvas_height//16,width//16),device=device))
        else:
            from lynn_decoder import LynnDecoder
            self.impl=LynnDecoder(directory,height=canvas_height,width=width).to(device)

    @torch.no_grad()
    def decode(self, z):
        length=z.shape[2]
        if self.compiled:
            if length==2: z=F.pad(z,(0,0,0,0,0,3)).contiguous()
            return self.impl.decode(z)[:17 if length==5 else 5]
        x=self.impl(z)[:,:,:17 if length==5 else 5]
        if self.height==540:x=x[:,:,:,2:-2]
        return (x[0].permute(1,2,3,0).clamp(0,1)*255).round().byte().cpu()


@torch.no_grad()
def decode_native(vae, latents, device, *, output_height=None):
    from video_decode import decode
    images=decode(vae,latents,device)[0].permute(1,2,3,0)
    if output_height==540 and images.shape[1]==544: images=images[:,2:-2]
    return (images.clamp(0,1)*255).round().byte().cpu()
