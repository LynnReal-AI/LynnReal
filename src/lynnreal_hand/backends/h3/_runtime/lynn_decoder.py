"""Differentiable Ultra-Fast-VAE weights, native 16x16 tiles on the 540p canvas.

No INT8/custom inference kernels, integer RGB packing, CUDA graphs or decoder
parameter updates. All normalization/RoPE precision follows the float package.
"""
import importlib.util
import json
import os
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class AutogradFA3:
    def __call__(self, attn, hidden_states, rotary_emb=None):
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        batch, tokens, _ = hidden_states.shape
        qkv = attn.to_qkv(hidden_states).reshape(batch, tokens, 3, attn.heads, attn.dim_head)
        query, key, value = [qkv[:, :, i] for i in range(3)]
        query = attn.norm_q(query.float()).to(query.dtype)
        key = attn.norm_k(key.float()).to(key.dtype)
        if rotary_emb is not None:
            cos, sin = (value.to(query.dtype) for value in rotary_emb)
            width = cos.shape[-1]

            def rotate(x):
                left, right = x[..., :width].chunk(2, -1)
                return torch.cat((x[..., :width] * cos + torch.cat((-right, left), -1) * sin,
                                  x[..., width:]), -1).contiguous()

            query, key = rotate(query), rotate(key)
        result = dispatch_attention_fn(query.contiguous(), key.contiguous(), value.contiguous(),
                                       attn_mask=None, dropout_p=0., is_causal=False,
                                       backend='_flash_3', parallel_config=None)
        return attn.to_out[0](result.reshape(batch, tokens, attn.heads * attn.dim_head))


def load_architecture(package):
    # Architecture belongs to the code release; HF packages contain weights/config only.
    import ultra_fast_vae.model as module
    return module


class LynnDecoder(nn.Module):
    def __init__(self, package, *, height=544, width=960, checkpoint_blocks=True):
        super().__init__()
        from safetensors.torch import load_file
        package = os.environ.get('DUOKL_LYNN_PACKAGE', package)
        self.package = Path(package)
        self.config = json.loads((self.package / 'config.json').read_text())
        self.arch = load_architecture(self.package)
        self.decoder = self.arch.ThinDecoder(self.config['model'], device='meta')
        self.post = nn.Conv3d(24, 24, 1, device='meta', dtype=torch.float16)
        state = load_file(str(self.package / 'weights.safetensors'), device='cpu')
        self.decoder.load_state_dict({k[len('decoder.'):]: v for k, v in state.items()
                                     if k.startswith('decoder.')}, strict=True, assign=True)
        self.post.load_state_dict({k[len('post.'):]: v for k, v in state.items()
                                  if k.startswith('post.')}, strict=True, assign=True)
        for block in self.decoder.transformer_blocks:
            block.attn.set_processor(AutogradFA3())
        self.height, self.width = height, width
        self.ys, self.hs, self.yo = self.arch.split_tiles(height, 256, 64)
        self.xs, self.ws, self.xo = self.arch.split_tiles(width, 256, 64)
        if any(value != 256 for value in self.hs + self.ws):
            raise ValueError('Ultra-Fast-VAE must retain its trained full16x16 latent tiles')
        norm = self.config['normalization']
        for name, key in [('latent_mean', 'latent_mean'), ('latent_std', 'latent_std'),
                          ('pixel_mean', 'pixel_mean'), ('pixel_std', 'pixel_std')]:
            values = norm[key]
            self.register_buffer(name, torch.tensor(values).view(1, len(values), 1, 1, 1))
        self.checkpoint_blocks = checkpoint_blocks
        self.eval().requires_grad_(False)

    def tile_decode(self, z):
        b, c, t, h, w = z.shape
        decoder = self.decoder
        x = decoder.proj_in(z.permute(0, 2, 3, 4, 1).reshape(b, t*h*w, c))
        count = x.shape[1]
        x = torch.cat((x, decoder.register_tokens.expand(b, -1, -1), torch.zeros_like(x[:, :1])), 1)
        grids = [2 * torch.arange(.5, n, device=x.device, dtype=torch.float32) / n - 1 for n in (t, h, w)]
        ids = torch.stack(torch.meshgrid(*grids, indexing='ij'), -1).flatten(0, 2)[None].expand(b, -1, -1)
        ids = torch.cat((ids, ids.new_zeros((b, decoder.num_register_tokens+1, 3))), 1)
        rotary = decoder.rope(ids)
        for block in decoder.transformer_blocks:
            if torch.is_grad_enabled() and x.requires_grad and self.checkpoint_blocks:
                x = checkpoint(block, x, rotary, use_reentrant=False)
            else:
                x = block(x, rotary)
        x = decoder.proj_out(decoder.norm_out(x.float()).to(x.dtype))[:, :count]
        p, pt = decoder.patch_size, decoder.patch_size_t
        x = x.reshape(b, t, h, w, 3, pt, p, p).permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.reshape(b, 3, t*pt, h*p, w*p)

    def forward(self, current):
        if current.shape[0:2] != (1, 24) or current.shape[-2:] != (self.height//16, self.width//16):
            raise ValueError('Lynn decoder canvas/channel contract differs')
        length = current.shape[2]
        if length not in (2, 5):
            raise ValueError('Only a native full5 chunk or final2 tail is allowed')
        z = current.float() * self.latent_std + self.latent_mean
        if length == 2:
            # Internal zero computation only; no padded rollout/score/GT frames.
            z = F.pad(z, (0, 0, 0, 0, 0, 3))
        z = z.to(self.post.weight.dtype)
        tiles = [z[..., y//16:(y+h)//16, x//16:(x+w)//16]
                 for y, h in zip(self.ys, self.hs) for x, w in zip(self.xs, self.ws)]
        raw = self.tile_decode(self.post(torch.cat(tiles)))[:, :, 3:3 + (17 if length == 5 else 5)]
        pieces = list(raw.split(1))
        rows = [pieces[i:i+len(self.xs)] for i in range(0, len(pieces), len(self.xs))]
        output = self.arch.stitch(rows, self.yo, self.xo)
        return (output.float() * self.pixel_std + self.pixel_mean).clamp(0, 1)
