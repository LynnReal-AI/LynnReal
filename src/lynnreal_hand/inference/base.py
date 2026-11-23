"""Bidirectional full-sequence generation with native hand conditions."""
import torch
from .sampling import shift_sigma


class BasePipeline:
    def __init__(self,config,device=None):
        from ..backends.h3.runtime import initialize
        from ..models.h3 import build_model
        self.config=config;self.device=device or initialize()
        self.model,self.adaln_memory=build_model(config,self.device,role='base')

    @torch.no_grad()
    def generate_latents(self,batch,*,seed=42):
        from base_conditioning import pack,captionless_without_hand,predict
        self.model.to(self.device)
        device=self.device;shape=(1,*batch.target_latents.shape)
        gen=torch.Generator(device=device).manual_seed(seed)
        z=torch.randn(shape,device=device,generator=gen)
        schedule=shift_sigma(torch.linspace(1,0,self.config['sampling']['schedule_points'],device=device),
                             self.config['sampling'].get('shift',3.))
        scale=self.config['sampling'].get('guidance_scale',3.)
        for sigma,next_sigma in zip(schedule[:-1],schedule[1:]):
            packed=pack(batch,z,1-float(sigma),device=device,latent_frames=shape[2],seed=seed)
            velocity=predict(self.model,packed)
            if scale!=1:
                negative=predict(self.model,captionless_without_hand(packed))
                velocity=negative+scale*(velocity-negative)
            z=z+(sigma-next_sigma)*velocity.float()
        return z

    @torch.no_grad()
    def generate(self,batch,*,seed=42):
        from ..models.codecs import native_encoder,decode_native
        z=self.generate_latents(batch,seed=seed)
        self.model.to('cpu');torch.cuda.empty_cache()
        vae=native_encoder(self.config['conditioning']['checkpoint_path'],self.device)
        images=decode_native(vae,z,self.device,output_height=self.config['video']['height'])
        return images,z.cpu()
