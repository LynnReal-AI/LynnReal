"""Causal W2 inference: self-generated history and three flow-map transitions."""
from dataclasses import replace
import torch


class FastPipeline:
    def __init__(self,config,device=None):
        from ..backends.h3.runtime import initialize
        from ..models.h3 import build_model
        from ..models.codecs import LynnRealDecoder
        self.config=config;self.device=device or initialize()
        self.model,self.adaln_memory=build_model(config,self.device,role='student')
        self.decoder=LynnRealDecoder(config['decoder'],self.device,compiled=config.get('decoder_compile',True),
            height=config['video']['height'],width=config['video']['width'])

    @torch.no_grad()
    def stream(self,batch,*,seed=42):
        from inference_core import Cache,Control,layout,student_layout,local_hand_positions,window_positions
        from lynnreal_core.backends.minimax_h3.layout import patchify_video,unpatchify_video
        from inference_core import ReferenceStudent as Student
        length=batch.target_latents.shape[1];device=self.device
        policy=Student(self.model,device,'HAND',lambda *a,**k:None,
                       latent_frames=length,window_chunks=2,maps=3)
        doc=student_layout(layout(batch,device,'HAND',latent_frames=length))
        prompt=batch.prompt_embeds[None].to(device,torch.bfloat16);empty=prompt[:,:0]
        gen=torch.Generator(device=device).manual_seed(seed)
        anchor=patchify_video(batch.anchor_latents[None].to(device).float())
        anchor=.999*anchor+.001*torch.randn(anchor.shape,device=device,generator=gen)
        cache=Cache();base=Control('prefix',0,cache,doc,doc.positions,
            batch.camera_viewmats[None].to(device).float(),batch.camera_K[None].to(device).float())
        policy.call(base,anchor,prompt,1.,1.)
        sigmas=policy.sigmas();count=(length+4)//5
        try:
            for index in range(count):
                cache.evict(index,2);n=min(5,length-index*5);rows=n*doc.rows_per_frame
                hand=patchify_video(batch.hand_latents[:,index*5:index*5+n][None].to(device).float())
                policy.call(replace(base,mode='hand',index=index,positions=local_hand_positions(doc,n)),hand,empty,1.,1.)
                control=replace(base,mode='noisy',index=index,positions=window_positions(doc,index,index,2))
                current=torch.randn((1,rows,96),device=device,generator=gen)
                for start,end in zip(sigmas[:-1],sigmas[1:]):
                    velocity=policy.call(control,current,empty,1-start,1-end)
                    current=current+(start-end)*velocity.float()
                z=unpatchify_video(current,n,doc.height,doc.width).contiguous().float()
                # Return owned frames: decoder's reusable output banks can be overwritten.
                yield self.decoder.decode(z).clone(),z.cpu()
                if index<count-1:policy.call(replace(control,mode='commit'),current,empty,1.,1.)
        finally:cache.clear()

    def generate(self,batch,*,seed=42):
        frames,latents=zip(*self.stream(batch,seed=seed))
        return torch.cat(frames),torch.cat(latents,2)
