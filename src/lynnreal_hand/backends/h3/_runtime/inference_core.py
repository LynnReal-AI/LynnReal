"""H3 inference geometry, cache visibility and causal model invocation."""
from dataclasses import dataclass,field,replace
import numpy as np
import torch
from lynnreal_core.backends.minimax_h3.layout import patchify_video,unpatchify_video
ROPE_POLICY="v6_sliding_local_raw_K_reencode"

@dataclass
class Layout:
    positions: object
    tags: object
    text_rows: int
    anchor_rows: int
    hand: bool
    camera: bool
    target_positions: object
    hand_positions: object
    height: int
    width: int

    @property
    def rows_per_frame(self):return self.height*self.width//4

def student_layout(doc):
    """v6 prefix anchor occupies the first HAND time, independently of clip length."""
    if not doc.hand:raise ValueError('v6 local HAND clock requires HAND conditioning')
    positions=doc.positions.clone()
    positions[doc.text_rows:,0]=doc.hand_positions[0,0]
    return replace(doc,positions=positions)

def local_hand_positions(doc,length=5):
    """Every physical HAND chunk reuses the first chunk's temporal slots."""
    if not 1<=length<=5:raise ValueError('invalid native HAND chunk length')
    return doc.hand_positions[:length*doc.rows_per_frame]

def local_video_positions(doc,slot,length=5):
    """Match v6 Control.video_positions: one HAND span precedes local RGB slots."""
    if slot<0:raise ValueError('negative local window slot')
    positions=local_hand_positions(doc,length).clone()
    positions[:,0]+=(slot+1)*17*(5/3)
    return positions

def window_positions(doc,index,physical_chunk,window=3):
    first=max(0,index-window+1)
    if not first<=physical_chunk<=index:raise ValueError('chunk outside local KV window')
    frames=len(doc.target_positions)//doc.rows_per_frame
    length=min(5,frames-physical_chunk*5)
    return local_video_positions(doc,physical_chunk-first,length)

def layout(batch,device,phase,with_hand=True,latent_frames=37):
    from h3_components.layout import build_condition_layout,H3VisualObservationSpec
    source_frames=int(batch.target_latents.shape[1]);h,w=batch.target_latents.shape[-2:]
    if h%2 or w%2 or not 0<latent_frames<=source_frames:raise ValueError('invalid native spatial/temporal geometry')
    key=(str(device),phase,with_hand,latent_frames)
    if not hasattr(batch,'_inference_layout_cache'):batch._inference_layout_cache={}
    if key in batch._inference_layout_cache:return batch._inference_layout_cache[key]
    tags=batch.text_token_tags.detach().cpu().numpy();conditional=phase=='HAND';hand=conditional and with_hand
    per_frame=h*w//4;rows=latent_frames*per_frame
    source=build_condition_layout(tags,source_frames,h,w,0,
        keyframe_anchors=('first',) if conditional else (),
        observations=(H3VisualObservationSpec(source_frames,h,w,tuple(range(source_frames))),) if hand else ())
    L=len(source.text_indices);A=per_frame if conditional else 0;cond=source.condition_video_indices
    positions=source.position_ids.copy()
    # Exact released Ref2VA reference clock, then Solar anchor=target0.
    # HAND/target spatial coordinates are untouched at either canvas size.
    if hand:
        spans=[(5./3.)*(1,4,4,4,4)[i%5] for i in range(source_frames)]
        times=float(L)+sum(spans)+torch.cat((torch.zeros(1,dtype=torch.float64),torch.tensor(spans[:-1],dtype=torch.float64).cumsum(0)))
        positions[source.target_video_indices,0]=times.repeat_interleave(per_frame).numpy()
    if conditional:positions[cond[:A],0]=positions[source.target_video_indices[:A],0]
    prefix=np.concatenate((source.text_indices,cond[:A]))
    target=positions[source.target_video_indices[:rows]]
    hp=positions[cond[A:A+rows]] if hand else np.empty((0,3))
    doc=Layout(torch.as_tensor(positions[prefix],device=device),torch.as_tensor(source.token_tags[prefix],device=device),L,A,
        hand,conditional,torch.as_tensor(target,device=device),torch.as_tensor(hp,device=device),h,w)
    batch._inference_layout_cache[key]=doc
    return doc

@dataclass
class Cache:
    prefix: dict=field(default_factory=dict)
    hands: dict=field(default_factory=dict)
    history: dict=field(default_factory=dict)
    cameras: dict=field(default_factory=dict)

    def snapshot(self):
        return Cache(dict(self.prefix),{k:dict(v) for k,v in self.hands.items()},
                     {k:dict(v) for k,v in self.history.items()},self.cameras)

    def clear(self):self.prefix.clear();self.hands.clear();self.history.clear();self.cameras.clear()

    def evict(self,index,window_chunks):
        if window_chunks is None:return
        for parts in self.history.values():
            for j in list(parts):
                if j<index-window_chunks+1:del parts[j]
        for parts in self.hands.values():
            for j in list(parts):
                if j<index:del parts[j]

@dataclass
class Control:
    mode: str
    index: int
    cache: Cache
    layout: Layout
    positions: object
    views: object
    K: object

    def prefix_camera(self):
        if "prefix" in self.cache.cameras:return self.cache.cameras["prefix"]
        L=self.layout.text_rows;P=L+self.layout.anchor_rows
        v=torch.eye(4,device=self.views.device).repeat(1,P,1,1)
        K=torch.eye(3,device=self.views.device).repeat(1,P,1,1)
        if self.layout.anchor_rows:v[:,L:]=self.views[:,:1];K[:,L:]=self.K[:,:1]
        mask=torch.arange(P,device=v.device)>=L
        self.cache.cameras["prefix"]=(v,K,mask)
        return v,K,mask

    def camera(self,index):
        if index in self.cache.cameras:return self.cache.cameras[index]
        end=min(index*5+5,len(self.layout.target_positions)//self.layout.rows_per_frame)
        ids=torch.arange(index*5,end,device=self.views.device)
        rows=self.layout.rows_per_frame
        result=(self.views[:,ids].repeat_interleave(rows,1),self.K[:,ids].repeat_interleave(rows,1),torch.ones(len(ids)*rows,device=self.views.device,dtype=torch.bool))
        self.cache.cameras[index]=result
        return result

def install_student_attention(model):
    from lynnreal_core.backends.minimax_h3.model import H3AttentionControl,SolarMiniMaxH3AttnProcessor
    from lynnreal_core.backends.minimax_h3.torch_prope import prope_qkv_separate
    from exact_kernels.rotary_dispatch import fused_rotary
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from reference_kv import transformer_blocks,disable_audio
    class Processor(SolarMiniMaxH3AttnProcessor):
        def __init__(self,index):super().__init__();self.index=index
        def __call__(self,attn,hidden_states,rotary_emb=None,attention_mask=None):
            c=(attention_mask.prope_kwargs or {}).get('three_chunk') if isinstance(attention_mask,H3AttentionControl) else None
            if c is None:return super().__call__(attn,hidden_states,rotary_emb,attention_mask)
            backend=self._attention_backend
            if str(getattr(backend,'value',backend))!='_flash_3':raise RuntimeError('native FA3 required')
            q=attn.norm_q(attn.to_q(hidden_states).unflatten(-1,(attn.heads,-1)))
            k=attn.norm_k(attn.to_k(hidden_states).unflatten(-1,(attn.heads,-1)))
            v=attn.to_v(hidden_states).unflatten(-1,(attn.heads,-1))
            def attend(q,k,v,qp,kp,qc,kc):
                q=fused_rotary(q,*model.rope(qp));k=fused_rotary(k,*model.rope(kp))
                undo=None
                if c.layout.camera:
                    original=q,k,v
                    qq,kk,vv,undo=prope_qkv_separate(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                        query_viewmats=qc[0],query_K=qc[1],key_viewmats=kc[0],key_K=kc[1])
                    q=torch.where(qc[2][None,:,None,None],qq.transpose(1,2),original[0])
                    k=torch.where(kc[2][None,:,None,None],kk.transpose(1,2),original[1])
                    v=torch.where(kc[2][None,:,None,None],vv.transpose(1,2),original[2])
                y=dispatch_attention_fn(q.contiguous(),k.contiguous(),v.contiguous(),attn_mask=None,
                    dropout_p=0.,is_causal=False,backend=backend,parallel_config=getattr(self,'_parallel_config',None))
                if undo is not None:y=torch.where(qc[2][None,:,None,None],undo(y.transpose(1,2)).transpose(1,2),y)
                return y
            pc=c.prefix_camera();idx=self.index
            if c.mode=='prefix':
                if torch.is_grad_enabled():raise RuntimeError('prefix KV must be detached')
                L=c.layout.text_rows;p=c.layout.positions
                qc=tuple(x[:,:L] if x.ndim>1 else x[:L] for x in pc)
                out=attend(q[:,:L],k[:,:L],v[:,:L],p[:L],p[:L],qc,qc)
                if c.layout.anchor_rows:
                    ac=tuple(x[:,L:] if x.ndim>1 else x[L:] for x in pc)
                    out=torch.cat((out,attend(q[:,L:],k,v,p[L:],p,ac,pc)),1)
                c.cache.prefix[idx]=(k.detach(),v.detach())
            elif c.mode=='hand':
                if torch.is_grad_enabled() or not c.layout.hand:raise RuntimeError('invalid HAND prefill')
                hc=c.camera(c.index)
                out=attend(q,k,v,c.positions,c.positions,hc,hc)
                c.cache.hands.setdefault(idx,{})[c.index]=(k.detach(),v.detach())
            else:
                keys=[];values=[];positions=[c.layout.positions];cameras=[pc]
                def add(pair):
                    if pair[0].device!=q.device or pair[1].device!=q.device:raise RuntimeError("KV must remain on the active GPU")
                    keys.append(pair[0]);values.append(pair[1])
                add(c.cache.prefix[idx])
                if c.layout.hand:
                    add(c.cache.hands[idx][c.index]);positions.append(local_hand_positions(c.layout,len(c.positions)//c.layout.rows_per_frame));cameras.append(c.camera(c.index))
                for j in sorted(j for j in c.cache.history.get(idx,{}) if j<c.index):
                    add(c.cache.history[idx][j]);positions.append(window_positions(c.layout,c.index,j));cameras.append(c.camera(j))
                keys.append(k);values.append(v);positions.append(c.positions);cameras.append(c.camera(c.index))
                kc=tuple(torch.cat([z[i] for z in cameras],dim=1 if i<2 else 0) for i in range(3))
                out=attend(q,torch.cat(keys,1),torch.cat(values,1),c.positions,torch.cat(positions),c.camera(c.index),kc)
                if c.mode=='commit':
                    if torch.is_grad_enabled():raise RuntimeError('history commit must be detached')
                    c.cache.history.setdefault(idx,{})[c.index]=(k.detach(),v.detach())
            return attn.to_out[1](attn.to_out[0](out.flatten(2,3).type_as(hidden_states)))
    for i,_,block in transformer_blocks(model):
        old=block.attn.processor;p=Processor(i);p._attention_backend=old._attention_backend;block.attn.set_processor(p)
    original=model.token_refiner.forward
    model.token_refiner.forward=lambda x,*a,**kw:x if x.shape[1]==0 else original(x,*a,**kw)
    disable_audio(model)
    return dict(chunks=7,tail_latents=2,training_latent_frames=37,history='detached_raw_GPU_KV',RoPE=ROPE_POLICY,backend='_flash_3')

class ReferenceStudent:
    def __init__(self,model,device,phase,progress,latent_frames=37,window_chunks=2,maps=3):
        self.model=model;self.device=device;self.phase=phase;self.progress=progress
        self.latent_frames=latent_frames;self.window_chunks=window_chunks;self.maps=maps

    def sigmas(self):
        x = torch.linspace(1.0, 0.0, self.maps + 1, device=self.device)
        return 3 * x / (1 + 2 * x)

    def call(self, c, video, prompt, t, r):
        N = len(c.positions)
        L = c.layout.text_rows if c.mode == 'prefix' else 0
        tags = c.layout.tags if c.mode == 'prefix' else torch.zeros(N, device=self.device, dtype=torch.long)
        tv = torch.ones((N, 2), device=self.device)
        if c.mode == 'prefix' and c.layout.anchor_rows:
            tv[L:] = 0.999
        elif c.mode == 'noisy':
            tv[:, 0] = t
            tv[:, 1] = r
        (unique, inverse) = torch.unique(tv, dim=0, sorted=True, return_inverse=True)
        from lynnreal_hand.inference.adaln import select_adaln
        select_adaln(self.model, unique[:, 0], unique[:, 1])
        vi = torch.arange(L, N, device=self.device)
        (out, _) = self.model(hidden_states=video, encoder_hidden_states=prompt, audio_hidden_states=video.new_empty((1, 0, 32)), timestep=unique[:, 0], r_timestep=unique[:, 1], timestep_indices=inverse, token_tags=torch.as_tensor(tags, device=self.device), position_ids=c.positions, video_indices=vi, audio_indices=vi[:0], text_indices=torch.arange(L, device=self.device), attention_mask=None, fused_prope=True, prope_kwargs={'three_chunk': c}, prope_token_indices=vi, prope_frame_ids=None, cam_viewmats=c.views, cam_K=c.K, packed_sequence_parallel=False, return_dict=False)
        return out
