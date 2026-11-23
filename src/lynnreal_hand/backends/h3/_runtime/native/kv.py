"""GPU packed KV; exact local time rebasing from raw temporal channels.

H3 rotates channels [0:96], with time pairs [0:16] and [48:64]; PRoPE
changes [96:128]. Store raw time32, encoded K128, camera-projected V128.
Sliding never rotates an already rotated time value. Raw full CPU archives
exist only for detached final-map replay; bounded staging overlaps archival IO.
"""
from dataclasses import dataclass,field
import torch
from sparse_hand import SETTINGS as SPARSE,subset_geometry,hand_attention,rgb_attention
MONARCH = {"factor": 0}
from condition_policy import POLICY,kv_only,probe_attention
from reference_kv import Control,transformer_blocks,disable_audio
from reference_kv import RawCache as ReferenceCache


def temporal_raw(key):
    assert key.shape[-1]==128
    return torch.cat((key[...,:16],key[...,48:64]),-1)


def temporal_frequencies(pair):
    assert pair[0].shape[-1]==96
    return tuple(torch.cat((x[...,:16],x[...,48:64]),-1).contiguous()for x in pair)


@dataclass
class RawCache(ReferenceCache):
    optimized: bool=True
    archive_async: bool=True
    archive_writer: object=None
    packed: dict=field(default_factory=dict)
    geometry: dict=field(default_factory=dict)
    replay_static: dict=field(default_factory=dict)
    counters: dict=field(default_factory=lambda:dict(history_H2D_bytes=0,raw_archive_D2H_bytes=0,window_rebases=0))

    def archive_pair(self,k,v):
        self.counters['raw_archive_D2H_bytes']+=(k.numel()+v.numel())*k.element_size()
        if not self.archive_async or k.device.type!='cuda':return k.detach().cpu(),v.detach().cpu()
        from kv_transfer import ArchiveWriter
        if self.archive_writer is None:self.archive_writer=ArchiveWriter(k.device)
        return self.archive_writer.submit((k,v))

    def finish_archive(self):
        if self.archive_writer is not None:
            self.archive_writer.finish()
            self.counters.update(archive_wait_seconds=self.archive_writer.wait_seconds,
                archive_flush_seconds=self.archive_writer.flush_seconds,
                archive_staging_peak_bytes=self.archive_writer.staging_peak_bytes)

    def clear(self):
        self.finish_archive()
        super().clear();self.packed.clear();self.geometry.clear();self.replay_static.clear()

    def begin_chunk(self,index):
        self.geometry.clear();self.replay_static.clear()

    def offload_for_replay(self):
        self.finish_archive()
        super().offload_for_replay();self.packed.clear();self.geometry.clear();self.replay_static.clear()

    def memory_summary(self):
        unique={}
        def visit(value):
            if isinstance(value,torch.Tensor):unique[(str(value.device),value.untyped_storage().data_ptr())]=value
            elif isinstance(value,dict):
                for x in value.values():visit(x)
            elif isinstance(value,(list,tuple)):
                for x in value:visit(x)
        for x in [self.prefix,self.hand,self.history,self.archive,self.hand_archive,self.packed,self.replay_static,self.geometry]:visit(x)
        return dict(cache_CPU_GiB=sum(x.untyped_storage().nbytes()for x in unique.values()if x.device.type=='cpu')/2**30,
            cache_GPU_GiB=sum(x.untyped_storage().nbytes()for x in unique.values()if x.device.type=='cuda')/2**30,
            cache_dtypes=sorted({str(x.dtype)for x in unique.values()}),history_storage='GPU_packed_KV_raw_time32'if self.optimized else'CPU_reference',**self.counters)


def install(model):
    import reference_kv
    report=reference_kv.install(model)
    from lynnreal_core.backends.minimax_h3.model import H3AttentionControl,SolarMiniMaxH3AttnProcessor
    from lynnreal_core.backends.minimax_h3.torch_prope import _prepare_suffix_transforms,logd4_relative_viewmats,_project
    from exact_kernels.rotary_dispatch import fused_rotary
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    def suffix(features,matrix):
        part=_project(features.transpose(1,2)[...,96:128],matrix=matrix).transpose(1,2)
        return torch.cat((features[...,:96],part),-1)

    def geometry(c,mode,dtype):
        key=(c.index,mode,dtype)
        if key in c.cache.geometry:return c.cache.geometry[key]
        P=c.text_rows+c.rows;FULL=5*c.rows;T=c.frames*c.rows
        if mode=='prefix':
            qp=kp=c.prefix_positions;qc=kc=c.prefix_cameras()
        elif mode=='hand':
            qp=kp=c.hand_positions[:T];a,b=c.cameras(range(c.index*5,min((c.index+1)*5,c.span)))
            qc=kc=(a,b,torch.ones(a.shape[1],device=a.device,dtype=torch.bool))
        else:
            first=max(0,c.index-(c.cache.window-1));vp=c.video_positions(c.index-first+1);qp=vp[-T:]
            kp=torch.cat((c.prefix_positions,c.hand_positions[:T],vp))
            pc=c.prefix_cameras();a,b=c.cameras(range(c.index*5,min((c.index+1)*5,c.span)))
            qc=(a,b,torch.ones(a.shape[1],device=a.device,dtype=torch.bool))
            a,b=c.cameras(range(first*5,min((c.index+1)*5,c.span)));wc=(a,b,torch.ones(a.shape[1],device=a.device,dtype=torch.bool))
            kc=tuple(torch.cat((a,b,d),dim=1 if a.ndim>1 else 0)for a,b,d in zip(pc,qc,wc))
        qv=logd4_relative_viewmats(qc[0].to(dtype));kv=logd4_relative_viewmats(kc[0].to(dtype))
        qm,_,out=_prepare_suffix_transforms(viewmats=qv,Ks=qc[1].to(dtype))
        _,km,_=_prepare_suffix_transforms(viewmats=kv,Ks=kc[1].to(dtype))
        g=dict(qrope=c.rope(qp),krope=c.rope(kp),qm=qm.keywords['matrix'],km=km.keywords['matrix'],
            output=out.keywords['matrix'],qmask=qc[2],kmask=kc[2])
        if mode=='hand'and SPARSE['pack']:g=subset_geometry(g,c.cache.hand_token_indices)
        c.cache.geometry[key]=g;return g

    def query(q,g,attn):
        q=fused_rotary(q,*g['qrope'])
        return torch.where(g['qmask'][None,:,None,None],suffix(q,g['qm']),q)

    def encode_kv(k,v,rope,matrix,mask):
        k=fused_rotary(k,*rope)
        return (torch.where(mask[None,:,None,None],suffix(k,matrix),k),
                torch.where(mask[None,:,None,None],suffix(v,matrix),v))

    class Processor(SolarMiniMaxH3AttnProcessor):
        def __init__(self,layer,reference):super().__init__();self.layer=layer;self.reference=reference
        def __call__(self,attn,hidden_states,rotary_emb=None,attention_mask=None):
            c=(attention_mask.prope_kwargs or{}).get('duo_rawkv')if isinstance(attention_mask,H3AttentionControl)else None
            if c is None or not c.cache.optimized or c.mode=='dense':return self.reference(attn,hidden_states,rotary_emb,attention_mask)
            backend=self._attention_backend;assert str(getattr(backend,'value',backend))=='_flash_3'
            if hasattr(attn, 'to_qkv_deploy'):
                if isinstance(hidden_states,tuple):
                    quantized,scale,shape=hidden_states
                    q,k,v=attn.to_qkv_deploy.forward_quantized(quantized,scale,shape[:-1]).chunk(3,dim=-1)
                else:q,k,v=attn.to_qkv_deploy(hidden_states).chunk(3,dim=-1)
                q=q.unflatten(-1,(attn.heads,-1))
                q=attn.norm_q(q)
                k=attn.norm_k(k.unflatten(-1,(attn.heads,-1)))
                v=v.unflatten(-1,(attn.heads,-1))
            else:
                q=attn.norm_q(attn.to_q(hidden_states).unflatten(-1,(attn.heads,-1)))
                k=attn.norm_k(attn.to_k(hidden_states).unflatten(-1,(attn.heads,-1)))
                v=attn.to_v(hidden_states).unflatten(-1,(attn.heads,-1))
            P=c.text_rows+c.rows;FULL=5*c.rows;T=c.frames*c.rows
            only_kv=kv_only(self.layer,c)
            def attend(q,k,v,g):
                if c.mode=='hand'and SPARSE['hand_frame']:
                    y=hand_attention(q,k,v,c.cache,c.rows)
                elif c.mode in('noisy','commit')and SPARSE['rgb']!='dense':
                    y=rgb_attention(q,k,v,c)
                else:
                    y=dispatch_attention_fn(q.contiguous(),k.contiguous(),v.contiguous(),attn_mask=None,dropout_p=0.,
                        is_causal=False,backend=backend,parallel_config=getattr(self,'_parallel_config',None))
                probe_attention(q,k,v,y,c,self.layer)
                return torch.where(g['qmask'][None,:,None,None],suffix(y,g['output']),y)
            # SCD-inspired tail: preserve full transformer depth, reduce only
            # conditional visibility. Prefix/HAND paths are kept explicit.
            if c.mode=='noisy' and self.layer>=POLICY['condition_depth'] and POLICY['tail_mode']!='full':
                g=geometry(c,'rgb',q.dtype)
                kk,vv=encode_kv(k,v,tuple(x[-T:].contiguous()for x in g['krope']),g['km'][:,-T:],g['kmask'][-T:])
                if POLICY['tail_mode']=='self':ks,vs=kk,vv
                else:
                    state=c.cache.packed[self.layer];state['k'][:,P+T:P+2*T].copy_(kk);state['v'][:,P+T:P+2*T].copy_(vv)
                    ks,vs=state['k'][:,:P+2*T],state['v'][:,:P+2*T]
                out=attend(query(q,g,attn),ks,vs,g)
                return attn.to_out[1](attn.to_out[0](out.flatten(2,3).to(q.dtype)))
            if c.replay:
                assert c.mode=='noisy' and torch.is_grad_enabled()
                g=geometry(c,'rgb',q.dtype);ns=len(g['kmask'])-T
                if self.layer not in c.cache.replay_static:
                    with torch.no_grad():
                        pk,pv=c.cache.prefix[self.layer];hk,hv=c.cache.hand_archive[self.layer][c.index]
                        history=[c.cache.archive[self.layer][i]for i in range(max(0,c.index-(c.cache.window-1)),c.index)]
                        keys=[pk,hk,*[x[0]for x in history]];values=[pv,hv,*[x[1]for x in history]]
                        c.cache.counters['history_H2D_bytes']+=sum(x.numel()*x.element_size()for x in keys+values if x.device.type=='cpu')
                        sk=torch.cat([x.to(q.device)for x in keys],1);sv=torch.cat([x.to(q.device)for x in values],1)
                        pair=encode_kv(sk,sv,tuple(x[:ns].contiguous()for x in g['krope']),g['km'][:,:ns],g['kmask'][:ns])
                        c.cache.replay_static[self.layer]=tuple(x.detach()for x in pair)
                sk,sv=c.cache.replay_static[self.layer]
                ck,cv=encode_kv(k,v,tuple(x[-T:].contiguous()for x in g['krope']),g['km'][:,-T:],g['kmask'][-T:])
                out=attend(query(q,g,attn),torch.cat((sk,ck),1),torch.cat((sv,cv),1),g)
            else:
                assert not torch.is_grad_enabled(),'packed buffer is exclusively no-grad rollout'
                if self.layer not in c.cache.packed:
                    c.cache.packed[self.layer]=dict(k=k.new_empty((1,P+(c.cache.window+1)*FULL,*k.shape[2:])),v=v.new_empty((1,P+(c.cache.window+1)*FULL,*v.shape[2:])),
                        rawtime=k.new_empty((1,c.cache.window*FULL,k.shape[2],32)),index=-1)
                state=c.cache.packed[self.layer];kb=state['k'];vb=state['v']
                if c.mode=='prefix':
                    g=geometry(c,'prefix',q.dtype);kk,vv=encode_kv(k,v,g['krope'],g['km'],g['kmask']);qq=query(q,g,attn)
                    kb[:,:P].copy_(kk);vb[:,:P].copy_(vv);L=c.text_rows
                    # Same prefix visibility as reference: Qwen self, anchor Qwen+self.
                    def subset(g,a,b):return{**g,'qmask':g['qmask'][a:b],'output':g['output'][:,a:b]}
                    out=torch.cat((attend(qq[:,:L],kk[:,:L],vv[:,:L],subset(g,0,L)),attend(qq[:,L:],kk,vv,subset(g,L,P))),1)
                    if c.cache.archive_enabled:
                        c.cache.prefix[self.layer]=c.cache.archive_pair(k,v)
                elif c.mode=='hand':
                    if state['index']!=c.index:
                        assert state['index']==c.index-1
                        if c.index>=c.cache.window and not(self.layer>=POLICY['condition_depth']and POLICY['tail_mode']!='full'):
                            a=P+FULL
                            kb[:,a:a+(c.cache.window-1)*FULL].copy_(kb[:,a+FULL:a+c.cache.window*FULL].clone());vb[:,a:a+(c.cache.window-1)*FULL].copy_(vb[:,a+FULL:a+c.cache.window*FULL].clone())
                            raw=state['rawtime'];raw[:,:(c.cache.window-1)*FULL].copy_(raw[:,FULL:c.cache.window*FULL].clone())
                            rebase_key=('rebase',c.index,k.dtype)
                            if rebase_key not in c.cache.geometry:
                                c.cache.geometry[rebase_key]=temporal_frequencies(c.rope(c.video_positions(c.cache.window)[:(c.cache.window-1)*FULL]))
                            rotated=fused_rotary(raw[:,:(c.cache.window-1)*FULL],*c.cache.geometry[rebase_key])
                            kb[:,a:a+(c.cache.window-1)*FULL,:,:16].copy_(rotated[...,:16]);kb[:,a:a+(c.cache.window-1)*FULL,:,48:64].copy_(rotated[...,16:])
                            c.cache.counters['window_rebases']+=1
                        state['index']=c.index
                    g=geometry(c,'hand',q.dtype);kk,vv=encode_kv(k,v,g['krope'],g['km'],g['kmask'])
                    if SPARSE['pack']:
                        nreal=c.cache.hand_valid_count
                        kb[:,P:P+T].index_copy_(1,c.cache.hand_real_indices,kk[:,:nreal]);vb[:,P:P+T].index_copy_(1,c.cache.hand_real_indices,vv[:,:nreal])
                    else:kb[:,P:P+T].copy_(kk);vb[:,P:P+T].copy_(vv)
                    if only_kv:return k[:,:1]
                    out=attend(query(q,g,attn),kk,vv,g)
                    if c.cache.archive_enabled:
                        c.cache.hand_archive.setdefault(self.layer,{})[c.index]=c.cache.archive_pair(k,v)
                else:
                    assert c.mode in{'noisy','commit'} and state['index']==c.index
                    g=geometry(c,'rgb',q.dtype);start=P+FULL+min(c.index,c.cache.window-1)*FULL;end=start+T
                    kk,vv=encode_kv(k,v,tuple(x[-T:].contiguous()for x in g['krope']),g['km'][:,-T:],g['kmask'][-T:])
                    kb[:,start:end].copy_(kk);vb[:,start:end].copy_(vv)
                    if c.mode=='commit' or getattr(c.cache,'_capture_noisy_raw',False):
                        state['rawtime'][:,min(c.index,c.cache.window-1)*FULL:min(c.index,c.cache.window-1)*FULL+T].copy_(temporal_raw(k))
                        if c.cache.archive_enabled:
                            c.cache.archive.setdefault(self.layer,{})[c.index]=c.cache.archive_pair(k,v)
                    if only_kv:return k[:,:1]
                    ks=kb[:,:end] if T==FULL else torch.cat((kb[:,:P+T],kb[:,P+FULL:end]),1)
                    vs=vb[:,:end] if T==FULL else torch.cat((vb[:,:P+T],vb[:,P+FULL:end]),1)
                    out=attend(query(q,g,attn),ks,vs,g)
            return attn.to_out[1](attn.to_out[0](out.flatten(2,3).to(q.dtype)))

    for i,_,block in transformer_blocks(model):
        old=block.attn.processor;new=Processor(i,old);new._attention_backend=old._attention_backend
        new.geometry_for=geometry
        block.attn.set_processor(new)
    return{**report,'optimization':'GPU_packed_KV_raw_time32_exact_rebase','history_H2D_rollout':0,'shared_geometry_across50layers':True}
