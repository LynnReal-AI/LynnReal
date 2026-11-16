"""Detached raw-KV W6, local MM-RoPE, and the Stage1 HAND visibility rules.

Raw K and V precede BOTH MM-RoPE and camera PRoPE. Camera IDs remain physical;
only the RGB window clock is rebased. No audio rows exist in this interface.
"""
from dataclasses import dataclass, field
import torch


@dataclass
class RawCache:
    prefix: dict = field(default_factory=dict)
    hand: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)
    archive: dict = field(default_factory=dict)
    hand_archive: dict = field(default_factory=dict)
    archive_enabled: bool = True
    window: int = 6

    def past(self, layer, index):
        values = self.history.get(layer, [])
        assert [x[0] for x in values] == list(range(max(0, index-(self.window-1)), index))
        return values

    def commit(self, layer, index, key, value):
        assert not torch.is_grad_enabled(), 'clean commits must be detached'
        old = self.past(layer, index)
        pair = (key.detach().to(device='cpu',copy=True),value.detach().to(device='cpu',copy=True))
        entry = (index, *pair)
        self.history[layer] = [*old, entry][-(self.window-1):]
        # CPU archive is for final-step gradient replay, never an extra visible KV.
        if self.archive_enabled:
            self.archive.setdefault(layer, {})[index] = pair

    def clear(self):
        self.prefix.clear(); self.hand.clear(); self.history.clear(); self.archive.clear(); self.hand_archive.clear()

    def offload_for_replay(self):
        self.prefix = {i:tuple(x.cpu()for x in pair)for i,pair in self.prefix.items()}
        self.history.clear(); self.hand.clear()

    def memory_summary(self):
        unique={}
        def visit(value):
            if isinstance(value,torch.Tensor):
                key=(str(value.device),value.untyped_storage().data_ptr())
                unique[key]=value
            elif isinstance(value,dict):
                for x in value.values():visit(x)
            elif isinstance(value,(list,tuple)):
                for x in value:visit(x)
        for x in [self.prefix,self.hand,self.history,self.archive,self.hand_archive]:visit(x)
        return dict(cache_CPU_GiB=sum(x.untyped_storage().nbytes()for x in unique.values()if x.device.type=='cpu')/2**30,
            cache_GPU_GiB=sum(x.untyped_storage().nbytes()for x in unique.values()if x.device.type=='cuda')/2**30,
            cache_dtypes=sorted({str(x.dtype)for x in unique.values()}),history_storage='CPU_lossless_per_layer_load')


@dataclass
class Control:
    mode: str
    cache: RawCache
    index: int
    prefix_positions: object
    hand_positions: object
    views: object
    intrinsics: object
    rope: object
    text_rows: int
    rows: int = 1008
    replay: bool = False

    def video_positions(self, slots):
        # One fixed HAND span precedes the local RGB clock, as in Stage1.
        positions = []
        for slot in range(slots):
            p = self.hand_positions.clone()
            p[:, 0] += (slot+1)*17*(5/3)
            positions.append(p)
        return torch.cat(positions)

    def cameras(self, frames):
        ids = torch.as_tensor(frames, device=self.views.device, dtype=torch.long)
        return (self.views[:, ids].repeat_interleave(self.rows, 1),
                self.intrinsics[:, ids].repeat_interleave(self.rows, 1))

    def prefix_cameras(self):
        n = len(self.prefix_positions)
        views = torch.eye(4, device=self.views.device).repeat(1, n, 1, 1)
        K = torch.eye(3, device=self.views.device).repeat(1, n, 1, 1)
        views[:, self.text_rows:] = self.views[:, :1]
        K[:, self.text_rows:] = self.intrinsics[:, :1]
        visible = torch.arange(n, device=views.device) >= self.text_rows
        return views, K, visible


def transformer_blocks(model):
    """Resolve the main blocks through PEFT/FSDP module-name prefixes."""
    found=[]
    for name,block in model.named_modules():
        parts=name.split('.')
        if len(parts)>=2 and parts[-2]=='transformer_blocks' and parts[-1].isdigit():
            found.append((int(parts[-1]),name,block))
    found.sort(key=lambda row:row[0])
    assert [row[0]for row in found]==list(range(50)), [row[1]for row in found]
    return found


def disable_audio(model):
    """Reject any audio input and bypass both native audio projection GEMMs."""
    audit=dict(forwards=0,audio_tokens=0,audio_projection_GEMMs=0)
    object.__setattr__(model,'_duo_audio_audit',audit)
    def check_input(_module,args,kwargs):
        audio=kwargs.get('audio_hidden_states')
        indices=kwargs.get('audio_indices')
        assert audio is not None and audio.shape[1]==0, 'Audio input is forbidden'
        assert indices is not None and indices.numel()==0, 'Audio rows are forbidden'
        audit['forwards']+=1
    model.register_forward_pre_hook(check_input,with_kwargs=True)
    ai,ao=model.audio_proj_in,model.audio_proj_out
    ai.forward=lambda x:x.new_empty((*x.shape[:-1],ai.out_features))
    ao.forward=lambda x:x.new_empty((*x.shape[:-1],ao.out_features))
    return audit


def install(model):
    from lynnreal_core.backends.minimax_h3.model import H3AttentionControl, SolarMiniMaxH3AttnProcessor
    from lynnreal_core.backends.minimax_h3.torch_prope import prope_qkv_separate
    from exact_kernels.rotary_dispatch import fused_rotary
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    class Processor(SolarMiniMaxH3AttnProcessor):
        def __init__(self, layer):
            super().__init__(); self.layer = layer

        def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
            c = (attention_mask.prope_kwargs or {}).get('duo_rawkv') if isinstance(attention_mask, H3AttentionControl) else None
            if c is None:
                return super().__call__(attn, hidden_states, rotary_emb, attention_mask)
            backend = self._attention_backend
            assert str(getattr(backend, 'value', backend)) == '_flash_3'
            q = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1)))
            k = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1)))
            v = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))

            def calculate(q, k, v, qp, kp, qc, kc):
                q = fused_rotary(q, *c.rope(qp)); k = fused_rotary(k, *c.rope(kp))
                # Official fixed-K helper also transforms identity cameras. Qwen
                # rows must instead remain camera-neutral, exactly as Stage1.
                original = q, k, v
                qq, kk, vv, undo = prope_qkv_separate(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
                    query_viewmats=qc[0], query_K=qc[1], key_viewmats=kc[0], key_K=kc[1])
                qq, kk, vv = qq.transpose(1,2), kk.transpose(1,2), vv.transpose(1,2)
                qq = torch.where(qc[2][None,:,None,None], qq, original[0])
                kk = torch.where(kc[2][None,:,None,None], kk, original[1])
                vv = torch.where(kc[2][None,:,None,None], vv, original[2])
                out = dispatch_attention_fn(qq.contiguous(), kk.contiguous(), vv.contiguous(),
                    attn_mask=None, dropout_p=0., is_causal=False, backend=backend,
                    parallel_config=getattr(self, '_parallel_config', None))
                projected = undo(out.transpose(1,2)).transpose(1,2)
                return torch.where(qc[2][None,:,None,None], projected, out)

            def cam(frames):
                a,b = c.cameras(frames)
                return a,b,torch.ones(a.shape[1],device=a.device,dtype=torch.bool)

            if c.mode == 'prefix':
                assert not torch.is_grad_enabled() and k.shape[1] == c.text_rows+c.rows
                pc = c.prefix_cameras(); L = c.text_rows; pp = c.prefix_positions
                outputs = [calculate(q[:,:L],k[:,:L],v[:,:L],pp[:L],pp[:L],
                    (pc[0][:,:L],pc[1][:,:L],pc[2][:L]),(pc[0][:,:L],pc[1][:,:L],pc[2][:L])),
                    calculate(q[:,L:],k,v,pp[L:],pp,(pc[0][:,L:],pc[1][:,L:],pc[2][L:]),pc)]
                c.cache.prefix[self.layer] = (k.detach().clone(),v.detach().clone())
                out = torch.cat(outputs,1)
            elif c.mode == 'dense':
                # Independent oracle packing: recompute every prior clean row,
                # with W6 visibility and a separate local clock per query group.
                L=c.text_rows;P=L+c.rows;T=5*c.rows;n=c.index+1
                assert k.shape[1]==P+2*n*T
                pp=c.prefix_positions;pc=c.prefix_cameras()
                output=[calculate(q[:,:L],k[:,:L],v[:,:L],pp[:L],pp[:L],
                    (pc[0][:,:L],pc[1][:,:L],pc[2][:L]),(pc[0][:,:L],pc[1][:,:L],pc[2][:L])),
                    calculate(q[:,L:P],k[:,:P],v[:,:P],pp[L:],pp,
                    (pc[0][:,L:],pc[1][:,L:],pc[2][L:]),pc)]
                for i in range(n):
                    a=P+i*T;b=a+T;hc=cam(range(i*5,(i+1)*5))
                    output.append(calculate(q[:,a:b],k[:,a:b],v[:,a:b],c.hand_positions,c.hand_positions,hc,hc))
                for i in range(n):
                    first=max(0,i-(c.cache.window-1));a=P+n*T+i*T;b=a+T
                    hand=slice(P+i*T,P+(i+1)*T);clean=slice(P+n*T+first*T,P+n*T+i*T)
                    key=torch.cat((k[:,:P],k[:,hand],k[:,clean],k[:,a:b]),1)
                    value=torch.cat((v[:,:P],v[:,hand],v[:,clean],v[:,a:b]),1)
                    vp=c.video_positions(i-first+1);kp=torch.cat((pp,c.hand_positions,vp))
                    hc=cam(range(i*5,(i+1)*5));wc=cam(range(first*5,(i+1)*5))
                    kc=tuple(torch.cat((x,y,z),dim=1 if x.ndim>1 else 0)for x,y,z in zip(pc,hc,wc))
                    output.append(calculate(q[:,a:b],key,value,vp[-T:],kp,hc,kc))
                out=torch.cat(output,1)
            elif c.mode == 'hand':
                assert not torch.is_grad_enabled() and k.shape[1] == 5*c.rows
                camera = cam(range(c.index*5,(c.index+1)*5))
                out = calculate(q,k,v,c.hand_positions,c.hand_positions,camera,camera)
                c.cache.hand[self.layer] = (k.detach().clone(),v.detach().clone())
                if c.cache.archive_enabled:
                    c.cache.hand_archive.setdefault(self.layer,{})[c.index] = (k.detach().cpu(),v.detach().cpu())
            else:
                assert c.mode in {'noisy','commit'} and k.shape[1] == 5*c.rows
                first = max(0,c.index-(c.cache.window-1)); count = c.index-first
                if c.replay:
                    history = [(i,*c.cache.archive[self.layer][i])for i in range(first,c.index)]
                else:
                    history = c.cache.past(self.layer,c.index)
                pk,pv = c.cache.prefix[self.layer]
                hk,hv = c.cache.hand_archive[self.layer][c.index] if c.replay else c.cache.hand[self.layer]
                keys = [pk.to(q.device),hk.to(q.device),*[x[1].to(q.device)for x in history],k]
                values = [pv.to(q.device),hv.to(q.device),*[x[2].to(q.device)for x in history],v]
                vp = c.video_positions(count+1)
                kp = torch.cat((c.prefix_positions,c.hand_positions,vp))
                pc = c.prefix_cameras(); hc = cam(range(c.index*5,(c.index+1)*5))
                wc = cam(range(first*5,(c.index+1)*5))
                kc = tuple(torch.cat((a,b,d),dim=1 if a.ndim>1 else 0)for a,b,d in zip(pc,hc,wc))
                out = calculate(q,torch.cat(keys,1),torch.cat(values,1),vp[-5*c.rows:],kp,hc,kc)
                if c.mode == 'commit':c.cache.commit(self.layer,c.index,k,v)
            return attn.to_out[1](attn.to_out[0](out.flatten(2,3).type_as(hidden_states)))

    names=[]
    for layer,name,block in transformer_blocks(model):
        old=block.attn.processor;new=Processor(layer)
        new._attention_backend=old._attention_backend;block.attn.set_processor(new);names.append(name)
    # Empty Qwen input after prefix prefill must not execute zero-length FA3.
    original=model.token_refiner.forward
    model.token_refiner.forward=lambda x,*a,**kw:x if x.shape[1]==0 else original(x,*a,**kw)
    # No audio rows and no audio projection GEMMs in this dedicated entry.
    disable_audio(model)
    return dict(blocks=names,cache='before_MM_RoPE_and_PRoPE',window=6,audio_tokens=0,
        Qwen_self=True,anchor_Qwen=True,HAND_self=True,clean_commit_current_HAND=True)
