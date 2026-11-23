"""Isolated inference ablations; original mask/RoPE is baseline.

Tail conditioning cuts and residual reuse change model behavior; measure their
incremental error against the accepted W8A8 implementation in this process.
"""
import inspect,textwrap,types
import torch
POLICY=dict(name='baseline',exact_tail=False,cache_only_head=False,condition_depth=50,tail_mode='full',commit_depth=50,noisy_reuse_layers=(),hand_reuse_layers=(),probe=False,monarch=False)
MODEL=None;PROBE_ROWS=[];PROBE_PATH=None

def depth(mode):
    if mode=='hand':return POLICY['condition_depth']if POLICY['tail_mode']=='self'else 50
    if mode=='commit':return min(POLICY['commit_depth'],POLICY['condition_depth'])if POLICY['tail_mode']!='full'else POLICY['commit_depth']
    return 50

def kv_only(layer,c):
    if c.mode not in('hand','commit'):return False
    return (POLICY['exact_tail']and layer==depth(c.mode)-1)or(c.mode=='hand'and layer in POLICY['hand_reuse_layers']and c.index%2==1)

def configure(**kw):
    POLICY.update(name='baseline',exact_tail=False,cache_only_head=False,condition_depth=50,tail_mode='full',commit_depth=50,noisy_reuse_layers=(),hand_reuse_layers=(),probe=False,monarch=False)
    POLICY.update(kw)

def _table(block,states,temb):
    key=MODEL._deploy_time_key
    t=block._fusion_tables.get(key)
    if t is None:
        t=torch.cat(block.adaln_proj(temb),-1).reshape(-1,6*states.shape[-1]).contiguous();block._fusion_tables[key]=t
    return t

def _cache_only_block(block,states,temb,indices,rotary,control):
    from lynn_kernels import kernels as k
    from lynn_kernels.int8 import Int8Linear
    table=_table(block,states,temb)
    if isinstance(block.attn.to_qkv_deploy,Int8Linear):
        q,s=k._rms_adaln_quant_int8(states,block.norm1,table,indices,0,1);value=(q,s,states.shape)
    else:value=k._rms_adaln(states,block.norm1,table,indices,0,1)
    block.attn(value,rotary,control)
    return states

def install(model,blocks_fn,write):
    global MODEL,PROBE_PATH
    MODEL=model;PROBE_PATH=write
    old_forward=model.forward;fn=inspect.unwrap(old_forward);source=textwrap.dedent(inspect.getsource(fn))
    start=source.index('def forward(');source=source[start:]
    end=source.index('        packed = self.norm_out')if '        packed = self.norm_out'in source else source.index('    packed = self.norm_out')
    source=source[:end]+('        'if source[end:].startswith('        ')else'    ')+'return hidden_states, audio_hidden_states\n'
    source=source.replace('def forward(','def cache_only_forward(',1)
    namespace=dict(fn.__globals__);exec(compile(source,__file__,'exec'),namespace)
    cache_forward=types.MethodType(namespace['cache_only_forward'],model)
    def dispatch_forward(*args,**kwargs):
        c=kwargs.get('prope_kwargs',{}).get('duo_rawkv')
        if c is not None and c.mode in('hand','commit')and POLICY['cache_only_head']:
            return cache_forward(*args,**kwargs)
        return old_forward(*args,**kwargs)
    model.forward=dispatch_forward
    for i,name,block in blocks_fn(model):
        old=block.forward
        def wrapped(self,states,temb,indices,rotary,control,layer=i,original=old):
            c=(control.prope_kwargs or{}).get('duo_rawkv')
            if c is None:return original(states,temb,indices,rotary,control)
            if c.mode in('hand','commit')and layer>=depth(c.mode):return states
            if kv_only(layer,c):
                _cache_only_block(self,states,temb,indices,rotary,control)
                if c.mode=='hand'and layer in POLICY['hand_reuse_layers']and c.index%2==1:
                    delta=getattr(c.cache,'_hand_deltas',{}).get(layer)
                    if delta is None:raise RuntimeError('HAND residual has no current-request reference')
                    return (states.float()+delta).to(states.dtype)
                return states
            selected=c.mode=='noisy'and layer in POLICY['noisy_reuse_layers']
            if selected:
                if getattr(c.cache,'_delta_chunk',None)!=c.index:c.cache._noisy_deltas={};c.cache._delta_chunk=c.index
                if MODEL._deploy_time_key[1]!=0.:
                    delta=c.cache._noisy_deltas[layer]
                    return (states.float()+delta).to(states.dtype)
            out=original(states,temb,indices,rotary,control)
            if selected:c.cache._noisy_deltas[layer]=out.float()-states.float()
            if c.mode=='hand'and layer in POLICY['hand_reuse_layers']:
                if not hasattr(c.cache,'_hand_deltas'):c.cache._hand_deltas={}
                c.cache._hand_deltas[layer]=out.float()-states.float()
            return out
        block.forward=types.MethodType(wrapped,block)
    write('POLICY_INSTALLED.json',dict(cache_only_head_from_original_source=True,blocks=50,baseline_has_no_changes=True))

def probe_attention(q,k,v,out,c,layer):
    if not POLICY['probe']or c.index!=5 or layer not in(0,5,10,20,30,40,49):return
    if c.mode=='noisy'and MODEL._deploy_time_key[1]!=.5:return
    if c.mode not in('hand','noisy'):return
    row=dict(chunk=c.index,layer=layer,mode=c.mode)
    if c.mode=='noisy':
        heads=torch.arange(0,56,7,device=q.device);ids=torch.linspace(0,q.shape[1]-1,32,device=q.device).long()
        qq=q.index_select(1,ids).index_select(2,heads).float();kk=k.index_select(2,heads).float()
        scores=torch.einsum('bqhd,bkhd->bhqk',qq,kk)*(q.shape[-1]**-.5);p=scores.softmax(-1)
        P=c.text_rows+c.rows;T=5*c.rows;regions={'Qwen':(0,c.text_rows),'anchor':(c.text_rows,P),'HAND':(P,P+T),'past_clean':(P+T,k.shape[1]-T),'current':(k.shape[1]-T,k.shape[1])}
        row['sampled_attention_mass']={n:float(p[...,a:b].sum(-1).mean())for n,(a,b)in regions.items()}
    if layer in(0,20,49):
        try:
            from bridge import mixed
            pred=mixed(q.contiguous(),k.contiguous(),v.contiguous(),0 if c.mode=='hand'else c.text_rows+c.rows)
            row['Monarch_relative_L2_actual_features']=float((pred.float()-out.float()).norm()/out.float().norm().clamp_min(1e-8))
        except Exception as exc:row['Monarch_error']=repr(exc)
    PROBE_ROWS.append(row);PROBE_PATH('ATTENTION_FEATURE_PROBE.json',PROBE_ROWS)
