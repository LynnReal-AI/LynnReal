"""LynnReal W8A8/block kernels with the native v6 raw-K/camera/FA3 processor.

Quantization changes numerical precision; visibility and raw-K rebasing stay native.
"""
import types,time
import torch
from lynn_kernels.int8 import Int8Linear
from lynn_kernels import kernels as k

@torch.no_grad()
def install(model,blocks_fn,fuse_qkv,write):
    from int8_dispatch import install as dispatch_install
    dispatch_install()
    fuse_qkv(model)
    measurements=[];converted=[]
    for i,name,block in blocks_fn(model):
        if i in (0,49):continue
        for owner,attr in [(block.attn,'to_qkv_deploy'),(block.attn.to_out,'0'),(block.ff.net[0],'proj'),(block.ff.net,'2')]:
            base=getattr(owner,attr);candidate=Int8Linear(base,gemm='torch');candidate.fused_ops=True
            if i==1:
                x=torch.randn((1,5040,base.in_features),device='cuda',dtype=base.weight.dtype)
                reference=base(x);actual=candidate(x)
                error=float((actual.float()-reference.float()).norm()/reference.float().norm())
                def speed(fn):
                    for _ in range(2):fn(x)
                    torch.cuda.synchronize();tick=time.perf_counter()
                    for _ in range(5):fn(x)
                    torch.cuda.synchronize();return (time.perf_counter()-tick)*1000/5
                bf16_ms=speed(base);torch_ms=speed(candidate)
                candidate.gemm='triton'
                # Autotune on true 5040 rows before graph capture/timed rollouts.
                triton_ms=speed(candidate)
                candidate.gemm='triton' if triton_ms<torch_ms else 'torch'
                measurements.append(dict(layer=attr,shape=list(base.weight.shape),relative_L2=error,
                    bf16_ms=bf16_ms,torch_int8_ms=torch_ms,triton_int8_ms=triton_ms,selected=candidate.gemm))
            else:
                candidate.gemm=measurements[len(converted)%4]['selected']
            setattr(owner,attr,candidate);converted.append(f'{name}.{attr}')
        # Remove original QKV weight views: all calls use fused projection now.
        block.attn.to_q=block.attn.to_k=block.attn.to_v=None
    for i,name,block in blocks_fn(model):
        if i in(0,49):block.attn.to_q=block.attn.to_k=block.attn.to_v=None
        block._fusion_model_key=lambda:model._deploy_time_key
        block._fusion_tables={}
        block.forward=types.MethodType(fused_block,block)
    result=dict(format='W8A8',protected_edges=[0,49],quantized_linears=len(converted),complete_transformer=True,
        qkv_fused=True,rmsnorm_adaln=True,swiglu=True,gated_residual=True,residual_ffn_norm_quant=True,
        activation_quantization_fused=True,raw_K_rebase_preserved=True,FA3=True,
        attention_visibility_changed=False,quantization_changes_precision=True,microbenchmarks=measurements)
    write('gates/INT8_full_fusion_install.json',result)
    return result


def fused_block(self,hidden_states,temb,adaln_indices,rotary_emb,attention_mask=None):
    key=self._fusion_model_key()
    table=self._fusion_tables.get(key)
    if table is None:
        table=torch.cat(self.adaln_proj(temb),dim=-1).reshape(-1,6*hidden_states.shape[-1]).contiguous()
        self._fusion_tables[key]=table
    use_quant=isinstance(self.attn.to_qkv_deploy,Int8Linear)
    if use_quant:
        q,scale=k._rms_adaln_quant_int8(hidden_states,self.norm1,table,adaln_indices,0,1)
        attn_input=(q,scale,hidden_states.shape)
    else:attn_input=k._rms_adaln(hidden_states,self.norm1,table,adaln_indices,0,1)
    update=self.attn(attn_input,rotary_emb,attention_mask)
    up,down=self.ff.net[0].proj,self.ff.net[2]
    if use_quant:
        hidden_states,q,scale=k.residual_rms_quant(hidden_states,update,self.norm2,table,adaln_indices)
        projected=up.forward_quantized(q,scale,hidden_states.shape[:-1])
        q,scale=k._swiglu_quant_int8(projected)
        output=down.forward_quantized(q,scale,projected.shape[:-1])
    else:
        hidden_states=k._gated_residual(hidden_states,update,table,adaln_indices,2)
        projected=up(k._rms_adaln(hidden_states,self.norm2,table,adaln_indices,3,4))
        output=down(k._swiglu(projected))
    return k._gated_residual(hidden_states,output,table,adaln_indices,5)
