"""Frozen-weight precision and quantization helpers for Ultra-Fast-VAE."""
import torch
from torch import nn
from torch.nn import functional as F

@torch.library.custom_op('lynnreal_vae::row_int8',mutates_args=())
def row_int8(x:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    from .vae_lowprecision import quantize_int8
    return quantize_int8(x)
@row_int8.register_fake
def _(x):return torch.empty_like(x,dtype=torch.int8),x.new_empty((x.shape[0],),dtype=torch.float32)

@torch.library.custom_op('lynnreal_vae::gemm_int8',mutates_args=())
def gemm_int8(x:torch.Tensor,weight:torch.Tensor,row_scale:torch.Tensor,column_scale:torch.Tensor,bias:torch.Tensor|None)->torch.Tensor:
    from .vae_int8_tma import matmul
    return matmul(x,weight.T,row_scale,column_scale,bias)
@gemm_int8.register_fake
def _(x,weight,row_scale,column_scale,bias):return x.new_empty((x.shape[0],weight.shape[0]),dtype=torch.float16)

class Int8Linear(nn.Module):
    def __init__(self,original):
        super().__init__();self.in_features=original.in_features;self.out_features=original.out_features
        w=original.weight.detach().float();scale=w.abs().amax(1).clamp_min(1e-8)/127
        self.register_buffer('weight',(w/scale[:,None]).round().clamp(-127,127).to(torch.int8).contiguous())
        self.register_buffer('scale',scale.contiguous())
        self.register_buffer('bias',None if original.bias is None else original.bias.detach().half().contiguous())
    def forward(self,x):
        assert x.dtype==torch.float16
        shape=x.shape;flat=x.reshape(-1,self.in_features).contiguous();q,s=row_int8(flat)
        return gemm_int8(q,self.weight,s,self.scale,self.bias).reshape(*shape[:-1],self.out_features)

class FusedProcessor:
    _attention_backend='_flash_3';_parallel_config=None
    def __call__(self,attn,hidden_states,rotary_emb=None):
        from flash_attn_interface import flash_attn_func
        batch,tokens,_=hidden_states.shape;dim=attn.heads*attn.dim_head
        qkv=attn.to_qkv(hidden_states).reshape(batch,tokens,3,attn.heads,attn.dim_head)
        query,key,value=[qkv[:,:,i]for i in range(3)]
        query=attn.norm_q(query.float()).to(query.dtype);key=attn.norm_k(key.float()).to(key.dtype)
        if rotary_emb is not None:
            cos,sin=rotary_emb;cos=cos.to(query.dtype);sin=sin.to(query.dtype);rotary=cos.shape[-1]
            def apply(x):
                a,b=x[...,:rotary].chunk(2,-1);turned=torch.cat((-b,a),-1)
                return torch.cat((x[...,:rotary]*cos+turned*sin,x[...,rotary:]),-1).contiguous()
            query,key=apply(query),apply(key)
        assert query.dtype==key.dtype==value.dtype==torch.float16
        result=flash_attn_func(query.contiguous(),key.contiguous(),value.contiguous(),causal=False,deterministic=True)
        return attn.to_out[0](result.reshape(batch,tokens,dim))

@torch.no_grad()
def static_half_and_fuse(light):
    decoder=light.core.decoder
    # Keep all normalization weights and RoPE buffers FP32. Linear weights,
    # register and residual scales must share the FP16 activation dtype.
    for module in decoder.modules():
        if isinstance(module,nn.Linear):module.half()
    light.core.post_quant_conv.half();decoder.register_tokens.data=decoder.register_tokens.data.half()
    saved=[]
    for block in decoder.transformer_blocks:
        block.scale1.data=block.scale1.data.half();block.scale2.data=block.scale2.data.half()
        a=block.attn;parts=[a.to_q,a.to_k,a.to_v]
        fused=nn.Linear(parts[0].in_features,sum(x.out_features for x in parts),bias=True,device=parts[0].weight.device,dtype=torch.float16)
        fused.weight.copy_(torch.cat([x.weight for x in parts]));fused.bias.copy_(torch.cat([x.bias for x in parts]))
        a.to_qkv=fused;a.to_q=a.to_k=a.to_v=None;a.set_processor(FusedProcessor())
        for owner,name in [(a,'to_qkv'),(a.to_out,'0'),(block.ff.net[0],'proj'),(block.ff.net,'2')]:saved.append((owner,name,getattr(owner,name)))
    return saved

def install_policy(saved,policy):
    assert policy in ('fp16','int8_ffn','int8_all')
    for owner,name,original in saved:
        quantize=policy=='int8_all'or(policy=='int8_ffn'and name in ('proj','2'))
        setattr(owner,name,Int8Linear(original)if quantize else original)

def core18_decode(light,context):
    """Cover768x1344 once using18native16x16 inputs and16x14 RGB cores.

    The last column takes2pixels-of-latent left halo, every other column takes
    2 right halo. Temporal/spatial input geometry remains native5x16x16.
    """
    assert context.shape==(1,24,5,48,84)
    starts=[0,14,28,42,56,68]
    packed=torch.cat([context[:,:,:,y:y+16,x:x+16]for y in (0,16,32)for x in starts],dim=0)
    raw=light.core.decoder(light.core.post_quant_conv(packed))[:,:,3:20]
    rows=[]
    for i in range(3):
        tiles=[]
        for j in range(6):
            left=32 if j==5 else 0
            tiles.append(raw[i*6+j:i*6+j+1,...,left:left+224])
        rows.append(torch.cat(tiles,dim=-1))
    return torch.cat(rows,dim=-2)
