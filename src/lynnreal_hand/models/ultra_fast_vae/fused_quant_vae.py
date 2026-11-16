"""Inference fusions that retain explicit FP16 activation rounding before INT8."""
import torch
from torch import nn
import triton
import triton.language as tl
from .stable_vae_ops import StableBlock,scaled_residual_half,qkv_rms_rope_half
from .vae_precision_runtime import gemm_int8

@triton.jit
def _rms_quant(X,W,Q,S,K:tl.constexpr,EPS:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BK);valid=col<K
    x=tl.load(X+row*K+col,valid,other=0).to(tl.float32)
    w=tl.load(W+col,valid,other=0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,0)/K+EPS)
    y=((x*inv)*w).to(tl.float16).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(y)),1e-8)/127.
    quant=tl.extra.cuda.libdevice.nearbyint(y/scale)
    quant=tl.maximum(-127.,tl.minimum(127.,quant)).to(tl.int8)
    tl.store(Q+row*K+col,quant,valid);tl.store(S+row,scale)

@torch.library.custom_op('lynnreal_vae::rms_int8',mutates_args=())
def rms_int8(x:torch.Tensor,weight:torch.Tensor,epsilon:float)->tuple[torch.Tensor,torch.Tensor]:
    assert x.dtype==torch.float16 and x.is_contiguous()
    k=x.shape[-1];m=x.numel()//k;q=torch.empty((m,k),device=x.device,dtype=torch.int8);s=torch.empty(m,device=x.device,dtype=torch.float32)
    _rms_quant[(m,)](x,weight,q,s,k,epsilon,triton.next_power_of_2(k),num_warps=4,enable_fp_fusion=False)
    return q,s
@rms_int8.register_fake
def _(x,weight,epsilon):
    m=x.numel()//x.shape[-1];return x.new_empty((m,x.shape[-1]),dtype=torch.int8),x.new_empty((m,),dtype=torch.float32)

@triton.jit
def _swiglu_quant(X,Q,S,K:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BK);valid=col<K
    x=tl.load(X+row*(2*K)+col,valid,other=0).to(tl.float32)
    gate=tl.load(X+row*(2*K)+K+col,valid,other=0).to(tl.float32)
    activated=(gate/(1+tl.exp(-gate))).to(tl.float16).to(tl.float32)
    y=(x*activated).to(tl.float16).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(y)),1e-8)/127.
    quant=tl.extra.cuda.libdevice.nearbyint(y/scale)
    quant=tl.maximum(-127.,tl.minimum(127.,quant)).to(tl.int8)
    tl.store(Q+row*K+col,quant,valid);tl.store(S+row,scale)

@torch.library.custom_op('lynnreal_vae::swiglu_int8',mutates_args=())
def swiglu_int8(x:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    assert x.dtype==torch.float16 and x.is_contiguous()
    k=x.shape[-1]//2;m=x.numel()//(2*k);q=torch.empty((m,k),device=x.device,dtype=torch.int8);s=torch.empty(m,device=x.device,dtype=torch.float32)
    _swiglu_quant[(m,)](x,q,s,k,triton.next_power_of_2(k),num_warps=4 if k<=4096 else 8,enable_fp_fusion=False)
    return q,s
@swiglu_int8.register_fake
def _(x):
    k=x.shape[-1]//2;m=x.numel()//(2*k);return x.new_empty((m,k),dtype=torch.int8),x.new_empty((m,),dtype=torch.float32)

@triton.jit
def _residual_rms_quant(X,U,RS,W,H,Q,S,K:tl.constexpr,EPS:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BK);valid=col<K
    x=tl.load(X+row*K+col,valid,other=0).to(tl.float32)
    u=tl.load(U+row*K+col,valid,other=0).to(tl.float32)
    residual_scale=tl.load(RS+col,valid,other=0).to(tl.float32)
    product=(u*residual_scale).to(tl.float16).to(tl.float32)
    hidden=(x+product).to(tl.float16).to(tl.float32)
    weight=tl.load(W+col,valid,other=0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(hidden*hidden,0)/K+EPS)
    y=((hidden*inv)*weight).to(tl.float16).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(y)),1e-8)/127.
    quant=tl.extra.cuda.libdevice.nearbyint(y/scale)
    quant=tl.maximum(-127.,tl.minimum(127.,quant)).to(tl.int8)
    tl.store(H+row*K+col,hidden,valid);tl.store(Q+row*K+col,quant,valid);tl.store(S+row,scale)

@torch.library.custom_op('lynnreal_vae::residual_rms_int8',mutates_args=())
def residual_rms_int8(x:torch.Tensor,update:torch.Tensor,residual_scale:torch.Tensor,weight:torch.Tensor,epsilon:float)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    assert x.is_contiguous()and update.is_contiguous()and x.dtype==update.dtype==torch.float16
    k=x.shape[-1];m=x.numel()//k;h=torch.empty_like(x);q=torch.empty((m,k),device=x.device,dtype=torch.int8);s=torch.empty(m,device=x.device,dtype=torch.float32)
    _residual_rms_quant[(m,)](x,update,residual_scale,weight,h,q,s,k,epsilon,triton.next_power_of_2(k),num_warps=4,enable_fp_fusion=False)
    return h,q,s
@residual_rms_int8.register_fake
def _(x,update,residual_scale,weight,epsilon):
    m=x.numel()//x.shape[-1];return torch.empty_like(x),x.new_empty((m,x.shape[-1]),dtype=torch.int8),x.new_empty((m,),dtype=torch.float32)

def projected(linear,q,s,batch,tokens):
    return gemm_int8(q,linear.weight,s,linear.scale,linear.bias).reshape(batch,tokens,linear.out_features)

class FusedQuantBlock(StableBlock):
    def forward(self,hidden_states,rotary_emb):
        from flash_attn_interface import flash_attn_func
        b,t,_=hidden_states.shape;h,d=self.attn.heads,self.attn.dim_head
        quant,scale=rms_int8(hidden_states,self.norm1.weight,self.norm1.eps)
        qkv=projected(self.attn.to_qkv,quant,scale,b,t).reshape(b,t,3,h,d)
        cosine,sine=rotary_emb;q,k,v=qkv_rms_rope_half(qkv,cosine,sine,self.attn.norm_q.eps)
        attention=flash_attn_func(q,k,v,causal=False,deterministic=True)
        update=self.attn.to_out[0](attention.reshape(b,t,h*d))
        hidden_states,quant,scale=residual_rms_int8(hidden_states,update,self.scale1,self.norm2.weight,self.norm2.eps)
        up=projected(self.ff.net[0].proj,quant,scale,b,t)
        quant,scale=swiglu_int8(up);update=projected(self.ff.net[2],quant,scale,b,t)
        return scaled_residual_half(hidden_states,update,self.scale2)
