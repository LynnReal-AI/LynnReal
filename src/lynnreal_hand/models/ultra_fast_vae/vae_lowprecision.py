"""Explicit low-precision linear policies; no silent precision/kernel fallback."""
import torch
from torch import nn
from torch.nn import functional as F
import triton
import triton.language as tl

@triton.jit
def _row_int8(X,Q,S,M:tl.constexpr,K:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BK)
    x=tl.load(X+row*K+col,col<K,other=0).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(x)),1e-8)/127.
    value=tl.extra.cuda.libdevice.nearbyint(x/scale)
    value=tl.maximum(-127.,tl.minimum(127.,value)).to(tl.int8)
    tl.store(Q+row*K+col,value,col<K);tl.store(S+row,scale)

def quantize_int8(x):
    assert x.is_cuda and x.ndim==2 and x.is_contiguous()
    m,k=x.shape;q=torch.empty_like(x,dtype=torch.int8);scale=torch.empty(m,device=x.device,dtype=torch.float32)
    _row_int8[(m,)](x,q,scale,m,k,triton.next_power_of_2(k),num_warps=4 if k<=4096 else 8)
    return q,scale

class LowPrecisionLinear(nn.Module):
    def __init__(self,weight,bias,policy):
        super().__init__();assert policy in ('fp16','fp8_tensor','fp8_rowwise','int8_tma_fp16')
        self.policy=policy;self.in_features=weight.shape[1];self.out_features=weight.shape[0]
        w=weight.detach().to(torch.float16).contiguous()
        self.register_buffer('bias',None if bias is None else bias.detach().to(torch.float16).contiguous())
        if policy=='fp16':self.register_buffer('weight',w);return
        if policy=='int8_tma_fp16':
            scale=w.float().abs().amax(1).clamp_min(1e-8)/127
            quant=(w.float()/scale[:,None]).round().clamp(-127,127).to(torch.int8)
        elif policy=='fp8_tensor':
            scale=w.float().abs().amax().clamp_min(1e-8).reshape(1)/448
            quant=(w.float()/scale).clamp(-448,448).to(torch.float8_e4m3fn)
        else:
            scale=w.float().abs().amax(1,keepdim=True).clamp_min(1e-8)/448
            quant=(w.float()/scale).clamp(-448,448).to(torch.float8_e4m3fn)
            scale=scale.T.contiguous()
        self.register_buffer('quantized',quant.contiguous());self.register_buffer('scale',scale.contiguous())
    def forward(self,x):
        shape=x.shape;flat=x.reshape(-1,self.in_features).to(torch.float16).contiguous()
        if self.policy=='fp16':return F.linear(flat,self.weight,self.bias).reshape(*shape[:-1],self.out_features)
        if self.policy=='int8_tma_fp16':
            from .vae_int8_tma import matmul
            q,s=quantize_int8(flat);out=matmul(q,self.quantized.T,s,self.scale,self.bias)
        else:
            if self.policy=='fp8_tensor':scale=flat.float().abs().amax().clamp_min(1e-8).reshape(1)/448
            else:scale=flat.float().abs().amax(1,keepdim=True).clamp_min(1e-8)/448
            quant=(flat.float()/scale).clamp(-448,448).to(torch.float8_e4m3fn)
            padding=(-flat.shape[0])%16
            if padding:
                quant=F.pad(quant,(0,0,0,padding))
                if self.policy=='fp8_rowwise':scale=F.pad(scale,(0,0,0,padding),value=1.)
            out=torch._scaled_mm(quant,self.quantized.T,scale_a=scale,scale_b=self.scale,bias=self.bias,
                out_dtype=torch.float16,use_fast_accum=True)[:flat.shape[0]]
        return out.reshape(*shape[:-1],self.out_features)
