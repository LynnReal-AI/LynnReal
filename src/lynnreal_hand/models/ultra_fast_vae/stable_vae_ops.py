"""Ultra-Fast-VAE operators with explicit FP16 rounding boundaries."""
import torch
from torch import nn
import triton
import triton.language as tl

@torch.library.custom_op('lynnreal_vae::denormalize_half',mutates_args=())
def denormalize_half(x:torch.Tensor,mean:torch.Tensor,std:torch.Tensor)->torch.Tensor:
    # Opaque boundary preserves the two FP32 operations before the FP16 cast.
    return (x.float()*std+mean).to(torch.float16)
@denormalize_half.register_fake
def _(x,mean,std):return torch.empty_like(x,dtype=torch.float16)

@torch.library.custom_op('lynnreal_vae::linear_half',mutates_args=())
def linear_half(x:torch.Tensor,weight:torch.Tensor,bias:torch.Tensor|None)->torch.Tensor:
    return torch.nn.functional.linear(x,weight,bias)
@linear_half.register_fake
def _(x,weight,bias):return x.new_empty((*x.shape[:-1],weight.shape[0]))

class OpaqueLinear(nn.Module):
    def __init__(self,original):
        super().__init__();self.weight=original.weight;self.bias=original.bias
    def forward(self,x):return linear_half(x,self.weight,self.bias)

@triton.jit
def _rms(X,W,Y,K:tl.constexpr,EPS:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BK)
    x=tl.load(X+row*K+col,col<K,other=0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,0)/K+EPS)
    weight=tl.load(W+col,col<K,other=0).to(tl.float32)
    tl.store(Y+row*K+col,(x*inv)*weight,col<K)

@torch.library.custom_op('lynnreal_vae::rms_half',mutates_args=())
def rms_half(x:torch.Tensor,weight:torch.Tensor,epsilon:float)->torch.Tensor:
    assert x.dtype==torch.float16 and x.is_contiguous()and weight.is_contiguous()
    k=x.shape[-1];out=torch.empty_like(x)
    _rms[(x.numel()//k,)](x,weight,out,k,epsilon,triton.next_power_of_2(k),num_warps=4,enable_fp_fusion=False)
    return out
@rms_half.register_fake
def _(x,weight,epsilon):return torch.empty_like(x)

@triton.jit
def _swiglu(X,Y,M:tl.constexpr,K:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);row=index//K;col=index%K
    x=tl.load(X+row*(2*K)+col,row<M,other=0).to(tl.float32)
    gate=tl.load(X+row*(2*K)+K+col,row<M,other=0).to(tl.float32)
    activated=(gate/(1+tl.exp(-gate))).to(tl.float16).to(tl.float32)
    tl.store(Y+index,x*activated,row<M)

@torch.library.custom_op('lynnreal_vae::swiglu_half',mutates_args=())
def swiglu_half(x:torch.Tensor)->torch.Tensor:
    assert x.dtype==torch.float16 and x.is_contiguous()and x.shape[-1]%2==0
    k=x.shape[-1]//2;m=x.numel()//(2*k);out=torch.empty((*x.shape[:-1],k),device=x.device,dtype=x.dtype)
    _swiglu[(triton.cdiv(m*k,1024),)](x,out,m,k,1024,enable_fp_fusion=False)
    return out
@swiglu_half.register_fake
def _(x):return x.new_empty((*x.shape[:-1],x.shape[-1]//2))

@triton.jit
def _residual(X,U,S,Y,SIZE:tl.constexpr,K:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+i,i<SIZE,other=0).to(tl.float32)
    u=tl.load(U+i,i<SIZE,other=0).to(tl.float32)
    s=tl.load(S+i%K).to(tl.float32)
    product=(u*s).to(tl.float16).to(tl.float32)
    tl.store(Y+i,x+product,i<SIZE)

@torch.library.custom_op('lynnreal_vae::scaled_residual_half',mutates_args=())
def scaled_residual_half(x:torch.Tensor,update:torch.Tensor,scale:torch.Tensor)->torch.Tensor:
    assert x.dtype==update.dtype==scale.dtype==torch.float16
    assert x.is_contiguous()and update.is_contiguous()and scale.is_contiguous()
    assert x.shape==update.shape and scale.shape==(x.shape[-1],)
    out=torch.empty_like(x)
    _residual[(triton.cdiv(x.numel(),1024),)](x,update,scale,out,x.numel(),x.shape[-1],1024,enable_fp_fusion=False)
    return out
@scaled_residual_half.register_fake
def _(x,update,scale):return torch.empty_like(x)

@triton.jit
def _qkv(QKV,COS,SIN,Q,K,V,ROWS:tl.constexpr,H:tl.constexpr,D:tl.constexpr,R:tl.constexpr,EPS:tl.constexpr,BR:tl.constexpr,BD:tl.constexpr):
    rows=tl.program_id(0)*BR+tl.arange(0,BR);cols=tl.arange(0,BD)
    bt=rows//H;head=rows%H;offset=bt[:,None]*(3*H*D)+head[:,None]*D+cols[None,:]
    mask=(rows[:,None]<ROWS)&(cols[None,:]<D)
    q=tl.load(QKV+offset,mask,other=0).to(tl.float32)
    k=tl.load(QKV+offset+H*D,mask,other=0).to(tl.float32)
    v=tl.load(QKV+offset+2*H*D,mask,other=0)
    iq=tl.rsqrt(tl.sum(q*q,1)/D+EPS);ik=tl.rsqrt(tl.sum(k*k,1)/D+EPS)
    qn=(q*iq[:,None]).to(tl.float16).to(tl.float32);kn=(k*ik[:,None]).to(tl.float16).to(tl.float32)
    rotated=tl.where(cols<R//2,cols+R//2,cols-R//2)
    roffset=bt[:,None]*(3*H*D)+head[:,None]*D+rotated[None,:]
    rmask=(rows[:,None]<ROWS)&(cols[None,:]<R)
    qr=tl.load(QKV+roffset,rmask,other=0).to(tl.float32)
    kr=tl.load(QKV+roffset+H*D,rmask,other=0).to(tl.float32)
    qr=(qr*iq[:,None]).to(tl.float16).to(tl.float32);kr=(kr*ik[:,None]).to(tl.float16).to(tl.float32)
    sign=tl.where(cols<R//2,-1.,1.)
    cosine=tl.load(COS+bt[:,None]*R+cols[None,:],rmask,other=0).to(tl.float16).to(tl.float32)
    sine=tl.load(SIN+bt[:,None]*R+cols[None,:],rmask,other=0).to(tl.float16).to(tl.float32)
    qrot=(qn*cosine).to(tl.float16).to(tl.float32)+(qr*sign[None,:]*sine).to(tl.float16).to(tl.float32)
    krot=(kn*cosine).to(tl.float16).to(tl.float32)+(kr*sign[None,:]*sine).to(tl.float16).to(tl.float32)
    dest=rows[:,None]*D+cols[None,:]
    tl.store(Q+dest,tl.where(cols[None,:]<R,qrot,qn),mask)
    tl.store(K+dest,tl.where(cols[None,:]<R,krot,kn),mask)
    tl.store(V+dest,v,mask)

@torch.library.custom_op('lynnreal_vae::qkv_rms_rope_half',mutates_args=())
def qkv_rms_rope_half(qkv:torch.Tensor,cosine:torch.Tensor,sine:torch.Tensor,epsilon:float)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    b,t,three,h,d=qkv.shape;assert three==3 and qkv.is_contiguous()and qkv.dtype==torch.float16
    assert cosine.shape==sine.shape and cosine.shape==(b,t,1,cosine.shape[-1])
    assert cosine.is_contiguous()and sine.is_contiguous()
    r=cosine.shape[-1];assert r%2==0 and r<=d
    q=torch.empty((b,t,h,d),device=qkv.device,dtype=qkv.dtype);k=torch.empty_like(q);v=torch.empty_like(q)
    _qkv[(triton.cdiv(b*t*h,8),)](qkv,cosine,sine,q,k,v,b*t*h,h,d,r,epsilon,8,triton.next_power_of_2(d),num_warps=4,enable_fp_fusion=False)
    return q,k,v
@qkv_rms_rope_half.register_fake
def _(qkv,cosine,sine,epsilon):
    b,t,_,h,d=qkv.shape;q=qkv.new_empty((b,t,h,d))
    return q,torch.empty_like(q),torch.empty_like(q)

class CachedReferenceRoPE(nn.Module):
    def __init__(self,decoder,device):
        super().__init__()
        grids=[2*(torch.arange(.5,n,device=device)/n)-1 for n in (5,16,16)]
        ids=torch.stack(torch.meshgrid(*grids,indexing='ij'),-1).flatten(0,2)[None]
        ids=torch.cat((ids,ids.new_zeros(1,decoder.num_register_tokens+1,3)),1)
        cosine,sine=decoder.rope(ids)
        self.register_buffer('cosine',cosine.contiguous());self.register_buffer('sine',sine.contiguous())
    def forward(self,ids):
        assert ids.shape[1:]==(1285,3)
        return self.cosine.expand(ids.shape[0],-1,-1,-1).contiguous(),self.sine.expand(ids.shape[0],-1,-1,-1).contiguous()

class StableProcessor:
    _attention_backend='_flash_3';_parallel_config=None
    def __call__(self,attn,hidden_states,rotary_emb=None):
        from flash_attn_interface import flash_attn_func
        b,t,_=hidden_states.shape;h,d=attn.heads,attn.dim_head
        qkv=attn.to_qkv(hidden_states).reshape(b,t,3,h,d)
        cosine,sine=rotary_emb
        q,k,v=qkv_rms_rope_half(qkv,cosine,sine,attn.norm_q.eps)
        result=flash_attn_func(q,k,v,causal=False,deterministic=True)
        return attn.to_out[0](result.reshape(b,t,h*d))

class StableBlock(nn.Module):
    def __init__(self,original):
        super().__init__()
        for name in ('norm1','norm2','attn','ff','scale1','scale2'):setattr(self,name,getattr(original,name))
        self.attn.set_processor(StableProcessor())
    def forward(self,hidden_states,rotary_emb):
        normalized=rms_half(hidden_states,self.norm1.weight,self.norm1.eps)
        hidden_states=scaled_residual_half(hidden_states,self.attn(normalized,rotary_emb),self.scale1)
        normalized=rms_half(hidden_states,self.norm2.weight,self.norm2.eps)
        update=self.ff.net[2](swiglu_half(self.ff.net[0].proj(normalized)))
        return scaled_residual_half(hidden_states,update,self.scale2)
