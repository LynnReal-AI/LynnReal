"""LynnReal Hopper persistent INT8 TMA GEMM adapted to Triton 3.5 descriptors."""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def _persistent(A,B,SA,SB,Bias,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                BIAS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SMS:tl.constexpr):
    nm,nn=tl.cdiv(M,BM),tl.cdiv(N,BN)
    for tile in tl.range(tl.program_id(0),nm*nn,SMS):
        group=tile//(8*nn);size=tl.minimum(nm-group*8,8)
        mi=group*8+(tile%(8*nn))%size;ni=(tile%(8*nn))//size
        acc=tl.full((BM,BN),0,tl.int32)
        for ki in range(tl.cdiv(K,BK)):
            av=A.load([mi*BM,ki*BK]);bv=B.load([ni*BN,ki*BK])
            acc=tl.dot(av,bv.T,acc,out_dtype=tl.int32)
        rows=mi.to(tl.int64)*BM+tl.arange(0,BM);cols=ni*BN+tl.arange(0,BN)
        value=acc.to(tl.float32)*tl.load(SA+rows,rows<M,other=0)[:,None]
        value=value*tl.load(SB+cols,cols<N,other=0)[None,:]
        if BIAS:value=value+tl.load(Bias+cols,cols<N,other=0).to(tl.float32)[None,:]
        tl.store(C+rows[:,None]*N+cols[None,:],value,(rows[:,None]<M)&(cols[None,:]<N))

def matmul(a,b,sa,sb,bias=None,tile=(128,256,128,8,3)):
    """Keep LynnReal's schedule/epilogue; only update the host descriptor API."""
    m,k=a.shape;n=b.shape[1];bm,bn,bk,warps,stages=tile
    if a.dtype!=torch.int8 or b.dtype!=torch.int8 or not a.is_contiguous() or b.stride()!=(1,k):
        raise ValueError('TMA requires row-major INT8 activations and transposed row-major weights')
    if torch.cuda.get_device_capability(a.device)[0]!=9:
        raise ValueError('persistent TMA kernel requires Hopper')
    da=TensorDescriptor.from_tensor(a,[bm,bk])
    db=TensorDescriptor.from_tensor(b.T,[bn,bk])
    out=torch.empty((m,n),device=a.device,dtype=torch.bfloat16)
    sms=torch.cuda.get_device_properties(a.device).multi_processor_count
    grid=(min(sms,triton.cdiv(m,bm)*triton.cdiv(n,bn)),)
    _persistent[grid](da,db,sa,sb,bias,out,m,n,k,bias is not None,bm,bn,bk,sms,
        num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    return out
