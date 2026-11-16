"""Student-only architecture; no encoder, teacher or project imports.

The forward and native geometry reproduce MiniMax H3 / LynnReal definitions.
Inference acceleration is installed separately after strict weight loading.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self,width,heads,head_dim,eps,device):
        super().__init__();self.heads=heads;self.dim_head=head_dim;self.processor=None
        self.norm_q=nn.RMSNorm(head_dim,eps=eps,elementwise_affine=False,device=device)
        self.norm_k=nn.RMSNorm(head_dim,eps=eps,elementwise_affine=False,device=device)
        self.to_qkv=nn.Linear(width,3*heads*head_dim,device=device,dtype=torch.float16)
        self.to_out=nn.ModuleList([nn.Linear(heads*head_dim,width,device=device,dtype=torch.float16),nn.Dropout(0.)])
    def set_processor(self,processor):self.processor=processor
    def forward(self,x,rotary_emb=None):
        if self.processor is None:raise RuntimeError('An explicit attention processor must be installed')
        return self.processor(self,x,rotary_emb)

class SwiGLU(nn.Module):
    def __init__(self,width,ffn,device):
        super().__init__();self.proj=nn.Linear(width,2*ffn,device=device,dtype=torch.float16)
    def forward(self,x):
        value,gate=self.proj(x).chunk(2,dim=-1);return value*F.silu(gate)

class FeedForward(nn.Module):
    def __init__(self,width,ffn,device):
        super().__init__();self.net=nn.ModuleList([SwiGLU(width,ffn,device),nn.Dropout(0.),nn.Linear(ffn,width,device=device,dtype=torch.float16)])
    def forward(self,x):
        for module in self.net:x=module(x)
        return x

class Block(nn.Module):
    def __init__(self,width,heads,head_dim,ffn,eps,device):
        super().__init__();self.norm1=nn.RMSNorm(width,eps=eps,device=device,dtype=torch.float32)
        self.norm2=nn.RMSNorm(width,eps=eps,device=device,dtype=torch.float32)
        self.attn=Attention(width,heads,head_dim,eps,device);self.ff=FeedForward(width,ffn,device)
        self.scale1=nn.Parameter(torch.zeros(width,device=device,dtype=torch.float16));self.scale2=nn.Parameter(torch.zeros(width,device=device,dtype=torch.float16))
    def forward(self,x,rotary_emb):
        n=self.norm1(x.float()).to(x.dtype);x=x+self.attn(n,rotary_emb)*self.scale1
        n=self.norm2(x.float()).to(x.dtype);return x+self.ff(n)*self.scale2

class FixedRoPE(nn.Module):
    def __init__(self,shape,device):
        super().__init__();self.register_buffer('cosine',torch.empty(shape,device=device,dtype=torch.float32));self.register_buffer('sine',torch.empty(shape,device=device,dtype=torch.float32))
    def forward(self,ids):
        assert ids.shape[1]==self.cosine.shape[1]
        return self.cosine.expand(ids.shape[0],-1,-1,-1).contiguous(),self.sine.expand(ids.shape[0],-1,-1,-1).contiguous()

class ThinDecoder(nn.Module):
    def __init__(self,config,device='meta'):
        super().__init__();c=config;w=c['width'];eps=c['norm_eps'];self.patch_size=c['patch_size'];self.patch_size_t=c['patch_size_t'];self.out_channels=c['out_channels'];self.num_register_tokens=c['register_tokens']
        self.proj_in=nn.Linear(c['latent_channels'],w,device=device,dtype=torch.float16)
        self.register_tokens=nn.Parameter(torch.zeros(1,self.num_register_tokens,w,device=device,dtype=torch.float16))
        self.rope=FixedRoPE(c['rope_shape'],device)
        self.transformer_blocks=nn.ModuleList([Block(w,c['heads'],c['head_dim'],c['ffn'],eps,device)for _ in range(c['layers'])])
        self.norm_out=nn.LayerNorm(w,eps=eps,device=device,dtype=torch.float32)
        self.proj_out=nn.Linear(w,self.out_channels*self.patch_size_t*self.patch_size**2,device=device,dtype=torch.float16)
    def forward(self,z):
        b,c,t,h,w=z.shape;x=z.permute(0,2,3,4,1).reshape(b,t*h*w,c);x=self.proj_in(x);patches=x.shape[1]
        x=torch.cat((x,self.register_tokens.expand(b,-1,-1),torch.zeros_like(x[:,:1])),1)
        grids=[2*(torch.arange(.5,n,dtype=torch.float32,device=x.device)/n)-1 for n in (t,h,w)]
        ids=torch.stack(torch.meshgrid(*grids,indexing='ij'),-1).flatten(0,2)[None].expand(b,-1,-1)
        ids=torch.cat((ids,ids.new_zeros((b,self.num_register_tokens+1,3))),1);rotary=self.rope(ids)
        for block in self.transformer_blocks:x=block(x,rotary)
        x=self.proj_out(self.norm_out(x))[:,:patches];p,pt=self.patch_size,self.patch_size_t
        x=x.reshape(b,t,h,w,self.out_channels,pt,p,p).permute(0,4,1,5,2,6,3,7).contiguous()
        return x.reshape(b,self.out_channels,t*pt,h*p,w*p)

def split_tiles(length,size,overlap,ratio=16):
    if size>=length:return [0],[length],[]
    count=math.ceil(length/size)
    while size*count-overlap*(count-1)-length<0:count+=1
    overlaps=[overlap]*(count-1);remaining=size*count-sum(overlaps)-length
    for i in range(remaining//ratio):overlaps[i%(count-1)]+=ratio
    starts=[0]
    for amount in overlaps:starts.append(starts[-1]+size-amount)
    return starts,[size]*count,overlaps

def blend(a,b,extent,dim):
    extent=min(a.shape[dim],b.shape[dim],extent);positions=torch.arange(extent,device=b.device,dtype=b.dtype);shape=[1]*a.ndim;shape[dim]=extent
    wa=(1-positions/extent).view(shape);wb=(positions/extent).view(shape);sa=[slice(None)]*a.ndim;sb=sa.copy();sa[dim]=slice(-extent,None);sb[dim]=slice(0,extent)
    value=a[tuple(sa)]*wa+b[tuple(sb)]*wb
    if extent==b.shape[dim]:return value
    rest=[slice(None)]*b.ndim;rest[dim]=slice(extent,None);return torch.cat((value,b[tuple(rest)]),dim=dim)

def stitch(tiles,y_overlap,x_overlap):
    rows=[]
    for i,row in enumerate(tiles):
        columns=[]
        for j,tile in enumerate(row):
            if i and y_overlap[i-1]>0:tile=blend(tiles[i-1][j],tile,y_overlap[i-1],-2)
            if j and x_overlap[j-1]>0:tile=blend(row[j-1],tile,x_overlap[j-1],-1)
            if i<len(tiles)-1 and y_overlap[i]>0:tile=tile[...,:-y_overlap[i],:]
            if j<len(row)-1 and x_overlap[j]>0:tile=tile[...,:,:-x_overlap[j]]
            columns.append(tile)
        rows.append(torch.cat(columns,dim=-1))
    return torch.cat(rows,dim=-2)

class NativeCanvas(nn.Module):
    def __init__(self,decoder,post):
        super().__init__();self.decoder=decoder;self.post=post
        self.ys,self.heights,self.y_overlap=split_tiles(768,256,64);self.xs,self.widths,self.x_overlap=split_tiles(1344,256,64)
        assert len(self.ys)*len(self.xs)==28
    def forward(self,z):
        assert z.shape==(1,24,5,48,84)
        tiles=[z[...,y//16:(y+h)//16,x//16:(x+w)//16]for y,h in zip(self.ys,self.heights)for x,w in zip(self.xs,self.widths)]
        decoded=self.decoder(self.post(torch.cat(tiles)));pieces=list(decoded.split(1));rows=[pieces[i:i+len(self.xs)]for i in range(0,len(pieces),len(self.xs))]
        return stitch(rows,self.y_overlap,self.x_overlap)
