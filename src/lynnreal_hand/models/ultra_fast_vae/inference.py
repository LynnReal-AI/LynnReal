"""Ultra-Fast-VAE decoder with explicit loading, compilation, and warmup.

Input: normalized native H3 latent CUDA FP32 [1,24,5,48,84].
Output: borrowed pinned CPU uint8 [17,768,1344,3], reusable after two calls.
One instance holds immutable weights; new_stream() owns independent buffers.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
from torch import nn
from .model import ThinDecoder,NativeCanvas

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for piece in iter(lambda:stream.read(16*1024*1024),b''):digest.update(piece)
    return digest.hexdigest()

def verify_artifacts(root):
    from lynnreal_hand.checkpoints import verify_checksums
    root=Path(root)
    if verify_checksums(root):return {'schema':'lynnreal-hand.weights.v1','verified':True}
    legacy=root/'ARTIFACT_HASHES.json'
    if legacy.is_file():
        manifest=json.loads(legacy.read_text())
        for name in ['config.json','weights.safetensors']:
            expected=manifest['files'].get(name)
            if expected is None or sha256(root/name)!=expected:
                raise RuntimeError('Artifact hash mismatch or missing hash: '+name)
        return {'schema':'thin-h3-export-v3','verified':True,'scope':'config and weights'}
    raise FileNotFoundError('VAE package requires SHA256SUMS.json; create it with local VAE export')

def load_float(root,device='cpu'):
    from safetensors.torch import load_file
    root=Path(root);verify_artifacts(root);config=json.loads((root/'config.json').read_text());state=load_file(str(root/'weights.safetensors'),device='cpu')
    decoder=ThinDecoder(config['model']);post=nn.Conv3d(24,24,1,device='meta',dtype=torch.float16)
    decoder.load_state_dict({k[len('decoder.'):]:v for k,v in state.items()if k.startswith('decoder.')},strict=True,assign=True)
    post.load_state_dict({k[len('post.'):]:v for k,v in state.items()if k.startswith('post.')},strict=True,assign=True)
    assert sum(p.numel()for p in decoder.parameters())==config['parameters']
    assert all('teacher'not in n and'encoder'not in n for n in state)
    return decoder.to(device).eval().requires_grad_(False),post.to(device).eval().requires_grad_(False),config

def validate_input(current,device):
    if current.shape!=(1,24,5,48,84)or current.dtype!=torch.float32 or current.device!=device or not current.is_contiguous():
        raise ValueError('Expected contiguous FP32 native normalized H3 latent [1,24,5,48,84] on '+str(device))

class Stream:
    def __init__(self,owner,fixture):
        self.owner=owner;self.count=0;self.device=fixture.device;validate_input(fixture,self.device)
        with torch.inference_mode():
            self.input=fixture.clone();warm=torch.cuda.Stream(device=self.device);warm.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(warm):
                for _ in range(3):owner.compiled(self.input)
            torch.cuda.current_stream(self.device).wait_stream(warm);torch.cuda.synchronize(self.device);self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.output=owner.compiled(self.input)
        self.banks=[torch.empty((17,768,1344,3),dtype=torch.uint8,pin_memory=True)for _ in range(2)]
    @torch.inference_mode()
    def decode_gpu(self,current):
        validate_input(current,self.device);self.input.copy_(current,non_blocking=True);self.graph.replay();return self.output
    @torch.inference_mode()
    def decode(self,current):
        output=self.decode_gpu(current);bank=self.banks[self.count%2];bank.copy_(output,non_blocking=True);torch.cuda.synchronize(self.device);self.count+=1;return bank
    @torch.inference_mode()
    def reset(self):self.input.zero_();self.count=0

class Decoder:
    def __init__(self,root,device='cuda:0'):
        from .h3_fa3_policy import require_h3_fa3_training_backend
        from .stable_vae_ops import StableBlock,OpaqueLinear,denormalize_half
        from .vae_precision_runtime import install_policy
        from .fused_quant_vae import FusedQuantBlock
        self.root=Path(root).resolve();self.device=torch.device(device)
        if self.device.index is None:self.device=torch.device(self.device.type,torch.cuda.current_device())
        require_h3_fa3_training_backend('_flash_3',self.device)
        if torch.cuda.device_count()!=1 or 'H200'not in torch.cuda.get_device_name(self.device):raise RuntimeError('The optimized decoder requires one visible NVIDIA H200')
        self.decoder,self.post,self.config=load_float(self.root,self.device);self.decoder.proj_in=OpaqueLinear(self.decoder.proj_in)
        blocks=[StableBlock(b)for b in self.decoder.transformer_blocks];self.decoder.transformer_blocks=nn.ModuleList(blocks);projections=[]
        for b in blocks:
            for owner,name in [(b.attn,'to_qkv'),(b.attn.to_out,'0'),(b.ff.net[0],'proj'),(b.ff.net,'2')]:projections.append((owner,name,getattr(owner,name)))
        install_policy(projections,'int8_all');self.decoder.transformer_blocks=nn.ModuleList([FusedQuantBlock(b)for b in blocks]);self.canvas=NativeCanvas(self.decoder,self.post)
        norm=self.config['normalization'];self.mean=torch.tensor(norm['latent_mean'],device=self.device).view(1,24,1,1,1);self.std=torch.tensor(norm['latent_std'],device=self.device).view(1,24,1,1,1)
        self.pixel_mean=torch.tensor(norm['pixel_mean'],device=self.device).view(1,3,1,1,1);self.pixel_std=torch.tensor(norm['pixel_std'],device=self.device).view(1,3,1,1,1)
        torch._inductor.config.freezing=False;torch._inductor.config.emulate_precision_casts=True;torch._dynamo.config.recompile_limit=32;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=True
        def packed(current):
            z=denormalize_half(current,self.mean,self.std)
            with torch.autocast('cuda',dtype=torch.float16):raw=self.canvas(z)[:,:,3:20]
            return (((raw.float()*self.pixel_std+self.pixel_mean).clamp(0,1)[0]*255).round().clamp(0,255).byte().permute(1,2,3,0).contiguous())
        self.packed=packed;self.compiled=torch.compile(packed,fullgraph=True,dynamic=False,mode='default');self.ready=False;self.default_stream=None
    @torch.inference_mode()
    def warmup(self,fixture):
        validate_input(fixture,self.device);expected=self.packed(fixture).clone();actual=self.compiled(fixture)
        eager_error=int((expected.short()-actual.short()).abs().max())
        if eager_error>1:raise RuntimeError('Compiled/eager RGB error '+str(eager_error))
        self.ready=True;self.default_stream=self.new_stream(fixture);expected=self.compiled(fixture).clone();actual=self.default_stream.decode_gpu(fixture)
        graph_error=int((expected.short()-actual.short()).abs().max());assert graph_error==0
        for _ in range(15):self.default_stream.decode(fixture)
        return dict(compiled_eager_max_code_error=eager_error,graph_max_code_error=graph_error,warmups=15)
    def new_stream(self,fixture):
        if not self.ready:raise RuntimeError('Call warmup once before creating streams')
        return Stream(self,fixture)
    def decode(self,current):
        if self.default_stream is None:raise RuntimeError('Call warmup before timed decode')
        return self.default_stream.decode(current)
    def reset(self):
        if self.default_stream is None:raise RuntimeError('Call warmup before reset')
        self.default_stream.reset()
    def benchmark(self,inputs,minimum=501):
        if len(inputs)<minimum:raise ValueError('Require >500 real current chunks for delivery timing')
        values=[]
        for current in inputs:
            start=time.perf_counter();self.decode(current);values.append((time.perf_counter()-start)*1000)
        return dict(p50_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),maximum_ms=max(values),steady_chunks=len(values),samples_ms=values,
            endpoint='Public Decoder.decode: inputD2D+allINT8/FA3/fusion/28tiles/stitch/pack/pinnedD2H/sync; no historicalCNN/KV; excludes load/compile/warmup/IO')
