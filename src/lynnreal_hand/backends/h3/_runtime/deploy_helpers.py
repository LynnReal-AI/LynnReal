import torch, statistics, time
from pathlib import Path
import json
from reference_kv import transformer_blocks
write = None
def relative(a, b):
    return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-8))

def gated_compile(fn, inputs, name, tolerance, *, graphs=False):
    eager=fn
    with torch.inference_mode():
        try:
            ref=eager(*inputs).clone()
            candidate=torch.compile(eager,fullgraph=True,dynamic=False,
                options={'triton.cudagraphs':graphs,'triton.cudagraph_trees':False,'emulate_precision_casts':True})
            pred=candidate(*inputs)
            error=relative(pred,ref)
            if error>tolerance:raise ValueError(f'compile relative L2 {error}')
            def speed(f):
                values=[]
                for _ in range(3):
                    torch.cuda.synchronize();start=time.perf_counter();f(*inputs);torch.cuda.synchronize()
                    values.append((time.perf_counter()-start)*1000)
                return statistics.median(values)
            old,new=speed(eager),speed(candidate)
            accepted=new<old
            write(f'gates/{name}.json',dict(passed=True,enabled=accepted,relative_L2=error,eager_ms=old,compiled_ms=new))
            return candidate if accepted else eager
        except Exception as exc:
            write(f'gates/{name}.json',dict(passed=False,enabled=False,error=str(exc)))
            return eager

def fuse_qkv(model):
    for _,_,block in transformer_blocks(model):
        a=block.attn;parts=[a.to_q,a.to_k,a.to_v]
        assert len({tuple(x.weight.shape) for x in parts})==1
        layer=torch.nn.Linear(parts[0].in_features,3*parts[0].out_features,
            bias=parts[0].bias is not None,device='meta',dtype=parts[0].weight.dtype)
        weight=torch.cat([x.weight.detach() for x in parts],0)
        layer.weight=torch.nn.Parameter(weight,requires_grad=False)
        if layer.bias is not None:
            layer.bias=torch.nn.Parameter(torch.cat([x.bias.detach()for x in parts]),requires_grad=False)
        # Keep reference projections as views: no duplicate 50-layer QKV weights.
        for i,x in enumerate(parts):
            x.weight.data=weight[i*x.out_features:(i+1)*x.out_features]
            if x.bias is not None:x.bias.data=layer.bias.data[i*x.out_features:(i+1)*x.out_features]
        a.to_qkv_deploy=layer

def time_cache(model):
    counts=dict(hits=0,misses=0)
    def memo(fn):
        cache={}
        def call(*args,**kwargs):
            key=model._deploy_time_key
            if key not in cache:
                cache[key]=fn(*args,**kwargs)
                if not torch.compiler.is_compiling():counts['misses']+=1
            elif not torch.compiler.is_compiling():counts['hits']+=1
            return cache[key]
        return call
    model._time_condition=memo(model._time_condition)
    for _,_,block in transformer_blocks(model):block.adaln_proj.forward=memo(block.adaln_proj.forward)
    return counts
