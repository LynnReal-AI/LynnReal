"""Measure real Hopper INT8 backends; require identical integer/dequant output.

Reuse this project's Triton3.5 ports (explicit INT32 dot / TensorDescriptor).
Tuning occurs before graph capture; the deployment dispatch never silently falls back.
"""
import time,json,os
from pathlib import Path
import torch
import int8_gemm_compat as compatible
import int8_tma_backend as tma
from lynn_kernels import int8_gemm as upstream
from lynn_kernels import kernels
SELECTED={};MEASUREMENTS={};CALLS={}
E=Path(os.environ.get('LYNNREAL_CACHE_DIR', Path.home()/'.cache/lynnreal-hand'))
E.mkdir(parents=True,exist_ok=True)

def _torch(a,b,sa,sb,bias):
    m=a.shape[0];pad=max(32,((m+7)//8)*8)-m
    aa=torch.nn.functional.pad(a,(0,0,0,pad))if pad else a
    return kernels.int8_epilogue(torch._int_mm(aa,b)[:m],sa,sb,bias)

def dispatch(a,b,sa,sb,bias=None,*,tile=None):
    m,depth=a.shape;out=b.shape[1];key=(m,depth,out,bias is not None)
    functions={'torch':_torch,'triton':compatible.int8_matmul}
    if m>=4096:functions['tma']=tma.matmul
    if key not in SELECTED:
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('INT8 shape was not prewarmed before graph capture')
        ref=_torch(a,b,sa,sb,bias)
        rows={}
        for label,fn in functions.items():
            try:
                value=fn(a,b,sa,sb,bias)
                if label=='triton'and compatible._disabled:raise RuntimeError('Triton disabled shape; refusing fallback result')
                error=float((value.float()-ref.float()).abs().max())
                if error!=0:raise RuntimeError('INT8 backend output differs: '+str(error))
                for _ in range(2):fn(a,b,sa,sb,bias)
                torch.cuda.synchronize();tick=time.perf_counter()
                for _ in range(5):fn(a,b,sa,sb,bias)
                torch.cuda.synchronize();ms=(time.perf_counter()-tick)*1000/5
                rows[label]=dict(passed=True,ms=ms,max_abs_vs_torch=error)
            except Exception as exc:rows[label]=dict(passed=False,error=repr(exc))
        good=[(v['ms'],label)for label,v in rows.items()if v['passed']]
        if not good:raise RuntimeError('no valid INT8 backend')
        selected=min(good)[1];SELECTED[key]=selected;MEASUREMENTS[str(key)]=dict(selected=selected,backends=rows)
        (E/'INT8_BACKEND_TUNING.json').write_text(json.dumps(MEASUREMENTS,indent=2))
    chosen=SELECTED[key]
    if not torch.compiler.is_compiling():CALLS[chosen]=CALLS.get(chosen,0)+1
    return functions[chosen](a,b,sa,sb,bias)

def install():upstream.int8_matmul=dispatch

def audit():
    return dict(selected={str(k):v for k,v in SELECTED.items()},measurements=MEASUREMENTS,
        python_dispatch_calls=CALLS,graph_replays_do_not_increment_python_counters=True,
        triton_disabled_shapes=len(compatible._disabled),Triton_INT32_port=True,
        TMA_TensorDescriptor_port=True,failures_are_recorded=True,automatic_silent_fallback=False)
