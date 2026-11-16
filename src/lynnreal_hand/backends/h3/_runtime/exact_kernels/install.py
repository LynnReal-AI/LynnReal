"""Instance-local exact pointwise/RoPE integration; retain LynnReal PRoPE/SP/FA3.

No global monkeypatch, parameter changes, FFN chunking or norm replacement.
The LynnReal attention function is cloned with one private global substituted:
the elementwise rotary implementation. All camera/collective/backend code is
the original function body. This avoids maintaining a divergent attention copy.
"""
import copy,hashlib,inspect,types
import torch
from .pointwise_ops import fused
from .rotary_dispatch import fused_rotary

def block_forward(self,hidden_states,temb,adaln_indices,rotary_emb,attention_mask=None):
    sm,cm,gm,sf,cf,gf=self.adaln_proj(temb)
    residual=hidden_states
    h=fused('modulate',self.norm1(hidden_states),cm.index_select(0,adaln_indices),sm.index_select(0,adaln_indices))
    h=self.attn(h,rotary_emb,attention_mask)
    hidden_states=fused('gated_residual',residual,gm.index_select(0,adaln_indices),h)
    residual=hidden_states
    h=fused('modulate',self.norm2(hidden_states),cf.index_select(0,adaln_indices),sf.index_select(0,adaln_indices))
    h=self.ff(h)
    return fused('gated_residual',residual,gf.index_select(0,adaln_indices),h)

def swiglu_forward(self,hidden_states):
    return fused('swiglu',self.proj(hidden_states))

def _clone(fn,changes):
    namespace=dict(fn.__globals__);namespace.update(changes)
    result=types.FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__,fn.__closure__)
    result.__kwdefaults__=fn.__kwdefaults__
    return result

def install(model):
    from diffusers.models.activations import SwiGLU
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
    from lynnreal_core.backends.minimax_h3.model import SolarMiniMaxH3Transformer3DModel as Solar
    from lynnreal_core.backends.minimax_h3.model import SolarMiniMaxH3AttnProcessor as Processor
    attention=Solar._attention_forward.__func__
    source=inspect.getsource(attention)
    assert source.count('_apply_rotary_emb(')==2 and '_prepare_prope(' in source
    assert 'sequence_model_parallel_all_to_all_4D' in source and 'dispatch_attention_fn(' in source
    private_type=type('FusedH3Attention',(Solar,),
                      {'_attention_forward':classmethod(_clone(attention,{'_apply_rotary_emb':fused_rotary}))})
    private_processor=type('FusedH3AttnProcessor',(Processor,),
                           {'__call__':_clone(Processor.__call__,{'SolarMiniMaxH3Transformer3DModel':private_type})})
    inventory=tuple((n,id(p),p.dtype,p.requires_grad)for n,p in model.named_parameters())
    report=dict(profile='exact_rope_pointwise',attention_body_sha256=hashlib.sha256(source.encode()).hexdigest(),
                block_modules=[],swiglu_modules=[],FFN_chunking=False,RMSNorm_changed=False,
                PRoPE_changed=False,SP_changed=False,global_patch=False)
    for name,module in model.named_modules():
        if type(module) is MiniMaxH3TransformerBlock:
            assert 'forward' not in module.__dict__,name
            original=module.attn.processor
            assert type(original)is Processor,(name,type(original))
            replacement=private_processor();replacement.__dict__.update(copy.copy(original.__dict__))
            assert getattr(replacement,'_attention_backend',None)==getattr(original,'_attention_backend',None)
            module.attn.set_processor(replacement)
            module.forward=types.MethodType(block_forward,module)
            report['block_modules'].append(name)
        elif type(module)is SwiGLU:
            assert type(module.activation)is torch.nn.SiLU and not module.activation.inplace
            assert 'forward'not in module.__dict__,name
            module.forward=types.MethodType(swiglu_forward,module)
            report['swiglu_modules'].append(name)
    assert report['block_modules'] and report['swiglu_modules']
    assert inventory==tuple((n,id(p),p.dtype,p.requires_grad)for n,p in model.named_parameters())
    return report
