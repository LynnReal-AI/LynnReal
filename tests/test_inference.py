"""CPU validation for the H3 inference-only distribution; no model weights."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from lynnreal_hand.camera import convert_camera
from lynnreal_hand.config import load_config,validate_config,canvas
from lynnreal_hand.data import save_cache,load_cache,NATIVE_LENGTHS


def test_public_qwen_conditioner_exposes_required_multimodal_api():
    # A fresh 4.57 installation imported successfully but failed only after
    # loading the real 32B conditioner. Catch that deployment mismatch early.
    from transformers import Qwen3VLProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    import inspect
    assert callable(getattr(Qwen3VLProcessor, 'create_mm_token_type_ids', None))
    assert 'mm_token_type_ids' in inspect.signature(Qwen3VLModel.forward).parameters
from lynnreal_hand.cli import parser

ROOT=Path(__file__).resolve().parents[1]


def activate():
    from lynnreal_hand.backends.h3.runtime import activate
    activate()


def test_h3_only_inference_distribution():
    assert {p.name for p in (ROOT/'src/lynnreal_core/backends').iterdir() if p.is_dir() and p.name!='__pycache__'}=={'minimax_h3'}
    assert {p.name for p in (ROOT/'src/lynnreal_hand/backends').iterdir() if p.is_dir() and p.name!='__pycache__'}=={'h3'}
    assert not (ROOT/'src/lynnreal_hand/training').exists()
    assert not (ROOT/'src/lynnreal_hand/losses').exists()
    assert not (ROOT/'src/lynnreal_core/training').exists()
    assert {p.name for p in (ROOT/'configs').glob('*.json')}=={'base.json','fast.json','fast_768.json'}
    for command in ['train','prepare-image-text']:
        with pytest.raises(SystemExit) as e:parser().parse_args([command])
        assert e.value.code==2
    with pytest.raises(SystemExit):parser().parse_args(['prepare','--config','fast.json','--output','out','--training'])


def test_config_and_fast_contract(tmp_path):
    c=load_config(ROOT/'configs/fast.json',weights_root=tmp_path)
    assert canvas(c)==(544,960)
    assert c['model']['checkpoint_path']==str(tmp_path/'fast')
    for k,v in [('steps',4),('window',3)]:
        changed=copy.deepcopy(c);changed['sampling'][k]=v
        with pytest.raises(ValueError):validate_config(changed)
    changed=copy.deepcopy(c);changed['training']={'max_steps':1}
    with pytest.raises(ValueError,match='inference only'):validate_config(changed)
    c['model']['attention_backend']='auto'
    with pytest.raises(ValueError):validate_config(c)


def test_native_768_fast_keeps_the_sampling_contract(tmp_path):
    native=load_config(ROOT/'configs/fast_768.json',weights_root=tmp_path)
    default=load_config(ROOT/'configs/fast.json',weights_root=tmp_path)
    assert canvas(native)==(768,1344)
    assert native['sampling']==default['sampling']
    assert native['model']==default['model']
    native['video']['width']=960
    with pytest.raises(ValueError,match='1344x768'):validate_config(native)


@pytest.mark.parametrize('height,width',[(540,960),(768,1344)])
def test_fast_decoder_tail_preserves_native_canvas(height,width):
    from lynnreal_hand.models.codecs import LynnRealDecoder
    decoder=LynnRealDecoder.__new__(LynnRealDecoder)
    decoder.compiled=False;decoder.height=height
    canvas_height=544 if height==540 else height
    def decode_tail(z):
        assert z.shape[2]==2
        rows=torch.linspace(0,1,canvas_height).reshape(1,1,1,canvas_height,1)
        return rows.expand(1,3,5,canvas_height,width)
    decoder.impl=decode_tail
    output=decoder.decode(torch.zeros(1,24,2,canvas_height//16,width//16))
    assert output.shape==(5,height,width,3)
    assert output.dtype==torch.uint8
    if height==768:
        assert output[0,0,0,0]==0 and output[0,-1,0,0]==255
    else:
        assert output[0,0,0,0]>0 and output[0,-1,0,0]<255


@pytest.mark.parametrize('frames,length',NATIVE_LENGTHS.items())
def test_physical_camera_clock(tmp_path,frames,length):
    poses=np.tile(np.eye(4,dtype=np.float32),(frames,1,1));poses[:,0,3]=3+np.arange(frames)*.01
    source=tmp_path/'camera.npz';np.savez(source,camera_c2w=poses)
    output=convert_camera(source,tmp_path/'aligned.npz',frames=frames);a=np.load(output)
    expected=np.array([start+offset for start in range(0,frames,17) for offset in [0,1,5,9,13] if start+offset<frames])
    assert len(expected)==length
    np.testing.assert_array_equal(a['source_frame_indices'],expected)
    np.testing.assert_allclose(a['camera_viewmats'][:,0,3],-expected*.01,atol=1e-6)
    assert a['camera_K'][0,0,2]==.5


def test_future_reference_tensors_are_ignored(tmp_path):
    shape=(24,37,2,2)
    values=dict(prompt_embeds=torch.zeros(3,8),text_token_tags=torch.zeros(3,dtype=torch.long),
        anchor_latents=torch.zeros(24,1,2,2),hand_latents=torch.zeros(shape),target_latents=torch.ones(shape)*.321,
        teacher_hand_latents=torch.ones(shape),camera_viewmats=torch.eye(4).repeat(37,1,1),camera_K=torch.eye(3).repeat(37,1,1))
    path=save_cache(values,dict(sample_id='one',caption='hands',pixel_frames=124,target_shape=shape,variant='stage3',fps=30),tmp_path/'cache')
    batch=load_cache(path)
    assert torch.isnan(batch.target_latents).all()
    assert not hasattr(batch,'teacher_hand_latents')
    assert torch.equal(batch.hand_latents,values['hand_latents'])


def test_installed_backend_import_closure():
    activate()
    names=['lynnreal_core.backends.minimax_h3.model','lynnreal_core.backends.minimax_h3.loading',
        'lynnreal_hand.models.h3','lynnreal_hand.models.codecs','lynnreal_hand.prepare',
        'lynnreal_hand.inference.base','lynnreal_hand.inference.fast','lynnreal_hand.inference.accelerated',
        'lynnreal_hand.inference.packed_graph','inference_core','base_conditioning',
        'full_fusion','native_kernel_ops','precision_boundary','codec540','ultra_fast_vae.model']
    for name in names:
        m=importlib.import_module(name)
        assert Path(m.__file__).is_relative_to(ROOT/'src')
    from inference_core import ReferenceStudent
    assert not hasattr(ReferenceStudent,'rollout') and not hasattr(ReferenceStudent,'replay')


def test_w2_positions_and_joint_cfg():
    activate()
    from types import SimpleNamespace
    from geometry import layout,student_layout,window_positions
    from base_conditioning import pack,captionless_without_hand
    batch=SimpleNamespace(target_latents=torch.full((24,37,2,2),float('nan')),hand_latents=torch.zeros(24,37,2,2),
        anchor_latents=torch.zeros(24,1,2,2),prompt_embeds=torch.zeros(4,8),text_token_tags=torch.tensor([1,0,0,1]),
        camera_viewmats=torch.eye(4).repeat(37,1,1),camera_K=torch.eye(3).repeat(37,1,1),task='i2v_cam_hand')
    batch.camera_viewmats[:,0,3]=torch.arange(37)
    doc=student_layout(layout(batch,'cpu','HAND',latent_frames=37))
    torch.testing.assert_close(window_positions(doc,1,1,2),window_positions(doc,5,5,2))
    assert len(window_positions(doc,7,7,2))==2
    positive=pack(batch,torch.zeros(1,24,37,2,2),.3,device='cpu',latent_frames=37,seed=42)
    negative=captionless_without_hand(positive)
    assert positive.hand_rows==37 and negative.hand_rows==0
    assert negative.anchor_rows==positive.anchor_rows==1
    torch.testing.assert_close(positive.inputs['cam_viewmats'][0,:,0,3],torch.arange(37).float())


def test_reference_schedule_has_three_maps():
    activate()
    from inference_core import ReferenceStudent
    s=ReferenceStudent(None,'cpu','HAND',lambda *a,**k:None)
    expected=3*torch.linspace(1,0,4)/(1+2*torch.linspace(1,0,4))
    torch.testing.assert_close(torch.as_tensor(s.sigmas()),expected)


def test_full_fast_roundtrip_and_checksums(tmp_path):
    activate()
    from lynnreal_core.backends.minimax_h3.model import SolarMiniMaxH3Transformer3DModel as Model
    from lynnreal_hand.checkpoints import verify_checksums
    # Exercise the actual Diffusers constructor and strict loader: interval
    # weights must exist before loading and must never be reset afterward.
    cfg=dict(num_attention_heads=2,attention_head_dim=16,hidden_size=32,
        num_layers=1,num_refiner_layers=1,ffn_dim=64,in_channels=4,audio_in_channels=4,
        text_dim=16,freq_dim=8,time_embed_hidden_dim=32,time_embed_dim=16,
        lynnreal_variant='fast',anyflow_gate=.25)
    model=Model.from_config(cfg)
    assert model.uses_anyflow
    model.to(torch.bfloat16)
    for name in ('proj_in','audio_proj_in','time_embedder','proj_out','audio_proj_out','delta_embedding'):
        getattr(model,name).float()
    with torch.no_grad():model.delta_embedding.linear_1.weight.add_(.125)
    model.save_pretrained(tmp_path/'fast',safe_serialization=True)
    restored=Model.strict_from_pretrained(tmp_path/'fast',local_files_only=True,torch_dtype=torch.bfloat16)
    assert restored.uses_anyflow
    for key,value in model.state_dict().items():
        assert restored.state_dict()[key].dtype==value.dtype
        torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)
    assert not torch.equal(restored.delta_embedding.linear_1.weight,restored.time_embedder.linear_1.weight)
    t=torch.tensor([.5]);r=torch.tensor([.7])
    kw=dict(parameter_dtype=lambda m:next(m.parameters()).dtype)
    torch.testing.assert_close(model._time_condition(t,r,**kw),restored._time_condition(t,r,**kw))
    (tmp_path/'fast/SHA256SUMS.json').write_text(json.dumps({'config.json':'0'*64}))
    with pytest.raises(ValueError,match='checksum'):verify_checksums(tmp_path/'fast')


def test_example_and_missing_weights_fail_before_cuda():
    example=ROOT/'examples/02/example.json'
    for name in ['base','fast']:
        result=subprocess.run([sys.executable,'-m','lynnreal_hand','validate','--config',str(ROOT/f'configs/{name}.json'),'--example',str(example)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        assert json.loads(result.stdout)['passed']
    failed=subprocess.run([sys.executable,'-m','lynnreal_hand','infer','--config',str(ROOT/'configs/fast.json'),'--example',str(example),'--output',str(ROOT/'outputs/not-created.mp4'),'--weights-root','/nonexistent/lynn-test'],capture_output=True,text=True)
    assert failed.returncode==2 and 'Weights are not bundled' in failed.stderr and 'CUDA' not in failed.stderr
