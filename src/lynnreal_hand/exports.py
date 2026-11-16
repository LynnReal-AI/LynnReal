"""Create local Hugging Face weight packages. This module never uploads anything."""
from pathlib import Path
import json
import hashlib
import shutil
import tempfile


def save_transformer(state,config,output,*,max_shard_bytes=4<<30):
    from safetensors.torch import save_file
    output=Path(output)/'transformer';output.mkdir(parents=True,exist_ok=True)
    groups=[];current={};size=0;total=0
    for name,value in state.items():
        value=value.detach().cpu().contiguous();n=value.numel()*value.element_size()
        if current and size+n>max_shard_bytes:groups.append(current);current={};size=0
        current[name]=value;size+=n;total+=n
    if current:groups.append(current)
    mapping={}
    for i,group in enumerate(groups,1):
        name=f'diffusion_pytorch_model-{i:05d}-of-{len(groups):05d}.safetensors'
        save_file(group,str(output/name),metadata={'format':'pt'})
        mapping.update({k:name for k in group})
    (output/'diffusion_pytorch_model.safetensors.index.json').write_text(
        json.dumps(dict(metadata={'total_size':total},weight_map=mapping),indent=2)+'\n')
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')


def export_weights(component,source,output,*,dry_run=False):
    source,output=Path(source),Path(output)
    plan=dict(component=component,source=str(source),output=str(output),upload=False)
    if dry_run:return plan
    if not source.exists():raise FileNotFoundError(source)
    output.mkdir(parents=True,exist_ok=True)
    if component=='hand_encoder':
        import torch
        from safetensors.torch import save_file
        state=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
        values=state.get('ema_encoder',state.get('encoder'))
        if values is None:raise ValueError('Expected encoder or ema_encoder tensors')
        save_file({k:v.contiguous() for k,v in values.items()},str(output/'encoder.safetensors'))
        (output/'config.json').write_text(json.dumps(dict(architecture='taeh3',latent_channels=24,
            group_frames=17,group_latents=5,source='ema_encoder' if 'ema_encoder' in state else 'encoder'),indent=2)+'\n')
    elif component=='vae':
        config=json.loads((source/'config.json').read_text())
        # Keep runtime fields; omit private training paths and obsolete audit prose.
        public={k:config[k] for k in ['schema','model','parameters','normalization','required'] if k in config}
        (output/'config.json').write_text(json.dumps(public,indent=2)+'\n')
        shutil.copy2(source/'weights.safetensors',output/'weights.safetensors')
        hashes={}
        for name in ['config.json','weights.safetensors']:
            h=hashlib.sha256()
            with (output/name).open('rb') as f:
                for block in iter(lambda:f.read(16<<20),b''):h.update(block)
            hashes[name]=h.hexdigest()
        (output/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n')
    elif component in ('base','fast'):
        if (source/'transformer/config.json').is_file():
            entry = source/'config.json'
            packaged = False
            if entry.is_file():
                from .config import INFERENCE_PACKAGE_SCHEMA
                package = json.loads(entry.read_text())
                if package.get('schema') == INFERENCE_PACKAGE_SCHEMA:
                    if package.get('component') != component:
                        raise ValueError('The exported component differs from the source inference configuration')
                    packaged = True
            shutil.copytree(source/'transformer',output/'transformer',dirs_exist_ok=True)
            if packaged:
                shutil.copy2(entry, output/'config.json')
                if (source/'model_index.json').is_file():
                    shutil.copy2(source/'model_index.json', output/'model_index.json')
        elif component=='base' and (source/'model/.metadata').is_file():
            import torch
            from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
            with tempfile.TemporaryDirectory(prefix='lynn-base-export-') as directory:
                archive=Path(directory)/'model.pt'
                dcp_to_torch_save(source/'model',archive)
                state=torch.load(archive,map_location='cpu',weights_only=True,mmap=True)['model']
                save_transformer(state,json.loads((source/'config.json').read_text()),output)
        else:raise ValueError('Source must contain a complete transformer directory')
    else:raise ValueError(component)
    return plan
