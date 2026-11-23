"""Public entry points. Argument help and schema validation work without CUDA."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import sys
from .config import load_config,canvas


def parser():
    p=argparse.ArgumentParser(prog='lynnreal-hand',description='LynnReal-Hand Base/Fast video inference')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ['validate','prepare','infer']:
        q=sub.add_parser(name);q.add_argument('--config',required=True);q.add_argument('--weights-root')
        if name in ('validate','prepare','infer'):q.add_argument('--example')
        if name=='validate':q.add_argument('--check-weights',action='store_true')
        if name=='prepare':
            q.add_argument('--output',required=True)
        if name=='infer':
            q.add_argument('--cache');q.add_argument('--output',required=True);q.add_argument('--seed',type=int,default=42)
            q.add_argument('--accelerated',action='store_true');q.add_argument('--save-latents',action='store_true')
    q=sub.add_parser('export');q.add_argument('--component',required=True,choices=['base','fast','hand_encoder','vae'])
    q.add_argument('--source',required=True);q.add_argument('--output',required=True);q.add_argument('--dry-run',action='store_true')
    sub.add_parser('doctor')
    return p


def check_weights(config,*,conditioning=False):
    required=[Path(config['model']['checkpoint_path'])/'transformer/config.json']
    if config['stage']=='stage3':
        required += [Path(config['model']['checkpoint_path'])/'transformer/diffusion_pytorch_model.safetensors.index.json',
                     Path(config['decoder'])/'weights.safetensors',
                     Path(config['decoder'])/'config.json',Path(config['decoder'])/'SHA256SUMS.json',
                     Path(config['hand_encoder'])/'encoder.safetensors',Path(config['hand_encoder'])/'config.json']
    if conditioning:required += [Path(config['conditioning']['checkpoint_path'])/n for n in ['vae','text_encoder','tokenizer','processor']]
    missing=[str(x) for x in required if not x.exists()]
    if missing:raise FileNotFoundError('Weights are not bundled. See docs/weights.md. Missing:\n'+'\n'.join(missing))


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command=='doctor':
            import shutil
            versions={}
            for name in ['torch','diffusers','transformers','safetensors','flash_attn_3']:
                try:versions[name]=importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:versions[name]=None
            print(json.dumps(dict(packages=versions,ffmpeg=shutil.which('ffmpeg'),ffprobe=shutil.which('ffprobe'),
                gpu_test_executed=False),indent=2));return
        if args.command=='export':
            from .exports import export_weights
            print(json.dumps(export_weights(args.component,args.source,args.output,dry_run=args.dry_run),indent=2));return
        config=load_config(args.config,weights_root=args.weights_root,output_dir=getattr(args,'output',None))
        if args.command=='validate':
            from .data import load_example
            example=load_example(args.example,verify_media=True) if args.example else None
            if example:
                import numpy as np
                from .data import NATIVE_LENGTHS
                camera=np.load(example['camera'],allow_pickle=False);n=NATIVE_LENGTHS[example['frames']]
                if camera['camera_viewmats'].shape!=(n,4,4) or camera['camera_K'].shape!=(n,3,3):
                    raise ValueError('Camera arrays must have one pose/intrinsic per native latent frame')
                if not all(np.isfinite(camera[k]).all() for k in ['camera_viewmats','camera_K']):raise ValueError('Nonfinite camera condition')
            if args.check_weights:check_weights(config)
            print(json.dumps(dict(passed=True,stage=config['stage'],canvas=canvas(config),
                example=example['id'] if example else None,weights_checked=args.check_weights),indent=2));return
        if args.command=='prepare':
            if not args.example:raise ValueError('--example is required')
            check_weights(config,conditioning=True)
            from .prepare import prepare_example
            print(prepare_example(config,args.example,args.output));return
        check_weights(config,conditioning=not args.cache)
        from .data import load_cache
        cache=args.cache
        if not cache:
            if not args.example:raise ValueError('Provide --cache or --example')
            from .prepare import prepare_example
            cache=Path(args.output).parent/'conditioning'
            prepare_example(config,args.example,cache)
        batch=load_cache(cache)
        if batch.variant!=config['stage']:raise ValueError('The conditioning cache belongs to another stage')
        if config['stage']=='stage1':
            if args.accelerated:raise ValueError('--accelerated is a streaming Fast option')
            from .inference.base import BasePipeline
            pipeline=BasePipeline(config)
        elif args.accelerated:
            from .inference.accelerated import AcceleratedFastPipeline
            pipeline=AcceleratedFastPipeline(config)
        else:
            from .inference.fast import FastPipeline
            pipeline=FastPipeline(config)
        frames,latents=pipeline.generate(batch,seed=args.seed)
        if getattr(pipeline,'gates',None):
            gate_path=Path(args.output).with_suffix('.gates.json');gate_path.parent.mkdir(parents=True,exist_ok=True)
            gate_path.write_text(json.dumps(pipeline.gates,indent=2)+'\n')
        if len(frames)!=batch.pixel_frames:raise ValueError('Generated frame count differs from the native source span')
        from .media import write_video
        write_video(frames,args.output,batch.fps)
        output=Path(args.output)
        output.with_suffix('.json').write_text(json.dumps(dict(model='LynnReal-Hand-Base' if config['stage']=='stage1' else 'LynnReal-Hand-Fast',
            frames=len(frames),fps=batch.fps,seed=args.seed,source_id=batch.sample_id),indent=2)+'\n')
        if args.save_latents:
            from safetensors.torch import save_file
            save_file({'latents':latents.cpu().contiguous()},str(output.with_suffix('.safetensors')))
        print(str(output.resolve()))
    except (ValueError,FileNotFoundError) as error:
        parser().exit(2,str(error)+'\n')


if __name__=='__main__':main()
