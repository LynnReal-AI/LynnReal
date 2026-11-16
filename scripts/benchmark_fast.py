"""Benchmark the released accelerated Fast pipeline in one warmed model session."""
from pathlib import Path
import argparse,gc,json,platform,time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/fast.json')
    parser.add_argument('--cache',required=True)
    parser.add_argument('--weights-root',default='checkpoints')
    parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--warmup-rollouts',type=int,default=1)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--cpu-threads',type=int,default=8)
    args=parser.parse_args()
    if args.warmup_rollouts<1 or args.repeats<1:parser.error('Require warmup>=1 and repeats>=1')
    import torch
    from lynnreal_hand.config import load_config
    from lynnreal_hand.data import load_cache
    from lynnreal_hand.inference.accelerated import AcceleratedFastPipeline
    from lynnreal_hand.benchmark import summarize
    from lynnreal_hand.media import write_video
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    assert torch.cuda.device_count()==1, 'Expose exactly one GPU for this benchmark'
    config=load_config(args.config,weights_root=args.weights_root)
    batch=load_cache(args.cache)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();pipeline=AcceleratedFastPipeline(config)
    records=[];frames=[];latents=[];trial=None
    last_chunk=(batch.target_latents.shape[1]+4)//5-1
    def collect(record):
        records.append(record)
        if record['trial']==args.warmup_rollouts-1 and record['chunk']==last_chunk:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    # File flushes (especially on a shared filesystem) can stall the next
    # request and dominate successive-ready FPS. Retain every timing sample,
    # but persist the journal after the measured generator has stopped.
    try:
        for rgb,z in pipeline.benchmark(batch,seed=args.seed,warmup_rollouts=args.warmup_rollouts,
                repeats=args.repeats,on_record=collect):
            if trial!=records[-1]['trial']:frames.clear();latents.clear();trial=records[-1]['trial']
            frames.append(rgb);latents.append(z)
    finally:
        (output/'chunks.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    report=summarize(records)
    report['memory']=dict(
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3,
        scope='all generation chunks after warmup; conditioning preparation and cold loading excluded',
        conditioning_offloaded=True,adaln=pipeline.adaln_memory)
    assert sum(len(frame) for frame in frames)==batch.pixel_frames
    assert torch.isfinite(torch.cat(latents,2)).all()
    write_video(torch.cat(frames),output/'generated.mp4',batch.fps)
    report.update(passed=True,case=batch.sample_id,seed=args.seed,warmup_rollouts=args.warmup_rollouts,
        repeats=args.repeats,frames=batch.pixel_frames,video_fps=batch.fps,
        total_wall_seconds=time.perf_counter()-start,
        journal_io='buffered in memory; persisted after measured generation',
        hardware=dict(gpu=torch.cuda.get_device_name(0),gpu_count=torch.cuda.device_count(),
            torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version(),
            cpu_threads=torch.get_num_threads(),tf32=False,cudnn_benchmark=True),
        warm_scope='loading, Qwen/anchor preparation, compilation/capture, warmup and MP4 encoding excluded',
        runtime_gates=pipeline.gates)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
