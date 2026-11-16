# Fast performance

The accelerated Fast pipeline supports 960 × 540 and native 1344 × 768 RGB output on one NVIDIA H200, with three flow-map transitions, W2 context, a fixed first-frame anchor, guidance 1, the online hand encoder, and Ultra-Fast-VAE.

## Results

| Resolution | Sustained throughput | Mean response | P95 response | Peak allocated memory | Measured full chunks |
| --- | ---: | ---: | ---: | ---: | ---: |
| 960 × 540 | 17.94 FPS | 0.743 s | 0.747 s | 34.51 GiB | 36 |

These measurements use the default model revision listed in [model weights](weights.md), a 243-frame input, seed 42, sampling-specific AdaLN tables, one warmup rollout, and three measured rollouts. The native VAE and Qwen encoder are off device after conditioning preparation; the memory peak covers all generation chunks in the measured rollouts. The reference runtime uses eight CPU threads, TF32 disabled, and cuDNN benchmark enabled. Performance depends on hardware, input shapes, and runtime settings.

The supplied examples illustrate generation separately from this timing input. The commands below reproduce the measurement procedure on the supplied inputs.

## Run the benchmark

Install the package and download the Fast components, then prepare the example and run the benchmark:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=backend:native,expandable_segments:True
lynnreal-hand prepare \
  --config configs/fast.json --weights-root checkpoints \
  --example examples/01/example.json \
  --output outputs/cache/fast/example_01

python scripts/benchmark_fast.py \
  --config configs/fast.json --weights-root checkpoints \
  --cache outputs/cache/fast/example_01 \
  --seed 42 --warmup-rollouts 1 --repeats 3 \
  --output /tmp/lynnreal-benchmark/example_01
```

The first rollout compiles and captures the required shapes. Measured rollouts reuse the model and graphs in the same process. Use a fresh output directory on local storage to limit filesystem stalls.

The command saves `chunks.jsonl`, `summary.json`, and the final generated video. The summary includes the maximum allocated and reserved GPU memory across all generation chunks after warmup, in GiB. Unused allocator cache is released after warmup. Conditioning preparation and cold loading are outside this memory scope; the native VAE and Qwen encoder are absent during Fast generation. Timing records are buffered during generation and written afterward. `total_wall_seconds` includes model initialization, warmup, measured generation, and final video encoding; downloads and the earlier preparation command are separate.

## Native 768p

Use the same Fast weights with `configs/fast_768.json` for native 1344 × 768 generation. Prepare a separate conditioning cache at this resolution:

```bash
lynnreal-hand prepare \
  --config configs/fast_768.json --weights-root checkpoints \
  --example examples/01/example.json \
  --output outputs/cache/fast_768/example_01

python scripts/benchmark_fast.py \
  --config configs/fast_768.json --weights-root checkpoints \
  --cache outputs/cache/fast_768/example_01 \
  --seed 42 --warmup-rollouts 1 --repeats 3 \
  --output /tmp/lynnreal-benchmark/768p_example_01
```

The three-step schedule, W2 context, first-frame anchor, online hand encoding, and timing definitions are the same at both resolutions. The 768p path uses native H3 conditioning and the decoder's 1344 × 768 output.

## Timing definitions

- **Response latency:** elapsed time from hand and camera inputs in host memory to synchronized RGB output in CPU memory. It includes transfer, online hand encoding, hand-condition processing, three DiT transitions, decoding, and RGB delivery.
- **Sustained throughput:** emitted RGB frames divided by successive CPU-ready delivery intervals. These intervals also include the preceding history-cache update, consumer overhead, and host stalls.
- **Measured chunks:** non-warmup rollouts, chunk index at least two, and complete chunks of five latent frames / 17 RGB frames. Initial chunks and shorter final chunks are retained in the journal and excluded from this steady-state statistic.

Preparation, model loading, initial compilation, warmup, and MP4 encoding are excluded from the warmed latency and throughput. The endpoint is CPU RGB availability; capture, pose estimation, rasterization, and display latency are outside the measurement. Playback frame rate is separate from generation throughput.

The standard `infer --accelerated` command includes cold startup and saves its MP4 after generation. Use this benchmark entry point when measuring warmed performance.
