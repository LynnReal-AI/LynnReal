# Input and output verification

## Check inputs

Validate the manifest, hand video, and camera arrays before preparing a cache:

```bash
lynnreal-hand validate --config configs/base.json --example examples/01/example.json
lynnreal-hand validate --config configs/fast.json --example examples/01/example.json
lynnreal-hand validate --config configs/fast_768.json --example examples/01/example.json
```

These commands check configuration, file presence, supported frame counts, video frame rate, and camera-array shapes. They run without model weights or a GPU. Add `--weights-root checkpoints --check-weights` after downloading the model components to check their entry files. Tensor compatibility is checked during model loading.

## Check generated video

Inspect the output dimensions, frame rate, and decoded frame count, then decode the full video:

```bash
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=width,height,avg_frame_rate,nb_read_frames \
  -of json outputs/fast/example_01.mp4
ffmpeg -v error -i outputs/fast/example_01.mp4 -f null -
```

Output frame count and playback frame rate should match the input manifest. Base outputs 1344 × 768 by default; Fast outputs 960 × 540 with `configs/fast.json` or native 1344 × 768 with `configs/fast_768.json`. Add `--save-latents` during inference to retain the latent tensors for inspection.

## Runtime checks

Model initialization checks the required FlashAttention-3 interface. The accelerated pipeline also checks new compiled DiT shapes and CUDA graph replay against fused eager execution, and checks the compiled hand encoder and decoder outputs.

W8A8 quantization changes numerical precision. Compilation and graph checks compare execution of the same quantized model.

For environment setup, see [installation](installation.md). For latency and throughput measurements, see [benchmarks](benchmark.md).
