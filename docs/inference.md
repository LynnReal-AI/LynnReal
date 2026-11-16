# Inference

LynnReal-Hand accepts a first image, caption, aligned hand-condition video, and camera poses. Base and Fast use different conditioning caches. All commands below run from the repository root after installing the required model components.

## Choose a model

| Setting | Base | Fast |
| --- | --- | --- |
| Config | `configs/base.json` | `configs/fast.json` |
| Generation | Bidirectional full sequence | Causal, chunk by chunk |
| Sampling | 30 schedule points; shift 3; CFG 3 | 3 flow-map transitions; shift 3; guidance 1 |
| Context | Full input span | W2: current chunk plus preceding generated chunk |
| Default output | 1344 × 768 | 960 × 540 |
| Model weights | Complete Base | Complete Fast transformer |
| Hand encoding | Original native encoder | Compact hand encoder |
| Decoding | Original native decoder | Ultra-Fast-VAE |

Both paths use the original first-image anchor and Qwen first-image/text conditions. The compact hand encoder never replaces the first-image anchor encoder. Fast loads a sampling-specialized export of the 33B transformer with streaming time conditioning and does not require a separate Base download.

## Inputs and native spans

The example schema is [example.json](../examples/01/example.json). Paths resolve relative to that file. The supported packaged spans are:

| RGB frames | Latent frames |
| ---: | ---: |
| 124 | 37 |
| 158 | 47 |
| 243 | 72 |
| 260 | 77 |

The source frame rate is preserved. Each full causal chunk contains five latent frames and produces 17 RGB frames; the native tail is kept. Spatial preparation uses the same field of view for image, HAND, and camera. The default Fast recipe works on a 960 × 544 latent-compatible canvas and crops two decoded rows from the top and bottom to deliver 960 × 540. `configs/fast_768.json` uses native 1344 × 768 encoding, generation, and decoding, with the same three-step schedule, W2 context, and fixed first-frame anchor. Prepare a separate cache for each resolution. Base preparation uses 1344 × 768.

Use one of the supported native spans, preserving the source frames and their order.

Inference uses the first image, caption, hands, and camera conditions. A paired reference video can support visual evaluation, but its future RGB frames do not become generation history. See [data](data.md) for camera conversion and cache tensors.

## Fast: reference sampler

```bash
lynnreal-hand prepare \
  --config configs/fast.json \
  --example examples/01/example.json \
  --output outputs/cache/fast/example_01

lynnreal-hand infer \
  --config configs/fast.json \
  --cache outputs/cache/fast/example_01 \
  --output outputs/fast/example_01.mp4 \
  --seed 42
```

Use `--weights-root /path/to/checkpoints` on both commands when the weights are outside `checkpoints/`. Prepare separate caches for Base and Fast.

The Python `FastPipeline.stream` interface yields a chunk's decoded frames and latents before committing its clean history. Advancing the iterator performs that commit before processing the next chunk. Generated history is retained with sliding local positional encoding; physical camera indices remain aligned to the input trajectory.

## Fast: accelerated sampler

```bash
lynnreal-hand infer \
  --config configs/fast.json \
  --cache outputs/cache/fast/example_01 \
  --output outputs/fast_accelerated/example_01.mp4 \
  --seed 42 --accelerated --save-latents
```

The accelerated path combines packed KV storage, W8A8 projections, fused operators, all-block DiT compilation, CUDA graphs, optimized hand encoding, and compiled Ultra-Fast-VAE decoding. It preserves the three-step/W2 sampling interface. Its online hand encoder also needs the original hand video, so keep cache metadata and source media in their expected relative locations.

## Base

```bash
lynnreal-hand prepare \
  --config configs/base.json \
  --example examples/01/example.json \
  --output outputs/cache/base/example_01

lynnreal-hand infer \
  --config configs/base.json \
  --cache outputs/cache/base/example_01 \
  --output outputs/base/example_01.mp4 \
  --seed 42
```

The `--accelerated` option is specific to Fast. Base uses full-sequence bidirectional sampling.

## Output files

For an output path `outputs/fast/example_01.mp4`, the CLI writes:

- `example_01.mp4`: generated video, with the input's native frame count and frame rate.
- `example_01.json`: model label, source ID, frame count, frame rate, and seed.
- `example_01.safetensors`: generated latents, when `--save-latents` is set.
- `example_01.gates.json`: runtime numerical-gate results, when the pipeline reports gates.

The CLI saves video output. For visual evaluation, compare against aligned reference media. See [output verification](VALIDATION.md) to inspect frame counts, dimensions, frame rate, and decoding.

## Acceleration and timing

First use includes compilation and CUDA graph capture. New compiled shapes undergo numerical checks against fused eager execution. W8A8 quantization changes precision relative to full-BF16 execution.

On one H200 at 540p, three steps, and W2, accelerated Fast measured **17.94 FPS** sustained throughput and **0.743 s** mean response on a 243-frame input. Response ends when synchronized RGB output is available in CPU memory. Throughput also includes history-cache updates between deliveries.

Use the [benchmark command](benchmark.md) to retain the model and graphs across warmup and measured rollouts. See [output verification](VALIDATION.md) for generated-media checks.

## AdaLN memory

AdaLN modulation is determined by time conditions. Once the sampling schedule is fixed, Base needs only its visited timesteps and the conditioning boundary; Fast needs only its visited start/end-time pairs and the prefix and clean-history boundaries.

The published packages store these modulation outputs in `transformer/adaln.safetensors`. For the default schedules, table payloads are 535.3 MiB for Base and 55.4 MiB for Fast, replacing about 24.2 GiB of learned block projections in each model. The loader constructs the compact table modules before reading weights, so the learned block projections are never loaded. Scene inputs and CFG reuse the same time-dependent values. Keep the packaged sampling schedule: changing step count or shift requires a new export and is rejected explicitly.

Conditioning preparation and generation are separate commands. Fast generation loads neither the native H3 VAE nor the Qwen encoder; they can be released after preparing the conditioning cache. Base still needs the native VAE for final video decoding. Total inference memory includes activations, KV caches, graph buffers, and the output decoder as well as transformer weights. Warm generation peaks exclude conditioning preparation and cold loading.

For a full training-format checkpoint, `sampling.precompute_adaln: true` provides an optional runtime specialization after loading the full model. To create a compact package ahead of loading, use `python scripts/export_compact.py --source /path/to/full-package --config configs/fast.json --output /path/to/compact-package`. The exporter writes `transformer/`; retain the matching conditioning components, hand encoder, and decoder when assembling an inference package. Export on the intended execution device and validate generated outputs before deployment. Packaged tables are selected automatically regardless of `precompute_adaln`.
