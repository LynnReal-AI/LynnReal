# Inference conditions

`examples/01/example.json` defines the input. Paths are relative to that JSON file. Required fields are `id`, `caption`, `image`, `hand_video`, `camera`, `frames` and `fps`. An optional `target_video` supplies reference media for visual evaluation only; future reference RGB is never used as generation history.

## HandWorldBench controls

The benchmark distributes camera-space 21-joint tracks, camera calibration and original hand-surface videos. Its human and robot surface renderings use different geometries. To evaluate LynnReal-Hand, fit both embodiments to MANO, then render the fitted hands in the model's training colors. The converter uses the wrist anchor for human tracks and the mean of the four finger knuckles for robot tracks. It projects each original frame with its lens distortion and preserves the benchmark frame rate.

Install the project environment and FFmpeg. For conversion, install the optional fitting and rendering dependencies with the same CUDA-enabled PyTorch environment. Obtain `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` under the [official MANO license](https://mano.is.tue.mpg.de/), and point `--mano-dir` to the directory containing them.

```bash
python -m pip install smplx chumpy opencv-python-headless --no-build-isolation
python -m pip install 'git+https://github.com/NVlabs/nvdiffrast.git' --no-build-isolation
python scripts/convert_handworldbench_mano.py \
  --dataset /path/to/HandWorldBench --case-id hw500_0001 --check-only
python scripts/convert_handworldbench_mano.py \
  --dataset /path/to/HandWorldBench --case-id hw500_0001 \
  --mano-dir /path/to/mano/models --output hand.mp4
```

Before inference, compare the hand location and motion to the source frames. Check for a black background, a red right hand, a blue left hand and the same camera field of view and timeline as the RGB. Set `caption` to the training-format prompt with its first-frame reference, shot description and audio fields. For the benchmark, the exact prompt is stored in `model_prompt` in each results manifest. The converted MP4 follows the source length, while the general-purpose inference CLI accepts the specific clip lengths listed below.

Download the reference inputs, original and adapted controls, and available predictions from the [Fast results](https://huggingface.co/shaohao011/LynnReal-Hand-Fast-preview/tree/main/HandWBench_Results) or [Base results](https://huggingface.co/shaohao011/LynnReal-Hand-Base-1.0/tree/main/HandWBench_Results).

## Camera alignment

Runtime `camera.npz` contains `camera_viewmats[L,4,4]` and `camera_K[L,3,3]`. Views are first-frame-relative world-to-camera poses. Intrinsics retain the fixed normalized H3 convention. Attention applies its pose transform later; do not apply it in advance.

If your source uses absolute camera-to-world poses for every RGB frame, store them in `raw_camera.npz` under `camera_c2w` (or `pose`) with shape `[frames,4,4]`, then convert once. Already latent-aligned runtime camera files can be used directly.

```bash
python scripts/convert_camera.py --input raw_camera.npz --output camera.npz --frames 243
```

Native spans are 124/37, 158/47, 243/72 and 260/77 RGB/latent frames. Camera conversion selects physical frame offsets `[0,1,5,9,13]` in each 17-frame group and `[0,1]` in a five-frame tail. Preserve source frame order, translation units and native frame rate.

## Conditioning cache

`lynnreal-hand prepare` writes `conditions.safetensors` and `sample.json`. Base and Fast caches are distinct.

| Tensor | Shape | Role |
| --- | --- | --- |
| `prompt_embeds` | `[tokens,5120]` | Joint first-image and text condition |
| `text_token_tags` | `[tokens]` | Image/text row tags |
| `anchor_latents` | `[24,1,H/16,W/16]` | Native first-frame latent |
| `hand_latents` | `[24,L,H/16,W/16]` | Aligned hand conditions |
| `camera_viewmats` | `[L,4,4]` | Physical camera views |
| `camera_K` | `[L,3,3]` | Normalized camera intrinsics |

Metadata records the sample ID, caption, native frame count/rate, variant, relative source hand-video path and latent geometry. `target_shape` describes that geometry; it does not provide future target content. If a legacy cache contains paired future tensors, the inference loader discards them and creates shape-only placeholders.

## HandiVid training data

**[HandiVid v1 (H3)](https://huggingface.co/datasets/shaohao011/handivid)** provides pre-encoded training data, including video and hand-motion latents, first-frame anchors, image/text conditioning features, frame alignment, and available camera arrays.

Training tensors are distributed in TAR shards with Safetensors/JSON samples and a Parquet index. Use the dataset's [format guide and reference PyTorch reader](https://huggingface.co/datasets/shaohao011/handivid/blob/main/v1/h3/README.md) to consume these training samples. Their schema is documented separately from the inference conditioning caches above.

The GitHub repository provides inference tooling; training optimizers, losses, and training recipes are not included here. Dataset licensing and source attribution are documented in the [HandiVid license](https://huggingface.co/datasets/shaohao011/handivid/blob/main/LICENSE.md) and [notices](https://huggingface.co/datasets/shaohao011/handivid/blob/main/NOTICE.md).
