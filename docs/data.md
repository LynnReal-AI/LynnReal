# Inference conditions

`examples/01/example.json` defines the input. Paths are relative to that JSON file. Required fields are `id`, `caption`, `image`, `hand_video`, `camera`, `frames` and `fps`. An optional `target_video` supplies reference media for visual evaluation only; future reference RGB is never used as generation history.

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

The public distribution does not include training data loaders, optimizers, losses or recipes.
