# Inference examples

Seven examples demonstrate hand-controlled interactions. Each directory includes the complete input bundle and Base/Fast reference videos.

The featured example is [Pet interaction](05/example.json). [Watch Fast](05/fast.mp4) · [Watch Base](05/base.mp4).

| Directory | Scene | Frames | Frame rate | Camera entries |
| --- | --- | ---: | ---: | ---: |
| [01](01/example.json) | Bottle handling | 124 | 24 fps | 37 |
| [02](02/example.json) | Preparing corn | 124 | 24 fps | 37 |
| [03](03/example.json) | Preparing ingredients | 124 | 24 fps | 37 |
| [04](04/example.json) | Opening a carton | 124 | 30 fps | 37 |
| [05](05/example.json) | Pet interaction | 243 | 24 fps | 72 |
| [06](06/example.json) | Clay modeling | 243 | 24 fps | 72 |
| [07](07/example.json) | Wooden puzzle | 243 | 24 fps | 72 |

Use `05`, the featured pet-interaction scene, for the quick start. `image.png` provides the initial image; `prompt.txt` and `example.json` contain the text condition. `hand.mp4` and `camera.npz` provide aligned hand motion and camera poses. The optional `reference.mp4` supports visual comparison.

To use your own inputs, replace the media and update `example.json` together. Keep the initial image, hand motion, and camera poses aligned in field of view and frame order. Set `frames` and `fps` to match the hand video, and align the camera arrays to the native latent-frame clock. See [data preparation](../docs/data.md) for the format and camera conversion.

Prepare separate Base and Fast caches with `lynnreal-hand prepare`, then run the corresponding inference command from the [quick start](../README.md#quick-start).

## Reference results

| Example | Base | Fast |
| --- | --- | --- |
| 01 | [Video](01/base.mp4) | [Video](01/fast.mp4) |
| 02 | [Video](02/base.mp4) | [Video](02/fast.mp4) |
| 03 | [Video](03/base.mp4) | [Video](03/fast.mp4) |
| 04 | [Video](04/base.mp4) | [Video](04/fast.mp4) |
| 05 | [Video](05/base.mp4) | [Video](05/fast.mp4) |
| 06 | [Video](06/base.mp4) | [Video](06/fast.mp4) |
| 07 | [Video](07/base.mp4) | [Video](07/fast.mp4) |

The reference videos use 1344 × 768 output. The released Fast quick-start configuration generates 960 × 540 output; `--accelerated` enables its optimized inference path. Output dimensions and frame counts for the supplied reference videos are listed in each directory's `results.json`.
