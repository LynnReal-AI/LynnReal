# Installation

## Reference environment

| Component | Reference configuration |
| --- | --- |
| Operating system | Linux |
| Python | 3.10 |
| GPU | NVIDIA H200; one GPU per inference, evaluation, or timing task |
| Reference host allocation | 20 CPUs and 200 GiB RAM |
| PyTorch | 2.9.1 + CUDA 12.8 |
| Torchvision | 0.24.1 |
| Transformers | 5.14.1 |
| Diffusers | 0.40.0 |
| Attention | FlashAttention-3 Hopper, exposing `flash_attn_interface` |
| Media tools | FFmpeg and ffprobe |

Use the declared Transformers version for the H3 Qwen conditioner's multimodal token-type API.

## Set up Python dependencies

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
```

Install the package and build the required H3 FlashAttention-3 kernels from pinned public upstream source. The build requires Git, a C++ compiler and a CUDA toolkit with `nvcc` (CUDA 12.3 or newer):

```bash
pip install -e .
bash scripts/install_fa3.sh
```

The repository includes the 13 H3-specific Diffusers modules and activates them without editing site-packages. The package pins Diffusers 0.40.0 for those APIs. Install the Hopper FA3 build below; the standard FlashAttention package does not provide the required FA3 interface.

Install FFmpeg and ffprobe with your system package manager. Complete model downloads are listed in [weights](weights.md). The official conditioning components are included in both packages; the matching hand encoder and Ultra-Fast-VAE are included in Fast.

Each complete Base or Fast package occupies approximately 120 GB. Each package includes its own official conditioning components. Allow additional space for Python dependencies, FA3 compilation and temporary downloads. Fast can be used independently of Base.

## Inspect the environment

```bash
lynnreal-hand doctor
```

`doctor` reports installed package versions and media-tool paths. It does not run CUDA or validate model outputs. You can separately check whether the FA3 entry point is importable:

```bash
python -c 'from flash_attn_interface import flash_attn_func; print("FA3 interface import passed")'
```

Model startup requires `_flash_3` and runs forward/backward FlashAttention-3 checks before generation.

## Validate the example without weights

```bash
lynnreal-hand validate --config configs/base.json --example examples/01/example.json
lynnreal-hand validate --config configs/fast.json --example examples/01/example.json
```

These checks validate configuration, native frame counts, media frame rate, camera-array geometry, and input presence. They do not test generation. After installing weights, add `--check-weights` to check required local files. That option checks file presence; model loading performs additional tensor-compatibility checks.

See [input and output verification](VALIDATION.md) for media checks and [inference](inference.md) for generation commands.

## Pinned FA3 build

`scripts/install_fa3.sh` fetches the pinned official FlashAttention source and its Cutlass submodule. It enables BF16/FP16 and head dimensions 64/128: the H3 DiT uses BF16/128 and the Ultra-Fast-VAE uses FP16/64. Both are required for Fast. Other head sizes and FP8 are omitted. The build uses four compiler workers by default. Set `CUDA_HOME` if the toolkit is outside `/usr/local/cuda`; set `MAX_JOBS` to reduce compiler memory use. Downloaded sources and wheels live in `~/.cache/lynnreal-hand/fa3`, or `LYNNREAL_FA3_BUILD_DIR` if provided. Building does not require loading model weights. When CUDA is visible, the helper smoke-tests both required kernel variants; otherwise it reports that the GPU checks were skipped.
