#!/usr/bin/env bash
# Build the DiT BF16/128 and Ultra-Fast-VAE FP16/64 Hopper kernels.
set -euo pipefail

LYNNREAL_FA3_COMMIT=e9cf2c1651d2303191eb40a739a3c135fda00999
LYNNREAL_FA3_BUILD_DIR=${LYNNREAL_FA3_BUILD_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/lynnreal-hand/fa3}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
command -v nvcc >/dev/null || { echo 'CUDA toolkit with nvcc is required; set CUDA_HOME.' >&2; exit 1; }
python -c 'import torch; assert torch.__version__.split("+")[0] == "2.9.1", "Install PyTorch 2.9.1 first"'
python -m pip install 'setuptools>=77' wheel packaging ninja

mkdir -p "$LYNNREAL_FA3_BUILD_DIR"
if [ ! -d "$LYNNREAL_FA3_BUILD_DIR/source/.git" ]; then
    git init --quiet "$LYNNREAL_FA3_BUILD_DIR/source"
    git -C "$LYNNREAL_FA3_BUILD_DIR/source" remote add origin https://github.com/Dao-AILab/flash-attention.git
fi
git -C "$LYNNREAL_FA3_BUILD_DIR/source" fetch --depth 1 origin "$LYNNREAL_FA3_COMMIT"
git -C "$LYNNREAL_FA3_BUILD_DIR/source" checkout --detach "$LYNNREAL_FA3_COMMIT"
git -C "$LYNNREAL_FA3_BUILD_DIR/source" submodule update --init --depth 1 csrc/cutlass

export MAX_JOBS=${MAX_JOBS:-4}
export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE
export FLASH_ATTENTION_DISABLE_SPLIT=TRUE FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE FLASH_ATTENTION_DISABLE_LOCAL=TRUE
export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE FLASH_ATTENTION_DISABLE_PACKGQA=TRUE
unset FLASH_ATTENTION_DISABLE_FP16 FLASH_ATTENTION_DISABLE_HDIM64
export FLASH_ATTENTION_DISABLE_FP8=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE
export FLASH_ATTENTION_DISABLE_HDIM192=TRUE FLASH_ATTENTION_DISABLE_HDIM256=TRUE
export FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
python -m pip wheel --no-build-isolation --no-deps --no-cache-dir \
    --wheel-dir "$LYNNREAL_FA3_BUILD_DIR/wheels" "$LYNNREAL_FA3_BUILD_DIR/source/hopper"
python -m pip install --force-reinstall --no-deps "$LYNNREAL_FA3_BUILD_DIR"/wheels/flash_attn_3-3.0.0-*.whl
python - <<'PY'
import torch
from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
if torch.cuda.is_available():
    for dtype, head_dim in [(torch.bfloat16, 128), (torch.float16, 64)]:
        values = [torch.randn(1, 17, 2, head_dim, device='cuda', dtype=dtype) for _ in range(3)]
        output = flash_attn_func(*values, causal=False, deterministic=True)
        assert output.shape == values[0].shape and torch.isfinite(output).all()
    print('FA3 BF16/128 and FP16/64 GPU smoke checks passed; model numerical gates still apply.')
else:
    print('FA3 import passed; GPU kernel checks were skipped because CUDA is unavailable.')
PY
