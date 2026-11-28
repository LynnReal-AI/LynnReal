# Model packages

Base and Fast are complete inference packages. Both include the original VAE, Qwen image/text encoder, tokenizer, vocabulary, processor, and scheduler from the official MiniMax H3 release. Fast additionally includes its compact hand encoder and Ultra-Fast-VAE decoder.

| Model | Repository | Local directory |
| --- | --- | --- |
| LynnReal-Hand-Base | [Hugging Face](https://huggingface.co/shaohao011/LynnReal-Hand-Base-1.0) | `checkpoints/base` |
| LynnReal-Hand-Fast | [Hugging Face](https://huggingface.co/shaohao011/LynnReal-Hand-Fast-preview) | `checkpoints/fast` |
| Ultra-Fast-VAE, standalone decoder | [Hugging Face](https://huggingface.co/shaohao011/LynnReal-ultra-fast-vae-1.0) | `checkpoints/vae` |

## Download

Run the command for your chosen model from the code repository:

```bash
# Base
python scripts/download_weights.py --component base

# Fast, including its hand encoder and decoder
python scripts/download_weights.py --component fast
```

Fast can be used independently of Base. The standalone decoder download is available for other integrations:

```bash
python scripts/download_weights.py --component vae
```

Add `--weights-root /path/to/checkpoints` to select another directory, and use that same root in preparation and inference commands. Add `--dry-run` to inspect the download plan. Downloads do not require login.

## Package layout

```text
base/
  config.json
  model_index.json
  transformer/
  vae/
  text_encoder/
  tokenizer/
  processor/
  scheduler/

fast/
  config.json
  model_index.json
  transformer/
  vae/
  text_encoder/
  tokenizer/
  processor/
  scheduler/
  hand_encoder/
  decoder/
```

`vae/` is the original H3 VAE used for native encoding and Base decoding. Fast's `decoder/` contains Ultra-Fast-VAE for video output. The tokenizer and processor directories include their vocabulary, merge rules, and templates.

The transformer retains the backbone and time-conditioning parameters and stores only the AdaLN modulation tables needed by its packaged sampling schedule. The learned block projection matrices are omitted. Base uses 30-step sampling for a single span or each sliding window. Fast uses three sampling steps. The tokenizer vocabulary is included in full and is independent of these AdaLN tables. See [AdaLN memory](inference.md#adaln-memory).

## Configuration

The root `model_index.json` identifies the LynnReal-Hand pipeline and points to its `config.json`. The nested `transformer/config.json` describes the transformer architecture. Either root entry point can be passed to validation, preparation, and inference:

```bash
lynnreal-hand validate --config checkpoints/fast/model_index.json --weights-root checkpoints
```

Downloaded package configurations resolve their components relative to the package directory. The checked-in `configs/base.json` and `configs/fast.json` also recognize complete local model packages. Existing installations using shared conditioner and decoder directories remain supported.

## Integrity and licensing

The downloader selects compatible component versions. Model loading checks tensor compatibility, and decoder packages include file integrity checks. Native H3 and Qwen components retain their official terms; see each model package's LICENSE and NOTICE.
