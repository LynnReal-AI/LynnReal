<div align="center">

# LynnReal-Hand

### Unlocking World Interaction with Hand Control

[English](README.md) · [简体中文](README_zh-CN.md)

[LynnReal-Hand (arXiv)](https://arxiv.org/pdf/2611.21479) · [HandiVid (arXiv)](https://arxiv.org/abs/2611.29693) · [Hugging Face Paper](https://huggingface.co/papers/2611.21479) · [YouTube](https://youtu.be/OdywWF0srpU)

[演示](#演示) · [模型](#模型) · [数据集](#数据集) · [快速开始](#快速开始) · [文档](#文档) · [许可证](LICENSE)

</div>

![LynnReal-Hand：通过手部控制探索日常与想象中的世界](assets/teaser-v2.jpg)

**LynnReal-Hand** 是一个由手部动作控制的视频世界模型。模型接收初始图像、文本描述、手部动作和相机位姿，生成日常与想象场景中的交互视频。

- **精细手物交互。** 通过关节化手部动作与相机轨迹，控制不同场景中的伸手、抓取、释放和双手协同交互。
- **交互保持训练（IPT）。** 基于 100K 交互片段，通过手部感知的多层次监督强化小尺度手部及接触区域，并在因果蒸馏中保持控制精度。
- **三步因果生成。** LynnReal-Hand-Base 生成完整序列；LynnReal-Hand-Fast 学习因果流映射，并在自生成历史上进一步优化。固定首帧锚点、有界滑动上下文和局部时间位置重置支持流式生成，避免注意力缓存持续增长。
- **Ultra-Fast-VAE 加速流式解码。** 紧凑解码器保留原生 H3 潜变量接口，通过蒸馏恢复空间与时间重建质量。
- **低延迟流式推理。** 优化后的 Fast 在单张 H200 上达到 **540p、18.03 FPS**，**预热后每块平均响应 741 ms**。采用按采样日程精简的 AdaLN 表并卸载原生 VAE/Qwen 编码器后，独立测得**预热后峰值已分配显存为 34.89 GiB**；测量范围见[性能](#性能)。

本仓库提供推理代码、模型下载和示例输入。预编码训练数据见 [HandiVid](#数据集)。

<a id="official-demo"></a>

## 演示

https://github.com/user-attachments/assets/3df7244c-e001-490f-8743-9b556942767b

演示涵盖自然环境、日常活动与想象场景中的手部交互。[YouTube 观看](https://youtu.be/OdywWF0srpU) · [下载演示视频](assets/demo.mp4)。

## 模型

| 模型 | 说明 | 默认输出 | 下载 |
| --- | --- | --- | --- |
| **LynnReal-Hand-Base** | 33B 主干，双向完整序列生成 | 1344 × 768 | [Hugging Face](https://huggingface.co/shaohao011/LynnReal-Hand-Base-1.0) |
| **LynnReal-Hand-Fast** | 33B 因果主干，三步采样、W2 上下文、首帧锚点 | 960 × 540 | [Hugging Face](https://huggingface.co/shaohao011/LynnReal-Hand-Fast-preview) |
| **Ultra-Fast-VAE** | LynnReal-Hand-Fast 使用的视频解码器 | Fast 流程中为 960 × 540 | [Hugging Face](https://huggingface.co/shaohao011/LynnReal-ultra-fast-vae-1.0) |

Base 和 Fast 均采用按采样日程精简的 Transformer 推理权重；Fast 还包含配套手部编码器。两个生成模型包均包含官方 MiniMax H3 VAE、Qwen 图像与文本编码器、词表和 processor；Fast 同时包含 Ultra-Fast-VAE。组件目录见[模型权重说明](docs/weights.md)。

## 数据集

**[HandiVid](https://huggingface.co/datasets/shaohao011/handivid)** 是 LynnReal-Hand 的预编码训练数据集。**v1 为 H3 编码版本**，已在 Hugging Face 上发布。

数据集的构建与标注流程见论文 [*HandiVid: A High-Quality Dataset for Hand-Driven Interactive Video Generation*](https://arxiv.org/abs/2611.29693)。

样本包含对齐的视频与手部潜变量、首帧锚点、图像/文本条件特征、帧对齐信息，以及可用的相机数组。数据以 TAR 分片发布，单个样本包含 Safetensors 和 JSON 记录，并提供 Parquet 索引及参考 PyTorch 读取器。

[数据集与下载](https://huggingface.co/datasets/shaohao011/handivid) · [格式与数据读取](https://huggingface.co/datasets/shaohao011/handivid/blob/main/v1/h3/README.md) · [许可与致谢](https://huggingface.co/datasets/shaohao011/handivid/blob/main/NOTICE.md)

Wan 2.2-5B 编码计划作为独立版本提供。

## 快速开始

### 安装

参考环境为 **Linux、Python 3.10、PyTorch 2.9.1 / CUDA 12.8，以及单张 NVIDIA H200**。请通过系统包管理器安装 FFmpeg 和 ffprobe；编译 FlashAttention-3 需要 C++ 编译器及包含 `nvcc` 的 CUDA toolkit。

```bash
git clone https://github.com/LynnReal-AI/LynnReal.git
cd LynnReal

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
bash scripts/install_fa3.sh
lynnreal-hand doctor

export CUDA_VISIBLE_DEVICES=0
```

后续命令均在仓库根目录运行。依赖版本和 FlashAttention-3 编译要求见[安装指南](docs/installation.md)。

### 模型下载

下载所选模型的完整模型包：

```bash
# LynnReal-Hand-Base
python scripts/download_weights.py --component base

# LynnReal-Hand-Fast
python scripts/download_weights.py --component fast
```

Fast 可独立于 Base 使用。如需更换权重目录，在下载命令中添加 `--weights-root /path/to/checkpoints`，并在后续准备和推理命令中使用同一路径。添加 `--dry-run` 可查看下载计划。

### 示例输入

七个示例均包含对齐的初始图像、手部动作视频、相机位姿、文本描述及 Base/Fast 参考结果：

| 示例 | 场景 | 输入 | Base | Fast |
| --- | --- | --- | --- | --- |
| 01 | 瓶体抓取与放置 | [输入](examples/01/example.json) | [视频](examples/01/base.mp4) | [视频](examples/01/fast.mp4) |
| 02 | 玉米备餐 | [输入](examples/02/example.json) | [视频](examples/02/base.mp4) | [视频](examples/02/fast.mp4) |
| 03 | 食材处理 | [输入](examples/03/example.json) | [视频](examples/03/base.mp4) | [视频](examples/03/fast.mp4) |
| 04 | 打开纸盒盖 | [输入](examples/04/example.json) | [视频](examples/04/base.mp4) | [视频](examples/04/fast.mp4) |
| 05 | 宠物互动 | [输入](examples/05/example.json) | [视频](examples/05/base.mp4) | [视频](examples/05/fast.mp4) |
| 06 | 黏土塑形 | [输入](examples/06/example.json) | [视频](examples/06/base.mp4) | [视频](examples/06/fast.mp4) |
| 07 | 木制解谜玩具 | [输入](examples/07/example.json) | [视频](examples/07/base.mp4) | [视频](examples/07/fast.mp4) |

参考结果的分辨率为 1344 × 768，Fast 快速开始默认输出 960 × 540。帧率和文件布局见[示例说明](examples/README.md)。生成前先检查输入：

```bash
lynnreal-hand validate --config configs/base.json --example examples/05/example.json
lynnreal-hand validate --config configs/fast.json --example examples/05/example.json
```

输入检查无需模型权重或 GPU。下载权重后，可添加 `--weights-root checkpoints --check-weights` 检查模型包入口文件。

<a id="base-inference"></a>

### LynnReal-Hand-Base

Base 使用 **1344 × 768** 的完整序列双向采样，默认 30 个采样调度点、CFG 3。先准备条件缓存，再生成视频：

```bash
lynnreal-hand prepare \
  --config configs/base.json --weights-root checkpoints \
  --example examples/05/example.json \
  --output outputs/cache/base/example_05

lynnreal-hand infer \
  --config configs/base.json --weights-root checkpoints \
  --cache outputs/cache/base/example_05 \
  --output outputs/base/example_05.mp4 \
  --seed 42
```

<a id="fast-inference"></a>

### LynnReal-Hand-Fast

Fast 使用 **960 × 540** 输出、三次流映射和 W2 上下文，即当前块及前一个已生成块，并始终保留首帧作为锚点。每个完整块输出 17 帧 RGB。

```bash
lynnreal-hand prepare \
  --config configs/fast.json --weights-root checkpoints \
  --example examples/05/example.json \
  --output outputs/cache/fast/example_05

lynnreal-hand infer \
  --config configs/fast.json --weights-root checkpoints \
  --cache outputs/cache/fast/example_05 \
  --output outputs/fast/example_05.mp4 \
  --seed 42
```

Base 和 Fast 需要分别准备条件缓存。使用加速 Fast 时，请保留原始手部视频，以供在线手部编码读取。

Fast 原生 **1344 × 768** 生成请在条件准备和推理时均使用 [`configs/fast_768.json`](configs/fast_768.json)。每种分辨率需要独立的条件缓存；命令见 [768p 测速说明](docs/benchmark.md#native-768p)。

<a id="accelerated-fast-inference"></a>

### 加速 Fast

准备好 Fast 缓存后，添加 `--accelerated` 启用单张 H200 加速路径：

```bash
lynnreal-hand infer \
  --config configs/fast.json --weights-root checkpoints \
  --cache outputs/cache/fast/example_05 \
  --output outputs/fast_accelerated/example_05.mp4 \
  --seed 42 --accelerated
```

首次运行包含编译和 CUDA Graph 捕获；同一进程中的后续生成可复用已编译的形状。稳态吞吐与响应时间的测量方式见[性能指南](docs/benchmark.md)。

<a id="inference-outputs"></a>

CLI 保存生成的 MP4，以及记录模型、帧数、帧率、随机种子和示例 ID 的 JSON 文件。添加 `--save-latents` 可保存生成 latent。流式 Python 接口和输出格式见[推理指南](docs/inference.md)。

**重点示例：宠物互动**

[![宠物互动：Fast 生成预览](examples/05/preview.gif)](examples/05/fast.mp4)

[Fast 视频](examples/05/fast.mp4) · [Base 视频](examples/05/base.mp4) · [示例输入](examples/05/example.json)

<a id="custom-inputs"></a>

## 自定义输入

测试 Base 或 Fast 前，请先对照仓库示例检查 prompt 和 MANO 控制视频。prompt 应按训练时的格式引用首帧并描述动作。对照首帧播放控制视频，确认手部位置、左右手和帧序正确，黑底上右手为红色、左手为蓝色。[HandWorldBench 转换指南](docs/data.md#handworldbench-controls) 提供转换脚本和命令。

将初始图像、手部动作视频和相机数组与 `example.json` 放在同一目录：

```json
{
  "id": "my_example",
  "image": "image.png",
  "hand_video": "hand.mp4",
  "camera": "camera.npz",
  "caption": "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description: [Shot 1] The target video continues from the scene shown in <Picture 1>. A hand picks up the cup from the table.\n\noverall_soundscape: N/A\n\nnon_diegetic_music: N/A",
  "frames": 124,
  "fps": 30.0
}
```

文件路径相对于 `example.json` 解析。文本描述、帧数和帧率应与实际输入一致。当前输入流程支持 **124、158、243 或 260 帧**，并保留输入帧率。

在 `validate` 和 `prepare` 命令中替换为自己的示例路径，再将生成的缓存传给 `infer`。图像、手部动作和相机位姿须保持相同视野及帧序。相机对齐和输入要求见[示例说明](examples/README.md)与[数据准备指南](docs/data.md)。

## 性能

### 视频生成

以下为论文中单张 H200 的结果。Fast 报告 540p 下热启动后的平均响应时间，其余方法报告 768p 下的延迟中位数；各方法的分辨率、片段长度和计时单位有所不同。

| 模型 | 主干参数 | 分辨率 | 计时单位 | 延迟（秒）↓ | FPS ↑ | 峰值显存（GiB） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| JointControlVideo | 14B | 768p | 完整序列 | 1320.324 | 0.053 | 59.50 |
| Generated Reality | 5B | 768p | 完整序列 | 446.484 | 0.221 | 15.87 |
| Hand2World-Bi | 5B | 768p | 完整序列 | 104.682 | 1.195 | 59.67 |
| Hand2World-AR | 5B | 768p | 4 帧 | 0.437 | 8.646 | 33.44 |
| **LynnReal-Hand-Base** | 33B | 768p | 完整序列 | 770.155 | 0.136 | 82.59 |
| **LynnReal-Hand-Fast** | 33B | 540p | 17 帧 | 0.741 | 18.03 | 34.89 |

Fast 的响应时间包含在线手部编码、生成、解码和传输至 CPU，持续 FPS 还包含历史缓存更新。其显存来自使用精简 AdaLN 表、原生 VAE 与 Qwen 编码器移出显存后的独立热态测量，不计入条件准备和冷启动。Base 的显存沿用完整 AdaLN 投影的原测量结果。详见[测速方法](docs/benchmark.md)。

### VAE 解码

以下为论文中 **1344 × 768、每 17 帧、单张 H200** 的解码器结果，延迟与 FPS 仅统计解码。时序误差为 deltaMAE × 10³。

| 解码器 | 参数量（M）↓ | PSNR（dB）↑ | LPIPS ↓ | 时序误差 ↓ | 延迟（毫秒）↓ | FPS ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native H3 | 2423.5 | 35.04 | 0.0434 | 9.05 | 831.30 | 20.44 |
| Tiny-H3 | 9.9 | 30.40 | 0.1040 | 13.78 | 10.90 | 1555.64 |
| StreamRender¹ | 1617.8 | 33.74 | 0.0533 | 10.09 | 143.31 | 118.63 |
| StreamRender² | 1617.8 | 33.95 | 0.0516 | 9.88 | 157.65 | 123.91 |
| SolarX V3 | 2423.5 | 34.09 | 0.0484 | 10.20 | 552.79 | 30.54 |
| LightVAE | 1752.1 | 35.13 | 0.0423 | 8.97 | 586.40 | 28.97 |
| **Ultra-Fast-VAE** | 361.2 | 34.43 | 0.0511 | 9.80 | 46.77 | 363.21 |

StreamRender¹/² 每次使用一个／两个 latent；LightVAE 使用七个 latent，其中两个来自未来。Ultra-Fast-VAE 使用 W8A8。完整生成流程还包含解码之外的计算。

发布运行时的测量结果和复现命令见[性能测量说明](docs/benchmark.md)。

## AdaLN 显存优化

Base 和 Fast 仅包含默认采样日程需要的 AdaLN 调制表，加载时直接使用这些精简表替代各层完整投影，从而减少下载体积和常驻参数显存。Fast 默认采用三步采样。原生 VAE 与 Qwen 编码器用于条件准备，完成后可在 Fast 生成前移出显存。采样日程限制及显存统计范围见[推理显存说明](docs/inference.md#adaln-memory)。

## 文档

| 指南 | 内容 |
| --- | --- |
| [安装](docs/installation.md) | 依赖、FlashAttention-3 和环境设置 |
| [模型权重](docs/weights.md) | 下载、固定版本和组件目录 |
| [推理](docs/inference.md) | Base、Fast、加速及流式接口 |
| [数据准备](docs/data.md) | 输入格式、相机对齐、条件缓存与 HandiVid 数据集 |
| [性能测量](docs/benchmark.md) | 吞吐、响应时间及测量定义 |
| [输出检查](docs/VALIDATION.md) | 输入和生成视频检查 |

## 相关项目

- [BiWM](https://github.com/LynnReal-AI/BiWM)
- [LynnReal-Omni](https://github.com/LynnReal-AI/LynnReal-Omni)

## 许可证

本仓库遵循 [MiniMax H3 Community License Agreement](LICENSE)。第三方组件保留各自许可证和声明，详见 [NOTICE](NOTICE) 与[第三方致谢](third_party/acknowledgements/README.md)。

HandiVid 的许可与第三方声明见其[数据集许可证](https://huggingface.co/datasets/shaohao011/handivid/blob/main/LICENSE.md)和[第三方说明](https://huggingface.co/datasets/shaohao011/handivid/blob/main/NOTICE.md)。

## 致谢

感谢 SolarWM、H3、Qwen、Diffusers、PyTorch、TAEHV 和 DuoMatching 贡献者提供的开源实现。各组件的来源与声明见 [third_party/README.md](third_party/README.md)。

## 引用

### LynnReal-Hand 模型与 Ultra-Fast-VAE

如果在研究中使用 LynnReal-Hand 或 Ultra-Fast-VAE，请引用 LynnReal-Hand 论文。

```bibtex
@article{rui2026lynnreal,
  title={{LynnReal-Hand}: Unlocking World Interaction with Hand Control},
  author={Rui, Shaohao and Zhang, Chaoyang and Mao, Xiaofeng and Ma, Weijie},
  journal={arXiv preprint arXiv:2611.21479},
  year={2026}
}
```

### HandiVid 数据集

如果在研究中使用 HandiVid 数据集，请引用 HandiVid 论文。

```bibtex
@article{zhang2026handivid,
  title={{HandiVid}: A High-Quality Dataset for Hand-Driven Interactive Video Generation},
  author={Zhang, Chaoyang and Rui, Shaohao and Mao, Xiaofeng and Xu, Yuanxun and Xia, Haoyuan and Liu, Xiaohong and Du, Qinghe and Ma, Weijie},
  journal={arXiv preprint arXiv:2611.29693},
  year={2026},
  url={https://arxiv.org/abs/2611.29693}
}
```

## 相关论文

相关论文：

- **BiWM: Advancing Open-Source Interactive Video World Models with Bidirectional Autoregression** — [Paper](https://arxiv.org/abs/2606.10135) · [Code](https://github.com/LynnReal-AI/BiWM)
- **LynnReal-Omni: Native multi-modal Video Generation for Agentic Visual Workflows** — [Paper](https://arxiv.org/abs/2609.15863) · [Code](https://github.com/LynnReal-AI/LynnReal-Omni)

```bibtex
@article{rui2026biwm,
  title={{BiWM}: Advancing Open-Source Interactive Video World Models with Bidirectional Autoregression},
  author={Rui, Shaohao and Mao, Xiaofeng and Zhang, Zhanyu and Lin, Peijia and Zhu, Yansong and Zhang, Yibo and Wan, Haibin and Zhao, Zhangrui and Ma, Weijie},
  journal={arXiv preprint arXiv:2606.10135},
  year={2026},
  url={https://arxiv.org/abs/2606.10135}
}

@misc{mao2026lynnrealomninativemultimodalvideo,
  title={{LynnReal-Omni}: Native multi-modal Video Generation for Agentic Visual Workflows},
  author={Xiaofeng Mao and Peijia Lin and Shaohao Rui and Yibo Zhang and Haibin Wan and Weijie Ma},
  year={2026},
  eprint={2609.15863},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.15863}
}
```
