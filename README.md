# DRFT: Dense Rank-Factored Transformer for Image Super-Resolution

A transformer architecture for single image super-resolution that combines rank-factored implicit neural bias, dense skip connections, and hybrid attention-convolution blocks.

## Architecture

DRFT builds on the Residual Hybrid Attention Group (RHAG) paradigm with several key components:

- **Rank-Factored Neural Bias** — Flash-compatible position bias via Q/K concatenation with learned low-rank factors, avoiding the full N×N bias matrix while remaining compatible with SDPA/FlashAttention backends.
- **Attention-Convolution Transformer (ACT) Blocks** — Each block combines window self-attention (with rank-factored bias), ECB reparameterizable convolution with ECA channel attention, and Conv-SwiGLU FFN with depthwise locality injection.
- **i-LN (Image Restoration Tailored Layer Normalization)** — Spatially holistic normalization with input-adaptive rescaling from ["Analyzing the Training Dynamics of Image Restoration Transformers"](https://openreview.net/forum?id=owziuM1nsR) (ICLR 2026). Applied to early stages for stable training.
- **Dense Skip Connections** — DRCT-style dense fusion within each RHAG for information preservation across transformer blocks.
- **Overlapping Cross-Attention (OCAB)** — HAT-style cross-window attention with asymmetric rank-factored bias for direct inter-window information flow.
- **Pixel Attention Upsampling** — Per-pixel attention weights refine features before PixelShuffle reconstruction.
- **ECB Reparameterizable Conv** — Multi-branch training (3x3 + 1x1 + identity) that folds into a single 3x3 conv at inference for zero-cost accuracy gain.
- **LayerScale** — Per-channel residual scaling initialized to small values for stable deep network training.

### Residual Formula

```
x + std * drop_path(ls(attn)) + ls(conv) * conv_scale    # Attention adapts to input via std
x + std * drop_path(ls(ffn))                              # FFN also rescaled by std
```

Where `std` is the stabilized standard deviation from i-LN (detached, compressed, clamped).

## Benchmark Results (x4 SR)

Preliminary results with DRFT-L at 706K/800K training iterations (training in progress):

| Method | Urban100 PSNR | Urban100 SSIM | Manga109 PSNR | Manga109 SSIM |
|--------|:-:|:-:|:-:|:-:|
| SwinIR | 27.45 | 0.8254 | 32.03 | 0.9260 |
| DAT | 27.87 | 0.8343 | 32.51 | 0.9291 |
| HAT-L | 28.93 | 0.8547 | 33.28 | 0.9348 |
| DRCT-L | 28.70 | 0.8508 | 33.14 | 0.9347 |
| **DRFT-L** (706K/800K) | **29.15** | **0.8602** | 32.27 | **0.9351** |

> Note: Results are preliminary. Training has not yet converged (706K of 800K iterations).

## Model Variants

| Variant | embed_dim | RHAG Layers | Heads | head_dim | Dense Skip |
|---------|:-:|:-:|:-:|:-:|:-:|
| DRFT-XS | 128 | 4 | 4 | 32 | Yes |
| DRFT-S | 160 | 6 | 5 | 32 | Yes |
| DRFT-M | 192 | 6 | 6 | 32 | Yes |
| DRFT-L | 192 | 12 | 6 | 32 | Yes |

All variants use `head_dim=32` and `rank=8` (augmented dim = 40), satisfying FlashAttention alignment constraints.

## Usage

```python
import torch
from drft_arch import DRFT, drft_l, drft_m, drft_s, drft_xs

# Using factory functions
model = drft_l(scale=4)

# Or direct construction with custom config
model = DRFT(
    upscale=4,
    embed_dim=192,
    depths=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
    num_heads=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
    window_size=32,
    overlap_ratio=0.5,
    mlp_ratio=2.667,
    drop_path_rate=0.1,
    dense_skip=True,
    use_iln=True,
    rank=8,
)

# Inference
x = torch.randn(1, 3, 64, 64)
with torch.no_grad():
    output = model(x)  # (1, 3, 256, 256) for scale=4
```

### Inference Optimization

```python
# Fold ECB multi-branch convs into single 3x3 for faster inference
model.fold_ecb()
```

### ONNX/TensorRT Export

```python
# Enable math-attention fallback for export compatibility
model.set_export_attention_mode(True)
```

## Requirements

- PyTorch >= 2.0
- CUDA with FlashAttention/SDPA support (recommended)

## Credits

This architecture builds on ideas and components from the following works:

| Component | Paper | Authors |
|-----------|-------|---------|
| RHAG + OCAB | [Activating More Pixels in Image Super-Resolution Transformer](https://arxiv.org/abs/2205.04437) (CVPR 2023) | Xiangyu Chen, Xintao Wang, Jiantao Zhou, Yu Qiao, Chao Dong |
| Dense Skip Connections | [DRCT: Saving Image Super-Resolution away from Information Bottleneck](https://arxiv.org/abs/2404.00722) | Chih-Chung Hsu, Chia-Ming Lee, Yi-Shiuan Chou |
| i-LN | [Analyzing the Training Dynamics of Image Restoration Transformers](https://openreview.net/forum?id=owziuM1nsR) (ICLR 2026) | Rui Qin, Ming Sun, Chao Zhou, Bin Wang |
| Rank-Factored Bias | [FlashBias: Fast Computation of Attention with Bias](https://arxiv.org/abs/2505.12044) (NeurIPS 2025) | Zihao Zheng, Jingze Shi, Hanqiu Chen, Xiangyu Zhang, Zhe Li |
| Shifted Window Attention | [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257) (ICCVW 2021) | Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, Radu Timofte |
| ECB Reparameterizable Conv | [Edge-oriented Convolution Block for Real-time Super Resolution](https://github.com/xindongzhang/ECBSR) (ACM MM 2021) | Xindong Zhang, Hui Zeng, Lei Zhang |
| ECA Channel Attention | [ECA-Net: Efficient Channel Attention](https://arxiv.org/abs/1910.03151) (CVPR 2020) | Qilong Wang, Banggu Wu, Pengfei Zhu, Peihua Li, Wangmeng Zuo, Qinghua Hu |
| SwiGLU FFN | [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | Noam Shazeer |
| LayerScale | [Going Deeper with Image Transformers](https://arxiv.org/abs/2103.17239) (ICCV 2021) | Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, Herve Jegou |
| Pixel Attention | [Efficient Image Super-Resolution Using Pixel Attention](https://arxiv.org/abs/2010.01073) (ECCVW 2020) | Hengyuan Zhao, Xiangtao Kong, Jingwen He, Yu Qiao, Chao Dong |

Trained with [traiNNer-redux](https://github.com/the-database/traiNNer-redux).

## License

This project is licensed under the [MIT License](LICENSE).
