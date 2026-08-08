# DRFT: Dense Rank-Factored Transformer for Image Super-Resolution

DRFT is a single-image super-resolution transformer designed around image-wide normalization, efficient local attention, dense residual refinement, and deployment-aware graph structure.

This repository tracks the current `drft_arch.py` from the DRFT traiNNer-redux development tree. The previous standalone implementation is preserved locally as `drft_legacy.bak` and intentionally ignored by Git.

> [!IMPORTANT]
> The current architecture expects the matching traiNNer integration, including `traiNNer.ops.drft_ocab_flex` for the compiled OCAB training path. Copying this file into an older stock traiNNer checkout without its companion runtime changes is not a complete installation.

## Current architecture

- **Full i-LN trunk** — every transformer normalization site uses paper-style image-wide i-LN. There is no legacy LayerNorm factory or `use_iln` compatibility switch.
- **Asymmetric attention width** — shifted local attention and OCAB retain the full trunk width, while unshifted blocks use a smaller physical Q/K/V space where the factory specifies it.
- **Rank-factored local position bias** — local attention injects learned low-rank query/key factors without materializing a full position-bias matrix during training.
- **Exact overlapping cross-attention** — a 32x32 query window uses centered 40x40 K/V context by default. This preserves OCAB behavior rather than replacing it with ordinary local attention.
- **Dense RHAG refinement** — ACT block endpoints are fused within each RHAG, followed by a residual group convolution.
- **EDBB convolution branch** — an Edge-Enhanced Diverse Branch Block provides multi-branch training and folds to one convolution for inference.
- **SwiGLU channel attention and FFN** — both global channel recalibration and the convolutional feed-forward path use gated activations.
- **LayerScale and DropPath** — ACT residual branches use LayerScale; DRFT-XL additionally enables RHAG-level LayerScale by default.
- **Compile-aware training graph** — the model advertises a full-graph, static-shape compile contract and a configuration-derived persistent-cache identity.
- **Deployment-aware export** — the canonical export path folds reparameterizable branches, captures static bias factors, uses the Q4R gather-free shifted-window split, and supports an admitted FP32/BF16/FP16 TensorRT layout.

## Model factories

All standard factories use `rank=32` and full i-LN with `iln_eps=1e-4` by default.

| Factory | Width | RHAGs | Blocks per RHAG | Full heads | Unshifted heads | Unshifted Q/K/V width | Window / OCAB span | Dense skip | Reconstruction |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| `drft_nano` | 32 | 1 | 2 | 1 | 1 | 16 | 16 / 20 | No | Direct |
| `drft_micro` | 64 | 1 | 6 | 2 | 1 | 32 | 32 / 40 | Yes | Progressive |
| `drft_light` | 96 | 2 | 6 | 3 | 1 | 48 | 32 / 40 | Yes | Progressive |
| `drft_xs` | 128 | 4 | 6 | 4 | 2 | 64 | 32 / 40 | Yes | Progressive |
| `drft_s` | 160 | 6 | 6 | 5 | 3 | 96 | 32 / 40 | Yes | Progressive |
| `drft_m` | 192 | 8 | 6 | 6 | 3 | 96 | 32 / 40 | Yes | Progressive |
| `drft_l` | 224 | 10 | 6 | 7 | 4 | 128 | 32 / 40 | Yes | Progressive |
| `drft_xl` | 256 | 14 | 6 | 8 | 4 | 128 | 32 / 40 | Yes | Progressive |

DRFT-XL defaults to `rhag_layer_scale_init=1e-4`. The same option can be passed explicitly to another factory when RHAG-level residual scaling is wanted.

### Teacher factories

Every student has a width-compatible teacher factory:

`drft_nano_teacher`, `drft_micro_teacher`, `drft_light_teacher`, `drft_xs_teacher`, `drft_s_teacher`, `drft_m_teacher`, `drft_l_teacher`, and `drft_xl_teacher`.

The teachers preserve the student's width and reconstruction topology, increase refinement depth through additional RHAGs, and use full-width unshifted attention. `distillation_feature_pairs()` exposes proportional RHAG endpoints plus the deep-trunk endpoint for feature distillation.

## Attention modes

| Mode | Shifted interior windows | Shifted boundary windows | Intended use |
|---|---|---|---|
| `masked` | PyTorch SDPA / Flash when eligible | Additive masked attention | Portable default |
| `hybrid` | PyTorch Flash SDPA | FlexAttention and the compiled OCAB training route | Fast compiled Linux training |

`hybrid` requires Linux, Triton, and the matching traiNNer runtime. The mathematical region mask is unchanged; routing only selects the backend used for each compatible path.

## traiNNer configuration

Minimal DRFT-Light generator configuration:

```yaml
use_amp: true
amp_bf16: true
use_channels_last: true
use_compile: true
compile_mode: max-autotune
compile_discriminator: false

network_g:
  type: drft_light
  scale: 4
  window_size: 32
  rank: 32
  iln_eps: 1.0e-4
  drop_path_rate: 0.1
  use_checkpoint: true
  use_checkpoint_ocab: false
  attn_type: hybrid

train:
  per_image_outlier_guard:
    enabled: true
    action: exclude
    start_iter: 10000
    max_absolute_error: 10.0
```

`use_checkpoint` controls ACT checkpointing unless `use_checkpoint_act` overrides it. OCAB checkpointing remains off by default and is controlled independently by `use_checkpoint_ocab`.

The per-image guard is a traiNNer feature, not part of the inference architecture. It rejects pathological samples before their update is accepted and has no deployment cost.

## Inference folding

```python
model.eval()
model.fold_reparam_conv()
```

Folding converts each EDBB training structure into its equivalent inference convolution. Do not use a folded model to resume ordinary multi-branch training. The ONNX preparation API folds an inference clone automatically, leaving the original training model unchanged.

## ONNX and TensorRT

Prepare the model from the canonical 96x96 NCHW example while keeping H/W symbolic in the portable ONNX:

```python
deployment = model.prepare_for_onnx_export(
    (1, 3, 96, 96),
    precision="tensorrt_mixed",
    dynamic_spatial=True,
)
```

For DRFT, the dynamic-spatial route uses PyTorch's legacy ONNX exporter. Batch and channels stay fixed; height and width are dynamic. Build a separate fixed-profile TensorRT engine for each actual deployment resolution.

The admitted `tensorrt_mixed` policy is:

- FP32 host input and output
- BF16 body
- FP16 upsampler and final convolution
- zero FP8 unless a separate calibrated quantization workflow is deliberately applied

The portable graph materializes exact OCAB bias so ordinary ONNX runtimes can execute it. When the separately built DRFT TensorRT plugin is available, create the compact plugin graph without modifying the portable source:

```python
from traiNNer.archs.drft_arch import rewrite_onnx_for_tensorrt_plugins

report = rewrite_onnx_for_tensorrt_plugins(
    "drft_portable.onnx",
    "drft_tensorrt_plugin.onnx",
    strategy="compact_bias",
)
```

`compact_bias` keeps TensorRT's native fused attention and expands the exact learned OCAB table at runtime. `fused_attention` is available as a lower-engine-memory fallback.

## Checkpoint compatibility

The current file is a clean, fully i-LN architecture line with revised factories from Nano through XL. Older DRFT checkpoints are not assumed to load strictly: factory widths, RHAG counts, normalization parameters, attention projections, and reconstruction choices may differ. Treat conversion or partial loading as an explicit migration rather than compatibility by default.

## Historical quality reference

The following result belongs to an earlier DRFT-L factory and is retained only as project lineage; it is not presented as a benchmark of the current C224/G10 fully i-LN factory.

| Model | Iterations | Urban100 PSNR | Urban100 SSIM | Manga109 PSNR | Manga109 SSIM |
|---|---:|---:|---:|---:|---:|
| Historical DRFT-L* | 761K | 29.31 | 0.8631 | 33.37 | 0.9357 |

\* Trained on the enhanced dataset used by the original comparison.

## Requirements

- The matching DRFT traiNNer-redux integration
- PyTorch with SDPA support
- Linux and Triton for `attn_type: hybrid`
- `onnx` and NumPy for plugin-graph rewriting
- TensorRT and the separately built DRFT plugin library for plugin deployment

## Research lineage

DRFT incorporates or builds upon ideas from:

- [HAT: Activating More Pixels in Image Super-Resolution Transformer](https://arxiv.org/abs/2205.04437) — RHAG and OCAB
- [DRCT](https://arxiv.org/abs/2404.00722) — dense transformer refinement
- [Analyzing the Training Dynamics of Image Restoration Transformers](https://arxiv.org/abs/2504.06629) — i-LN
- [FlashBias](https://arxiv.org/abs/2505.12044) — rank-factored attention-bias formulation
- [SwinIR](https://arxiv.org/abs/2108.10257) — shifted-window image restoration
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) — channel recalibration
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU
- [Going Deeper with Image Transformers](https://arxiv.org/abs/2103.17239) — LayerScale
- [Efficient Image Super-Resolution Using Pixel Attention](https://arxiv.org/abs/2010.01073) — pixel attention reconstruction

Training and evaluation use [traiNNer-redux](https://github.com/the-database/traiNNer-redux).

## License

This project is licensed under the [MIT License](LICENSE).
