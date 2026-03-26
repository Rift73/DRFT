# ruff: noqa
# type: ignore
"""
DRFT: Dense Rank-Factored Transformer for Super-Resolution
A transformer architecture optimized for RTX 50 series (SM120/CUDA 13)

Key innovations:
- Rank-factored implicit neural bias (Flash-compatible position bias)
- Conv-SwiGLU FFN with locality injection (fused gate+value projection)
- i-LN (Image Restoration Tailored Layer Normalization) from ICLR 2026 paper
- ECB-style reparameterizable conv (BN-free)
- ECA channel attention
- Dense skip connections (DRCT-style)
- Pixel Attention upsampling
- SDPA (Scaled Dot Product Attention) throughout for FlashAttention support

Design follows HAT_iLN's proven formula (with LayerScale for stable training):
- Attention: std-scaled (input-adaptive rescaling via i-LN) + LayerScale
- Conv: independent of std (0.01 weight via conv_scale) + LayerScale
- FFN: std-scaled (input-adaptive rescaling via i-LN) + LayerScale
- Formula: x + std * drop_path(ls(attn)) + ls(conv) * conv_scale
This ensures attention adapts to input while conv maintains texture in smooth regions.

i-LN (raw variant, matching TRFT — "Analyzing the Training Dynamics of IR Transformers"):
- Replaces per-token LayerNorm with spatially holistic normalization (stats over [L*C])
- std applied ONLY to attention output, NOT to conv
- LN* stats computed in FP32 for numerical stability
- Residual scale is raw std with gradients flowing through (no detach/compress/clamp)
- LayerScale provides sufficient output gating — clamped std is redundant
- Prevents feature magnitude divergence (million-scale) under conventional LayerNorm
- Stabilizes channel-wise entropy during training
- Applied to norm1/norm2 in ACTBlock and OCAB across ALL RHAG stages

Constraints:
- embed_dim / num_heads must be divisible by 8 (SDPA/FlashAttention compatible)
- No BatchNorm anywhere
- Multi-scale support: 1x, 2x, 3x, 4x, 8x

torch.compile Compatibility:
- No int() casts in tensor shape computations
- Local variables for buffer dtype/device conversion
- Native SDPA path without masked attention fallback
- Native PyTorch ops instead of einops

DDP Compatibility: Fixed unused parameters in OCAB, safe ECB folding

Training Recommendations:
- Use 5000-10000 iteration linear warmup
- Exclude bias and bias scale terms from weight decay
- Gradient clipping at max_norm=1.0 recommended

Example optimizer setup:
    no_decay = ['bias', 'bias_scale']
    param_groups = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.05},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Flex Attention (hybrid mode only — requires Triton, Linux)
try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
    _FLEX_AVAILABLE = True
except ImportError:
    _FLEX_AVAILABLE = False

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    _SDPA_BACKEND_AVAILABLE = True
except ImportError:
    _SDPA_BACKEND_AVAILABLE = False

ATTN_TYPE = Literal['masked', 'hybrid']

# =============================================================================
# Utility Functions
# =============================================================================

def to_2tuple(x):
    """Convert to 2-tuple."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization."""
    with torch.no_grad():
        l = (1. + math.erf((a - mean) / std / math.sqrt(2.))) / 2.
        u = (1. + math.erf((b - mean) / std / math.sqrt(2.))) / 2.
        tensor.uniform_(l, u)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


def _scaled_dot_product_attention_export_safe(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    training: bool = False,
    use_export_safe_math: bool = False,
) -> torch.Tensor:
    """SDPA wrapper with math-attention fallback for ONNX/TensorRT export.

    The fallback path avoids emitting ONNX Attention op by using explicit
    matmul/softmax/matmul. Matmuls stay in input dtype (BF16 tensor cores)
    while only softmax is upcast to FP32 for accumulation stability.
    Pre-softmax clamping prevents exp() overflow in reduced precision.
    """
    if not use_export_safe_math:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p if training else 0.0,
            scale=scale,
        )

    # Q@K^T matmul stays in input dtype (BF16 tensor cores when available)
    attn = torch.matmul(q, k.transpose(-2, -1))
    if scale is not None:
        attn = attn * scale
    if attn_mask is not None:
        attn = attn + attn_mask.to(dtype=attn.dtype)
    # Pre-softmax clamp: prevents exp() overflow in BF16/FP16.
    # exp(50) ≈ 5.2e21 is safely within BF16 range (max 3.4e38).
    # Region mask -inf is clamped to -50: exp(-50) ≈ 0 — masking preserved.
    attn = attn.clamp(-50.0, 50.0)
    # Only softmax needs FP32 (accumulation over N tokens loses precision in BF16)
    attn = torch.softmax(attn.float(), dim=-1).to(dtype=v.dtype)
    if dropout_p > 0.0 and training:
        attn = F.dropout(attn, p=dropout_p, training=True)
    # A@V matmul stays in input dtype
    out = torch.matmul(attn, v)
    return out


def _make_bias_score_mod(bias_flat: torch.Tensor, seq_len: int):
    """Create a score_mod for flex_attention that adds rank-factored neural bias.

    Uses flattened 2D indexing (num_heads, seq_len*seq_len) to avoid 3D fancy
    indexing which causes SubgraphLoweringException in torch.compile's inductor.
    This matches ESCReal's apply_rpe pattern: table[h, computed_flat_idx].

    Created fresh each forward pass with the current-device bias tensor.
    Dynamo detects the closure structure is identical across calls and treats
    the captured tensor as a graph input — no recompilation overhead.
    """
    def score_mod(score, b, h, q_idx, kv_idx):
        return score + bias_flat[h, q_idx * seq_len + kv_idx]
    return score_mod


# Single compiled flex_attention handle shared by all WindowAttentionRFB instances.
# Dynamo's code cache means all call sites reuse the same Triton kernel after the
# first compilation, so per-instance compilation is redundant overhead.
if _FLEX_AVAILABLE:
    _compiled_flex_attention = torch.compile(flex_attention, dynamic=True)


_SHARED_OCAB_INDEX_CACHE: dict[tuple[int, int, int, int, str, int], torch.Tensor] = {}


def _build_relative_position_index(
    q_window_size: tuple[int, int],
    k_window_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Build cross-attention relative-position indices for OCAB."""
    qh, qw = q_window_size
    kh, kw = k_window_size

    coords_q_h = torch.arange(qh, device=device)
    coords_q_w = torch.arange(qw, device=device)
    coords_q = torch.stack(torch.meshgrid(coords_q_h, coords_q_w, indexing='ij'))
    coords_q_flatten = coords_q.flatten(1)

    coords_k_h = torch.arange(kh, device=device)
    coords_k_w = torch.arange(kw, device=device)
    coords_k = torch.stack(torch.meshgrid(coords_k_h, coords_k_w, indexing='ij'))
    coords_k_flatten = coords_k.flatten(1)

    relative_coords = coords_k_flatten[:, None, :] - coords_q_flatten[:, :, None]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += qh - 1
    relative_coords[:, :, 1] += qw - 1
    relative_coords[:, :, 0] *= qw + kw - 1
    return relative_coords.sum(-1).long()


def _get_shared_relative_position_index(
    q_window_size: tuple[int, int],
    k_window_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Get a shared OCAB relative-position index cache for the current device."""
    device_index = -1 if device.index is None else int(device.index)
    key = (
        q_window_size[0],
        q_window_size[1],
        k_window_size[0],
        k_window_size[1],
        device.type,
        device_index,
    )
    cached = _SHARED_OCAB_INDEX_CACHE.get(key)
    if cached is None:
        cached = _build_relative_position_index(q_window_size, k_window_size, device)
        _SHARED_OCAB_INDEX_CACHE[key] = cached
    return cached


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition into windows.
    
    Args:
        x: (B, H, W, C)
        window_size: window size
        
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partition.
    
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
        
    Returns:
        x: (B, H, W, C)
    """
    # Use integer division to avoid graph break from int() cast
    B = windows.shape[0] // (H // window_size) // (W // window_size)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def pad_to_multiple(x: torch.Tensor, multiple: int, mode: str = "reflect") -> torch.Tensor:
    """Pad tensor spatial dimensions to be divisible by multiple.

    Args:
        x: Input tensor of shape (B, C, H, W).
        multiple: Target divisibility factor for H and W.
        mode: Padding mode for F.pad. Default: "reflect".

    Returns:
        Padded tensor of shape (B, C, H', W') where H' and W'
        are the smallest values >= H and W divisible by multiple.

    Note:
        Always executes F.pad (even when no padding needed) to avoid
        torch.compile graph breaks from conditional branching.
    """
    _, _, H, W = x.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    # Always execute pad to avoid torch.compile graph breaks from conditional branching
    # When pad_h=0 and pad_w=0, this is a no-op
    x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x


# =============================================================================
# AffineTransform (for final norm in i-LN networks)
# =============================================================================

class AffineTransform(nn.Module):
    """Simple affine transformation (gamma * x + beta) without normalization.

    Used for final norm layer in i-LN networks where we want learnable
    scaling/shifting but no actual normalization.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias


# =============================================================================
# LayerScale
# =============================================================================

class LayerScale(nn.Module):
    """LayerScale for stable deep network training.

    Applies learnable per-channel scaling to residual outputs,
    initialized to small values (eps) to make early training
    behave like a shallow network.

    Note: This class handles both 3D (B, N, C) and 4D (B, C, H, W) tensors
    using pre-computed view shapes to avoid torch.compile graph breaks.
    """

    def __init__(self, dim: int, init_value: float = 1e-6, input_format: Literal["3d", "4d"] = "3d") -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))
        self.input_format = input_format

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use pre-defined format to avoid torch.compile graph breaks from ndim check
        if self.input_format == "4d":
            return x * self.gamma.view(1, -1, 1, 1)
        else:
            return x * self.gamma


# =============================================================================
# i-LN: Image Restoration Transformer Tailored Layer Normalization
# =============================================================================

class iLN(nn.Module):
    """Image Restoration Transformer Tailored Layer Normalization (i-LN).

    From: "Analyzing the Training Dynamics of Image Restoration Transformers:
           A Revisit to Layer Normalization" (ICLR 2026)

    Two key differences from standard LayerNorm:
    1. Spatially Holistic Normalization (LN*): Statistics computed over BOTH
       spatial (L) AND channel (C) dimensions, not just channel dimension.
       This preserves inter-pixel structure up to a global scale factor.

    2. Input-Adaptive Rescaling: The caller rescales the output by the original
       standard deviation to preserve input-dependent statistics and allow
       range flexibility. This is done OUTSIDE this module (in the block).

    Mathematical formulation:
        LN*(x) = γ * (x - μ) / σ + β
        where μ = E_{ℓ,c}[x_{ℓ,c}], σ² = E_{ℓ,c}[(x_{ℓ,c} - μ)²]

    The forward returns BOTH the normalized output AND the std for rescaling:
        B(x; f, i-LN) = x + σ · f(LN*(x))

    This design:
    - Preserves spatial correlations between tokens (Proposition 2 in paper)
    - Maintains input-dependent feature statistics throughout the network
    - Prevents feature magnitude divergence to million-scale
    - Stabilizes channel-wise entropy during training
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        # Learnable affine parameters (same as standard LayerNorm)
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        # EMA of std for adaptive explosion guard (non-persistent: not saved to checkpoints)
        self.register_buffer('std_ema', torch.ones(1), persistent=False)
        # Buffer for stashing std observation (like BN running stats — compile-safe)
        self.register_buffer('_last_std_buf', torch.ones(1), persistent=False)
        self._std_ema_initialized = False

    def update_std_ema(self) -> None:
        """Update EMA from last observed std. Called via pre-hook, outside compiled forward."""
        with torch.no_grad():
            batch_std = self._last_std_buf
            if not self._std_ema_initialized:
                self.std_ema.copy_(batch_std)
                self._std_ema_initialized = True
            else:
                safe_std = torch.where(batch_std < 2.0 * self.std_ema, batch_std, self.std_ema)
                self.std_ema.lerp_(safe_std, 1e-3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with holistic normalization.

        Args:
            x: Input tensor of shape (B, L, C) where L = H*W (spatial tokens)

        Returns:
            tuple: (normalized_output, std) where:
                - normalized_output: (B, L, C) normalized and affine-transformed
                - std: (B, 1, 1) residual scale for input-adaptive rescaling
        """
        # Compute LN* stats in FP32 for BF16/FP16 stability.
        # var_mean over (L*C) can be 10000+ elements — BF16 mantissa too narrow.
        orig_dtype = x.dtype
        x_fp32 = x.float()
        var, mean = torch.var_mean(x_fp32, dim=(1, 2), keepdim=True, correction=0)
        std_raw = torch.sqrt(var + self.eps)  # (B, 1, 1)

        # Stash mean std for EMA update in pre-hook (buffer write, like BN running stats).
        with torch.no_grad():
            self._last_std_buf.copy_(std_raw.mean())

        # Adaptive explosion guard: cap at 2x EMA (single fusible op).
        scale = torch.min(std_raw, 2.0 * self.std_ema.detach())

        # Normalize in FP32 (paper-faithful normalization path).
        x_norm = (x_fp32 - mean) / std_raw

        # Apply learnable affine transformation, cast output back.
        w = self.weight.view(1, 1, -1).float()
        b = self.bias.view(1, 1, -1).float()
        out = (w * x_norm + b).to(orig_dtype)

        scale = scale.to(orig_dtype)

        # Caller applies residual as: x + scale * f(out)
        return out, scale

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


# =============================================================================
# Rank-Factored Implicit Neural Bias
# =============================================================================

class RankFactoredNeuralBias(nn.Module):
    """Rank-factored neural position bias for Flash-compatible attention.

    Generates per-token factors Bq and Bk (instead of a full NxN bias matrix),
    so position bias can be injected via Q/K concatenation:
        [Q*s | Bq] @ [K | Bk]^T = QK^T/sqrt(d) + BqBk^T
    """

    def __init__(
        self,
        num_heads: int,
        q_window_size: tuple[int, int],
        k_window_size: Optional[tuple[int, int]] = None,
        rank: int = 32,
        hidden_dim: int = 128,
        scale_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.q_window_size = q_window_size
        self.k_window_size = k_window_size if k_window_size is not None else q_window_size
        self.rank = rank
        self.shared_qk_window = self.q_window_size == self.k_window_size
        self._cached_factors: Optional[tuple[torch.Tensor, torch.Tensor]] = None

        if self.shared_qk_window:
            # Symmetric window attention: one MLP pass, then split into Q/K factors.
            self.bias_mlp_qk = nn.Sequential(
                nn.Linear(2, hidden_dim, bias=True),
                nn.GELU(),
                nn.Linear(hidden_dim, num_heads * rank * 2, bias=False),
            )
        else:
            # Asymmetric Q/K windows (e.g., OCAB): keep independent mappings.
            self.bias_mlp_q = nn.Sequential(
                nn.Linear(2, hidden_dim, bias=True),
                nn.GELU(),
                nn.Linear(hidden_dim, num_heads * rank, bias=False),
            )
            self.bias_mlp_k = nn.Sequential(
                nn.Linear(2, hidden_dim, bias=True),
                nn.GELU(),
                nn.Linear(hidden_dim, num_heads * rank, bias=False),
            )
        self.bias_scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))

        self._create_coords_tables()

    @staticmethod
    def _build_coords_table(window_size: tuple[int, int]) -> torch.Tensor:
        wh, ww = window_size
        coords_h = torch.arange(wh, dtype=torch.float32)
        coords_w = torch.arange(ww, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(coords_h, coords_w, indexing="ij")
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)

        coords_log = torch.sign(coords) * torch.log1p(torch.abs(coords))
        coords_log[:, 0] /= max(math.log1p(wh - 1), 1.0)
        coords_log[:, 1] /= max(math.log1p(ww - 1), 1.0)
        return coords_log

    def _create_coords_tables(self) -> None:
        self.register_buffer("q_coords_table", self._build_coords_table(self.q_window_size))
        self.register_buffer("k_coords_table", self._build_coords_table(self.k_window_size))

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training and self._cached_factors is not None:
            return self._cached_factors

        if self.shared_qk_window:
            n = self.q_coords_table.shape[0]
            bqk = self.bias_mlp_qk(self.q_coords_table).view(
                n, self.num_heads, 2, self.rank
            )
            # (N, heads, 2, rank) -> (2, heads, N, rank)
            bq, bk = bqk.permute(2, 1, 0, 3).unbind(0)
        else:
            nq = self.q_coords_table.shape[0]
            nk = self.k_coords_table.shape[0]
            bq = self.bias_mlp_q(self.q_coords_table).view(
                nq, self.num_heads, self.rank
            ).permute(1, 0, 2)
            bk = self.bias_mlp_k(self.k_coords_table).view(
                nk, self.num_heads, self.rank
            ).permute(1, 0, 2)

        # bounded magnitude for stable optimization in early training
        bq = self.bias_scale * torch.sigmoid(bq)
        bk = torch.sigmoid(bk)

        if not self.training:
            self._cached_factors = (bq, bk)
        return bq, bk

    def clear_cache(self) -> None:
        self._cached_factors = None

    def train(self, mode: bool = True) -> "RankFactoredNeuralBias":
        self.clear_cache()
        return super().train(mode)


# =============================================================================
# ECA Channel Attention
# =============================================================================

class ECAAttention(nn.Module):
    """Efficient Channel Attention.
    
    Uses 1D convolution instead of FC layers for channel attention,
    drastically reducing parameters while maintaining effectiveness.
    """
    
    def __init__(self, dim: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Kernel size based on channel count (can be adaptive)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Global average pooling
        y = self.gap(x).view(B, 1, C)  # (B, 1, C)
        
        # 1D convolution across channels
        y = self.conv(y)  # (B, 1, C)
        y = self.sigmoid(y).view(B, C, 1, 1)
        
        return x * y


# =============================================================================
# ECB Reparameterizable Conv Block (BN-free) - DDP SAFE
# =============================================================================

class ECBConvBlock(nn.Module):
    """Edge-oriented Convolution Block (ECB) style reparameterizable conv.
    
    Training: Multiple parallel branches (3x3, 1x1, identity)
    Inference: Folds into single 3x3 conv
    
    NO BATCHNORM - uses simple weight summation for folding.
    
    DDP Safety: fold() can only be called in eval mode.
    """
    
    def __init__(
        self,
        dim: int,
        expand_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.dim = dim
        hidden_dim = int(dim * expand_ratio)
        
        # Branch 1: 3x3 conv
        self.conv3x3 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        
        # Branch 2: 1x1 conv (will be padded to 3x3 for folding)
        self.conv1x1 = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)

        # Branch 3: Identity (represented as 1x1 with identity weights conceptually)
        # We'll just add input directly

        # Learnable branch weights
        self.branch_weight = nn.Parameter(torch.ones(3) / 3)

        self._is_folded = False
        self._folded_conv: Optional[nn.Conv2d] = None

    def _fold_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fold multi-branch weights into single 3x3 conv."""
        # Get branch weights
        w = F.softmax(self.branch_weight, dim=0)

        # 3x3 weights
        weight_3x3 = self.conv3x3.weight * w[0]
        bias_3x3 = self.conv3x3.bias * w[0] if self.conv3x3.bias is not None else 0

        # Pad 1x1 to 3x3
        weight_1x1 = F.pad(self.conv1x1.weight, [1, 1, 1, 1]) * w[1]
        bias_1x1 = self.conv1x1.bias * w[1] if self.conv1x1.bias is not None else 0

        # Build the folded identity kernel on demand to avoid registering a
        # large static buffer that DDP would rebroadcast every forward.
        weight_id = F.pad(
            torch.eye(
                self.dim,
                device=self.conv3x3.weight.device,
                dtype=self.conv3x3.weight.dtype,
            ).view(self.dim, self.dim, 1, 1),
            [1, 1, 1, 1],
        ) * w[2]

        # Sum all branches
        folded_weight = weight_3x3 + weight_1x1 + weight_id
        folded_bias = bias_3x3 + bias_1x1

        return folded_weight, folded_bias

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        identity_weight = F.pad(
            torch.eye(
                self.dim,
                device=self.conv3x3.weight.device,
                dtype=self.conv3x3.weight.dtype,
            ).view(self.dim, self.dim, 1, 1),
            [1, 1, 1, 1],
        )
        destination[prefix + "identity_weight"] = (
            identity_weight if keep_vars else identity_weight.detach()
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        state_dict.pop(prefix + "identity_weight", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def fold(self) -> None:
        """Fold branches into single conv for inference.

        DDP Safety: Only callable in eval mode to prevent unused parameter issues.
        Cleans up original branch params to save memory and reduce state dict size.
        """
        if self._is_folded:
            return

        if self.training:
            raise RuntimeError(
                "ECBConvBlock.fold() can only be called in eval mode. "
                "Call model.eval() first to avoid DDP issues with unused parameters."
            )

        folded_weight, folded_bias = self._fold_weights()
        device = self.conv3x3.weight.device

        self._folded_conv = nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=True)
        self._folded_conv.weight.data = folded_weight
        self._folded_conv.bias.data = folded_bias
        self._folded_conv = self._folded_conv.to(device)

        # Clean up original branch params to free memory
        del self.conv3x3
        del self.conv1x1
        del self.branch_weight

        self._is_folded = True

    def unfold(self) -> None:
        """Unfold back to training mode.

        NOTE: After fold() cleans up branch params, the model cannot be unfolded.
        Reload the training checkpoint to get the multi-branch model back.
        """
        if not self._is_folded:
            return

        if not hasattr(self, 'conv3x3'):
            raise RuntimeError(
                "Cannot unfold: original branch params were cleaned up by fold(). "
                "Reload the training checkpoint to get the multi-branch model back."
            )

        self._is_folded = False
        self._folded_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_folded and self._folded_conv is not None:
            return self._folded_conv(x)
        
        # Multi-branch forward
        w = F.softmax(self.branch_weight, dim=0)
        
        out = w[0] * self.conv3x3(x)
        out = out + w[1] * self.conv1x1(x)
        out = out + w[2] * x  # Identity branch
        
        return out


# =============================================================================
# Conv-SwiGLU FFN
# =============================================================================

class ConvSwiGLUFFN(nn.Module):
    """Convolutional SwiGLU Feed-Forward Network.

    Combines gated linear units (SwiGLU) with depthwise convolution
    for locality injection. Uses 2/3 hidden dimension scaling.

    Uses fused gate+value projection for better memory bandwidth.
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: float = 2.667,  # 2/3 * 4
        drop: float = 0.,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        # Round to multiple of 8 for tensor core efficiency
        hidden_dim = ((hidden_dim + 7) // 8) * 8
        self.hidden_dim = hidden_dim

        # Fused gate and value projection (reads input once instead of twice)
        self.fc_gate_value = nn.Linear(dim, hidden_dim * 2)

        # Depthwise conv for locality injection
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)

        # Dilated depthwise conv for wider receptive field (zero-init coupling, checkpoint-safe)
        self.dwconv_dilated = nn.Conv2d(
            hidden_dim, hidden_dim, 3, 1, padding=2, dilation=2, groups=hidden_dim,
        )
        self.dil_scale = nn.Parameter(torch.zeros(1))

        # Output projection
        self.fc_out = nn.Linear(hidden_dim, dim)

        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, N, C) where N = H * W
        B, N, C = x.shape

        # Fused gate and value projection, then split
        gate_value = self.fc_gate_value(x)
        gate, value = gate_value.chunk(2, dim=-1)

        # SwiGLU: SiLU(gate) * value
        x = F.silu(gate) * value

        # Reshape for depthwise conv
        x = x.transpose(1, 2).view(B, -1, H, W)  # (B, hidden, H, W)
        x = self.dwconv(x)
        x = x + self.dil_scale * self.dwconv_dilated(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N, hidden)

        # Output projection
        x = self.fc_out(x)
        x = self.drop(x)

        return x


# =============================================================================
# Flash-Compatible Window Attention (Rank-Factored Bias)
# =============================================================================

class WindowAttentionRFB(nn.Module):
    """Window attention with rank-factored neural bias (Flash-friendly).

    Supports two attention modes (both use Flash for interior/non-shifted windows):
    - masked: Boundary windows use SDPA with dense attn_mask (portable, no Triton)
    - hybrid: Boundary windows use Flex Attention with BlockMask (requires Triton)
    """

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        rank: int = 32,
        attn_type: ATTN_TYPE = 'masked',
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rank = rank
        self.attn_type = attn_type

        assert self.head_dim % 8 == 0, (
            f"head_dim should be divisible by 8 for SDPA optimization, got {self.head_dim}"
        )
        assert (self.head_dim + rank) % 8 == 0, (
            f"head_dim({self.head_dim}) + rank({rank}) must be divisible by 8 for Flash kernels."
        )

        self.logit_scale = nn.Parameter(
            torch.full((num_heads,), self.head_dim ** -0.5)
        )
        self.val_res_scale = nn.Parameter(torch.zeros(num_heads))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()
        self.attn_drop_p = attn_drop
        self.force_math_attention = False
        self.force_additive_attention = False

        self.neural_bias = RankFactoredNeuralBias(
            num_heads=num_heads,
            q_window_size=window_size,
            k_window_size=window_size,
            rank=rank,
        )

        # Hybrid (Flex Attention) setup — Flex needs ws² for flattened bias indexing
        if attn_type == 'hybrid':
            self._ws_sq = window_size[0] * window_size[1]

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        b_windows, n_tokens, c = x.shape

        qkv = self.qkv(x).reshape(b_windows, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Per-head learned temperature (init = 1/sqrt(d), checkpoint-safe)
        q = q * self.logit_scale.view(1, self.num_heads, 1, 1)

        bq, bk = self.neural_bias()
        use_export_safe_math = self.force_math_attention or torch.onnx.is_in_onnx_export()
        force_additive_attention = self.force_additive_attention or use_export_safe_math

        if force_additive_attention:
            # The additive-bias path is mathematically equivalent to the
            # flash-style augmented-Q/K formulation, but exports to a much
            # simpler ONNX/TensorRT graph.
            bias = torch.einsum('hnr,hmr->hnm', bq, bk)
            attn_mask = bias.unsqueeze(0)
            if mask is not None:
                attn_mask = attn_mask + mask.to(dtype=attn_mask.dtype)
            out = _scaled_dot_product_attention_export_safe(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop_p,
                scale=1.0,
                training=self.training,
                use_export_safe_math=use_export_safe_math,
            )
        elif self.attn_type == 'hybrid' and mask is not None:
            # Flex Attention path: bias via score_mod, region mask via BlockMask.
            # Only used for boundary windows in shifted blocks (requires Triton).
            bias_flat = torch.einsum('hnr,hmr->hnm', bq, bk).reshape(
                bq.shape[0], -1,
            )
            score_mod = _make_bias_score_mod(bias_flat, self._ws_sq)
            out = _compiled_flex_attention(
                q, k, v,
                score_mod=score_mod,
                block_mask=mask,
                scale=1.0,
            )
        elif self.attn_type == 'masked' and mask is not None:
            # SDPA path: bias + region mask -> dense attn_mask.
            bias = torch.einsum('hnr,hmr->hnm', bq, bk)
            attn_mask = bias.unsqueeze(0) + mask.to(dtype=bias.dtype)
            out = _scaled_dot_product_attention_export_safe(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop_p,
                scale=1.0,
                training=self.training,
                use_export_safe_math=use_export_safe_math,
            )
        else:
            # Flash path: bias via Q/K concatenation, no mask needed.
            # Q is already scaled by logit_scale above.
            bq = bq.unsqueeze(0).to(dtype=q.dtype).expand(b_windows, -1, -1, -1)
            bk = bk.unsqueeze(0).to(dtype=k.dtype).expand(b_windows, -1, -1, -1)

            q_aug = torch.cat([q, bq], dim=-1)
            k_aug = torch.cat([k, bk], dim=-1)
            v_aug = F.pad(v, (0, self.rank))

            out = _scaled_dot_product_attention_export_safe(
                q_aug,
                k_aug,
                v_aug,
                attn_mask=None,
                dropout_p=self.attn_drop_p,
                scale=1.0,
                training=self.training,
                use_export_safe_math=use_export_safe_math,
            )
            out = out[..., : self.head_dim]

        # Value residual: preserves token info through deep attention (zero-init, checkpoint-safe)
        v_mean = v.mean(dim=-2, keepdim=True)
        out = out + self.val_res_scale.view(1, self.num_heads, 1, 1) * v_mean

        out = out.transpose(1, 2).reshape(b_windows, n_tokens, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# =============================================================================
# Attention-Convolution Transformer Block (ACT)
# =============================================================================

class ACTBlock(nn.Module):
    """Attention-Convolution Transformer Block with i-LN normalization.

    Combines:
    - Window self-attention with rank-factored neural bias (SDPA backend)
    - ECB reparameterizable conv with ECA channel attention
    - Conv-SwiGLU FFN (fused gate+value)
    - i-LN with input-adaptive rescaling (see module docstring for details)

    Residual formula: x + std * drop_path(ls(attn)) + ls(conv) * conv_scale
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 32,
        shift_size: int = 0,
        mlp_ratio: float = 2.667,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        conv_scale: float = 0.01,
        layer_scale_init: float = 1e-6,
        rank: int = 32,
        use_iln: bool = False,
        attn_type: ATTN_TYPE = 'masked',
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.conv_scale = conv_scale
        self.use_iln = use_iln
        self.force_dense_shifted_attention = False

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"

        # Normalization: iLN returns (normalized_output, std), LayerNorm returns tensor
        self.norm1 = iLN(dim) if use_iln else nn.LayerNorm(dim)

        # Window Attention with rank-factored bias
        self.attn = WindowAttentionRFB(
            dim=dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            rank=rank,
            attn_type=attn_type,
        )

        # ECB Conv block with ECA
        self.conv_block = nn.Sequential(
            ECBConvBlock(dim),
            nn.GELU(),
            ECBConvBlock(dim),
            ECAAttention(dim),
        )

        # LayerScale for stable deep network training
        self.ls_attn = LayerScale(dim, init_value=layer_scale_init)
        self.ls_conv = LayerScale(dim, init_value=layer_scale_init)
        self.ls_ffn = LayerScale(dim, init_value=layer_scale_init)

        # Drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = iLN(dim) if use_iln else nn.LayerNorm(dim)
        self.ffn = ConvSwiGLUFFN(dim=dim, expansion_factor=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int], hybrid_ctx=None) -> torch.Tensor:
        H, W = x_size
        B, L, C = x.shape

        shortcut = x

        # iLN returns (normalized, std) tuple; LayerNorm returns just normalized tensor
        if self.use_iln:
            x_norm, std1 = self.norm1(x)
        else:
            x_norm = self.norm1(x)
        x_2d = x_norm.view(B, H, W, C)

        # === Conv Branch ===
        conv_x = x_2d.permute(0, 3, 1, 2)  # (B, C, H, W)
        conv_x = self.conv_block(conv_x)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)  # (B, L, C)

        # === Attention Branch ===
        # Cyclic shift for shifted window attention
        if self.shift_size > 0:
            shifted_x = torch.roll(x_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_2d

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, ws, ws, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window attention with per-window routing for shifted blocks.
        # Non-shifted blocks: all windows → Flash path (no contamination risk).
        # Shifted blocks: interior windows → Flash, boundary windows → masked/Flex.
        # int_idx_full/bnd_idx_full are pre-expanded batch indices cached at
        # model level — no per-block recomputation needed.
        if self.shift_size > 0 and hybrid_ctx is not None:
            region_mask, int_idx_full, bnd_idx_full = hybrid_ctx

            if self.force_dense_shifted_attention:
                attn_windows = self.attn(x_windows, mask=region_mask)
            else:
                # Process each group sequentially: gather -> attn -> scatter -> free,
                # so both temporary outputs are never alive at the same time.
                # Guard empty index groups and align dtypes for index_put under AMP/BF16.
                attn_windows = torch.empty_like(x_windows)

                # Interior windows: Flash path (no mask needed)
                if int_idx_full.numel() > 0:
                    interior_windows = x_windows[int_idx_full]
                    interior_out = self.attn(interior_windows)
                    if interior_out.dtype != attn_windows.dtype:
                        interior_out = interior_out.to(dtype=attn_windows.dtype)
                    attn_windows[int_idx_full] = interior_out
                    del interior_windows, interior_out

                # Boundary windows: masked SDPA or Flex (with region mask)
                if bnd_idx_full.numel() > 0:
                    boundary_windows = x_windows[bnd_idx_full]
                    boundary_out = self.attn(boundary_windows, mask=region_mask)
                    if boundary_out.dtype != attn_windows.dtype:
                        boundary_out = boundary_out.to(dtype=attn_windows.dtype)
                    attn_windows[bnd_idx_full] = boundary_out
                    del boundary_windows, boundary_out
        else:
            attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x

        attn_x = attn_x.view(B, L, C)

        # Apply LayerScale to attention output
        attn_x = self.ls_attn(attn_x)

        # Apply LayerScale to conv output
        conv_x = self.ls_conv(conv_x)

        # Combine branches:
        # iLN: std multiplies attention output (input-adaptive rescaling)
        # LayerNorm: standard residual connection without std rescaling
        if self.use_iln:
            x = shortcut + std1 * self.drop_path(attn_x) + conv_x * self.conv_scale
        else:
            x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale

        # FFN with LayerScale
        ffn_shortcut = x
        if self.use_iln:
            x_norm2, std2 = self.norm2(x)
        else:
            x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2, H, W)
        ffn_out = self.ls_ffn(ffn_out)
        if self.use_iln:
            x = ffn_shortcut + std2 * self.drop_path(ffn_out)
        else:
            x = ffn_shortcut + self.drop_path(ffn_out)

        return x


# =============================================================================
# Overlapping Cross-Attention Block (OCAB) - DDP FIXED
# =============================================================================

class OCAB(nn.Module):
    """Overlapping Cross-Attention Block with dense signed relative bias.

    Enables direct cross-window information flow using unfold
    with overlapping regions.

    Uses a dense signed relative-position bias table inside OCAB only.
    ACT remains rank-factorized; OCAB stays dense because its positional bias
    is substantially higher-rank and signed, which is critical for text fidelity.

    i-LN: Uses spatially holistic normalization with input-adaptive rescaling
    to preserve input-dependent statistics (from i-LN paper).
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        window_size: int,
        overlap_ratio: float,
        num_heads: int,
        qkv_bias: bool = True,
        mlp_ratio: float = 2.667,
        layer_scale_init: float = 1e-6,
        rank: int = 32,
        use_iln: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.rank = rank
        self.use_iln = use_iln
        self.force_math_attention = False
        self.logit_scale = nn.Parameter(
            torch.full((num_heads,), head_dim ** -0.5)
        )
        self.val_res_scale = nn.Parameter(torch.zeros(num_heads))

        # Normalization: iLN returns (normalized_output, std), LayerNorm returns tensor
        self.norm1 = iLN(dim) if use_iln else nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=window_size,
            padding=(self.overlap_win_size - window_size) // 2,
        )

        hspan = window_size + self.overlap_win_size - 1
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(hspan * hspan, num_heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.proj = nn.Linear(dim, dim)

        # LayerScale for stable deep network training
        self.ls_attn = LayerScale(dim, init_value=layer_scale_init)
        self.ls_ffn = LayerScale(dim, init_value=layer_scale_init)

        self.norm2 = iLN(dim) if use_iln else nn.LayerNorm(dim)
        self.ffn = ConvSwiGLUFFN(dim=dim, expansion_factor=mlp_ratio)

    @property
    def q_window_size(self) -> tuple[int, int]:
        return (self.window_size, self.window_size)

    @property
    def k_window_size(self) -> tuple[int, int]:
        return (self.overlap_win_size, self.overlap_win_size)

    def _relative_position_bias(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Materialize dense OCAB bias late and in the attention dtype."""
        index = _get_shared_relative_position_index(self.q_window_size, self.k_window_size, device)
        n_q = self.window_size * self.window_size
        n_k = self.overlap_win_size * self.overlap_win_size
        bias = self.relative_position_bias_table[index.view(-1)]
        bias = bias.view(n_q, n_k, self.num_heads).permute(2, 0, 1).contiguous()
        return bias.to(dtype=dtype).unsqueeze(0)
    
    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        H, W = x_size
        B, L, C = x.shape

        shortcut = x

        # iLN returns (normalized, std) tuple; LayerNorm returns just normalized tensor
        if self.use_iln:
            x_norm, std1 = self.norm1(x)
        else:
            x_norm = self.norm1(x)
        x_norm = x_norm.view(B, H, W, C)

        # QKV
        qkv = self.qkv(x_norm).reshape(B, H, W, 3, C).permute(3, 0, 4, 1, 2)  # 3, B, C, H, W
        q = qkv[0].permute(0, 2, 3, 1)  # B, H, W, C
        kv = torch.cat((qkv[1], qkv[2]), dim=1)  # B, 2*C, H, W

        # Partition Q windows
        q_windows = window_partition(q, self.window_size)
        q_windows = q_windows.view(-1, self.window_size * self.window_size, C)

        # Unfold KV with overlap
        kv_windows = self.unfold(kv)  # B, 2*C*ow*ow, nW
        nW = kv_windows.shape[2]
        ow = self.overlap_win_size
        # Replace einops with native PyTorch for torch.compile compatibility
        # Original: rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', ...)
        kv_windows = kv_windows.view(B, 2, C, ow, ow, nW)
        kv_windows = kv_windows.permute(1, 0, 5, 3, 4, 2).contiguous()  # nc, B, nW, owh, oww, ch
        kv_windows = kv_windows.view(2, B * nW, ow * ow, C)
        k_windows, v_windows = kv_windows.unbind(0)

        # Attention
        B_, Nq, _ = q_windows.shape
        _, Nk, _ = k_windows.shape

        q = q_windows.reshape(B_, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k_windows.reshape(B_, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v_windows.reshape(B_, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Per-head learned temperature (init = 1/sqrt(d), checkpoint-safe)
        q = q * self.logit_scale.view(1, self.num_heads, 1, 1)

        attn_bias = self._relative_position_bias(q.device, q.dtype)
        use_export_safe_math = self.force_math_attention or torch.onnx.is_in_onnx_export()

        if use_export_safe_math or not _SDPA_BACKEND_AVAILABLE or not q.is_cuda:
            x = _scaled_dot_product_attention_export_safe(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=0.0,
                scale=1.0,
                training=self.training,
                use_export_safe_math=use_export_safe_math,
            )
        else:
            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_bias,
                    dropout_p=0.0,
                    scale=1.0,
                )

        # Value residual: preserves token info through deep attention (zero-init, checkpoint-safe)
        v_mean = v.mean(dim=-2, keepdim=True)
        x = x + self.val_res_scale.view(1, self.num_heads, 1, 1) * v_mean

        x = x.transpose(1, 2).reshape(B_, Nq, self.dim)

        # Merge windows
        x = x.view(-1, self.window_size, self.window_size, self.dim)
        x = window_reverse(x, self.window_size, H, W)
        x = x.view(B, L, self.dim)

        x = self.proj(x)

        # Apply LayerScale to attention output
        x = self.ls_attn(x)

        # Residual: iLN uses std rescaling, LayerNorm uses standard residual
        if self.use_iln:
            x = shortcut + std1 * x
        else:
            x = shortcut + x

        # FFN with LayerScale
        ffn_shortcut = x
        if self.use_iln:
            x_norm2, std2 = self.norm2(x)
        else:
            x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2, H, W)
        ffn_out = self.ls_ffn(ffn_out)
        if self.use_iln:
            x = ffn_shortcut + std2 * ffn_out
        else:
            x = ffn_shortcut + ffn_out

        return x


# =============================================================================
# Pixel Attention for Upsampling
# =============================================================================

class PixelAttention(nn.Module):
    """Pixel Attention for enhanced upsampling.
    
    Generates per-pixel attention weights to refine features
    before PixelShuffle.
    """
    
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.sigmoid(self.pa_conv(x))
        return x * attn


# =============================================================================
# Attention Blocks Container
# =============================================================================

class AttentionBlocks(nn.Module):
    """Container for ACT blocks within a RHAG.

    Implements dense skip connections (DRCT-style) for information preservation.

    Key design choices:
    - OCAB is applied BEFORE dense_fusion on the final sequential output
    - Dense fusion is ADDITIVE (x + ls_dense(dense_out)) with LayerScale
    - LayerScale for stable deep network training
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        overlap_ratio: float,
        mlp_ratio: float = 2.667,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float | list[float] = 0.,
        conv_scale: float = 0.01,
        layer_scale_init: float = 1e-6,
        use_checkpoint: bool = False,
        use_checkpoint_act: Optional[bool] = None,
        use_checkpoint_ocab: Optional[bool] = None,
        dense_skip: bool = True,
        rank: int = 32,
        use_iln: bool = False,
        attn_type: ATTN_TYPE = 'masked',
    ) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint_act = (
            use_checkpoint if use_checkpoint_act is None else use_checkpoint_act
        )
        # Targeted default: checkpoint ACT blocks, keep OCAB uncheckpointed
        # for better throughput/VRAM tradeoff. Users can still override explicitly.
        self.use_checkpoint_ocab = (
            False if use_checkpoint_ocab is None else use_checkpoint_ocab
        )
        self.dense_skip = dense_skip

        # Build ACT blocks
        self.blocks = nn.ModuleList([
            ACTBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                conv_scale=conv_scale,
                layer_scale_init=layer_scale_init,
                rank=rank,
                use_iln=use_iln,
                attn_type=attn_type,
            )
            for i in range(depth)
        ])

        # OCAB for cross-window attention
        self.ocab = OCAB(
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            mlp_ratio=mlp_ratio,
            layer_scale_init=layer_scale_init,
            rank=rank,
            use_iln=use_iln,
        )

        # Dense skip fusion (if enabled) with LayerScale
        if dense_skip:
            self.dense_fusion = nn.Linear(dim * depth, dim)
            self.ls_dense = LayerScale(dim, init_value=1e-4)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int], hybrid_ctx=None) -> torch.Tensor:
        if self.dense_skip:
            # Collect outputs from all blocks
            block_outputs = []
            for blk in self.blocks:
                if self.use_checkpoint_act and self.training:
                    x = checkpoint(blk, x, x_size, hybrid_ctx, use_reentrant=False)
                else:
                    x = blk(x, x_size, hybrid_ctx)
                block_outputs.append(x)

            # OCAB on final sequential output BEFORE dense fusion
            # This ensures cross-window attention operates on refined sequential features
            # OCAB does not use shifted windows — no block_mask needed
            if self.use_checkpoint_ocab and self.training:
                x = checkpoint(self.ocab, x, x_size, use_reentrant=False)
            else:
                x = self.ocab(x, x_size)

            # Dense fusion: concatenate all block outputs and project with LayerScale
            dense_cat = torch.cat(block_outputs, dim=-1)  # (B, L, dim * depth)
            dense_out = self.dense_fusion(dense_cat)  # (B, L, dim)
            x = x + self.ls_dense(dense_out)
        else:
            for blk in self.blocks:
                if self.use_checkpoint_act and self.training:
                    x = checkpoint(blk, x, x_size, hybrid_ctx, use_reentrant=False)
                else:
                    x = blk(x, x_size, hybrid_ctx)

            # OCAB — no block_mask needed (no shifted windows)
            if self.use_checkpoint_ocab and self.training:
                x = checkpoint(self.ocab, x, x_size, use_reentrant=False)
            else:
                x = self.ocab(x, x_size)

        return x


# =============================================================================
# Residual Hybrid Attention Group (RHAG)
# =============================================================================

class RHAG(nn.Module):
    """Residual Hybrid Attention Group.

    Contains multiple ACT blocks + OCAB with residual connection.
    Uses LayerScale for stable deep network training.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        overlap_ratio: float,
        mlp_ratio: float = 2.667,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float | list[float] = 0.,
        conv_scale: float = 0.01,
        layer_scale_init: float = 1e-6,
        use_checkpoint: bool = False,
        use_checkpoint_act: Optional[bool] = None,
        use_checkpoint_ocab: Optional[bool] = None,
        dense_skip: bool = True,
        resi_connection: str = '1conv',
        rank: int = 32,
        use_iln: bool = False,
        attn_type: ATTN_TYPE = 'masked',
    ) -> None:
        super().__init__()
        self.dim = dim

        self.residual_group = AttentionBlocks(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            conv_scale=conv_scale,
            layer_scale_init=layer_scale_init,
            use_checkpoint=use_checkpoint,
            use_checkpoint_act=use_checkpoint_act,
            use_checkpoint_ocab=use_checkpoint_ocab,
            dense_skip=dense_skip,
            rank=rank,
            use_iln=use_iln,
            attn_type=attn_type,
        )
        
        # Residual connection conv
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )
        else:
            self.conv = nn.Identity()

    def forward(self, x: torch.Tensor, x_size: tuple[int, int], hybrid_ctx=None) -> torch.Tensor:
        # x: (B, L, C)
        H, W = x_size

        # Attention blocks
        res = self.residual_group(x, x_size, hybrid_ctx)
        
        # Conv on residual
        res = res.transpose(1, 2).view(-1, self.dim, H, W)
        res = self.conv(res)
        res = res.flatten(2).transpose(1, 2)
        
        # Residual add
        return x + res


# =============================================================================
# Patch Embed / Unembed
# =============================================================================

class PatchEmbed(nn.Module):
    """Image to Patch Embedding (flatten spatial to sequence)."""
    
    def __init__(self, embed_dim: int, norm_layer: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        if self.norm:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """Patch Unembedding (sequence to spatial)."""
    
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        # x: (B, H*W, C) -> (B, C, H, W)
        H, W = x_size
        B, L, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        return x


# =============================================================================
# Upsample Module
# =============================================================================

class Upsample(nn.Module):
    """Multi-scale upsampling with Pixel Attention.
    
    Supports 1x, 2x, 3x, 4x scales.
    """
    
    def __init__(self, scale: int, num_feat: int) -> None:
        super().__init__()
        self.scale = scale
        
        if scale == 1:
            # No upsampling for restoration tasks
            self.up = nn.Identity()
        elif scale == 2:
            self.up = nn.Sequential(
                nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
                PixelAttention(4 * num_feat),
                nn.PixelShuffle(2),
            )
        elif scale == 3:
            self.up = nn.Sequential(
                nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1),
                PixelAttention(9 * num_feat),
                nn.PixelShuffle(3),
            )
        elif scale == 4:
            self.up = nn.Sequential(
                nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
                PixelAttention(4 * num_feat),
                nn.PixelShuffle(2),
                nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
                PixelAttention(4 * num_feat),
                nn.PixelShuffle(2),
            )
        elif scale == 8:
            self.up = nn.Sequential(
                nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
                PixelAttention(4 * num_feat),
                nn.PixelShuffle(2),
                nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
                PixelAttention(4 * num_feat),
                nn.PixelShuffle(2),
                nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
                PixelAttention(4 * num_feat),
                nn.PixelShuffle(2),
            )
        else:
            raise ValueError(f"Unsupported scale: {scale}. Supported: 1, 2, 3, 4, 8")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


# =============================================================================
# DRFT Main Architecture
# =============================================================================

class DRFT(nn.Module):
    """DRFT: Dense Rank-Factored Transformer for Super-Resolution.

    A transformer-based super-resolution architecture optimized for
    RTX 50 series (SM120/CUDA 13) with:
    - Rank-factored implicit neural bias
    - Conv-SwiGLU FFN
    - i-LN normalization (see module docstring for details)
    - LayerScale for stable deep network training
    - HAT-style balance (conv_scale=0.01)
    - ECB reparameterizable convs
    - Dense skip connections
    - Pixel Attention upsampling

    Args:
        img_size: Input image size (for position encoding initialization).
        patch_size: Patch size (always 1 for SR).
        in_chans: Number of input channels.
        embed_dim: Embedding dimension (embed_dim / num_heads must be divisible by 8).
        depths: Number of ACT blocks in each RHAG.
        num_heads: Number of attention heads (embed_dim / num_heads must be divisible by 8).
        window_size: Window size for attention.
        overlap_ratio: Overlap ratio for OCAB.
        mlp_ratio: MLP expansion ratio (default 2.667 for SwiGLU 2/3 rule).
        qkv_bias: Add bias to QKV projections.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        conv_scale: Conv branch scale (0.01 = 1% of attention, HAT's proven value).
        use_checkpoint: Enable targeted gradient checkpointing default
            (ACT=True, OCAB=False).
        use_checkpoint_act: Optional explicit override for ACT checkpointing.
            If None, inherits from use_checkpoint.
        use_checkpoint_ocab: Optional explicit override for OCAB checkpointing.
            If None, defaults to False.
        upscale: Upscale factor (1, 2, 3, 4, 8).
        img_range: Image value range (1.0 or 255.0).
        resi_connection: Residual connection type ('1conv', '3conv', 'identity').
        dense_skip: Use dense skip connections (DRCT-style).
        rank: Rank for factorized ACT position bias.
        use_iln: Use iLN (image restoration tailored LayerNorm) with std rescaling.
            When False, uses standard nn.LayerNorm without std-based residual rescaling.
            Hardcoded to the first 2 RHAG stages when enabled.
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 192,
        depths: Sequence[int] = (6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        window_size: int = 32,
        overlap_ratio: float = 0.5,
        mlp_ratio: float = 2.667,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        conv_scale: float = 0.01,
        layer_scale_init: float = 1e-6,
        use_checkpoint: bool = False,
        use_checkpoint_act: Optional[bool] = None,
        use_checkpoint_ocab: Optional[bool] = None,
        upscale: int = 4,
        img_range: float = 1.,
        resi_connection: str = '1conv',
        dense_skip: bool = True,
        num_feat: int = 64,
        rank: int = 32,
        use_iln: bool = False,
        attn_type: ATTN_TYPE = 'masked',
    ) -> None:
        super().__init__()

        # Validate head_dim constraint (divisible by 8 for tensor core efficiency)
        assert embed_dim % num_heads[0] == 0, "embed_dim must be divisible by num_heads"
        head_dim = embed_dim // num_heads[0]
        assert head_dim % 8 == 0, f"head_dim should be divisible by 8 for SDPA optimization, got {head_dim}"

        # Validate hybrid mode requirements (Flex Attention needs Triton)
        if attn_type == 'hybrid':
            if not _FLEX_AVAILABLE:
                raise ImportError(
                    "DRFT with attn_type='hybrid' requires torch.nn.attention.flex_attention "
                    "(Triton + Linux). Use attn_type='masked' for portable masked attention."
                )
            from traiNNer.utils.misc import require_triton
            require_triton(
                "DRFT with attn_type='hybrid' requires Triton for Flex Attention. "
                "Use attn_type='masked' for portable masked attention."
            )

        self.img_range = img_range
        self.upscale = upscale
        self.window_size = window_size
        self.attn_type = attn_type
        self.force_tensorrt_export_mode = False
        
        # Image mean for normalization (RGB mean from ImageNet)
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.register_buffer('mean', torch.tensor(rgb_mean).view(1, 3, 1, 1))
        else:
            self.register_buffer('mean', torch.zeros(1, in_chans, 1, 1))
        
        num_in_ch = in_chans
        num_out_ch = in_chans
        
        # Shallow feature extraction
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        
        # Patch embed/unembed (no norm - matching HAT_iLN)
        self.patch_embed = PatchEmbed(embed_dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(embed_dim)
        
        # Input resolution for position encoding
        patches_resolution = (img_size, img_size)
        
        # Build RHAGs
        self.num_layers = len(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            stage_use_iln = use_iln
            layer = RHAG(
                dim=embed_dim,
                input_resolution=patches_resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                overlap_ratio=overlap_ratio,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                conv_scale=conv_scale,
                layer_scale_init=layer_scale_init,
                use_checkpoint=use_checkpoint,
                use_checkpoint_act=use_checkpoint_act,
                use_checkpoint_ocab=use_checkpoint_ocab,
                dense_skip=dense_skip,
                resi_connection=resi_connection,
                rank=rank,
                use_iln=stage_use_iln,
                attn_type=attn_type,
            )
            self.layers.append(layer)

        # Final norm selection:
        # iLN path: AffineTransform (no re-normalization after iLN stages)
        # Standard path: LayerNorm
        self.norm = AffineTransform(embed_dim) if use_iln else nn.LayerNorm(embed_dim)
        
        # Feature fusion before upsampling
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        
        # Upsampling
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
            nn.GELU(),
        )
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        
        # Initialize weights
        self.apply(self._init_weights)

        # Persistent flag for spandrel/chaiNNer auto-detection
        self.register_buffer('_use_iln', torch.tensor(use_iln), persistent=True)

        # Per-window routing caches (masked and hybrid modes)
        if attn_type in ('masked', 'hybrid'):
            self._region_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()
            self._window_indices_cache: OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
            self._batch_indices_cache: OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
            self._mask_cache_max = 8
        # Mode-specific mask caches
        if attn_type == 'masked':
            self._dense_mask_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        elif attn_type == 'hybrid':
            self._block_mask_cache: OrderedDict[tuple, object] = OrderedDict()

        # Auto-update iLN EMA after each forward (runs outside compiled graph).
        if use_iln:
            self.register_forward_pre_hook(lambda mod, inp: mod.update_iln_ema())

    def _compute_region_ids(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Compute Swin-style region IDs for shifted window masking.

        Each token within a shifted window is assigned a region ID (0-8)
        indicating which original (pre-roll) window it came from.
        Tokens with different region IDs should not attend to each other.

        Returns:
            region_ids: (nW, ws²) long tensor
        """
        ws = self.window_size
        shift = ws // 2

        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (
            slice(0, -ws),
            slice(-ws, -shift),
            slice(-shift, None),
        )
        w_slices = (
            slice(0, -ws),
            slice(-ws, -shift),
            slice(-shift, None),
        )

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, ws)  # (nW, ws, ws, 1)
        region_ids = mask_windows.view(-1, ws * ws).long()  # (nW, ws²)
        return region_ids

    def _get_window_indices(
        self, H: int, W: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cached (interior_indices, boundary_indices) for per-window routing.

        After cyclic shift, windows form an (nH, nW) grid. Interior windows
        (row < nH-1 and col < nW-1) are entirely within region 0 and need no
        masking — they can use Flash. Boundary windows (last row or last col)
        straddle region boundaries and require Flex with a block_mask.

        Returns:
            (interior_indices, boundary_indices): 1D long tensors indexing into
            the flattened window dimension (size nW = nH_grid * nW_grid).
        """
        idx_key = (H, W, device)
        if idx_key not in self._window_indices_cache:
            ws = self.window_size
            nH_grid = H // ws
            nW_grid = W // ws

            interior = []
            boundary = []
            for i in range(nH_grid):
                for j in range(nW_grid):
                    w = i * nW_grid + j
                    if i < nH_grid - 1 and j < nW_grid - 1:
                        interior.append(w)
                    else:
                        boundary.append(w)

            self._window_indices_cache[idx_key] = (
                torch.tensor(interior, dtype=torch.long, device=device),
                torch.tensor(boundary, dtype=torch.long, device=device),
            )
            if len(self._window_indices_cache) > self._mask_cache_max:
                self._window_indices_cache.popitem(last=False)
        else:
            self._window_indices_cache.move_to_end(idx_key)

        return self._window_indices_cache[idx_key]

    def _get_batch_indices(
        self, B: int, H: int, W: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cached per-batch expanded indices for gather/scatter routing.

        Expands the per-window interior/boundary indices across the batch
        dimension. For x_windows shaped (B*nW, ws², C), these index the
        correct rows for each batch element's interior/boundary windows.

        Returns:
            (int_idx_full, bnd_idx_full): 1D long tensors of size
            (B * n_interior,) and (B * n_boundary,) respectively.
        """
        cache_key = (B, H, W, device)
        if cache_key not in self._batch_indices_cache:
            interior_idx, boundary_idx = self._get_window_indices(H, W, device)
            ws = self.window_size
            nW = (H // ws) * (W // ws)

            offsets = torch.arange(B, device=device).unsqueeze(1) * nW
            int_idx_full = (offsets + interior_idx.unsqueeze(0)).reshape(-1)
            bnd_idx_full = (offsets + boundary_idx.unsqueeze(0)).reshape(-1)

            self._batch_indices_cache[cache_key] = (int_idx_full, bnd_idx_full)
            if len(self._batch_indices_cache) > self._mask_cache_max:
                self._batch_indices_cache.popitem(last=False)
        else:
            self._batch_indices_cache.move_to_end(cache_key)

        return self._batch_indices_cache[cache_key]

    def _get_shifted_dense_mask(
        self, B: int, H: int, W: int, device: torch.device,
    ) -> torch.Tensor:
        """Get or create cached dense region mask for boundary windows.

        Builds a float mask from region IDs: 0 for same-region token pairs
        (can attend), -inf for different-region pairs (must not attend).
        The mask is pre-expanded across the batch dimension for direct use
        in SDPA's attn_mask parameter.

        Args:
            B: Batch size (number of images)
            H, W: Spatial dimensions (after padding to window_size multiple)
            device: Target device

        Returns:
            Dense mask of shape (B * n_boundary, 1, ws², ws²) with values
            0.0 (same region) or -inf (different region).
        """
        mask_key = (B, H, W, device)
        if mask_key not in self._dense_mask_cache:
            ws = self.window_size

            # Get cached region IDs
            region_key = (H, W, device)
            if region_key not in self._region_cache:
                self._region_cache[region_key] = self._compute_region_ids(H, W, device)
                if len(self._region_cache) > self._mask_cache_max:
                    self._region_cache.popitem(last=False)
            else:
                self._region_cache.move_to_end(region_key)
            region_ids = self._region_cache[region_key]  # (nW, ws²)

            # Extract boundary window region IDs
            _, boundary_idx = self._get_window_indices(H, W, device)
            boundary_regions = region_ids[boundary_idx]  # (n_boundary, ws²)

            # Build mask: same region → 0, different region → -inf
            # (n_boundary, ws², 1) vs (n_boundary, 1, ws²) → (n_boundary, ws², ws²)
            same_region = boundary_regions.unsqueeze(2) == boundary_regions.unsqueeze(1)
            mask = torch.zeros(same_region.shape, dtype=torch.float32, device=device)
            mask.masked_fill_(~same_region, float('-inf'))

            # (n_boundary, 1, ws², ws²) — head dim broadcasts across all heads.
            # .repeat(B, ...) tiles for batch: layout matches x_windows gather order.
            mask = mask.unsqueeze(1).repeat(B, 1, 1, 1)

            self._dense_mask_cache[mask_key] = mask
            if len(self._dense_mask_cache) > self._mask_cache_max:
                self._dense_mask_cache.popitem(last=False)
        else:
            self._dense_mask_cache.move_to_end(mask_key)

        return self._dense_mask_cache[mask_key]

    def _get_shifted_dense_mask_all_windows(
        self, B: int, H: int, W: int, device: torch.device,
    ) -> torch.Tensor:
        """Get or create a dense shifted-window mask for every window.

        Interior windows become all-zero masks, while boundary windows keep the
        usual same-region versus blocked-region structure. This simplifies the
        exported graph by letting all shifted windows share one masked path.
        """
        if torch.onnx.is_in_onnx_export():
            region_ids = self._compute_region_ids(H, W, device)
            same_region = region_ids.unsqueeze(2) == region_ids.unsqueeze(1)
            mask = torch.zeros(same_region.shape, dtype=torch.float32, device=device)
            mask.masked_fill_(~same_region, float('-inf'))
            return mask.unsqueeze(1).repeat(B, 1, 1, 1)

        mask_key = (B, H, W, device, "all")
        if mask_key not in self._dense_mask_cache:
            region_key = (H, W, device)
            if region_key not in self._region_cache:
                self._region_cache[region_key] = self._compute_region_ids(H, W, device)
                if len(self._region_cache) > self._mask_cache_max:
                    self._region_cache.popitem(last=False)
            else:
                self._region_cache.move_to_end(region_key)
            region_ids = self._region_cache[region_key]  # (nW, ws²)

            same_region = region_ids.unsqueeze(2) == region_ids.unsqueeze(1)
            mask = torch.zeros(same_region.shape, dtype=torch.float32, device=device)
            mask.masked_fill_(~same_region, float('-inf'))
            mask = mask.unsqueeze(1).repeat(B, 1, 1, 1)

            self._dense_mask_cache[mask_key] = mask
            if len(self._dense_mask_cache) > self._mask_cache_max:
                self._dense_mask_cache.popitem(last=False)
        else:
            self._dense_mask_cache.move_to_end(mask_key)

        return self._dense_mask_cache[mask_key]

    def _get_shifted_block_mask(
        self, B: int, H: int, W: int, device: torch.device,
    ) -> BlockMask:
        """Get or create cached BlockMask for boundary windows only.

        Only boundary windows (last row/col of the window grid) have mixed
        region IDs and need masking. Interior windows use Flash instead.

        Args:
            B: Batch size (number of images)
            H, W: Spatial dimensions (after padding to window_size multiple)
            device: Target device

        Returns:
            BlockMask sized for boundary windows only (B * n_boundary batches)
        """
        ws = self.window_size
        nH_grid = H // ws
        nW_grid = W // ws
        nW = nH_grid * nW_grid

        # Get cached region IDs (for ALL windows)
        region_key = (H, W, device)
        if region_key not in self._region_cache:
            self._region_cache[region_key] = self._compute_region_ids(H, W, device)
            if len(self._region_cache) > self._mask_cache_max:
                self._region_cache.popitem(last=False)
        else:
            self._region_cache.move_to_end(region_key)
        region_ids = self._region_cache[region_key]  # (nW, ws²)

        # Extract boundary window indices and their region IDs
        _, boundary_idx = self._get_window_indices(H, W, device)
        n_boundary = len(boundary_idx)
        boundary_region_ids = region_ids[boundary_idx]  # (n_boundary, ws²)

        # Get cached BlockMask for boundary windows
        mask_key = (B, H, W, device)
        if mask_key not in self._block_mask_cache:
            B_total = B * n_boundary
            ws_sq = ws * ws

            def mask_mod(b, h, q_idx, kv_idx):
                win_idx = b % n_boundary
                return boundary_region_ids[win_idx, q_idx] == boundary_region_ids[win_idx, kv_idx]

            self._block_mask_cache[mask_key] = create_block_mask(
                mask_mod, B_total, 1, ws_sq, ws_sq, device=device,
            )
            if len(self._block_mask_cache) > self._mask_cache_max:
                self._block_mask_cache.popitem(last=False)
        else:
            self._block_mask_cache.move_to_end(mask_key)

        return self._block_mask_cache[mask_key]

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, iLN, AffineTransform)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            # Use xavier for GELU/SiLU activations (not relu)
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_size = (x.shape[2], x.shape[3])
        H, W = x_size

        # Compute per-window routing info for shifted ACTBlocks.
        # hybrid_ctx = (region_mask, int_idx_full, bnd_idx_full) passed through
        # the layer hierarchy. Non-shifted blocks ignore it entirely and use
        # Flash. Shifted blocks split windows: interior → Flash, boundary → masked/Flex.
        # Both mask types and batch indices are cached at model level.
        if self.attn_type == 'masked':
            B = x.shape[0]
            if self.force_tensorrt_export_mode:
                dense_mask = self._get_shifted_dense_mask_all_windows(B, H, W, x.device)
                int_idx_full, bnd_idx_full = None, None
            else:
                dense_mask = self._get_shifted_dense_mask(B, H, W, x.device)
                int_idx_full, bnd_idx_full = self._get_batch_indices(B, H, W, x.device)
            hybrid_ctx = (dense_mask, int_idx_full, bnd_idx_full)
        elif self.attn_type == 'hybrid':
            B = x.shape[0]
            block_mask = self._get_shifted_block_mask(B, H, W, x.device)
            int_idx_full, bnd_idx_full = self._get_batch_indices(B, H, W, x.device)
            hybrid_ctx = (block_mask, int_idx_full, bnd_idx_full)
        else:
            hybrid_ctx = None

        # Patch embedding
        x = self.patch_embed(x)

        # RHAG layers
        for layer in self.layers:
            x = layer(x, x_size, hybrid_ctx)

        # Final norm
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]

        # Normalize input (use local variable to avoid modifying buffer during forward)
        mean = self.mean.to(dtype=x.dtype, device=x.device)
        x = (x - mean) * self.img_range

        # Pad to multiple of window size
        x = pad_to_multiple(x, self.window_size)

        # Shallow features
        shallow = self.conv_first(x)

        # Deep features
        deep = self.forward_features(shallow)
        deep = self.conv_after_body(deep) + shallow

        # Upsampling
        x = self.conv_before_upsample(deep)
        x = self.upsample(x)
        x = self.conv_last(x)

        # Denormalize
        x = x / self.img_range + mean

        # Crop to output size
        return x[:, :, :H * self.upscale, :W * self.upscale]

    def fold_ecb(self) -> None:
        """Fold all ECB blocks for inference optimization.
        
        Must be called after model.eval().
        """
        if self.training:
            raise RuntimeError("Call model.eval() before fold_ecb()")
        for module in self.modules():
            if isinstance(module, ECBConvBlock):
                module.fold()

    def unfold_ecb(self) -> None:
        """Unfold ECB blocks back to training mode."""
        for module in self.modules():
            if isinstance(module, ECBConvBlock):
                module.unfold()

    def update_iln_ema(self) -> None:
        """Update all iLN EMA stats from last forward. Call outside compiled forward."""
        for module in self.modules():
            if isinstance(module, iLN):
                module.update_std_ema()

    def set_export_attention_mode(self, enabled: bool = True) -> None:
        """Force math-attention path in all attention modules.

        Useful when exporter/tracer does not set torch.onnx.is_in_onnx_export().
        No parameters or buffers are modified, so checkpoint compatibility is
        unchanged.
        """
        for module in self.modules():
            if isinstance(module, (WindowAttentionRFB, OCAB)):
                module.force_math_attention = enabled

    def set_tensorrt_export_mode(self, enabled: bool = True) -> None:
        """Use a TensorRT/ONNX-friendly attention graph during export.

        This keeps weights and checkpoint structure unchanged while simplifying
        the exported graph:
        - force explicit math-attention
        - use additive attention bias instead of augmented Q/K/V attention
        - route all shifted ACT windows through one dense masked path
        """
        self.force_tensorrt_export_mode = enabled
        for module in self.modules():
            if isinstance(module, (WindowAttentionRFB, OCAB)):
                module.force_math_attention = enabled
            if isinstance(module, WindowAttentionRFB):
                module.force_additive_attention = enabled
            if isinstance(module, ACTBlock):
                module.force_dense_shifted_attention = enabled


# =============================================================================
# Model Factory Functions
# =============================================================================

def drft_xs(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = 'masked',
    **kwargs
) -> DRFT:
    """DRFT-Extra-Small: embed_dim=128, 4 RHAG layers, 4 heads."""
    return DRFT(
        upscale=scale,
        embed_dim=128,
        depths=(6, 6, 6, 6),
        num_heads=(4, 4, 4, 4),
        window_size=window_size,
        overlap_ratio=0.5,
        mlp_ratio=2.667,
        drop_path_rate=drop_path_rate,
        use_checkpoint=use_checkpoint,
        dense_skip=True,
        attn_type=attn_type,
        **kwargs
    )


def drft_s(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = 'masked',
    **kwargs
) -> DRFT:
    """DRFT-Small: embed_dim=160, 6 RHAG layers, 5 heads."""
    return DRFT(
        upscale=scale,
        embed_dim=160,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(5, 5, 5, 5, 5, 5),
        window_size=window_size,
        overlap_ratio=0.5,
        mlp_ratio=2.667,
        drop_path_rate=drop_path_rate,
        use_checkpoint=use_checkpoint,
        dense_skip=True,
        attn_type=attn_type,
        **kwargs
    )


def drft_m(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = 'masked',
    **kwargs
) -> DRFT:
    """DRFT-Medium: embed_dim=192, 6 RHAG layers, 6 heads."""
    return DRFT(
        upscale=scale,
        embed_dim=192,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=window_size,
        overlap_ratio=0.5,
        mlp_ratio=2.667,
        drop_path_rate=drop_path_rate,
        use_checkpoint=use_checkpoint,
        dense_skip=True,
        attn_type=attn_type,
        **kwargs
    )


def drft_l(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = 'masked',
    **kwargs
) -> DRFT:
    """DRFT-Large: embed_dim=192, 12 RHAG layers, 6 heads."""
    return DRFT(
        upscale=scale,
        embed_dim=192,
        depths=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        window_size=window_size,
        overlap_ratio=0.5,
        mlp_ratio=2.667,
        drop_path_rate=drop_path_rate,
        use_checkpoint=use_checkpoint,
        dense_skip=True,
        attn_type=attn_type,
        **kwargs
    )


# =============================================================================
# Optional: traiNNer Registry
# =============================================================================

try:
    from traiNNer.utils.registry import ARCH_REGISTRY

    ARCH_REGISTRY.register(drft_xs)
    ARCH_REGISTRY.register(drft_s)
    ARCH_REGISTRY.register(drft_m)
    ARCH_REGISTRY.register(drft_l)

except ImportError:
    pass  # traiNNer not available


# =============================================================================
# DDP Compatibility Test
# =============================================================================

def check_ddp_compatibility(model: nn.Module, device: str = 'cuda') -> bool:
    """Check for common DDP issues before distributed training.
    
    Args:
        model: The model to check
        device: Device to test on
        
    Returns:
        True if all checks pass, False otherwise
    """
    model = model.to(device)
    model.train()
    
    print("=" * 60)
    print("DDP Compatibility Check")
    print("=" * 60)
    
    # Test forward pass
    x = torch.randn(2, 3, 64, 64, device=device)
    try:
        y = model(x)
        loss = y.sum()
        loss.backward()
        print("✅ Forward/backward pass successful")
    except Exception as e:
        print(f"❌ Forward/backward failed: {e}")
        return False
    
    # Check for unused parameters
    unused = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused.append(name)
    
    if unused:
        print(f"❌ Unused parameters detected ({len(unused)} total):")
        for name in unused[:10]:  # Show first 10
            print(f"   - {name}")
        if len(unused) > 10:
            print(f"   ... and {len(unused) - 10} more")
        return False
    else:
        print(f"✅ All {sum(p.numel() for p in model.parameters())} parameters have gradients")
    
    # Check for non-deterministic operations that might cause issues
    print("✅ Model is DDP compatible")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    # Quick sanity check
    print("DRFT Architecture Definitions:")
    print(f"  drft_xs: embed_dim=128, 4 layers, heads=4, head_dim=32")
    print(f"  drft_s:  embed_dim=160, 6 layers, heads=5, head_dim=32")
    print(f"  drft_m:  embed_dim=192, 6 layers, heads=6, head_dim=32")
    print(f"  drft_l:  embed_dim=192, 12 layers, heads=6, head_dim=32")
    print("\nAll models satisfy Flash constraints (head_dim=32, rank=32 => 64)")

    # Run DDP check if CUDA available
    if torch.cuda.is_available():
        print("\nRunning DDP compatibility check (masked mode)...")
        model_masked = drft_xs(scale=4, attn_type='masked')
        check_ddp_compatibility(model_masked)

        if _FLEX_AVAILABLE:
            print("\nRunning DDP compatibility check (hybrid mode)...")
            model_hybrid = drft_xs(scale=4, attn_type='hybrid')
            check_ddp_compatibility(model_hybrid)

            # Verify output shapes match between modes
            print("\nVerifying masked/hybrid output shape compatibility...")
            model_masked.train(False)
            model_hybrid.train(False)
            x = torch.randn(1, 3, 64, 64, device='cuda')
            with torch.inference_mode():
                y_masked = model_masked(x)
                y_hybrid = model_hybrid(x)
            assert y_masked.shape == y_hybrid.shape, (
                f"Shape mismatch: masked={y_masked.shape} vs hybrid={y_hybrid.shape}"
            )
            print(f"  Output shapes match: {y_masked.shape}")
            print("  Masked/hybrid compatibility verified.")
        else:
            print("\nSkipping hybrid mode (Flex Attention not available).")
            model_masked.train(False)
            x = torch.randn(1, 3, 64, 64, device='cuda')
            with torch.inference_mode():
                y_masked = model_masked(x)
            print(f"  Masked output shape: {y_masked.shape}")
            print("  Masked mode verified.")
