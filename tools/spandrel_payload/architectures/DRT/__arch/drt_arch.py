# ruff: noqa
# type: ignore
"""Dense Rotary Transformer (DRT) for single-image super-resolution.

DRT deliberately keeps DRFT's proven SISR body: full-width ACT blocks,
shifted windows, i-LN, EDBB convolution, ConvSwiGLU FFNs, OCAB placement,
dense fusion, RHAG residuals, and progressive PixelAttention reconstruction.
Only four reviewed changes are made:

* rank-factored ACT bias and dense OCAB bias are replaced by learnable mixed
  two-dimensional RoPE;
* Q/K receive FP32-statistics RMS normalization and a learnable per-head scale;
* OCAB obtains its halo from explicit reflection padding;
* the convolution branch uses its existing LayerScale as the sole learnable
  gate.  Its initial effective gain matches DRFT, but its gradient is no longer
  attenuated by a second fixed 0.01 multiplier.

The implementation reuses the canonical DRFT shell instead of forking its
deployment and cache machinery.  This keeps the two architectures isolated at
the checkpoint/registry level while preventing fixes to shared SISR machinery
from silently diverging.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...DRFT.__arch import drft_arch as _drft
from ...DRFT.__arch.drft_arch import (
    ATTN_TYPE,
    ACTBlock,
    AttentionBlocks,
    DRFT,
    OCAB,
    WindowAttentionRFB,
    _DRFTTensorRTMixedPrecision,
    _scaled_dot_product_attention_export_safe,
    window_partition,
    window_reverse,
)


DRT_CHECKPOINT_VERSION = 1


def drt_checkpoint_metadata(
    *,
    window_size: int,
    overlap_window_size: int,
    rope_base: float,
    qk_eps: float,
    iln_eps: float,
) -> dict[str, torch.Tensor]:
    """Return the persistent tensors that identify and reconstruct DRT."""

    return {
        "_drt_metadata": torch.tensor(
            [DRT_CHECKPOINT_VERSION, window_size, overlap_window_size],
            dtype=torch.int64,
        ),
        "_drt_hyperparameters": torch.tensor(
            [rope_base, qk_eps, iln_eps],
            dtype=torch.float64,
        ),
    }


def _coords_2d(
    height: int,
    width: int,
    *,
    offset_y: int = 0,
    offset_x: int = 0,
) -> torch.Tensor:
    """Return row-major ``(y, x)`` coordinates as an FP32 tensor."""

    y = torch.arange(offset_y, offset_y + height, dtype=torch.float32)
    x = torch.arange(offset_x, offset_x + width, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((yy, xx), dim=-1).reshape(-1, 2)


class MixedRoPE2D(nn.Module):
    """Learnable per-head mixed-direction two-dimensional rotary embedding.

    Each real/imaginary feature pair owns a learnable 2-D frequency vector.
    The deterministic initialization covers directions around the unit circle
    and geometrically spaced magnitudes.  Using the same frequencies for query
    and key makes their dot product depend only on coordinate displacement.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
        if rope_base <= 1.0:
            raise ValueError(f"rope_base must be greater than one, got {rope_base}")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.pairs = head_dim // 2
        self.rope_base = float(rope_base)

        pair_index = torch.arange(self.pairs, dtype=torch.float32)
        band_index = torch.div(pair_index, 2, rounding_mode="floor")
        band_count = max(1, (self.pairs + 1) // 2)
        magnitudes = self.rope_base ** (-band_index / band_count)

        head_index = torch.arange(num_heads, dtype=torch.float32).unsqueeze(1)
        # Interleave horizontal/vertical anchors and rotate them slightly per
        # head.  They are fully learnable after initialization, hence "mixed".
        angles = pair_index.unsqueeze(0) * (math.pi / 2.0) + head_index * (
            math.pi / (2.0 * max(1, num_heads))
        )
        frequencies = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
        frequencies = frequencies * magnitudes.view(1, self.pairs, 1)
        self.frequencies = nn.Parameter(frequencies)

    def cos_sin(
        self,
        coordinates: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords = coordinates.to(device=device, dtype=torch.float32)
        frequencies = self.frequencies.to(device=device, dtype=torch.float32)
        angles = torch.einsum("nd,hfd->hnf", coords, frequencies)
        return torch.cos(angles).to(dtype=dtype), torch.sin(angles).to(dtype=dtype)

    @staticmethod
    def apply_rotary(
        tensor: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, heads, tokens, head_dim = tensor.shape
        pairs = head_dim // 2
        paired = tensor.reshape(batch, heads, tokens, pairs, 2)
        real, imaginary = paired.unbind(dim=-1)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        rotated = torch.stack(
            (
                real * cos - imaginary * sin,
                real * sin + imaginary * cos,
            ),
            dim=-1,
        )
        return rotated.flatten(-2)


class _QKRoPEMixin:
    """Shared QK-RMSNorm, temperature, RoPE, and deployment freezing."""

    qk_eps: float
    head_dim: int
    num_heads: int
    logit_scale: nn.Parameter
    rope: MixedRoPE2D

    def _qk_rms_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        # Statistics stay in FP32 under AMP; the normalized tensor returns to
        # the activation dtype before attention.
        inverse_rms = torch.rsqrt(
            tensor.float().square().mean(dim=-1, keepdim=True) + self.qk_eps
        ).to(dtype=tensor.dtype)
        return tensor * inverse_rms

    def _attention_scale(self, reference: torch.Tensor) -> torch.Tensor:
        frozen = getattr(self, "_deployment_attention_scale", None)
        if frozen is not None:
            return frozen.to(device=reference.device, dtype=reference.dtype)
        maximum = math.log(100.0)
        return (
            self.logit_scale.float()
            .clamp(max=maximum)
            .exp()
            .to(
                device=reference.device,
                dtype=reference.dtype,
            )
        )

    def _freeze_attention_scale(self) -> None:
        value = self.logit_scale.detach().float().clamp(max=math.log(100.0)).exp()
        self.register_buffer(
            "_deployment_attention_scale",
            value,
            persistent=False,
        )


class DRTWindowAttention(WindowAttentionRFB, _QKRoPEMixin):
    """Full-width window attention with mixed 2-D RoPE and QK normalization."""

    def __init__(
        self,
        original: WindowAttentionRFB,
        *,
        rope_base: float,
        qk_eps: float,
    ) -> None:
        # Intentionally bypass WindowAttentionRFB.__init__: DRT has no RIB.
        nn.Module.__init__(self)
        self.dim = original.dim
        self.window_size = original.window_size
        self.num_heads = original.num_heads
        self.head_dim = original.head_dim
        self.rank = 0
        self.attn_type = original.attn_type
        self.content_scale = 1.0
        self.qkv = original.qkv
        self.proj = original.proj
        self.proj_drop = original.proj_drop
        self.attn_drop_p = original.attn_drop_p
        self.force_math_attention = False
        self.force_additive_attention = False
        self.qk_eps = float(qk_eps)

        self.rope = MixedRoPE2D(self.num_heads, self.head_dim, rope_base)
        self.logit_scale = nn.Parameter(
            torch.full((self.num_heads, 1, 1), -0.5 * math.log(self.head_dim))
        )
        self.register_buffer(
            "window_coordinates",
            _coords_2d(*self.window_size),
            persistent=False,
        )

    def neural_bias(self) -> None:
        """Compatibility hook for DRFT's selective-checkpoint container."""

        return None

    def freeze_for_export(self) -> None:
        cos, sin = self.rope.cos_sin(
            self.window_coordinates,
            dtype=torch.float32,
            device=self.logit_scale.device,
        )
        self.register_buffer("_deployment_rope_cos", cos, persistent=False)
        self.register_buffer("_deployment_rope_sin", sin, persistent=False)
        self._freeze_attention_scale()

    def _prepare_qkv(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_windows, tokens, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch_windows,
            tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query = self._qk_rms_norm(query)
        key = self._qk_rms_norm(key)

        cos = getattr(self, "_deployment_rope_cos", None)
        sin = getattr(self, "_deployment_rope_sin", None)
        if cos is None or sin is None:
            cos, sin = self.rope.cos_sin(
                self.window_coordinates,
                dtype=query.dtype,
                device=query.device,
            )
        else:
            cos = cos.to(device=query.device, dtype=query.dtype)
            sin = sin.to(device=query.device, dtype=query.dtype)
        query = self.rope.apply_rotary(query, cos, sin)
        key = self.rope.apply_rotary(key, cos, sin)
        query = query * self._attention_scale(query).unsqueeze(0)
        return query, key, value

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask=None,
    ) -> torch.Tensor:
        export_math = self.force_math_attention or torch.onnx.is_in_onnx_export()
        if self.attn_type == "hybrid" and mask is not None and not export_math:
            return _drft._compiled_flex_attention(
                query,
                key,
                value,
                block_mask=mask,
                scale=1.0,
            )
        return _scaled_dot_product_attention_export_safe(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.attn_drop_p,
            scale=1.0,
            training=self.training,
            use_export_safe_math=export_math,
        )

    def _forward_with_static_bias(
        self,
        x: torch.Tensor,
        additive_bias: torch.Tensor,
    ) -> torch.Tensor:
        # Kept for the inherited ACT deployment interface.  In DRT this
        # argument is only a shifted-window region mask, never positional bias.
        return self.forward(x, mask=additive_bias)

    def forward(
        self,
        x: torch.Tensor,
        mask=None,
        bias_factors=None,
    ) -> torch.Tensor:
        del bias_factors
        batch_windows, tokens, channels = x.shape
        query, key, value = self._prepare_qkv(x)
        output = self._attend(query, key, value, mask)
        output = output.transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.proj_drop(self.proj(output))


class DRTOCAB(OCAB, _QKRoPEMixin):
    """Bias-free overlapping cross-attention with reflected K/V halos."""

    def __init__(
        self,
        original: OCAB,
        *,
        rope_base: float,
        qk_eps: float,
    ) -> None:
        # Bypass OCAB.__init__ so no dense relative-position table is created.
        nn.Module.__init__(self)
        self.dim = original.dim
        self.input_resolution = original.input_resolution
        self.window_size = original.window_size
        self.num_heads = original.num_heads
        self.head_dim = original.head_dim
        self.overlap_win_size = original.overlap_win_size
        self.rank = 0
        self.attn_type = original.attn_type
        self.force_math_attention = False
        self._use_batch_axis_kv = False
        self.use_compact_relative_bias_training = False
        self.use_compact_unfold_training = False
        self.use_fused_qkv_windows_training = False
        self.use_window_order_projection_training = False
        self.scale = 1.0
        self.qk_eps = float(qk_eps)

        self.norm1 = original.norm1
        self.qkv = original.qkv
        self.proj = original.proj
        self.ls_attn = original.ls_attn
        self.ls_ffn = original.ls_ffn
        self.norm2 = original.norm2
        self.ffn = original.ffn

        self.halo = (self.overlap_win_size - self.window_size) // 2
        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=self.window_size,
            padding=0,
        )
        self.rope = MixedRoPE2D(self.num_heads, self.head_dim, rope_base)
        self.logit_scale = nn.Parameter(
            torch.full((self.num_heads, 1, 1), -0.5 * math.log(self.head_dim))
        )
        self.register_buffer(
            "query_coordinates",
            _coords_2d(self.window_size, self.window_size),
            persistent=False,
        )
        self.register_buffer(
            "key_coordinates",
            _coords_2d(
                self.overlap_win_size,
                self.overlap_win_size,
                offset_y=-self.halo,
                offset_x=-self.halo,
            ),
            persistent=False,
        )

    def freeze_for_export(self) -> None:
        device = self.logit_scale.device
        q_cos, q_sin = self.rope.cos_sin(
            self.query_coordinates,
            dtype=torch.float32,
            device=device,
        )
        k_cos, k_sin = self.rope.cos_sin(
            self.key_coordinates,
            dtype=torch.float32,
            device=device,
        )
        for name, value in (
            ("_deployment_q_rope_cos", q_cos),
            ("_deployment_q_rope_sin", q_sin),
            ("_deployment_k_rope_cos", k_cos),
            ("_deployment_k_rope_sin", k_sin),
        ):
            self.register_buffer(name, value, persistent=False)
        self._freeze_attention_scale()

    def _overlap_unfold(self, kv: torch.Tensor) -> torch.Tensor:
        if min(kv.shape[-2:]) <= self.halo:
            raise ValueError(
                "DRT OCAB reflection padding requires spatial dimensions "
                f"greater than halo={self.halo}, got {tuple(kv.shape[-2:])}"
            )
        padded = F.pad(
            kv,
            (self.halo, self.halo, self.halo, self.halo),
            mode="reflect",
        )
        return self.unfold(padded)

    def _rope_pair(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self._qk_rms_norm(query)
        key = self._qk_rms_norm(key)
        q_cos = getattr(self, "_deployment_q_rope_cos", None)
        q_sin = getattr(self, "_deployment_q_rope_sin", None)
        k_cos = getattr(self, "_deployment_k_rope_cos", None)
        k_sin = getattr(self, "_deployment_k_rope_sin", None)
        if q_cos is None:
            q_cos, q_sin = self.rope.cos_sin(
                self.query_coordinates,
                dtype=query.dtype,
                device=query.device,
            )
            k_cos, k_sin = self.rope.cos_sin(
                self.key_coordinates,
                dtype=key.dtype,
                device=key.device,
            )
        else:
            q_cos = q_cos.to(device=query.device, dtype=query.dtype)
            q_sin = q_sin.to(device=query.device, dtype=query.dtype)
            k_cos = k_cos.to(device=key.device, dtype=key.dtype)
            k_sin = k_sin.to(device=key.device, dtype=key.dtype)
        query = self.rope.apply_rotary(query, q_cos, q_sin)
        key = self.rope.apply_rotary(key, k_cos, k_sin)
        query = query * self._attention_scale(query).unsqueeze(0)
        return query, key

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        height, width = x_size
        batch, length, channels = x.shape
        shortcut = x

        x_norm, residual_scale = self.norm1(x)
        x_norm = x_norm.view(batch, height, width, channels)
        qkv = self.qkv(x_norm).reshape(batch, height, width, 3, channels)
        qkv = qkv.permute(3, 0, 4, 1, 2)
        query_image = qkv[0].permute(0, 2, 3, 1)
        query_windows = window_partition(query_image, self.window_size).view(
            -1,
            self.window_size * self.window_size,
            channels,
        )

        if self._use_batch_axis_kv:
            kv = qkv[1:3].reshape(2 * batch, channels, height, width)
            kv_windows = self._overlap_unfold(kv)
            window_count = kv_windows.shape[2]
            kv_windows = (
                kv_windows.view(
                    2,
                    batch,
                    channels,
                    self.overlap_win_size,
                    self.overlap_win_size,
                    window_count,
                )
                .permute(0, 1, 5, 3, 4, 2)
                .contiguous()
            )
        else:
            kv = torch.cat((qkv[1], qkv[2]), dim=1)
            kv_windows = self._overlap_unfold(kv)
            window_count = kv_windows.shape[2]
            kv_windows = (
                kv_windows.view(
                    batch,
                    2,
                    channels,
                    self.overlap_win_size,
                    self.overlap_win_size,
                    window_count,
                )
                .permute(1, 0, 5, 3, 4, 2)
                .contiguous()
            )
        kv_windows = kv_windows.view(
            2,
            batch * window_count,
            self.overlap_win_size * self.overlap_win_size,
            channels,
        )
        key_windows, value_windows = kv_windows.unbind(0)

        batch_windows, query_tokens, _ = query_windows.shape
        key_tokens = key_windows.shape[1]
        query = query_windows.reshape(
            batch_windows, query_tokens, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        key = key_windows.reshape(
            batch_windows, key_tokens, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        value = value_windows.reshape(
            batch_windows, key_tokens, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        query, key = self._rope_pair(query, key)

        use_math = self.force_math_attention or torch.onnx.is_in_onnx_export()
        output = _scaled_dot_product_attention_export_safe(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            scale=1.0,
            training=self.training,
            use_export_safe_math=use_math,
        )
        output = output.transpose(1, 2).reshape(batch_windows, query_tokens, self.dim)

        project_in_window_order = (
            self.use_window_order_projection_training
            and self.training
            and output.is_cuda
        )
        if project_in_window_order:
            output = self.proj(output)
        output = output.view(-1, self.window_size, self.window_size, self.dim)
        output = window_reverse(output, self.window_size, height, width)
        output = output.view(batch, length, self.dim)
        if not project_in_window_order:
            output = self.proj(output)

        output = self.ls_attn(output) * residual_scale
        x = shortcut + output
        ffn_shortcut = x
        x_norm2, ffn_scale = self.norm2(x)
        ffn_output = self.ffn(x_norm2, height, width)
        ffn_output = self.ls_ffn(ffn_output) * ffn_scale
        return ffn_shortcut + ffn_output


class DRT(DRFT):
    """Dense Rotary Transformer built on DRFT's proven SISR shell."""

    architecture_id = "drt_rope_qknorm_reflect_ocab_shifted_v1"

    def __init__(
        self,
        *args: Any,
        rope_base: float = 10_000.0,
        qk_eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rope_base = float(rope_base)
        self.qk_eps = float(qk_eps)

        first_block = self.layers[0].residual_group.blocks[0]
        metadata = drt_checkpoint_metadata(
            window_size=int(self.window_size),
            overlap_window_size=int(self.overlap_window_size),
            rope_base=self.rope_base,
            qk_eps=self.qk_eps,
            iln_eps=float(first_block.norm1.eps),
        )
        for name, value in metadata.items():
            self.register_buffer(name, value, persistent=True)

        for group in self.layers:
            residual_group = group.residual_group
            if not isinstance(residual_group, AttentionBlocks):
                raise RuntimeError("DRT requires canonical AttentionBlocks")
            for block in residual_group.blocks:
                if not isinstance(block, ACTBlock):
                    raise RuntimeError("DRT requires canonical ACTBlock modules")
                if not isinstance(block.attn, WindowAttentionRFB):
                    raise RuntimeError("DRT requires full-width DRFT attention")
                block.attn = DRTWindowAttention(
                    block.attn,
                    rope_base=self.rope_base,
                    qk_eps=self.qk_eps,
                )
                # Merge LayerScale and fixed conv_scale into one trainable gate.
                # Preserve the exact initial effective gain while removing the
                # 0.01 multiplier from the gate's gradient.
                legacy_conv_scale = float(block.conv_scale)
                with torch.no_grad():
                    block.ls_conv.gamma.mul_(legacy_conv_scale)
                block.conv_scale = 1.0
            residual_group.ocab = DRTOCAB(
                residual_group.ocab,
                rope_base=self.rope_base,
                qk_eps=self.qk_eps,
            )

        self.conv_gain = 1.0
        self.compile_cache_key_tag = (
            f"{self.compile_cache_key_tag}_{self.architecture_id}"
        )

    def set_conv_gain(self, gain: float) -> None:
        """Set the non-learned inference texture/detail gain."""

        if not math.isfinite(gain) or gain < 0.0:
            raise ValueError(f"conv_gain must be finite and non-negative, got {gain}")
        self.conv_gain = float(gain)
        for module in self.modules():
            if isinstance(module, ACTBlock):
                module.conv_scale = self.conv_gain

    @torch.no_grad()
    def prepare_for_compile(
        self,
        input_shape: tuple[int, int, int, int] | None = None,
        input_dtype: torch.dtype | None = None,
    ) -> None:
        del input_dtype
        device = self.mean.device
        if self.training and device.type == "cuda":
            for module in self.modules():
                if isinstance(module, AttentionBlocks):
                    module.use_compiled_checkpoint_policy = True
                elif isinstance(module, DRTOCAB):
                    module._use_batch_axis_kv = True
                    module.use_window_order_projection_training = True
                    # Canonical OCAB fused ops encode zero padding and dense
                    # relative bias, so they are intentionally inapplicable.
                    module.use_compact_relative_bias_training = False
                    module.use_compact_unfold_training = False
                    module.use_fused_qkv_windows_training = False

        if input_shape is None:
            return
        batch, _, height, width = input_shape
        padded_height = math.ceil(height / self.window_size) * self.window_size
        padded_width = math.ceil(width / self.window_size) * self.window_size
        if self.attn_type == "masked":
            self._get_shifted_dense_mask(batch, padded_height, padded_width, device)
            self._get_batch_indices(batch, padded_height, padded_width, device)
        elif self.attn_type == "hybrid":
            self._get_shifted_block_mask(batch, padded_height, padded_width, device)
            self._get_batch_indices(batch, padded_height, padded_width, device)

    def forward_features(
        self,
        x: torch.Tensor,
        return_distillation_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x_size = (x.shape[2], x.shape[3])
        height, width = x_size
        distillation_features: list[torch.Tensor] = []

        if self.force_tensorrt_export_mode:
            # Boundary patterns depend only on shifted-window geometry.  A
            # canonical 2x2 grid keeps them constant while H/W remain symbolic.
            canonical = 2 * self.window_size
            boundary_masks = self._get_shifted_export_boundary_masks(
                canonical, canonical, x.device
            )
            hybrid_ctx = (boundary_masks, None, None)
        elif self.attn_type == "masked":
            batch = x.shape[0]
            dense_mask = self._get_shifted_dense_mask(batch, height, width, x.device)
            interior, boundary = self._get_batch_indices(batch, height, width, x.device)
            hybrid_ctx = (dense_mask, interior, boundary)
        elif self.attn_type == "hybrid":
            batch = x.shape[0]
            block_mask = self._get_shifted_block_mask(batch, height, width, x.device)
            interior, boundary = self._get_batch_indices(batch, height, width, x.device)
            hybrid_ctx = (block_mask, interior, boundary)
        else:
            hybrid_ctx = None

        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x, x_size, hybrid_ctx)
            if return_distillation_features:
                distillation_features.append(
                    x.transpose(1, 2).reshape(x.shape[0], self.embed_dim, height, width)
                )
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        if return_distillation_features:
            return x, tuple(distillation_features)
        return x

    def prepare_for_onnx_export(
        self,
        input_shape: Sequence[int],
        *,
        clone: bool = True,
        dynamic_spatial: bool = False,
        precision: str = "native",
    ) -> nn.Module:
        """Create a folded, bias-free DRT deployment graph."""

        if precision not in ("native", "tensorrt_mixed"):
            raise ValueError(f"unsupported DRT ONNX precision: {precision}")
        shape = tuple(int(value) for value in input_shape)
        if len(shape) != 4 or min(shape) < 1:
            raise ValueError(f"expected positive NCHW deployment shape, got {shape}")
        if shape[1] != int(self.mean.shape[1]):
            raise ValueError(
                f"expected {int(self.mean.shape[1])} channels, got {shape[1]}"
            )
        feature_height = math.ceil(shape[2] / self.window_size) * self.window_size
        feature_width = math.ceil(shape[3] / self.window_size) * self.window_size
        if min(feature_height, feature_width) < 2 * self.window_size:
            raise ValueError(
                "shifted-window export requires at least a 2x2 window grid"
            )

        deployment = copy.deepcopy(self) if clone else self
        deployment.eval()
        deployment.fold_reparam_conv()
        for module in deployment.modules():
            if isinstance(module, (DRTWindowAttention, DRTOCAB)):
                module.freeze_for_export()
        deployment._onnx_deployment_input_shape = shape
        deployment._onnx_deployment_dynamic_spatial = bool(dynamic_spatial)
        # DRT.forward_features supplies the canonical shared masks; do not
        # enable canonical DRFT's RIB-specific static split buffers.
        deployment._onnx_deployment_feature_shape = None
        deployment.set_tensorrt_export_mode(True)
        if dynamic_spatial:
            for module in deployment.modules():
                if isinstance(module, (DRTWindowAttention, DRTOCAB)):
                    module.force_math_attention = True
        deployment.requires_grad_(False)
        deployment.onnx_deployment_inventory = {
            "architecture": self.architecture_id,
            "input_shape": shape,
            "dynamic_spatial": bool(dynamic_spatial),
            "attention": "mixed_rope_qk_rmsnorm",
            "ocab_halo": "reflect",
            "shifted_windows": True,
            "conv_gate": "single_layerscale",
            "precision": precision,
        }
        if precision == "tensorrt_mixed":
            return _DRFTTensorRTMixedPrecision(deployment)
        return deployment


def _resolve_overlap(window_size: int, overlap_window_size: int | None) -> int:
    canonical = window_size + window_size // 2
    if overlap_window_size is not None and overlap_window_size != canonical:
        raise ValueError(
            f"DRT requires overlap_window_size={canonical}, got {overlap_window_size}"
        )
    return canonical


def _build_drt(
    *,
    scale: int,
    embed_dim: int,
    groups: int,
    depth: int,
    full_heads: int,
    use_checkpoint: bool,
    window_size: int,
    overlap_window_size: int | None,
    drop_path_rate: float,
    attn_type: ATTN_TYPE,
    iln_eps: float,
    dense_skip: bool = True,
    reconstruction: str = "progressive",
    rhag_layer_scale_init: float | None = None,
    rope_base: float = 10_000.0,
    qk_eps: float = 1e-6,
    **kwargs: Any,
) -> DRT:
    if embed_dim % full_heads != 0 or embed_dim // full_heads != 32:
        raise ValueError("DRT factories require head_dim=32")
    overlap = _resolve_overlap(window_size, overlap_window_size)
    return DRT(
        upscale=scale,
        embed_dim=embed_dim,
        depths=(depth,) * groups,
        num_heads=(full_heads,) * groups,
        unshifted_num_heads=(full_heads,) * groups,
        unshifted_attention_dim=(embed_dim,) * groups,
        window_size=window_size,
        overlap_window_size=overlap,
        mlp_ratio=2.667,
        drop_path_rate=drop_path_rate,
        rhag_layer_scale_init=rhag_layer_scale_init,
        use_checkpoint=use_checkpoint,
        dense_skip=dense_skip,
        attn_type=attn_type,
        rank=32,
        reconstruction=reconstruction,
        full_width_unshifted=True,
        iln_eps=iln_eps,
        rope_base=rope_base,
        qk_eps=qk_eps,
        **kwargs,
    )


def drt_light(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = "masked",
    iln_eps: float = 1e-4,
    **kwargs: Any,
) -> DRT:
    """DRT Light: C96/G2 with three 32-D heads."""

    return _build_drt(
        scale=scale,
        embed_dim=96,
        groups=2,
        depth=6,
        full_heads=3,
        use_checkpoint=use_checkpoint,
        window_size=window_size,
        overlap_window_size=overlap_window_size,
        drop_path_rate=drop_path_rate,
        attn_type=attn_type,
        iln_eps=iln_eps,
        **kwargs,
    )


def drt_xs(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = "masked",
    iln_eps: float = 1e-4,
    **kwargs: Any,
) -> DRT:
    return _build_drt(
        scale=scale,
        embed_dim=128,
        groups=4,
        depth=6,
        full_heads=4,
        use_checkpoint=use_checkpoint,
        window_size=window_size,
        overlap_window_size=overlap_window_size,
        drop_path_rate=drop_path_rate,
        attn_type=attn_type,
        iln_eps=iln_eps,
        **kwargs,
    )


def drt_s(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = "masked",
    iln_eps: float = 1e-4,
    **kwargs: Any,
) -> DRT:
    return _build_drt(
        scale=scale,
        embed_dim=160,
        groups=6,
        depth=6,
        full_heads=5,
        use_checkpoint=use_checkpoint,
        window_size=window_size,
        overlap_window_size=overlap_window_size,
        drop_path_rate=drop_path_rate,
        attn_type=attn_type,
        iln_eps=iln_eps,
        **kwargs,
    )


def drt_m(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = "masked",
    iln_eps: float = 1e-4,
    **kwargs: Any,
) -> DRT:
    return _build_drt(
        scale=scale,
        embed_dim=192,
        groups=8,
        depth=6,
        full_heads=6,
        use_checkpoint=use_checkpoint,
        window_size=window_size,
        overlap_window_size=overlap_window_size,
        drop_path_rate=drop_path_rate,
        attn_type=attn_type,
        iln_eps=iln_eps,
        **kwargs,
    )


def drt_l(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = "masked",
    iln_eps: float = 1e-4,
    **kwargs: Any,
) -> DRT:
    return _build_drt(
        scale=scale,
        embed_dim=224,
        groups=10,
        depth=6,
        full_heads=7,
        use_checkpoint=use_checkpoint,
        window_size=window_size,
        overlap_window_size=overlap_window_size,
        drop_path_rate=drop_path_rate,
        attn_type=attn_type,
        iln_eps=iln_eps,
        **kwargs,
    )


def drt_xl(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = "masked",
    iln_eps: float = 1e-4,
    rhag_layer_scale_init: float = 1e-4,
    **kwargs: Any,
) -> DRT:
    return _build_drt(
        scale=scale,
        embed_dim=256,
        groups=14,
        depth=6,
        full_heads=8,
        use_checkpoint=use_checkpoint,
        window_size=window_size,
        overlap_window_size=overlap_window_size,
        drop_path_rate=drop_path_rate,
        attn_type=attn_type,
        iln_eps=iln_eps,
        rhag_layer_scale_init=rhag_layer_scale_init,
        **kwargs,
    )
