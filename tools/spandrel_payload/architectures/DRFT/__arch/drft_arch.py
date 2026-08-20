# ruff: noqa
# type: ignore
"""Dense rank-factored transformer for single-image super-resolution.

The residual trunk and every local-attention block stay at full channel
width. Position bias is rank-factored for local attention and dense for the
original half-window-overlap cross-attention path.

The implementation uses image-wide i-LN throughout and supports masked or
hybrid shifted-window routing, reparameterizable convolutions, gradient
checkpointing, and export-friendly attention paths.

For ONNX deployment, call
``model.prepare_for_onnx_export((batch, channels, height, width),
dynamic_spatial=True)`` and export the returned inference clone from the
canonical 96x96 example with dynamic H/W axes. Use
``precision="tensorrt_mixed"`` for FP32 I/O, a BF16 body, and the FP16
reconstruction tail.  traiNNer's ONNX converter selects that policy
automatically for BF16 DRFT exports. TensorRT should specialize the portable
ONNX to one fixed min/opt/max spatial shape per engine.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections import OrderedDict
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

_NATIVE_FLEX_HOP = getattr(torch.ops.higher_order, "flex_attention", None)


def _is_checkpointable_attention_op(op: Any) -> bool:
    return (
        op is torch.ops.aten._scaled_dot_product_flash_attention.default
        or (_NATIVE_FLEX_HOP is not None and op is _NATIVE_FLEX_HOP)
    )


def _selective_flash_qkv_policy(
    _ctx: Any,
    op: Any,
    *args: Any,
    **_kwargs: Any,
) -> CheckpointPolicy:
    if _is_checkpointable_attention_op(op):
        return CheckpointPolicy.MUST_SAVE
    if op is torch.ops.aten.addmm.default and len(args) >= 3:
        matrix = args[2]
        if (
            isinstance(matrix, torch.Tensor)
            and matrix.ndim == 2
            and matrix.shape[1] == 3 * matrix.shape[0]
        ):
            return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def _selective_flash_qkv_norm_stats_policy(
    ctx: Any,
    op: Any,
    *args: Any,
    **kwargs: Any,
) -> CheckpointPolicy:
    base_policy = _selective_flash_qkv_policy(ctx, op, *args, **kwargs)
    if base_policy is CheckpointPolicy.MUST_SAVE:
        return base_policy
    return _selective_small_reduction_policy(op, *args)


def _selective_small_reduction_policy(
    op: Any,
    *args: Any,
) -> CheckpointPolicy:
    if op is torch.ops.aten.mean.dim and len(args) >= 2:
        source = args[0]
        dimensions = tuple(args[1])
        if (
            isinstance(source, torch.Tensor)
            and source.ndim == 3
            and dimensions == (1, 2)
        ):
            # Each image-wide i-LN statistic is Bx1x1. Retaining it avoids
            # rerunning the expensive spatial/channel reduction while adding
            # negligible checkpoint memory.
            return CheckpointPolicy.MUST_SAVE
        if (
            isinstance(source, torch.Tensor)
            and source.ndim == 4
            and dimensions in ((2, 3), (-2, -1))
        ):
            # SwiGLU channel attention consumes only BxCx1x1 pooled values.
            # Cache that tiny reduction output, not its full feature input.
            return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE

_FLASH_QKV_NORM_STATS_SELECTIVE_CHECKPOINT_CONTEXT = partial(
    create_selective_checkpoint_contexts,
    _selective_flash_qkv_norm_stats_policy,
)
try:
    from traiNNer.ops.drft_ocab_relative_bias import (
        ocab_bias_dtype_code,
        ocab_relative_bias,
    )
    _OCAB_COMPACT_BIAS_AVAILABLE = True
except ImportError:
    _OCAB_COMPACT_BIAS_AVAILABLE = False

try:
    from traiNNer.ops.drft_ocab_unfold import ocab_unfold2d

    _OCAB_COMPACT_UNFOLD_AVAILABLE = True
except ImportError:
    _OCAB_COMPACT_UNFOLD_AVAILABLE = False

try:
    from traiNNer.ops.drft_ocab_qkv_windows import ocab_qkv_windows

    _OCAB_FUSED_QKV_WINDOWS_AVAILABLE = True
except ImportError:
    _OCAB_FUSED_QKV_WINDOWS_AVAILABLE = False

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
ONNX_DEPLOYMENT_PRECISION = Literal['native', 'tensorrt_mixed']
RECONSTRUCTION_TYPE = Literal['progressive', 'direct']

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


class ImageLayerNorm(nn.Module):
    """Paper i-LN with raw FP32 statistics in training and inference."""

    def __init__(self, dim: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # AMP inputs still use FP32 accumulation for the image-wide statistics.
        # Cast the normalized result and raw scale back to the activation dtype
        # so i-LN does not turn the rest of the mixed-precision trunk into an
        # FP32 island. Every finite scale is preserved: NaN/Inf deliberately
        # propagates to the trainer's existing whole-step rejection checks.
        stats_x = x.float()
        mean = stats_x.mean(dim=(1, 2), keepdim=True)
        centered = stats_x - mean
        variance = (centered * centered).mean(dim=(1, 2), keepdim=True)
        std = torch.sqrt(variance + self.eps)
        normalized = (centered / std).to(dtype=x.dtype)
        scale = std.to(dtype=x.dtype)
        return normalized * self.weight + self.bias, scale


class AffineTransform(nn.Module):
    """Learned affine used where a final normalization has no residual pair."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias


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

    The fallback path avoids emitting an ONNX Attention op by using explicit
    matmul/softmax/matmul. Matmuls and softmax remain in the attention tensor
    dtype, so TensorRT mixed-precision export keeps this path in BF16.
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
    # Softmax is numerically stable without clipping finite logits. Keep it in
    # the attention dtype; clamping would alter the trained attention weights.
    attn = torch.softmax(attn, dim=-1).to(dtype=v.dtype)
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


# Keep Flex Attention behind its own compiled boundary.  This is the measured
# pre-fusion training arrangement: outer DRFT compilation may graph-break at
# this reusable kernel instead of absorbing every shifted-boundary call into a
# single monolithic graph.
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


def rewrite_onnx_for_tensorrt_plugins(
    source: str | Path,
    output: str | Path,
    *,
    strategy: Literal[
        "optimized", "compact_bias", "fused_attention"
    ] = "optimized",
) -> dict[str, Any]:
    """Create a TensorRT-specialized ONNX from a portable DRFT ONNX.

    The portable export deliberately materializes OCAB's additive relative
    position bias so ordinary ONNX runtimes can execute the graph. TensorRT
    otherwise serializes the dense ``[1, heads, Q^2, KV^2]`` tensor once per
    OCAB. This deployment-only rewrite restores the exact learned
    ``(Q + KV - 1)^2`` table for the admitted Q32/K40 and Q32/K48 geometries.

    ``optimized`` is the default fast path. It retains TensorRT's native fused
    attention and applies the admitted DRFT deployment rewrites in one pass:
    compact OCAB relative bias, blockwise dense-fusion MatMuls, exact image-LN
    plugins, and byte-preserving token-major OCAB Q/K/V packing.

    ``compact_bias`` applies only ``DrftOCABRelativeBias_TRT`` while retaining
    TensorRT's native fused attention. ``fused_attention`` replaces only
    ``QK -> bias -> softmax -> PV`` with ``DrftOCABAttention_TRT`` and is kept
    as a lower-engine-memory fallback. All projections, halo extraction,
    window partition/reversal, residuals, and reconstruction remain normal
    ONNX ops in both strategies.

    The custom node requires the separately distributed DRFT TensorRT plugin
    library. The source ONNX is never modified and remains the portable
    fallback.
    """
    try:
        import numpy as np
        import onnx
        from onnx import helper
    except ImportError as error:
        raise RuntimeError(
            "ONNX plugin rewriting requires the optional onnx and numpy packages"
        ) from error

    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path == output_path:
        raise ValueError("Plugin ONNX output must differ from the portable source")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_path}")
    if strategy not in {"optimized", "compact_bias", "fused_attention"}:
        raise ValueError(f"Unknown TensorRT plugin strategy: {strategy}")
    bias_strategy = "compact_bias" if strategy == "optimized" else strategy

    model = onnx.load(str(source_path), load_external_data=True)
    graph = model.graph
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    producer_indices = {
        name: index
        for index, node in enumerate(graph.node)
        for name in node.output
    }
    constant_tensors: dict[str, tuple[int, Any, Any]] = {}
    for node_index, node in enumerate(graph.node):
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        tensor_attributes = [
            attribute.t
            for attribute in node.attribute
            if attribute.type == onnx.AttributeProto.TENSOR
        ]
        if len(tensor_attributes) == 1:
            constant_tensors[node.output[0]] = (
                node_index,
                node,
                tensor_attributes[0],
            )
    consumers: dict[str, list[Any]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    removed_node_indices: set[int] = set()
    replacements: dict[int, Any] = {}
    insertions: dict[int, list[Any]] = {}
    removed_initializers: set[str] = set()
    compact_initializers: list[Any] = []
    reports: list[dict[str, Any]] = []
    relative_geometry: dict[
        tuple[int, int], tuple[int, Any, Any]
    ] = {}

    for add_index, add_node in enumerate(graph.node):
        if add_node.op_type != "Add":
            continue
        bias_name = None
        bias_tensor = None
        bias_constant_index = None
        bias_source = None
        for name in add_node.input:
            initializer = initializers.get(name)
            constant = constant_tensors.get(name)
            candidate = initializer if initializer is not None else (
                constant[2] if constant is not None else None
            )
            if candidate is None:
                continue
            dims = list(candidate.dims)
            named_deployment_bias = name.endswith(
                "._deployment_relative_position_bias"
            )
            q_tokens_candidate = int(dims[-2]) if len(dims) == 4 else 0
            kv_tokens_candidate = int(dims[-1]) if len(dims) == 4 else 0
            q_window_candidate = math.isqrt(q_tokens_candidate)
            kv_window_candidate = math.isqrt(kv_tokens_candidate)
            admitted_geometry = (
                q_window_candidate == 32
                and q_window_candidate * q_window_candidate == q_tokens_candidate
                and kv_window_candidate in {40, 48}
                and kv_window_candidate * kv_window_candidate == kv_tokens_candidate
            )
            exact_dynamic_constant = (
                constant is not None
                and dims[:1] == [1]
                and admitted_geometry
                and "/ocab/" in constant[1].name
            )
            if named_deployment_bias or exact_dynamic_constant:
                bias_name = name
                bias_tensor = candidate
                bias_constant_index = constant[0] if constant is not None else None
                bias_source = "constant" if constant is not None else "initializer"
                break
        if bias_name is None:
            continue
        assert bias_tensor is not None
        bias_dims = list(bias_tensor.dims)
        q_tokens = int(bias_dims[2]) if len(bias_dims) == 4 else 0
        kv_tokens = int(bias_dims[3]) if len(bias_dims) == 4 else 0
        q_window = math.isqrt(q_tokens)
        kv_window = math.isqrt(kv_tokens)
        if (
            len(bias_dims) != 4
            or bias_dims[0] != 1
            or q_window != 32
            or q_window * q_window != q_tokens
            or kv_window not in {40, 48}
            or kv_window * kv_window != kv_tokens
        ):
            raise RuntimeError(
                f"Unexpected OCAB bias shape for {bias_name}: {bias_dims}"
            )
        if bias_strategy == "fused_attention" and kv_window != 40:
            raise RuntimeError(
                "The fused OCAB attention plugin admits only Q32/K40; "
                "use compact_bias for exact Q32/K48 OCAB"
            )
        geometry_key = (q_window, kv_window)
        geometry = relative_geometry.get(geometry_key)
        if geometry is None:
            bias_span = q_window + kv_window - 1
            qy = np.repeat(np.arange(q_window, dtype=np.int64), q_window)
            qx = np.tile(np.arange(q_window, dtype=np.int64), q_window)
            ky = np.repeat(np.arange(kv_window, dtype=np.int64), kv_window)
            kx = np.tile(np.arange(kv_window, dtype=np.int64), kv_window)
            relative_index = (
                (ky[None, :] - qy[:, None] + q_window - 1) * bias_span
                + (kx[None, :] - qx[:, None] + q_window - 1)
            )
            unique_index, first_flat_position = np.unique(
                relative_index.reshape(-1), return_index=True
            )
            expected_index = np.arange(bias_span * bias_span, dtype=np.int64)
            if not np.array_equal(unique_index, expected_index):
                raise RuntimeError(
                    "OCAB relative-position index is not a complete table cover"
                )
            geometry = (bias_span, relative_index, first_flat_position)
            relative_geometry[geometry_key] = geometry
        bias_span, relative_index, first_flat_position = geometry
        heads = int(bias_dims[1])
        if not bias_tensor.raw_data:
            raise RuntimeError(
                f"OCAB bias must use raw tensor storage for bit-exact compression: {bias_name}"
            )
        if bias_tensor.data_type == onnx.TensorProto.BFLOAT16:
            storage_dtype = np.dtype("<u2")
        elif bias_tensor.data_type == onnx.TensorProto.FLOAT16:
            storage_dtype = np.dtype("<u2")
        elif bias_tensor.data_type == onnx.TensorProto.FLOAT:
            storage_dtype = np.dtype("<u4")
        else:
            raise RuntimeError(
                f"Unsupported OCAB bias dtype {bias_tensor.data_type} for {bias_name}"
            )
        dense_bits = np.frombuffer(bias_tensor.raw_data, dtype=storage_dtype).reshape(
            1, heads, q_tokens, kv_tokens
        )
        table_bits = np.empty((bias_span * bias_span, heads), dtype=storage_dtype)
        for head in range(heads):
            dense_flat = dense_bits[0, head].reshape(-1)
            table_bits[:, head] = dense_flat[first_flat_position]
            if not np.array_equal(table_bits[:, head][relative_index], dense_bits[0, head]):
                raise RuntimeError(
                    f"Dense OCAB bias is not an exact table expansion: {bias_name}, head {head}"
                )

        score_name = next(name for name in add_node.input if name != bias_name)
        score_index = producer_indices.get(score_name)
        if score_index is None:
            raise RuntimeError(f"OCAB bias input has no producer: {bias_name}")
        score_node = graph.node[score_index]
        if bias_strategy == "compact_bias":
            score_is_qk = score_node.op_type == "MatMul"
            if score_node.op_type == "Mul":
                score_is_qk = any(
                    (
                        producer_indices.get(name) is not None
                        and graph.node[producer_indices[name]].op_type == "MatMul"
                    )
                    for name in score_node.input
                )
            if not score_is_qk:
                raise RuntimeError(
                    f"OCAB bias input is not produced by QK or scaled QK: {bias_name}"
                )
            softmax_node = None
            softmax_index = None
            pv_node = None
            pv_index = None
        else:
            if score_node.op_type != "MatMul":
                raise RuntimeError(
                    "Fused OCAB plugin requires the direct QK graph; "
                    f"got {score_node.op_type} for {bias_name}"
                )
            add_users = consumers.get(add_node.output[0], [])
            if len(add_users) != 1 or add_users[0].op_type != "Softmax":
                raise RuntimeError(
                    f"OCAB Add does not feed exactly one Softmax: {bias_name}"
                )
            softmax_node = add_users[0]
            softmax_index = producer_indices[softmax_node.output[0]]
            softmax_users = consumers.get(softmax_node.output[0], [])
            if len(softmax_users) != 1 or softmax_users[0].op_type != "MatMul":
                raise RuntimeError(
                    f"OCAB Softmax does not feed exactly one PV MatMul: {bias_name}"
                )
            pv_node = softmax_users[0]
            pv_index = producer_indices[pv_node.output[0]]
            if pv_node.input[0] != softmax_node.output[0]:
                raise RuntimeError(
                    f"OCAB probability tensor is not PV input zero: {bias_name}"
                )

        table_name = bias_name.removesuffix(
            "_deployment_relative_position_bias"
        ) + "tensorrt_relative_position_bias_table"
        table_tensor = onnx.TensorProto()
        table_tensor.name = table_name
        table_tensor.data_type = bias_tensor.data_type
        table_tensor.dims.extend([bias_span * bias_span, heads])
        table_tensor.raw_data = table_bits.tobytes(order="C")
        compact_initializers.append(table_tensor)

        if bias_strategy == "compact_bias":
            plugin_op = (
                "DrftOCABRelativeBias_TRT"
                if kv_window == 40
                else "DrftOCABRelativeBiasSafe_TRT"
            )
            plugin_node = helper.make_node(
                plugin_op,
                [table_name],
                [bias_name],
                name=add_node.name.replace("node_Add", "drft_ocab_relative_bias"),
                plugin_version="1",
                plugin_namespace="",
                q_window_size=q_window,
                kv_window_size=kv_window,
            )
            insertion_index = (
                bias_constant_index
                if bias_constant_index is not None
                else add_index
            )
            insertions.setdefault(insertion_index, []).append(plugin_node)
            if bias_constant_index is not None:
                removed_node_indices.add(bias_constant_index)
            plugin_output = bias_name
        else:
            plugin_op = "DrftOCABAttention_TRT"
            assert pv_node is not None
            assert softmax_index is not None
            assert pv_index is not None
            plugin_node = helper.make_node(
                "DrftOCABAttention_TRT",
                [score_node.input[0], score_node.input[1], pv_node.input[1], table_name],
                list(pv_node.output),
                name=add_node.name.replace("node_Add", "drft_ocab_attention"),
                plugin_version="1",
                plugin_namespace="",
                q_window_size=q_window,
                kv_window_size=kv_window,
            )
            replacements[score_index] = plugin_node
            removed_node_indices.update(
                (score_index, add_index, softmax_index, pv_index)
            )
            if bias_constant_index is not None:
                removed_node_indices.add(bias_constant_index)
            plugin_output = pv_node.output[0]
        if bias_source == "initializer":
            removed_initializers.add(bias_name)
        reports.append(
            {
                "bias": bias_name,
                "source": bias_source,
                "heads": heads,
                "q_window_size": q_window,
                "kv_window_size": kv_window,
                "dense_bytes": len(bias_tensor.raw_data),
                "table_bytes": len(table_tensor.raw_data),
                "output": plugin_output,
                "plugin": plugin_op,
            }
        )

    if not reports:
        raise RuntimeError("No dense OCAB attention pattern was found")

    original_nodes = list(graph.node)
    del graph.node[:]
    for index, node in enumerate(original_nodes):
        graph.node.extend(insertions.get(index, []))
        replacement = replacements.get(index)
        if replacement is not None:
            graph.node.append(replacement)
        if index not in removed_node_indices:
            graph.node.append(node)

    retained_initializers = [
        tensor for tensor in graph.initializer if tensor.name not in removed_initializers
    ]
    del graph.initializer[:]
    graph.initializer.extend(retained_initializers)
    graph.initializer.extend(compact_initializers)

    optimization_reports: dict[str, Any] | None = None
    if strategy == "optimized":
        from traiNNer.archs.drft_tensorrt_rewrite import (
            decompose_dense_fusion,
            fuse_image_layer_norm,
            fuse_ocab_qkv_pack,
        )

        dense_report = decompose_dense_fusion(model)
        image_ln_report = fuse_image_layer_norm(
            model,
            plugin_op="DrftImageLayerNorm_TRT",
        )
        channels = {int(item["channels"]) for item in image_ln_report}
        heads = {int(item["heads"]) for item in reports}
        geometries = {
            (int(item["q_window_size"]), int(item["kv_window_size"]))
            for item in reports
        }
        if len(channels) != 1 or len(heads) != 1 or len(geometries) != 1:
            raise RuntimeError(
                "Optimized DRFT TensorRT export requires one shared channel, "
                "head, and OCAB geometry across all residual groups"
            )
        q_window, kv_window = next(iter(geometries))
        qkv_report = fuse_ocab_qkv_pack(
            model,
            channels=next(iter(channels)),
            heads=next(iter(heads)),
            query_window=q_window,
            key_window=kv_window,
            token_major=True,
        )
        optimization_reports = {
            "dense_fusion": dense_report,
            "image_layer_norm": image_ln_report,
            "ocab_qkv_pack": qkv_report,
        }

    plugin_names = sorted({item["plugin"] for item in reports})
    if len(plugin_names) != 1:
        raise RuntimeError(
            f"A plugin ONNX may use only one admitted OCAB geometry: {plugin_names}"
        )
    plugin_name = plugin_names[0]
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata.update(
        {
            "drft.tensorrt_plugin_mode": f"{plugin_name} v1",
            "drft.tensorrt_plugin_strategy": strategy,
            "drft.tensorrt_plugin_ocab_count": str(len(reports)),
            "drft.portable_source": str(source_path),
        }
    )
    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    external_data_path = output_path.with_name(f"{output_path.name}.data")
    if external_data_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite plugin ONNX external data: {external_data_path}"
        )
    raw_initializer_bytes = sum(
        len(tensor.raw_data) for tensor in model.graph.initializer
    )
    raw_constant_bytes = sum(
        len(attribute.t.raw_data)
        for node in model.graph.node
        for attribute in node.attribute
        if attribute.type == onnx.AttributeProto.TENSOR
    )
    raw_tensor_bytes = raw_initializer_bytes + raw_constant_bytes
    uses_external_data = raw_tensor_bytes >= 1536 * 1024 * 1024
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=uses_external_data,
        all_tensors_to_one_file=uses_external_data,
        location=external_data_path.name if uses_external_data else None,
        size_threshold=1024,
        convert_attribute=uses_external_data,
    )
    plugin_counts: dict[str, int] = {}
    for node in model.graph.node:
        if node.op_type.startswith("Drft") and node.op_type.endswith("_TRT"):
            plugin_counts[node.op_type] = plugin_counts.get(node.op_type, 0) + 1

    return {
        "source": str(source_path),
        "output": str(output_path),
        "plugin": plugin_name,
        "plugin_version": "1",
        "strategy": strategy,
        "plugin_counts": dict(sorted(plugin_counts.items())),
        "optimization_reports": optimization_reports,
        "ocab_count": len(reports),
        "dense_bias_bytes": sum(item["dense_bytes"] for item in reports),
        "compact_table_bytes": sum(item["table_bytes"] for item in reports),
        "geometries": sorted(
            {
                (item["q_window_size"], item["kv_window_size"])
                for item in reports
            }
        ),
        "external_data": str(external_data_path) if uses_external_data else None,
        "raw_initializer_bytes": raw_initializer_bytes,
        "raw_constant_bytes": raw_constant_bytes,
        "raw_tensor_bytes": raw_tensor_bytes,
        "instances": reports,
    }


def _get_shared_relative_position_index(
    q_window_size: tuple[int, int],
    k_window_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Get a shared OCAB relative-position index cache for the current device."""
    if torch.onnx.is_in_onnx_export():
        return _build_relative_position_index(q_window_size, k_window_size, device)
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
        self.force_tensorrt_export_mode = False

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
        export_mode = self.force_tensorrt_export_mode or torch.onnx.is_in_onnx_export()
        if not self.training and not export_mode and self._cached_factors is not None:
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

        if not self.training and not export_mode:
            self._cached_factors = (bq, bk)
        return bq, bk

    def clear_cache(self) -> None:
        self._cached_factors = None

    def train(self, mode: bool = True) -> "RankFactoredNeuralBias":
        self.clear_cache()
        return super().train(mode)


class _StaticRankFactoredBias(nn.Module):
    """Precomputed rank factors used by a shape-static deployment clone."""

    def __init__(self, bq: torch.Tensor, bk: torch.Tensor) -> None:
        super().__init__()
        if bq.ndim != 3 or bq.shape != bk.shape:
            raise ValueError(
                f"rank-factor shapes must match and be three-dimensional, got "
                f"{tuple(bq.shape)} and {tuple(bk.shape)}"
            )
        self.register_buffer(
            "bq", bq.detach().contiguous().clone(), persistent=False,
        )
        self.register_buffer(
            "bk", bk.detach().contiguous().clone(), persistent=False,
        )

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.bq, self.bk


# =============================================================================
# Channel Attention Blocks
# =============================================================================

class ChannelAttention(nn.Module):
    """Global channel recalibration with a SwiGLU bottleneck."""

    def __init__(self, dim: int, squeeze_factor: int = 16) -> None:
        super().__init__()
        hidden = max(1, dim // squeeze_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Conv2d(dim, hidden * 2, 1, padding=0, bias=True)
        self.expand = nn.Conv2d(hidden, dim, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.force_tensorrt_export_mode = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.force_tensorrt_export_mode and not self.training:
            y = x.mean(dim=(2, 3), keepdim=True)
        else:
            y = self.pool(x)
        gate, value = self.project(y).chunk(2, dim=1)
        y = F.silu(gate) * value
        y = self.sigmoid(self.expand(y))
        return x * y


def _multiscale_pad(kernel: torch.Tensor, target_size: int) -> torch.Tensor:
    """Pad a smaller kernel to the requested size."""
    pad = (target_size - kernel.size(2)) // 2
    return F.pad(kernel, [pad, pad, pad, pad])


class _EDBBSeqConv3x3(nn.Module):
    """Sequential 1x1 -> 3x3 branch used by EDBB."""

    def __init__(
        self,
        seq_type: str,
        dim: int,
        depth_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.seq_type = seq_type
        self.dim = dim

        if seq_type == "conv1x1-conv3x3":
            self.mid_dim = int(dim * depth_multiplier)
            conv0 = nn.Conv2d(dim, self.mid_dim, 1, bias=True)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            conv1 = nn.Conv2d(self.mid_dim, dim, 3, bias=True)
            self.k1 = conv1.weight
            self.b1 = conv1.bias
        elif seq_type in ("conv1x1-sobelx", "conv1x1-sobely", "conv1x1-laplacian"):
            conv0 = nn.Conv2d(dim, dim, 1, bias=True)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            self.scale = nn.Parameter(torch.randn(dim, 1, 1, 1) * 1e-3)
            self.bias = nn.Parameter(torch.randn(dim) * 1e-3)

            mask = torch.zeros(dim, 1, 3, 3)
            if seq_type == "conv1x1-sobelx":
                kernel = torch.tensor(
                    [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                    dtype=torch.float32,
                )
            elif seq_type == "conv1x1-sobely":
                kernel = torch.tensor(
                    [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                    dtype=torch.float32,
                )
            else:
                kernel = torch.tensor(
                    [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                    dtype=torch.float32,
                )
            for i in range(dim):
                mask[i, 0] = kernel
            self.register_buffer("mask", mask)
        else:
            raise ValueError(f"Unsupported seq_type: {seq_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Equivalent to the original explicit border writes, but friendlier to
        # torch.compile because it avoids in-place slice updates.
        y0 = F.conv2d(x, self.k0, None, stride=1)
        y0 = F.pad(y0, (1, 1, 1, 1), "constant", 0)
        y0 = y0 + self.b0.view(1, -1, 1, 1)

        if self.seq_type == "conv1x1-conv3x3":
            return F.conv2d(y0, self.k1, self.b1, stride=1)

        return F.conv2d(
            y0,
            self.scale * self.mask,
            self.bias,
            stride=1,
            groups=self.dim,
        )

    def rep_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fold this branch into a single 3x3 kernel and bias."""
        device = self.k0.device
        dtype = self.k0.dtype

        if self.seq_type == "conv1x1-conv3x3":
            rep_kernel = F.conv2d(self.k1, self.k0.permute(1, 0, 2, 3))
            rep_bias = (
                torch.ones(1, self.mid_dim, 3, 3, device=device, dtype=dtype)
                * self.b0.view(1, -1, 1, 1)
            )
            rep_bias = F.conv2d(rep_bias, self.k1).view(-1) + self.b1
            return rep_kernel, rep_bias

        tmp = self.scale * self.mask
        k1 = torch.zeros(self.dim, self.dim, 3, 3, device=device, dtype=dtype)
        for i in range(self.dim):
            k1[i, i, :, :] = tmp[i, 0, :, :]
        rep_bias = (
            torch.ones(1, self.dim, 3, 3, device=device, dtype=dtype)
            * self.b0.view(1, -1, 1, 1)
        )
        rep_bias = F.conv2d(rep_bias, k1).view(-1) + self.bias
        rep_kernel = F.conv2d(k1, self.k0.permute(1, 0, 2, 3))
        return rep_kernel, rep_bias


class EDBBConvBlock(nn.Module):
    """Edge-Enhanced Diverse Branch Block."""

    def __init__(
        self,
        dim: int,
        depth_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.use_train_reparam = True

        self.conv3x3 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.conv1x1 = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)
        self.conv1x1_3x3 = _EDBBSeqConv3x3("conv1x1-conv3x3", dim, depth_multiplier)
        self.conv1x1_sbx = _EDBBSeqConv3x3("conv1x1-sobelx", dim)
        self.conv1x1_sby = _EDBBSeqConv3x3("conv1x1-sobely", dim)
        self.conv1x1_lpl = _EDBBSeqConv3x3("conv1x1-laplacian", dim)

        # Keep this three-dimensional so ``model.to(memory_format=...)`` does
        # not rewrite its strides.  The view below then matches the contiguous
        # identity tensor formerly created inside every training forward.
        identity_weight = torch.zeros(dim, dim, 9)
        channel_index = torch.arange(dim)
        identity_weight[channel_index, channel_index, 4] = 1.0
        self.register_buffer(
            "identity_weight",
            identity_weight,
            persistent=False,
        )

        # ``eval()`` uses a reversible, non-persistent folded kernel.  Keep it
        # separate from the terminal deployment fold below: ordinary PyTorch
        # train/eval transitions must never delete parameters or alter the
        # checkpoint structure.
        self.register_buffer("_eval_folded_weight", None, persistent=False)
        self.register_buffer("_eval_folded_bias", None, persistent=False)
        self._is_folded = False
        self._folded_conv: Optional[nn.Conv2d] = None
        self.register_load_state_dict_post_hook(self._refresh_eval_cache_after_load)

    def _fold_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fold all branches into a single 3x3 kernel and bias."""
        k0, b0 = self.conv3x3.weight, self.conv3x3.bias
        k1 = _multiscale_pad(self.conv1x1.weight, 3)
        b1 = self.conv1x1.bias
        k2, b2 = self.conv1x1_3x3.rep_params()
        k3, b3 = self.conv1x1_sbx.rep_params()
        k4, b4 = self.conv1x1_sby.rep_params()
        k5, b5 = self.conv1x1_lpl.rep_params()

        identity_weight = self.identity_weight.view(self.dim, self.dim, 3, 3)
        rep_kernel = k0 + k1 + k2 + k3 + k4 + k5 + identity_weight
        rep_bias = b0 + b1 + b2 + b3 + b4 + b5
        return rep_kernel, rep_bias

    def _seq3x3_rep_params_vectorized(self) -> tuple[torch.Tensor, torch.Tensor]:
        branch = self.conv1x1_3x3
        k0 = branch.k0.squeeze(-1).squeeze(-1)
        k1 = branch.k1
        rep_kernel = torch.einsum("omxy,mi->oixy", k1, k0)
        rep_bias = torch.einsum("omxy,m->o", k1, branch.b0) + branch.b1
        return rep_kernel, rep_bias

    @staticmethod
    def _edge_rep_params_vectorized(branch: _EDBBSeqConv3x3) -> tuple[torch.Tensor, torch.Tensor]:
        edge_kernel = branch.scale * branch.mask.to(
            device=branch.scale.device,
            dtype=branch.scale.dtype,
        )
        rep_kernel = branch.k0 * edge_kernel
        rep_bias = branch.bias
        return rep_kernel, rep_bias

    def _fold_weights_vectorized(self) -> tuple[torch.Tensor, torch.Tensor]:
        k0 = self.conv3x3.weight
        b0 = self.conv3x3.bias
        k1 = _multiscale_pad(self.conv1x1.weight, 3)
        b1 = self.conv1x1.bias
        k2, b2 = self._seq3x3_rep_params_vectorized()
        k3, b3 = self._edge_rep_params_vectorized(self.conv1x1_sbx)
        k4, b4 = self._edge_rep_params_vectorized(self.conv1x1_sby)
        k5, b5 = self._edge_rep_params_vectorized(self.conv1x1_lpl)

        identity_weight = self.identity_weight.view(self.dim, self.dim, 3, 3)
        rep_kernel = k0 + k1 + k2 + k3 + k4 + k5 + identity_weight
        rep_bias = b0 + b1 + b2 + b3 + b4 + b5
        return rep_kernel, rep_bias

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

    def _clear_eval_cache(self) -> None:
        self._eval_folded_weight = None
        self._eval_folded_bias = None

    @torch.no_grad()
    def _refresh_eval_cache(self) -> None:
        if self._is_folded:
            return
        folded_weight, folded_bias = self._fold_weights()
        # Preserve channels-last weight layout when the model was converted
        # before eval; forcing default contiguous layout costs eager cuDNN
        # convolution performance.
        self._eval_folded_weight = folded_weight.detach()
        self._eval_folded_bias = folded_bias.detach()

    def _refresh_eval_cache_after_load(
        self,
        module: nn.Module,
        incompatible_keys: Any,
    ) -> None:
        del module, incompatible_keys
        if self._is_folded:
            return
        self._clear_eval_cache()
        if not self.training:
            self._refresh_eval_cache()

    def train(self, mode: bool = True) -> "EDBBConvBlock":
        """Switch between train branches and the reversible eval fast path."""
        super().train(mode)
        if self._is_folded:
            return self
        if mode:
            self._clear_eval_cache()
        else:
            # Refresh even when already in eval mode.  EMA updates parameters
            # in-place while the EMA module stays in eval mode.
            self._refresh_eval_cache()
        return self

    def fold(self) -> None:
        """Permanently fold branches into a single deployment convolution."""
        if self._is_folded:
            return

        if self.training:
            raise RuntimeError(
                "EDBBConvBlock.fold() can only be called in eval mode. "
                "Call model.eval() first to avoid DDP issues with unused parameters."
            )

        folded_weight, folded_bias = self._fold_weights()
        device = self.conv3x3.weight.device

        self._folded_conv = nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=True)
        self._folded_conv.weight.data = folded_weight
        self._folded_conv.bias.data = folded_bias
        self._folded_conv = self._folded_conv.to(device)
        self._clear_eval_cache()

        del self.conv3x3
        del self.conv1x1
        del self.conv1x1_3x3
        del self.conv1x1_sbx
        del self.conv1x1_sby
        del self.conv1x1_lpl

        self._is_folded = True

    def unfold(self) -> None:
        """Unfold back to training mode.

        After fold() cleans up branch params, unfold requires reloading weights.
        """
        if not self._is_folded:
            return

        if not hasattr(self, "conv3x3"):
            raise RuntimeError(
                "Cannot unfold: original branch params were cleaned up by fold(). "
                "Reload the training checkpoint to get the multi-branch model back."
            )

        self._is_folded = False
        self._folded_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_folded and self._folded_conv is not None:
            return self._folded_conv(x)

        if not self.training:
            if self._eval_folded_weight is None or self._eval_folded_bias is None:
                self._refresh_eval_cache()
            return F.conv2d(
                x,
                self._eval_folded_weight,
                self._eval_folded_bias,
                stride=1,
                padding=1,
            )

        if self.use_train_reparam and self.training:
            rep_kernel, rep_bias = self._fold_weights_vectorized()
            return F.conv2d(x, rep_kernel, rep_bias, stride=1, padding=1)

        out = self.conv3x3(x)
        out = out + self.conv1x1(x)
        out = out + self.conv1x1_3x3(x)
        out = out + self.conv1x1_sbx(x)
        out = out + self.conv1x1_sby(x)
        out = out + self.conv1x1_lpl(x)
        out = out + x
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
        # Keep the channel dimension explicit. This is equivalent in PyTorch,
        # but it prevents TensorRT from trying to infer the grouped-convolution
        # channel count through a symbolic H*W reshape in dynamic ONNX graphs.
        x = x.transpose(1, 2).view(B, self.hidden_dim, H, W)
        x = self.dwconv(x)
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

        self.content_scale = self.head_dim ** -0.5
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

    def _prepare_query_key(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hook for architecture variants that condition Q/K before SDPA."""

        return query, key

    def _forward_with_static_bias(
        self,
        x: torch.Tensor,
        additive_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Apply attention with a precomputed dense bias.

        Shape-static shifted-window deployment uses this path for its three
        boundary classes.  Keeping each class as a separate attention call
        exposes the regular fused-MHA pattern to ONNX and TensorRT.
        """
        batch_windows, token_count, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_windows,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query, key = self._prepare_query_key(query, key)
        bias = additive_bias.to(device=query.device, dtype=query.dtype)
        output = _scaled_dot_product_attention_export_safe(
            query,
            key,
            value,
            attn_mask=bias,
            dropout_p=self.attn_drop_p,
            scale=self.content_scale,
            training=self.training,
            use_export_safe_math=torch.onnx.is_in_onnx_export(),
        )
        output = output.transpose(1, 2).reshape(
            batch_windows, token_count, channels,
        )
        return self.proj_drop(self.proj(output))

    def forward(
        self,
        x: torch.Tensor,
        mask=None,
        bias_factors: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        b_windows, n_tokens, c = x.shape

        qkv = self.qkv(x).reshape(b_windows, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self._prepare_query_key(q, k)

        bq, bk = (
            self.neural_bias() if bias_factors is None else bias_factors
        )
        export_mode = torch.onnx.is_in_onnx_export()
        use_export_safe_math = self.force_math_attention or (export_mode and mask is not None)
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
                scale=self.content_scale,
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
                scale=self.content_scale,
            )
        elif self.attn_type == 'masked' and mask is not None:
            # SDPA path: bias + region mask -> dense attn_mask.
            bias = torch.einsum('hnr,hmr->hnm', bq, bk)
            attn_mask = bias.unsqueeze(0) + mask.to(dtype=bias.dtype)
            out = _scaled_dot_product_attention_export_safe(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop_p,
                scale=self.content_scale,
                training=self.training,
                use_export_safe_math=use_export_safe_math,
            )
        else:
            # Flash path: bias via Q/K concatenation, no mask needed.
            bq = bq.unsqueeze(0).to(dtype=q.dtype).expand(b_windows, -1, -1, -1)
            bk = bk.unsqueeze(0).to(dtype=k.dtype).expand(b_windows, -1, -1, -1)

            q_aug = torch.cat([q * self.content_scale, bq], dim=-1)
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

        out = out.transpose(1, 2).reshape(b_windows, n_tokens, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class NarrowWindowAttention(nn.Module):
    """Unshifted attention with full-width endpoints and explicit-width Q/K/V."""

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        attention_dim: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        rank: int = 32,
    ) -> None:
        super().__init__()
        if not 0 < attention_dim <= dim:
            raise ValueError(
                f"attention_dim must be in (0, {dim}], got {attention_dim}"
            )
        if attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim={attention_dim} must be divisible by "
                f"num_heads={num_heads}"
            )

        head_dim = attention_dim // num_heads
        if head_dim % 8 != 0:
            raise ValueError(
                f"head_dim must be divisible by 8 for SDPA optimization, got {head_dim}"
            )
        if (head_dim + rank) % 8 != 0:
            raise ValueError(
                f"head_dim({head_dim}) + rank({rank}) must be divisible by 8"
            )

        self.dim = dim
        self.attention_dim = attention_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rank = rank
        self.content_scale = head_dim ** -0.5
        self.attn_drop_p = attn_drop
        self.force_math_attention = False
        self.force_additive_attention = False

        self.qkv = nn.Linear(dim, 3 * attention_dim, bias=qkv_bias)
        self.proj = nn.Linear(attention_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()
        self.neural_bias = RankFactoredNeuralBias(
            num_heads=num_heads,
            q_window_size=window_size,
            k_window_size=window_size,
            rank=rank,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        bias_factors: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            raise RuntimeError("NarrowWindowAttention only supports unshifted windows")

        b_windows, n_tokens, channels = x.shape
        export_tracing = torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()
        if not export_tracing and channels != self.dim:
            raise RuntimeError(
                f"expected {self.dim} endpoint channels, got {channels}"
            )

        qkv = self.qkv(x).reshape(
            b_windows,
            n_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        bq, bk = (
            self.neural_bias() if bias_factors is None else bias_factors
        )
        use_export_safe_math = self.force_math_attention
        force_additive_attention = self.force_additive_attention or use_export_safe_math

        if force_additive_attention:
            bias = torch.einsum('hnr,hmr->hnm', bq, bk).unsqueeze(0)
            out = _scaled_dot_product_attention_export_safe(
                q,
                k,
                v,
                attn_mask=bias,
                dropout_p=self.attn_drop_p,
                scale=self.content_scale,
                training=self.training,
                use_export_safe_math=use_export_safe_math,
            )
        else:
            bq = bq.unsqueeze(0).to(dtype=q.dtype).expand(b_windows, -1, -1, -1)
            bk = bk.unsqueeze(0).to(dtype=k.dtype).expand(b_windows, -1, -1, -1)
            q_aug = torch.cat([q * self.content_scale, bq], dim=-1)
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
            )[..., :self.head_dim]

        out = out.transpose(1, 2).reshape(
            b_windows,
            n_tokens,
            self.attention_dim,
        )
        return self.proj_drop(self.proj(out))


# =============================================================================
# Attention-Convolution Transformer Block (ACT)
# =============================================================================

class ACTBlock(nn.Module):
    """Attention-Convolution Transformer Block.

    Combines:
    - Window self-attention with rank-factored neural bias (SDPA backend)
    - EDBB reparameterizable conv with SwiGLU channel attention
    - Conv-SwiGLU FFN (fused gate+value)
    - Image-wide i-LN before attention/convolution and feed-forward processing
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        unshifted_num_heads: int | None = None,
        unshifted_attention_dim: int | None = None,
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
        attn_type: ATTN_TYPE = 'masked',
        full_width_unshifted: bool = False,
        edbb_depth_multiplier: float = 1.0,
        channel_squeeze_factor: int = 16,
        iln_eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.conv_scale = conv_scale
        self.force_dense_shifted_attention = False
        self.force_tensorrt_export_mode = False
        self._use_static_split_attention = False
        self.use_grouped_shifted_routing = attn_type == "hybrid"

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"

        self.norm1 = ImageLayerNorm(dim, eps=iln_eps)

        if (
            self.shift_size == 0
            and unshifted_num_heads is not None
            and not full_width_unshifted
        ):
            if unshifted_attention_dim is None:
                raise ValueError(
                    "unshifted_attention_dim is required for narrow unshifted attention"
                )
            self.attn = NarrowWindowAttention(
                dim=dim,
                window_size=to_2tuple(self.window_size),
                num_heads=unshifted_num_heads,
                attention_dim=unshifted_attention_dim,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                rank=rank,
            )
        else:
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

        # Reparameterizable conv block with SwiGLU channel attention.
        self.conv_block = nn.Sequential(
            EDBBConvBlock(dim, depth_multiplier=edbb_depth_multiplier),
            nn.GELU(),
            EDBBConvBlock(dim, depth_multiplier=edbb_depth_multiplier),
            ChannelAttention(dim, squeeze_factor=channel_squeeze_factor),
        )

        # LayerScale for stable deep network training
        self.ls_attn = LayerScale(dim, init_value=layer_scale_init)
        self.ls_conv = LayerScale(dim, init_value=layer_scale_init)
        self.ls_ffn = LayerScale(dim, init_value=layer_scale_init)

        # Drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = ImageLayerNorm(dim, eps=iln_eps)
        self.ffn = ConvSwiGLUFFN(dim=dim, expansion_factor=mlp_ratio, drop=drop)

    def _forward_shifted_windows_export_split(
        self,
        x_windows: torch.Tensor,
        B: int,
        H: int,
        W: int,
        boundary_masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Export-safe shifted-window split using reshape/slice/concat only.

        This preserves the normal masked-attention semantics:
        - interior shifted windows stay unmasked
        - only boundary shifted windows use the dense region mask
        - boundary masks stay shared across windows so TRT can keep FMHA
        """
        if self._use_static_split_attention:
            return self._forward_shifted_windows_static_split(
                x_windows, B, H, W,
            )

        nH = H // self.window_size
        nW = W // self.window_size
        interior_h = nH - 1
        interior_w = nW - 1
        right_count = interior_h
        bottom_left_count = interior_w
        ws_sq = x_windows.shape[1]
        C = x_windows.shape[2]
        right_mask, bottom_mask, corner_mask = boundary_masks

        x_grid = x_windows.reshape(B, nH, nW, ws_sq, C)
        interior_windows = x_grid[:, :interior_h, :interior_w, :, :].reshape(
            B * interior_h * interior_w, ws_sq, C,
        )
        right_windows = x_grid[:, :right_count, nW - 1, :, :].reshape(B * right_count, ws_sq, C)
        bottom_left_windows = x_grid[:, nH - 1, :interior_w, :, :].reshape(
            B * bottom_left_count, ws_sq, C,
        )
        corner_windows = x_grid[:, nH - 1, nW - 1, :, :].reshape(B, ws_sq, C)

        interior_out = self.attn(interior_windows)
        right_out = self.attn(right_windows, mask=right_mask)
        bottom_left_out = self.attn(bottom_left_windows, mask=bottom_mask)
        corner_out = self.attn(corner_windows, mask=corner_mask)
        if interior_out.dtype != x_windows.dtype:
            interior_out = interior_out.to(dtype=x_windows.dtype)
        if right_out.dtype != x_windows.dtype:
            right_out = right_out.to(dtype=x_windows.dtype)
        if bottom_left_out.dtype != x_windows.dtype:
            bottom_left_out = bottom_left_out.to(dtype=x_windows.dtype)
        if corner_out.dtype != x_windows.dtype:
            corner_out = corner_out.to(dtype=x_windows.dtype)

        interior_grid = interior_out.reshape(B, interior_h, interior_w, ws_sq, C)
        right_grid = right_out.reshape(B, right_count, 1, ws_sq, C)
        bottom_left_grid = bottom_left_out.reshape(B, 1, bottom_left_count, ws_sq, C)
        corner_grid = corner_out.reshape(B, 1, 1, ws_sq, C)
        top_rows = torch.cat([interior_grid, right_grid], dim=2)
        bottom_row = torch.cat([bottom_left_grid, corner_grid], dim=2)
        out_grid = torch.cat([top_rows, bottom_row], dim=1)
        return out_grid.reshape(B * nH * nW, ws_sq, C)

    def _forward_shifted_windows_static_split(
        self,
        x_windows: torch.Tensor,
        batch: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Run the four fixed shifted-window classes without gather/scatter."""
        if not isinstance(self.attn, WindowAttentionRFB):
            raise RuntimeError("static shifted deployment requires full-width attention")

        n_h = height // self.window_size
        n_w = width // self.window_size
        interior_h = n_h - 1
        interior_w = n_w - 1

        token_count = x_windows.shape[1]
        channels = x_windows.shape[2]
        grid = x_windows.reshape(
            batch, n_h, n_w, token_count, channels,
        )
        interior = grid[:, :interior_h, :interior_w].reshape(
            batch * interior_h * interior_w, token_count, channels,
        )
        right = grid[:, :interior_h, interior_w:n_w].reshape(
            batch * interior_h, token_count, channels,
        )
        bottom = grid[:, interior_h:n_h, :interior_w].reshape(
            batch * interior_w, token_count, channels,
        )
        corner = grid[:, interior_h:n_h, interior_w:n_w].reshape(
            batch, token_count, channels,
        )

        outputs = (
            self.attn(interior),
            self.attn._forward_with_static_bias(
                right, self.attn._deployment_right_bias,
            ),
            self.attn._forward_with_static_bias(
                bottom, self.attn._deployment_bottom_bias,
            ),
            self.attn._forward_with_static_bias(
                corner, self.attn._deployment_corner_bias,
            ),
        )
        outputs = tuple(
            value.to(dtype=x_windows.dtype)
            if value.dtype != x_windows.dtype else value
            for value in outputs
        )

        interior_grid = outputs[0].reshape(
            batch, interior_h, interior_w, token_count, channels,
        )
        right_grid = outputs[1].reshape(
            batch, interior_h, 1, token_count, channels,
        )
        bottom_grid = outputs[2].reshape(
            batch, 1, interior_w, token_count, channels,
        )
        corner_grid = outputs[3].reshape(
            batch, 1, 1, token_count, channels,
        )
        top = torch.cat((interior_grid, right_grid), dim=2)
        final_row = torch.cat((bottom_grid, corner_grid), dim=2)
        return torch.cat((top, final_row), dim=1).reshape(
            batch * n_h * n_w, token_count, channels,
        )

    def _forward_shifted_windows_grouped(
        self,
        x_windows: torch.Tensor,
        batch: int,
        height: int,
        width: int,
        region_mask,
        bias_factors: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Route shifted windows without index gather/scatter.

        The boundary order is kept batch-major as right-edge windows followed
        by the complete bottom row.  This exactly matches the cached dense and
        Flex masks while exposing slice/concat operations that Inductor can
        fuse with the surrounding window layout work.
        """
        n_h = height // self.window_size
        n_w = width // self.window_size
        interior_h = n_h - 1
        interior_w = n_w - 1
        token_count = x_windows.shape[1]
        channels = x_windows.shape[2]

        grid = x_windows.reshape(
            batch, n_h, n_w, token_count, channels,
        )
        interior_windows = grid[:, :interior_h, :interior_w].reshape(
            batch * interior_h * interior_w, token_count, channels,
        )
        right_windows = grid[:, :interior_h, n_w - 1]
        bottom_windows = grid[:, n_h - 1]
        boundary_windows = torch.cat(
            (right_windows, bottom_windows), dim=1,
        ).reshape(
            batch * (interior_h + n_w), token_count, channels,
        )

        interior_out = self.attn(
            interior_windows, bias_factors=bias_factors,
        )
        boundary_out = self.attn(
            boundary_windows,
            mask=region_mask,
            bias_factors=bias_factors,
        )
        if interior_out.dtype != x_windows.dtype:
            interior_out = interior_out.to(dtype=x_windows.dtype)
        if boundary_out.dtype != x_windows.dtype:
            boundary_out = boundary_out.to(dtype=x_windows.dtype)

        interior_grid = interior_out.reshape(
            batch, interior_h, interior_w, token_count, channels,
        )
        boundary_grid = boundary_out.reshape(
            batch, interior_h + n_w, token_count, channels,
        )
        right_grid = boundary_grid[:, :interior_h].reshape(
            batch, interior_h, 1, token_count, channels,
        )
        bottom_grid = boundary_grid[:, interior_h:].reshape(
            batch, 1, n_w, token_count, channels,
        )
        top_grid = torch.cat((interior_grid, right_grid), dim=2)
        return torch.cat((top_grid, bottom_grid), dim=1).reshape(
            batch * n_h * n_w, token_count, channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_size: tuple[int, int],
        hybrid_ctx=None,
        attention_bias_factors: tuple[
            torch.Tensor, torch.Tensor
        ] | None = None,
    ) -> torch.Tensor:
        H, W = x_size
        B, L, C = x.shape

        shortcut = x

        x_norm, residual_scale = self.norm1(x)
        x_2d = x_norm.view(B, H, W, C)

        # === Conv Branch ===
        conv_x = x_2d.permute(0, 3, 1, 2)  # (B, C, H, W)
        conv_x = self.conv_block(conv_x)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)  # (B, L, C)

        # === Attention Branch ===
        # Cyclic shift for shifted window attention
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x_2d,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        else:
            shifted_x = x_2d

        # Partition windows
        x_windows = window_partition(
            shifted_x, self.window_size,
        )  # (nW*B, ws, ws, C)
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C,
        )

        # Window attention with per-window routing for shifted blocks.
        # Non-shifted blocks: all windows → Flash path (no contamination risk).
        # Shifted blocks: interior windows → Flash, boundary windows → masked/Flex.
        # int_idx_full/bnd_idx_full are pre-expanded batch indices cached at
        # model level — no per-block recomputation needed.
        if self.shift_size > 0 and hybrid_ctx is not None:
            region_mask, int_idx_full, bnd_idx_full = hybrid_ctx

            if self.force_dense_shifted_attention:
                attn_windows = self.attn(
                    x_windows,
                    mask=region_mask,
                    bias_factors=attention_bias_factors,
                )
            elif int_idx_full is None or bnd_idx_full is None:
                if self.force_tensorrt_export_mode:
                    attn_windows = self._forward_shifted_windows_export_split(
                        x_windows, B, H, W, region_mask,
                    )
                else:
                    attn_windows = self.attn(
                        x_windows,
                        mask=region_mask,
                        bias_factors=attention_bias_factors,
                    )
            elif (
                self.use_grouped_shifted_routing
                and H >= 2 * self.window_size
                and W >= 2 * self.window_size
            ):
                attn_windows = self._forward_shifted_windows_grouped(
                    x_windows,
                    B,
                    H,
                    W,
                    region_mask,
                    attention_bias_factors,
                )
            else:
                # Process each group sequentially: gather -> attn -> scatter -> free,
                # so both temporary outputs are never alive at the same time.
                # Guard empty index groups and align dtypes for index_put under AMP/BF16.
                attn_windows = torch.empty_like(x_windows)
                export_mode = self.attn.force_additive_attention or torch.onnx.is_in_onnx_export()

                # Interior windows: Flash path (no mask needed)
                if export_mode:
                    interior_windows = x_windows[int_idx_full]
                    interior_out = self.attn(
                        interior_windows,
                        bias_factors=attention_bias_factors,
                    )
                    if interior_out.dtype != attn_windows.dtype:
                        interior_out = interior_out.to(dtype=attn_windows.dtype)
                    attn_windows[int_idx_full] = interior_out
                    del interior_windows, interior_out
                elif int_idx_full.numel() > 0:
                    interior_windows = x_windows[int_idx_full]
                    interior_out = self.attn(
                        interior_windows,
                        bias_factors=attention_bias_factors,
                    )
                    if interior_out.dtype != attn_windows.dtype:
                        interior_out = interior_out.to(dtype=attn_windows.dtype)
                    attn_windows[int_idx_full] = interior_out
                    del interior_windows, interior_out

                # Boundary windows: masked SDPA or Flex (with region mask)
                if export_mode:
                    boundary_windows = x_windows[bnd_idx_full]
                    boundary_out = self.attn(
                        boundary_windows,
                        mask=region_mask,
                        bias_factors=attention_bias_factors,
                    )
                    if boundary_out.dtype != attn_windows.dtype:
                        boundary_out = boundary_out.to(dtype=attn_windows.dtype)
                    attn_windows[bnd_idx_full] = boundary_out
                    del boundary_windows, boundary_out
                elif bnd_idx_full.numel() > 0:
                    boundary_windows = x_windows[bnd_idx_full]
                    boundary_out = self.attn(
                        boundary_windows,
                        mask=region_mask,
                        bias_factors=attention_bias_factors,
                    )
                    if boundary_out.dtype != attn_windows.dtype:
                        boundary_out = boundary_out.to(dtype=attn_windows.dtype)
                    attn_windows[bnd_idx_full] = boundary_out
                    del boundary_windows, boundary_out
        else:
            attn_windows = self.attn(
                x_windows, bias_factors=attention_bias_factors,
            )

        # Merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C,
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, H, W,
        )

        # Reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            attn_x = shifted_x

        attn_x = attn_x.view(B, L, C)

        # Apply LayerScale to attention output
        attn_x = self.ls_attn(attn_x)

        # Apply LayerScale to conv output
        conv_x = self.ls_conv(conv_x)

        # Combine attention and convolution branches. Paper i-LN restores the
        # input-specific magnitude of the complete residual sublayer.
        residual = (
            self.drop_path(attn_x) + conv_x * self.conv_scale
        ) * residual_scale
        x = shortcut + residual

        # FFN with LayerScale
        ffn_shortcut = x
        x_norm2, ffn_scale = self.norm2(x)
        ffn_out = self.ffn(x_norm2, H, W)
        ffn_out = self.ls_ffn(ffn_out)
        ffn_out = self.drop_path(ffn_out) * ffn_scale
        x = ffn_shortcut + ffn_out

        return x


# =============================================================================
# Overlapping Cross-Attention Block (OCAB)
# =============================================================================

class OCAB(nn.Module):
    """Overlapping Cross-Attention Block with dense signed relative bias.

    Enables direct cross-window information flow using unfold
    with overlapping regions.

    Uses a dense signed relative-position bias table inside OCAB only.
    ACT remains rank-factorized; OCAB stays dense because its positional bias
    is substantially higher-rank and signed, which is critical for text fidelity.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        window_size: int,
        overlap_window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        mlp_ratio: float = 2.667,
        layer_scale_init: float = 1e-6,
        rank: int = 32,
        attn_type: ATTN_TYPE = "masked",
        iln_eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        if overlap_window_size < window_size:
            raise ValueError("overlap_window_size must be at least window_size")
        if (overlap_window_size - window_size) % 2 != 0:
            raise ValueError(
                "overlap_window_size - window_size must be even for symmetric padding"
            )
        self.overlap_win_size = overlap_window_size
        self.rank = rank
        self.attn_type = attn_type
        self.force_math_attention = False
        self._use_batch_axis_kv = False
        self.use_compact_relative_bias_training = False
        self.use_compact_unfold_training = False
        self.use_fused_qkv_windows_training = False
        self.use_window_order_projection_training = False
        self.scale = head_dim ** -0.5

        self.norm1 = ImageLayerNorm(dim, eps=iln_eps)
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

        self.norm2 = ImageLayerNorm(dim, eps=iln_eps)
        self.ffn = ConvSwiGLUFFN(dim=dim, expansion_factor=mlp_ratio)

    @property
    def q_window_size(self) -> tuple[int, int]:
        return (self.window_size, self.window_size)

    @property
    def k_window_size(self) -> tuple[int, int]:
        return (self.overlap_win_size, self.overlap_win_size)

    def _relative_position_bias(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Materialize dense OCAB bias late and in the attention dtype."""
        static_bias = getattr(self, "_deployment_relative_position_bias", None)
        if static_bias is not None:
            return static_bias.to(device=device, dtype=dtype)

        if self.use_compact_relative_bias_training and self.training and device.type == "cuda":
            if not _OCAB_COMPACT_BIAS_AVAILABLE:
                raise RuntimeError(
                    "compact OCAB training bias requires Triton and torch.library.triton_op"
                )
            return ocab_relative_bias(
                self.relative_position_bias_table,
                self.q_window_size[0],
                self.q_window_size[1],
                self.k_window_size[0],
                self.k_window_size[1],
                ocab_bias_dtype_code(dtype),
            )

        index = _get_shared_relative_position_index(self.q_window_size, self.k_window_size, device)
        n_q = self.window_size * self.window_size
        n_k = self.overlap_win_size * self.overlap_win_size
        bias = self.relative_position_bias_table[index.view(-1)]
        bias = bias.view(n_q, n_k, self.num_heads).permute(2, 0, 1).contiguous()
        return bias.to(dtype=dtype).unsqueeze(0)

    def _overlap_unfold(self, kv: torch.Tensor) -> torch.Tensor:
        if self.use_compact_unfold_training and self.training and kv.is_cuda:
            if not _OCAB_COMPACT_UNFOLD_AVAILABLE:
                raise RuntimeError(
                    "compact OCAB training unfold requires Triton and "
                    "torch.library.triton_op"
                )
            padding = (self.overlap_win_size - self.window_size) // 2
            return ocab_unfold2d(
                kv,
                self.overlap_win_size,
                self.overlap_win_size,
                self.window_size,
                self.window_size,
                padding,
                padding,
            )
        return self.unfold(kv)

    def _prepare_query_key(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hook for architecture variants that condition Q/K before SDPA."""

        return query, key

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        H, W = x_size
        B, L, C = x.shape

        shortcut = x

        x_norm, residual_scale = self.norm1(x)
        x_norm = x_norm.view(B, H, W, C)

        # QKV
        qkv = self.qkv(x_norm).reshape(B, H, W, 3, C)
        ow = self.overlap_win_size
        if (
            self.use_fused_qkv_windows_training
            and self.training
            and qkv.is_cuda
        ):
            if not _OCAB_FUSED_QKV_WINDOWS_AVAILABLE:
                raise RuntimeError(
                    "fused OCAB Q/K/V windows require Triton and "
                    "torch.library.triton_op"
                )
            q_windows, kv_windows = ocab_qkv_windows(
                qkv,
                self.window_size,
                ow,
            )
        else:
            qkv = qkv.permute(3, 0, 4, 1, 2)  # 3, B, C, H, W
            q = qkv[0].permute(0, 2, 3, 1)  # B, H, W, C

            # Partition Q windows
            q_windows = window_partition(q, self.window_size)
            q_windows = q_windows.view(
                -1,
                self.window_size * self.window_size,
                C,
            )

            # Unfold K/V with overlap.
            if self._use_batch_axis_kv:
                # Keep K/V on a leading 2*batch axis through Unfold.  Unfold
                # acts independently per batch item, so this is exactly
                # equivalent to channel concatenation.
                kv = qkv[1:3].reshape(2 * B, C, H, W)
                kv_windows = self._overlap_unfold(kv)
                nW = kv_windows.shape[2]
                kv_windows = kv_windows.view(2, B, C, ow, ow, nW)
                kv_windows = kv_windows.permute(
                    0,
                    1,
                    5,
                    3,
                    4,
                    2,
                ).contiguous()
            else:
                kv = torch.cat((qkv[1], qkv[2]), dim=1)  # B, 2*C, H, W
                kv_windows = self._overlap_unfold(kv)
                nW = kv_windows.shape[2]
                kv_windows = kv_windows.view(B, 2, C, ow, ow, nW)
                kv_windows = kv_windows.permute(
                    1,
                    0,
                    5,
                    3,
                    4,
                    2,
                ).contiguous()
            kv_windows = kv_windows.view(2, B * nW, ow * ow, C)
        k_windows, v_windows = kv_windows.unbind(0)

        # Attention
        B_, Nq, _ = q_windows.shape
        _, Nk, _ = k_windows.shape

        q = q_windows.reshape(B_, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k_windows.reshape(B_, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v_windows.reshape(B_, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self._prepare_query_key(q, k)

        use_export_safe_math = self.force_math_attention
        attn_bias = self._relative_position_bias(q.device, q.dtype)

        if use_export_safe_math or not _SDPA_BACKEND_AVAILABLE or not q.is_cuda:
            x = _scaled_dot_product_attention_export_safe(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=0.0,
                scale=self.scale,
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
                    scale=self.scale,
                )

        x = x.transpose(1, 2).reshape(B_, Nq, self.dim)

        # Projection is tokenwise and therefore commutes exactly with window
        # reversal.  In compiled training, applying it while tokens are still
        # in contiguous window order lets Inductor fuse the later permutation
        # into residual/statistics work instead of materializing an extra
        # spatial-order input for the GEMM.  Keep inference/export ordering
        # unchanged so their established deployment graph is untouched.
        project_in_window_order = (
            self.use_window_order_projection_training
            and self.training
            and x.is_cuda
        )
        if project_in_window_order:
            x = self.proj(x)

        # Merge windows
        x = x.view(-1, self.window_size, self.window_size, self.dim)
        x = window_reverse(x, self.window_size, H, W)
        x = x.view(B, L, self.dim)

        if not project_in_window_order:
            x = self.proj(x)

        # Apply LayerScale to attention output
        x = self.ls_attn(x) * residual_scale

        x = shortcut + x

        # FFN with LayerScale
        ffn_shortcut = x
        x_norm2, ffn_scale = self.norm2(x)
        ffn_out = self.ffn(x_norm2, H, W)
        ffn_out = self.ls_ffn(ffn_out) * ffn_scale
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
        unshifted_num_heads: int,
        unshifted_attention_dim: int,
        window_size: int,
        overlap_window_size: int,
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
        attn_type: ATTN_TYPE = 'masked',
        full_width_unshifted: bool = False,
        edbb_depth_multiplier: float = 1.0,
        channel_squeeze_factor: int = 16,
        iln_eps: float = 1e-4,
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
        self.use_compiled_checkpoint_policy = False
        self.compiled_checkpoint_context_fn = (
            _FLASH_QKV_NORM_STATS_SELECTIVE_CHECKPOINT_CONTEXT
        )

        # Build ACT blocks
        self.blocks = nn.ModuleList([
            ACTBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                unshifted_num_heads=unshifted_num_heads if i % 2 == 0 else None,
                unshifted_attention_dim=(
                    unshifted_attention_dim if i % 2 == 0 else None
                ),
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
                attn_type=attn_type,
                full_width_unshifted=full_width_unshifted,
                edbb_depth_multiplier=edbb_depth_multiplier,
                channel_squeeze_factor=channel_squeeze_factor,
                iln_eps=iln_eps,
            )
            for i in range(depth)
        ])

        # OCAB for cross-window attention
        self.ocab = OCAB(
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            overlap_window_size=overlap_window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            mlp_ratio=mlp_ratio,
            layer_scale_init=layer_scale_init,
            rank=rank,
            attn_type=attn_type,
            iln_eps=iln_eps,
        )

        # Dense skip fusion (if enabled) with LayerScale
        if dense_skip:
            self.dense_fusion = nn.Linear(dim * depth, dim)
            self.ls_dense = LayerScale(dim, init_value=1e-4)

    def forward(
        self,
        x: torch.Tensor,
        x_size: tuple[int, int],
        hybrid_ctx=None,
    ) -> torch.Tensor:
        checkpoint_options: dict[str, Any] = {"use_reentrant": False}
        if self.use_compiled_checkpoint_policy:
            checkpoint_options["context_fn"] = self.compiled_checkpoint_context_fn
        if self.dense_skip:
            # Collect outputs from all blocks
            block_outputs = []
            for blk in self.blocks:
                if self.use_checkpoint_act and self.training:
                    attention_bias_factors = (
                        blk.attn.neural_bias()
                        if self.use_compiled_checkpoint_policy
                        else None
                    )
                    if attention_bias_factors is not None:
                        x = checkpoint(
                            blk,
                            x,
                            x_size,
                            hybrid_ctx,
                            attention_bias_factors,
                            **checkpoint_options,
                        )
                    else:
                        x = checkpoint(
                            blk, x, x_size, hybrid_ctx, **checkpoint_options,
                        )
                else:
                    x = blk(x, x_size, hybrid_ctx)
                block_outputs.append(x)

            # OCAB on final sequential output BEFORE dense fusion
            # This ensures cross-window attention operates on refined sequential features
            # OCAB does not use shifted windows — no block_mask needed
            if self.use_checkpoint_ocab and self.training:
                x = checkpoint(self.ocab, x, x_size, **checkpoint_options)
            else:
                x = self.ocab(x, x_size)

            # Dense fusion: concatenate all block outputs and project with LayerScale
            dense_cat = torch.cat(block_outputs, dim=-1)  # (B, L, dim * depth)
            dense_out = self.dense_fusion(dense_cat)  # (B, L, dim)
            x = x + self.ls_dense(dense_out)
        else:
            for blk in self.blocks:
                if self.use_checkpoint_act and self.training:
                    attention_bias_factors = (
                        blk.attn.neural_bias()
                        if self.use_compiled_checkpoint_policy
                        else None
                    )
                    if attention_bias_factors is not None:
                        x = checkpoint(
                            blk,
                            x,
                            x_size,
                            hybrid_ctx,
                            attention_bias_factors,
                            **checkpoint_options,
                        )
                    else:
                        x = checkpoint(
                            blk, x, x_size, hybrid_ctx, **checkpoint_options,
                        )
                else:
                    x = blk(x, x_size, hybrid_ctx)

            # OCAB — no block_mask needed (no shifted windows)
            if self.use_checkpoint_ocab and self.training:
                x = checkpoint(self.ocab, x, x_size, **checkpoint_options)
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
        unshifted_num_heads: int,
        unshifted_attention_dim: int,
        window_size: int,
        overlap_window_size: int,
        mlp_ratio: float = 2.667,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float | list[float] = 0.,
        conv_scale: float = 0.01,
        layer_scale_init: float = 1e-6,
        rhag_layer_scale_init: float | None = None,
        use_checkpoint: bool = False,
        use_checkpoint_act: Optional[bool] = None,
        use_checkpoint_ocab: Optional[bool] = None,
        dense_skip: bool = True,
        resi_connection: str = '1conv',
        rank: int = 32,
        attn_type: ATTN_TYPE = 'masked',
        full_width_unshifted: bool = False,
        edbb_depth_multiplier: float = 1.0,
        channel_squeeze_factor: int = 16,
        iln_eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.dim = dim

        self.residual_group = AttentionBlocks(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            unshifted_num_heads=unshifted_num_heads,
            unshifted_attention_dim=unshifted_attention_dim,
            window_size=window_size,
            overlap_window_size=overlap_window_size,
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
            attn_type=attn_type,
            full_width_unshifted=full_width_unshifted,
            edbb_depth_multiplier=edbb_depth_multiplier,
            channel_squeeze_factor=channel_squeeze_factor,
            iln_eps=iln_eps,
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

        # The ACT/OCAB LayerScale modules stabilize residuals *inside* a group,
        # but do not constrain the complete HAT-style RHAG residual.  Deep DRFT
        # variants can opt into this additional boundary without changing the
        # parameter/state contract of the already-stable shallower variants.
        if rhag_layer_scale_init is not None and rhag_layer_scale_init < 0:
            raise ValueError("rhag_layer_scale_init must be non-negative or None")
        self.ls_rhag = (
            LayerScale(dim, init_value=rhag_layer_scale_init)
            if rhag_layer_scale_init is not None
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        x_size: tuple[int, int],
        hybrid_ctx=None,
    ) -> torch.Tensor:
        # x: (B, L, C)
        H, W = x_size

        # Attention blocks
        res = self.residual_group(x, x_size, hybrid_ctx)

        # Conv on residual
        res = res.transpose(1, 2).view(-1, self.dim, H, W)
        res = self.conv(res)
        res = res.flatten(2).transpose(1, 2)

        # Residual add. Deep variants gate the complete RHAG branch here;
        # shallower variants retain their exact historical x + res behavior.
        x = x + self.ls_rhag(res)
        return x


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
    """Dense rank-factored image-restoration transformer.

    ``embed_dim`` is the width of the trunk, every residual endpoint, and every
    local-attention block. The canonical defaults keep unshifted and shifted
    attention full-width, use the original half-window OCAB overlap, and apply
    image-wide i-LN at every normalization site.
    """

    # Bump only when DRFT's compiled graph contract changes incompatibly.
    # PyTorch includes this in its persistent compile-cache key, allowing safe
    # reuse across retraining while preventing pre-fix binaries from being
    # resurrected after compiler-facing graph changes.
    # Teacher and student instances share Python code but can have different
    # fixed attention scales. Keep Python float attributes specialized so
    # AOTAutograd never lifts those constants into CPU scalar graph inputs.
    compile_specialize_float = True
    compile_cache_key_tag = (
        "drft-iln-family-v20-sdpa-ocab-layout-fusion-raw-fp32-stats"
    )

    # Dynamic DRFT ONNX exports always trace this representative shape. The
    # dimensions remain symbolic in ONNX; this value is only example data.
    onnx_export_example_size = 96
    # torch.export currently specializes DRFT's pad/window/crop shape algebra
    # even with named H/W dimensions. The legacy exporter preserves the same
    # graph symbolically and is therefore the required dynamic-spatial route.
    onnx_dynamic_spatial_exporter = "legacy"

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 192,
        depths: Sequence[int] = (6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        unshifted_num_heads: Sequence[int] | None = None,
        unshifted_attention_dim: Sequence[int] | None = None,
        window_size: int = 32,
        overlap_window_size: int | None = None,
        mlp_ratio: float = 2.667,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        conv_scale: float = 0.01,
        layer_scale_init: float = 1e-6,
        rhag_layer_scale_init: float | None = None,
        use_checkpoint: bool = False,
        use_checkpoint_act: Optional[bool] = None,
        use_checkpoint_ocab: Optional[bool] = None,
        upscale: int = 4,
        img_range: float = 1.,
        resi_connection: str = '1conv',
        dense_skip: bool = True,
        num_feat: int = 64,
        rank: int = 32,
        attn_type: ATTN_TYPE = 'masked',
        reconstruction: RECONSTRUCTION_TYPE = 'progressive',
        full_width_unshifted: bool = True,
        edbb_depth_multiplier: float = 1.0,
        channel_squeeze_factor: int = 16,
        iln_eps: float = 1e-4,
    ) -> None:
        super().__init__()

        if len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must have the same length")
        if attn_type not in ("masked", "hybrid"):
            raise ValueError(
                "attn_type must be 'masked' or 'hybrid', "
                f"got {attn_type!r}"
            )
        if reconstruction not in ("progressive", "direct"):
            raise ValueError(
                "reconstruction must be 'progressive' or 'direct', "
                f"got {reconstruction!r}"
            )

        num_heads = tuple(num_heads)
        if unshifted_num_heads is None:
            unshifted_num_heads = num_heads
        else:
            unshifted_num_heads = tuple(unshifted_num_heads)

        if len(unshifted_num_heads) != len(depths):
            raise ValueError(
                "depths and unshifted_num_heads must have the same length"
            )
        if unshifted_attention_dim is None:
            unshifted_attention_dim = (embed_dim,) * len(depths)
        else:
            unshifted_attention_dim = tuple(unshifted_attention_dim)
        if len(unshifted_attention_dim) != len(depths):
            raise ValueError(
                "depths and unshifted_attention_dim must have the same length"
            )

        for stage, (full_heads, narrow_heads, attention_dim) in enumerate(
            zip(
                num_heads,
                unshifted_num_heads,
                unshifted_attention_dim,
                strict=True,
            )
        ):
            if embed_dim % full_heads != 0:
                raise ValueError(
                    f"stage {stage}: embed_dim={embed_dim} is not divisible by "
                    f"num_heads={full_heads}"
                )
            if not 0 < attention_dim <= embed_dim:
                raise ValueError(
                    f"stage {stage}: unshifted_attention_dim={attention_dim} must "
                    f"be in (0, {embed_dim}]"
                )
            if attention_dim % narrow_heads != 0:
                raise ValueError(
                    f"stage {stage}: unshifted_attention_dim={attention_dim} is not "
                    f"divisible by unshifted_num_heads={narrow_heads}"
                )
            full_head_dim = embed_dim // full_heads
            narrow_head_dim = attention_dim // narrow_heads
            if full_head_dim % 8 != 0 or narrow_head_dim % 8 != 0:
                raise ValueError(
                    f"stage {stage}: attention head dimensions must be divisible by 8"
                )
            if (full_head_dim + rank) % 8 != 0 or (narrow_head_dim + rank) % 8 != 0:
                raise ValueError(
                    f"stage {stage}: attention head dimension plus rank must be divisible by 8"
                )

        if overlap_window_size is None:
            overlap_window_size = window_size + window_size // 2

        resolved_checkpoint_act = (
            use_checkpoint if use_checkpoint_act is None else use_checkpoint_act
        )
        resolved_checkpoint_ocab = (
            False if use_checkpoint_ocab is None else use_checkpoint_ocab
        )
        compile_contract = (
            "drft-iln-family-v13-prefusion-execution",
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            tuple(depths),
            num_heads,
            unshifted_num_heads,
            unshifted_attention_dim,
            window_size,
            overlap_window_size,
            mlp_ratio,
            qkv_bias,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            conv_scale,
            layer_scale_init,
            rhag_layer_scale_init,
            resolved_checkpoint_act,
            resolved_checkpoint_ocab,
            upscale,
            img_range,
            resi_connection,
            dense_skip,
            num_feat,
            rank,
            attn_type,
            reconstruction,
            full_width_unshifted,
            edbb_depth_multiplier,
            channel_squeeze_factor,
            iln_eps,
        )
        compile_contract_sha = hashlib.sha256(
            repr(compile_contract).encode("utf-8")
        ).hexdigest()[:20]
        self.compile_cache_key_tag = (
            "drft-iln-family-v20-sdpa-ocab-layout-fusion-raw-fp32-stats-"
            f"{compile_contract_sha}"
        )

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
        self.embed_dim = embed_dim
        self.depths = tuple(depths)
        self.window_size = window_size
        self.overlap_window_size = overlap_window_size
        self.num_heads = num_heads
        self.unshifted_num_heads = unshifted_num_heads
        self.unshifted_attention_dim = unshifted_attention_dim
        self.attn_type = attn_type
        self.reconstruction = reconstruction
        self.force_tensorrt_export_mode = False
        self._onnx_deployment_input_shape: tuple[int, int, int, int] | None = None
        self._onnx_deployment_feature_shape: tuple[int, int] | None = None
        self._onnx_deployment_dynamic_spatial = False
        self.onnx_deployment_inventory: dict[str, Any] = {}

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

        # Patch embed/unembed without an additional normalization layer.
        self.patch_embed = PatchEmbed(embed_dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(embed_dim)

        # Input resolution for position encoding
        patches_resolution = (img_size, img_size)

        # Build RHAGs
        self.num_layers = len(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHAG(
                dim=embed_dim,
                input_resolution=patches_resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                unshifted_num_heads=unshifted_num_heads[i_layer],
                unshifted_attention_dim=unshifted_attention_dim[i_layer],
                window_size=window_size,
                overlap_window_size=overlap_window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                conv_scale=conv_scale,
                layer_scale_init=layer_scale_init,
                rhag_layer_scale_init=rhag_layer_scale_init,
                use_checkpoint=use_checkpoint,
                use_checkpoint_act=use_checkpoint_act,
                use_checkpoint_ocab=use_checkpoint_ocab,
                dense_skip=dense_skip,
                resi_connection=resi_connection,
                rank=rank,
                attn_type=attn_type,
                full_width_unshifted=full_width_unshifted,
                edbb_depth_multiplier=edbb_depth_multiplier,
                channel_squeeze_factor=channel_squeeze_factor,
                iln_eps=iln_eps,
            )
            self.layers.append(layer)

        self.norm = AffineTransform(embed_dim)

        # Feature fusion before upsampling
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # Reconstruction. The direct head keeps all convolution at LR and
        # performs a single terminal PixelShuffle, which is preferable for
        # latency-constrained variants.
        if reconstruction == 'progressive':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.GELU(),
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        else:
            self.conv_before_upsample = nn.Identity()
            self.upsample = nn.Sequential(
                nn.Conv2d(
                    embed_dim,
                    num_out_ch * upscale * upscale,
                    3,
                    1,
                    1,
                ),
                nn.PixelShuffle(upscale),
            )
            self.conv_last = nn.Identity()

        # Initialize weights
        self.apply(self._init_weights)

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
        export_mode = self.force_tensorrt_export_mode or torch.onnx.is_in_onnx_export()
        if export_mode:
            ws = self.window_size
            nH_grid, nW_grid = H // ws, W // ws
            nW_total = nH_grid * nW_grid
            window_ids = torch.arange(nW_total, device=device).reshape(nH_grid, nW_grid)
            interior_idx = window_ids[: nH_grid - 1, : nW_grid - 1].reshape(-1)
            boundary_idx = torch.cat(
                [
                    window_ids[: nH_grid - 1, nW_grid - 1 : nW_grid].reshape(-1),
                    window_ids[nH_grid - 1 : nH_grid, :].reshape(-1),
                ],
                dim=0,
            )
            offsets = torch.arange(B, device=device).unsqueeze(1) * nW_total
            int_idx_full = (offsets + interior_idx.unsqueeze(0)).reshape(-1)
            bnd_idx_full = (offsets + boundary_idx.unsqueeze(0)).reshape(-1)
            return int_idx_full, bnd_idx_full

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
        export_mode = self.force_tensorrt_export_mode or torch.onnx.is_in_onnx_export()
        if export_mode:
            ws = self.window_size
            nH_grid, nW_grid = H // ws, W // ws
            region_ids = self._compute_region_ids(H, W, device)
            region_grid = region_ids.reshape(nH_grid, nW_grid, ws * ws)
            boundary_regions = torch.cat(
                [
                    region_grid[: nH_grid - 1, nW_grid - 1 : nW_grid, :].reshape(nH_grid - 1, ws * ws),
                    region_grid[nH_grid - 1 : nH_grid, :, :].reshape(nW_grid, ws * ws),
                ],
                dim=0,
            )
            same_region = boundary_regions.unsqueeze(2) == boundary_regions.unsqueeze(1)
            mask = torch.zeros(same_region.shape, dtype=torch.float32, device=device)
            mask.masked_fill_(~same_region, float('-inf'))
            return mask.unsqueeze(1).repeat(B, 1, 1, 1)

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

    def _get_shifted_export_boundary_masks(
        self, H: int, W: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the three shared boundary masks used by shifted-window export.

        Right-edge windows, bottom-edge windows, and the bottom-right corner
        each have a distinct region pattern, but every window within a group
        shares the same mask. Returning broadcastable `(1, 1, ws^2, ws^2)`
        masks keeps TensorRT on the small-memory FMHA path.
        """
        ws = self.window_size
        nH_grid, nW_grid = H // ws, W // ws
        region_ids = self._compute_region_ids(H, W, device)
        region_grid = region_ids.reshape(nH_grid, nW_grid, ws * ws)

        def build_mask(region_vec: torch.Tensor) -> torch.Tensor:
            same_region = region_vec.unsqueeze(1) == region_vec.unsqueeze(0)
            mask = torch.zeros((1, 1, ws * ws, ws * ws), dtype=torch.float32, device=device)
            mask[0, 0].masked_fill_(~same_region, float('-inf'))
            return mask

        right_mask = build_mask(region_grid[0, nW_grid - 1, :])
        bottom_mask = build_mask(region_grid[nH_grid - 1, 0, :])
        corner_mask = build_mask(region_grid[nH_grid - 1, nW_grid - 1, :])
        return right_mask, bottom_mask, corner_mask

    def _get_shifted_dense_mask_all_windows(
        self, B: int, H: int, W: int, device: torch.device,
    ) -> torch.Tensor:
        """Get or create a dense shifted-window mask for every window.

        Interior windows become all-zero masks, while boundary windows keep the
        usual same-region versus blocked-region structure. This simplifies the
        exported graph by letting all shifted windows share one masked path.
        """
        export_mode = self.force_tensorrt_export_mode or torch.onnx.is_in_onnx_export()
        if export_mode:
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

    @staticmethod
    def _set_deployment_buffer(
        module: nn.Module,
        name: str,
        value: torch.Tensor,
    ) -> None:
        value = value.detach().contiguous()
        if name in module._buffers:
            setattr(module, name, value)
        else:
            module.register_buffer(name, value, persistent=False)

    def prepare_for_onnx_export(
        self,
        input_shape: Sequence[int],
        *,
        clone: bool = True,
        precision: ONNX_DEPLOYMENT_PRECISION = "native",
        dynamic_spatial: bool = False,
    ) -> nn.Module:
        """Create the optimized portable ONNX deployment graph.

        The returned model is a terminal inference clone by default: training
        parameters and checkpoint structure on ``self`` are not modified.
        Preparation folds reparameterizable convolutions, captures local
        position factors, materializes overlapping-attention bias, uses a
        Q4R gather-free four-way shifted-window split, and keeps overlapping
        K/V on a leading batch axis through Unfold.

        ``input_shape`` is the concrete trace example, not necessarily a fixed
        deployment contract. With ``dynamic_spatial=True``, N/C remain fixed
        while H/W stay symbolic in ONNX. TensorRT should then build a separate
        fixed-profile engine for each deployment resolution. The optional
        ``tensorrt_mixed`` policy keeps host I/O in FP32, runs the body in BF16,
        and keeps the upsampler plus final convolution in FP16.
        """
        if precision not in ("native", "tensorrt_mixed"):
            raise ValueError(f"unsupported DRFT ONNX precision policy: {precision}")

        shape = tuple(input_shape)
        if len(shape) != 4 or any(not isinstance(value, int) for value in shape):
            raise ValueError(
                "DRFT ONNX deployment requires a concrete integer NCHW shape"
            )
        batch, channels, height, width = shape
        expected_channels = int(self.mean.shape[1])
        if batch < 1 or height < 1 or width < 1:
            raise ValueError(f"N, H, and W must be positive, got {shape}")
        if channels != expected_channels:
            raise ValueError(
                f"expected {expected_channels} input channels, got {channels}"
            )

        feature_height = math.ceil(height / self.window_size) * self.window_size
        feature_width = math.ceil(width / self.window_size) * self.window_size
        if (
            feature_height // self.window_size < 2
            or feature_width // self.window_size < 2
        ):
            raise ValueError(
                "the fast shifted-window ONNX route requires a padded feature "
                "grid of at least 2x2 windows"
            )

        deployment = copy.deepcopy(self) if clone else self
        if deployment._onnx_deployment_input_shape is not None:
            same_contract = (
                deployment._onnx_deployment_dynamic_spatial == dynamic_spatial
                and (
                    deployment._onnx_deployment_input_shape[:2] == shape[:2]
                    if dynamic_spatial
                    else deployment._onnx_deployment_input_shape == shape
                )
            )
            if not same_contract:
                raise RuntimeError(
                    "this deployment model is already prepared for "
                    f"{deployment._onnx_deployment_input_shape} "
                    f"(dynamic_spatial={deployment._onnx_deployment_dynamic_spatial}), "
                    f"not {shape} (dynamic_spatial={dynamic_spatial})"
                )
            if precision == "tensorrt_mixed":
                return _DRFTTensorRTMixedPrecision(deployment)
            return deployment

        deployment.eval()
        for module in deployment.modules():
            if isinstance(module, RankFactoredNeuralBias):
                module.clear_cache()

        deployment.fold_reparam_conv()
        folded_edbb = sum(
            isinstance(module, EDBBConvBlock) and module._is_folded
            for module in deployment.modules()
        )

        factor_records: list[dict[str, Any]] = []
        static_constant_bytes = 0
        for path, module in list(deployment.named_modules()):
            if not isinstance(module, (WindowAttentionRFB, NarrowWindowAttention)):
                continue
            dynamic_bias = module.neural_bias
            if not isinstance(dynamic_bias, RankFactoredNeuralBias):
                raise RuntimeError(
                    f"{path}.neural_bias was already replaced or has an unsupported type"
                )
            with torch.inference_mode():
                bq, bk = dynamic_bias()
            module.neural_bias = _StaticRankFactoredBias(bq, bk)
            factor_bytes = (bq.numel() + bk.numel()) * bq.element_size()
            static_constant_bytes += factor_bytes
            factor_records.append(
                {
                    "path": path,
                    "shape": tuple(bq.shape),
                    "dtype": str(bq.dtype),
                    "bytes": factor_bytes,
                }
            )

        ocab_records: list[dict[str, Any]] = []
        for path, module in deployment.named_modules():
            if not isinstance(module, OCAB):
                continue
            table = module.relative_position_bias_table
            index = _build_relative_position_index(
                module.q_window_size,
                module.k_window_size,
                table.device,
            )
            query_tokens = module.window_size * module.window_size
            key_tokens = module.overlap_win_size * module.overlap_win_size
            with torch.inference_mode():
                bias = table[index.reshape(-1)]
                bias = bias.view(
                    query_tokens, key_tokens, module.num_heads,
                ).permute(2, 0, 1).contiguous().unsqueeze(0)
            deployment._set_deployment_buffer(
                module, "_deployment_relative_position_bias", bias,
            )
            module._use_batch_axis_kv = True
            bias_bytes = bias.numel() * bias.element_size()
            static_constant_bytes += bias_bytes
            ocab_records.append(
                {
                    "path": path,
                    "shape": tuple(bias.shape),
                    "dtype": str(bias.dtype),
                    "bytes": bias_bytes,
                }
            )

        device = deployment.conv_first.weight.device
        dtype = deployment.conv_first.weight.dtype
        # These three boundary classes depend only on window geometry. Build
        # them from the smallest valid grid so dynamic H/W never leak the trace
        # example into the exported constants.
        canonical_feature_size = 2 * deployment.window_size
        right_mask, bottom_mask, corner_mask = (
            deployment._get_shifted_export_boundary_masks(
                canonical_feature_size, canonical_feature_size, device,
            )
        )
        right_mask = right_mask.to(dtype=dtype)
        bottom_mask = bottom_mask.to(dtype=dtype)
        corner_mask = corner_mask.to(dtype=dtype)

        shifted_records: list[dict[str, Any]] = []
        for path, module in deployment.named_modules():
            if not isinstance(module, ACTBlock) or module.shift_size <= 0:
                continue
            if not isinstance(module.attn, WindowAttentionRFB):
                raise RuntimeError(
                    f"{path} is shifted but does not use full-width attention"
                )
            with torch.inference_mode():
                bq, bk = module.attn.neural_bias()
                rank_bias = torch.einsum("hnr,hmr->hnm", bq, bk).unsqueeze(0)
                combined = (
                    (rank_bias + right_mask).contiguous(),
                    (rank_bias + bottom_mask).contiguous(),
                    (rank_bias + corner_mask).contiguous(),
                )
            for name, value in zip(
                (
                    "_deployment_right_bias",
                    "_deployment_bottom_bias",
                    "_deployment_corner_bias",
                ),
                combined,
                strict=True,
            ):
                deployment._set_deployment_buffer(module.attn, name, value)
                static_constant_bytes += value.numel() * value.element_size()
            module._use_static_split_attention = True
            shifted_records.append(
                {
                    "path": path,
                    "bias_shape": tuple(combined[0].shape),
                    "dtype": str(combined[0].dtype),
                    "bytes": sum(
                        value.numel() * value.element_size()
                        for value in combined
                    ),
                }
            )

        deployment._onnx_deployment_input_shape = shape
        deployment._onnx_deployment_dynamic_spatial = dynamic_spatial
        deployment._onnx_deployment_feature_shape = None if dynamic_spatial else (
            feature_height, feature_width,
        )
        deployment.set_tensorrt_export_mode(True)
        if dynamic_spatial:
            # The legacy exporter is required to retain symbolic H/W, but
            # PyTorch 2.14 lowers SDPA's scale through COMPLEX128 casts that
            # TensorRT cannot parse. The explicit QK^T/softmax/V path is
            # mathematically equivalent and still matches TensorRT's fused-MHA
            # pattern after a fixed-shape engine profile is applied.
            for module in deployment.modules():
                if isinstance(
                    module,
                    (WindowAttentionRFB, NarrowWindowAttention, OCAB),
                ):
                    module.force_math_attention = True
        deployment.requires_grad_(False)
        deployment.onnx_deployment_inventory = {
            "input_shape": shape,
            "example_shape": shape,
            "dynamic_spatial": dynamic_spatial,
            "feature_shape": (
                "dynamic" if dynamic_spatial
                else (feature_height, feature_width)
            ),
            "attention_export": (
                "explicit_math" if dynamic_spatial else "sdpa"
            ),
            "q4r_route": "split4",
            "folded_edbb": folded_edbb,
            "static_rank_factors": factor_records,
            "static_ocab_biases": ocab_records,
            "static_shifted_splits": shifted_records,
            "batch_axis_ocab": len(ocab_records),
            "static_constant_bytes": static_constant_bytes,
        }
        if precision == "tensorrt_mixed":
            return _DRFTTensorRTMixedPrecision(deployment)
        return deployment

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (ImageLayerNorm, AffineTransform)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            # Use xavier for GELU/SiLU activations (not relu)
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def prepare_for_compile(
        self,
        input_shape: tuple[int, int, int, int] | None = None,
        input_dtype: torch.dtype | None = None,
    ) -> None:
        """Prime compile-only training rewrites and immutable device caches.

        OCAB's relative-position index is shared by every compatible block and
        does not depend on input data. Creating it during the first compiled
        forward mutates the global cache and otherwise forces a full second
        compilation when the cache changes from cold to warm.
        """
        del input_dtype
        device = self.mean.device
        if self.training and device.type == "cuda":
            for module in self.modules():
                if isinstance(module, AttentionBlocks):
                    module.use_compiled_checkpoint_policy = True
                elif isinstance(module, OCAB):
                    module._use_batch_axis_kv = True
                    module.use_fused_qkv_windows_training = (
                        _OCAB_FUSED_QKV_WINDOWS_AVAILABLE
                    )
                    module.use_window_order_projection_training = True
                    module.use_compact_relative_bias_training = (
                        _OCAB_COMPACT_BIAS_AVAILABLE
                    )
                    module.use_compact_unfold_training = (
                        _OCAB_COMPACT_UNFOLD_AVAILABLE
                    )

        for module in self.modules():
            if isinstance(module, OCAB):
                _get_shared_relative_position_index(
                    module.q_window_size,
                    module.k_window_size,
                    device,
                )

        if input_shape is None:
            return

        batch, _, height, width = input_shape
        padded_height = (
            (height + self.window_size - 1) // self.window_size
        ) * self.window_size
        padded_width = (
            (width + self.window_size - 1) // self.window_size
        ) * self.window_size
        if self.attn_type == "masked":
            self._get_shifted_dense_mask(
                batch,
                padded_height,
                padded_width,
                device,
            )
            self._get_batch_indices(
                batch,
                padded_height,
                padded_width,
                device,
            )
        elif self.attn_type == "hybrid":
            self._get_shifted_block_mask(
                batch,
                padded_height,
                padded_width,
                device,
            )

            self._get_batch_indices(
                batch,
                padded_height,
                padded_width,
                device,
            )

    def distillation_feature_names(self) -> tuple[str, ...]:
        """Sparse semantic endpoints for frozen-teacher feature distillation."""
        return tuple(
            [f"rhag.{group_index}.residual" for group_index in range(self.num_layers)]
            + ["trunk.deep"]
        )

    @staticmethod
    def _proportional_endpoint(
        source_index: int, source_count: int, target_count: int
    ) -> int:
        """Map a source endpoint to the same relative target depth."""
        return min(
            target_count - 1,
            ((source_index + 1) * target_count + source_count - 1)
            // source_count
            - 1,
        )

    def distillation_feature_pairs(
        self, teacher: "DRFT"
    ) -> tuple[tuple[int, int, str, str], ...]:
        """Map student RHAG endpoints to proportional teacher endpoints."""
        if not isinstance(teacher, DRFT):
            raise TypeError("DRFT distillation requires a DRFT teacher")
        if self.upscale != teacher.upscale or self.reconstruction != teacher.reconstruction:
            raise ValueError(
                "DRFT student and teacher must share scale and reconstruction topology"
            )
        if teacher.num_layers < self.num_layers:
            raise ValueError(
                "DRFT teacher must have at least as many RHAGs as the student"
            )

        pairs: list[tuple[int, int, str, str]] = []
        for student_group in range(self.num_layers):
            teacher_group = self._proportional_endpoint(
                student_group, self.num_layers, teacher.num_layers
            )
            student_name = f"rhag.{student_group}.residual"
            teacher_name = f"rhag.{teacher_group}.residual"
            pairs.append(
                (student_group, teacher_group, student_name, teacher_name)
            )
        pairs.append(
            (
                self.num_layers,
                teacher.num_layers,
                "trunk.deep",
                "trunk.deep",
            )
        )
        return tuple(pairs)

    def forward_features(
        self,
        x: torch.Tensor,
        return_distillation_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x_size = (x.shape[2], x.shape[3])
        H, W = x_size
        distillation_features: list[torch.Tensor] = []

        # Compute per-window routing info for shifted ACTBlocks.
        # hybrid_ctx = (region_mask, int_idx_full, bnd_idx_full) passed through
        # the layer hierarchy. Non-shifted blocks ignore it entirely and use
        # Flash. Shifted blocks split windows: interior → Flash, boundary → masked/Flex.
        # Both mask types and batch indices are cached at model level.
        # TensorRT export must never trace the FlexAttention BlockMask route,
        # even when the checkpoint was trained with attn_type='hybrid'.  The
        # masked route is weight-identical and encodes the same region semantics.
        if self._onnx_deployment_dynamic_spatial:
            # Q4R blocks carry the three shape-independent boundary biases.
            # Window-grid extents remain symbolic inside their slice/reshape
            # path, so no runtime mask construction is needed.
            hybrid_ctx = ((), None, None)
        elif self._onnx_deployment_feature_shape is not None:
            if (H, W) != self._onnx_deployment_feature_shape:
                raise RuntimeError(
                    "this DRFT deployment graph is static to feature shape "
                    f"{self._onnx_deployment_feature_shape}, got {(H, W)}"
                )
            # Static split blocks carry their own three boundary biases and do
            # not need runtime masks or gather/scatter indices.
            hybrid_ctx = ((), None, None)
        elif self.force_tensorrt_export_mode:
            B = x.shape[0]
            boundary_masks = self._get_shifted_export_boundary_masks(H, W, x.device)
            hybrid_ctx = (boundary_masks, None, None)
        elif self.attn_type == 'masked':
            B = x.shape[0]
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
            layer_output = layer(x, x_size, hybrid_ctx)
            assert isinstance(layer_output, torch.Tensor)
            x = layer_output
            if return_distillation_features:
                distillation_features.append(
                    x.transpose(1, 2).reshape(
                        x.shape[0], self.embed_dim, x_size[0], x_size[1]
                    )
                )

        # Final norm
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        if return_distillation_features:
            return x, tuple(distillation_features)
        return x

    def forward(
        self,
        x: torch.Tensor,
        return_distillation_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        export_tracing = torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()
        if self._onnx_deployment_input_shape is not None and not export_tracing:
            expected_shape = self._onnx_deployment_input_shape
            actual_shape = tuple(x.shape)
            shape_matches = (
                actual_shape[:2] == expected_shape[:2]
                if self._onnx_deployment_dynamic_spatial
                else actual_shape == expected_shape
            )
            if not shape_matches:
                contract = "fixed N/C with dynamic H/W" if (
                    self._onnx_deployment_dynamic_spatial
                ) else "static input shape"
                raise RuntimeError(
                    f"this DRFT deployment graph requires {contract} "
                    f"{expected_shape}, got {actual_shape}"
                )
        H, W = x.shape[2:]

        # Normalize input (use local variable to avoid modifying buffer during forward)
        mean = self.mean.to(dtype=x.dtype, device=x.device)
        x = (x - mean) * self.img_range

        # Pad to multiple of window size
        x = pad_to_multiple(x, self.window_size)

        # Shallow features
        shallow = self.conv_first(x)
        distillation_features: list[torch.Tensor] = []

        # Deep features
        deep_output = self.forward_features(
            shallow,
            return_distillation_features=return_distillation_features,
        )
        if return_distillation_features:
            assert isinstance(deep_output, tuple)
            deep, trunk_features = deep_output
            distillation_features.extend(trunk_features)
        else:
            assert isinstance(deep_output, torch.Tensor)
            deep = deep_output
        deep = self.conv_after_body(deep) + shallow
        if return_distillation_features:
            distillation_features.append(deep)

        # Upsampling
        x = self.conv_before_upsample(deep)
        x = self.upsample(x)
        x = self.conv_last(x)

        # Denormalize
        x = x / self.img_range + mean

        # Crop to output size
        x = x[:, :, :H * self.upscale, :W * self.upscale]
        if return_distillation_features:
            return x, tuple(distillation_features)
        return x

    def fold_reparam_conv(self) -> None:
        """Fold all reparameterizable conv blocks for inference optimization."""
        if self.training:
            raise RuntimeError("Call model.eval() before fold_reparam_conv()")
        for module in self.modules():
            if isinstance(module, EDBBConvBlock):
                module.fold()

    def unfold_reparam_conv(self) -> None:
        """Unfold reparameterizable conv blocks back to training mode."""
        for module in self.modules():
            if isinstance(module, EDBBConvBlock):
                module.unfold()

    def set_export_attention_mode(self, enabled: bool = True) -> None:
        """Force math-attention path in all attention modules.

        Useful when exporter/tracer does not set torch.onnx.is_in_onnx_export().
        No parameters or buffers are modified, so checkpoint compatibility is
        unchanged.
        """
        for module in self.modules():
            if isinstance(module, (WindowAttentionRFB, NarrowWindowAttention, OCAB)):
                module.force_math_attention = enabled

    def set_tensorrt_export_mode(self, enabled: bool = True) -> None:
        """Use a TensorRT/ONNX-friendly attention graph during export.

        This keeps weights and checkpoint structure unchanged while preparing
        the graph for TensorRT export:
        - disable module-local eval caches that poison export
        - only fall back to explicit math attention where masking requires it
        """
        self.force_tensorrt_export_mode = enabled
        for module in self.modules():
            if isinstance(module, (WindowAttentionRFB, NarrowWindowAttention, OCAB)):
                module.force_math_attention = False
            if isinstance(module, (WindowAttentionRFB, NarrowWindowAttention)):
                module.force_additive_attention = False
            if isinstance(module, ACTBlock):
                module.force_tensorrt_export_mode = enabled
            if isinstance(module, (ChannelAttention, RankFactoredNeuralBias)):
                module.force_tensorrt_export_mode = enabled


class _DRFTTensorRTMixedPrecision(nn.Module):
    """TensorRT export wrapper with the admitted DRFT precision layout."""

    body_dtype = torch.bfloat16
    tail_dtype = torch.float16
    io_dtype = torch.float32
    fp16_regions = ("upsample", "conv_last")
    precision_policy_id = "drft_fp32_io_bf16_body_fp16_tail_v1"
    onnx_deployment_precision_handled = True

    def __init__(self, inner: DRFT) -> None:
        super().__init__()
        if inner._onnx_deployment_input_shape is None:
            raise RuntimeError(
                "mixed-precision export requires a prepared DRFT deployment"
            )

        self.inner = inner.to(dtype=self.body_dtype)
        self.inner.upsample.to(dtype=self.tail_dtype)
        self.inner.conv_last.to(dtype=self.tail_dtype)

        inventory = dict(self.inner.onnx_deployment_inventory)
        inventory.update(
            {
                "precision_policy": self.precision_policy_id,
                "io_dtype": str(self.io_dtype),
                "body_dtype": str(self.body_dtype),
                "fp16_regions": list(self.fp16_regions),
            }
        )
        self.onnx_deployment_inventory = inventory
        self.eval()

    def set_tensorrt_export_mode(self, enabled: bool = True) -> None:
        self.inner.set_tensorrt_export_mode(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected_shape = self.inner._onnx_deployment_input_shape
        export_tracing = torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()
        if not export_tracing:
            actual_shape = tuple(x.shape)
            shape_matches = (
                actual_shape[:2] == expected_shape[:2]
                if self.inner._onnx_deployment_dynamic_spatial
                else actual_shape == expected_shape
            )
            if not shape_matches:
                contract = "fixed N/C with dynamic H/W" if (
                    self.inner._onnx_deployment_dynamic_spatial
                ) else "static input shape"
                raise RuntimeError(
                    f"this DRFT deployment graph requires {contract} "
                    f"{expected_shape}, got {actual_shape}"
                )
        if x.dtype is not self.io_dtype:
            raise RuntimeError(
                f"DRFT TensorRT export requires FP32 input, got {x.dtype}"
            )

        height, width = x.shape[2:]
        x = x.to(dtype=self.body_dtype)
        mean = self.inner.mean.to(dtype=self.body_dtype, device=x.device)
        x = (x - mean) * self.inner.img_range
        x = pad_to_multiple(x, self.inner.window_size)

        shallow = self.inner.conv_first(x)
        deep = self.inner.forward_features(shallow)
        deep = self.inner.conv_after_body(deep) + shallow

        x = self.inner.conv_before_upsample(deep)
        x = self.inner.upsample(x.to(dtype=self.tail_dtype))
        x = self.inner.conv_last(x)
        x = x.to(dtype=self.body_dtype)

        x = x / self.inner.img_range + mean
        x = x[:, :, : height * self.inner.upscale, : width * self.inner.upscale]
        return x.to(dtype=self.io_dtype)


# =============================================================================
# Model Factory Functions
# =============================================================================

def _resolve_ocab_window_size(
    window_size: int,
    overlap_window_size: int | None,
) -> int:
    """Resolve the canonical OCAB span with 50% window overlap."""
    canonical_size = window_size + window_size // 2
    if overlap_window_size is not None and overlap_window_size != canonical_size:
        raise ValueError(
            "canonical DRFT requires overlap_window_size == "
            f"window_size + window_size // 2 ({canonical_size}), got "
            f"{overlap_window_size}"
        )
    return canonical_size


def _build_drft(
    *,
    scale: int,
    embed_dim: int,
    groups: int,
    full_heads: int,
    use_checkpoint: bool,
    window_size: int,
    overlap_window_size: int | None,
    drop_path_rate: float,
    attn_type: ATTN_TYPE,
    rank: int,
    mlp_ratio: float = 2.667,
    depth: int = 6,
    dense_skip: bool = True,
    reconstruction: RECONSTRUCTION_TYPE = 'progressive',
    edbb_depth_multiplier: float = 1.0,
    channel_squeeze_factor: int = 16,
    iln_eps: float = 1e-4,
    rhag_layer_scale_init: float | None = None,
    **kwargs,
) -> DRFT:
    overlap_window_size = _resolve_ocab_window_size(
        window_size,
        overlap_window_size,
    )
    return DRFT(
        upscale=scale,
        embed_dim=embed_dim,
        depths=(depth,) * groups,
        num_heads=(full_heads,) * groups,
        unshifted_num_heads=(full_heads,) * groups,
        unshifted_attention_dim=(embed_dim,) * groups,
        window_size=window_size,
        overlap_window_size=overlap_window_size,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        rhag_layer_scale_init=rhag_layer_scale_init,
        use_checkpoint=use_checkpoint,
        dense_skip=dense_skip,
        attn_type=attn_type,
        rank=rank,
        reconstruction=reconstruction,
        full_width_unshifted=True,
        edbb_depth_multiplier=edbb_depth_multiplier,
        channel_squeeze_factor=channel_squeeze_factor,
        iln_eps=iln_eps,
        **kwargs,
    )


def drft_nano(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 16,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.05,
    attn_type: ATTN_TYPE = 'masked',
    rank: int = 32,
    iln_eps: float = 1e-4,
    **kwargs
) -> DRFT:
    """C32, one two-block RHAG, one full-width head, and direct reconstruction."""
    return _build_drft(
        scale=scale,
        embed_dim=32,
        groups=1,
        full_heads=1,
        use_checkpoint=use_checkpoint,
        window_size=window_size,
        overlap_window_size=overlap_window_size,
        drop_path_rate=drop_path_rate,
        attn_type=attn_type,
        rank=rank,
        iln_eps=iln_eps,
        depth=2,
        dense_skip=False,
        reconstruction='direct',
        **kwargs,
    )


def drft_micro(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = 'masked',
    rank: int = 32,
    iln_eps: float = 1e-4,
    **kwargs
) -> DRFT:
    """C64, one RHAG, and two full-width heads in every local block."""
    return _build_drft(
        scale=scale,
        embed_dim=64,
        groups=1,
        depth=6,
        full_heads=2,
        use_checkpoint=use_checkpoint,
        window_size=window_size,
        overlap_window_size=overlap_window_size,
        drop_path_rate=drop_path_rate,
        attn_type=attn_type,
        rank=rank,
        iln_eps=iln_eps,
        **kwargs,
    )


def drft_light(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = 'masked',
    rank: int = 32,
    iln_eps: float = 1e-4,
    embed_dim: int = 96,
    groups: int = 2,
    full_heads: int = 3,
    **kwargs
) -> DRFT:
    """DRFT-Light, canonically C96/G2 with three full-width heads."""
    return _build_drft(
        scale=scale,
        embed_dim=embed_dim,
        groups=groups,
        depth=6,
        full_heads=full_heads,
        use_checkpoint=use_checkpoint,
        window_size=window_size,
        overlap_window_size=overlap_window_size,
        drop_path_rate=drop_path_rate,
        attn_type=attn_type,
        rank=rank,
        iln_eps=iln_eps,
        **kwargs,
    )


def drft_xs(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = 'masked',
    rank: int = 32,
    iln_eps: float = 1e-4,
    **kwargs
) -> DRFT:
    """C128, four RHAGs, and four full-width heads in every local block."""
    return _build_drft(
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
        rank=rank,
        iln_eps=iln_eps,
        **kwargs,
    )


def drft_s(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = 'masked',
    rank: int = 32,
    iln_eps: float = 1e-4,
    **kwargs
) -> DRFT:
    """C160/G6 with 5x32 full-width attention in every local block."""
    return _build_drft(
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
        rank=rank,
        iln_eps=iln_eps,
        **kwargs,
    )


def drft_m(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = 'masked',
    rank: int = 32,
    iln_eps: float = 1e-4,
    **kwargs
) -> DRFT:
    """C192/G8 with 6x32 full-width attention in every local block."""
    return _build_drft(
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
        rank=rank,
        iln_eps=iln_eps,
        **kwargs,
    )


def drft_m_safe(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = 'masked',
    rank: int = 32,
    iln_eps: float = 1e-4,
    **kwargs,
) -> DRFT:
    """Compatibility spelling for the now-canonical full-width DRFT-M."""
    return drft_m(
        scale=scale,
        use_checkpoint=use_checkpoint,
        window_size=window_size,
        overlap_window_size=overlap_window_size,
        drop_path_rate=drop_path_rate,
        attn_type=attn_type,
        rank=rank,
        iln_eps=iln_eps,
        **kwargs,
    )


def drft_l(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = 'masked',
    rank: int = 32,
    iln_eps: float = 1e-4,
    **kwargs
) -> DRFT:
    """C224/G10 with 7x32 full-width attention in every local block."""
    return _build_drft(
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
        rank=rank,
        iln_eps=iln_eps,
        **kwargs,
    )


def drft_xl(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = 'masked',
    rank: int = 32,
    iln_eps: float = 1e-4,
    rhag_layer_scale_init: float = 1e-4,
    **kwargs,
) -> DRFT:
    """C256/G14 with 8x32 full-width attention in every local block."""
    return _build_drft(
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
        rank=rank,
        iln_eps=iln_eps,
        rhag_layer_scale_init=rhag_layer_scale_init,
        **kwargs,
    )


def _build_drft_teacher(
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
    rank: int,
    mlp_ratio: float,
    edbb_depth_multiplier: float,
    channel_squeeze_factor: int,
    rhag_layer_scale_init: float | None = None,
    dense_skip: bool = True,
    reconstruction: RECONSTRUCTION_TYPE = "progressive",
    **kwargs,
) -> DRFT:
    """Build a width-compatible teacher with a deeper refinement hierarchy.

    Teacher strength comes from additional RHAG/ACT refinement and full-width
    unshifted attention. FFN, EDBB, and channel-attention internals retain the
    canonical DRFT ratios unless explicitly overridden.
    """
    return _build_drft(
        scale=scale,
        embed_dim=embed_dim,
        groups=groups,
        depth=depth,
        full_heads=full_heads,
        use_checkpoint=use_checkpoint,
        window_size=window_size,
        overlap_window_size=overlap_window_size,
        drop_path_rate=drop_path_rate,
        attn_type=attn_type,
        rank=rank,
        mlp_ratio=mlp_ratio,
        edbb_depth_multiplier=edbb_depth_multiplier,
        channel_squeeze_factor=channel_squeeze_factor,
        rhag_layer_scale_init=rhag_layer_scale_init,
        dense_skip=dense_skip,
        reconstruction=reconstruction,
        **kwargs,
    )


def drft_nano_teacher(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 16,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.05,
    attn_type: ATTN_TYPE = 'masked',
    rank: int = 32,
    groups: int = 2,
    depth: int = 4,
    mlp_ratio: float = 2.667,
    edbb_depth_multiplier: float = 1.0,
    channel_squeeze_factor: int = 16,
    **kwargs,
) -> DRFT:
    return _build_drft_teacher(
        scale=scale, embed_dim=32, groups=groups, depth=depth,
        full_heads=1,
        use_checkpoint=use_checkpoint,
        window_size=window_size, overlap_window_size=overlap_window_size,
        drop_path_rate=drop_path_rate, attn_type=attn_type, rank=rank,
        mlp_ratio=mlp_ratio, edbb_depth_multiplier=edbb_depth_multiplier,
        channel_squeeze_factor=channel_squeeze_factor, dense_skip=False,
        reconstruction="direct", **kwargs,
    )


def drft_micro_teacher(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = 'masked',
    rank: int = 32,
    groups: int = 2,
    depth: int = 6,
    mlp_ratio: float = 2.667,
    edbb_depth_multiplier: float = 1.0,
    channel_squeeze_factor: int = 16,
    **kwargs,
) -> DRFT:
    return _build_drft_teacher(
        scale=scale, embed_dim=64, groups=groups, depth=depth,
        full_heads=2,
        use_checkpoint=use_checkpoint,
        window_size=window_size, overlap_window_size=overlap_window_size,
        drop_path_rate=drop_path_rate, attn_type=attn_type, rank=rank,
        mlp_ratio=mlp_ratio, edbb_depth_multiplier=edbb_depth_multiplier,
        channel_squeeze_factor=channel_squeeze_factor, **kwargs,
    )


def drft_light_teacher(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = 'masked',
    rank: int = 32,
    groups: int = 4,
    depth: int = 6,
    mlp_ratio: float = 2.667,
    edbb_depth_multiplier: float = 1.0,
    channel_squeeze_factor: int = 16,
    **kwargs,
) -> DRFT:
    return _build_drft_teacher(
        scale=scale, embed_dim=96, groups=groups, depth=depth,
        full_heads=3,
        use_checkpoint=use_checkpoint,
        window_size=window_size, overlap_window_size=overlap_window_size,
        drop_path_rate=drop_path_rate, attn_type=attn_type, rank=rank,
        mlp_ratio=mlp_ratio, edbb_depth_multiplier=edbb_depth_multiplier,
        channel_squeeze_factor=channel_squeeze_factor, **kwargs,
    )


def drft_xs_teacher(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = 'masked',
    rank: int = 32,
    groups: int = 6,
    depth: int = 6,
    mlp_ratio: float = 2.667,
    edbb_depth_multiplier: float = 1.0,
    channel_squeeze_factor: int = 16,
    **kwargs,
) -> DRFT:
    return _build_drft_teacher(
        scale=scale, embed_dim=128, groups=groups, depth=depth,
        full_heads=4,
        use_checkpoint=use_checkpoint,
        window_size=window_size, overlap_window_size=overlap_window_size,
        drop_path_rate=drop_path_rate, attn_type=attn_type, rank=rank,
        mlp_ratio=mlp_ratio, edbb_depth_multiplier=edbb_depth_multiplier,
        channel_squeeze_factor=channel_squeeze_factor, **kwargs,
    )


def drft_s_teacher(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = 'masked',
    rank: int = 32,
    groups: int = 9,
    depth: int = 6,
    mlp_ratio: float = 2.667,
    edbb_depth_multiplier: float = 1.0,
    channel_squeeze_factor: int = 16,
    **kwargs,
) -> DRFT:
    return _build_drft_teacher(
        scale=scale, embed_dim=160, groups=groups, depth=depth,
        full_heads=5,
        use_checkpoint=use_checkpoint,
        window_size=window_size, overlap_window_size=overlap_window_size,
        drop_path_rate=drop_path_rate, attn_type=attn_type, rank=rank,
        mlp_ratio=mlp_ratio, edbb_depth_multiplier=edbb_depth_multiplier,
        channel_squeeze_factor=channel_squeeze_factor, **kwargs,
    )


def drft_m_teacher(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = 'masked',
    rank: int = 32,
    groups: int = 12,
    depth: int = 6,
    mlp_ratio: float = 2.667,
    edbb_depth_multiplier: float = 1.0,
    channel_squeeze_factor: int = 16,
    **kwargs,
) -> DRFT:
    return _build_drft_teacher(
        scale=scale, embed_dim=192, groups=groups, depth=depth,
        full_heads=6,
        use_checkpoint=use_checkpoint,
        window_size=window_size, overlap_window_size=overlap_window_size,
        drop_path_rate=drop_path_rate, attn_type=attn_type, rank=rank,
        mlp_ratio=mlp_ratio, edbb_depth_multiplier=edbb_depth_multiplier,
        channel_squeeze_factor=channel_squeeze_factor, **kwargs,
    )


def drft_l_teacher(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = 'masked',
    rank: int = 32,
    groups: int = 15,
    depth: int = 6,
    mlp_ratio: float = 2.667,
    edbb_depth_multiplier: float = 1.0,
    channel_squeeze_factor: int = 16,
    rhag_layer_scale_init: float = 1e-4,
    **kwargs,
) -> DRFT:
    return _build_drft_teacher(
        scale=scale, embed_dim=224, groups=groups, depth=depth,
        full_heads=7,
        use_checkpoint=use_checkpoint,
        window_size=window_size, overlap_window_size=overlap_window_size,
        drop_path_rate=drop_path_rate, attn_type=attn_type, rank=rank,
        mlp_ratio=mlp_ratio, edbb_depth_multiplier=edbb_depth_multiplier,
        channel_squeeze_factor=channel_squeeze_factor,
        rhag_layer_scale_init=rhag_layer_scale_init, **kwargs,
    )


def drft_xl_teacher(
    scale: int = 4,
    use_checkpoint: bool = False,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    drop_path_rate: float = 0.1,
    attn_type: ATTN_TYPE = 'masked',
    rank: int = 32,
    groups: int = 21,
    depth: int = 6,
    mlp_ratio: float = 2.667,
    edbb_depth_multiplier: float = 1.0,
    channel_squeeze_factor: int = 16,
    rhag_layer_scale_init: float = 1e-4,
    **kwargs,
) -> DRFT:
    return _build_drft_teacher(
        scale=scale, embed_dim=256, groups=groups, depth=depth,
        full_heads=8,
        use_checkpoint=use_checkpoint,
        window_size=window_size, overlap_window_size=overlap_window_size,
        drop_path_rate=drop_path_rate, attn_type=attn_type, rank=rank,
        mlp_ratio=mlp_ratio, edbb_depth_multiplier=edbb_depth_multiplier,
        channel_squeeze_factor=channel_squeeze_factor,
        rhag_layer_scale_init=rhag_layer_scale_init, **kwargs,
    )


# =============================================================================
# Optional: traiNNer Registry
# =============================================================================

try:
    from traiNNer.utils.registry import ARCH_REGISTRY

    ARCH_REGISTRY.register(drft_nano)
    ARCH_REGISTRY.register(drft_micro)
    ARCH_REGISTRY.register(drft_light)
    ARCH_REGISTRY.register(drft_xs)
    ARCH_REGISTRY.register(drft_s)
    ARCH_REGISTRY.register(drft_m)
    ARCH_REGISTRY.register(drft_l)
    ARCH_REGISTRY.register(drft_xl)
    ARCH_REGISTRY.register(drft_nano_teacher)
    ARCH_REGISTRY.register(drft_micro_teacher)
    ARCH_REGISTRY.register(drft_light_teacher)
    ARCH_REGISTRY.register(drft_xs_teacher)
    ARCH_REGISTRY.register(drft_s_teacher)
    ARCH_REGISTRY.register(drft_m_teacher)
    ARCH_REGISTRY.register(drft_l_teacher)
    ARCH_REGISTRY.register(drft_xl_teacher)

except ImportError:
    pass  # traiNNer not available
