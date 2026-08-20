# ruff: noqa: ANN001, ANN202, N803
"""Fused deterministic Q/K/V window preparation for compiled DRFT OCAB.

The OCAB projection naturally produces ``B,H,W,3,C``.  The ordinary PyTorch
route then performs several large layout copies before and after ``Unfold`` to
construct the query windows and overlapping key/value windows consumed by
SDPA.  These Triton gathers write the final compact window layouts directly.

Backward assigns one program lane to each projected Q/K/V element.  Query
gradients have one source; overlapping key/value gradients are accumulated in
a fixed ascending window order.  There are no atomics or result-dependent
paths, so strict deterministic training remains supported.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _ocab_query_windows_forward_kernel(
    qkv_ptr,
    query_ptr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    window_size: tl.constexpr,
    output_width: tl.constexpr,
    windows_per_image: tl.constexpr,
    query_tokens: tl.constexpr,
    qkv_stride_batch: tl.constexpr,
    qkv_stride_height: tl.constexpr,
    qkv_stride_width: tl.constexpr,
    qkv_stride_kind: tl.constexpr,
    qkv_stride_channel: tl.constexpr,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < total_elements

    channel = offsets % channels
    remainder = offsets // channels
    query_index = remainder % query_tokens
    batch_window = remainder // query_tokens
    window_index = batch_window % windows_per_image
    batch = batch_window // windows_per_image

    window_y = window_index // output_width
    window_x = window_index - window_y * output_width
    query_y = query_index // window_size
    query_x = query_index - query_y * window_size
    input_y = window_y * window_size + query_y
    input_x = window_x * window_size + query_x

    input_offset = (
        batch * qkv_stride_batch
        + input_y * qkv_stride_height
        + input_x * qkv_stride_width
        + 0 * qkv_stride_kind
        + channel * qkv_stride_channel
    )
    value = tl.load(qkv_ptr + input_offset, mask=valid)
    tl.store(query_ptr + offsets, value, mask=valid)


@triton.jit
def _ocab_kv_windows_forward_kernel(
    qkv_ptr,
    kv_ptr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    window_size: tl.constexpr,
    overlap_size: tl.constexpr,
    padding: tl.constexpr,
    output_width: tl.constexpr,
    windows_per_image: tl.constexpr,
    overlap_tokens: tl.constexpr,
    batch_windows: tl.constexpr,
    qkv_stride_batch: tl.constexpr,
    qkv_stride_height: tl.constexpr,
    qkv_stride_width: tl.constexpr,
    qkv_stride_kind: tl.constexpr,
    qkv_stride_channel: tl.constexpr,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < total_elements

    channel = offsets % channels
    remainder = offsets // channels
    overlap_index = remainder % overlap_tokens
    remainder = remainder // overlap_tokens
    batch_window = remainder % batch_windows
    kind = remainder // batch_windows

    window_index = batch_window % windows_per_image
    batch = batch_window // windows_per_image
    window_y = window_index // output_width
    window_x = window_index - window_y * output_width
    overlap_y = overlap_index // overlap_size
    overlap_x = overlap_index - overlap_y * overlap_size
    input_y = window_y * window_size + overlap_y - padding
    input_x = window_x * window_size + overlap_x - padding
    valid_input = (
        valid & (input_y >= 0) & (input_y < height) & (input_x >= 0) & (input_x < width)
    )

    input_offset = (
        batch * qkv_stride_batch
        + input_y * qkv_stride_height
        + input_x * qkv_stride_width
        + (kind + 1) * qkv_stride_kind
        + channel * qkv_stride_channel
    )
    value = tl.load(qkv_ptr + input_offset, mask=valid_input, other=0.0)
    tl.store(kv_ptr + offsets, value, mask=valid)


@triton.jit
def _ocab_qkv_windows_backward_kernel(
    grad_query_ptr,
    grad_kv_ptr,
    grad_qkv_ptr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    window_size: tl.constexpr,
    overlap_size: tl.constexpr,
    padding: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    windows_per_image: tl.constexpr,
    query_tokens: tl.constexpr,
    overlap_tokens: tl.constexpr,
    max_cover: tl.constexpr,
    grad_query_stride_batch_window: tl.constexpr,
    grad_query_stride_token: tl.constexpr,
    grad_query_stride_channel: tl.constexpr,
    grad_kv_stride_kind: tl.constexpr,
    grad_kv_stride_batch_window: tl.constexpr,
    grad_kv_stride_token: tl.constexpr,
    grad_kv_stride_channel: tl.constexpr,
    grad_qkv_stride_batch: tl.constexpr,
    grad_qkv_stride_height: tl.constexpr,
    grad_qkv_stride_width: tl.constexpr,
    grad_qkv_stride_kind: tl.constexpr,
    grad_qkv_stride_channel: tl.constexpr,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < total_elements

    channel = offsets % channels
    remainder = offsets // channels
    kind = remainder % 3
    remainder = remainder // 3
    input_x = remainder % width
    remainder = remainder // width
    input_y = remainder % height
    batch = remainder // height

    query_window_y = input_y // window_size
    query_window_x = input_x // window_size
    query_window = query_window_y * output_width + query_window_x
    query_batch_window = batch * windows_per_image + query_window
    query_y = input_y - query_window_y * window_size
    query_x = input_x - query_window_x * window_size
    query_index = query_y * window_size + query_x
    query_offset = (
        query_batch_window * grad_query_stride_batch_window
        + query_index * grad_query_stride_token
        + channel * grad_query_stride_channel
    )
    query_gradient = tl.load(
        grad_query_ptr + query_offset,
        mask=valid & (kind == 0),
        other=0.0,
    ).to(tl.float32)

    kv_kind = tl.maximum(kind - 1, 0)
    base_window_y = (input_y + padding) // window_size
    base_window_x = (input_x + padding) // window_size
    kv_gradient = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Enumerate candidate windows in ascending row-major order.  With the
    # canonical 48/32 overlap, max_cover is two in each dimension.
    for cover_y in range(max_cover):
        window_y = base_window_y - (max_cover - 1 - cover_y)
        overlap_y = input_y + padding - window_y * window_size
        valid_y = (
            (window_y >= 0)
            & (window_y < output_height)
            & (overlap_y >= 0)
            & (overlap_y < overlap_size)
        )
        for cover_x in range(max_cover):
            window_x = base_window_x - (max_cover - 1 - cover_x)
            overlap_x = input_x + padding - window_x * window_size
            valid_window = (
                valid
                & (kind > 0)
                & valid_y
                & (window_x >= 0)
                & (window_x < output_width)
                & (overlap_x >= 0)
                & (overlap_x < overlap_size)
            )
            window_index = window_y * output_width + window_x
            batch_window = batch * windows_per_image + window_index
            overlap_index = overlap_y * overlap_size + overlap_x
            kv_offset = (
                kv_kind * grad_kv_stride_kind
                + batch_window * grad_kv_stride_batch_window
                + overlap_index * grad_kv_stride_token
                + channel * grad_kv_stride_channel
            )
            kv_gradient += tl.load(
                grad_kv_ptr + kv_offset,
                mask=valid_window,
                other=0.0,
            ).to(tl.float32)

    gradient = tl.where(kind == 0, query_gradient, kv_gradient)
    output_offset = (
        batch * grad_qkv_stride_batch
        + input_y * grad_qkv_stride_height
        + input_x * grad_qkv_stride_width
        + kind * grad_qkv_stride_kind
        + channel * grad_qkv_stride_channel
    )
    tl.store(grad_qkv_ptr + output_offset, gradient, mask=valid)


@triton_op("trainner_drft::ocab_qkv_windows", mutates_args={})
def ocab_qkv_windows(
    qkv: torch.Tensor,
    window_size: int,
    overlap_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, height, width, kinds, channels = qkv.shape
    if kinds != 3:
        raise ValueError(f"expected three Q/K/V projections, got {kinds}")
    if height % window_size != 0 or width % window_size != 0:
        raise ValueError("OCAB Q/K/V input must be divisible by window_size")
    if (overlap_size - window_size) % 2 != 0:
        raise ValueError("OCAB overlap padding must be symmetric")

    output_height = height // window_size
    output_width = width // window_size
    windows_per_image = output_height * output_width
    batch_windows = batch * windows_per_image
    query_tokens = window_size * window_size
    overlap_tokens = overlap_size * overlap_size
    padding = (overlap_size - window_size) // 2
    query = torch.empty(
        (batch_windows, query_tokens, channels),
        device=qkv.device,
        dtype=qkv.dtype,
    )
    kv = torch.empty(
        (2, batch_windows, overlap_tokens, channels),
        device=qkv.device,
        dtype=qkv.dtype,
    )
    qkv_strides = qkv.stride()

    query_elements = query.numel()

    def query_grid(meta):
        return (triton.cdiv(query_elements, meta["BLOCK_SIZE"]),)

    wrap_triton(_ocab_query_windows_forward_kernel)[query_grid](
        qkv,
        query,
        channels,
        height,
        width,
        window_size,
        output_width,
        windows_per_image,
        query_tokens,
        *qkv_strides,
        query_elements,
        BLOCK_SIZE=256,
        num_warps=4,
    )

    kv_elements = kv.numel()

    def kv_grid(meta):
        return (triton.cdiv(kv_elements, meta["BLOCK_SIZE"]),)

    wrap_triton(_ocab_kv_windows_forward_kernel)[kv_grid](
        qkv,
        kv,
        channels,
        height,
        width,
        window_size,
        overlap_size,
        padding,
        output_width,
        windows_per_image,
        overlap_tokens,
        batch_windows,
        *qkv_strides,
        kv_elements,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return query, kv


@triton_op("trainner_drft::ocab_qkv_windows_backward", mutates_args={})
def ocab_qkv_windows_backward(
    grad_query: torch.Tensor,
    grad_kv: torch.Tensor,
    batch: int,
    height: int,
    width: int,
    channels: int,
    qkv_stride_batch: int,
    qkv_stride_height: int,
    qkv_stride_width: int,
    qkv_stride_kind: int,
    qkv_stride_channel: int,
    window_size: int,
    overlap_size: int,
) -> torch.Tensor:
    qkv_shape = (batch, height, width, 3, channels)
    qkv_strides = (
        qkv_stride_batch,
        qkv_stride_height,
        qkv_stride_width,
        qkv_stride_kind,
        qkv_stride_channel,
    )
    grad_qkv = torch.empty_strided(
        qkv_shape,
        qkv_strides,
        device=grad_query.device,
        dtype=grad_query.dtype,
    )
    output_height = height // window_size
    output_width = width // window_size
    windows_per_image = output_height * output_width
    query_tokens = window_size * window_size
    overlap_tokens = overlap_size * overlap_size
    padding = (overlap_size - window_size) // 2
    max_cover = triton.cdiv(overlap_size, window_size)
    total_elements = grad_qkv.numel()

    def grid(meta):
        return (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)

    wrap_triton(_ocab_qkv_windows_backward_kernel)[grid](
        grad_query,
        grad_kv,
        grad_qkv,
        channels,
        height,
        width,
        window_size,
        overlap_size,
        padding,
        output_height,
        output_width,
        windows_per_image,
        query_tokens,
        overlap_tokens,
        max_cover,
        *grad_query.stride(),
        *grad_kv.stride(),
        *qkv_strides,
        total_elements,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return grad_qkv


def _setup_context(ctx, inputs, output) -> None:
    del output
    qkv, window_size, overlap_size = inputs
    ctx.qkv_shape = tuple(qkv.shape)
    ctx.qkv_strides = tuple(qkv.stride())
    ctx.window_size = window_size
    ctx.overlap_size = overlap_size


def _backward(
    ctx,
    grad_query: torch.Tensor,
    grad_kv: torch.Tensor,
):
    batch, height, width, _kinds, channels = ctx.qkv_shape
    grad_qkv = ocab_qkv_windows_backward(
        grad_query,
        grad_kv,
        batch,
        height,
        width,
        channels,
        *ctx.qkv_strides,
        ctx.window_size,
        ctx.overlap_size,
    )
    return grad_qkv, None, None


torch.library.register_autograd(
    ocab_qkv_windows,
    _backward,
    setup_context=_setup_context,
)


__all__ = ["ocab_qkv_windows"]
