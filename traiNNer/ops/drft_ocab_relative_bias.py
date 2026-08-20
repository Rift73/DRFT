# ruff: noqa: ANN001, ANN202, I001, N803
"""Structure-aware OCAB relative-bias expansion for compiled training.

The dense OCAB bias is a gather from a compact learned table. PyTorch's
generic deterministic indexing backward sorts and reduces the repeated table
indices. The geometry is regular, so one Triton program can instead own each
table entry and reduce exactly the query/key pairs that map to it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


_FP32 = 0
_BF16 = 1
_FP16 = 2


def _dtype_from_code(code: int) -> torch.dtype:
    if code == _FP32:
        return torch.float32
    if code == _BF16:
        return torch.bfloat16
    if code == _FP16:
        return torch.float16
    raise ValueError(f"unsupported OCAB bias dtype code: {code}")


def ocab_bias_dtype_code(dtype: torch.dtype) -> int:
    if dtype is torch.float32:
        return _FP32
    if dtype is torch.bfloat16:
        return _BF16
    if dtype is torch.float16:
        return _FP16
    raise ValueError(f"unsupported OCAB bias dtype: {dtype}")


@triton.jit
def _expand_relative_bias_kernel(
    table_ptr,
    output_ptr,
    query_tokens: tl.constexpr,
    key_tokens: tl.constexpr,
    query_width: tl.constexpr,
    key_width: tl.constexpr,
    query_height: tl.constexpr,
    num_heads: tl.constexpr,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < total_elements

    per_head = query_tokens * key_tokens
    head = offsets // per_head
    head_offset = offsets - head * per_head
    query_index = head_offset // key_tokens
    key_index = head_offset - query_index * key_tokens

    query_y = query_index // query_width
    query_x = query_index - query_y * query_width
    key_y = key_index // key_width
    key_x = key_index - key_y * key_width

    relative_y = key_y - query_y + query_height - 1
    relative_x = key_x - query_x + query_width - 1
    relative_width = query_width + key_width - 1
    table_row = relative_y * relative_width + relative_x
    value = tl.load(table_ptr + table_row * num_heads + head, mask=valid)
    tl.store(output_ptr + offsets, value, mask=valid)


@triton.jit
def _reduce_relative_bias_gradient_kernel(
    grad_output_ptr,
    grad_table_ptr,
    query_tokens: tl.constexpr,
    key_tokens: tl.constexpr,
    query_height: tl.constexpr,
    query_width: tl.constexpr,
    key_height: tl.constexpr,
    key_width: tl.constexpr,
    relative_height: tl.constexpr,
    relative_width: tl.constexpr,
    num_heads: tl.constexpr,
    BLOCK_QUERY: tl.constexpr,
):
    output_index = tl.program_id(0)
    head = output_index % num_heads
    relative_index = output_index // num_heads
    relative_y = relative_index // relative_width
    relative_x = relative_index - relative_y * relative_width
    delta_y = relative_y - (query_height - 1)
    delta_x = relative_x - (query_width - 1)

    query_index = tl.arange(0, BLOCK_QUERY)
    query_y = query_index // query_width
    query_x = query_index - query_y * query_width
    key_y = query_y + delta_y
    key_x = query_x + delta_x
    valid = (
        (query_index < query_tokens)
        & (key_y >= 0)
        & (key_y < key_height)
        & (key_x >= 0)
        & (key_x < key_width)
    )
    key_index = key_y * key_width + key_x
    dense_offset = (head * query_tokens + query_index) * key_tokens + key_index
    gradient = tl.load(
        grad_output_ptr + dense_offset,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    reduced = tl.sum(gradient, axis=0)
    tl.store(grad_table_ptr + relative_index * num_heads + head, reduced)


@triton_op("trainner_drft::ocab_relative_bias", mutates_args={})
def ocab_relative_bias(
    table: torch.Tensor,
    query_height: int,
    query_width: int,
    key_height: int,
    key_width: int,
    output_dtype_code: int,
) -> torch.Tensor:
    num_heads = table.shape[1]
    query_tokens = query_height * query_width
    key_tokens = key_height * key_width
    output = torch.empty(
        (1, num_heads, query_tokens, key_tokens),
        device=table.device,
        dtype=_dtype_from_code(output_dtype_code),
    )
    total_elements = output.numel()

    def grid(meta):
        return (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)

    wrap_triton(_expand_relative_bias_kernel)[grid](
        table,
        output,
        query_tokens,
        key_tokens,
        query_width,
        key_width,
        query_height,
        num_heads,
        total_elements,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return output


@triton_op("trainner_drft::ocab_relative_bias_backward", mutates_args={})
def ocab_relative_bias_backward(
    grad_output: torch.Tensor,
    table: torch.Tensor,
    query_height: int,
    query_width: int,
    key_height: int,
    key_width: int,
) -> torch.Tensor:
    grad_output = grad_output.contiguous()
    grad_table = torch.empty_like(table, memory_format=torch.contiguous_format)
    num_heads = table.shape[1]
    query_tokens = query_height * query_width
    key_tokens = key_height * key_width
    relative_height = query_height + key_height - 1
    relative_width = query_width + key_width - 1
    grid = (relative_height * relative_width * num_heads,)
    wrap_triton(_reduce_relative_bias_gradient_kernel)[grid](
        grad_output,
        grad_table,
        query_tokens,
        key_tokens,
        query_height,
        query_width,
        key_height,
        key_width,
        relative_height,
        relative_width,
        num_heads,
        BLOCK_QUERY=triton.next_power_of_2(query_tokens),
        num_warps=8,
    )
    return grad_table


def _setup_context(ctx, inputs, output) -> None:
    del output
    (
        table,
        query_height,
        query_width,
        key_height,
        key_width,
        _output_dtype_code,
    ) = inputs
    ctx.save_for_backward(table)
    ctx.query_height = query_height
    ctx.query_width = query_width
    ctx.key_height = key_height
    ctx.key_width = key_width


def _backward(ctx, grad_output: torch.Tensor):
    (table,) = ctx.saved_tensors
    grad_table = ocab_relative_bias_backward(
        grad_output,
        table,
        ctx.query_height,
        ctx.query_width,
        ctx.key_height,
        ctx.key_width,
    )
    return grad_table, None, None, None, None, None


torch.library.register_autograd(
    ocab_relative_bias,
    _backward,
    setup_context=_setup_context,
)


__all__ = ["ocab_bias_dtype_code", "ocab_relative_bias"]
