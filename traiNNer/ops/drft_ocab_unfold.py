# ruff: noqa: ANN001, ANN202, N803
"""Deterministic OCAB overlap extraction for compiled DRFT training.

PyTorch's deterministic compiled backward for ``nn.Unfold`` lowers the
overlapping col2im reduction through a generic index/sort path.  OCAB has a
regular, fixed geometry, so every input element can instead gather the small
set of output-window gradients that contain it.  One Triton program owns each
input element: there are no atomics, races, or result-dependent paths.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _ocab_unfold_forward_kernel(
    input_ptr,
    output_ptr,
    channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    kernel_height: tl.constexpr,
    kernel_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    kernel_elements: tl.constexpr,
    output_windows: tl.constexpr,
    input_batch_stride: tl.constexpr,
    input_channel_stride: tl.constexpr,
    input_height_stride: tl.constexpr,
    input_width_stride: tl.constexpr,
    BLOCK_PATCH: tl.constexpr,
):
    program = tl.program_id(0)
    window_index = program % output_windows
    batch_channel = program // output_windows
    channel = batch_channel % channels
    batch = batch_channel // channels

    patch_index = tl.arange(0, BLOCK_PATCH)
    valid_patch = patch_index < kernel_elements
    kernel_y = patch_index // kernel_width
    kernel_x = patch_index - kernel_y * kernel_width
    output_y = window_index // output_width
    output_x = window_index - output_y * output_width
    input_y = output_y * stride_height + kernel_y - padding_height
    input_x = output_x * stride_width + kernel_x - padding_width
    valid_input = (
        valid_patch
        & (input_y >= 0)
        & (input_y < input_height)
        & (input_x >= 0)
        & (input_x < input_width)
    )

    input_offset = (
        batch * input_batch_stride
        + channel * input_channel_stride
        + input_y * input_height_stride
        + input_x * input_width_stride
    )
    value = tl.load(input_ptr + input_offset, mask=valid_input, other=0.0)
    output_offset = (
        (batch_channel * kernel_elements + patch_index) * output_windows
        + window_index
    )
    tl.store(output_ptr + output_offset, value, mask=valid_patch)


@triton.jit
def _ocab_unfold_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    kernel_height: tl.constexpr,
    kernel_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    kernel_elements: tl.constexpr,
    output_windows: tl.constexpr,
    input_elements: tl.constexpr,
    grad_input_batch_stride: tl.constexpr,
    grad_input_channel_stride: tl.constexpr,
    grad_input_height_stride: tl.constexpr,
    grad_input_width_stride: tl.constexpr,
    BLOCK_INPUT: tl.constexpr,
):
    batch_channel = tl.program_id(0)
    pixel_index = tl.arange(0, BLOCK_INPUT)
    valid_pixel = pixel_index < input_elements
    input_y = pixel_index // input_width
    input_x = pixel_index - input_y * input_width
    gradient = tl.zeros((BLOCK_INPUT,), dtype=tl.float32)

    # Fixed loop order is deterministic and covers every window containing the
    # input pixel.  OCAB's normal 48/32 geometry executes only four iterations.
    for output_y in range(output_height):
        kernel_y = input_y + padding_height - output_y * stride_height
        valid_y = (kernel_y >= 0) & (kernel_y < kernel_height)
        for output_x in range(output_width):
            kernel_x = input_x + padding_width - output_x * stride_width
            valid = (
                valid_pixel
                & valid_y
                & (kernel_x >= 0)
                & (kernel_x < kernel_width)
            )
            patch_index = kernel_y * kernel_width + kernel_x
            window_index = output_y * output_width + output_x
            grad_offset = (
                (batch_channel * kernel_elements + patch_index) * output_windows
                + window_index
            )
            gradient += tl.load(
                grad_output_ptr + grad_offset,
                mask=valid,
                other=0.0,
            ).to(tl.float32)

    channel = batch_channel % channels
    batch = batch_channel // channels
    output_offset = (
        batch * grad_input_batch_stride
        + channel * grad_input_channel_stride
        + input_y * grad_input_height_stride
        + input_x * grad_input_width_stride
    )
    tl.store(grad_input_ptr + output_offset, gradient, mask=valid_pixel)


@triton_op("trainner_drft::ocab_unfold2d", mutates_args={})
def ocab_unfold2d(
    input: torch.Tensor,
    kernel_height: int,
    kernel_width: int,
    stride_height: int,
    stride_width: int,
    padding_height: int,
    padding_width: int,
) -> torch.Tensor:
    batch, channels, input_height, input_width = input.shape
    output_height = (
        input_height + 2 * padding_height - kernel_height
    ) // stride_height + 1
    output_width = (
        input_width + 2 * padding_width - kernel_width
    ) // stride_width + 1
    kernel_elements = kernel_height * kernel_width
    output_windows = output_height * output_width
    input_batch_stride, input_channel_stride, input_height_stride, input_width_stride = (
        input.stride()
    )
    output = torch.empty(
        (batch, channels * kernel_elements, output_windows),
        device=input.device,
        dtype=input.dtype,
    )
    grid = (batch * channels * output_windows,)
    wrap_triton(_ocab_unfold_forward_kernel)[grid](
        input,
        output,
        channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        output_height,
        output_width,
        kernel_elements,
        output_windows,
        input_batch_stride,
        input_channel_stride,
        input_height_stride,
        input_width_stride,
        BLOCK_PATCH=triton.next_power_of_2(kernel_elements),
        num_warps=8,
    )
    return output


@triton_op("trainner_drft::ocab_unfold2d_backward", mutates_args={})
def ocab_unfold2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    kernel_height: int,
    kernel_width: int,
    stride_height: int,
    stride_width: int,
    padding_height: int,
    padding_width: int,
) -> torch.Tensor:
    grad_output = grad_output.contiguous()
    grad_input = torch.empty_like(input, memory_format=torch.preserve_format)
    batch, channels, input_height, input_width = input.shape
    output_height = (
        input_height + 2 * padding_height - kernel_height
    ) // stride_height + 1
    output_width = (
        input_width + 2 * padding_width - kernel_width
    ) // stride_width + 1
    kernel_elements = kernel_height * kernel_width
    output_windows = output_height * output_width
    input_elements = input_height * input_width
    (
        grad_input_batch_stride,
        grad_input_channel_stride,
        grad_input_height_stride,
        grad_input_width_stride,
    ) = grad_input.stride()
    grid = (batch * channels,)
    wrap_triton(_ocab_unfold_backward_kernel)[grid](
        grad_output,
        grad_input,
        channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        output_height,
        output_width,
        kernel_elements,
        output_windows,
        input_elements,
        grad_input_batch_stride,
        grad_input_channel_stride,
        grad_input_height_stride,
        grad_input_width_stride,
        BLOCK_INPUT=triton.next_power_of_2(input_elements),
        num_warps=8,
    )
    return grad_input


def _setup_context(ctx, inputs, output) -> None:
    del output
    (
        input,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
    ) = inputs
    ctx.save_for_backward(input)
    ctx.geometry = (
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
    )


def _backward(ctx, grad_output: torch.Tensor):
    (input,) = ctx.saved_tensors
    grad_input = ocab_unfold2d_backward(grad_output, input, *ctx.geometry)
    return grad_input, None, None, None, None, None, None


torch.library.register_autograd(
    ocab_unfold2d,
    _backward,
    setup_context=_setup_context,
)


__all__ = ["ocab_unfold2d"]
