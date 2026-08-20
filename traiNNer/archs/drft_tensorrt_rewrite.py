#!/usr/bin/env python3
"""Reusable DRFT TensorRT graph rewrites and experimental candidate CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import onnx
from onnx import TensorProto, helper

ELEMENT_BYTES = {
    TensorProto.FLOAT: 4,
    TensorProto.FLOAT16: 2,
    TensorProto.BFLOAT16: 2,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def decompose_dense_fusion(model: onnx.ModelProto) -> list[dict[str, object]]:
    """Replace ``Concat(x_i) @ W`` with ``Sum(x_i @ W_i)``.

    Splitting W by input-channel rows is exact over real arithmetic and removes
    the full-resolution ``depth * channels`` concatenation.  The unchanged
    Linear bias Add remains after the replacement Sum.
    """
    graph = model.graph
    nodes = list(graph.node)
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    producer_index = {
        output: index
        for index, node in enumerate(nodes)
        for output in node.output
    }
    consumers: dict[str, list[int]] = {}
    for node_index, node in enumerate(nodes):
        for name in node.input:
            consumers.setdefault(name, []).append(node_index)

    insertions: dict[int, list[onnx.NodeProto]] = {}
    removed_nodes: set[int] = set()
    removed_initializers: set[str] = set()
    new_initializers: list[onnx.TensorProto] = []
    reports: list[dict[str, object]] = []

    for matmul_index, matmul in enumerate(nodes):
        if (
            matmul.op_type != "MatMul"
            or not matmul.name.endswith("/dense_fusion/MatMul")
            or len(matmul.input) != 2
            or len(matmul.output) != 1
        ):
            continue
        concat_index = producer_index.get(matmul.input[0])
        if concat_index is None:
            raise RuntimeError(f"Dense fusion input has no producer: {matmul.name}")
        concat = nodes[concat_index]
        if concat.op_type != "Concat" or len(concat.input) < 2:
            raise RuntimeError(
                f"Dense fusion input is not a multi-input Concat: {matmul.name}"
            )
        if consumers.get(concat.output[0]) != [matmul_index]:
            raise RuntimeError(
                f"Dense fusion Concat has unexpected consumers: {concat.name}"
            )
        axis = next(
            (attribute.i for attribute in concat.attribute if attribute.name == "axis"),
            None,
        )
        if axis != -1:
            raise RuntimeError(f"Dense fusion Concat axis must be -1, got {axis}")

        weight = initializers.get(matmul.input[1])
        if weight is None or len(weight.dims) != 2 or not weight.raw_data:
            raise RuntimeError("Dense fusion weight is not a raw rank-2 initializer")
        depth = len(concat.input)
        input_channels, output_channels = map(int, weight.dims)
        if input_channels % depth != 0:
            raise RuntimeError(
                f"Dense fusion input width {input_channels} is not divisible by {depth}"
            )
        channels = input_channels // depth
        element_bytes = ELEMENT_BYTES.get(weight.data_type)
        if element_bytes is None:
            raise RuntimeError(
                f"Unsupported dense fusion dtype: {weight.data_type}"
            )
        block_bytes = channels * output_channels * element_bytes
        if len(weight.raw_data) != block_bytes * depth:
            raise RuntimeError(
                f"Dense fusion raw size mismatch: {len(weight.raw_data)}"
            )

        partial_outputs: list[str] = []
        replacement_nodes: list[onnx.NodeProto] = []
        for block_index, block_input in enumerate(concat.input):
            weight_name = f"{weight.name}.drft_block_{block_index}"
            partial_output = f"{matmul.output[0]}.drft_partial_{block_index}"
            sliced_weight = onnx.TensorProto()
            sliced_weight.CopyFrom(weight)
            sliced_weight.name = weight_name
            del sliced_weight.dims[:]
            sliced_weight.dims.extend([channels, output_channels])
            start = block_index * block_bytes
            sliced_weight.raw_data = weight.raw_data[start : start + block_bytes]
            new_initializers.append(sliced_weight)
            partial_outputs.append(partial_output)
            replacement_nodes.append(
                helper.make_node(
                    "MatMul",
                    [block_input, weight_name],
                    [partial_output],
                    name=f"{matmul.name}.drft_block_{block_index}",
                )
            )
        replacement_nodes.append(
            helper.make_node(
                "Sum",
                partial_outputs,
                list(matmul.output),
                name=f"{matmul.name}.drft_sum",
            )
        )
        insertions.setdefault(concat_index, []).extend(replacement_nodes)
        removed_nodes.update((concat_index, matmul_index))
        removed_initializers.add(weight.name)
        reports.append(
            {
                "matmul": matmul.name,
                "concat": concat.name,
                "depth": depth,
                "channels": channels,
                "output_channels": output_channels,
                "weight": weight.name,
                "weight_bytes": len(weight.raw_data),
                "replacement_matmuls": depth,
            }
        )

    if not reports:
        raise RuntimeError("No DRFT dense-fusion pattern was found")

    del graph.node[:]
    for node_index, node in enumerate(nodes):
        graph.node.extend(insertions.get(node_index, []))
        if node_index not in removed_nodes:
            graph.node.append(node)
    retained = [
        tensor
        for tensor in graph.initializer
        if tensor.name not in removed_initializers
    ]
    del graph.initializer[:]
    graph.initializer.extend(retained)
    graph.initializer.extend(new_initializers)

    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata["drft.tensorrt_lossless_dense_fusion"] = "block_matmul_sum_v1"
    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    return reports


def _bf16_scalar(tensor: onnx.TensorProto) -> float:
    if (
        tensor.data_type != TensorProto.BFLOAT16
        or list(tensor.dims) not in ([], [1])
        or len(tensor.raw_data) != 2
    ):
        raise RuntimeError("Expected one raw BF16 scalar")
    bits = int.from_bytes(tensor.raw_data, "little") << 16
    return struct.unpack("<f", bits.to_bytes(4, "little"))[0]


def _float_scalar(tensor: onnx.TensorProto) -> float:
    if tensor.data_type != TensorProto.FLOAT or list(tensor.dims) not in ([], [1]):
        raise RuntimeError("Expected one FP32 scalar")
    if len(tensor.raw_data) == 4:
        return struct.unpack("<f", tensor.raw_data)[0]
    if len(tensor.float_data) == 1:
        return float(tensor.float_data[0])
    raise RuntimeError("Expected one raw or float_data FP32 scalar")


def fuse_image_layer_norm(
    model: onnx.ModelProto,
    *,
    plugin_op: str | None = None,
) -> list[dict[str, object]]:
    """Replace the paper i-LN primitive chain with ONNX LayerNormalization.

    Axis 1 normalizes the complete token/channel plane per image.  The 1-D
    channel affine parameters broadcast across tokens.  ONNX supplies inverse
    standard deviation, so a scalar Reciprocal and BF16 Cast reconstruct the
    unchanged residual-scale output.
    """
    graph = model.graph
    nodes = list(graph.node)
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    producers = {
        output: node
        for node in nodes
        for output in node.output
    }
    by_name = {node.name: (index, node) for index, node in enumerate(nodes)}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in nodes:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    removed_nodes: set[int] = set()
    replacements: dict[int, list[onnx.NodeProto]] = {}
    reports: list[dict[str, object]] = []
    suffixes = (
        "ReduceMean",
        "Sub",
        "Mul",
        "ReduceMean_1",
        "Constant",
        "Add",
        "Sqrt",
        "Div",
        "Mul_1",
        "Add_1",
    )

    for reduce_index, reduce_mean in enumerate(nodes):
        if (
            reduce_mean.op_type != "ReduceMean"
            or not reduce_mean.name.endswith(("/norm1/ReduceMean", "/norm2/ReduceMean"))
        ):
            continue
        prefix = reduce_mean.name.removesuffix("/ReduceMean")
        chain: dict[str, tuple[int, onnx.NodeProto]] = {}
        for suffix in suffixes:
            item = by_name.get(f"{prefix}/{suffix}")
            if item is None:
                raise RuntimeError(f"Incomplete i-LN chain at {prefix}: missing {suffix}")
            chain[suffix] = item
        expected_types = {
            "ReduceMean": "ReduceMean",
            "Sub": "Sub",
            "Mul": "Mul",
            "ReduceMean_1": "ReduceMean",
            "Constant": "Constant",
            "Add": "Add",
            "Sqrt": "Sqrt",
            "Div": "Div",
            "Mul_1": "Mul",
            "Add_1": "Add",
        }
        for suffix, expected_type in expected_types.items():
            if chain[suffix][1].op_type != expected_type:
                raise RuntimeError(
                    f"Unexpected {prefix}/{suffix} type: {chain[suffix][1].op_type}"
                )

        axes = next(
            (
                list(attribute.ints)
                for attribute in reduce_mean.attribute
                if attribute.name == "axes"
            ),
            None,
        )
        if axes != [1, 2]:
            raise RuntimeError(f"i-LN axes must be [1, 2], got {axes}")
        constant = chain["Constant"][1]
        tensor_attributes = [
            attribute.t
            for attribute in constant.attribute
            if attribute.type == onnx.AttributeProto.TENSOR
        ]
        if len(tensor_attributes) != 1:
            raise RuntimeError(f"i-LN epsilon must be one tensor constant: {prefix}")
        epsilon = _bf16_scalar(tensor_attributes[0])
        if abs(epsilon - 1.0e-4) > 1.0e-6:
            raise RuntimeError(f"Unexpected i-LN epsilon {epsilon} at {prefix}")

        sub = chain["Sub"][1]
        square = chain["Mul"][1]
        variance = chain["ReduceMean_1"][1]
        add_epsilon = chain["Add"][1]
        sqrt = chain["Sqrt"][1]
        divide = chain["Div"][1]
        affine_mul = chain["Mul_1"][1]
        affine_add = chain["Add_1"][1]
        input_name = reduce_mean.input[0]
        if (
            list(sub.input) != [input_name, reduce_mean.output[0]]
            or list(square.input) != [sub.output[0], sub.output[0]]
            or list(variance.input) != [square.output[0]]
            or variance.output[0] not in add_epsilon.input
            or list(sqrt.input) != [add_epsilon.output[0]]
            or list(divide.input) != [sub.output[0], sqrt.output[0]]
            or divide.output[0] not in affine_mul.input
            or affine_mul.output[0] not in affine_add.input
        ):
            raise RuntimeError(f"Unexpected i-LN data flow at {prefix}")
        weight_name = next(name for name in affine_mul.input if name != divide.output[0])
        bias_name = next(name for name in affine_add.input if name != affine_mul.output[0])
        def resolve_initializer(name: str) -> onnx.TensorProto | None:
            seen: set[str] = set()
            while name not in seen:
                seen.add(name)
                initializer = initializers.get(name)
                if initializer is not None:
                    return initializer
                producer = producers.get(name)
                if (
                    producer is None
                    or producer.op_type != "Identity"
                    or len(producer.input) != 1
                ):
                    return None
                name = producer.input[0]
            return None

        weight = resolve_initializer(weight_name)
        bias = resolve_initializer(bias_name)
        if (
            weight is None
            or bias is None
            or len(weight.dims) != 1
            or list(bias.dims) != list(weight.dims)
        ):
            raise RuntimeError(f"i-LN affine parameters are not matching vectors at {prefix}")
        channels = int(weight.dims[0])

        internal_outputs = {
            reduce_mean.output[0],
            sub.output[0],
            square.output[0],
            variance.output[0],
            constant.output[0],
            add_epsilon.output[0],
            divide.output[0],
            affine_mul.output[0],
        }
        chain_nodes = {item[1].name for item in chain.values()}
        for output_name in internal_outputs:
            external_users = [
                user.name
                for user in consumers.get(output_name, [])
                if user.name not in chain_nodes
            ]
            if external_users:
                raise RuntimeError(
                    f"Internal i-LN tensor {output_name} has external users {external_users}"
                )

        if plugin_op is not None:
            plugin = helper.make_node(
                plugin_op,
                [input_name, weight_name, bias_name],
                [affine_add.output[0], sqrt.output[0]],
                name=f"{prefix}/{plugin_op.removesuffix('_TRT')}",
                plugin_version="1",
                plugin_namespace="",
                channels=channels,
                epsilon=float(epsilon),
            )
            replacements[reduce_index] = [plugin]
        else:
            mean_output = f"{prefix}/LayerNormalization_mean_output_0"
            invstd_output = f"{prefix}/LayerNormalization_invstd_output_0"
            std_float_output = f"{prefix}/LayerNormalization_std_float_output_0"
            layer_norm = helper.make_node(
                "LayerNormalization",
                [input_name, weight_name, bias_name],
                [affine_add.output[0], mean_output, invstd_output],
                name=f"{prefix}/LayerNormalization",
                axis=1,
                epsilon=float(epsilon),
                stash_type=TensorProto.FLOAT,
            )
            reciprocal = helper.make_node(
                "Reciprocal",
                [invstd_output],
                [std_float_output],
                name=f"{prefix}/StdReciprocal",
            )
            cast = helper.make_node(
                "Cast",
                [std_float_output],
                list(sqrt.output),
                name=f"{prefix}/StdCast",
                to=TensorProto.BFLOAT16,
            )
            replacements[reduce_index] = [layer_norm, reciprocal, cast]
        removed_nodes.update(item[0] for item in chain.values())
        reports.append(
            {
                "prefix": prefix,
                "input": input_name,
                "output": affine_add.output[0],
                "std_output": sqrt.output[0],
                "weight": weight_name,
                "bias": bias_name,
                "channels": channels,
                "epsilon": epsilon,
                "implementation": (
                    plugin_op if plugin_op is not None
                    else "ONNX LayerNormalization"
                ),
            }
        )

    if not reports:
        raise RuntimeError("No DRFT ImageLayerNorm pattern was found")
    del graph.node[:]
    for node_index, node in enumerate(nodes):
        graph.node.extend(replacements.get(node_index, []))
        if node_index not in removed_nodes:
            graph.node.append(node)

    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata["drft.tensorrt_lossless_image_layer_norm"] = (
        (
            "drft_image_layer_norm_fast_plugin_v1"
            if plugin_op == "DrftImageLayerNormFast_TRT"
            else "drft_image_layer_norm_plugin_v1"
        )
        if plugin_op is not None else "onnx_layer_normalization_v1"
    )
    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    return reports


def fuse_next_image_layer_norm_plugin(
    model: onnx.ModelProto,
) -> list[dict[str, object]]:
    """Fuse DRFT-Next's FP32-statistics NCHW ImageLayerNorm chain.

    The legacy DRFT plugin consumes token-major BLC tensors and deliberately
    reproduces BF16-rounded intermediate statistics. DRFT-Next instead casts
    NCHW input to FP32, computes image-wide mean and centered variance in FP32,
    then casts only the normalized tensor and standard deviation back to BF16.
    A distinct operator name keeps those incompatible contracts ABI-safe.
    """
    plugin_op = "DrftImageLayerNormNCHW_TRT"
    graph = model.graph
    nodes = list(graph.node)
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    producers = {output: node for node in nodes for output in node.output}
    by_name = {node.name: (index, node) for index, node in enumerate(nodes)}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in nodes:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    suffixes = (
        "Cast",
        "ReduceMean",
        "Sub",
        "Mul",
        "ReduceMean_1",
        "Constant",
        "Add",
        "Sqrt",
        "Div",
        "Cast_1",
        "Mul_1",
        "Add_1",
        "Cast_2",
    )
    expected_types = {
        "Cast": "Cast",
        "ReduceMean": "ReduceMean",
        "Sub": "Sub",
        "Mul": "Mul",
        "ReduceMean_1": "ReduceMean",
        "Constant": "Constant",
        "Add": "Add",
        "Sqrt": "Sqrt",
        "Div": "Div",
        "Cast_1": "Cast",
        "Mul_1": "Mul",
        "Add_1": "Add",
        "Cast_2": "Cast",
    }
    removed_nodes: set[int] = set()
    replacements: dict[int, list[onnx.NodeProto]] = {}
    reports: list[dict[str, object]] = []

    def resolve_initializer(name: str) -> onnx.TensorProto | None:
        seen: set[str] = set()
        while name not in seen:
            seen.add(name)
            initializer = initializers.get(name)
            if initializer is not None:
                return initializer
            producer = producers.get(name)
            if (
                producer is None
                or producer.op_type != "Identity"
                or len(producer.input) != 1
            ):
                return None
            name = producer.input[0]
        return None

    def cast_target(node: onnx.NodeProto) -> int | None:
        return next(
            (attribute.i for attribute in node.attribute if attribute.name == "to"),
            None,
        )

    for reduce_mean in nodes:
        if reduce_mean.op_type != "ReduceMean" or not reduce_mean.name.endswith(
            "/ReduceMean"
        ):
            continue
        prefix = reduce_mean.name.removesuffix("/ReduceMean")
        if "norm" not in prefix.rsplit("/", 1)[-1].lower():
            continue
        axes = next(
            (
                list(attribute.ints)
                for attribute in reduce_mean.attribute
                if attribute.name == "axes"
            ),
            None,
        )
        if axes != [1, 2, 3]:
            continue

        chain: dict[str, tuple[int, onnx.NodeProto]] = {}
        for suffix in suffixes:
            item = by_name.get(f"{prefix}/{suffix}")
            if item is None:
                raise RuntimeError(
                    f"Incomplete DRFT-Next i-LN chain at {prefix}: missing {suffix}"
                )
            chain[suffix] = item
        for suffix, expected_type in expected_types.items():
            if chain[suffix][1].op_type != expected_type:
                raise RuntimeError(
                    f"Unexpected {prefix}/{suffix} type: "
                    f"{chain[suffix][1].op_type}"
                )

        statistics_cast = chain["Cast"][1]
        sub = chain["Sub"][1]
        square = chain["Mul"][1]
        variance = chain["ReduceMean_1"][1]
        constant = chain["Constant"][1]
        add_epsilon = chain["Add"][1]
        sqrt = chain["Sqrt"][1]
        divide = chain["Div"][1]
        normalized_cast = chain["Cast_1"][1]
        affine_mul = chain["Mul_1"][1]
        affine_add = chain["Add_1"][1]
        std_cast = chain["Cast_2"][1]
        if (
            cast_target(statistics_cast) != TensorProto.FLOAT
            or cast_target(normalized_cast) != TensorProto.BFLOAT16
            or cast_target(std_cast) != TensorProto.BFLOAT16
        ):
            raise RuntimeError(f"Unexpected DRFT-Next i-LN cast types at {prefix}")

        tensor_attributes = [
            attribute.t
            for attribute in constant.attribute
            if attribute.type == onnx.AttributeProto.TENSOR
        ]
        if len(tensor_attributes) != 1:
            raise RuntimeError(f"i-LN epsilon must be one tensor constant: {prefix}")
        epsilon = _float_scalar(tensor_attributes[0])
        if abs(epsilon - 1.0e-4) > 1.0e-8:
            raise RuntimeError(f"Unexpected i-LN epsilon {epsilon} at {prefix}")

        input_name = statistics_cast.input[0]
        if (
            list(reduce_mean.input) != [statistics_cast.output[0]]
            or list(sub.input)
            != [statistics_cast.output[0], reduce_mean.output[0]]
            or list(square.input) != [sub.output[0], sub.output[0]]
            or list(variance.input) != [square.output[0]]
            or variance.output[0] not in add_epsilon.input
            or constant.output[0] not in add_epsilon.input
            or list(sqrt.input) != [add_epsilon.output[0]]
            or list(divide.input) != [sub.output[0], sqrt.output[0]]
            or list(normalized_cast.input) != [divide.output[0]]
            or normalized_cast.output[0] not in affine_mul.input
            or affine_mul.output[0] not in affine_add.input
            or list(std_cast.input) != [sqrt.output[0]]
        ):
            raise RuntimeError(f"Unexpected DRFT-Next i-LN data flow at {prefix}")

        weight_name = next(
            name for name in affine_mul.input if name != normalized_cast.output[0]
        )
        bias_name = next(
            name for name in affine_add.input if name != affine_mul.output[0]
        )
        weight = resolve_initializer(weight_name)
        bias = resolve_initializer(bias_name)
        if (
            weight is None
            or bias is None
            or list(weight.dims[:1]) != [1]
            or len(weight.dims) != 4
            or int(weight.dims[2]) != 1
            or int(weight.dims[3]) != 1
            or list(bias.dims) != list(weight.dims)
            or weight.data_type != TensorProto.BFLOAT16
            or bias.data_type != TensorProto.BFLOAT16
        ):
            raise RuntimeError(
                f"i-LN affine parameters are not matching BF16 NCHW tensors at {prefix}"
            )
        channels = int(weight.dims[1])

        internal_outputs = {
            statistics_cast.output[0],
            reduce_mean.output[0],
            sub.output[0],
            square.output[0],
            variance.output[0],
            constant.output[0],
            add_epsilon.output[0],
            sqrt.output[0],
            divide.output[0],
            normalized_cast.output[0],
            affine_mul.output[0],
        }
        chain_names = {item[1].name for item in chain.values()}
        for output_name in internal_outputs:
            external_users = [
                user.name
                for user in consumers.get(output_name, [])
                if user.name not in chain_names
            ]
            if external_users:
                raise RuntimeError(
                    f"Internal i-LN tensor {output_name} has external users "
                    f"{external_users}"
                )

        plugin = helper.make_node(
            plugin_op,
            [input_name, weight_name, bias_name],
            [affine_add.output[0], std_cast.output[0]],
            name=f"{prefix}/DrftImageLayerNormNCHW",
            plugin_version="1",
            plugin_namespace="",
            channels=channels,
            epsilon=float(epsilon),
        )
        replacements[chain["Cast"][0]] = [plugin]
        removed_nodes.update(item[0] for item in chain.values())
        reports.append(
            {
                "prefix": prefix,
                "input": input_name,
                "output": affine_add.output[0],
                "std_output": std_cast.output[0],
                "weight": weight_name,
                "bias": bias_name,
                "channels": channels,
                "epsilon": epsilon,
                "statistics_dtype": "FP32",
                "layout": "NCHW",
                "implementation": plugin_op,
            }
        )

    if not reports:
        raise RuntimeError("No DRFT-Next NCHW ImageLayerNorm pattern was found")
    del graph.node[:]
    for node_index, node in enumerate(nodes):
        graph.node.extend(replacements.get(node_index, []))
        if node_index not in removed_nodes:
            graph.node.append(node)
    plugin_count = sum(node.op_type == plugin_op for node in graph.node)
    if plugin_count != len(reports):
        raise RuntimeError(
            f"Expected {len(reports)} live NCHW i-LN plugins, found {plugin_count}"
        )

    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata["drft.tensorrt_lossless_image_layer_norm"] = (
        "drft_next_fp32_statistics_nchw_plugin_v1"
    )
    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    return reports


def _eliminate_dead_nodes(model: onnx.ModelProto) -> dict[str, int]:
    """Keep only nodes reachable from graph outputs.

    ONNX graphs have no stateful DRFT operators.  Backward reachability is
    therefore a conservative way to delete the replaced OCAB packing chain
    while retaining any dynamic-shape expressions shared by later window
    reversal nodes.
    """
    graph = model.graph
    nodes = list(graph.node)
    producers = {
        output: index
        for index, node in enumerate(nodes)
        for output in node.output
    }
    live_nodes: set[int] = set()
    pending = [output.name for output in graph.output]
    visited_tensors: set[str] = set()
    while pending:
        tensor_name = pending.pop()
        if tensor_name in visited_tensors:
            continue
        visited_tensors.add(tensor_name)
        producer_index = producers.get(tensor_name)
        if producer_index is None:
            continue
        if producer_index in live_nodes:
            continue
        live_nodes.add(producer_index)
        pending.extend(nodes[producer_index].input)

    del graph.node[:]
    graph.node.extend(
        node for index, node in enumerate(nodes) if index in live_nodes
    )

    used_initializers = {
        name
        for node in graph.node
        for name in node.input
    }
    used_initializers.update(output.name for output in graph.output)
    initializers = list(graph.initializer)
    del graph.initializer[:]
    graph.initializer.extend(
        tensor for tensor in initializers if tensor.name in used_initializers
    )
    return {
        "nodes_removed": len(nodes) - len(live_nodes),
        "initializers_removed": len(initializers) - len(graph.initializer),
    }


def fuse_next_ocab_qkv_reflect_pack(
    model: onnx.ModelProto,
    *,
    channels: int,
    heads: int,
    query_window: int,
    key_window: int,
) -> list[dict[str, object]]:
    """Fuse DRFT-Next's reflection pad, unfold, and Q/K/V window layouts.

    The plugin is arithmetic-free: it reads the projected ``[B,H,W,3,C]``
    tensor with reflection coordinates and writes token-major Q/K/V. Three
    metadata-only transposes restore the layouts expected by QK-RoPE and MHA.
    """

    if (
        channels <= 0
        or heads <= 0
        or channels % heads != 0
        or channels // heads != 32
        or channels % 8 != 0
        or query_window <= 0
        or key_window < query_window
        or (key_window - query_window) % 2 != 0
    ):
        raise ValueError("Invalid DRFT-Next reflected OCAB packing dimensions")

    graph = model.graph
    nodes = list(graph.node)
    by_name = {node.name: (index, node) for index, node in enumerate(nodes)}
    produced = {output for node in nodes for output in node.output}
    insertions: dict[int, list[onnx.NodeProto]] = {}
    reports: list[dict[str, object]] = []

    for group_index in range(64):
        prefix = f"/layers.{group_index}/residual_group/ocab/"
        boundary_item = by_name.get(f"{prefix}Reshape_1")
        if boundary_item is None:
            continue
        boundary_index, boundary = boundary_item
        if boundary.op_type != "Reshape" or len(boundary.output) != 1:
            raise RuntimeError(f"Unexpected DRFT-Next OCAB boundary at {prefix}")

        old_query = f"{prefix}Transpose_5_output_0"
        old_key = f"{prefix}Transpose_6_output_0"
        old_value = f"{prefix}Transpose_7_output_0"
        if not all(name in produced for name in (old_query, old_key, old_value)):
            raise RuntimeError(f"Incomplete DRFT-Next OCAB packing at {prefix}")

        raw_query = f"{prefix}DrftNextOCABReflectPack_query_token_major"
        raw_key = f"{prefix}DrftNextOCABReflectPack_key_token_major"
        raw_value = f"{prefix}DrftNextOCABReflectPack_value_token_major"
        query_output = f"{prefix}DrftNextOCABReflectPack_query"
        key_output = f"{prefix}DrftNextOCABReflectPack_key"
        value_output = f"{prefix}DrftNextOCABReflectPack_value"
        plugin = helper.make_node(
            "DrftNextOCABQKVPackReflectTokenMajor_TRT",
            [boundary.output[0]],
            [raw_query, raw_key, raw_value],
            name=f"{prefix}DrftNextOCABQKVPackReflect",
            plugin_version="1",
            plugin_namespace="",
            channels=channels,
            heads=heads,
            query_window_size=query_window,
            key_window_size=key_window,
        )
        layouts = [
            helper.make_node(
                "Transpose",
                [raw_query],
                [query_output],
                name=f"{prefix}DrftNextOCABReflectPack_query_layout",
                perm=[1, 2, 0, 3],
            ),
            helper.make_node(
                "Transpose",
                [raw_key],
                [key_output],
                name=f"{prefix}DrftNextOCABReflectPack_key_layout",
                perm=[1, 2, 0, 3],
            ),
            helper.make_node(
                "Transpose",
                [raw_value],
                [value_output],
                name=f"{prefix}DrftNextOCABReflectPack_value_layout",
                perm=[1, 2, 0, 3],
            ),
        ]
        insertions.setdefault(boundary_index + 1, []).extend([plugin, *layouts])

        replacements = {
            old_query: query_output,
            old_key: key_output,
            old_value: value_output,
        }
        for node in nodes:
            for input_index, name in enumerate(node.input):
                replacement = replacements.get(name)
                if replacement is not None:
                    node.input[input_index] = replacement
        reports.append(
            {
                "group": group_index,
                "boundary": boundary.output[0],
                "channels": channels,
                "heads": heads,
                "head_dim": channels // heads,
                "query_window": query_window,
                "key_window": key_window,
                "padding": "reflect",
                "arithmetic": "none",
                "token_major": True,
            }
        )

    if not reports:
        raise RuntimeError("No DRFT-Next reflected OCAB packing pattern was found")

    del graph.node[:]
    for node_index, node in enumerate(nodes):
        graph.node.extend(insertions.get(node_index, []))
        graph.node.append(node)
    graph.node.extend(insertions.get(len(nodes), []))
    dead_code = _eliminate_dead_nodes(model)
    for report in reports:
        report["dead_code_elimination"] = dead_code

    plugin_count = sum(
        node.op_type == "DrftNextOCABQKVPackReflectTokenMajor_TRT"
        for node in graph.node
    )
    if plugin_count != len(reports):
        raise RuntimeError(
            f"Expected {len(reports)} reflected OCAB plugins, found {plugin_count}"
        )

    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata["drft.next.tensorrt_ocab_reflect_pack"] = (
        "direct_bf16_reflect_window_pack_token_major_v1"
    )
    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    return reports


def fuse_next_qk_rmsnorm_rope(
    model: onnx.ModelProto,
    *,
    epsilon: float = 1.0e-6,
) -> list[dict[str, object]]:
    """Fuse FP32 QK-RMSNorm, BF16 mixed RoPE, and query temperature.

    TensorRT retains native QK/softmax/PV attention. The plugin consumes the
    raw rank-4 Q/K tensors plus frozen BF16 cosines, sines, and head scales.
    """

    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    graph = model.graph
    nodes = list(graph.node)
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    producers = {
        output: node
        for node in nodes
        for output in node.output
    }
    node_indices = {id(node): index for index, node in enumerate(nodes)}
    by_name = {node.name: node for node in nodes}

    def producer(tensor_name: str, expected: str | None = None) -> onnx.NodeProto:
        node = producers.get(tensor_name)
        if node is None or (expected is not None and node.op_type != expected):
            actual = None if node is None else node.op_type
            raise RuntimeError(
                f"Expected {expected} producer for {tensor_name}, got {actual}"
            )
        return node

    def is_constant(tensor_name: str) -> bool:
        node = producers.get(tensor_name)
        return tensor_name in initializers or (
            node is not None and node.op_type == "Constant"
        )

    def constant_input(node: onnx.NodeProto) -> str:
        constants = [name for name in node.input if is_constant(name)]
        if len(constants) != 1:
            raise RuntimeError(
                f"Expected one constant input for {node.name}, got {constants}"
            )
        return constants[0]

    def constant_fingerprint(tensor_name: str) -> bytes:
        tensor = initializers.get(tensor_name)
        if tensor is not None:
            return tensor.SerializeToString()
        node = producer(tensor_name, "Constant")
        values = [attribute.t for attribute in node.attribute if attribute.name == "value"]
        if len(values) != 1:
            raise RuntimeError(f"Constant {node.name} has no tensor value")
        return values[0].SerializeToString()

    def raw_input_from_reduce(reduce_node: onnx.NodeProto) -> str:
        square = producer(reduce_node.input[0], "Mul")
        if len(square.input) != 2 or square.input[0] != square.input[1]:
            raise RuntimeError(f"Unexpected QK square at {square.name}")
        cast = producer(square.input[0], "Cast")
        if len(cast.input) != 1:
            raise RuntimeError(f"Unexpected QK FP32 cast at {cast.name}")
        return cast.input[0]

    def rotary_constants(rotated_tensor: str) -> tuple[str, str]:
        reshape = producer(rotated_tensor, "Reshape")
        concat = producer(reshape.input[0], "Concat")
        if len(concat.input) != 2:
            raise RuntimeError(f"Unexpected rotary concat at {concat.name}")
        real_unsqueeze = producer(concat.input[0], "Unsqueeze")
        imaginary_unsqueeze = producer(concat.input[1], "Unsqueeze")
        real_rotation = producer(real_unsqueeze.input[0], "Sub")
        imaginary_rotation = producer(imaginary_unsqueeze.input[0], "Add")
        real_cos_mul = producer(real_rotation.input[0], "Mul")
        real_sin_mul = producer(real_rotation.input[1], "Mul")
        imaginary_sin_mul = producer(imaginary_rotation.input[0], "Mul")
        imaginary_cos_mul = producer(imaginary_rotation.input[1], "Mul")
        cosine = constant_input(real_cos_mul)
        sine = constant_input(real_sin_mul)
        if constant_fingerprint(cosine) != constant_fingerprint(
            constant_input(imaginary_cos_mul)
        ):
            raise RuntimeError(f"Rotary cosine mismatch at {concat.name}")
        if constant_fingerprint(sine) != constant_fingerprint(
            constant_input(imaginary_sin_mul)
        ):
            raise RuntimeError(f"Rotary sine mismatch at {concat.name}")
        return cosine, sine

    q_reductions = [
        node
        for node in nodes
        if node.op_type == "ReduceMean"
        and node.name.endswith("/ReduceMean")
        and (
            "/attn/" in node.name
            or "/attn_" in node.name
            or "/ocab/" in node.name
        )
    ]
    insertions: dict[int, list[onnx.NodeProto]] = {}
    reports: list[dict[str, object]] = []
    for q_reduce in q_reductions:
        prefix = q_reduce.name[: -len("ReduceMean")]
        k_reduce = by_name.get(f"{prefix}ReduceMean_1")
        score = by_name.get(f"{prefix}MatMul")
        if k_reduce is None or score is None or score.op_type != "MatMul":
            raise RuntimeError(f"Incomplete QK attention pattern at {prefix}")
        raw_query = raw_input_from_reduce(q_reduce)
        raw_key = raw_input_from_reduce(k_reduce)

        query_final = score.input[0]
        query_scale_mul = producer(query_final, "Mul")
        query_scale = constant_input(query_scale_mul)
        query_rotated_inputs = [
            name for name in query_scale_mul.input if name != query_scale
        ]
        if len(query_rotated_inputs) != 1:
            raise RuntimeError(f"Unexpected query temperature at {prefix}")
        query_rotated = query_rotated_inputs[0]

        key_transpose = producer(score.input[1], "Transpose")
        if len(key_transpose.input) != 1:
            raise RuntimeError(f"Unexpected key transpose at {prefix}")
        key_rotated = key_transpose.input[0]
        query_cosine, query_sine = rotary_constants(query_rotated)
        key_cosine, key_sine = rotary_constants(key_rotated)

        query_output = f"{prefix}DrftNextQKRoPE_query"
        key_output = f"{prefix}DrftNextQKRoPE_key"
        plugin = helper.make_node(
            "DrftNextQKRoPE_TRT",
            [
                raw_query,
                raw_key,
                query_cosine,
                query_sine,
                key_cosine,
                key_sine,
                query_scale,
            ],
            [query_output, key_output],
            name=f"{prefix}DrftNextQKRoPE",
            plugin_version="1",
            plugin_namespace="",
            epsilon=float(epsilon),
        )
        score.input[0] = query_output
        key_transpose.input[0] = key_output
        insertions.setdefault(node_indices[id(score)], []).append(plugin)
        reports.append(
            {
                "scope": prefix.rstrip("/"),
                "raw_query": raw_query,
                "raw_key": raw_key,
                "query_cosine": query_cosine,
                "query_sine": query_sine,
                "key_cosine": key_cosine,
                "key_sine": key_sine,
                "query_scale": query_scale,
                "epsilon": epsilon,
                "head_dim": 32,
            }
        )

    if not reports:
        raise RuntimeError("No DRFT-Next QK-RMSNorm/RoPE patterns were found")

    del graph.node[:]
    for node_index, node in enumerate(nodes):
        graph.node.extend(insertions.get(node_index, []))
        graph.node.append(node)
    graph.node.extend(insertions.get(len(nodes), []))
    dead_code = _eliminate_dead_nodes(model)
    for report in reports:
        report["dead_code_elimination"] = dead_code

    plugin_count = sum(
        node.op_type == "DrftNextQKRoPE_TRT" for node in graph.node
    )
    if plugin_count != len(reports):
        raise RuntimeError(
            f"Expected {len(reports)} QK-RoPE plugins, found {plugin_count}"
        )

    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata["drft.next.tensorrt_qk_rope"] = (
        "fp32_rms_bf16_step_rounded_rope_warp32_v1"
    )
    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    return reports


def fuse_ocab_qkv_pack(
    model: onnx.ModelProto,
    *,
    channels: int,
    heads: int,
    query_window: int,
    key_window: int,
    token_major: bool = False,
) -> list[dict[str, object]]:
    """Replace OCAB Q/K/V window packing with one data-movement plugin.

    The plugin copies the same BF16 Q/K/V elements into the same attention
    layouts, including zero padding for the overlapping K/V windows.  It does
    no floating-point arithmetic.  TensorRT's native fused MHA remains the
    consumer of the three outputs.
    """
    if (
        channels <= 0
        or heads <= 0
        or channels % heads != 0
        or query_window <= 0
        or key_window < query_window
        or (key_window - query_window) % 2 != 0
    ):
        raise ValueError("Invalid OCAB Q/K/V packing dimensions")

    graph = model.graph
    nodes = list(graph.node)
    by_name = {node.name: (index, node) for index, node in enumerate(nodes)}
    insertions: dict[int, list[onnx.NodeProto]] = {}
    reports: list[dict[str, object]] = []

    for group_index in range(64):
        prefix = f"/layers.{group_index}/residual_group/ocab/"
        boundary_item = by_name.get(f"{prefix}Reshape_1")
        if boundary_item is None:
            continue
        boundary_index, boundary = boundary_item
        if boundary.op_type != "Reshape" or len(boundary.output) != 1:
            raise RuntimeError(f"Unexpected OCAB QKV boundary at {prefix}")

        query_name = f"{prefix}Transpose_4_output_0"
        key_name = f"{prefix}Transpose_6_output_0"
        value_name = f"{prefix}Transpose_5_output_0"
        score_item = by_name.get(f"{prefix}MatMul")
        value_matmul_item = by_name.get(f"{prefix}MatMul_1")
        if score_item is None or value_matmul_item is None:
            raise RuntimeError(f"Incomplete OCAB attention consumers at {prefix}")
        _, score_matmul = score_item
        _, value_matmul = value_matmul_item
        if (
            score_matmul.op_type != "MatMul"
            or list(score_matmul.input) != [query_name, key_name]
            or value_matmul.op_type != "MatMul"
            or len(value_matmul.input) != 2
            or value_matmul.input[1] != value_name
        ):
            raise RuntimeError(f"Unexpected OCAB attention data flow at {prefix}")

        query_output = f"{prefix}DrftOCABQKVPack_query_output_0"
        key_output = f"{prefix}DrftOCABQKVPack_key_output_0"
        value_output = f"{prefix}DrftOCABQKVPack_value_output_0"
        plugin_op = (
            "DrftOCABQKVPackTokenMajor_TRT"
            if token_major
            else "DrftOCABQKVPack_TRT"
        )
        plugin_outputs = [query_output, key_output, value_output]
        layout_nodes: list[onnx.NodeProto] = []
        if token_major:
            raw_query = f"{query_output}_token_major"
            raw_key = f"{key_output}_token_major"
            raw_value = f"{value_output}_token_major"
            plugin_outputs = [raw_query, raw_key, raw_value]
            layout_nodes = [
                helper.make_node(
                    "Transpose",
                    [raw_query],
                    [query_output],
                    name=f"{prefix}DrftOCABQKVPack_query_layout",
                    perm=[1, 2, 0, 3],
                ),
                helper.make_node(
                    "Transpose",
                    [raw_key],
                    [key_output],
                    name=f"{prefix}DrftOCABQKVPack_key_layout",
                    perm=[1, 2, 3, 0],
                ),
                helper.make_node(
                    "Transpose",
                    [raw_value],
                    [value_output],
                    name=f"{prefix}DrftOCABQKVPack_value_layout",
                    perm=[1, 2, 0, 3],
                ),
            ]
        plugin = helper.make_node(
            plugin_op,
            [boundary.output[0]],
            plugin_outputs,
            name=f"{prefix}DrftOCABQKVPack",
            plugin_version="1",
            plugin_namespace="",
            channels=channels,
            heads=heads,
            query_window_size=query_window,
            key_window_size=key_window,
        )
        insertions.setdefault(boundary_index + 1, []).extend(
            [plugin, *layout_nodes]
        )
        score_matmul.input[0] = query_output
        score_matmul.input[1] = key_output
        value_matmul.input[1] = value_output
        reports.append(
            {
                "group": group_index,
                "boundary": boundary.output[0],
                "query_replaced": query_name,
                "key_replaced": key_name,
                "value_replaced": value_name,
                "channels": channels,
                "heads": heads,
                "head_dim": channels // heads,
                "query_window": query_window,
                "key_window": key_window,
                "arithmetic": "none",
                "token_major": token_major,
            }
        )

    if not reports:
        raise RuntimeError("No DRFT OCAB Q/K/V packing patterns were found")

    del graph.node[:]
    for node_index, node in enumerate(nodes):
        graph.node.extend(insertions.get(node_index, []))
        graph.node.append(node)
    graph.node.extend(insertions.get(len(nodes), []))
    dead_code = _eliminate_dead_nodes(model)
    for report in reports:
        report["dead_code_elimination"] = dead_code

    expected_plugin_op = (
        "DrftOCABQKVPackTokenMajor_TRT"
        if token_major
        else "DrftOCABQKVPack_TRT"
    )
    plugin_count = sum(
        node.op_type == expected_plugin_op for node in graph.node
    )
    if plugin_count != len(reports):
        raise RuntimeError(
            f"Expected {len(reports)} live Q/K/V plugins, found {plugin_count}"
        )

    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata["drft.tensorrt_lossless_ocab_qkv_pack"] = (
        "direct_bf16_window_pack_token_major_v2"
        if token_major
        else "direct_bf16_window_pack_v1"
    )
    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    return reports


def retarget_image_layer_norm_plugin(
    model: onnx.ModelProto,
    *,
    source_op: str,
    target_op: str,
) -> list[dict[str, object]]:
    """Retarget an already-fused i-LN node to an equivalent implementation."""
    reports: list[dict[str, object]] = []
    for node in model.graph.node:
        if node.op_type != source_op:
            continue
        reports.append(
            {
                "name": node.name,
                "source_op": source_op,
                "target_op": target_op,
                "inputs": list(node.input),
                "outputs": list(node.output),
            }
        )
        node.op_type = target_op
    if not reports:
        raise RuntimeError(f"No {source_op} nodes were found")
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata["drft.tensorrt_lossless_image_layer_norm"] = (
        "drft_image_layer_norm_cooperative_plugin_v1"
    )
    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    return reports


def fuse_ffn_swiglu_depthwise(
    model: onnx.ModelProto,
) -> list[dict[str, object]]:
    """Fuse BF16 SwiGLU, layout, FP16 depthwise 3x3, and BF16 bias.

    The surrounding BF16 GEMMs remain native TensorRT layers.  Dynamic H/W is
    preserved by reshaping the two token-major halves to BHWC views before the
    plugin and flattening its BHWC output back to BLC afterward.
    """
    graph = model.graph
    nodes = list(graph.node)
    by_name = {node.name: (index, node) for index, node in enumerate(nodes)}
    producer = {output: node for node in nodes for output in node.output}
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    insertions: dict[int, list[onnx.NodeProto]] = {}
    reports: list[dict[str, object]] = []

    for add in nodes:
        suffix = "/ffn/fc_gate_value/Add"
        if add.op_type != "Add" or not add.name.endswith(suffix):
            continue
        prefix = add.name.removesuffix("/fc_gate_value/Add")
        required_names = {
            "gate": f"{prefix}/Slice",
            "value": f"{prefix}/Slice_1",
            "shape": f"{prefix}/Concat",
            "depthwise": f"{prefix}/dwconv/Conv",
            "fc_out": f"{prefix}/fc_out/MatMul",
        }
        required = {key: by_name.get(name) for key, name in required_names.items()}
        missing = [key for key, item in required.items() if item is None]
        if missing:
            raise RuntimeError(f"Incomplete FFN pattern at {prefix}: {missing}")
        gate_index, gate = required["gate"]  # type: ignore[misc]
        _, value = required["value"]  # type: ignore[misc]
        shape_index, nchw_shape = required["shape"]  # type: ignore[misc]
        _, depthwise = required["depthwise"]  # type: ignore[misc]
        _, fc_out = required["fc_out"]  # type: ignore[misc]
        if (
            gate.op_type != "Slice"
            or value.op_type != "Slice"
            or nchw_shape.op_type != "Concat"
            or len(nchw_shape.input) != 4
            or depthwise.op_type != "Conv"
            or len(depthwise.input) != 3
            or fc_out.op_type != "MatMul"
            or len(fc_out.input) != 2
        ):
            raise RuntimeError(f"Unexpected FFN data flow at {prefix}")
        def resolve_identity_initializer(name: str) -> onnx.TensorProto | None:
            visited: set[str] = set()
            while name not in initializers:
                if name in visited:
                    return None
                visited.add(name)
                alias = producer.get(name)
                if alias is None or alias.op_type != "Identity" or len(alias.input) != 1:
                    return None
                name = alias.input[0]
            return initializers[name]

        weight = resolve_identity_initializer(depthwise.input[1])
        bias = resolve_identity_initializer(depthwise.input[2])
        conv_attributes = {
            attribute.name: (
                list(attribute.ints)
                if attribute.type == onnx.AttributeProto.INTS
                else int(attribute.i)
            )
            for attribute in depthwise.attribute
        }
        if (
            weight is None
            or bias is None
            or len(weight.dims) != 4
            or list(weight.dims[1:]) != [1, 3, 3]
            or list(bias.dims) != [int(weight.dims[0])]
            or weight.data_type != TensorProto.BFLOAT16
            or bias.data_type != TensorProto.BFLOAT16
            or conv_attributes
            != {
                "dilations": [1, 1],
                "group": int(weight.dims[0]),
                "kernel_shape": [3, 3],
                "pads": [1, 1, 1, 1],
                "strides": [1, 1],
            }
        ):
            raise RuntimeError(f"Unsupported FFN depthwise parameters at {prefix}")
        channels = int(weight.dims[0])

        shape_3d = f"{prefix}/DrftFFNFused_shape_3d_output_0"
        shape_bhwc = f"{prefix}/DrftFFNFused_shape_bhwc_output_0"
        gate_bhwc = f"{prefix}/DrftFFNFused_gate_bhwc_output_0"
        value_bhwc = f"{prefix}/DrftFFNFused_value_bhwc_output_0"
        fused_bhwc = f"{prefix}/DrftFFNFused_bhwc_output_0"
        fused_flat = f"{prefix}/DrftFFNFused_flat_output_0"
        replacement = [
            helper.make_node(
                "Shape",
                [gate.output[0]],
                [shape_3d],
                name=f"{prefix}/DrftFFNFusedShape3D",
            ),
            helper.make_node(
                "Concat",
                [
                    nchw_shape.input[0],
                    nchw_shape.input[2],
                    nchw_shape.input[3],
                    nchw_shape.input[1],
                ],
                [shape_bhwc],
                name=f"{prefix}/DrftFFNFusedShapeBHWC",
                axis=0,
            ),
            helper.make_node(
                "Reshape",
                [gate.output[0], shape_bhwc],
                [gate_bhwc],
                name=f"{prefix}/DrftFFNFusedGateBHWC",
                allowzero=0,
            ),
            helper.make_node(
                "Reshape",
                [value.output[0], shape_bhwc],
                [value_bhwc],
                name=f"{prefix}/DrftFFNFusedValueBHWC",
                allowzero=0,
            ),
            helper.make_node(
                "DrftFFNSwiGLUDepthwise_TRT",
                [gate_bhwc, value_bhwc, depthwise.input[1], depthwise.input[2]],
                [fused_bhwc],
                name=f"{prefix}/DrftFFNSwiGLUDepthwise",
                plugin_version="1",
                plugin_namespace="",
                channels=channels,
            ),
            helper.make_node(
                "Reshape",
                [fused_bhwc, shape_3d],
                [fused_flat],
                name=f"{prefix}/DrftFFNFusedFlatten",
                allowzero=0,
            ),
        ]
        insertions.setdefault(shape_index + 1, []).extend(replacement)
        fc_out.input[0] = fused_flat
        reports.append(
            {
                "prefix": prefix,
                "gate": gate.output[0],
                "value": value.output[0],
                "depthwise_weight": depthwise.input[1],
                "depthwise_bias": depthwise.input[2],
                "channels": channels,
                "kernel": [3, 3],
                "padding": [1, 1],
                "stride": [1, 1],
                "output": fused_flat,
                "insertion_after": nodes[shape_index].name,
                "gate_index": gate_index,
            }
        )

    if not reports:
        raise RuntimeError("No DRFT FFN SwiGLU/depthwise patterns were found")

    del graph.node[:]
    for node_index, node in enumerate(nodes):
        graph.node.extend(insertions.get(node_index, []))
        graph.node.append(node)
    graph.node.extend(insertions.get(len(nodes), []))
    dead_code = _eliminate_dead_nodes(model)
    for report in reports:
        report["dead_code_elimination"] = dead_code
    plugin_count = sum(
        node.op_type == "DrftFFNSwiGLUDepthwise_TRT" for node in graph.node
    )
    if plugin_count != len(reports):
        raise RuntimeError(
            f"Expected {len(reports)} live fused FFN plugins, found {plugin_count}"
        )

    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata["drft.tensorrt_lossless_ffn_swiglu_depthwise"] = (
        "bf16_swiglu_fp16_depthwise_bf16_bias_v1"
    )
    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    return reports


def fuse_residual_image_layer_norm(
    model: onnx.ModelProto,
) -> list[dict[str, object]]:
    """Fuse a BF16 residual Add into the following exact i-LN plugin.

    The replacement keeps the rounded residual tensor as its first output, so
    every non-normalization consumer observes the same value.  The existing
    normalized tensor and scalar standard-deviation outputs are preserved.
    """
    graph = model.graph
    nodes = list(graph.node)
    producer_index = {
        output: index
        for index, node in enumerate(nodes)
        for output in node.output
    }
    replacements: dict[int, onnx.NodeProto] = {}
    removed: set[int] = set()
    reports: list[dict[str, object]] = []

    for norm_index, norm in enumerate(nodes):
        if norm.op_type != "DrftImageLayerNorm_TRT":
            continue
        if len(norm.input) != 3 or len(norm.output) != 2:
            raise RuntimeError(f"Unexpected i-LN arity at {norm.name}")
        add_index = producer_index.get(norm.input[0])
        if add_index is None:
            continue
        add = nodes[add_index]
        if add.op_type != "Add":
            continue
        if len(add.input) != 2 or len(add.output) != 1:
            raise RuntimeError(f"Unexpected residual Add arity at {add.name}")
        attributes = {attribute.name: attribute for attribute in norm.attribute}
        channels_attribute = attributes.get("channels")
        epsilon_attribute = attributes.get("epsilon")
        if (
            channels_attribute is None
            or channels_attribute.type != onnx.AttributeProto.INT
            or channels_attribute.i <= 0
            or epsilon_attribute is None
            or epsilon_attribute.type != onnx.AttributeProto.FLOAT
            or epsilon_attribute.f <= 0.0
        ):
            raise RuntimeError(f"Missing i-LN fields at {norm.name}")
        if add_index in replacements or add_index in removed:
            raise RuntimeError(f"Residual Add already rewritten: {add.name}")

        plugin = helper.make_node(
            "DrftResidualImageLayerNorm_TRT",
            [add.input[0], add.input[1], norm.input[1], norm.input[2]],
            [add.output[0], norm.output[0], norm.output[1]],
            name=f"{norm.name}.ResidualFused",
            plugin_version="1",
            plugin_namespace="",
            channels=int(channels_attribute.i),
            epsilon=float(epsilon_attribute.f),
        )
        replacements[add_index] = plugin
        removed.add(add_index)
        removed.add(norm_index)
        reports.append(
            {
                "residual_add": add.name,
                "image_layer_norm": norm.name,
                "left": add.input[0],
                "right": add.input[1],
                "residual_output": add.output[0],
                "normalized_output": norm.output[0],
                "std_output": norm.output[1],
                "channels": int(channels_attribute.i),
                "epsilon": float(epsilon_attribute.f),
            }
        )

    if not reports:
        raise RuntimeError("No residual Add -> i-LN patterns were found")
    del graph.node[:]
    for node_index, node in enumerate(nodes):
        replacement = replacements.get(node_index)
        if replacement is not None:
            graph.node.append(replacement)
        elif node_index not in removed:
            graph.node.append(node)
    plugin_count = sum(
        node.op_type == "DrftResidualImageLayerNorm_TRT"
        for node in graph.node
    )
    if plugin_count != len(reports):
        raise RuntimeError(
            f"Expected {len(reports)} residual i-LN plugins, found {plugin_count}"
        )
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata["drft.tensorrt_lossless_residual_image_layer_norm"] = (
        "bf16_residual_add_plus_two_pass_image_layer_norm_v1"
    )
    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "dense_fusion",
            "image_layer_norm",
            "image_layer_norm_plugin",
            "image_layer_norm_fast_plugin",
            "image_layer_norm_cooperative_plugin",
            "next_image_layer_norm_plugin",
            "next_ocab_reflect_pack_plugin",
            "next_qk_rope_plugin",
            "ocab_qkv_pack",
            "ocab_qkv_pack_token_major",
            "ffn_swiglu_depthwise_plugin",
            "residual_image_layer_norm_plugin",
        ),
        default="dense_fusion",
    )
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--query-window", type=int, default=32)
    parser.add_argument("--key-window", type=int, default=48)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    report_path = args.report.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite candidate artifacts")

    model = onnx.load(str(source), load_external_data=True)
    nodes_before = len(model.graph.node)
    initializers_before = len(model.graph.initializer)
    if args.mode == "dense_fusion":
        reports = decompose_dense_fusion(model)
        mode = "dense_fusion_block_matmul_sum_v1"
        identity = "concat(x_i) @ concat_rows(W_i) == sum(x_i @ W_i)"
    elif args.mode == "image_layer_norm":
        reports = fuse_image_layer_norm(model)
        mode = "onnx_layer_normalization_v1"
        identity = "affine((x - mean(x)) / sqrt(mean((x - mean(x))^2) + eps))"
    elif args.mode == "image_layer_norm_plugin":
        reports = fuse_image_layer_norm(
            model, plugin_op="DrftImageLayerNorm_TRT",
        )
        mode = "drft_image_layer_norm_plugin_v1"
        identity = "affine((x - mean(x)) / sqrt(mean((x - mean(x))^2) + eps))"
    elif args.mode == "image_layer_norm_fast_plugin":
        reports = fuse_image_layer_norm(
            model, plugin_op="DrftImageLayerNormFast_TRT",
        )
        mode = "drft_image_layer_norm_fast_plugin_v1"
        identity = "affine((x - mean(x)) / sqrt(mean((x - mean(x))^2) + eps))"
    elif args.mode == "next_image_layer_norm_plugin":
        reports = fuse_next_image_layer_norm_plugin(model)
        mode = "drft_next_fp32_statistics_nchw_plugin_v1"
        identity = (
            "BF16 affine(FP32 (x - mean(x)) / "
            "sqrt(mean((x - mean(x))^2) + eps)) over NCHW"
        )
    elif args.mode == "next_ocab_reflect_pack_plugin":
        reports = fuse_next_ocab_qkv_reflect_pack(
            model,
            channels=args.channels,
            heads=args.heads,
            query_window=args.query_window,
            key_window=args.key_window,
        )
        mode = "drft_next_ocab_reflect_pack_token_major_plugin_v1"
        identity = (
            "BF16 element-preserving Q/K/V window gather with exact reflect "
            "coordinates and metadata-only attention layout transposes"
        )
    elif args.mode == "next_qk_rope_plugin":
        reports = fuse_next_qk_rmsnorm_rope(model)
        mode = "drft_next_qk_rmsnorm_rope_plugin_v1"
        identity = (
            "FP32 QK RMS statistics with BF16-rounded normalization, mixed "
            "RoPE operations, and query temperature"
        )
    elif args.mode == "ocab_qkv_pack":
        reports = fuse_ocab_qkv_pack(
            model,
            channels=args.channels,
            heads=args.heads,
            query_window=args.query_window,
            key_window=args.key_window,
        )
        mode = "drft_ocab_qkv_pack_plugin_v1"
        identity = "BF16 element-preserving Q/K/V window gather and layout"
    elif args.mode == "ocab_qkv_pack_token_major":
        reports = fuse_ocab_qkv_pack(
            model,
            channels=args.channels,
            heads=args.heads,
            query_window=args.query_window,
            key_window=args.key_window,
            token_major=True,
        )
        mode = "drft_ocab_qkv_pack_token_major_plugin_v2"
        identity = (
            "BF16 element-preserving Q/K/V window gather with metadata-only "
            "attention layout transposes"
        )
    elif args.mode == "image_layer_norm_cooperative_plugin":
        reports = retarget_image_layer_norm_plugin(
            model,
            source_op="DrftImageLayerNorm_TRT",
            target_op="DrftImageLayerNormCooperative_TRT",
        )
        mode = "drft_image_layer_norm_cooperative_plugin_v1"
        identity = (
            "unchanged two-pass centered-variance reduction and BF16 rounding"
        )
    elif args.mode == "ffn_swiglu_depthwise_plugin":
        reports = fuse_ffn_swiglu_depthwise(model)
        mode = "drft_ffn_swiglu_depthwise_plugin_v1"
        identity = (
            "BF16 SwiGLU followed by FP16 3x3 depthwise convolution and BF16 bias"
        )
    else:
        reports = fuse_residual_image_layer_norm(model)
        mode = "drft_residual_image_layer_norm_plugin_v1"
        identity = (
            "BF16 residual Add output plus unchanged two-pass image layer norm"
        )
    onnx.save(model, str(output))
    report = {
        "status": "rewritten",
        "mode": mode,
        "mathematical_identity": identity,
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "nodes_before": nodes_before,
        "nodes_after": len(model.graph.node),
        "initializers_before": initializers_before,
        "initializers_after": len(model.graph.initializer),
        "instances": reports,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
