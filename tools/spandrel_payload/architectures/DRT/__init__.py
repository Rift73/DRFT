# ruff: noqa
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

try:
    from typing_extensions import override
except ImportError:

    def override(function):  # type: ignore[no-untyped-def]
        return function


try:
    from spandrel.util import KeyCondition, get_seq_len
except ImportError:

    class KeyCondition:
        """Compatibility subset for Spandrel versions before ``spandrel.util``."""

        @staticmethod
        def has_all(*keys: str):
            return lambda state_dict: all(key in state_dict for key in keys)

    def get_seq_len(state_dict: Mapping[str, object], seq_key: str) -> int:
        prefix = seq_key + "."
        indices = {
            int(key[len(prefix) :].split(".", maxsplit=1)[0])
            for key in state_dict
            if key.startswith(prefix)
        }
        return max(indices) + 1 if indices else 0


try:
    from ...__helpers import model_descriptor as _model_descriptor
except ImportError:
    import spandrel as _model_descriptor  # type: ignore[no-redef]

ImageModelDescriptor = _model_descriptor.ImageModelDescriptor
SizeRequirements = _model_descriptor.SizeRequirements
StateDict = _model_descriptor.StateDict
_MODERN_ARCHITECTURE_API = hasattr(_model_descriptor, "Architecture")

if _MODERN_ARCHITECTURE_API:
    Architecture = _model_descriptor.Architecture
else:

    class Architecture:
        """Facade matching the Architecture API introduced in Spandrel 0.3."""

        def __init__(self, *, id: str, detect) -> None:  # type: ignore[no-untyped-def]
            self.id = id
            self.detect = detect

        def __class_getitem__(cls, _item: Any):
            return cls


from ..DRFT import (
    _detect_channel_squeeze_factor,
    _detect_mlp_ratio,
    _detect_residual_connection,
    _get_drft_upscale,
    _prepare_folded_edbb,
    _size_tag,
)
from .__arch import DRT, DRT_CHECKPOINT_VERSION


def _descriptor(
    architecture: "DRTArch",
    model: DRT,
    state_dict: StateDict,
    *,
    upscale: int,
    in_chans: int,
    tags: list[str],
) -> ImageModelDescriptor[DRT]:
    return ImageModelDescriptor(
        model,
        state_dict,
        architecture=architecture if _MODERN_ARCHITECTURE_API else architecture.id,
        purpose="Restoration" if upscale == 1 else "SR",
        tags=tags,
        supports_half=True,
        supports_bfloat16=True,
        scale=upscale,
        input_channels=in_chans,
        output_channels=in_chans,
        size_requirements=SizeRequirements(minimum=16),
    )


def _metadata_values(
    state_dict: StateDict,
) -> tuple[int, int, float, float, float]:
    metadata = state_dict["_drt_metadata"].reshape(-1)  # type: ignore[attr-defined]
    hyperparameters = state_dict["_drt_hyperparameters"].reshape(-1)  # type: ignore[attr-defined]
    if metadata.numel() != 3 or hyperparameters.numel() != 3:  # type: ignore[attr-defined]
        raise ValueError("invalid DRT checkpoint metadata shape")

    version, window_size, overlap_window_size = (
        int(value)
        for value in metadata.tolist()  # type: ignore[attr-defined]
    )
    rope_base, qk_eps, iln_eps = (
        float(value)
        for value in hyperparameters.tolist()  # type: ignore[attr-defined]
    )
    if version != DRT_CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported DRT checkpoint version {version}; "
            f"expected {DRT_CHECKPOINT_VERSION}"
        )
    if window_size < 1 or overlap_window_size < window_size:
        raise ValueError("invalid DRT window geometry")
    if (overlap_window_size - window_size) % 2:
        raise ValueError("DRT overlap halo must be symmetric")
    if not math.isfinite(rope_base) or rope_base <= 1.0:
        raise ValueError("invalid DRT RoPE base")
    if not math.isfinite(qk_eps) or qk_eps <= 0.0:
        raise ValueError("invalid DRT Q/K epsilon")
    if not math.isfinite(iln_eps) or iln_eps <= 0.0:
        raise ValueError("invalid DRT i-LN epsilon")
    return window_size, overlap_window_size, rope_base, qk_eps, iln_eps


def _get_structure(state_dict: StateDict) -> tuple[tuple[int, ...], tuple[int, ...]]:
    num_layers = get_seq_len(state_dict, "layers")
    if num_layers < 1:
        raise ValueError("DRT checkpoint has no residual groups")
    depths = tuple(
        get_seq_len(state_dict, f"layers.{stage}.residual_group.blocks")
        for stage in range(num_layers)
    )
    num_heads = tuple(
        int(
            state_dict[f"layers.{stage}.residual_group.ocab.logit_scale"].shape[0]  # type: ignore[index]
        )
        for stage in range(num_layers)
    )
    if any(depth < 1 for depth in depths) or any(heads < 1 for heads in num_heads):
        raise ValueError("invalid DRT group structure")
    return depths, num_heads


class DRTArch(Architecture[DRT]):
    def __init__(self) -> None:
        super().__init__(
            id="DRT",
            detect=KeyCondition.has_all(
                "_drt_metadata",
                "_drt_hyperparameters",
                "conv_first.weight",
                "conv_after_body.weight",
                "layers.0.residual_group.blocks.0.attn.rope.frequencies",
                "layers.0.residual_group.blocks.0.attn.logit_scale",
                "layers.0.residual_group.blocks.0.ffn.fc_gate_value.weight",
                "layers.0.residual_group.ocab.rope.frequencies",
                "layers.0.residual_group.ocab.logit_scale",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[DRT]:
        in_chans = int(state_dict["conv_first.weight"].shape[1])  # type: ignore[index]
        embed_dim = int(state_dict["conv_first.weight"].shape[0])  # type: ignore[index]
        reconstruction = "progressive" if "conv_last.weight" in state_dict else "direct"
        num_feat = (
            int(state_dict["conv_last.weight"].shape[1])  # type: ignore[index]
            if reconstruction == "progressive"
            else 64
        )
        upscale = _get_drft_upscale(
            state_dict,
            in_chans=in_chans,
            num_feat=num_feat,
        )
        depths, num_heads = _get_structure(state_dict)
        window_size, overlap_window_size, rope_base, qk_eps, iln_eps = _metadata_values(
            state_dict
        )

        if any(embed_dim % heads for heads in num_heads):
            raise ValueError("DRT channel width must be divisible by every head count")
        hidden_dim = (
            int(
                state_dict[
                    "layers.0.residual_group.blocks.0.ffn.fc_gate_value.weight"
                ].shape[0]  # type: ignore[index]
            )
            // 2
        )
        mlp_ratio = _detect_mlp_ratio(embed_dim, hidden_dim)
        qkv_bias = "layers.0.residual_group.blocks.0.attn.qkv.bias" in state_dict
        dense_skip = "layers.0.residual_group.dense_fusion.weight" in state_dict
        resi_connection = _detect_residual_connection(state_dict)
        folded = (
            "layers.0.residual_group.blocks.0.conv_block.0._folded_conv.weight"
            in state_dict
        )

        edbb_depth_multiplier = 1.0
        edbb_probe = "layers.0.residual_group.blocks.0.conv_block.0.conv1x1_3x3.k0"
        if edbb_probe in state_dict:
            edbb_depth_multiplier = (
                int(state_dict[edbb_probe].shape[0]) / embed_dim  # type: ignore[index]
            )
        channel_squeeze_factor = _detect_channel_squeeze_factor(
            state_dict,
            embed_dim,
        )
        rhag_layer_scale_init = 1e-4 if "layers.0.ls_rhag.gamma" in state_dict else None

        model = DRT(
            img_size=64,
            patch_size=1,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            unshifted_num_heads=num_heads,
            unshifted_attention_dim=(embed_dim,) * len(depths),
            window_size=window_size,
            overlap_window_size=overlap_window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            conv_scale=0.01,
            layer_scale_init=1e-6,
            rhag_layer_scale_init=rhag_layer_scale_init,
            use_checkpoint=False,
            upscale=upscale,
            img_range=1.0,
            resi_connection=resi_connection,
            dense_skip=dense_skip,
            num_feat=num_feat,
            rank=32,
            attn_type="masked",
            reconstruction=reconstruction,
            full_width_unshifted=True,
            edbb_depth_multiplier=edbb_depth_multiplier,
            channel_squeeze_factor=channel_squeeze_factor,
            iln_eps=iln_eps,
            rope_base=rope_base,
            qk_eps=qk_eps,
        )
        if folded:
            _prepare_folded_edbb(model)

        tags = [
            _size_tag(len(depths)),
            "Dense-Rotary",
            "iLN",
            "EDBB",
            f"G{len(depths)}",
            f"w{window_size}",
            f"{embed_dim}dim",
            resi_connection,
            reconstruction,
        ]
        if reconstruction == "progressive":
            tags.append(f"{num_feat}nf")
        if folded:
            tags.append("folded")
        if rhag_layer_scale_init is not None:
            tags.append("RHAG-LS")

        return _descriptor(
            self,
            model,
            state_dict,
            upscale=upscale,
            in_chans=in_chans,
            tags=tags,
        )


__all__ = ["DRTArch", "DRT"]
