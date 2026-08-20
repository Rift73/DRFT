#!/usr/bin/env python3
"""Mark a DRFT-Next checkpoint as DRT without changing learned tensors.

DRFT-Next and DRT use the same parameter paths and tensor shapes. DRT adds two
small persistent metadata tensors so deployment loaders can identify the model
and reconstruct window/numerical settings that cannot be inferred from learned
weights. This converter adds only those tensors and always preserves the source.
"""

from __future__ import annotations

import argparse
import math
import os
import uuid
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import torch

DRT_CHECKPOINT_VERSION = 1
_TORCH_SUFFIXES = {".ckpt", ".pt", ".pth"}
_KNOWN_STATE_KEYS = (
    "params_ema",
    "params-ema",
    "params",
    "model_state_dict",
    "state_dict",
    "model",
    "net_g_ema",
    "net_g",
    "net",
)
_PREFIXES = ("", "module.", "_orig_mod.", "module._orig_mod.", "net_g.", "netG.")
_SIGNATURE = (
    "conv_first.weight",
    "conv_after_body.weight",
    "layers.0.residual_group.blocks.0.attn.rope.frequencies",
    "layers.0.residual_group.blocks.0.attn.logit_scale",
    "layers.0.residual_group.blocks.0.ffn.fc_gate_value.weight",
    "layers.0.residual_group.ocab.rope.frequencies",
    "layers.0.residual_group.ocab.logit_scale",
)


def _find_prefix(state_dict: MutableMapping[str, Any]) -> str | None:
    for prefix in _PREFIXES:
        if all(prefix + key in state_dict for key in _SIGNATURE):
            return prefix
    return None


def _state_dict_targets(
    checkpoint: MutableMapping[str, Any],
    param_key: str | None,
) -> list[tuple[str, MutableMapping[str, Any], str]]:
    if param_key is not None:
        candidate = checkpoint.get(param_key)
        if not isinstance(candidate, MutableMapping):
            raise KeyError(f"checkpoint has no mapping at --param-key {param_key!r}")
        prefix = _find_prefix(candidate)
        if prefix is None:
            raise ValueError(f"{param_key!r} is not a DRFT-Next state dict")
        return [(param_key, candidate, prefix)]

    root_prefix = _find_prefix(checkpoint)
    if root_prefix is not None:
        return [("<root>", checkpoint, root_prefix)]

    targets: list[tuple[str, MutableMapping[str, Any], str]] = []
    visited: set[int] = set()
    names = (*_KNOWN_STATE_KEYS, *checkpoint.keys())
    for name in names:
        candidate = checkpoint.get(name)
        if not isinstance(candidate, MutableMapping) or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        prefix = _find_prefix(candidate)
        if prefix is not None:
            targets.append((str(name), candidate, prefix))
    if not targets:
        raise ValueError(
            "no DRFT-Next state dict was found; expected RoPE/QK-normalized "
            "DRFT-Next tensor keys"
        )
    return targets


def _metadata_tensors(
    *,
    window_size: int,
    overlap_window_size: int,
    rope_base: float,
    qk_eps: float,
    iln_eps: float,
) -> dict[str, torch.Tensor]:
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


def _validate_settings(
    *,
    window_size: int,
    overlap_window_size: int,
    rope_base: float,
    qk_eps: float,
    iln_eps: float,
) -> None:
    if window_size < 1:
        raise ValueError("window size must be positive")
    if overlap_window_size < window_size:
        raise ValueError("overlap window must be at least the attention window")
    if (overlap_window_size - window_size) % 2:
        raise ValueError(
            "overlap minus window size must be even for reflection padding"
        )
    if not math.isfinite(rope_base) or rope_base <= 1.0:
        raise ValueError("RoPE base must be finite and greater than one")
    if not math.isfinite(qk_eps) or qk_eps <= 0.0:
        raise ValueError("Q/K epsilon must be finite and positive")
    if not math.isfinite(iln_eps) or iln_eps <= 0.0:
        raise ValueError("i-LN epsilon must be finite and positive")


def _add_drt_metadata(
    checkpoint: MutableMapping[str, Any],
    *,
    param_key: str | None,
    window_size: int,
    overlap_window_size: int,
    rope_base: float,
    qk_eps: float,
    iln_eps: float,
) -> tuple[list[str], int, int]:
    targets = _state_dict_targets(checkpoint, param_key)
    metadata = _metadata_tensors(
        window_size=window_size,
        overlap_window_size=overlap_window_size,
        rope_base=rope_base,
        qk_eps=qk_eps,
        iln_eps=iln_eps,
    )
    converted: list[str] = []
    tensor_count = 0
    parameter_elements = 0
    for name, state_dict, prefix in targets:
        marker = prefix + "_drt_metadata"
        hyperparameters = prefix + "_drt_hyperparameters"
        if marker in state_dict or hyperparameters in state_dict:
            raise ValueError(f"state dict {name!r} is already marked as DRT")

        original = dict(state_dict)
        state_dict[marker] = metadata["_drt_metadata"].clone()
        state_dict[hyperparameters] = metadata["_drt_hyperparameters"].clone()
        if any(state_dict[key] is not value for key, value in original.items()):
            raise RuntimeError("conversion unexpectedly replaced a source tensor")

        learned = [
            value for value in original.values() if isinstance(value, torch.Tensor)
        ]
        tensor_count += len(learned)
        parameter_elements += sum(int(value.numel()) for value in learned)
        converted.append(name)
    return converted, tensor_count, parameter_elements


def _load_safetensors(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required for .safetensors conversion"
        ) from exc

    state = dict(load_file(str(path), device="cpu"))
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    return state, metadata


def _atomic_destination(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")


def _save_safetensors(
    state: dict[str, torch.Tensor],
    metadata: dict[str, str],
    destination: Path,
) -> None:
    from safetensors.torch import save_file

    temporary = _atomic_destination(destination)
    try:
        save_file(state, str(temporary), metadata=metadata)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _save_torch(checkpoint: Any, destination: Path) -> None:
    temporary = _atomic_destination(destination)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def default_destination(source: Path) -> Path:
    return source.with_name(f"{source.stem}_drt{source.suffix}")


def convert_checkpoint(
    source: Path,
    destination: Path,
    *,
    window_size: int = 32,
    overlap_window_size: int | None = None,
    rope_base: float = 10_000.0,
    qk_eps: float = 1e-6,
    iln_eps: float = 1e-4,
    param_key: str | None = None,
    unsafe_load: bool = False,
    overwrite: bool = False,
) -> tuple[list[str], int, int]:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == destination:
        raise ValueError(
            "source and destination must be different; conversion is non-destructive"
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"destination exists (pass --force to replace it): {destination}"
        )

    overlap = (
        window_size + window_size // 2
        if overlap_window_size is None
        else overlap_window_size
    )
    _validate_settings(
        window_size=window_size,
        overlap_window_size=overlap,
        rope_base=rope_base,
        qk_eps=qk_eps,
        iln_eps=iln_eps,
    )

    suffix = source.suffix.lower()
    if destination.suffix.lower() != suffix:
        raise ValueError("source and destination checkpoint formats must match")
    if suffix == ".safetensors":
        checkpoint, file_metadata = _load_safetensors(source)
        converted = _add_drt_metadata(
            checkpoint,
            param_key=param_key,
            window_size=window_size,
            overlap_window_size=overlap,
            rope_base=rope_base,
            qk_eps=qk_eps,
            iln_eps=iln_eps,
        )
        file_metadata.update(
            {
                "architecture": "DRT",
                "converted_from": "DRFT-Next",
                "drt_checkpoint_version": str(DRT_CHECKPOINT_VERSION),
            }
        )
        _save_safetensors(checkpoint, file_metadata, destination)
        return converted

    if suffix not in _TORCH_SUFFIXES:
        raise ValueError(f"unsupported checkpoint format: {source.suffix}")
    checkpoint = torch.load(
        source,
        map_location="cpu",
        weights_only=not unsafe_load,
    )
    if not isinstance(checkpoint, MutableMapping):
        raise TypeError("PyTorch checkpoint root must be a mapping")
    converted = _add_drt_metadata(
        checkpoint,
        param_key=param_key,
        window_size=window_size,
        overlap_window_size=overlap,
        rope_base=rope_base,
        qk_eps=qk_eps,
        iln_eps=iln_eps,
    )
    _save_torch(checkpoint, destination)
    return converted


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="DRFT-Next checkpoint")
    parser.add_argument(
        "destination",
        type=Path,
        nargs="?",
        help="output path (default: SOURCE_drt with the same extension)",
    )
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--overlap-window-size", type=int)
    parser.add_argument("--rope-base", type=float, default=10_000.0)
    parser.add_argument("--qk-eps", type=float, default=1e-6)
    parser.add_argument("--iln-eps", type=float, default=1e-4)
    parser.add_argument(
        "--param-key", help="explicit top-level state-dict key for .pth"
    )
    parser.add_argument(
        "--unsafe-load",
        action="store_true",
        help="allow pickle-based loading for trusted legacy PyTorch checkpoints",
    )
    parser.add_argument("--force", action="store_true", help="replace destination")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    destination = args.destination or default_destination(args.source)
    converted, tensor_count, elements = convert_checkpoint(
        args.source,
        destination,
        window_size=args.window_size,
        overlap_window_size=args.overlap_window_size,
        rope_base=args.rope_base,
        qk_eps=args.qk_eps,
        iln_eps=args.iln_eps,
        param_key=args.param_key,
        unsafe_load=args.unsafe_load,
        overwrite=args.force,
    )
    print(f"Wrote: {destination.expanduser().resolve()}")
    print(f"Converted state dicts: {', '.join(converted)}")
    print(f"Preserved learned tensors: {tensor_count:,} ({elements:,} elements)")
    print("Added: _drt_metadata, _drt_hyperparameters")


if __name__ == "__main__":
    main()
