# ruff: noqa
"""Register bundled architectures without depending on Spandrel's source layout."""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Iterable, Mapping
from typing import Any

from . import DRFTArch
from ..DRT import DRTArch
from ..ESCRIB import ESCRIBArch


def _architecture_id(value: Any) -> str | None:
    architecture = getattr(value, "architecture", None)
    identifier = getattr(architecture, "id", None)
    if identifier is None:
        identifier = getattr(value, "id", None)
    if identifier is None and isinstance(value, tuple) and value:
        identifier = value[0]
    return str(identifier) if identifier is not None else None


def _registry_has(registry: Any, identifier: str) -> bool:
    getter = getattr(registry, "get", None)
    if callable(getter):
        try:
            if getter(identifier) is not None:
                return True
        except Exception:
            pass

    try:
        if identifier in registry:
            return True
    except Exception:
        pass

    return identifier in _registry_ids(registry)


def _registry_ids(registry: Any) -> tuple[str, ...]:
    values: Iterable[Any] | None = None
    if isinstance(registry, Mapping):
        values = registry.values()
    else:
        for name in ("architectures", "items", "values"):
            accessor = getattr(registry, name, None)
            if not callable(accessor):
                continue
            for arguments in ((), ("insertion",), ("detection",)):
                try:
                    candidate = accessor(*arguments)
                except Exception:
                    continue
                if isinstance(candidate, Iterable):
                    values = candidate
                    break
            if values is not None:
                break
    if values is None:
        try:
            values = iter(registry)
        except Exception:
            return ()
    try:
        return tuple(
            identifier
            for value in values
            if (identifier := _architecture_id(value)) is not None
        )
    except Exception:
        return ()


def _find_registry(namespace: Mapping[str, Any]) -> Any:
    direct = namespace.get("MAIN_REGISTRY")
    if direct is not None:
        return direct

    loader_type = namespace.get("ModelLoader")
    if loader_type is not None:
        try:
            registry = getattr(loader_type(), "registry", None)
        except Exception:
            registry = None
        if registry is not None:
            return registry

    candidates: list[Any] = []
    for module_name, module in tuple(sys.modules.items()):
        if module is None or not module_name.startswith("spandrel"):
            continue
        registry = getattr(module, "MAIN_REGISTRY", None)
        if registry is not None:
            return registry
        for name, value in vars(module).items():
            if "registry" not in name.lower() or inspect.ismodule(value):
                continue
            if any(
                callable(getattr(value, method, None))
                for method in ("add", "register", "append", "load")
            ):
                candidates.append(value)
    if candidates:
        return candidates[0]
    raise RuntimeError(
        "Architecture patch could not discover Spandrel's live architecture registry"
    )


def _find_arch_support(namespace: Mapping[str, Any]) -> type[Any] | None:
    support = namespace.get("ArchSupport")
    if isinstance(support, type):
        return support
    for module_name in (
        "spandrel.__helpers.registry",
        "spandrel.registry",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        support = getattr(module, "ArchSupport", None)
        if isinstance(support, type):
            return support
    for module_name, module in tuple(sys.modules.items()):
        if module is None or not module_name.startswith("spandrel"):
            continue
        support = getattr(module, "ArchSupport", None)
        if isinstance(support, type):
            return support
    return None


def _make_support(
    namespace: Mapping[str, Any],
    registry: Any,
    architecture: Any,
    identifier: str,
) -> Any | None:
    support_type = _find_arch_support(namespace)
    if support_type is None:
        return None
    before = tuple(
        existing for existing in _registry_ids(registry) if existing != identifier
    )
    factory = getattr(support_type, "from_architecture", None)
    if callable(factory):
        for kwargs in ({"before": before}, {}):
            try:
                return factory(architecture, **kwargs)
            except (AttributeError, TypeError):
                continue
    for kwargs in (
        {
            "id": identifier,
            "detect": architecture.detect,
            "load": architecture.load,
            "before": before,
        },
        {"architecture": architecture, "detect": architecture.detect, "before": before},
    ):
        try:
            return support_type(**kwargs)
        except (TypeError, ValueError):
            continue
    return None


def _registration_attempts(
    registry: Any,
    architecture: Any,
    support: Any | None,
    identifier: str,
):
    values = [value for value in (support, architecture) if value is not None]
    add = getattr(registry, "add", None)
    if callable(add):
        for value in values:
            yield f"add({type(value).__name__})", lambda value=value: add(value)
            yield (
                f"add({type(value).__name__}, ignore_duplicates=True)",
                lambda value=value: add(value, ignore_duplicates=True),
            )

    register = getattr(registry, "register", None)
    if callable(register):
        for value in values:
            yield (
                f"register({type(value).__name__})",
                lambda value=value: register(value),
            )
        yield (
            "register(id/detect/load)",
            lambda: register(
                id=identifier, detect=architecture.detect, load=architecture.load
            ),
        )

    append = getattr(registry, "append", None)
    if callable(append):
        for value in values:
            yield f"append({type(value).__name__})", lambda value=value: append(value)


def _register_architecture(
    namespace: Mapping[str, Any],
    registry: Any,
    architecture_type: type[Any],
) -> Any:
    architecture = architecture_type()
    identifier = str(architecture.id)
    if _registry_has(registry, identifier):
        return registry

    support = _make_support(namespace, registry, architecture, identifier)
    failures: list[str] = []
    for label, attempt in _registration_attempts(
        registry, architecture, support, identifier
    ):
        try:
            attempt()
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
        if _registry_has(registry, identifier):
            return registry

    methods = sorted(
        name
        for name in dir(registry)
        if not name.startswith("_") and callable(getattr(registry, name, None))
    )
    detail = "; ".join(failures[-6:]) or "no compatible mutator was exposed"
    raise RuntimeError(
        "Architecture patch found registry "
        f"{type(registry).__module__}.{type(registry).__qualname__} but could not "
        f"register {identifier}; methods={methods}; attempts={detail}"
    )


def register_drft(namespace: Mapping[str, Any]) -> Any:
    """Register all bundled architectures through the legacy DRFT hook."""
    registry = _find_registry(namespace)
    for architecture_type in (DRFTArch, DRTArch, ESCRIBArch):
        _register_architecture(namespace, registry, architecture_type)
    return registry


__all__ = ["register_drft"]
