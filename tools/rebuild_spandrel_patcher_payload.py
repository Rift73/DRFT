#!/usr/bin/env python3
"""Rebuild the patcher's deterministic embedded ZIP from source overlays."""

from __future__ import annotations

import base64
import hashlib
import io
import re
import textwrap
import zipfile
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parent
PATCHER = TOOLS_ROOT / "patch_spandrel_drft.py"
OVERLAY_ROOT = TOOLS_ROOT / "spandrel_payload"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _existing_payload(text: str) -> dict[str, bytes]:
    match = re.search(r'_PAYLOAD_B64 = """\n(.*?)\n"""', text, re.DOTALL)
    if match is None:
        raise RuntimeError("patcher has no embedded payload")
    archive = base64.b64decode("".join(match.group(1).split()), validate=True)
    with zipfile.ZipFile(io.BytesIO(archive), "r") as bundle:
        return {
            info.filename: bundle.read(info)
            for info in bundle.infolist()
            if not info.is_dir()
        }


def _overlay(files: dict[str, bytes]) -> None:
    for path in sorted(OVERLAY_ROOT.rglob("*.py")):
        relative = path.relative_to(OVERLAY_ROOT).as_posix()
        files[relative] = path.read_bytes()


def _archive(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as bundle:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            bundle.writestr(info, files[name])
    return output.getvalue()


def main() -> None:
    text = PATCHER.read_text(encoding="utf-8")
    files = _existing_payload(text)
    _overlay(files)
    archive = _archive(files)

    manifest = "\n".join(
        f"    {name!r}: {_sha256(content)!r},"
        for name, content in sorted(files.items())
    )
    encoded = "\n".join(textwrap.wrap(base64.b64encode(archive).decode("ascii"), 100))
    replacement = (
        f'PAYLOAD_SHA256 = "{_sha256(archive)}"\n'
        "PAYLOAD_FILES: dict[str, str] = {\n"
        f"{manifest}\n"
        "}\n\n"
        '_PAYLOAD_B64 = """\n'
        f"{encoded}\n"
        '"""'
    )
    pattern = re.compile(
        r'PAYLOAD_SHA256 = ".*?"\n'
        r"PAYLOAD_FILES: dict\[str, str\] = \{\n.*?\n\}\n\n"
        r'_PAYLOAD_B64 = """\n.*?\n"""',
        re.DOTALL,
    )
    updated, count = pattern.subn(lambda _match: replacement, text, count=1)
    if count != 1:
        raise RuntimeError("unable to replace patcher payload block")
    PATCHER.write_text(updated, encoding="utf-8", newline="")
    print(f"Updated {PATCHER}")
    print(f"Payload files: {len(files)}")
    print(f"Payload SHA-256: {_sha256(archive)}")


if __name__ == "__main__":
    main()
