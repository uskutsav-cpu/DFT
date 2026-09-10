"""Content hashes, atomic artifacts, and non-secret reproducibility records."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: str | Path, text: str, *, overwrite: bool = True) -> None:
    """Atomic replacement, or an atomic create-if-absent using a hard link."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: str | Path, value: Any, *, overwrite: bool = True) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n"
    atomic_text(path, text, overwrite=overwrite)


def read_json(path: str | Path) -> Any:
    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant: {value}")

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    with Path(path).open(encoding="utf-8") as stream:
        return json.load(
            stream, parse_constant=reject_constant, object_pairs_hook=reject_duplicates
        )


def environment_record(root: str | Path | None = None) -> dict:
    """Whitelist runtime information. Do not capture env vars or credentials."""
    versions = {}
    for name in ("refblind", "numpy", "PyYAML", "ase", "fairchem-core", "mace-torch", "pyscf"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    revision = None
    if root is not None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            revision = result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "created_utc": utc_now(),
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "versions": versions,
        "git_commit": revision,
    }


def freeze_protocol(config: dict, files: list[str | Path], output: str | Path) -> dict:
    """Write a content-addressed snapshot, never silently rewrite a freeze."""
    inputs = [{"path": str(p), "sha256": file_digest(p)} for p in files]
    content = {"config": config, "inputs": inputs}
    result = {**content, "protocol_sha256": digest(content), "created_utc": utc_now()}
    write_json(output, result, overwrite=False)
    return result
