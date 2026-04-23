from __future__ import annotations

from pathlib import Path


def resolve_path(path_value: str, base_dir: str | Path = ".") -> Path:
    """Resolve relative paths against project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(base_dir).resolve() / path
