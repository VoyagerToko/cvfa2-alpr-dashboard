from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config into a dictionary."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_directories(config: dict[str, Any]) -> None:
    """Create key project directories if they do not exist."""
    path_cfg = config.get("paths", {})
    for key in ["processed_dir", "checkpoints_dir", "artifacts_dir"]:
        if key in path_cfg:
            Path(path_cfg[key]).mkdir(parents=True, exist_ok=True)
