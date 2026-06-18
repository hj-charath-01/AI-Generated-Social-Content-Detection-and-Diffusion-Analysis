"""
Configuration loader and manager.
"""

import os
from pathlib import Path
from typing import Any

import yaml


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
_config_cache: dict[str, Any] | None = None


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Optional path to config file. Defaults to project root config.yaml.

    Returns:
        Configuration dictionary.
    """
    global _config_cache

    if config_path is not None:
        path = Path(config_path)
    else:
        path = _CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    _config_cache = config
    return config


def get_config() -> dict[str, Any]:
    """
    Return cached config, loading if necessary.

    Returns:
        Configuration dictionary.
    """
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config()
    return _config_cache


def get_path(key: str) -> Path:
    """
    Resolve a named path from config relative to the project root.

    Args:
        key: Key under config['paths'], e.g. 'data_dir', 'models_dir'.

    Returns:
        Absolute Path object.
    """
    cfg = get_config()
    project_root = Path(__file__).resolve().parents[2]
    relative = cfg["paths"].get(key)
    if relative is None:
        raise KeyError(f"Path key '{key}' not found in config.paths")
    return project_root / relative


def ensure_dirs() -> None:
    """Create all configured directories if they don't exist."""
    cfg = get_config()
    project_root = Path(__file__).resolve().parents[2]
    for key in cfg.get("paths", {}):
        directory = project_root / cfg["paths"][key]
        directory.mkdir(parents=True, exist_ok=True)