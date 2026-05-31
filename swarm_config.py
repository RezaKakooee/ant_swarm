"""Load the single-source-of-truth ``config.yaml`` into an attribute namespace.

    from swarm_config import load_config
    cfg = load_config()
    cfg.world.width          # 1.25
    cfg.walls.x_columns      # [0.46, 0.70]

Lists of scalars stay plain lists; nested mappings become ``SimpleNamespace``
so they read like attributes.
"""
from __future__ import annotations

import types
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _to_ns(obj):
    if isinstance(obj, dict):
        return types.SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(x) for x in obj]
    return obj


def load_config_dict(path: str | Path | None = None) -> dict:
    """Return the raw nested dict (used by the web-config generator)."""
    path = Path(path) if path else CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(path: str | Path | None = None) -> types.SimpleNamespace:
    """Return the config as a nested attribute namespace."""
    return _to_ns(load_config_dict(path))
