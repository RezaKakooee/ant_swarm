"""Gym / Gymnasium dual-import shim.

Classic OpenAI ``gym`` and ``gymnasium`` share the same Env/spaces/wrappers
surface but live under different package names.  Import ``gym`` and ``spaces``
from here so the rest of the package is agnostic.
"""
from __future__ import annotations

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_PACKAGE = "gymnasium"
except ImportError:  # pragma: no cover
    import gym  # type: ignore[no-redef]
    from gym import spaces  # type: ignore[no-redef]
    GYM_PACKAGE = "gym"

__all__ = ["gym", "spaces", "GYM_PACKAGE"]
