"""Ant-Swarm T-Barrier environment package.

Public API:
    from ant_swarm import AntSwarmEnv, GymCompatWrapper, load_config

Modular pieces (see each module's docstring):
    config       — load config.yaml into a namespace
    geometry     — LocalRect + SAT collision helpers (utils)
    layout       — world outline + inner barrier walls + goal
    tshape       — the movable T-shaped object
    state        — dynamic world state + physics integration
    action       — action space + per-ant force → wrench
    observation  — observation space + per-ant observation builder
    reward       — reward function
    ant_swarm    — the gym ``AntSwarmEnv`` composing the above
"""
from __future__ import annotations

from .config import load_config, load_config_dict
from .layout import Layout
from .tshape import TShape
from .state import SwarmState
from .action import ActionModel
from .observation import ObservationModel
from .reward import RewardModel
from .render import Renderer
from .ant_swarm import AntSwarmEnv, GymCompatWrapper, make_compat_env
from ._gym import gym

__all__ = [
    "AntSwarmEnv", "GymCompatWrapper", "make_compat_env", "Renderer",
    "load_config", "load_config_dict",
    "Layout", "TShape", "SwarmState",
    "ActionModel", "ObservationModel", "RewardModel",
]


def _register():
    try:
        ids = {spec.id for spec in gym.envs.registry.values()}
    except Exception:
        ids = set()
    max_steps = int(load_config().env.max_steps)
    if "AntSwarmBarrier-v0" not in ids:
        gym.register(id="AntSwarmBarrier-v0",
                     entry_point="ant_swarm:AntSwarmEnv", max_episode_steps=max_steps)
    if "AntSwarmBarrier-v0-compat" not in ids:
        gym.register(id="AntSwarmBarrier-v0-compat",
                     entry_point="ant_swarm:make_compat_env", max_episode_steps=max_steps)


_register()
