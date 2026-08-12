"""Config loading — OmegaConf-backed, with a Hydra-compose CLI entry.

Two ways to get a config:

1. Library / replay consumers (env registration, interactive sandbox, replay
   of run snapshots) call :func:`load_config`, which reads a single yaml —
   the ``ANT_SWARM_CONFIG`` env var if set, else ``configs/rl/config.yaml``.

2. Entry scripts call :func:`load_config_cli`, which adds Hydra on top:

       python scripts/rl/train_sac.py --config-name pnas_kin_geo
       python scripts/rl/train_sac.py env.reward_mode=sparse sac.timesteps=2e6
       ANT_SWARM_CONFIG=path/to/variant.yaml python scripts/rl/train_sac.py

   We use Hydra's *compose API* (not ``@hydra.main``) on purpose: run dirs,
   logging, and working directory stay under our control (storage_local/),
   and the yamls stay plain — no ``hydra:`` plumbing inside them.

Both return an ``omegaconf.DictConfig`` (attribute access: ``cfg.world.width``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

# package dir → project root → configs/rl/config.yaml
_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = _ROOT / "configs" / "rl" / "config.yaml"


def _default_path() -> Path:
    """Project config.yaml, unless ANT_SWARM_CONFIG points at a variant
    (used to run parallel sweeps, each SLURM job with its own config)."""
    return Path(os.environ.get("ANT_SWARM_CONFIG") or CONFIG_PATH)


def load_config(path: str | Path | None = None) -> DictConfig:
    """Load one yaml as a DictConfig (no CLI, no composition)."""
    return OmegaConf.load(Path(path) if path else _default_path())


def load_config_dict(path: str | Path | None = None) -> dict:
    """Return the config as a plain nested dict (used by the web-config generator)."""
    return OmegaConf.to_container(load_config(path), resolve=True)


def load_config_cli(config_dir: str | Path | None = None,
                    default_name: str = "config",
                    argv: list[str] | None = None) -> DictConfig:
    """Hydra-compose a config from the command line.

    Accepts ``--config-name/-cn <name>`` plus any ``key=value`` overrides.
    Priority: --config-name > ANT_SWARM_CONFIG > <config_dir>/<default_name>.
    """
    from hydra import compose, initialize_config_dir

    argv = list(sys.argv[1:] if argv is None else argv)
    config_dir = Path(config_dir) if config_dir else (_ROOT / "configs" / "rl")
    name = None
    for flag in ("--config-name", "-cn"):
        if flag in argv:
            i = argv.index(flag)
            name = argv[i + 1]
            del argv[i:i + 2]
    if name is None and os.environ.get("ANT_SWARM_CONFIG"):
        p = Path(os.environ["ANT_SWARM_CONFIG"])
        config_dir, name = p.parent, p.stem
    if name is None:
        name = default_name

    overrides = [a for a in argv if "=" in a and not a.startswith("-")]
    unknown = [a for a in argv if a not in overrides]
    if unknown:
        raise SystemExit(f"unknown args {unknown}; use key=value overrides "
                         f"or --config-name <name>")
    with initialize_config_dir(config_dir=str(config_dir.resolve()),
                               version_base=None):
        cfg = compose(config_name=name, overrides=overrides)
    # let downstream consumers (run-id tag, snapshot fallback) see the choice
    chosen = config_dir / f"{name}.yaml"
    if chosen.exists():
        os.environ.setdefault("ANT_SWARM_CONFIG", str(chosen.resolve()))
    return cfg
