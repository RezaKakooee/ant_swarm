"""Snapshot the source code + config into a run directory for reproducibility.

    from ant_swarm import save_code
    save_code(run_dir, __file__)

Copies into ``<run_dir>/code/``:
  * the ``ant_swarm`` package source (minus __pycache__),
  * the project ``config.yaml`` (the exact params used),
  * the entry script (``train_ppo.py`` / ``train_sac.py`` / ``random_agent.py``).
"""
from __future__ import annotations

import shutil
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent        # .../ant_swarm/ant_swarm
_ROOT = _PKG_DIR.parent                           # project root


def save_code(run_dir, script_path: str | None = None) -> Path:
    dest = Path(run_dir) / "code"
    dest.mkdir(parents=True, exist_ok=True)

    # package source
    shutil.copytree(
        _PKG_DIR, dest / "ant_swarm",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        dirs_exist_ok=True,
    )
    # config.yaml
    cfg = _ROOT / "config.yaml"
    if cfg.exists():
        shutil.copy2(cfg, dest / "config.yaml")
    # entry script
    if script_path:
        sp = Path(script_path)
        if sp.exists():
            shutil.copy2(sp, dest / sp.name)
    return dest
