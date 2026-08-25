"""Run Go-Explore phase 1 on the configured env. No policy is trained.

    python scripts/rl/run_go_explore.py [key=value ...]

Reads the usual config (ANT_SWARM_CONFIG) plus a ``go_explore:`` section:

    go_explore:
      total_steps: 2000000
      cell_xy: 0.02
      cell_deg: 30
      explore_steps: 60
      select_power: 1.0

Found solutions are saved as success JSONs under
``storage_local/<run_id>/successes`` — ready for ``run.seed_successes_from``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ant_swarm import AntSwarmEnv, build_run_id, load_config_cli, setup_logging  # noqa: E402
from go_explore import GoExplore  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    cfg = load_config_cli(sys.argv[1:])
    g = getattr(cfg, "go_explore", None)
    run_dir = PROJECT_ROOT / "storage_local" / build_run_id("go_explore", int(cfg.ants.n))
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir)

    wandb_run = None
    if bool(getattr(g, "wandb", True)):
        try:
            import wandb
            wandb_run = wandb.init(project="ant_swarm", entity="kakooee", name=run_dir.name,
                                   group="go_explore",
                                   tags=["go_explore"])
            logger.info(f"W&B run: {wandb_run.url}")
        except Exception as e:
            logger.warning(f"wandb init failed — continuing without: {e}")

    env = AntSwarmEnv(config=cfg, seed=0)
    ge = GoExplore(
        env,
        cell_xy=float(getattr(g, "cell_xy", 0.02)),
        cell_deg=float(getattr(g, "cell_deg", 30.0)),
        explore_steps=int(getattr(g, "explore_steps", 60)),
        select_power=float(getattr(g, "select_power", 1.0)),
        rng=np.random.default_rng(0),
    )
    total = int(float(getattr(g, "total_steps", 2_000_000)))
    logger.info(f"Go-Explore: {total} env steps, archive grid "
                f"{ge.cell_xy} m x {math_deg(ge.cell_th)} deg")
    def log(msg):
        logger.info(msg)
        if wandb_run is not None:
            s = ge.stats()
            wandb_run.log({"go_explore/steps": s["steps"],
                           "go_explore/cells": s["cells"],
                           "go_explore/successes": s["successes"],
                           "go_explore/frontier_x": s["frontier_x"]})

    stats = ge.run(total, log=log)
    n = ge.save_successes(run_dir / "successes")
    logger.info(f"Done: {stats}; {n} solutions saved to {run_dir/'successes'}")
    if wandb_run is not None:
        wandb_run.summary.update(stats)
        wandb_run.finish()


def math_deg(rad: float) -> float:
    import math
    return round(math.degrees(rad), 1)


if __name__ == "__main__":
    main()
