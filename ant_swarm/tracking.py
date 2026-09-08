"""Weights & Biases tracking that never breaks a run.

Same convention as scripts/rl/train_sac.py: project ``ant_swarm``, run name = the run id (see ant_swarm/run_id.py), so the W&B run,
the run dir and the ops .out log share one name.  Any failure -- no network,
no login, package missing -- is logged and tracking is simply off.

    from ant_swarm.tracking import Tracker
    tr = Tracker(name=run_dir.name, group="dagger", tags=["marl"], config=vars(args),
                 enabled=args.wandb)
    tr.log({"eval/sr": 73.3}, step=8)
    tr.finish()
"""
from __future__ import annotations

from loguru import logger

import os

WANDB_PROJECT = os.environ.get("ANT_SWARM_WANDB_PROJECT", "ant_swarm")
# None = the entity the API key belongs to. The key on this cluster is user
# `rkakooee` (entities: rkakooee, gen-xr); the `kakooee` entity hardcoded in
# scripts/rl/train_sac.py is refused with "permission denied".
WANDB_ENTITY = os.environ.get("ANT_SWARM_WANDB_ENTITY") or None


class Tracker:
    def __init__(self, name: str, group: str, tags: list[str] | None = None,
                 config: dict | None = None, enabled: bool = True):
        self.run = None
        if not enabled:
            return
        try:
            import wandb
            self.run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=name,
                                  group=group, tags=list(tags or []), config=config or {},
                                  save_code=False)
            logger.info(f"W&B run   : {self.run.url}")
        except Exception as e:                      # noqa: BLE001
            logger.warning(f"wandb init failed -- continuing without tracking: {e}")
            self.run = None

    def log(self, data: dict, step: int | None = None) -> None:
        if self.run is None:
            return
        try:
            self.run.log(data, step=step)
        except Exception as e:                      # noqa: BLE001
            logger.warning(f"wandb log failed: {e}")

    def summary(self, data: dict) -> None:
        if self.run is None:
            return
        try:
            for k, v in data.items():
                self.run.summary[k] = v
        except Exception as e:                      # noqa: BLE001
            logger.warning(f"wandb summary failed: {e}")

    def finish(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:                       # noqa: BLE001
                pass
            self.run = None
