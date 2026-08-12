"""Run identifier helpers (mirrors phy_nav_1's phynav/utils/run_id.py).

Each run gets ONE id shared by the ops .out log file, the experiment directory
under storage_local/, and the wandb run name. Ops wrappers (ops/sb_train.sh)
mint the id up front and export it as ANT_SWARM_RUN_ID; Python generates one
only when the environment variable is absent (e.g. local runs).

Format (when generated here):
    ant__<YYYYMMDD_HHMM>__<slurm job id | local>__<script>[__<single|multi>][__<config tag>]
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path


def normalize_name(name: str) -> str:
    """Return a filesystem-friendly name segment."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name).strip())
    return cleaned.strip("_") or "run"


def build_run_id(script_name: str = "run", n_ants: int | None = None) -> str:
    """One id for run dir + wandb + logs. ANT_SWARM_RUN_ID overrides."""
    override = os.environ.get("ANT_SWARM_RUN_ID")
    if override:
        return normalize_name(override)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    parts = ["ant", ts, job_id, normalize_name(script_name)]
    if n_ants is not None:
        parts.append("single" if int(n_ants) == 1 else "multi")
    tag = Path(os.environ.get("ANT_SWARM_CONFIG", "")).stem
    if tag:
        parts.append(normalize_name(tag))
    return "__".join(parts)
