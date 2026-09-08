"""Audit/repair cached goal vectors using the original demo episode targets.

Only obs[:, 4:6] is changed. Physics replay is unnecessary: goals affect
observations and termination, but not the recorded actions or object dynamics.
The old cache is retained; use --out to write a separate corrected cache.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ant_swarm import load_config
from ant_swarm.tshape import TShape


def goal_vectors(obs, episode_ids, goals, cfg):
    """Reconstruct the true goal vector, including the configured tracked point."""
    if int(cfg.ants.n) != 1 or obs.ndim != 2 or obs.shape[1] not in (25, 27):
        raise ValueError("repair supports flat single-ant 25-D/27-D observations")
    ids = np.asarray(episode_ids)
    if (ids.shape != (len(obs),) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or np.any(ids >= len(goals))):
        raise ValueError("invalid cache episode IDs")
    size = np.array([cfg.world.width, cfg.world.height], dtype=np.float32)
    size *= float(cfg.scene_scale)
    local = TShape(cfg).track_local_point(getattr(cfg.env, "goal_track", "center"))
    sin, cos = obs[:, 6], obs[:, 7]
    rotated = np.stack([cos * local[0] - sin * local[1],
                        sin * local[0] + cos * local[1]], axis=1)
    tracked = obs[:, 2:4] * size + rotated
    return (np.asarray(goals, dtype=np.float32)[ids] - tracked) / size


def audit_goals(obs, episode_ids, goals, cfg, *, repair=False, batch_size=100_000):
    """Check every row in bounded memory. Optionally repair in place."""
    if len(obs) == 0:
        raise ValueError("empty replay cache")
    size = np.array([cfg.world.width, cfg.world.height]) * float(cfg.scene_scale)
    total, worst, bad = 0.0, 0.0, 0
    for start in range(0, len(obs), batch_size):
        rows = slice(start, start + batch_size)
        expected = goal_vectors(obs[rows], episode_ids[rows], goals, cfg)
        error = np.linalg.norm((obs[rows, 4:6] - expected) * size, axis=1)
        if not np.isfinite(error).all():
            raise ValueError("nonfinite cached goal vector")
        total += float(error.sum())
        worst = max(worst, float(error.max()))
        bad += int(np.count_nonzero(error > 1e-5))
        if repair:
            obs[rows, 4:6] = expected
    return {"transitions": len(obs), "incorrect_goal_rows": bad,
            "mean_goal_error_m": total / len(obs), "max_goal_error_m": worst}


def require_correct_goals(obs, episode_ids, goals, cfg):
    result = audit_goals(obs, episode_ids, goals, cfg)
    if result["incorrect_goal_rows"]:
        raise ValueError(f"Stale goal vectors in replay cache: {result}. "
                         "Run scripts/il/repair_replay_goals.py with --out first.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/il/il_augmented_bc.yaml")
    parser.add_argument("--dataset", default="storage_local/datasets/successes_v1/dataset.npz")
    parser.add_argument("--cache", default="storage_local/cache/replay_full.npz")
    parser.add_argument("--out", help="write repaired cache here; default audits only")
    args = parser.parse_args()
    if args.out and Path(args.out).resolve() == Path(args.cache).resolve():
        parser.error("--out must differ from --cache to retain original evidence")
    if args.out and Path(args.out).exists():
        parser.error("--out already exists")
    cfg = load_config(args.config)
    with np.load(args.dataset) as dataset:
        goals = dataset["goal"]
    with np.load(args.cache) as cache:
        data = {key: cache[key] for key in cache.files}
    result = audit_goals(data["obs"], data["epid"], goals, cfg, repair=bool(args.out))
    print(json.dumps(result, indent=2), flush=True)
    if args.out:
        require_correct_goals(data["obs"], data["epid"], goals, cfg)
        data["goal_observation_version"] = np.array(2, dtype=np.int32)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # File handle preserves the exact output name (np.savez otherwise adds .npz).
        with out.open("wb") as handle:
            np.savez(handle, **data)
        out.with_suffix(".audit.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"Repaired cache: {out}", flush=True)


if __name__ == "__main__":
    main()
