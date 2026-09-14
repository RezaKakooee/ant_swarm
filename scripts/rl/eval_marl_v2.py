"""Held-out evaluation of MARL v2 actors on fresh seed blocks.

Loads one or more ``.pt`` files saved by ``train_marl_v2.py`` (best.pt,
final.pt, ckpt_*.pt) and scores each on fixed seed blocks that training
never used. Default: seeds 60000-60099 and 70000-70099 (200 episodes).

    python scripts/rl/eval_marl_v2.py storage_local/<run>/final.pt --history 4
    python scripts/rl/eval_marl_v2.py storage_local/<run>/ckpt_*.pt --history 4 --out heldout.json

The config is read from ``results.json`` next to the checkpoint when
``--config`` is not given. The history length is read from there too.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "rl"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
from train_marl_v2 import make_actor, evaluate  # noqa: E402


def _run_args(ckpt: Path) -> dict:
    """Training args from results.json in the checkpoint's folder, if present."""
    rj = ckpt.parent / "results.json"
    if rj.exists():
        return json.loads(rj.read_text()).get("args", {})
    return {}


def score_checkpoint(ckpt: Path, config: str | None, history: int | None, seeds, episodes):
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    ra = _run_args(ckpt)
    config = config or ra.get("config")
    history = history or ra.get("history") or 1
    if config is None:
        raise SystemExit(f"{ckpt}: no --config and no results.json next to it")
    independent = bool(blob.get("independent", False))
    mu_key = "actors.0.mu.weight" if independent else "mu.weight"
    act_dim = int(blob["actor"][mu_key].shape[0])
    obs_dim = int(blob["obs_dim"])
    actor = make_actor(obs_dim * int(history), act_dim, int(blob.get("n", 1)), independent)
    actor.load_state_dict(blob["actor"]); actor.eval()
    row = {"ckpt": str(ckpt), "steps": blob.get("steps"), "n": blob.get("n"), "history": int(history)}
    srs = []
    for s in seeds:
        sr, md = evaluate(actor, config, blob.get("layout"), episodes, int(s), K=int(history))
        row[f"sr_{s}"] = sr; row[f"dist_{s}"] = md; srs.append(sr)
    row["sr_mean"] = sum(srs) / len(srs)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--config", default=None)
    p.add_argument("--history", type=int, default=None)
    p.add_argument("--seeds", default="60000,70000", help="first seed of each block")
    p.add_argument("--episodes", type=int, default=100, help="episodes per block")
    p.add_argument("--out", default=None, help="write all rows as JSON here")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    rows = []
    hdr = "| checkpoint | steps | " + " | ".join(f"seeds {s}+" for s in seeds) + " | mean |"
    print(hdr); print("|" + "---|" * (len(seeds) + 3))
    for c in args.ckpts:
        row = score_checkpoint(Path(c), args.config, args.history, seeds, args.episodes)
        rows.append(row)
        steps = row["steps"]
        steps_s = f"{steps/1e6:.1f}M" if isinstance(steps, (int, float)) else "?"
        cells = " | ".join(f"{row[f'sr_{s}']:.0f}%" for s in seeds)
        print(f"| {Path(c).parent.name}/{Path(c).name} | {steps_s} | {cells} | {row['sr_mean']:.0f}% |", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
