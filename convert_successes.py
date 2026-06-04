"""Convert stored success trajectories from .npz to .json.

Older runs saved successful episodes as compressed ``.npz``; the current
pipeline saves ``.json``. This converts existing files to match.

Usage:
    python convert_successes.py <path> [--delete]

  <path>    a run dir, a successes/ dir, or any parent dir (searched
            recursively for success_*.npz). Default: storage_local
  --delete  remove each .npz after a successful conversion

Examples:
    python convert_successes.py storage_local/ant__20260602_2331__13328496__train_ppo__single
    python convert_successes.py storage_local --delete
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def convert_one(npz_path: Path, delete: bool = False) -> bool:
    try:
        z = np.load(npz_path)
    except Exception as e:
        print(f"  SKIP (unreadable): {npz_path.name}  ({e})")
        return False

    payload = {}
    for k in z.files:
        v = z[k]
        if v.ndim == 0:
            payload[k] = v.item()                       # scalar metadata
        else:
            payload[k] = np.asarray(v, dtype=float).tolist()
    out = npz_path.with_suffix(".json")
    with open(out, "w") as f:
        json.dump(payload, f)
    if delete:
        npz_path.unlink()
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    delete = "--delete" in sys.argv[1:]
    root = Path(args[0]) if args else Path("storage_local")

    files = sorted(root.rglob("success_*.npz")) if root.is_dir() else [root]
    print(f"Found {len(files)} .npz file(s) under {root}  (delete={delete})")
    done = 0
    for i, f in enumerate(files, 1):
        if convert_one(f, delete=delete):
            done += 1
        if i % 2000 == 0:
            print(f"  {i}/{len(files)} converted...", flush=True)
    print(f"Converted {done}/{len(files)} → .json")


if __name__ == "__main__":
    main()
