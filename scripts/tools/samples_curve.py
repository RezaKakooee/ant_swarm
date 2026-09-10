"""Chapter 04 §12, experiment 2: success versus actor rows seen.

Reads the 30-episode evaluation curves from each run's ``results.json`` and
plots success against actor rows. A run with n ants and E envs per worker has
seen n*E*steps actor rows at step count ``steps``. So the 5-ant swarm at step S
is compared with the single ant at step 5*S.

    python scripts/tools/samples_curve.py \\
        --run single=storage_local/ant__20260909_2334__242212__marl_v2_h4 \\
        --run swarm=storage_local/ant__20260909_2337__242213__marl_v2_h4 \\
        --out docs/project_journey/figures/ch04_s12_samples_curve.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(run_dir: Path):
    d = json.loads((run_dir / "results.json").read_text())
    a = d["args"]; E = int(a.get("envs_per_worker", 1))
    h = d["history"]; n = int(h[0]["stage_n"]) if h else 1
    steps = np.array([x["steps"] for x in h], float)
    sr = np.array([x["sr"] for x in h], float)
    return dict(steps=steps, sr=sr, rows=steps * n * E, n=n, E=E)


def smooth(y, k=3):
    """Centred moving median over k evaluations (k odd)."""
    out = np.empty_like(y); r = k // 2
    for i in range(len(y)):
        out[i] = np.median(y[max(0, i - r):i + r + 1])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, help="label=run_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--rows", default="5,10,15,20,25,50,75,100", help="actor-row marks (millions) for the table")
    p.add_argument("--window", type=int, default=5, help="evaluations averaged around each mark")
    args = p.parse_args()

    runs = {}
    for spec in args.run:
        lab, path = spec.split("=", 1); runs[lab] = load(Path(path))

    # ---- table: mean of the `window` evaluations nearest each row mark
    marks = [float(x) * 1e6 for x in args.rows.split(",")]
    print("| actor rows | " + " | ".join(f"{lab} (steps)" for lab in runs) + " |")
    print("|" + "---|" * (len(runs) + 1))
    for m in marks:
        cells = []
        for lab, r in runs.items():
            if m > r["rows"].max() * 1.02:
                cells.append("—"); continue
            i = np.argsort(np.abs(r["rows"] - m))[:args.window]
            cells.append(f"{r['sr'][i].mean():.0f}% ({r['steps'][i].mean()/1e6:.1f}M)")
        print(f"| {m/1e6:.0f}M | " + " | ".join(cells) + " |")

    # ---- plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    palette = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    for ax, key, xl in ((axes[0], "steps", "worker steps = PPO updates x 15,360"), (axes[1], "rows", "actor rows seen")):
        for (lab, r), c in zip(runs.items(), palette):
            ax.plot(r[key] / 1e6, r["sr"], ".", alpha=0.35, color=c)
            ax.plot(r[key] / 1e6, smooth(r["sr"], 5), "-", lw=2, color=c, label=f"{lab} (n={r['n']}, E={r['E']})")
        ax.set_xlabel(xl + " [millions]"); ax.grid(alpha=0.3); ax.set_ylim(-3, 103)
    axes[0].set_ylabel("30-episode success [%]"); axes[1].legend(loc="lower right")
    axes[0].set_title("same x-axis as W&B (equal update count)"); axes[1].set_title("x = n_ants x E x steps (equal row count)")
    fig.tight_layout(); Path(args.out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
