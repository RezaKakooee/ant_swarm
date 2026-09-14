"""Figures for blog post 02 (the swarm). Drawn from the real env geometry.

    python blogs/02-can-a-swarm-solve-the-ant-puzzle/make_figures.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle, Circle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ant_swarm import load_config          # noqa: E402
from ant_swarm.tshape import TShape        # noqa: E402

OUT = Path(__file__).resolve().parent / "ant-swarm-rl" / "assets"; OUT.mkdir(exist_ok=True)
TSH = TShape(load_config(ROOT / "configs/rl/marl_v2_5ants.yaml"))
RED, INK, MUTED = "#c0392b", "#222", "#777"


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", pad_inches=0.05); plt.close(fig); print("saved", name)


def draw_T(ax, alpha=1.0):
    for r in TSH.rects:
        hx, hy = r.half_size
        c = r.center + np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
        ax.add_patch(Polygon(c, closed=True, fc=RED, ec=RED, alpha=alpha, lw=0, zorder=2))


def fig_layouts():
    """Where the ants hold the load: 1, 5 and 10 ants."""
    five = load_config(ROOT / "configs/rl/marl_v2_5ants.yaml").ants.offsets
    ten = load_config(ROOT / "configs/rl/marl_v2_10ants.yaml").ants.offsets
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, (title, pts) in zip(axes, [("one ant (it may also spin the load)", [[0, 0]]), ("five ants", five), ("ten ants", ten)]):
        draw_T(ax)
        for i, (x, y) in enumerate(pts):
            ax.add_patch(Circle((x, y), 0.011, fc="white", ec=INK, lw=1.2, zorder=4))
            ax.text(x, y, str(i + 1), ha="center", va="center", fontsize=6.5, zorder=5)
        ax.set_xlim(-0.24, 0.24); ax.set_ylim(-0.13, 0.13); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=11, color=INK)
    axes[1].text(0, -0.12, "big head on the left, small head on the right", ha="center", fontsize=9, color=MUTED)
    save(fig, "ant_layouts.png")


def fig_split():
    """The split formula, drawn: one push + one turn -> N small pushes.
    Uses the real 5-ant layout and the closed form of chapter 04 §2:
        f_i = F/n + lambda * perp(a_i - abar),  lambda = (T - abar x F) / sum |a_j - abar|^2
    """
    from matplotlib.patches import FancyArrowPatch
    pts = np.array(load_config(ROOT / "configs/rl/marl_v2_5ants.yaml").ants.offsets, float)
    n = len(pts); F = np.array([0.0, 0.20]); T = 0.045           # a push "up" plus an anti-clockwise turn (example)
    abar = pts.mean(0); ac = pts - abar
    lam = (T - (abar[0] * F[1] - abar[1] * F[0])) / (ac ** 2).sum()
    perp = np.stack([-ac[:, 1], ac[:, 0]], 1)                   # rotate arm by +90 deg
    f_push = np.tile(F / n, (n, 1)); f_turn = lam * perp; f = f_push + f_turn
    K = 1.0                                                      # arrow scale: force -> metres on the drawing (same in every panel)

    def arrow(ax, p0, v, color, lw=2.6, ms=16, z=6):
        if np.linalg.norm(v) < 1e-6: return
        ax.add_patch(FancyArrowPatch(p0, p0 + K * v, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=z))

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0))
    titles = ["1. what the single agent wants", "2. equal share of the push", "3. sideways shares make the turn", "4. each ant: 2 + 3"]
    for ax, t in zip(axes, titles):
        draw_T(ax, alpha=0.9)
        for i, (x, y) in enumerate(pts):
            ax.add_patch(Circle((x, y), 0.011, fc="white", ec=INK, lw=1.2, zorder=4))
        ax.set_xlim(-0.27, 0.27); ax.set_ylim(-0.20, 0.25); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(t, fontsize=10.5, color=INK)
    # 1: total push at the centre + a curved turn arrow
    arrow(axes[0], np.zeros(2), F, "#1565c0", lw=3, ms=18)
    axes[0].text(0.03, 0.13, "push F", color="#1565c0", fontsize=10, fontweight="bold")
    # anticlockwise (T > 0): below the centre the motion goes from LEFT to RIGHT
    axes[0].add_patch(FancyArrowPatch((-0.09, -0.07), (0.09, -0.07), connectionstyle="arc3,rad=0.7", arrowstyle="-|>",
                                      mutation_scale=14, color="#6a1b9a", lw=2.2, zorder=6))
    axes[0].text(0.0, -0.165, "turn T (anticlockwise)", color="#6a1b9a", fontsize=10, fontweight="bold", ha="center")
    # 2: F/n at every ant
    for i in range(n): arrow(axes[1], pts[i], f_push[i], "#1565c0")
    axes[1].text(0, -0.165, "every ant: F / 5, same direction", color="#1565c0", fontsize=9.5, ha="center")
    # 3: sideways shares, larger far from the centre, opposite on opposite sides
    for i in range(n): arrow(axes[2], pts[i], f_turn[i], "#6a1b9a")
    axes[2].text(0, -0.165, "far from centre: big; opposite sides: opposite\nthese add up to zero push, but to the turn", color="#6a1b9a", fontsize=8.8, ha="center")
    # 4: the sum
    for i in range(n): arrow(axes[3], pts[i], f[i], "#c0392b", lw=2.8)
    axes[3].text(0, -0.165, "the push each ant is told to make", color="#c0392b", fontsize=9.5, ha="center")
    fig.text(0.5, 0.10, "all arrows on the same scale", ha="center", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(wspace=0.05)
    save(fig, "split_formula.png")


def fig_cancel():
    """The two-ant example of §5: 5% error per ant -> a push nobody asked for."""
    from matplotlib.patches import FancyArrowPatch
    A = np.array([-0.166, 0.0]); B = np.array([0.166, 0.0])       # the two heads of the T
    K = 0.16                                                      # arrow scale (per unit push)
    E = 6.0                                                       # the 0.05 errors are drawn 6x larger, so they can be seen
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    def arrow(ax, p0, v, color, lw=2.6, ms=16, z=6, ls="-"):
        if np.linalg.norm(v) < 1e-9: return
        ax.add_patch(FancyArrowPatch(p0, p0 + K * np.asarray(v), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=z, linestyle=ls))

    def panel(ax, title, fa, fb, note, note_color):
        draw_T(ax, alpha=0.9)
        for q in (A, B): ax.add_patch(Circle(q, 0.011, fc="white", ec=INK, lw=1.2, zorder=4))
        ax.text(A[0] - 0.03, 0.02, "A", fontsize=11, fontweight="bold", color=INK, ha="right")
        ax.text(B[0] + 0.03, 0.02, "B", fontsize=11, fontweight="bold", color=INK, ha="left")
        arrow(ax, A, fa, "#1565c0"); arrow(ax, B, fb, "#1565c0")
        ax.set_xlim(-0.30, 0.30); ax.set_ylim(-0.25, 0.25); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=11, color=INK)
        ax.text(0, -0.225, note, fontsize=9.5, color=note_color, ha="center", va="center", linespacing=1.4)

    # 1: what the formula says
    # anticlockwise turn: the LEFT end goes down and the RIGHT end goes up (same sense as the split figure)
    panel(axes[0], "1. what the formula says", [0, -1.0], [0, 1.0],
          "A: 1.00 down.  B: 1.00 up.\nnet push 0, a clean anticlockwise turn", "#1565c0")
    # 2: each ant 5% off, both leaning left
    panel(axes[1], "2. each ant 5% off", [-0.05, -1.0], [-0.05, 1.0],
          "A: 1.00 down + 0.05 left.  B: 1.00 up + 0.05 left.\n(dashed: what was asked. errors drawn 6x larger)", "#c0392b")
    arrow(axes[1], A, [0, -1.0], "#1565c0", lw=1.2, ms=10, z=5, ls="--"); arrow(axes[1], B, [0, 1.0], "#1565c0", lw=1.2, ms=10, z=5, ls="--")
    arrow(axes[1], A + K * np.array([0, -1.0]), [-0.05 * E, 0], "#c0392b", lw=3.0, ms=16, z=7)
    arrow(axes[1], B + K * np.array([0, 1.0]), [-0.05 * E, 0], "#c0392b", lw=3.0, ms=16, z=7)
    axes[1].text(-0.21, -0.19, "error", color="#c0392b", fontsize=10, fontweight="bold", ha="center")
    axes[1].text(0.10, 0.19, "error", color="#c0392b", fontsize=10, fontweight="bold", ha="center")
    # 3: what the load gets: the turn (fine) + a slide of 0.10 nobody asked for
    ax = axes[2]; draw_T(ax, alpha=0.9)
    for q in (A, B): ax.add_patch(Circle(q, 0.011, fc="white", ec=INK, lw=1.2, zorder=4))
    ax.add_patch(FancyArrowPatch((-0.08, -0.05), (0.08, -0.05), connectionstyle="arc3,rad=0.6", arrowstyle="-|>",
                                 mutation_scale=14, color="#6a1b9a", lw=2.2, zorder=6))
    ax.text(0, -0.155, "turn: correct", color="#6a1b9a", fontsize=10, fontweight="bold", ha="center")
    ax.add_patch(FancyArrowPatch((0.06, 0.11), (0.06 - 0.10 * K * E, 0.11), arrowstyle="-|>", mutation_scale=20, color="#c0392b", lw=3.2, zorder=7))
    ax.text(0.0, 0.18, "slide left 0.10: nobody asked for it", color="#c0392b", fontsize=10, fontweight="bold", ha="center")
    ax.set_xlim(-0.30, 0.30); ax.set_ylim(-0.25, 0.25); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("3. what the load gets", fontsize=11, color=INK)
    ax.text(0, -0.225, "the 1.00 pushes cancel, the 0.05 errors add.\nin 500 steps that slide is about one metre", fontsize=9.5, color=INK, ha="center", va="center", linespacing=1.4)
    fig.subplots_adjust(wspace=0.05)
    save(fig, "cancel_example.png")


def fig_wrench_errors():
    """Chapter 04 §4: per-ant error vs summed torque error."""
    n = [2, 4, 10, 50, 100]; per = [4.6, 3.2, 1.6, 0.9, 1.3]; force = [13.7, 16.0, 19.0, 48.4, 136.0]; torque = [90.2, 77.2, 76.5, 136.9, 356.7]
    fig, ax = plt.subplots(figsize=(8, 3.8)); x = np.arange(len(n)); w = 0.27
    ax.bar(x - w, per, w, color="#2e7d32", label="each ant's own push error")
    ax.bar(x, force, w, color="#f2b705", label="error of the total push")
    ax.bar(x + w, torque, w, color=RED, label="error of the total turn (torque)")
    for i, v in enumerate(torque): ax.text(x[i] + w, v + 6, f"{v:.0f}%", ha="center", fontsize=8.5, color=RED)
    for i, v in enumerate(per): ax.text(x[i] - w, v + 6, f"{v:.1f}%", ha="center", fontsize=8.5, color="#2e7d32")
    ax.set_xticks(x); ax.set_xticklabels([f"{k} ants" for k in n]); ax.set_ylabel("error [%]"); ax.set_ylim(0, 400)
    ax.legend(frameon=False, fontsize=9, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Each ant is nearly right. Their sum is very wrong.", fontsize=11, color=INK)
    save(fig, "wrench_errors.png")


def _hist(run):
    d = json.loads((ROOT / run / "results.json").read_text()); h = d["history"]
    return np.array([x["steps"] for x in h]) / 1e6, np.array([x["sr"] for x in h])


def _smooth(y, k=5):
    r = k // 2; return np.array([np.median(y[max(0, i - r):i + r + 1]) for i in range(len(y))])


def fig_curves(name, runs, title, xlabel="million steps"):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for lab, run, c in runs:
        s, y = _hist(run); ax.plot(s, y, ".", alpha=0.25, color=c); ax.plot(s, _smooth(y), lw=2, color=c, label=lab)
    ax.set_xlabel(xlabel); ax.set_ylabel("success on 30 test episodes [%]"); ax.set_ylim(-3, 103); ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc="lower right"); ax.spines[["top", "right"]].set_visible(False); ax.set_title(title, fontsize=11, color=INK)
    save(fig, name)


def fig_samples():
    """§12: the same curves on a 'pushes seen by the network' axis."""
    runs = [("one ant", "storage_local/ant__20260909_2334__242212__marl_v2_h4", "#1f77b4", 1),
            ("one ant, five copies of the maze", "storage_local/ant__20260910_1013__242283__marl_v2_h4_x5", "#2ca02c", 5),
            ("five ants, one shared brain", "storage_local/ant__20260909_2337__242213__marl_v2_h4", RED, 5)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for lab, run, c, mult in runs:
        s, y = _hist(run)
        axes[0].plot(s, _smooth(y), lw=2, color=c, label=lab); axes[0].plot(s, y, ".", alpha=0.2, color=c)
        axes[1].plot(s * mult, _smooth(y), lw=2, color=c, label=lab); axes[1].plot(s * mult, y, ".", alpha=0.2, color=c)
    axes[0].set_xlabel("million steps of the maze"); axes[1].set_xlabel("million pushes the network learned from")
    axes[0].set_ylabel("success on 30 test episodes [%]"); axes[1].legend(frameon=False, fontsize=9, loc="lower right")
    for ax in axes: ax.set_ylim(-3, 103); ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("by time", fontsize=11, color=INK); axes[1].set_title("by data", fontsize=11, color=INK)
    save(fig, "swarm_vs_single_samples.png")


if __name__ == "__main__":
    fig_layouts(); fig_split(); fig_cancel(); fig_wrench_errors(); fig_samples()
    fig_curves("shared_vs_independent.png",
               [("5 ants, one shared brain", "storage_local/ant__20260909_2337__242213__marl_v2_h4", "#1f77b4"),
                ("5 ants, one brain each", "storage_local/ant__20260911_1512__242601__marl_v2_h4_ind", "#2ca02c"),
                ("10 ants, one shared brain", "storage_local/ant__20260911_0016__242469__marl_v2_h4_10ants", RED),
                ("10 ants, one brain each", "storage_local/ant__20260911_1512__242603__marl_v2_h4_10ants_ind", "#9467bd")],
               "Sharing one brain is faster at first. Separate brains end just as high, and do not fall back.")
    fig_curves("no_map.png",
               [("5 ants, with the BFS teacher", "storage_local/ant__20260911_1512__242601__marl_v2_h4_ind", "#2ca02c"),
                ("5 ants, straight-line reward, PPO", "storage_local/ant__20260912_0111__242707__marl_v2_h4_ind_nomap", RED),
                ("5 ants, straight-line reward, SAC", "storage_local/ant__20260912_0120__242710__masac_h4_ind_nomap", "#ff7f0e")],
               "Without the teacher, nothing happens in 20 million steps.")
