"""Regenerate the figures used in this blog post.

    python blogs/01_single_agent/make_figures.py

Everything is drawn from the REAL environment: the same config the agent was
trained on, the same collision geometry, and (for the curriculum figure) the
same BFS field the teacher used. So the pictures cannot drift from the code.

Output: blogs/01_single_agent/assets/*.png
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/rl"))
from ant_swarm import load_config                      # noqa: E402
from ant_swarm.layout import Layout                    # noqa: E402
from ant_swarm.tshape import TShape                    # noqa: E402

OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(exist_ok=True)

CFG = load_config(ROOT / "configs/rl/pnas_kin_geo_v2.yaml")
LAY, TSH = Layout(CFG), TShape(CFG)
W, H = LAY.world_size
GOAL = (float(LAY.goal[0]), float(LAY.goal[1]))
HALF = TSH.stem_len / 2
X1A, X1B = LAY.walls_aabb[0][0], LAY.walls_aabb[0][1]
X2A, X2B = LAY.walls_aabb[2][0], LAY.walls_aabb[2][1]


# ---------------------------------------------------------------- helpers
def draw_walls(ax, color="#4a4a52"):
    for a in LAY.walls_aabb:
        ax.add_patch(Rectangle((a[0], a[2]), a[1] - a[0], a[3] - a[2], color=color, zorder=3))
    ax.add_patch(Rectangle((0, 0), W, H, fill=False, ec="#888", lw=1.2, zorder=1))


def draw_load(ax, x, y, theta, color="#c0392b", alpha=1.0):
    """The load, drawn from the env's own rectangles — never hand-drawn."""
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    for r in TSH.rects:
        hx, hy = r.half_size
        corners = r.center + np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
        ax.add_patch(Polygon(corners @ R.T + np.array([x, y]), closed=True,
                             fc=color, ec=color, alpha=alpha, lw=0, zorder=4))


def base(ax):
    draw_walls(ax)
    ax.add_patch(Rectangle((GOAL[0] - 0.012, GOAL[1] - 0.012), 0.024, 0.024,
                           color="#2e8b2e", zorder=5))
    ax.set_aspect("equal")
    ax.axis("off")


def lab(ax, x, y, s, c, size=11, weight="bold", ha="center"):
    ax.text(x, y, s, color=c, fontsize=size, fontweight=weight, ha=ha, va="center", zorder=9)


def arrow(ax, a, b, c, lw=1.5, style="->"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=13,
                                 color=c, lw=lw, zorder=9, shrinkA=3, shrinkB=3))


def save(fig, name):
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", OUT / name)


# ------------------------------------------------------- 1. the vocabulary
def fig_vocabulary():
    """Names every part of the task. No numbers — this is for the reader's words."""
    fig, ax = plt.subplots(figsize=(11.5, 4.9))
    base(ax)
    ax.set_xlim(-0.05, W + 0.05)
    ax.set_ylim(-0.135, H + 0.125)
    mid, gap_lo = (X1B + X2A) / 2, LAY.wall_len

    # room shading
    ax.add_patch(Rectangle((0, 0), X1A, H, color="#f2b705", alpha=0.10, zorder=0))
    ax.add_patch(Rectangle((X1B, 0), X2A - X1B, H, color="#6a1b9a", alpha=0.10, zorder=0))
    ax.add_patch(Rectangle((X2B, 0), W - X2B, H, color="#2e8b2e", alpha=0.10, zorder=0))

    sx, sy = 0.30, 0.36
    draw_load(ax, sx, sy, math.pi)
    draw_load(ax, 0.762, 0.335, math.radians(212), alpha=0.30)

    lab(ax, 0.20, H + 0.062, "start room", "#8a6d00", 11.5)
    lab(ax, mid, H + 0.068, "middle room\n(the corridor)", "#6a1b9a", 10.5)
    lab(ax, 1.33, H + 0.062, "goal room", "#2e7d32", 11.5)

    lab(ax, 0.585, 0.655, "wall", "#333", 11.5)
    arrow(ax, (0.625, 0.655), (X1A - 0.005, 0.60), "#333")
    lab(ax, 1.145, 0.655, "wall", "#333", 11.5)
    arrow(ax, (1.105, 0.655), (X2B + 0.005, 0.60), "#333")

    lab(ax, 0.62, -0.082, "slit", "#1565c0", 11.5)
    arrow(ax, (0.638, -0.062), (X1A - 0.006, gap_lo + 0.02), "#1565c0")
    lab(ax, 1.12, -0.082, "slit", "#1565c0", 11.5)
    arrow(ax, (1.102, -0.062), (X2B + 0.006, gap_lo + 0.02), "#1565c0")

    lab(ax, sx, sy + 0.15, "the load\n(T-shape)", "#c0392b", 11)
    lab(ax, sx + HALF + 0.075, sy - 0.10, "big head", "#b8860b", 11)
    arrow(ax, (sx + HALF + 0.045, sy - 0.075), (sx + HALF, sy - 0.04), "#b8860b")
    lab(ax, sx - HALF - 0.02, sy - 0.135, "small head", "#777", 11)   # below, not left
    arrow(ax, (sx - HALF - 0.02, sy - 0.11), (sx - HALF, sy - 0.03), "#777")
    lab(ax, sx + 0.16, 0.145, "start pose", "#c0392b", 11)
    lab(ax, GOAL[0], GOAL[1] - 0.07, "goal", "#2e7d32", 11.5)

    lab(ax, 1.24, 0.60, "the load must tilt\nto pass through a slit", "#c0392b", 10, "normal")
    arrow(ax, (1.13, 0.575), (0.86, 0.44), "#c0392b")
    save(fig, "task_vocabulary.png")


# -------------------------------------------------------- 2. the dimensions
def fig_dimensions():
    """Same scene, with the measurements that make the task hard."""
    fig, ax = plt.subplots(figsize=(12, 5.2))
    base(ax)
    ax.set_xlim(-0.05, W + 0.05)
    ax.set_ylim(-0.155, H + 0.105)
    sx, sy = 0.30, 0.36
    draw_load(ax, sx, sy, math.pi)
    draw_load(ax, 0.762, 0.335, math.radians(212), alpha=0.28)

    def dim(a, b, c, lw=1.8):
        arrow(ax, a, b, c, lw, style="<->")

    bx = sx + HALF + 0.020
    dim((bx, sy - TSH.cap_big_len / 2), (bx, sy + TSH.cap_big_len / 2), "#b8860b")
    lab(ax, bx + 0.012, sy, "big head\n0.175", "#b8860b", 10.5, ha="left")
    smx = sx - HALF - 0.020
    dim((smx, sy - TSH.cap_small_len / 2), (smx, sy + TSH.cap_small_len / 2), "#777", 1.4)
    lab(ax, smx - 0.012, sy, "small head\n0.087", "#777", 9.5, "normal", ha="right")
    ly = sy - 0.115
    dim((sx - HALF - 0.0135, ly), (sx + HALF + 0.0135, ly), "#555", 1.4)
    lab(ax, sx, ly - 0.035, "load length 0.359", "#555", 10, "normal")
    arrow(ax, (0.175, 0.545), (sx - 0.055, sy + TSH.thickness / 2 + 0.002), "#555", 1.3)
    lab(ax, 0.175, 0.565, "bar thickness 0.027", "#555", 9.5, "normal")

    dim((X1B + 0.030, LAY.wall_len), (X1B + 0.030, H - LAY.wall_len), "#1565c0", 2.2)
    lab(ax, X1B + 0.042, 0.36, "slit\n0.150", "#1565c0", 10.5, ha="left")
    dim((X1A - 0.030, H - LAY.wall_len), (X1A - 0.030, H), "#333", 1.6)
    lab(ax, X1A - 0.042, H - LAY.wall_len / 2, "wall length\n0.285", "#333", 9.5, "normal", ha="right")
    arrow(ax, (1.10, 0.115), (X2B + 0.003, 0.155), "#333", 1.3)
    lab(ax, 1.175, 0.10, "wall thickness 0.010", "#333", 9.5, "normal")
    dim((X1B, H + 0.030), (X2A, H + 0.030), "#6a1b9a", 2)
    lab(ax, (X1B + X2A) / 2, H + 0.052, "corridor 0.211", "#6a1b9a", 10.5)
    dim((0, -0.065), (W, -0.065), "#999", 1.4)
    lab(ax, W / 2, -0.105, "world 1.65 x 0.72", "#999", 9.5, "normal")

    lab(ax, sx, sy + 0.155, "start", "#c0392b", 10.5)
    lab(ax, GOAL[0], GOAL[1] - 0.055, "goal", "#2e7d32", 10.5)
    save(fig, "maze_dimensions.png")


# --------------------------------------------------- 3. curriculum stages
def fig_curriculum(field_path="storage_local/fields/pnas_gap015.npz", stages=16):
    """The curriculum poses, taken from the BFS solution path itself."""
    from ant_swarm import PoseGeodesicField, curriculum_anchors
    f = ROOT / field_path
    if not f.exists():
        print("skip curriculum figure — no field at", f)
        return
    field = PoseGeodesicField(f)
    anchors = curriculum_anchors(field, (0.299, 0.359, -math.pi), stages)

    fig, ax = plt.subplots(figsize=(11, 5.0))
    base(ax)
    ax.set_xlim(-0.02, W + 0.02)
    ax.set_ylim(-0.02, H + 0.02)
    cmap = plt.get_cmap("turbo")
    for i, a in enumerate(anchors):
        draw_load(ax, float(a[0]), float(a[1]), float(a[2]),
                  color=cmap(i / (len(anchors) - 1)), alpha=0.8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, len(anchors) - 1))
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.01,
                 label="curriculum stage   (0 = easiest start, %d = the real start)" % (len(anchors) - 1))
    save(fig, "curriculum_anchors.png")


if __name__ == "__main__":
    fig_vocabulary()
    fig_dimensions()
    fig_curriculum()
