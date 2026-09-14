"""Figures for the addendum of blog post 01 (what happened after the post).

    python blogs/01-can-ai-solve-the-ant-puzzle/make_figures_addendum.py

Drawn from the real env geometry through the helpers of make_figures.py.
Output: blogs/01-can-ai-solve-the-ant-puzzle/ant-piano-movers-rl/assets/*.png
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_figures as M                                   # noqa: E402
from make_figures import W, H, GOAL, HALF, X1A, X1B, X2A, X2B, draw_load, draw_walls, base, lab, arrow, save  # noqa: E402

M.OUT = HERE / "ant-piano-movers-rl" / "assets"; M.OUT.mkdir(exist_ok=True)
INK, MUTED, RED, BLUE, VIOLET, GREEN = "#222", "#777", "#c0392b", "#1565c0", "#6a1b9a", "#2e7d32"


# --------------------------------------------------- A. any start, any goal + two legs
def fig_any_start_any_goal():
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    base(ax); ax.set_xlim(-0.03, W + 0.03); ax.set_ylim(-0.16, H + 0.12)
    # start zone: x in [0.06, 0.52], any angle
    ax.add_patch(Rectangle((0.06, 0.06), 0.46, H - 0.12, fc="#f2b705", alpha=0.15, ec="#b8860b", lw=1.2, ls="--", zorder=1))
    rng = np.random.default_rng(3)
    for _ in range(4):
        draw_load(ax, rng.uniform(0.18, 0.40), rng.uniform(0.22, 0.50), rng.uniform(-math.pi, math.pi), alpha=0.25)
    lab(ax, 0.29, H + 0.05, "start: anywhere here, at any angle", "#8a6d00", 10.5)
    # goal box
    ax.add_patch(Rectangle((1.05, 0.10), 0.50, 0.52, fc=GREEN, alpha=0.12, ec=GREEN, lw=1.2, ls="--", zorder=1))
    for gx, gy in rng.uniform([1.08, 0.13], [1.52, 0.59], size=(7, 2)):
        ax.add_patch(Rectangle((gx - 0.010, gy - 0.010), 0.02, 0.02, color=GREEN, alpha=0.55, zorder=5))
    lab(ax, 1.30, H + 0.05, "goal: anywhere in this box", GREEN, 10.5)
    # the two legs
    ax.add_patch(Circle((1.20, 0.36), 0.014, fc="white", ec=VIOLET, lw=2.2, ls="--", zorder=7))
    lab(ax, 1.20, 0.36 - 0.075, "where the map points\n(never has to be reached)", VIOLET, 9, "normal")
    arrow(ax, (0.40, 0.14), (0.74, 0.30), VIOLET, lw=2.4); arrow(ax, (0.80, 0.40), (0.985, 0.38), VIOLET, lw=2.4)
    lab(ax, 0.50, -0.085, "leg 1: follow the map, until the whole load\nis past the second wall (same map for every goal)", VIOLET, 9.5, "normal")
    ax.add_patch(FancyArrowPatch((1.00, 0.38), (1.46, 0.55), arrowstyle="-|>", mutation_scale=14, color=GREEN, lw=2.4, zorder=7))
    lab(ax, 1.30, -0.085, "leg 2: straight line to the actual goal\n(the room is open, so it is safe)", GREEN, 9.5, "normal")
    ax.axvline(0.989, color=VIOLET, lw=1.2, ls=":", zorder=2); lab(ax, 0.989, -0.02, "hand-over: x = 0.989", VIOLET, 8.5, "normal")
    save(fig, "any_start_any_goal.png")


# --------------------------------------------------- B. the wall at x = 0.900
def fig_frontier():
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    base(ax); ax.set_xlim(-0.03, W + 0.03); ax.set_ylim(-0.19, H + 0.10)
    # the reached region and its edge, both kept INSIDE the maze box
    ax.add_patch(Rectangle((0, 0), 0.900, H, fc=BLUE, alpha=0.10, zorder=0))
    ax.plot([0.900, 0.900], [0, H], color=RED, lw=2.4, zorder=6)
    draw_load(ax, 0.30, 0.36, math.pi, alpha=0.9)
    draw_load(ax, 0.900, 0.36, math.radians(82), alpha=0.55)        # the furthest pose: centre on the line, standing in the corridor
    ax.add_patch(Circle((0.900, 0.36), 0.012, fc="white", ec=RED, lw=2, zorder=8))
    # title, left-aligned above the box, clear of the line
    ax.text(0.0, H + 0.045, "everything ever reached without the map, in six runs and 14.6 million steps", color=BLUE,
            fontsize=10.5, fontweight="bold", ha="left", va="center", zorder=9)
    # the wall label, below the box, starting at the line
    ax.text(0.900, -0.06, "x = 0.900", color=RED, fontsize=11, fontweight="bold", ha="center", va="top", zorder=9)
    ax.text(0.0, -0.125, "the furthest the load's centre ever got (Go-Explore, both grids). HER runs barely left the first room;\n"
                         "gSDE reached the slit. Nothing ever crossed it.", color=RED, fontsize=9.5, ha="left", va="top", zorder=9)
    lab(ax, 1.33, 0.20, "never reached", MUTED, 12, "normal")
    # the second slit: label to the right of the wall, arrow into the gap
    lab(ax, 1.22, 0.60, "the second slit", RED, 9.5, "normal")
    arrow(ax, (1.13, 0.575), (X2B + 0.02, 0.45), RED, lw=1.5)
    save(fig, "frontier_wall.png")


# --------------------------------------------------- B2. the four methods, sketched
def _mini_maze(ax, title):
    base(ax); ax.set_xlim(-0.02, W + 0.02); ax.set_ylim(-0.02, H + 0.02); ax.set_title(title, fontsize=10.5, color=INK)


def fig_methods():
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.2))
    a, b, c, d = axes.ravel()
    # HER: relabel the goal to where the load ended
    _mini_maze(a, "HER: pretend where you ended was the goal")
    draw_load(a, 0.22, 0.50, math.pi, alpha=0.35); draw_load(a, 0.55, 0.42, math.radians(160), alpha=0.9)
    arrow(a, (0.30, 0.44), (0.48, 0.40), BLUE, lw=2)
    bx, by = 0.55 - HALF * math.cos(math.radians(160)), 0.42 - HALF * math.sin(math.radians(160))   # the big head sits at local -stem/2
    a.add_patch(Rectangle((bx - 0.012, by - 0.012), 0.024, 0.024, color=GREEN, zorder=6))
    lab(a, 0.36, 0.14, "new goal = where the big head ended,\nso this episode counts as a success", GREEN, 9, "normal")
    arrow(a, (0.50, 0.21), (bx - 0.005, by - 0.03), GREEN, lw=1.2)
    a.add_patch(Rectangle((GOAL[0] - 0.012, GOAL[1] - 0.012), 0.024, 0.024, fc="none", ec=GREEN, lw=1.5, ls="--", zorder=6))
    lab(a, GOAL[0], GOAL[1] - 0.08, "real goal, never reached", MUTED, 8.5, "normal")
    # intrinsic bonus: reward for new cells
    _mini_maze(b, "Exploration bonus: a reward for new poses")
    rng = np.random.default_rng(1)
    for x, y in rng.uniform([0.05, 0.05], [0.72, 0.67], size=(60, 2)):
        b.add_patch(Rectangle((x, y), 0.03, 0.03, fc=BLUE, alpha=0.25, lw=0, zorder=2))
    b.add_patch(Rectangle((0.80, 0.30), 0.03, 0.03, fc="#f2b705", alpha=0.95, lw=0, zorder=5))
    lab(b, 0.40, 0.66, "seen often: bonus ~ 0", BLUE, 9, "normal"); lab(b, 0.83, 0.20, "new cell:\nbig bonus", "#b8860b", 9, "normal")
    # gSDE: noise held for 64 steps
    _mini_maze(c, "gSDE: keep the same random nudge for 64 steps")
    t = np.linspace(0.08, 0.70, 40); jitter = 0.36 + 0.06 * np.sin(t * 90)
    c.plot(t, jitter, color=MUTED, lw=1.2, zorder=5); lab(c, 0.39, 0.52, "ordinary noise: a new wobble every step", MUTED, 9, "normal")
    c.plot([0.08, 0.35, 0.35, 0.70], [0.22, 0.22, 0.14, 0.14], color=BLUE, lw=2.2, zorder=5); lab(c, 0.39, 0.06, "gSDE: one direction, kept for a while", BLUE, 9, "normal")
    # Go-Explore: archive, return, explore
    _mini_maze(d, "Go-Explore: return to a saved pose, then explore")
    cells = [(0.20, 0.36, math.pi), (0.52, 0.36, math.radians(200)), (0.84, 0.34, math.radians(222))]
    for i, (x, y, th) in enumerate(cells):
        draw_load(d, x, y, th, alpha=0.3 + 0.3 * i)
    arrow(d, (0.22, 0.12), (0.82, 0.12), VIOLET, lw=2)
    lab(d, 0.52, 0.05, "replay the saved actions back to the furthest pose reached", VIOLET, 9, "normal")
    for ang in (50, 90, 130):
        arrow(d, (0.84, 0.50), (0.84 + 0.08 * math.cos(math.radians(ang)), 0.50 + 0.08 * math.sin(math.radians(ang))), RED, lw=1.5)
    lab(d, 0.84, 0.65, "then try random pushes from there", RED, 9, "normal")
    fig.tight_layout()
    save(fig, "four_methods.png")


# --------------------------------------------------- C. the frozen goal
def fig_frozen_goal():
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    draw_walls(ax); ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.03, W + 0.03); ax.set_ylim(-0.14, H + 0.06)
    ax.add_patch(Rectangle((1.05, 0.10), 0.50, 0.52, fc=GREEN, alpha=0.10, ec=GREEN, lw=1.2, ls="--", zorder=1))
    rng = np.random.default_rng(7)
    for gx, gy in rng.uniform([1.07, 0.12], [1.53, 0.60], size=(40, 2)):
        ax.add_patch(Rectangle((gx - 0.008, gy - 0.008), 0.016, 0.016, color=GREEN, alpha=0.5, zorder=4))
    lab(ax, W / 2, H + 0.03, "the goals the demonstrations really had: 34,777 of them, all over the box", GREEN, 10)
    ax.add_patch(Circle((1.5175, 0.5242), 0.02, fc=RED, ec="white", lw=1.5, zorder=8))
    lab(ax, 1.15, 0.66, "the one goal the network was told, every single time", RED, 10)
    arrow(ax, (1.40, 0.635), (1.50, 0.548), RED, lw=1.5)
    draw_load(ax, 0.30, 0.36, math.pi, alpha=0.8)
    lab(ax, W / 2, -0.08, "one line of code held the goal by reference. The reward saw the real goal.\nThe observation kept the first one, for eleven days.", INK, 10, "normal")
    save(fig, "frozen_goal.png")


# --------------------------------------------------- C2. the numbers
def fig_il_numbers():
    fig, ax = plt.subplots(figsize=(9, 3.6))
    names = ["copying, with the bug", "copying, bug fixed", "+ RL, free to change\neverything", "+ RL, small bounded\ncorrection only"]
    vals = [2.5, 91.5, 0, 97]; cols = [RED, BLUE, RED, GREEN]
    bars = ax.bar(names, vals, color=cols, width=0.6)
    for b_, v, t in zip(bars, vals, ["0-5%", "88-95%", "0%\n(peak 87%)", "97%"]):
        ax.text(b_.get_x() + b_.get_width() / 2, max(v, 2) + 3, t, ha="center", fontsize=10, color=INK)
    ax.set_ylim(0, 112); ax.set_ylabel("success on fresh episodes [%]"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Imitation learning, single agent, random start and goal. The policy never sees the map.", fontsize=11, color=INK)
    save(fig, "il_numbers.png")


if __name__ == "__main__":
    fig_any_start_any_goal(); fig_frontier(); fig_methods(); fig_frozen_goal(); fig_il_numbers()
