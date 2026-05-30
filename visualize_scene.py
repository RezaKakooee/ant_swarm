"""Standalone 2-D scene visualiser — no MuJoCo / RoboVerse required.

Draws the world boundary, walls, T-shape, goal, and spawn centre with
dimension annotations so you can confirm geometry before running the full
RoboVerse script.

Usage:
    python new_exps/visualize_scene.py                  # saves PNG + shows window
    python new_exps/visualize_scene.py --no-show        # save only
    python new_exps/visualize_scene.py --scene-scale 2  # test scaling
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Re-use geometry from ant_swarm_roboverse without importing metasim.
# We mock the metasim imports so the module loads cleanly.
# ---------------------------------------------------------------------------
import sys
import types
import rootutils
rootutils.setup_root(__file__, pythonpath=True)

for _mod in [
    "metasim", "metasim.constants", "metasim.scenario",
    "metasim.scenario.cameras", "metasim.scenario.objects",
    "metasim.scenario.scenario", "metasim.utils",
    "metasim.utils.obs_utils", "metasim.utils.setup_util",
]:
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

# Provide the symbols actually referenced at module level.
import enum
class _PhysicStateType(enum.Enum):
    RIGIDBODY = "rigidbody"
sys.modules["metasim.constants"].PhysicStateType = _PhysicStateType  # type: ignore
for _attr in ("PinholeCameraCfg", "PrimitiveCubeCfg", "PrimitiveSphereCfg",
              "RigidObjCfg", "ScenarioCfg", "ObsSaver", "get_handler", "configclass"):
    mod_name = {
        "PinholeCameraCfg":     "metasim.scenario.cameras",
        "PrimitiveCubeCfg":     "metasim.scenario.objects",
        "PrimitiveSphereCfg":   "metasim.scenario.objects",
        "RigidObjCfg":          "metasim.scenario.objects",
        "ScenarioCfg":          "metasim.scenario.scenario",
        "ObsSaver":             "metasim.utils.obs_utils",
        "get_handler":          "metasim.utils.setup_util",
        "configclass":          "metasim.utils",
    }[_attr]
    setattr(sys.modules[mod_name], _attr, None)  # type: ignore

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "ant_swarm_roboverse",
    Path(__file__).parent / "ant_swarm_roboverse.py",
)
_mod = _ilu.module_from_spec(_spec)  # type: ignore
_spec.loader.exec_module(_mod)       # type: ignore
MovableTShape = _mod.MovableTShape

# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------

def _tshape_patches(cfg, color="#b04030", alpha=1.0):
    """Return a list of matplotlib patches for the T-shape at init pose."""
    patches = []
    angle = cfg.init_angle
    cx, cy = cfg.init_center
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]])

    for rect in cfg.rects:
        lx, ly = rect.center
        hw, hh = rect.half_size
        # Four corners in local frame
        corners_local = np.array([
            [lx - hw, ly - hh],
            [lx + hw, ly - hh],
            [lx + hw, ly + hh],
            [lx - hw, ly + hh],
        ])
        corners_world = (R @ corners_local.T).T + np.array([cx, cy])
        poly = mpatches.Polygon(corners_world, closed=True,
                                facecolor=color, edgecolor="#5a1810",
                                linewidth=0.8, alpha=alpha)
        patches.append(poly)
    return patches


def _wall_patches(cfg, color="#8888aa"):
    """Return matplotlib patches for all wall segments."""
    patches = []
    ht = cfg.thickness / 2
    hl = cfg.wall_len / 2
    for wx, wy in cfg.wall_positions:
        corners = np.array([
            [wx - ht, wy - hl],
            [wx + ht, wy - hl],
            [wx + ht, wy + hl],
            [wx - ht, wy + hl],
        ])
        poly = mpatches.Polygon(corners, closed=True,
                                facecolor=color, edgecolor="#444466",
                                linewidth=0.8)
        patches.append(poly)
    return patches


def _dim_arrow(ax, x0, y0, x1, y1, label, offset=(0, 0), fontsize=7, color="k"):
    """Draw a double-headed dimension arrow with a centred label."""
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="<->", color=color, lw=0.8),
    )
    mx, my = (x0 + x1) / 2 + offset[0], (y0 + y1) / 2 + offset[1]
    ax.text(mx, my, label, ha="center", va="center", fontsize=fontsize,
            color=color, bbox=dict(fc="white", ec="none", pad=1.0))


# ------------------------------------------------------------------
# Main draw function
# ------------------------------------------------------------------

def draw_scene(cfg, *, save_path: str | None = None, show: bool = True):
    W, H = cfg.world_size
    fig, ax = plt.subplots(figsize=(10 * W, 10 * H))
    ax.set_aspect("equal")
    ax.set_xlim(-0.15, W + 0.15)
    ax.set_ylim(-0.15, H + 0.15)
    ax.set_facecolor("#d8d8d8")

    # World boundary
    world_rect = mpatches.Rectangle((0, 0), W, H,
                                     linewidth=1.5, edgecolor="#222",
                                     facecolor="none", zorder=5)
    ax.add_patch(world_rect)
    ax.text(W / 2, H + 0.07, f"world  {W:.2f} m × {H:.2f} m",
            ha="center", va="bottom", fontsize=8, color="#222")

    # Walls
    for p in _wall_patches(cfg):
        ax.add_patch(p)

    # T-shape
    for p in _tshape_patches(cfg):
        ax.add_patch(p)

    # Goal
    gx, gy = cfg.goal
    ax.plot(gx, gy, marker="s", markersize=8, color="#228822", zorder=6)
    ax.text(gx + 0.02, gy + 0.02, "goal", fontsize=6, color="#228822")

    # Spawn centre
    sx, sy = cfg.spawn_center
    ax.plot(sx, sy, marker="x", markersize=8, color="#555", zorder=6, mew=1.5)
    ax.text(sx + 0.02, sy + 0.02, "spawn", fontsize=6, color="#555")

    # ------------------------------------------------------------------
    # Dimension annotations
    # ------------------------------------------------------------------
    # World width & height
    _dim_arrow(ax, 0, -0.08, W, -0.08, f"{W:.2f} m", fontsize=7)
    _dim_arrow(ax, -0.10, 0, -0.10, H, f"{H:.2f} m", fontsize=7)

    # T-shape: stem length along stem axis
    angle = cfg.init_angle
    c, s_ = math.cos(angle), math.sin(angle)
    cx, cy = cfg.init_center
    stem_end_l = np.array([cx, cy]) + np.array([-c, -s_]) * cfg.stem_len / 2
    stem_end_r = np.array([cx, cy]) + np.array([c, s_]) * cfg.stem_len / 2
    _dim_arrow(ax, *stem_end_l, *stem_end_r,
               f"stem {cfg.stem_len:.2f}", offset=(0.04, 0.04), fontsize=6, color="#802010")

    # T-shape: thickness (perpendicular to stem at centre)
    perp = np.array([-s_, c])
    p0 = np.array([cx, cy]) - perp * cfg.thickness / 2
    p1 = np.array([cx, cy]) + perp * cfg.thickness / 2
    _dim_arrow(ax, *p0, *p1,
               f"{cfg.thickness:.2f}", offset=(0.03, -0.03), fontsize=6, color="#802010")

    # Big cap length
    cap_big_tip = np.array([cx, cy]) + np.array([-c, -s_]) * cfg.stem_len / 2
    cb0 = cap_big_tip + perp * cfg.cap_big_len / 2
    cb1 = cap_big_tip - perp * cfg.cap_big_len / 2
    _dim_arrow(ax, *cb0, *cb1,
               f"big cap {cfg.cap_big_len:.2f}", offset=(0.03, 0.0), fontsize=6, color="#802010")

    # Small cap length
    cap_small_tip = np.array([cx, cy]) + np.array([c, s_]) * cfg.stem_len / 2
    cs0 = cap_small_tip + perp * cfg.cap_small_len / 2
    cs1 = cap_small_tip - perp * cfg.cap_small_len / 2
    _dim_arrow(ax, *cs0, *cs1,
               f"small cap {cfg.cap_small_len:.2f}", offset=(0.03, 0.0), fontsize=6, color="#802010")

    # Wall positions and passage gaps (for left gate only)
    ht = cfg.thickness / 2
    hl = cfg.wall_len / 2
    wx_left = cfg.wall_positions[0][0]
    upper_cy = cfg.wall_positions[0][1]
    lower_cy = cfg.wall_positions[1][1]
    upper_bottom = upper_cy - hl
    lower_top    = lower_cy + hl

    # Wall x position
    _dim_arrow(ax, 0, H + 0.08, wx_left, H + 0.08,
               f"x={wx_left:.2f}", fontsize=6, color="#334")

    # Passage gap
    if lower_top < upper_bottom:
        _dim_arrow(ax, wx_left + 0.05, lower_top, wx_left + 0.05, upper_bottom,
                   f"gap {upper_bottom - lower_top:.2f} m", offset=(0.12, 0), fontsize=6, color="#446")

    # Wall length
    _dim_arrow(ax, wx_left + 0.04, upper_cy - hl, wx_left + 0.04, upper_cy + hl,
               f"wall_len {cfg.wall_len:.2f}", offset=(0.14, 0), fontsize=6, color="#334")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Scene geometry check — no physics, no MuJoCo")
    ax.grid(True, linestyle=":", linewidth=0.4, color="#aaa")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    no_show     = "--no-show" in sys.argv
    scene_scale = 1.0
    for arg in sys.argv[1:]:
        if arg.startswith("--scene-scale="):
            scene_scale = float(arg.split("=")[1])
        elif arg == "--scene-scale" and sys.argv.index(arg) + 1 < len(sys.argv):
            scene_scale = float(sys.argv[sys.argv.index(arg) + 1])

    cfg = MovableTShape(scene_scale=scene_scale)
    out = Path(__file__).parent / "output" / f"scene_check_s{scene_scale:.1f}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    draw_scene(cfg, save_path=str(out), show=not no_show)
