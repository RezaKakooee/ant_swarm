"""Render a saved success trajectory (.json) to a GIF.

The trajectory's observations encode the T pose each step
(obj_x, obj_y, sin, cos), so we reconstruct the exact frames — no policy or
replay needed. The barrier difficulty (`wall_len`) and, if available, the run's
snapshot config (`<run>/code/config.yaml`) are used so the geometry matches what
the trajectory was generated with.

Usage:
    python render_success.py <success.json> [--index N] [--out file.gif] [--fps N]

  <success.json>  a single success_*.json, OR a combined_*.json (a list) — use
                  --index to pick which trajectory (default 0).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ant_swarm import AntSwarmEnv, load_config  # noqa: E402


def _opt(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv else default


def _cfg_for(json_path: Path):
    """Run's snapshot config if present (so geometry matches the trajectory)."""
    snap_cfg = json_path.resolve().parent.parent / "code" / "config.yaml"
    return load_config(snap_cfg) if snap_cfg.exists() else load_config()


def _setup_env(traj: dict, cfg):
    """Build an env at the trajectory's layout + initial pose."""
    env = AntSwarmEnv(config=cfg, seed=0)
    wall_len = traj.get("wall_len")
    if wall_len is not None and wall_len == wall_len:   # not NaN
        env.set_wall_length(float(wall_len))
    env.reset()
    if "init_pose" in traj:                              # new (replay) format
        cx, cy, ang = traj["init_pose"]
        env.state.obj.set_pose([cx, cy], ang)
        env.state._update_ants()
        env._prev_dist = env.state.distance_to_goal()
    return env


def replay_obs(traj: dict, cfg) -> np.ndarray:
    """Deterministically replay stored actions → array of per-step observations.

    Use this to regenerate BC (obs→action) data from the minimal stored form.
    """
    env = _setup_env(traj, cfg)
    ashape = env.action_space.shape
    obs_seq = [env.obs_model.observe(env.state)]
    for a in traj["actions"]:
        env.step(np.asarray(a, dtype=np.float32).reshape(ashape))
        obs_seq.append(env.obs_model.observe(env.state))
    return np.asarray(obs_seq, dtype=np.float32)


def _frames_for(traj: dict, cfg, img_width: int = 650, stride: int = 1) -> list:
    """Rendered frames for one trajectory.

    New format (`init_pose` + `actions`): deterministically *replay* the actions.
    Legacy format (`obs`): reconstruct pose directly from stored observations.
    `img_width` sets render resolution; `stride` subsamples rendered frames.
    """
    from ant_swarm import Renderer
    env = _setup_env(traj, cfg)
    renderer = Renderer(cfg, env.layout, img_width=img_width)
    W, H = env.layout.world_size
    stride = max(1, stride)

    if "init_pose" in traj:                              # replay actions
        ashape = env.action_space.shape
        frames = [renderer.render(env.state)]
        for k, a in enumerate(traj["actions"]):
            env.step(np.asarray(a, dtype=np.float32).reshape(ashape))
            if (k + 1) % stride == 0:
                frames.append(renderer.render(env.state))
        return frames

    # legacy: reconstruct from stored obs
    frames = []
    for row in np.asarray(traj["obs"], dtype=float)[::stride]:
        x, y = float(row[2]) * W, float(row[3]) * H
        ang = math.atan2(float(row[6]), float(row[7]))
        env.state.obj.set_pose([x, y], ang)
        env.state._update_ants()
        frames.append(renderer.render(env.state))
    return frames


def _save_gif(frames: list, out: Path, fps: int):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    h, w = frames[0].shape[:2]
    fig, ax = plt.subplots(figsize=(6.5, 6.5 * h / w))
    im = ax.imshow(frames[0]); ax.set_axis_off()
    anim = FuncAnimation(fig, lambda i: [im.set_data(frames[i]) or im],
                         frames=len(frames), interval=1000 / fps, blit=True)
    anim.save(str(out), writer=PillowWriter(fps=fps))
    plt.close(fig)


def render_trajectory(json_path: Path, index: int = 0, out: Path | None = None, fps: int = 30) -> Path:
    data = json.load(open(json_path))
    if isinstance(data, list):                     # combined file → pick one
        data = data[index]
    frames = _frames_for(data, _cfg_for(json_path))
    out = Path(out) if out else json_path.with_suffix(".gif")
    _save_gif(frames, out, fps)
    print(f"Saved → {out}  ({len(frames)} frames, len={data.get('length')}, "
          f"wall_len={data.get('wall_len')}, return={data.get('episode_return')})")
    return out


def render_grid(json_path: Path, n: int = 9, out: Path | None = None, fps: int = 30) -> Path:
    """Tile up to `n` trajectories from a combined file into one GIF (looped)."""
    data = json.load(open(json_path))
    if not isinstance(data, list):
        data = [data]
    trajs = data[:n]
    cfg = _cfg_for(json_path)
    # low-res + time-subsampled so a grid of long clips fits in memory
    tile_w = 240
    clips = []
    for t in trajs:
        T_full = len(t["obs"])
        stride = max(1, T_full // 150)        # cap each clip at ~150 frames
        clips.append(_frames_for(t, cfg, img_width=tile_w, stride=stride))
    cols = int(math.ceil(math.sqrt(len(clips))))
    rows = int(math.ceil(len(clips) / cols))
    fh, fw = clips[0][0].shape[:2]
    T = max(len(c) for c in clips)

    tiled = []
    for t in range(T):
        canvas = np.full((rows * fh, cols * fw, 3), 255, dtype=np.uint8)
        for k, clip in enumerate(clips):
            frame = clip[min(t, len(clip) - 1)]              # hold last frame when done
            r, c = divmod(k, cols)
            canvas[r*fh:(r+1)*fh, c*fw:(c+1)*fw] = frame
        tiled.append(canvas)

    out = Path(out) if out else json_path.with_name(json_path.stem + "_grid.gif")
    _save_gif(tiled, out, fps)
    print(f"Saved grid → {out}  ({len(clips)} trajectories, {rows}x{cols}, {T} frames)")
    return out


def main():
    argv = sys.argv[1:]
    pos = [a for a in argv if not a.startswith("--")]
    if not pos:
        raise SystemExit("usage: python render_success.py <success.json> "
                         "[--index N | --grid [N]] [--out f.gif] [--fps N]")
    path = Path(pos[0])
    out = _opt(argv, "--out", None)
    fps = int(_opt(argv, "--fps", 30))
    if "--grid" in argv:
        gi = argv.index("--grid")
        n = 9
        if gi + 1 < len(argv) and not argv[gi + 1].startswith("--"):
            try:
                n = int(argv[gi + 1])
            except ValueError:
                pass
        render_grid(path, n=n, out=out, fps=fps)
    else:
        render_trajectory(path, index=int(_opt(argv, "--index", 0)), out=out, fps=fps)


if __name__ == "__main__":
    main()
