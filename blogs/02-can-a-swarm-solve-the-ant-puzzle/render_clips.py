"""Render swarm rollouts (MARL v2 / MASAC checkpoints) to GIF and MP4.

    python blogs/02-can-a-swarm-solve-the-ant-puzzle/render_clips.py

Picks fresh seeds (80000+) that training and the held-out tables never used.
Writes assets/<name>.gif and .mp4.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/rl"))
from ant_swarm import AntSwarmEnv, load_config          # noqa: E402
import train_marl_v2 as M                               # noqa: E402

OUT = Path(__file__).resolve().parent / "ant-swarm-rl" / "assets"; OUT.mkdir(exist_ok=True)

CLIPS = [
    # name, run dir, config, want success?
    ("swarm5_success", "storage_local/ant__20260911_1512__242601__marl_v2_h4_ind", "configs/rl/marl_v2_5ants.yaml", True),
    ("swarm10_success", "storage_local/ant__20260911_1512__242603__marl_v2_h4_10ants_ind", "configs/rl/marl_v2_10ants.yaml", True),
    ("swarm5_nomap_stuck", "storage_local/ant__20260912_0111__242707__marl_v2_h4_ind_nomap", "configs/rl/marl_v2_5ants_nomap.yaml", False),
    ("single_success", "storage_local/ant__20260910_1013__242283__marl_v2_h4_x5", "configs/rl/marl_v2_1ant.yaml", True),
]


def load_actor(run, K=4):
    blob = torch.load(Path(run) / "final.pt", map_location="cpu", weights_only=True)
    ind = bool(blob.get("independent", False))
    act_dim = int(blob["actor"]["actors.0.mu.weight" if ind else "mu.weight"].shape[0])
    actor = M.make_actor(int(blob["obs_dim"]) * K, act_dim, int(blob["n"]), ind)
    actor.load_state_dict(blob["actor"]); actor.eval()
    return actor, blob.get("layout")


def rollout(actor, cfg_path, layout, seed, K=4, max_steps=500):
    cfg = load_config(cfg_path).copy()
    if layout is not None:
        cfg.ants.n = len(layout); cfg.ants.offsets = [list(map(float, p)) for p in layout]
    env = AntSwarmEnv(config=cfg, seed=seed); env.reset(seed=seed)
    ring = M.Ring(int(cfg.ants.n), int(env.obs_model.obs_dim), K)
    frames, done, k, info = [env.render()], False, 0, {}
    with torch.no_grad():
        while not done and k < max_steps:
            obs = env.obs_model.observe(env.state)
            a, _, _ = actor.act(torch.as_tensor(ring.push(obs), dtype=torch.float32), deterministic=True)
            _, _, tm, tr, info = env.step(M.to_env_action(a.numpy())); k += 1; done = tm or tr
            if k % 2 == 0 or done:
                frames.append(env.render())
    ok = bool(info.get("is_success", False)); env.close()
    return frames, ok, k


def save(frames, name, fps=12):
    imgs = [Image.fromarray(f) for f in frames]
    gif = OUT / f"{name}.gif"
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], loop=0, duration=int(1000 / fps), optimize=True)
    mp4 = OUT / f"{name}.mp4"
    h, w = frames[0].shape[:2]; w2, h2 = w - w % 2, h - h % 2
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                          "-vf", f"crop={w2}:{h2}:0:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(mp4)], stdin=subprocess.PIPE)
    for f in frames: p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close(); p.wait()
    print(f"wrote {gif.name} ({len(frames)} frames) and {mp4.name}")


if __name__ == "__main__":
    for name, run, cfg, want in CLIPS:
        actor, layout = load_actor(run)
        for seed in range(80000, 80040):
            frames, ok, k = rollout(actor, cfg, layout, seed)
            if ok == want and (want or k >= 500):
                print(f"{name}: seed {seed} success={ok} steps={k}")
                save(frames, name); break
        else:
            print(f"{name}: no matching episode found")
