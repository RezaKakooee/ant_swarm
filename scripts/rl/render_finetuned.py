"""Render rollouts of a fine-tuned SAC checkpoint on fresh random starts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT, ROOT / "scripts" / "il", ROOT / "scripts" / "rl"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
from ant_swarm import AntSwarmEnv, load_config       # noqa: E402
from ant_swarm.compute import resolve_device        # noqa: E402
from bc_finetune import AnchoredSAC, BCReference    # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rl/ft_residual.yaml")
    p.add_argument("--checkpoint",
                   default="storage_local/ant__20260906_0950__240957__finetune_bc_residual/best.zip")
    p.add_argument("--out", default=None)
    p.add_argument("--episodes", type=int, default=60, help="episodes to scan")
    p.add_argument("--successes", type=int, default=3)
    p.add_argument("--failures", type=int, default=3)
    p.add_argument("--seed", type=int, default=30000)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    from PIL import Image
    from gymnasium.wrappers import FlattenObservation
    from stable_baselines3.common.vec_env import DummyVecEnv

    device = resolve_device(args.device)
    cfg = load_config(args.config)
    ft = cfg.finetune
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "renders"
    out.mkdir(parents=True, exist_ok=True)

    probe = FlattenObservation(AntSwarmEnv(config=cfg, seed=0))
    venv = DummyVecEnv([lambda: FlattenObservation(AntSwarmEnv(config=cfg, seed=0))])
    obs_dim = int(np.prod(probe.observation_space.shape))
    low = np.asarray(probe.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(probe.action_space.high, dtype=np.float32).reshape(-1)
    probe.close()
    reference = BCReference(ft.bc_checkpoint, obs_dim, low, high, torch.device(device))
    model = AnchoredSAC.load(args.checkpoint, env=venv, reference=reference, device=device)
    logger.info(f"loaded {args.checkpoint} on {device}")

    env = AntSwarmEnv(config=cfg, seed=args.seed)
    reach = float(cfg.goal.reach_radius) * float(cfg.scene_scale)
    got = {"success": 0, "failure": 0}
    want = {"success": args.successes, "failure": args.failures}

    for ep in range(args.episodes):
        if got["success"] >= want["success"] and got["failure"] >= want["failure"]:
            break
        obs, _ = env.reset(seed=args.seed + ep)
        frames, info, done, steps = [env.render()], {}, False, 0
        while not done and steps < args.max_steps:
            a, _ = model.predict(obs.reshape(-1), deterministic=True)
            obs, _, term, trunc, info = env.step(
                np.asarray(a, dtype=np.float32).reshape(env.action_space.shape))
            steps += 1
            done = term or trunc
            if steps % 2 == 0 or done:
                frames.append(env.render())
        d = float(info.get("object_distance", env.state.distance_to_goal()))
        ok = bool(info.get("is_success", False) or d < reach)
        tag = "success" if ok else "failure"
        if got[tag] >= want[tag]:
            continue
        got[tag] += 1
        name = out / f"{tag}_{got[tag]}_ep{ep}.gif"
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(name, save_all=True, append_images=imgs[1:],
                     loop=0, duration=80, optimize=True)
        logger.info(f"{name.name}: dist={d:.4f} steps={steps}")
    env.close()
    venv.close()
    logger.info(f"wrote {sum(got.values())} gifs -> {out}")


if __name__ == "__main__":
    main()
