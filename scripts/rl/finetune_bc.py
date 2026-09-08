"""Fine-tune a corrected-goal BC checkpoint with sparse-reward SAC.

The mode ('residual' or 'gaussian') and every hyper-parameter come from the
config's ``finetune:`` block, so the two arms differ only by config. See
``scripts/rl/bc_finetune.py`` for what each mode does.

    sbatch ops/sb_train.sh scripts/rl/finetune_bc.py configs/rl/ft_residual.yaml
    sbatch ops/sb_train.sh scripts/rl/finetune_bc.py configs/rl/ft_gaussian.yaml

No geodesic field, no curriculum, no new demonstrations.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "il", PROJECT_ROOT / "scripts" / "rl"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
from ant_swarm import AntSwarmEnv, load_config           # noqa: E402
from ant_swarm.compute import resolve_device             # noqa: E402
from ant_swarm.run_id import build_run_id                # noqa: E402
from ant_swarm.tracking import Tracker                  # noqa: E402
from bc_finetune import BCReference, build_finetune_model, distill_into_actor  # noqa: E402
from train_chunked_bc import ChunkPolicy, pick_eval_episodes                   # noqa: E402


def make_env(cfg, seed=0):
    def _init():
        return FlattenObservation(AntSwarmEnv(config=cfg, seed=seed))
    return _init


class SuccessRateCallback(BaseCallback):
    """Log the rolling success rate so the anchor schedule can be judged."""

    def __init__(self, window: int = 100):
        super().__init__()
        self.window, self.hits = window, []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("TimeLimit.truncated") or "episode" in info:
                self.hits.append(float(bool(info.get("is_success", False))))
                self.hits = self.hits[-self.window:]
        if self.hits:
            self.logger.record("rollout/success_rate", float(np.mean(self.hits)))
        return True


class BestEvalCallback(BaseCallback):
    """Evaluate periodically and keep the best policy.

    The first fine-tuning runs returned the LAST policy and so reported a loss
    (94% -> 64%) even though the policy was almost certainly better earlier in
    the run. Keeping the best-by-eval checkpoint makes fine-tuning unable to
    return something worse than it started with.
    """

    def __init__(self, cfg, every: int, episodes: int, seed: int, save_to: Path, tracker=None):
        super().__init__()
        self.tracker = tracker
        self.cfg, self.every, self.episodes = cfg, int(every), int(episodes)
        self.seed, self.save_to = int(seed), Path(save_to)
        self.best = -1.0
        self.history = []

    def _on_step(self) -> bool:
        if self.every <= 0 or self.num_timesteps % self.every != 0:
            return True
        r = evaluate(self.model, self.cfg, self.episodes, self.seed)
        self.history.append({'timesteps': int(self.num_timesteps), **r})
        self.logger.record('eval/success_rate_pct', r['success_rate_pct'])
        self.logger.record('eval/mean_distance_m', r['mean_distance_m'])
        if self.tracker is not None:
            self.tracker.log({'eval/success_rate_pct': r['success_rate_pct'],
                              'eval/mean_distance_m': r['mean_distance_m']}, step=int(self.num_timesteps))
        if r['success_rate_pct'] > self.best:
            self.best = r['success_rate_pct']
            self.model.save(self.save_to)
            logger.info(f"new best at {self.num_timesteps} steps: "
                        f"{r['success_rate_pct']:.1f}% -> {self.save_to}")
        return True


@torch.no_grad()
def evaluate(model, cfg, episodes, seed_base, max_steps=500):
    """Fresh random starts, deterministic policy. Same protocol as the BC eval."""
    env = AntSwarmEnv(config=cfg, seed=seed_base)
    reach = float(cfg.goal.reach_radius) * float(cfg.scene_scale)
    dists, succ = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        info, done, steps = {}, False, 0
        while not done and steps < max_steps:
            action, _ = model.predict(obs.reshape(-1), deterministic=True)
            obs, _, term, trunc, info = env.step(
                np.asarray(action, dtype=np.float32).reshape(env.action_space.shape))
            steps += 1
            done = term or trunc
        d = float(info.get("object_distance", env.state.distance_to_goal()))
        dists.append(d)
        succ.append(bool(info.get("is_success", False) or d < reach))
    env.close()
    return dict(episodes=episodes, successes=int(np.sum(succ)),
                success_rate_pct=float(np.mean(succ) * 100),
                mean_distance_m=float(np.mean(dists)),
                median_distance_m=float(np.median(dists)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rl/ft_residual.yaml")
    p.add_argument("--out", default=None, help="default: storage_local/<run id>")
    p.add_argument("--timesteps", type=int, default=None, help="override finetune.timesteps")
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--eval-seed", type=int, default=30000)
    p.add_argument("--device", default=None,
                   help="override finetune.device from the config")
    p.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    args = p.parse_args()

    cfg = load_config(args.config)
    ft = cfg.finetune
    device = resolve_device(args.device or ft.get("device", "cpu"))
    out = Path(args.out) if args.out else Path("storage_local") / build_run_id(
        f"finetune_bc_{ft.mode}")
    out.mkdir(parents=True, exist_ok=True)
    logger.add(out / "train.log", level="INFO")
    logger.info(f"mode={ft.mode}  device={device}  out={out}")
    tr = Tracker(name=out.name, group="finetune_bc", tags=["finetune", str(ft.mode)],
                 config={**dict(ft), "config": args.config}, enabled=args.wandb)

    if cfg.env.reward_mode != "sparse":
        raise ValueError("fine-tuning must use sparse reward; no geodesic field")
    if cfg.get("curriculum") and cfg.curriculum.get("enabled"):
        raise ValueError("fine-tuning must run without a curriculum")

    env = VecMonitor(DummyVecEnv([make_env(cfg, seed=0)]))
    eval_env_cfg = cfg.copy()

    obs_dim = int(np.prod(env.observation_space.shape))
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    reference = BCReference(ft.bc_checkpoint, obs_dim, low, high, torch.device(device))
    logger.info(f"frozen BC reference loaded from {ft.bc_checkpoint}")

    model = build_finetune_model(cfg, env, reference, tb_dir=out / "tb", device=device)

    if str(ft.mode).lower() == "gaussian" and bool(ft.get("distill_init", True)):
        cache = np.load(ft.distill_cache)
        idx = np.random.default_rng(0).choice(
            len(cache["obs"]), size=min(int(ft.distill_samples), len(cache["obs"])),
            replace=False)
        loss = distill_into_actor(model, reference, cache["obs"][idx],
                                  steps=int(ft.distill_steps))
        logger.info(f"distilled BC into the SAC actor, final MSE={loss:.6f}")

    before = evaluate(model, eval_env_cfg, args.eval_episodes, args.eval_seed)
    logger.info(f"before fine-tuning: {before}")
    tr.log({f"eval/before_{k}": v for k, v in before.items()}, step=0)

    steps = int(args.timesteps if args.timesteps is not None else ft.timesteps)
    best_cb = BestEvalCallback(eval_env_cfg, int(ft.get('eval_every', 20000)),
                               int(ft.get('eval_episodes', 30)),
                               args.eval_seed, out / 'best', tracker=tr)
    t0 = time.time()
    model.learn(total_timesteps=steps,
                callback=[SuccessRateCallback(), best_cb], progress_bar=False)
    model.save(out / "finetuned")
    logger.info(f"trained {steps} steps in {time.time()-t0:.0f}s -> {out/'finetuned.zip'}")

    last = evaluate(model, eval_env_cfg, args.eval_episodes, args.eval_seed)
    logger.info(f"last policy:  {last}")
    after = last
    best_path = out / 'best.zip'
    if best_path.exists():
        best_model = type(model).load(best_path, env=env, reference=reference)
        after = evaluate(best_model, eval_env_cfg, args.eval_episodes, args.eval_seed)
        logger.info(f"best policy:  {after}")

    report = dict(config=args.config, mode=str(ft.mode), timesteps=steps,
                  eval_seed=args.eval_seed, before=before, after=after,
                  last=last, eval_history=best_cb.history,
                  finetune={k: (list(v) if hasattr(v, "__iter__") and not isinstance(v, str) else v)
                            for k, v in dict(ft).items()})
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    tr.summary({f"final/{k}": v for k, v in after.items()}); tr.finish()

    print("\n" + "=" * 66)
    print(f"BC FINE-TUNING — mode={ft.mode}, {steps} steps, "
          f"{args.eval_episodes} fresh starts (seed {args.eval_seed})")
    print("=" * 66)
    print(f"{'stage':<8} {'SR %':>7} {'mean dist':>11} {'median':>10}")
    print("-" * 66)
    for tag, r in (("before", before), ("last", last), ("best", after)):
        print(f"{tag:<8} {r['success_rate_pct']:>7.1f} {r['mean_distance_m']:>11.4f} "
              f"{r['median_distance_m']:>10.4f}")
    print()


if __name__ == "__main__":
    main()
