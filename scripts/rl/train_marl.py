"""Multi-agent RL: one shared policy, each ant sees only its own observation.

Parameter-sharing PPO (the MAPPO recipe without a centralised critic input).
Every ant runs the SAME actor on ITS OWN 27-D row and outputs its own
``[push_angle, force]``. There is no communication, no global state and no
ant-count input -- the ant must read the swarm size from the load's response,
which a probe showed is possible (R2 0.86 from one frame, 0.94 from four).

Why RL rather than more imitation
---------------------------------
Three supervised attempts failed. Each ant copies the reference controller to
1-5%, but the total torque is a near-cancelling sum, so the summed error reached
312% at n=100. Regression has to reproduce an exact formula; RL only has to find
SOME split that moves the load, and the reward measures the load, not per-ant
accuracy. That is a much weaker requirement.

Team reward: every ant receives the same env reward. The critic is per-ant on
the same row, so the whole thing stays decentralised at execution.

    sbatch ... --wrap "python scripts/rl/train_marl.py --ants 4 --timesteps 2000000"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "il", PROJECT_ROOT / "scripts" / "rl"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
from ant_swarm import AntSwarmEnv, load_config       # noqa: E402
from ant_swarm.compute import resolve_device        # noqa: E402
from ant_swarm.run_id import build_run_id           # noqa: E402
from ant_swarm.tracking import Tracker             # noqa: E402
from pose_curriculum import build_pose_path_curriculum  # noqa: E402

OBS_DIM, ACT_DIM = 27, 2


class SharedActorCritic(nn.Module):
    """One network, applied per ant. Input is that ant's row only."""

    def __init__(self, obs_dim=OBS_DIM, hidden=256, init_log_std=-1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, ACT_DIM)
        self.v = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), float(init_log_std)))

    def forward(self, obs):
        h = self.body(obs)
        return self.mu(h), self.log_std.expand_as(self.mu(h)), self.v(h).squeeze(-1)

    def act(self, obs, deterministic=False):
        mu, log_std, value = self(obs)
        if deterministic:
            return torch.tanh(mu), None, value
        dist = torch.distributions.Normal(mu, log_std.exp())
        raw = dist.rsample()
        logp = dist.log_prob(raw).sum(-1)
        act = torch.tanh(raw)
        # tanh change of variables
        logp = logp - torch.log1p(-act.pow(2) + 1e-6).sum(-1)
        return act, logp, value

    def evaluate(self, obs, act):
        mu, log_std, value = self(obs)
        raw = torch.atanh(act.clamp(-0.999999, 0.999999))
        dist = torch.distributions.Normal(mu, log_std.exp())
        logp = dist.log_prob(raw).sum(-1) - torch.log1p(-act.pow(2) + 1e-6).sum(-1)
        return logp, dist.entropy().sum(-1), value


def to_env_action(act01: np.ndarray) -> np.ndarray:
    """tanh output in [-1,1]^2 -> [push_angle in +-pi, force in 0..1]."""
    out = np.empty_like(act01)
    out[:, 0] = act01[:, 0] * np.pi
    out[:, 1] = (act01[:, 1] + 1.0) * 0.5
    return out.astype(np.float32)


class PosePathDriver:
    """Drive the pose_path curriculum from a plain loop (no SB3 callback plumbing).

    Same rules as PosePathCurriculumCallback: a stage advances only when the
    rolling success over `window` episodes that STARTED at that stage reaches
    `threshold`; after the last anchor is held for `stop_window` episodes at
    `stop_success`, the spawn is released to `final_spawn_x_range`.
    """

    def __init__(self, cfg, env):
        cb = build_pose_path_curriculum(cfg, cfg.curriculum, float(env.layout.reach_radius))
        self.anchors, self.threshold, self.window = cb.anchors, cb.threshold, cb.window
        self.stop_success, self.stop_window = cb.stop_success, cb._target_success.maxlen
        self.final_range = cb.final_spawn_x_range
        self.jitter = (float(getattr(cfg.curriculum, "xy_jitter", 0.0)),
                       float(getattr(cfg.curriculum, "angle_jitter", 0.0)))
        self.stage, self.free = int(getattr(cfg.curriculum, "start_stage", 0)), False
        self.succ, self.target_succ = [], []
        self.env = env
        self._pin()
        logger.info(f"[curriculum:pose_path] {len(self.anchors)} anchors, start stage {self.stage}")

    def _pin(self):
        self.env.set_spawn_pose([self.anchors[self.stage].tolist()], self.stage, *self.jitter)

    @property
    def at_target(self):
        return self.stage == len(self.anchors) - 1

    def episode_done(self, info, step):
        if self.free or int(info.get("curriculum_stage", -1)) != self.stage:
            return
        ok = float(bool(info.get("is_success", False)))
        self.succ = (self.succ + [ok])[-self.window:]
        if self.at_target:
            self.target_succ = (self.target_succ + [ok])[-self.stop_window:]
        rate = sum(self.succ) / len(self.succ)
        if not self.at_target and len(self.succ) == self.window and rate >= self.threshold:
            logger.info(f"[curriculum:pose_path] stage {self.stage} mastered (success {rate:.2f}) at step {step}")
            self.stage += 1; self.succ = []; self._pin()
        elif self.at_target and len(self.target_succ) == self.stop_window:
            trate = sum(self.target_succ) / len(self.target_succ)
            if trate >= self.stop_success and self.final_range is not None:
                lo, hi = self.final_range
                self.env.set_spawn_x_range(lo, hi); self.free = True
                logger.info(f"[curriculum:pose_path] TARGET MASTERED ({trate:.2f}) -> FREE SPAWN x=[{lo},{hi}] at step {step}")


def rollout(env, net, steps, dev, gamma, lam, curriculum=None, step0=0):
    """Collect `steps` env steps. Every ant is a separate training sample."""
    n = env.obs_model.n_ants
    O, A, LP, V, R, D = [], [], [], [], [], []
    obs = env.obs_model.observe(env.state)
    ep_ret, ep_rets, ep_succ = 0.0, [], []
    for _ in range(steps):
        with torch.no_grad():
            a, logp, v = net.act(torch.as_tensor(obs, dtype=torch.float32, device=dev))
        a_np = a.cpu().numpy()
        _, rew, term, trunc, info = env.step(to_env_action(a_np))
        done = term or trunc
        O.append(obs.copy()); A.append(a_np); LP.append(logp.cpu().numpy())
        V.append(v.cpu().numpy()); R.append(np.full(n, rew, dtype=np.float32))
        D.append(np.full(n, float(done), dtype=np.float32))
        ep_ret += float(rew)
        if done:
            ep_rets.append(ep_ret); ep_succ.append(float(bool(info.get("is_success", False))))
            ep_ret = 0.0
            if curriculum is not None:
                curriculum.episode_done(info, step0 + len(R))
            env.reset()
        obs = env.obs_model.observe(env.state)

    with torch.no_grad():
        _, _, last_v = net(torch.as_tensor(obs, dtype=torch.float32, device=dev))
    last_v = last_v.cpu().numpy()

    V, R, D = np.array(V), np.array(R), np.array(D)
    adv = np.zeros_like(R)
    gae = np.zeros(n, dtype=np.float32)
    for t in reversed(range(len(R))):
        nxt = last_v if t == len(R) - 1 else V[t + 1]
        delta = R[t] + gamma * nxt * (1 - D[t]) - V[t]
        gae = delta + gamma * lam * (1 - D[t]) * gae
        adv[t] = gae
    ret = adv + V
    flat = lambda z: torch.as_tensor(np.concatenate(z if isinstance(z, list) else list(z)),
                                     dtype=torch.float32, device=dev)
    return (flat(O), flat(A), flat(LP), flat(adv), flat(ret),
            (float(np.mean(ep_rets)) if ep_rets else float("nan")),
            (float(np.mean(ep_succ)) if ep_succ else float("nan")))


def ppo_update(net, opt, batch, epochs, minibatch, clip, ent_coef, vf_coef):
    O, A, LP, ADV, RET = batch
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
    n = len(O)
    for _ in range(epochs):
        perm = torch.randperm(n, device=O.device)
        for k in range(0, n, minibatch):
            i = perm[k:k + minibatch]
            logp, ent, v = net.evaluate(O[i], A[i])
            ratio = (logp - LP[i]).exp()
            pg = -torch.min(ratio * ADV[i],
                            ratio.clamp(1 - clip, 1 + clip) * ADV[i]).mean()
            vloss = nn.functional.mse_loss(v, RET[i])
            loss = pg + vf_coef * vloss - ent_coef * ent.mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
    return float(pg), float(vloss)


@torch.no_grad()
def evaluate(net, cfg, n_ants, episodes, seed, dev, max_steps=500):
    cfg = cfg.copy(); cfg.ants.n = n_ants
    env = AntSwarmEnv(config=cfg, seed=seed)
    reach = float(cfg.goal.reach_radius) * float(cfg.scene_scale)
    dists, succ = [], []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        info, done, k = {}, False, 0
        while not done and k < max_steps:
            obs = env.obs_model.observe(env.state)
            a, _, _ = net.act(torch.as_tensor(obs, dtype=torch.float32, device=dev),
                              deterministic=True)
            _, _, term, trunc, info = env.step(to_env_action(a.cpu().numpy()))
            k += 1; done = term or trunc
        d = float(info.get("object_distance", env.state.distance_to_goal()))
        dists.append(d); succ.append(bool(info.get("is_success", False) or d < reach))
    env.close()
    return dict(ants=n_ants, success_rate_pct=float(np.mean(succ) * 100),
                mean_distance_m=float(np.mean(dists)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/il/il_augmented_bc.yaml")
    p.add_argument("--ants", type=int, default=4)
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.003)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--init-log-std", type=float, default=-1.0)
    p.add_argument("--eval-every", type=int, default=100_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-ants", default="2,4,10,50,100")
    p.add_argument("--seed", type=int, default=30000)
    p.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    p.add_argument("--out", default=None)
    p.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    args = p.parse_args()

    dev = torch.device(resolve_device(args.device))
    torch.manual_seed(args.seed)
    out = Path(args.out) if args.out else Path("storage_local") / build_run_id(
        f"marl_n{args.ants}")
    out.mkdir(parents=True, exist_ok=True)
    logger.add(out / "train.log", level="INFO")
    tr = Tracker(name=out.name, group="marl_ppo", tags=["marl", "ppo", f"n{args.ants}"],
                 config=vars(args), enabled=args.wandb)

    cfg = load_config(args.config).copy()
    if getattr(cfg.ants, "offsets", None) is not None:
        args.ants = int(cfg.ants.n)          # an explicit layout decides n
    cfg.ants.n = args.ants
    if cfg.env.reward_mode != "sparse":
        logger.warning(f"reward_mode={cfg.env.reward_mode} (not sparse)")
    env = AntSwarmEnv(config=cfg, seed=args.seed)
    env.reset()
    logger.info(f"MARL: {args.ants} ants, shared policy, own-row observations, "
                f"device={dev}")

    curriculum = None
    if cfg.get("curriculum") and cfg.curriculum.get("enabled"):
        curriculum = PosePathDriver(cfg, env); env.reset()
    net = SharedActorCritic(OBS_DIM, init_log_std=args.init_log_std).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    steps_done, best, history, t0 = 0, -1.0, [], time.time()
    while steps_done < args.timesteps:
        O, A, LP, ADV, RET, ret, succ = rollout(
            env, net, args.rollout_steps, dev, args.gamma, args.lam,
            curriculum=curriculum, step0=steps_done)
        pg, vl = ppo_update(net, opt, (O, A, LP, ADV, RET),
                            args.epochs, args.minibatch, args.clip,
                            args.ent_coef, args.vf_coef)
        steps_done += args.rollout_steps
        if steps_done % (args.rollout_steps * 10) == 0:
            tr.log({"rollout/ep_return": ret, "rollout/ep_success": succ,
                    **({"curriculum/stage": curriculum.stage, "curriculum/free_spawn": int(curriculum.free)} if curriculum else {}),
                    "train/pg_loss": pg, "train/vf_loss": vl}, step=steps_done)
            logger.info(f"{steps_done:>9} steps  ep_ret={ret:.3f}  ep_succ={succ:.3f}  "
                        f"pg={pg:.4f} vf={vl:.4f}  ({time.time()-t0:.0f}s)")
        if steps_done % args.eval_every < args.rollout_steps:
            r = evaluate(net, cfg, args.ants, args.eval_episodes, args.seed, dev)
            history.append({"steps": steps_done, **r})
            tr.log({"eval/success_rate_pct": r["success_rate_pct"],
                    "eval/mean_distance_m": r["mean_distance_m"]}, step=steps_done)
            logger.info(f"  eval n={args.ants}: SR={r['success_rate_pct']:.1f}%  "
                        f"mean={r['mean_distance_m']:.4f}")
            if r["success_rate_pct"] > best:
                best = r["success_rate_pct"]
                torch.save({"state_dict": net.state_dict(), "ants": args.ants},
                           out / "best.pt")
    env.close()

    torch.save({"state_dict": net.state_dict(), "ants": args.ants}, out / "final.pt")
    blob = torch.load(out / "best.pt", map_location=dev, weights_only=True)
    net.load_state_dict(blob["state_dict"])
    rows = [evaluate(net, cfg, int(n), args.eval_episodes, args.seed, dev)
            for n in args.eval_ants.split(",")]
    (out / "results.json").write_text(json.dumps(
        {"args": vars(args), "history": history, "final": rows}, indent=2) + "\n")
    tr.summary({f"final/n{r['ants']}_sr": r["success_rate_pct"] for r in rows}); tr.finish()

    print("\n" + "=" * 58)
    print(f"MARL — shared policy, trained on n={args.ants}, "
          f"{args.timesteps} steps")
    print("=" * 58)
    print(f"{'ants':>5} {'SR %':>8} {'mean dist':>11}")
    print("-" * 58)
    for r in rows:
        print(f"{r['ants']:>5} {r['success_rate_pct']:>8.1f} {r['mean_distance_m']:>11.4f}")
    print("\nEach ant sees ONLY its own row. No ant-count input, no teacher.\n")


if __name__ == "__main__":
    main()
