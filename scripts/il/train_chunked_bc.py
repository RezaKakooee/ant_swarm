"""Action-chunked behavioural cloning (ACT-style), pure end-to-end.

Why
---
Measured facts from this repo's diagnostics:
  * the torque sign IS predictable from the 27-D observation (AUC 0.93), so the
    observation is not missing a feature;
  * a discrete torque head fixes MSE magnitude collapse but changes nothing;
  * the policy is ~85% correct on expert states yet 0% closed-loop.
That gap is compounding error, and the one thing the policy lacks is a sense of
where to go over the next ~0.1 m.

The demonstrations already contain that: the actions over the next H steps.
So predict the whole chunk [a_t .. a_{t+H-1}] instead of a single action.  No
geodesic field, no curriculum, no privileged input -- only the demo data the
single-step BC already uses.

Execution modes
---------------
  chunk      predict once, run all H actions open-loop, then re-plan
  ensemble   re-plan every step and average the overlapping predictions with
             exponential weights (ACT temporal ensembling).  Push angle is
             circular, so it is averaged as a unit vector, not as a number.

Usage
-----
    python scripts/il/train_chunked_bc.py --cache storage_local/cache/replay_4000.npz \
        --horizons 1,8,32 --exec ensemble
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces.utils import flatten
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "il"))
from ant_swarm import AntSwarmEnv, load_config          # noqa: E402
from test_torque_observability import collect           # noqa: E402


# --------------------------------------------------------------------------- #
def pick_eval_episodes(init_poses, n, seed=0):
    """Episodes with DISTINCT start poses, spread over the whole dataset.

    The dataset is stored run by run and repeats start poses heavily: the
    first 50 episodes all share one start pose, and the first 534 contain only
    20 distinct ones.  Evaluating on `range(n)` therefore measures a single
    start pose with n different goals, not n independent episodes.
    """
    _, first = np.unique(init_poses, axis=0, return_index=True)
    rng = np.random.default_rng(seed)
    k = min(n, len(first))
    return np.sort(rng.choice(first, size=k, replace=False))


class ChunkPolicy(nn.Module):
    """obs -> H future actions.  Same trunk as the single-step baseline."""

    def __init__(self, obs_dim, horizon, hidden=256):
        super().__init__()
        self.horizon = horizon
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, horizon * 3)

    def forward(self, obs):
        z = self.head(self.trunk(obs)).view(-1, self.horizon, 3)
        angle = torch.tanh(z[..., 0]) * math.pi
        force = (torch.tanh(z[..., 1]) + 1.0) * 0.5
        torque = torch.tanh(z[..., 2])
        return torch.stack([angle, force, torque], dim=-1)


def build_chunk_index(epid, horizon):
    """(N, H) index of the next H steps, clipped at the episode end, plus a mask."""
    n = len(epid)
    idx = np.arange(n, dtype=np.int64)[:, None] + np.arange(horizon)[None, :]
    idx = np.minimum(idx, n - 1)
    # last index of each episode, broadcast per step
    ep_end = np.zeros(n, dtype=np.int64)
    bounds = np.flatnonzero(np.diff(epid)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds - 1, [n - 1]))
    for s, e in zip(starts, ends):
        ep_end[s:e + 1] = e
    mask = idx <= ep_end[:, None]
    idx = np.minimum(idx, ep_end[:, None])          # pad by repeating the last action
    return idx.astype(np.int32), mask


def train(obs, act, idx, mask, horizon, epochs, batch_size, lr, device, seed=0):
    torch.manual_seed(seed)
    dev = torch.device(device)
    net = ChunkPolicy(obs.shape[1], horizon).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    obs_t = torch.tensor(obs, device=dev)
    act_t = torch.tensor(act, device=dev)
    idx_t = torch.tensor(idx, device=dev, dtype=torch.long)
    mask_t = torch.tensor(mask, device=dev, dtype=torch.float32)

    n = len(obs_t)
    logger.info(f"[H={horizon}] training on {n} chunks, {epochs} epochs")
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n, device=dev)
        tot, nb = 0.0, 0
        for k in range(0, n - batch_size + 1, batch_size):
            b = perm[k:k + batch_size]
            target = act_t[idx_t[b]]                     # (B, H, 3)
            m = mask_t[b].unsqueeze(-1)                  # (B, H, 1)
            pred = net(obs_t[b])
            loss = (((pred - target) ** 2) * m).sum() / (m.sum() * 3.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.detach().item()
            nb += 1
        sched.step()
        if ep % 10 == 0 or ep == 1 or ep == epochs:
            logger.info(f"  [H={horizon}] epoch {ep:3d}/{epochs} loss={tot/max(nb,1):.5f}")
    return net


# --------------------------------------------------------------------------- #
class Ensembler:
    """ACT temporal ensembling.  Angle is circular -> averaged as a unit vector."""

    def __init__(self, horizon, decay=0.01):
        self.h = horizon
        self.w = np.exp(-decay * np.arange(horizon)).astype(np.float32)
        self.buf = []            # list of (age, chunk)

    def reset(self):
        self.buf = []

    def push_and_pop(self, chunk):
        self.buf.append([0, chunk])
        self.buf = [b for b in self.buf if b[0] < self.h]
        cs, sn, frc, trq, wsum = 0.0, 0.0, 0.0, 0.0, 0.0
        for age, ch in self.buf:
            a = ch[age]
            w = self.w[age]
            cs += w * math.cos(float(a[0]))
            sn += w * math.sin(float(a[0]))
            frc += w * float(a[1])
            trq += w * float(a[2])
            wsum += w
        for b in self.buf:
            b[0] += 1
        return np.array([math.atan2(sn, cs), frc / wsum, trq / wsum], dtype=np.float32)


@torch.no_grad()
def evaluate(net, cfg, dataset_path, n_episodes, device, mode, decay,
             max_steps=500, seed=42):
    net.eval()
    dev = torch.device(device)
    z = np.load(dataset_path)
    init_poses, goals = z["init_pose"], z["goal"]
    eval_eps = pick_eval_episodes(init_poses, n_episodes)
    env = AntSwarmEnv(config=cfg, seed=seed)
    reach = float(cfg.goal.reach_radius) * float(cfg.scene_scale)
    H = net.horizon
    ens = Ensembler(H, decay)
    dists, succ = [], []

    for ep in eval_eps:
        ep = int(ep)
        env.reset(seed=int(seed) + int(ep), options={
            "init_pose": init_poses[ep], "goal": goals[ep]})
        ens.reset()

        info, done, steps, pending = {}, False, 0, None
        while not done and steps < max_steps:
            if mode == "ensemble" or pending is None or len(pending) == 0:
                o = flatten(env.observation_space,
                            env.obs_model.observe(env.state)).astype(np.float32)
                chunk = net(torch.tensor(o[None, :], device=dev)).cpu().numpy()[0]
                if mode == "ensemble":
                    a = ens.push_and_pop(chunk)
                else:
                    pending = list(chunk)
                    a = pending.pop(0)
            else:
                a = pending.pop(0)
            _, _, term, trunc, info = env.step(
                np.asarray(a, dtype=np.float32).reshape(env.action_space.shape))
            steps += 1
            done = term or trunc

        d = float(info.get("object_distance", env.state.distance_to_goal()))
        dists.append(d)
        succ.append(bool(info.get("is_success", False) or d < reach))

    env.close()
    return (float(np.mean(succ) * 100.0), float(np.mean(dists)),
            float(np.median(dists)), float(np.min(dists)))


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/il/il_augmented_bc.yaml")
    p.add_argument("--dataset", default="storage_local/datasets/successes_v1/dataset.npz")
    p.add_argument("--cache", default="storage_local/cache/replay_4000.npz")
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--horizons", default="1,8,32")
    p.add_argument("--exec", dest="exec_modes", default="ensemble",
                   help="comma list: ensemble and/or chunk")
    p.add_argument("--decay", type=float, default=0.01, help="ensembling weight decay")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--holdout", type=float, default=0.05)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-dir", default="storage_local/checkpoints")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}")
    cfg = load_config(args.config)

    cache = Path(args.cache)
    if cache.exists():
        logger.info(f"Loading replay cache {cache}")
        z = np.load(cache)
        data = {k: z[k] for k in z.files}
    else:
        data = collect(cfg, args.dataset, args.episodes)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **data)
    obs, act, epid = data["obs"], data["act"], data["epid"]
    from repair_replay_goals import require_correct_goals
    with np.load(args.dataset) as demonstrations:
        require_correct_goals(obs, epid, demonstrations["goal"], cfg)
    logger.info(f"{len(obs)} transitions from {len(np.unique(epid))} episodes")

    uniq = np.unique(epid)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    held = set(uniq[:max(1, int(len(uniq) * args.holdout))].tolist())
    keep = ~np.array([e in held for e in epid])

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for hs in [h.strip() for h in args.horizons.split(",") if h.strip()]:
        H = int(hs)
        idx, mask = build_chunk_index(epid, H)
        net = train(obs[keep], act, idx[keep], mask[keep], H,
                    args.epochs, args.batch_size, args.lr, device)
        torch.save({"state_dict": net.state_dict(), "horizon": H},
                   save_dir / f"bc_chunk_H{H}.pt")
        for mode in [m.strip() for m in args.exec_modes.split(",") if m.strip()]:
            t0 = time.time()
            sr, mean_d, med_d, best_d = evaluate(
                net, cfg, args.dataset, args.eval_episodes, device, mode, args.decay)
            logger.info(f"[H={H} {mode}] SR={sr:.1f}% mean={mean_d:.4f} "
                        f"median={med_d:.4f} best={best_d:.4f} ({time.time()-t0:.0f}s)")
            results.append((H, mode, sr, mean_d, med_d, best_d))

    print("\n" + "=" * 72)
    print(f"ACTION-CHUNKED BC  ({args.epochs} epochs, {len(obs)} transitions, "
          f"{args.eval_episodes} eval episodes)")
    print("=" * 72)
    print(f"{'H':>4} {'exec':<10} {'SR %':>7} {'mean dist':>11} {'median':>10} {'best':>9}")
    print("-" * 72)
    for H, mode, sr, mean_d, med_d, best_d in results:
        print(f"{H:>4} {mode:<10} {sr:>7.1f} {mean_d:>11.4f} {med_d:>10.4f} {best_d:>9.4f}")
    print("\nH=1 is the single-step baseline. No geodesic field is used anywhere.\n")


if __name__ == "__main__":
    main()
