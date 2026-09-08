"""A/B test: does a discrete torque head beat MSE regression on torque?

Motivation
----------
`scripts/il/test_torque_observability.py` showed that the torque sign IS
predictable from the 27-D observation (AUC 0.93 in the tight, near-wall
subset).  So the observation is fine and the suspect is the loss: MSE on a
bimodal target (+1 / -1 recovery torque) collapses to ~0 and the load never
un-wedges.

This script trains the SAME network on the SAME data twice, changing exactly
one thing -- the torque head:

    mse    torque = tanh(z),  MSE loss            (the current baseline)
    bins   torque = argmax over K bins, cross-entropy loss

Push angle and force keep the MSE parameterisation in both runs, so any
difference is attributable to the torque head alone.  (Push angle is circular,
but only 0.5% of demo steps sit near the +-pi wrap, so the wrap is not worth
changing here.)

Besides closed-loop success it reports **mean |torque|** predicted on held-out
near-wall states.  If the MSE run collapses (predicted |torque| far below the
expert's) that is the smoothing effect, measured directly.

Usage
-----
    # reuse the replay cache written by test_torque_observability.py
    python scripts/il/train_discrete_torque_bc.py --cache <path>/replay_4000.npz

    # or replay from scratch (slow: ~30 min for 4000 episodes)
    python scripts/il/train_discrete_torque_bc.py --episodes 4000
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
from ant_swarm import AntSwarmEnv, load_config

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "il"))
from test_torque_observability import collect, min_tip_head_gap  # noqa: E402


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #
class BCPolicy(nn.Module):
    """Shared trunk, separate heads for push angle, force and spin torque."""

    def __init__(self, obs_dim, torque_head="bins", n_bins=21, hidden=256):
        super().__init__()
        self.torque_head = torque_head
        self.n_bins = n_bins
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_angle = nn.Linear(hidden, 1)
        self.head_force = nn.Linear(hidden, 1)
        self.head_torque = nn.Linear(hidden, n_bins if torque_head == "bins" else 1)
        # bin centres spread over [-1, 1]
        self.register_buffer("bin_centers", torch.linspace(-1.0, 1.0, n_bins))

    def forward(self, obs):
        h = self.trunk(obs)
        return self.head_angle(h), self.head_force(h), self.head_torque(h)

    def act(self, obs):
        """Deterministic action in env units: [angle in +-pi, force in 0..1, torque in +-1]."""
        z_ang, z_frc, z_trq = self(obs)
        angle = torch.tanh(z_ang).squeeze(-1) * math.pi
        force = (torch.tanh(z_frc).squeeze(-1) + 1.0) * 0.5
        if self.torque_head == "bins":
            torque = self.bin_centers[z_trq.argmax(dim=-1)]
        else:
            torque = torch.tanh(z_trq).squeeze(-1)
        return torch.stack([angle, force, torque], dim=-1)


def torque_to_bin(torque, n_bins):
    """Nearest-bin index for a torque value in [-1, 1]."""
    idx = np.rint((np.clip(torque, -1.0, 1.0) + 1.0) / 2.0 * (n_bins - 1))
    return idx.astype(np.int64)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train(obs, act, torque_head, n_bins, epochs, batch_size, lr, device, seed=0):
    torch.manual_seed(seed)
    dev = torch.device(device)
    net = BCPolicy(obs.shape[1], torque_head, n_bins).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    obs_t = torch.tensor(obs, device=dev)
    ang_t = torch.tensor(act[:, 0], device=dev)
    frc_t = torch.tensor(act[:, 1], device=dev)
    if torque_head == "bins":
        trq_t = torch.tensor(torque_to_bin(act[:, 2], n_bins), device=dev)
        torque_loss = nn.CrossEntropyLoss()
    else:
        trq_t = torch.tensor(act[:, 2], device=dev)
        torque_loss = nn.MSELoss()
    mse = nn.MSELoss()

    n = len(obs_t)
    logger.info(f"[{torque_head}] training on {n} transitions, {epochs} epochs, "
                f"device={device}")
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n, device=dev)
        tot, nb = 0.0, 0
        for k in range(0, n - batch_size + 1, batch_size):
            idx = perm[k:k + batch_size]
            z_ang, z_frc, z_trq = net(obs_t[idx])
            loss = (mse(torch.tanh(z_ang).squeeze(-1) * math.pi, ang_t[idx])
                    + mse((torch.tanh(z_frc).squeeze(-1) + 1.0) * 0.5, frc_t[idx]))
            if torque_head == "bins":
                loss = loss + torque_loss(z_trq, trq_t[idx])
            else:
                loss = loss + torque_loss(torch.tanh(z_trq).squeeze(-1), trq_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.detach().item()
            nb += 1
        sched.step()
        if ep % 5 == 0 or ep == 1 or ep == epochs:
            logger.info(f"  [{torque_head}] epoch {ep:3d}/{epochs} loss={tot/max(nb,1):.5f}")
    return net


# --------------------------------------------------------------------------- #
# measurements
# --------------------------------------------------------------------------- #
@torch.no_grad()
def torque_magnitude(net, obs, act, device, tag):
    """Mean |torque| predicted vs the expert's, on the given states."""
    net.eval()
    pred = net.act(torch.tensor(obs, device=torch.device(device))).cpu().numpy()
    exp_mag = float(np.abs(act[:, 2]).mean())
    pred_mag = float(np.abs(pred[:, 2]).mean())
    sign_acc = float((np.sign(pred[:, 2]) == np.sign(act[:, 2])).mean())
    logger.info(f"  [{tag}] |torque| expert={exp_mag:.3f} predicted={pred_mag:.3f} "
                f"(ratio {pred_mag/max(exp_mag,1e-8):.2f}), sign match={sign_acc:.3f}")
    return exp_mag, pred_mag, sign_acc


@torch.no_grad()
def evaluate(net, cfg, dataset_path, n_episodes, device, max_steps=500, seed=42):
    """Closed-loop rollouts from demo start poses / goals."""
    net.eval()
    dev = torch.device(device)
    z = np.load(dataset_path)
    init_poses, goals = z["init_pose"], z["goal"]

    env = AntSwarmEnv(config=cfg, seed=seed)
    reach = float(cfg.goal.reach_radius) * float(cfg.scene_scale)
    dists, succ = [], []

    from train_chunked_bc import pick_eval_episodes
    for ep in pick_eval_episodes(init_poses, n_episodes):
        env.reset(seed=int(seed) + int(ep), options={
            "init_pose": init_poses[ep], "goal": goals[ep]})

        info, done, steps = {}, False, 0
        while not done and steps < max_steps:
            obs = flatten(env.observation_space,
                          env.obs_model.observe(env.state)).astype(np.float32)
            a = net.act(torch.tensor(obs[None, :], device=dev)).cpu().numpy()[0]
            _, _, terminated, truncated, info = env.step(
                a.astype(np.float32).reshape(env.action_space.shape))
            steps += 1
            done = terminated or truncated

        d = float(info.get("object_distance", env.state.distance_to_goal()))
        dists.append(d)
        succ.append(bool(info.get("is_success", False) or d < reach))

    env.close()
    return float(np.mean(succ) * 100.0), float(np.mean(dists)), float(np.min(dists))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/il/il_augmented_bc.yaml")
    p.add_argument("--dataset", default="storage_local/datasets/successes_v1/dataset.npz")
    p.add_argument("--cache", default="", help="replay cache .npz (skips the slow replay)")
    p.add_argument("--episodes", type=int, default=4000, help="episodes to replay if no cache")
    p.add_argument("--heads", default="mse,bins", help="comma list: mse and/or bins")
    p.add_argument("--bins", type=int, default=21)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=6,
                   help="torch CPU threads (this box has no GPU)")
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--holdout", type=float, default=0.05, help="episode fraction held out")
    p.add_argument("--slit", type=int, default=2, choices=[1, 2])
    p.add_argument("--tight", type=float, default=0.05)
    p.add_argument("--margin", type=float, default=0.12)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-dir", default="storage_local/checkpoints")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and args.threads > 0:
        torch.set_num_threads(args.threads)
        logger.info(f"cpu training with {args.threads} threads")
    cfg = load_config(args.config)
    wall_x = float(cfg.walls.x_columns[args.slit - 1]) * float(cfg.scene_scale)

    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists():
        logger.info(f"Loading replay cache {cache}")
        z = np.load(cache)
        data = {k: z[k] for k in z.files}
    else:
        data = collect(cfg, args.dataset, args.episodes)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **data)

    obs, act, epid = data["obs"], data["act"], data["epid"]
    from repair_replay_goals import require_correct_goals
    with np.load(args.dataset) as demonstrations:
        require_correct_goals(obs, epid, demonstrations["goal"], cfg)
    logger.info(f"{len(obs)} transitions from {len(np.unique(epid))} episodes")

    # hold out whole episodes so the |torque| check is not measured on train data
    uniq = np.unique(epid)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    held = set(uniq[:max(1, int(len(uniq) * args.holdout))].tolist())
    is_held = np.array([e in held for e in epid])
    gap = min_tip_head_gap(data, wall_x)
    tight = is_held & (np.abs(data["cx"] - wall_x) < args.margin) & (gap < args.tight)
    logger.info(f"held-out: {int(is_held.sum())} steps, of which {int(tight.sum())} "
                f"are tight near-wall states")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for head in [h.strip() for h in args.heads.split(",") if h.strip()]:
        t0 = time.time()
        net = train(obs[~is_held], act[~is_held], head, args.bins,
                    args.epochs, args.batch_size, args.lr, device)
        ckpt = save_dir / f"bc_torque_{head}.pt"
        torch.save({"state_dict": net.state_dict(), "torque_head": head,
                    "n_bins": args.bins, "obs_dim": obs.shape[1]}, ckpt)

        logger.info(f"[{head}] measuring torque magnitude ...")
        _, mag_all, sign_all = torque_magnitude(net, obs[is_held], act[is_held],
                                                device, f"{head}/held-out")
        exp_t, mag_t, sign_t = torque_magnitude(net, obs[tight], act[tight],
                                                device, f"{head}/tight")
        logger.info(f"[{head}] closed-loop eval on {args.eval_episodes} episodes ...")
        sr, mean_d, best_d = evaluate(net, cfg, args.dataset, args.eval_episodes, device)
        logger.info(f"[{head}] SR={sr:.1f}%  mean_dist={mean_d:.4f}  best={best_d:.4f}  "
                    f"({time.time()-t0:.0f}s)  -> {ckpt}")
        results.append((head, exp_t, mag_t, sign_t, sr, mean_d, best_d))

    print("\n" + "=" * 78)
    print(f"TORQUE HEAD A/B  (slit {args.slit}, {args.bins} bins, "
          f"{args.epochs} epochs, {len(obs)} transitions)")
    print("=" * 78)
    print(f"{'head':<6} {'|trq| exp':>10} {'|trq| pred':>11} {'sign acc':>9} "
          f"{'SR %':>7} {'mean dist':>10} {'best dist':>10}")
    print("-" * 78)
    for head, exp_t, mag_t, sign_t, sr, mean_d, best_d in results:
        print(f"{head:<6} {exp_t:>10.3f} {mag_t:>11.3f} {sign_t:>9.3f} "
              f"{sr:>7.1f} {mean_d:>10.4f} {best_d:>10.4f}")
    print("\n(|trq| columns are measured on held-out TIGHT near-wall states.)")
    print("If mse shows |trq| pred << |trq| exp and bins does not, the MSE head is")
    print("collapsing on the bimodal target -- which is the thing being tested.\n")


if __name__ == "__main__":
    main()
