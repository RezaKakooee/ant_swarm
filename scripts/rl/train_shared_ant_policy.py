"""Option B: one shared policy, each ant sees only its own observation row.

Every ant runs the SAME network on ITS OWN 27-D row. No communication, no
global state, no ant-count input. The policy must work at any swarm size.

Why that is hard
----------------
The reference controller (scripts/rl/test_multiagent_split.py) splits the
single-ant wrench by least squares:

    f_i = F/n + lambda * perp(a_i),      lambda = T / sum_j |a_j|^2

``sum_j |a_j|^2 ~ n * E|a|^2``, so BOTH terms shrink like 1/n. An ant cannot
choose its force without knowing how many others are pushing -- and it is not
told.

The count is recoverable from physics: ``obs[9:11]`` is the load velocity
normalised by ``push_strength * n_ants / mass / damping``. An ant that knows its
own strength can compare it against how fast the load actually moves. But that
only works once the load is moving; at step 0 the velocity is zero and n is
unknowable. Hence the history option.

    --history 1   single frame  -- can one observation carry the swarm size?
    --history K   K frames      -- does short memory recover it?

Neither adds an input channel. The target is the force VECTOR (fx, fy)/push,
not the angle, so the +-pi wrap never enters the loss.

    python scripts/rl/train_shared_ant_policy.py --train-ants 2,4,8,16,32 \
        --test-ants 1,2,4,10,50,100 --history 1
"""
from __future__ import annotations

import argparse
import json
import math
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
from ant_swarm import AntSwarmEnv, load_config          # noqa: E402
from ant_swarm.run_id import build_run_id               # noqa: E402
from ant_swarm.compute import resolve_device            # noqa: E402
from train_chunked_bc import ChunkPolicy                # noqa: E402
from test_multiagent_split import single_ant_obs, split_wrench   # noqa: E402


class WrenchHeadPolicy(nn.Module):
    """obs row -> the SHARED split coefficients; geometry does the rest.

    Predicting f_i directly fails: each ant is ~1-5% wrong, but the torque is a
    near-cancelling sum of large opposing terms, so the errors survive while the
    signal cancels. Measured total-torque error was 90% at n=2 and 357% at n=100.

    Here the network outputs only what is common to every ant:

        (c_Fx, c_Fy, lam)   ->   f_i = c_F + lam * perp(a_i)

    with a_i the ant's own arm, recovered from obs[0:2] and obs[6:8] by fixed
    geometry, not learned. Total torque becomes sum_i lam^i |a_i|^2, so per-ant
    errors AVERAGE (1/sqrt(n)) instead of accumulating.
    """

    def __init__(self, obs_dim=27, history=1, hidden=256,
                 stem_half=0.16575, cap_half=0.0874):
        super().__init__()
        self.history, self.obs_dim = int(history), int(obs_dim)
        self.stem_half, self.cap_half = float(stem_half), float(cap_half)
        self.net = nn.Sequential(
            nn.Linear(obs_dim * self.history, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        z = self.net(x)
        cF, lam = z[:, :2], z[:, 2:3]
        last = x[:, -self.obs_dim:]                 # newest frame
        att = torch.stack([last[:, 0] * self.stem_half,
                           last[:, 1] * self.cap_half], dim=1)
        sin, cos = last[:, 6], last[:, 7]
        ax = cos * att[:, 0] - sin * att[:, 1]      # world arm = R(theta) @ att
        ay = sin * att[:, 0] + cos * att[:, 1]
        perp = torch.stack([-ay, ax], dim=1)
        return cF + lam * perp


class WrenchLogPolicy(nn.Module):
    """Like WrenchHeadPolicy, but the 1/n scale is factored out explicitly.

    Both split coefficients shrink like 1/n:

        c_F = F/n        lambda = T / sum_j |a_j|^2  ~  T / (n * E|a|^2)

    Across n = 2..100 that is a 50x range. Predicting them on a linear scale
    puts nearly all the regression effort on small swarms and leaves large ones
    badly biased -- measured torque error 312% at n=100 even though each ant was
    only 1.3% wrong. A separate probe showed the ant CAN read n from its own
    observation (R2 0.86 at 1 frame, 0.94 at 4), so the information is there; the
    parameterisation was wrong.

    Here the network predicts a log scale k ~ log(1/n) and two O(1) shape terms:

        c_F = cf_hat * exp(k)        lambda = lam_hat * exp(k)

    so every swarm size is represented with the same relative precision.
    """

    def __init__(self, obs_dim=27, history=1, hidden=256,
                 stem_half=0.16575, cap_half=0.0874, k_range=(-6.0, 2.0)):
        super().__init__()
        self.history, self.obs_dim = int(history), int(obs_dim)
        self.stem_half, self.cap_half = float(stem_half), float(cap_half)
        self.k_lo, self.k_hi = k_range
        self.net = nn.Sequential(
            nn.Linear(obs_dim * self.history, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, x):
        z = self.net(x)
        k = self.k_lo + (self.k_hi - self.k_lo) * torch.sigmoid(z[:, 3:4])
        scale = torch.exp(k)
        cF = z[:, :2] * scale
        lam = z[:, 2:3] * scale
        last = x[:, -self.obs_dim:]
        att = torch.stack([last[:, 0] * self.stem_half,
                           last[:, 1] * self.cap_half], dim=1)
        sin, cos = last[:, 6], last[:, 7]
        ax = cos * att[:, 0] - sin * att[:, 1]
        ay = sin * att[:, 0] + cos * att[:, 1]
        return cF + lam * torch.stack([-ay, ax], dim=1)


class SharedAntPolicy(nn.Module):
    """obs row (x history) -> that ant's force vector, in units of push_strength."""

    def __init__(self, obs_dim=27, history=1, hidden=256):
        super().__init__()
        self.history = int(history)
        self.obs_dim = int(obs_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim * self.history, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):                      # (B, obs_dim*history)
        return self.net(x)


class AntHistory:
    """Per-ant ring of the last K observation rows, oldest first."""

    def __init__(self, n_ants, obs_dim, history):
        self.k = int(history)
        self.buf = np.zeros((n_ants, self.k, obs_dim), dtype=np.float32)
        self.primed = False

    def push(self, obs_rows):
        if not self.primed:
            self.buf[:] = obs_rows[:, None, :]
            self.primed = True
        else:
            self.buf[:, :-1] = self.buf[:, 1:]
            self.buf[:, -1] = obs_rows
        return self.buf.reshape(len(self.buf), -1)


# --------------------------------------------------------------------------- #
# teacher data: the wrench-split controller, recorded per ant
# --------------------------------------------------------------------------- #
def collect(cfg_path, ckpt, ant_counts, episodes, history, seed, max_steps, dev):
    blob = torch.load(ckpt, map_location=dev, weights_only=True)
    if blob.get("goal_observation_version") != 2:
        raise SystemExit("single-ant checkpoint predates the goal fix")
    teacher = ChunkPolicy(27, int(blob.get("horizon", 1))).to(dev)
    teacher.load_state_dict(blob["state_dict"])
    teacher.eval()

    X, Y, t0 = [], [], time.time()
    for n in ant_counts:
        cfg = load_config(cfg_path).copy()
        cfg.ants.n = n
        env = AntSwarmEnv(config=cfg, seed=seed)
        s = float(cfg.scene_scale)
        push = float(cfg.physics.push_strength) * s
        spin_cfg = cfg.physics.get("spin_strength", None)
        spin_strength = (float(spin_cfg) * s if spin_cfg is not None
                         else push * (float(cfg.tshape.stem_len) * s) / 2)
        offsets = env.attachment_offsets

        for ep in range(episodes):
            env.reset(seed=seed + ep)
            hist = AntHistory(n, 27, history)
            done, steps = False, 0
            while not done and steps < max_steps:
                obs = env.obs_model.observe(env.state)
                feats = hist.push(obs)
                with torch.no_grad():
                    a = teacher(torch.tensor(single_ant_obs(obs, n)[None, :],
                                             device=dev))[0, 0].cpu().numpy()
                angle, mag = float(a[0]), float(np.clip(a[1], 0, 1))
                spin = float(np.clip(a[2], -1, 1))
                F = push * mag * np.array([math.cos(angle), math.sin(angle)])
                T = spin * spin_strength
                arms = offsets @ env.state.obj.rot().T
                f, _ = split_wrench(F, T, arms, push)

                X.append(feats.copy())
                Y.append((f / push).astype(np.float32))       # units of push_strength

                action = np.zeros((n, 2), dtype=np.float32)
                action[:, 0] = np.arctan2(f[:, 1], f[:, 0])
                action[:, 1] = np.clip(np.linalg.norm(f, axis=1) / push, 0.0, 1.0)
                _, _, term, trunc, _ = env.step(action)
                steps += 1
                done = term or trunc
        env.close()
        logger.info(f"  n={n}: {sum(len(x) for x in X)} ant-samples so far "
                    f"({time.time()-t0:.0f}s)")
    return (np.concatenate(X).astype(np.float32),
            np.concatenate(Y).astype(np.float32))


def build_policy(head, history, cfg_path):
    """Single switch point between the two per-ant heads."""
    cfg = load_config(cfg_path)
    stem_half = float(cfg.tshape.stem_len) * float(cfg.scene_scale) / 2
    cap_half = max(float(cfg.tshape.cap_big_len),
                   float(cfg.tshape.cap_small_len)) * float(cfg.scene_scale) / 2
    if head == 'direct':
        return SharedAntPolicy(27, history)
    if head == 'wrench':
        return WrenchHeadPolicy(27, history, stem_half=stem_half, cap_half=cap_half)
    if head == 'wrenchlog':
        return WrenchLogPolicy(27, history, stem_half=stem_half, cap_half=cap_half)
    raise ValueError(f"unknown head {head!r}; use 'direct' or 'wrench'")


def train(X, Y, history, epochs, batch_size, lr, dev, seed=0,
          head='direct', cfg_path='configs/il/il_augmented_bc.yaml',
          loss='mse'):
    torch.manual_seed(seed)
    net = build_policy(head, history, cfg_path).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt, Yt = torch.tensor(X, device=dev), torch.tensor(Y, device=dev)
    n = len(Xt)
    logger.info(f"training shared policy on {n} ant-samples, "
                f"history={history}, head={head}")
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=dev)
        tot, nb = 0.0, 0
        for k in range(0, n - batch_size + 1, batch_size):
            idx = perm[k:k + batch_size]
            pred, target = net(Xt[idx]), Yt[idx]
            if loss == 'relative':
                # per-sample relative error: a 100-ant step counts as much as
                # a 2-ant step, even though its forces are 50x smaller
                denom = (target ** 2).sum(1, keepdim=True) + 1e-8
                err = (((pred - target) ** 2) / denom).mean()
            else:
                err = nn.functional.mse_loss(pred, target)
            opt.zero_grad(); err.backward(); opt.step()
            tot += err.detach().item(); nb += 1
        sched.step()
        if ep % 10 == 0 or ep == 1 or ep == epochs:
            logger.info(f"  epoch {ep:3d}/{epochs} MSE={tot/max(nb,1):.6f}")
    return net


@torch.no_grad()
def evaluate(net, cfg_path, n, episodes, history, seed, max_steps, dev):
    cfg = load_config(cfg_path).copy()
    cfg.ants.n = n
    env = AntSwarmEnv(config=cfg, seed=seed)
    if env.action_model.single_spin:
        # n=1 attaches at the centre of the stem, so its arm is zero and a
        # push-only ant can produce no torque at all. The env compensates with
        # a direct spin action, which a shared push-only policy does not have.
        env.close()
        raise ValueError(
            f'n={n} uses the single-agent spin action; a push-only shared '
            'policy cannot drive it. Use n>=2, or set ants.single_agent_spin=false.')
    reach = float(cfg.goal.reach_radius) * float(cfg.scene_scale)
    dists, succ = [], []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        hist = AntHistory(n, 27, history)
        info, done, steps = {}, False, 0
        while not done and steps < max_steps:
            obs = env.obs_model.observe(env.state)
            feats = hist.push(obs)
            f = net(torch.tensor(feats, device=dev)).cpu().numpy()    # (n, 2)
            action = np.zeros((n, 2), dtype=np.float32)
            action[:, 0] = np.arctan2(f[:, 1], f[:, 0])
            action[:, 1] = np.clip(np.linalg.norm(f, axis=1), 0.0, 1.0)
            _, _, term, trunc, info = env.step(action)
            steps += 1
            done = term or trunc
        d = float(info.get("object_distance", env.state.distance_to_goal()))
        dists.append(d)
        succ.append(bool(info.get("is_success", False) or d < reach))
    env.close()
    return dict(ants=n, episodes=episodes, successes=int(np.sum(succ)),
                success_rate_pct=float(np.mean(succ) * 100),
                mean_distance_m=float(np.mean(dists)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/il/il_augmented_bc.yaml")
    p.add_argument("--teacher",
                   default="storage_local/ant__20260905_1322__240805__train_goal_bc/bc_goal_fixed.pt")
    p.add_argument("--train-ants", default="2,4,8,16,32")
    p.add_argument("--test-ants", default="2,3,4,10,50,100")
    p.add_argument("--history", type=int, default=1)
    p.add_argument("--loss", default="relative", choices=["mse", "relative"],
                   help="relative = weight every swarm size equally")
    p.add_argument("--head", default="wrenchlog", choices=["direct", "wrench", "wrenchlog"],
                   help="direct = predict f_i; wrench = predict the shared "
                        "(c_F, lambda) and apply the exact split formula")
    p.add_argument("--episodes", type=int, default=40, help="teacher episodes per ant count")
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=30000)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--device", default="cpu",
                   help="cpu | cuda | auto; cuda fails loudly if unavailable")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    dev = torch.device(resolve_device(args.device))
    out = Path(args.out) if args.out else Path("storage_local") / build_run_id(
        f"shared_ant_{args.head}_{args.loss}_h{args.history}")
    out.mkdir(parents=True, exist_ok=True)
    logger.add(out / "train.log", level="INFO")

    train_ants = [int(x) for x in args.train_ants.split(",") if x.strip()]
    test_ants = [int(x) for x in args.test_ants.split(",") if x.strip()]
    logger.info(f"device={dev}  history={args.history}  train_ants={train_ants}")

    X, Y = collect(args.config, args.teacher, train_ants, args.episodes,
                   args.history, args.seed, args.max_steps, dev)
    logger.info(f"dataset: {X.shape} -> {Y.shape}")
    net = train(X, Y, args.history, args.epochs, args.batch_size, args.lr, dev,
                head=args.head, cfg_path=args.config, loss=args.loss)
    torch.save({"state_dict": net.state_dict(), "history": args.history,
                "head": args.head, "train_ants": train_ants},
               out / "shared_ant_policy.pt")

    rows = [evaluate(net, args.config, n, args.eval_episodes, args.history,
                     args.seed, args.max_steps, dev) for n in test_ants]
    for r in rows:
        logger.info(f"  n={r['ants']:>3}: SR={r['success_rate_pct']:.1f}%  "
                    f"mean={r['mean_distance_m']:.4f}")
    (out / "results.json").write_text(json.dumps(
        {"args": vars(args), "results": rows}, indent=2) + "\n")

    print("\n" + "=" * 62)
    print(f"SHARED PER-ANT POLICY — head={args.head}, history={args.history}, "
          f"trained on n={train_ants}")
    print("=" * 62)
    print(f"{'ants':>5} {'trained on':>11} {'SR %':>8} {'mean dist':>11}")
    print("-" * 62)
    for r in rows:
        print(f"{r['ants']:>5} {'yes' if r['ants'] in train_ants else 'no':>11} "
              f"{r['success_rate_pct']:>8.1f} {r['mean_distance_m']:>11.4f}")
    print("\nEach ant sees ONLY its own observation row. No ant-count input.\n")


if __name__ == "__main__":
    main()
