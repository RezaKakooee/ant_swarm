"""Chapter 04 §7, first step: does the decentralised base controller work alone?

Each ant, from ITS OWN observation row only:
  1. rebuilds the single-ant observation and runs the single-ant BC -> (F, T)
  2. estimates the swarm size N from the load's velocity (a small learned probe)
  3. takes its share with the exact least-norm split, written per ant:

         f_i = F/N + lambda * perp(a_i - abar)
         lambda = (T - abar x F) / sum_j |a_j - abar|^2

     abar is the mean arm. Dropping it (the first version of this script) gives
     the wrong wrench whenever the attachment points are not symmetric -- the
     torque can even flip sign. A decentralised ant does not know abar, but it
     knows the T-shape and the attachment rule, so it can use the rule's
     expected centroid and spread as geometry constants (levels 1-2).

No learning in the loop, no communication, no ant-count input, no map.

The pieces are removed one at a time so the failure, if any, has an address:

    level 0   true N, true abar and spread   -> equals option C exactly. sanity.
    level 1   true N, geometry constants     -> cost of the approximation
    level 2   estimated N, geometry constants -> the fully decentralised base

    python scripts/rl/test_decentralised_base.py --test-ants 2,4,10,50,100
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
from ant_swarm import AntSwarmEnv, load_config                    # noqa: E402
from ant_swarm.compute import resolve_device                     # noqa: E402
from ant_swarm.run_id import build_run_id                        # noqa: E402
from ant_swarm.tshape import TShape, make_attachment_offsets     # noqa: E402
from train_chunked_bc import ChunkPolicy                         # noqa: E402
from test_multiagent_split import single_ant_obs, split_wrench   # noqa: E402
from train_shared_ant_policy import AntHistory                   # noqa: E402

OBS = 27
VEL_SCALE_ANTS = None   # env.velocity_scale_ants override; None = live n_ants
FIXED_N = None          # level 2 with a constant assumed N instead of a counter


def cfg_load(path):
    cfg = load_config(path).copy()
    if VEL_SCALE_ANTS is not None:
        cfg.env.velocity_scale_ants = int(VEL_SCALE_ANTS)
    return cfg


# --------------------------------------------------------------------------- #
# the swarm-size probe
# --------------------------------------------------------------------------- #
class NProbe(nn.Module):
    """own row (x history) [+ own previous log2 estimate] -> log2(N).

    With ``feedback`` the ant feeds its last estimate back in, so it can read
    the ratio of the load's response to what it assumed and correct over time.
    """

    def __init__(self, history=4, hidden=256, feedback=False):
        super().__init__()
        self.history, self.feedback = int(history), bool(feedback)
        d = OBS * history + (1 if feedback else 0)
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sd", torch.ones(d))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd).squeeze(-1)


def teacher_wrench(teacher, obs_rows, n, push, spin_strength, dev):
    """Single-ant BC -> (F, T) in env units, from ONE ant's row."""
    vel_n = VEL_SCALE_ANTS if VEL_SCALE_ANTS is not None else n
    with torch.no_grad():
        a = teacher(torch.tensor(single_ant_obs(obs_rows, vel_n)[None, :],
                                 device=dev))[0, 0].cpu().numpy()
    ang, mag, spin = float(a[0]), float(np.clip(a[1], 0, 1)), float(np.clip(a[2], -1, 1))
    return push * mag * np.array([math.cos(ang), math.sin(ang)]), spin * spin_strength


def collect_probe_data(cfg_path, teacher, ant_counts, episodes, history, seed,
                       cap_per_n, dev):
    X, Y = [], []
    for n in ant_counts:
        cfg = cfg_load(cfg_path); cfg.ants.n = n
        env = AntSwarmEnv(config=cfg, seed=seed)
        s = float(cfg.scene_scale); push = float(cfg.physics.push_strength) * s
        spin_strength = push * float(cfg.tshape.stem_len) * s / 2
        offs = env.attachment_offsets
        Xn = []
        for ep in range(episodes):
            env.reset(seed=seed + ep); hist = AntHistory(n, OBS, history)
            done, k = False, 0
            while not done and k < 500:
                obs = env.obs_model.observe(env.state)
                Xn.append(hist.push(obs).copy())
                F, T = teacher_wrench(teacher, obs, n, push, spin_strength, dev)
                f, _ = split_wrench(F, T, offs @ env.state.obj.rot().T, push)
                act = np.stack([np.arctan2(f[:, 1], f[:, 0]),
                                np.clip(np.linalg.norm(f, axis=1) / push, 0, 1)], 1)
                _, _, tm, tr, _ = env.step(act.astype(np.float32)); k += 1; done = tm or tr
        env.close()
        Xn = np.concatenate(Xn)
        if len(Xn) > cap_per_n:
            Xn = Xn[np.random.default_rng(n).choice(len(Xn), cap_per_n, replace=False)]
        X.append(Xn); Y.append(np.full(len(Xn), math.log2(n), dtype=np.float32))
        logger.info(f"  probe data n={n}: {len(Xn)} samples")
    return np.concatenate(X).astype(np.float32), np.concatenate(Y)


def base_action(obs, n_est, teacher, push, spin_strength, dev, stem_half, cap_half,
                c_local, spread, stem_end_spread):
    """One step of the decentralised base. n_est: per-ant array of assumed N."""
    n = len(obs)
    act = np.zeros((n, 2), dtype=np.float32)
    cache = {}
    for i in range(n):
        ne = float(n_est[i])
        if ne not in cache:                       # identical rows -> identical (F,T)
            cache[ne] = teacher_wrench(teacher, obs[i:i + 1], ne, push, spin_strength, dev)
        F, T = cache[ne]
        a_i = own_arm(obs[i], stem_half, cap_half)
        if round(ne) == 2:
            cl, sp = np.zeros(2), stem_end_spread
        else:
            cl, sp = c_local, spread
        abar = rotate(cl, obs[i]); ac = a_i - abar
        lam = (T - (abar[0] * F[1] - abar[1] * F[0])) / max(ne * sp, 1e-9)
        f = F / ne + lam * np.array([-ac[1], ac[0]])
        act[i, 0] = math.atan2(f[1], f[0])
        act[i, 1] = min(np.linalg.norm(f) / push, 1.0)
    return act


def collect_probe_data_randomized(cfg_path, teacher, ant_counts, episodes, history,
                                  seed, cap_per_n, dev, geom, n_range=(1.0, 200.0)):
    """Rollouts where the ants ASSUME a random N (log-uniform), re-drawn every 25
    steps. The probe then sees jammed and mis-driven loads, and its previous
    assumption is recorded as the feedback input. Label: the true N."""
    stem_half, cap_half, c_local, spread, stem_end_spread = geom
    rng = np.random.default_rng(seed + 7)
    X, Y = [], []
    for n in ant_counts:
        cfg = cfg_load(cfg_path); cfg.ants.n = n
        env = AntSwarmEnv(config=cfg, seed=seed)
        s_ = float(cfg.scene_scale); push = float(cfg.physics.push_strength) * s_
        spin_strength = push * float(cfg.tshape.stem_len) * s_ / 2
        Xn = []
        for ep in range(episodes):
            env.reset(seed=seed + ep); hist = AntHistory(n, OBS, history)
            done, k = False, 0
            n_ass = float(np.exp(rng.uniform(np.log(n_range[0]), np.log(n_range[1]))))
            while not done and k < 500:
                if k % 25 == 0 and k > 0 and rng.random() < 0.5:
                    n_ass = float(np.exp(rng.uniform(np.log(n_range[0]), np.log(n_range[1]))))
                obs = env.obs_model.observe(env.state)
                feats = hist.push(obs)
                Xn.append(np.concatenate([feats, np.full((n, 1), math.log2(n_ass),
                                                        dtype=np.float32)], 1))
                act = base_action(obs, np.full(n, n_ass), teacher, push, spin_strength,
                                  dev, stem_half, cap_half, c_local, spread, stem_end_spread)
                _, _, tm, tr, _ = env.step(act); k += 1; done = tm or tr
        env.close()
        Xn = np.concatenate(Xn)
        if len(Xn) > cap_per_n:
            Xn = Xn[np.random.default_rng(n).choice(len(Xn), cap_per_n, replace=False)]
        X.append(Xn); Y.append(np.full(len(Xn), math.log2(n), dtype=np.float32))
        logger.info(f"  probe data (randomised N) n={n}: {len(Xn)} samples")
    return np.concatenate(X).astype(np.float32), np.concatenate(Y)


def train_probe(X, Y, history, epochs, dev, feedback=False):
    net = NProbe(history, feedback=feedback).to(dev)
    net.mu.copy_(torch.tensor(X.mean(0))); net.sd.copy_(torch.tensor(X.std(0) + 1e-6))
    Xt, Yt = torch.tensor(X, device=dev), torch.tensor(Y, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(Xt)
    idx = torch.randperm(n, device=dev); cut = int(0.9 * n)
    tr, te = idx[:cut], idx[cut:]
    for ep in range(epochs):
        perm = tr[torch.randperm(len(tr), device=dev)]
        for k in range(0, len(perm) - 2048, 2048):
            i = perm[k:k + 2048]
            loss = nn.functional.mse_loss(net(Xt[i]), Yt[i])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(Xt[te]); y = Yt[te]
        r2 = 1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        err = (pred - y).abs().mean()
    logger.info(f"N-probe (history={history}, feedback={feedback}): held-out R2={r2:.3f}, "
                f"mean |log2 N| error={err:.3f}")
    return net, float(r2), float(err)


# --------------------------------------------------------------------------- #
# the decentralised base controller
# --------------------------------------------------------------------------- #
def attachment_stats(cfg, samples=20000):
    """(centroid, E|o - centroid|^2) of the random-perimeter rule, local frame."""
    tsh = TShape(cfg)
    offs = make_attachment_offsets(tsh, samples, np.random.default_rng(0))
    c = offs.mean(0)
    return c.astype(np.float64), float(((offs - c) ** 2).sum(1).mean())


def rotate(v, row):
    """Local -> world using the ant's own sin/cos entries."""
    sin, cos = row[6], row[7]
    return np.array([cos * v[0] - sin * v[1], sin * v[0] + cos * v[1]])


def own_arm(row, stem_half, cap_half):
    return rotate(np.array([row[0] * stem_half, row[1] * cap_half]), row)


@torch.no_grad()
def run_level(level, n, cfg_path, teacher, probe, history, episodes, seed, dev,
              stem_half, cap_half, centroid_local, spread, stem_end_spread,
              n_prior=16.0):
    cfg = cfg_load(cfg_path); cfg.ants.n = n
    env = AntSwarmEnv(config=cfg, seed=seed)
    s = float(cfg.scene_scale); push = float(cfg.physics.push_strength) * s
    spin_strength = push * float(cfg.tshape.stem_len) * s / 2
    reach = float(cfg.goal.reach_radius) * s
    offs = env.attachment_offsets.astype(np.float64)
    true_centroid_local = offs.mean(0)
    true_spread = float(((offs - true_centroid_local) ** 2).sum(1).mean())
    dists, succ, n_est_log = [], [], []

    for ep in range(episodes):
        env.reset(seed=seed + ep); hist = AntHistory(n, OBS, history)
        info, done, k = {}, False, 0
        n_est = np.full(n, float(n_prior))          # each ant's own running estimate
        while not done and k < 500:
            obs = env.obs_model.observe(env.state)
            feats = hist.push(obs)
            if level == 2 and FIXED_N is not None:
                n_est = np.full(n, float(FIXED_N))     # no counter: a constant guess
                n_est_log.append(float(FIXED_N))
            elif level == 2:
                x = torch.tensor(feats, device=dev)
                if probe.feedback:
                    x = torch.cat([x, torch.tensor(np.log2(n_est)[:, None],
                                                   dtype=torch.float32, device=dev)], 1)
                n_est = np.clip(2.0 ** probe(x).cpu().numpy(), 1.0, 400.0)
                n_est_log.append(float(np.median(n_est)))
            else:
                n_est = np.full(n, float(n))
            if level == 0:
                act = np.zeros((n, 2), dtype=np.float32)
                F, T = teacher_wrench(teacher, obs[0:1], float(n), push, spin_strength, dev)
                for i in range(n):
                    a_i = own_arm(obs[i], stem_half, cap_half)
                    abar = rotate(true_centroid_local, obs[i]); ac = a_i - abar
                    lam = (T - (abar[0] * F[1] - abar[1] * F[0])) / max(n * true_spread, 1e-9)
                    f = F / n + lam * np.array([-ac[1], ac[0]])
                    act[i, 0] = math.atan2(f[1], f[0]); act[i, 1] = min(np.linalg.norm(f) / push, 1.0)
            else:
                act = base_action(obs, n_est, teacher, push, spin_strength, dev,
                                  stem_half, cap_half, centroid_local, spread, stem_end_spread)
            _, _, tm, tr, info = env.step(act); k += 1; done = tm or tr

        d = float(info.get("object_distance", env.state.distance_to_goal()))
        dists.append(d); succ.append(bool(info.get("is_success", False) or d < reach))
    env.close()
    out = dict(level=level, ants=n, success_rate_pct=float(np.mean(succ) * 100),
               mean_distance_m=float(np.mean(dists)))
    if n_est_log:
        out["n_est_median"] = float(np.median(n_est_log))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/il/il_augmented_bc.yaml")
    p.add_argument("--teacher",
                   default="storage_local/ant__20260905_1322__240805__train_goal_bc/bc_goal_fixed.pt")
    p.add_argument("--probe-ants", default="2,3,4,6,8,12,16,24,32,48,64,100")
    p.add_argument("--probe-episodes", type=int, default=15)
    p.add_argument("--probe-cap", type=int, default=30000, help="samples per N")
    p.add_argument("--probe-epochs", type=int, default=40)
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--probe-mode", default="randomized", choices=["exact", "randomized"],
                   help="exact = trained on correctly-driven loads only (the first run); "
                        "randomized = ants assume a random N during collection")
    p.add_argument("--feedback", action="store_true", default=True,
                   help="probe also gets the ant's own previous estimate")
    p.add_argument("--no-feedback", dest="feedback", action="store_false")
    p.add_argument("--n-prior", type=float, default=16.0, help="initial guess")
    p.add_argument("--fixed-n", type=float, default=None,
                   help="level 2: assume this N for every ant, no counter")
    p.add_argument("--velocity-scale-ants", type=int, default=None,
                   help="set env.velocity_scale_ants (e.g. 1 = absolute velocity)")
    p.add_argument("--test-ants", default="2,4,10,50,100")
    p.add_argument("--levels", default="0,1,2")
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=30000)
    p.add_argument("--eval-seed", type=int, default=40000, help="unseen by the probe")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    global VEL_SCALE_ANTS, FIXED_N
    VEL_SCALE_ANTS = args.velocity_scale_ants
    FIXED_N = args.fixed_n
    dev = torch.device(resolve_device(args.device))
    out = Path(args.out) if args.out else Path("storage_local") / build_run_id(
        f"decentralised_base_{args.probe_mode}{'_fb' if args.feedback else ''}"
        f"{'_absvel' if args.velocity_scale_ants else ''}"
        f"{f'_fixedN{int(args.fixed_n)}' if args.fixed_n else ''}_h{args.history}")
    out.mkdir(parents=True, exist_ok=True)
    logger.add(out / "train.log", level="INFO")

    blob = torch.load(args.teacher, map_location=dev, weights_only=True)
    if blob.get("goal_observation_version") != 2:
        raise SystemExit("teacher predates the goal fix")
    teacher = ChunkPolicy(OBS, 1).to(dev); teacher.load_state_dict(blob["state_dict"]); teacher.eval()

    cfg = cfg_load(args.config)
    s = float(cfg.scene_scale)
    stem_half = float(cfg.tshape.stem_len) * s / 2
    cap_half = max(float(cfg.tshape.cap_big_len), float(cfg.tshape.cap_small_len)) * s / 2
    centroid_local, spread = attachment_stats(cfg)
    stem_end_spread = stem_half ** 2
    logger.info(f"attachment rule: centroid={centroid_local.round(4)}  "
                f"E|o-c|^2={spread:.5f}   (n=2 stem ends: {stem_end_spread:.5f})")

    probe_ants = [int(x) for x in args.probe_ants.split(",")]
    t0 = time.time()
    geom = (stem_half, cap_half, centroid_local, spread, stem_end_spread)
    if FIXED_N is not None:
        probe, r2, err = NProbe(args.history), float("nan"), float("nan")
        args.feedback = False
    elif args.probe_mode == "randomized":
        X, Y = collect_probe_data_randomized(args.config, teacher, probe_ants,
                                             args.probe_episodes, args.history,
                                             args.seed, args.probe_cap, dev, geom)
        if not args.feedback:
            X = X[:, :-1]
    else:
        X, Y = collect_probe_data(args.config, teacher, probe_ants, args.probe_episodes,
                                  args.history, args.seed, args.probe_cap, dev)
        args.feedback = False
    if FIXED_N is None:
        probe, r2, err = train_probe(X, Y, args.history, args.probe_epochs, dev,
                                     feedback=args.feedback)
    probe.eval()
    torch.save({"state_dict": probe.state_dict(), "history": args.history,
                "feedback": args.feedback, "mode": args.probe_mode,
                "r2": r2, "err": err}, out / "n_probe.pt")
    logger.info(f"probe ready ({time.time()-t0:.0f}s)")

    rows = []
    for level in [int(x) for x in args.levels.split(",")]:
        for n in [int(x) for x in args.test_ants.split(",")]:
            r = run_level(level, n, args.config, teacher, probe, args.history,
                          args.eval_episodes, args.eval_seed, dev,
                          stem_half, cap_half, centroid_local, spread, stem_end_spread,
                          n_prior=args.n_prior)
            rows.append(r)
            extra = f"  N_est~{r['n_est_median']:.1f}" if "n_est_median" in r else ""
            logger.info(f"  level {level}  n={n:>3}: SR={r['success_rate_pct']:.1f}%  "
                        f"mean={r['mean_distance_m']:.4f}{extra}")
    (out / "results.json").write_text(json.dumps(
        {"args": vars(args), "probe": {"r2": r2, "err": err}, "results": rows}, indent=2) + "\n")

    names = {0: "true N, true geometry", 1: "true N, geom. constants", 2: "est. N, geom. constants"}
    print("\n" + "=" * 70)
    print(f"DECENTRALISED BASE CONTROLLER  ({args.eval_episodes} episodes per cell, "
          f"probe={args.probe_mode}{'+feedback' if args.feedback else ''}, R2={r2:.3f})")
    print("=" * 70)
    print(f"{'level':<6} {'what each ant knows':<24} {'ants':>5} {'SR %':>7} {'mean dist':>10} {'N_est':>7}")
    print("-" * 70)
    for r in rows:
        ne = f"{r['n_est_median']:.1f}" if "n_est_median" in r else "-"
        print(f"{r['level']:<6} {names[r['level']]:<24} {r['ants']:>5} "
              f"{r['success_rate_pct']:>7.1f} {r['mean_distance_m']:>10.4f} {ne:>7}")
    print("\nlevel 2 is the fully decentralised base: own row only, no N input.\n")


if __name__ == "__main__":
    main()
