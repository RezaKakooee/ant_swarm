"""Is the torque sign predictable from the observation?

Diagnosis for the Slit-2 jamming problem.  If the expert's recovery torque is
strictly bimodal (+1 vs -1) but the 27-D observation cannot tell the two apart,
then MSE regression *must* collapse to ~0 torque and no loss function, action
chunk or diffusion head can fix it -- the observation is simply missing the cue.

The script replays demonstrations to regenerate observations, keeps the steps
near a slit, labels them by sign(spin torque), and trains small classifiers on
several feature sets.  A high AUC means the cue is there; a chance-level AUC
means it is not.

Feature sets compared:
    base        obs[0:11]   pose / goal / velocity block
    barrier     obs[11:27]  16 unsigned tip->wall-head distances
    obs         obs[0:27]   what the policy actually sees today
    obs+route   obs + the per-episode route label (up / down)
    signed      32 signed (dx, dy) tip->wall-head offsets  (privileged)
    obs+signed  obs + the signed offsets

Usage:
    python scripts/il/test_torque_observability.py --episodes 4000
"""
from __future__ import annotations

import argparse
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

FEATURE_SETS = ["base", "barrier", "obs", "obs+route", "signed", "obs+signed"]


# --------------------------------------------------------------------------- #
# 1. replay demonstrations -> observations
# --------------------------------------------------------------------------- #
def collect(cfg, dataset_path, n_episodes, seed=0, sample="spread"):
    """Replay episodes and return per-step obs, actions and side channels."""
    z = np.load(dataset_path)
    offsets, actions_flat = z["offsets"], z["actions"]
    init_poses, goals, routes = z["init_pose"], z["goal"], z["route"]

    cfg = cfg.copy()
    cfg.env.reward_mode = "sparse"
    env = AntSwarmEnv(config=cfg, seed=seed)

    tip_local = env.obs_model.tip_local           # (4, 2) T arm tips, local frame
    heads = env.layout.wall_heads                 # (4, 2) gap-facing wall corners
    total = len(offsets) - 1
    n_eps = min(total, n_episodes)
    # the dataset is stored run by run, so the first N episodes all come from
    # one source run / one route -- spread the sample over the whole file
    if sample == "spread":
        ep_ids = np.linspace(0, total - 1, n_eps).astype(np.int64)
    elif sample == "random":
        ep_ids = np.sort(np.random.default_rng(seed).choice(total, n_eps, replace=False))
    else:
        ep_ids = np.arange(n_eps, dtype=np.int64)
    logger.info(f"Replaying {n_eps}/{total} demonstration episodes (sample={sample}) ...")

    obs_l, act_l, signed_l, route_l, epid_l, cx_l = [], [], [], [], [], []
    t0 = time.time()
    kept = 0

    for step_i, i in enumerate(ep_ids):
        i = int(i)
        actions = actions_flat[offsets[i]:offsets[i + 1]]
        center = np.asarray(init_poses[i][:2], dtype=np.float32)
        angle = float(init_poses[i][2])

        env.reset()
        env.layout.goal = np.asarray(goals[i], dtype=np.float32)
        env.state.reset(center, angle)
        if env.state._bad_pose():
            continue
        env.init_center, env.init_angle = center.copy(), angle
        env.reward_model.reset(env.state)
        kept += 1

        for raw_action in actions:
            obj = env.state.obj
            obs = flatten(env.observation_space,
                          env.obs_model.observe(env.state)).astype(np.float32)

            # privileged: signed (dx, dy) from each arm tip to each wall head
            tips = obj.center[None, :] + tip_local @ obj.rot().T        # (4, 2)
            signed = (tips[:, None, :] - heads[None, :, :]).reshape(-1)  # (32,)

            obs_l.append(obs)
            act_l.append(np.asarray(raw_action, dtype=np.float32).ravel())
            signed_l.append(signed.astype(np.float32))
            route_l.append(routes[i])
            epid_l.append(i)
            cx_l.append(float(obj.center[0]))

            _, _, terminated, truncated, _ = env.step(
                np.asarray(raw_action, dtype=np.float32).reshape(env.action_space.shape))
            if terminated or truncated:
                break

        if (step_i + 1) % 1000 == 0 or (step_i + 1) == n_eps:
            logger.info(f"  {step_i+1}/{n_eps} episodes, {len(obs_l)} steps, "
                        f"{time.time()-t0:.0f}s")

    env.close()
    logger.info(f"Replay done: {kept} usable episodes, {len(obs_l)} transitions.")
    return {
        "obs": np.asarray(obs_l, dtype=np.float32),
        "act": np.asarray(act_l, dtype=np.float32),
        "signed": np.asarray(signed_l, dtype=np.float32),
        "route": np.asarray(route_l, dtype=np.float32).reshape(-1, 1),
        "epid": np.asarray(epid_l, dtype=np.int64),
        "cx": np.asarray(cx_l, dtype=np.float32),
        "tip_local": tip_local,
        "heads": heads,
    }


# --------------------------------------------------------------------------- #
# 2. subset selection
# --------------------------------------------------------------------------- #
def min_tip_head_gap(data, wall_x, tol=1e-3):
    """Smallest unsigned tip->head distance to the two heads of one column."""
    cols = np.abs(data["heads"][:, 0] - wall_x) < tol
    d = data["signed"].reshape(-1, 4, 4, 2)[:, :, cols, :]   # (N, tips, heads, 2)
    return np.linalg.norm(d, axis=-1).reshape(len(d), -1).min(axis=1)


def build_subsets(data, wall_x, margin, tight):
    near = np.abs(data["cx"] - wall_x) < margin
    gap = min_tip_head_gap(data, wall_x)
    return {
        f"near slit (|dx|<{margin:.2f} m)": near,
        f"tight (near + tip-head<{tight:.3f} m)": near & (gap < tight),
    }


# --------------------------------------------------------------------------- #
# 3. classifier
# --------------------------------------------------------------------------- #
def auc_score(y, score):
    """Rank-based ROC AUC (no sklearn needed)."""
    y = np.asarray(y, dtype=np.int64)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks over ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fit_classifier(Xtr, ytr, Xte, yte, epochs, device, seed=0, hidden=128):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    dev = torch.device(device)
    Xtr_t = torch.tensor(Xtr, device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev)
    Xte_t = torch.tensor(Xte, device=dev)

    net = nn.Sequential(
        nn.Linear(Xtr.shape[1], hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    # the two torque signs are usually very imbalanced -- reweight so the
    # net cannot win by always predicting the majority sign
    n_pos = float(ytr.sum())
    pos_w = torch.tensor([(len(ytr) - n_pos) / max(n_pos, 1.0)], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    n, bs = len(Xtr_t), 4096
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for k in range(0, n, bs):
            idx = perm[k:k + bs]
            opt.zero_grad()
            loss = lossf(net(Xtr_t[idx]).squeeze(-1), ytr_t[idx])
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        logits = net(Xte_t).squeeze(-1).cpu().numpy()
    acc = float(((logits > 0).astype(np.int64) == yte).mean())
    return auc_score(yte, logits), acc


def features(data, name, mask):
    obs, signed, route = data["obs"][mask], data["signed"][mask], data["route"][mask]
    return {
        "base": obs[:, :11],
        "barrier": obs[:, 11:27],
        "obs": obs,
        "obs+route": np.concatenate([obs, route], axis=1),
        "signed": signed,
        "obs+signed": np.concatenate([obs, signed], axis=1),
    }[name]


# --------------------------------------------------------------------------- #
# 4. main
# --------------------------------------------------------------------------- #
def run_subset(data, mask, label, deadband, epochs, device, test_frac=0.2, seed=0):
    torque = data["act"][:, 2]
    live = mask & (np.abs(torque) > deadband)
    n = int(live.sum())
    if n < 2000:
        logger.warning(f"[{label}] only {n} usable steps -- results will be noisy.")
    if n == 0:
        return None

    y = (torque[live] > 0).astype(np.int64)
    eps = data["epid"][live]
    uniq = np.unique(eps)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    test_eps = set(uniq[:max(1, int(len(uniq) * test_frac))].tolist())
    is_test = np.array([e in test_eps for e in eps])

    majority = float(max(y.mean(), 1 - y.mean()))
    logger.info(f"[{label}] {n} steps ({100*n/max(1,int(mask.sum())):.0f}% of subset "
                f"survive the deadband), {len(uniq)} episodes, "
                f"pos={y.mean():.3f}, majority baseline={majority:.3f}")

    rows = []
    for fs in FEATURE_SETS:
        X = features(data, fs, live)
        auc, acc = fit_classifier(X[~is_test], y[~is_test], X[is_test], y[is_test],
                                  epochs, device, seed=seed)
        rows.append((fs, X.shape[1], auc, acc))
        logger.info(f"    {fs:<12} dim={X.shape[1]:<3} AUC={auc:.3f} acc={acc:.3f}")
    return {"label": label, "n": n, "majority": majority, "rows": rows}


def report(results, deadband):
    print("\n" + "=" * 74)
    print("TORQUE-SIGN OBSERVABILITY  (test episodes held out, deadband "
          f"|torque|>{deadband})")
    print("=" * 74)
    for r in results:
        if r is None:
            continue
        print(f"\n{r['label']}   n={r['n']}   majority baseline acc={r['majority']:.3f}")
        print(f"  {'feature set':<12} {'dim':>4} {'AUC':>7} {'acc':>7}")
        print("  " + "-" * 32)
        for fs, dim, auc, acc in r["rows"]:
            print(f"  {fs:<12} {dim:>4} {auc:>7.3f} {acc:>7.3f}")
    print("\nHow to read it (look at the 'obs' row, tight subset):")
    print("  AUC ~0.50-0.65  observation cannot tell top-touch from bottom-touch.")
    print("                  -> fix the observation first; diffusion/ACT alone")
    print("                     will only pick a random sign.")
    print("  AUC >0.80       the cue is there; the MSE loss is the problem.")
    print("                  -> go to discrete torque bins / mixture head / diffusion.")
    print("  If 'signed' beats 'obs' by a lot, replace the 16 unsigned tip-head")
    print("  distances with the 32 signed (dx, dy) offsets.")
    print("  If 'obs+route' beats 'obs' by a lot, the demos mix two incompatible")
    print("  strategies that the policy cannot tell apart.\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/il/il_augmented_bc.yaml")
    p.add_argument("--dataset", default="storage_local/datasets/successes_v1/dataset.npz")
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--sample", default="spread", choices=["spread", "random", "head"],
                   help="which episodes to replay; 'head' takes the first N "
                        "(one source run only)")
    p.add_argument("--slit", type=int, default=2, choices=[1, 2],
                   help="which barrier column to analyse")
    p.add_argument("--margin", type=float, default=0.12,
                   help="|obj_x - wall_x| window, metres")
    p.add_argument("--tight", type=float, default=0.05,
                   help="min tip->wall-head distance for the 'tight' subset, metres")
    p.add_argument("--deadband", type=float, default=0.05,
                   help="ignore steps with |torque| below this (sign is meaningless)")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--device", default="auto")
    p.add_argument("--cache", default="", help="optional .npz path to cache the replay")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}")

    cfg = load_config(args.config)
    wall_x = float(cfg.walls.x_columns[args.slit - 1]) * float(cfg.scene_scale)
    logger.info(f"Slit {args.slit} at x={wall_x:.3f} m")

    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists():
        logger.info(f"Loading cached replay from {cache}")
        z = np.load(cache)
        data = {k: z[k] for k in z.files}
    else:
        data = collect(cfg, args.dataset, args.episodes, sample=args.sample)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **data)
            logger.info(f"Cached replay -> {cache}")

    routes, counts = np.unique(data["route"], return_counts=True)
    logger.info("route mix (0=down, 1=up): "
                + ", ".join(f"{int(r)}:{c}" for r, c in zip(routes, counts)))
    subsets = build_subsets(data, wall_x, args.margin, args.tight)
    results = [run_subset(data, m, lab, args.deadband, args.epochs, device)
               for lab, m in subsets.items()]
    report(results, args.deadband)


if __name__ == "__main__":
    main()
