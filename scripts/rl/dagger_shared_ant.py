"""Step 1 of MARL-without-a-formula: DAgger a shared per-ant policy at fixed N.

Each ant runs the SAME network on ITS OWN observation row and outputs its own
force vector. No N input, no formula, no communication at execution.

The teacher is the split controller (single-ant BC + least-norm split), used
ONLY as a labeller during training: at any state it can say what each ant
should push. DAgger loop:

    it 0     roll out the teacher, record (own row -> own force)
    it 1..K  roll out the STUDENT, label every visited state with the teacher,
             aggregate, retrain from scratch on everything

That addresses the covariate shift plain BC cannot: the student is trained on
the states it actually reaches.

Why a single fixed N: the earlier shared-policy runs regressed forces across
n = 2..32 at once, and the 1/N scaling of the targets is what broke the summed
torque (chapter 04 §4). At one N the target is a fixed function of the ant's
row.

Attachment points are fixed per env seed. Evaluation reports the training
layout AND unseen layouts (other seeds), so overfitting to positions shows.

    python scripts/rl/dagger_shared_ant.py --ants 5 --iters 6
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
from ant_swarm.tracking import Tracker                          # noqa: E402
from train_chunked_bc import ChunkPolicy                         # noqa: E402
from test_multiagent_split import single_ant_obs, split_wrench   # noqa: E402
from train_shared_ant_policy import SharedAntPolicy, AntHistory  # noqa: E402

OBS = 27          # replaced at runtime by the env's obs_dim (29 with the contact-velocity block)
STUDENT_IN = 27   # what the single-ant teacher consumes


class Teacher:
    """Split controller: labels every ant at any state. Training-time only."""

    def __init__(self, ckpt, cfg, dev):
        blob = torch.load(ckpt, map_location=dev, weights_only=True)
        if blob.get("goal_observation_version") != 2:
            raise SystemExit("teacher predates the goal fix")
        self.net = ChunkPolicy(OBS, 1).to(dev); self.net.load_state_dict(blob["state_dict"]); self.net.eval()
        s = float(cfg.scene_scale)
        self.push = float(cfg.physics.push_strength) * s
        self.spin = self.push * float(cfg.tshape.stem_len) * s / 2
        self.dev = dev

    @torch.no_grad()
    def forces(self, env, obs):
        """(n, 2) forces in units of push_strength."""
        n = len(obs)
        # the single-ant BC was trained on 27-D rows at the n=1 velocity scale. If the env
        # emits the contact-velocity block (29-D) drop it; if it normalises velocity by
        # the live ant count, undo that (x n); with velocity_scale_ants set, do not.
        o = obs if obs.shape[1] == STUDENT_IN else np.concatenate([obs[:, :9], obs[:, 9:11], obs[:, 13:]], 1)
        vel_n = 1 if getattr(env.cfg.env, "velocity_scale_ants", None) else n
        a = self.net(torch.tensor(single_ant_obs(o, vel_n)[None, :], device=self.dev))[0, 0].cpu().numpy()
        ang, mag, spin = float(a[0]), float(np.clip(a[1], 0, 1)), float(np.clip(a[2], -1, 1))
        F = self.push * mag * np.array([math.cos(ang), math.sin(ang)])
        arms = env.attachment_offsets @ env.state.obj.rot().T
        f, _ = split_wrench(F, spin * self.spin, arms, self.push)
        return (f / self.push).astype(np.float32)


def to_action(f):
    return np.stack([np.arctan2(f[:, 1], f[:, 0]),
                     np.clip(np.linalg.norm(f, axis=1), 0.0, 1.0)], 1).astype(np.float32)


@torch.no_grad()
def rollout(env, actor, teacher, history, seed, max_steps, dev, record=True):
    """actor: 'teacher' or a SharedAntPolicy. Returns (X, Y, success, dist)."""
    n = env.obs_model.n_ants
    reach = float(env.layout.reach_radius)
    env.reset(seed=seed); hist = AntHistory(n, OBS, history)
    X, Y, info, done, k = [], [], {}, False, 0
    while not done and k < max_steps:
        obs = env.obs_model.observe(env.state); feats = hist.push(obs)
        f_teacher = teacher.forces(env, obs)
        if record:
            X.append(feats.copy()); Y.append(f_teacher.copy())
        if actor == "teacher":
            f = f_teacher
        else:
            f = actor(torch.tensor(feats, device=dev)).cpu().numpy()
        _, _, tm, tr, info = env.step(to_action(f)); k += 1; done = tm or tr
    d = float(info.get("object_distance", env.state.distance_to_goal()))
    ok = bool(info.get("is_success", False) or d < reach)
    return X, Y, ok, d


def fit(X, Y, history, epochs, dev, seed=0, batch=4096, lr=1e-3):
    torch.manual_seed(seed)
    net = SharedAntPolicy(OBS, history).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt, Yt = torch.tensor(X, device=dev), torch.tensor(Y, device=dev)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev); tot, nb = 0.0, 0
        for k in range(0, n - batch + 1, batch):
            i = perm[k:k + batch]
            loss = nn.functional.mse_loss(net(Xt[i]), Yt[i])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        sched.step()
    net.eval()
    return net, tot / max(nb, 1)


def evaluate(net, cfg, teacher, history, episodes, layout_seed, ep_seed, dev):
    env = AntSwarmEnv(config=cfg, seed=layout_seed)       # layout_seed fixes attachments
    succ, dists = [], []
    for e in range(episodes):
        _, _, ok, d = rollout(env, net, teacher, history, ep_seed + e, 500, dev, record=False)
        succ.append(ok); dists.append(d)
    env.close()
    return float(np.mean(succ) * 100), float(np.mean(dists))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/il/il_augmented_bc.yaml")
    p.add_argument("--teacher", default="storage_local/ant__20260905_1322__240805__train_goal_bc/bc_goal_fixed.pt")
    p.add_argument("--ants", type=int, default=5)
    p.add_argument("--history", type=int, default=1)
    p.add_argument("--iters", type=int, default=6, help="DAgger iterations after the teacher pass")
    p.add_argument("--episodes", type=int, default=40, help="episodes collected per iteration")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--heldout-episodes", type=int, default=100,
                   help="episodes for the final held-out score of the chosen checkpoint")
    p.add_argument("--layout-seed", type=int, default=0, help="attachment layout used for training")
    p.add_argument("--unseen-layouts", default="1,2", help="other attachment layouts to test")
    p.add_argument("--seed", type=int, default=30000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    p.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    args = p.parse_args()

    dev = torch.device(resolve_device(args.device))
    out = Path(args.out) if args.out else Path("storage_local") / build_run_id(f"dagger_shared_n{args.ants}_h{args.history}")
    out.mkdir(parents=True, exist_ok=True); logger.add(out / "train.log", level="INFO")
    tr = Tracker(name=out.name, group="dagger_shared_ant",
                 tags=["marl", "dagger", f"n{args.ants}", f"h{args.history}"],
                 config=vars(args), enabled=args.wandb)

    cfg = load_config(args.config).copy()
    if getattr(cfg.ants, "offsets", None) is not None:
        args.ants = int(cfg.ants.n)          # the config's layout decides n
    cfg.ants.n = args.ants
    if args.ants < 2:
        raise SystemExit("needs n>=2: a single push-only ant at the stem centre has no torque")
    teacher = Teacher(args.teacher, cfg, dev)
    env = AntSwarmEnv(config=cfg, seed=args.layout_seed)
    global OBS
    OBS = int(env.obs_model.obs_dim)
    logger.info(f"student rows: {OBS}-D")
    logger.info(f"n={args.ants} history={args.history} layout_seed={args.layout_seed} device={dev}")
    logger.info(f"attachments: {np.round(env.attachment_offsets, 4).tolist()}")

    X, Y, log, t0 = [], [], [], time.time()
    # iteration 0: the teacher drives
    ok_t = []
    for e in range(args.episodes):
        x, y, ok, _ = rollout(env, "teacher", teacher, args.history, args.seed + e, 500, dev)
        X += x; Y += y; ok_t.append(ok)
    logger.info(f"it 0 (teacher): SR={100*np.mean(ok_t):.1f}%  {sum(len(x) for x in X)} ant-samples")
    net, mse = fit(np.concatenate(X), np.concatenate(Y), args.history, args.epochs, dev)
    sr, md = evaluate(net, cfg, teacher, args.history, args.eval_episodes, args.layout_seed, args.seed + 10_000, dev)
    log.append(dict(it=0, samples=int(sum(len(x) for x in X)), mse=mse, sr=sr, dist=md))
    tr.log({"dagger/student_sr": sr, "dagger/mean_dist": md, "dagger/mse": mse,
            "dagger/samples": log[-1]["samples"], "dagger/teacher_sr": float(100*np.mean(ok_t))}, step=0)
    logger.info(f"  after fit: MSE={mse:.5f}  student SR={sr:.1f}%  mean={md:.4f}  ({time.time()-t0:.0f}s)")

    # DAgger: the student drives, the teacher labels
    for it in range(1, args.iters + 1):
        ok_s = []
        for e in range(args.episodes):
            x, y, ok, _ = rollout(env, net, teacher, args.history, args.seed + 1000 * it + e, 500, dev)
            X += x; Y += y; ok_s.append(ok)
        net, mse = fit(np.concatenate(X), np.concatenate(Y), args.history, args.epochs, dev, seed=it)
        # a FRESH eval block every round: selecting the best round on one fixed block
        # of 50 episodes inflated the reported number (98% chosen vs 89-93% held out)
        sr, md = evaluate(net, cfg, teacher, args.history, args.eval_episodes, args.layout_seed,
                          args.seed + 10_000 + 1_000 * it, dev)
        log.append(dict(it=it, samples=int(sum(len(x) for x in X)), mse=mse, sr=sr, dist=md,
                        collect_sr=float(100 * np.mean(ok_s))))
        tr.log({"dagger/student_sr": sr, "dagger/mean_dist": md, "dagger/mse": mse,
                "dagger/samples": log[-1]["samples"], "dagger/collect_sr": log[-1]["collect_sr"]}, step=it)
        logger.info(f"it {it}: collected at SR={100*np.mean(ok_s):.1f}%  -> {log[-1]['samples']} samples  "
                    f"MSE={mse:.5f}  student SR={sr:.1f}%  mean={md:.4f}  ({time.time()-t0:.0f}s)")
        torch.save({"state_dict": net.state_dict(), "history": args.history, "ants": args.ants,
                    "layout_seed": args.layout_seed, "it": it}, out / "shared_ant_policy_last.pt")
        if sr >= max((r["sr"] for r in log[:-1]), default=-1.0):
            torch.save({"state_dict": net.state_dict(), "history": args.history, "ants": args.ants,
                        "layout_seed": args.layout_seed, "it": it, "sr": sr}, out / "shared_ant_policy.pt")
            logger.info(f"  best so far ({sr:.1f}%) -> shared_ant_policy.pt")
    env.close()

    # headline number: the CHOSEN checkpoint on a held-out block it never touched,
    # and the teacher on the same block, so the comparison is fair
    best = torch.load(out / "shared_ant_policy.pt", map_location=dev, weights_only=True)
    net.load_state_dict(best["state_dict"]); net.eval()
    ho_seed, ho_n = args.seed + 90_000, int(args.heldout_episodes)
    ho_sr, ho_md = evaluate(net, cfg, teacher, args.history, ho_n, args.layout_seed, ho_seed, dev)
    env_ho = AntSwarmEnv(config=cfg, seed=args.layout_seed)
    ho_teacher = float(100 * np.mean([rollout(env_ho, "teacher", teacher, args.history, ho_seed + e, 500, dev, record=False)[2]
                                      for e in range(ho_n)]))
    env_ho.close()
    logger.info(f"HELD-OUT ({ho_n} eps, seeds {ho_seed}+): best-round student SR={ho_sr:.1f}%  mean={ho_md:.4f}  teacher SR={ho_teacher:.1f}%")
    tr.summary({"heldout/student_sr": ho_sr, "heldout/teacher_sr": ho_teacher, "heldout/mean_dist": ho_md})
    unseen = []
    explicit = getattr(cfg.ants, "offsets", None) is not None
    if explicit:
        logger.info("explicit ants.offsets in config: layout is fixed, skipping seed-based unseen-layout eval")
    for ls in ([] if explicit else [int(x) for x in args.unseen_layouts.split(",") if x.strip()]):
        sr_u, md_u = evaluate(net, cfg, teacher, args.history, args.eval_episodes, ls, args.seed + 10_000, dev)
        unseen.append(dict(layout_seed=ls, sr=sr_u, dist=md_u))
        logger.info(f"unseen layout {ls}: SR={sr_u:.1f}%  mean={md_u:.4f}")
        tr.summary({f"unseen_layout_{ls}/sr": sr_u, f"unseen_layout_{ls}/mean_dist": md_u})
    (out / "results.json").write_text(json.dumps(dict(args=vars(args), dagger=log, unseen=unseen,
        heldout=dict(seed=ho_seed, episodes=ho_n, student_sr=ho_sr, teacher_sr=ho_teacher, mean_dist=ho_md)), indent=2) + "\n")
    tr.summary({"final/student_sr": log[-1]["sr"], "final/teacher_sr": float(100*np.mean(ok_t))})
    tr.finish()

    print("\n" + "=" * 64)
    print(f"DAGGER SHARED PER-ANT POLICY  n={args.ants}, history={args.history}")
    print("=" * 64)
    print(f"{'it':>3} {'samples':>9} {'MSE':>9} {'student SR %':>13} {'mean dist':>10}")
    print("-" * 64)
    for r in log:
        print(f"{r['it']:>3} {r['samples']:>9} {r['mse']:>9.5f} {r['sr']:>13.1f} {r['dist']:>10.4f}")
    print(f"\nteacher during collection: {100*np.mean(ok_t):.1f}%")
    print(f"HELD-OUT ({ho_n} episodes): student {ho_sr:.1f}%  |  teacher {ho_teacher:.1f}%   <- the number to quote")
    for u in unseen:
        print(f"unseen attachment layout {u['layout_seed']}: SR={u['sr']:.1f}%  mean={u['dist']:.4f}")
    print("\nEach ant: own row only, no N input, no formula at execution.\n")


if __name__ == "__main__":
    main()
