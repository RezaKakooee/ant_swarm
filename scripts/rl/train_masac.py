"""Multi-agent SAC with independent actors and two replay buffers (chapter 04 §15).

Same setting as train_marl_v2.py (workers, env, 29-D rows, history, independent
per-ant actors, held-out evaluation, checkpoints) but off-policy:

  * actors     one network per ant (--independent-actors, default on), each
               sees only its own row; tanh-Gaussian, entropy-regularised
  * critics    two centralised Q networks (all rows + all actions), team reward
  * buffers    MAIN: uniform ring of every transition.
               SUCCESS: ring of transitions from episodes that reached the goal.
               Every minibatch takes --success-frac of its rows from SUCCESS
               once it holds at least one batch (else all from MAIN).
  * alpha      automatic, target entropy = -act_dim per ant
  * bootstrap  only true termination (success) ends a value chain; the
               500-step time limit does not

    python scripts/rl/train_masac.py --config configs/rl/marl_v2_5ants_nomap.yaml --workers 30 --history 4

Checkpoints (`best.pt`, `final.pt`, `ckpt_<steps>.pt`) have the same actor
blob as train_marl_v2.py, so `eval_marl_v2.py` scores them unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "rl"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
from ant_swarm import AntSwarmEnv, load_config       # noqa: E402
from ant_swarm.run_id import build_run_id           # noqa: E402
from ant_swarm.tracking import Tracker             # noqa: E402
import train_marl_v2 as M                            # noqa: E402  (workers, actors, evaluate)


# --------------------------------------------------------------------------- #
# worker rollout: transitions (x, a, r, term, x') per env stream
# --------------------------------------------------------------------------- #
def _w_rollout_sac(args):
    state_dict, steps, max_steps, drive = args
    W = M._W; n, obs_dim, K = W["n"], W["obs_dim"], W["hist_k"]
    act_dim = int(W["env"].action_space.shape[-1])
    actor = M.make_actor(obs_dim * K, act_dim, n, W["independent"]); actor.load_state_dict(state_dict); actor.eval()
    out = []
    for env in W["envs"]:
        ring = W.setdefault("rings", {}).get(id(env)) or M.Ring(n, obs_dim, K)
        X, A, R, T, X2, EP = [], [], [], [], [], []
        rets, succ = [], []
        st = W.setdefault("ep_state", {}).get(id(env), (0.0, 0))
        ep_ret, k = st
        obs = env.obs_model.observe(env.state)
        x = ring.push(obs) if not ring.primed else ring.buf.reshape(n, -1)
        ep_start = 0
        with torch.no_grad():
            for t in range(steps):
                if drive == "random":
                    a_np = np.random.uniform(-1, 1, size=(n, act_dim)).astype(np.float32)
                else:
                    a, _, _ = actor.act(torch.as_tensor(x, dtype=torch.float32)); a_np = a.numpy()
                _, rew, term, trunc, info = env.step(M.to_env_action(a_np)); k += 1
                obs2 = env.obs_model.observe(env.state)
                x2 = ring.push(obs2)
                X.append(x.copy()); A.append(a_np); R.append(float(rew)); T.append(float(bool(term))); X2.append(x2.copy()); EP.append(-1)
                ep_ret += float(rew)
                if term or trunc or k >= max_steps:
                    ok = bool(info.get("is_success", False))
                    rets.append(ep_ret); succ.append(float(ok))
                    for j in range(ep_start, len(EP)): EP[j] = int(ok)      # label the whole episode
                    ep_ret, k, ep_start = 0.0, 0, len(EP)
                    env.reset(); ring = M.Ring(n, obs_dim, K); x = ring.push(env.obs_model.observe(env.state))
                else:
                    x = x2
        W["rings"][id(env)] = ring; W["ep_state"][id(env)] = (ep_ret, k)
        out.append(dict(X=np.array(X, np.float32), A=np.array(A, np.float32), R=np.array(R, np.float32), T=np.array(T, np.float32),
                        X2=np.array(X2, np.float32), EP=np.array(EP, np.int8), rets=rets, succ=succ))
    return out


# --------------------------------------------------------------------------- #
class Ring:
    """Flat replay ring over (x, a, r, term, x')."""

    def __init__(self, size, n, dim, act):
        self.size, self.i, self.full = int(size), 0, False
        self.X = np.zeros((size, n, dim), np.float32); self.X2 = np.zeros_like(self.X)
        self.A = np.zeros((size, n, act), np.float32); self.R = np.zeros(size, np.float32); self.T = np.zeros(size, np.float32)

    def add(self, X, A, R, T, X2):
        m = len(R); idx = (self.i + np.arange(m)) % self.size
        self.X[idx], self.A[idx], self.R[idx], self.T[idx], self.X2[idx] = X, A, R, T, X2
        self.i = (self.i + m) % self.size; self.full = self.full or (self.i < m) or m >= self.size

    def __len__(self):
        return self.size if self.full else self.i

    def sample(self, b, dev):
        j = np.random.randint(0, len(self), size=b)
        f = lambda z: torch.as_tensor(z[j], device=dev)
        return f(self.X), f(self.A), f(self.R), f(self.T), f(self.X2)


class TwinQ(nn.Module):
    def __init__(self, n, dim, act, hidden=256):
        super().__init__()
        inp = n * dim + n * act
        mk = lambda: nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.q1, self.q2 = mk(), mk()

    def forward(self, x, a):
        z = torch.cat([x.flatten(1), a.flatten(1)], 1)
        return self.q1(z).squeeze(-1), self.q2(z).squeeze(-1)


def policy_sample(actor, x, independent):
    """x: (B, n, D) -> actions (B, n, act) in [-1,1] and summed log-prob (B,)."""
    B, n, D = x.shape
    if independent:
        acts, lps = [], []
        for i, a in enumerate(actor.actors):
            ai, lpi, _ = a.act(x[:, i]); acts.append(ai); lps.append(lpi)
        return torch.stack(acts, 1), torch.stack(lps, 1).sum(1)
    a, lp, _ = actor.act(x.reshape(B * n, D))
    return a.reshape(B, n, -1), lp.reshape(B, n).sum(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/rl/marl_v2_5ants_nomap.yaml")
    p.add_argument("--workers", type=int, default=30)
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--timesteps", type=int, default=20_000_000)
    p.add_argument("--rollout-steps", type=int, default=512, help="per worker per round")
    p.add_argument("--envs-per-worker", type=int, default=1)
    p.add_argument("--independent-actors", dest="independent", action="store_true", default=True)
    p.add_argument("--shared-actor", dest="independent", action="store_false")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--utd", type=float, default=0.1, help="gradient updates per env step")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--buffer", type=int, default=500_000)
    p.add_argument("--success-buffer", type=int, default=200_000)
    p.add_argument("--success-frac", type=float, default=0.25)
    p.add_argument("--start-steps", type=int, default=100_000, help="random actions before learning")
    p.add_argument("--init-log-std", type=float, default=-1.0)
    p.add_argument("--init-alpha", type=float, default=1.0,
                   help="starting entropy coefficient. Per-step rewards here are ~0.01, so 1.0 inflates Q by the entropy bonus for ~300k steps; 0.01 avoids that")
    p.add_argument("--reward-scale", type=float, default=1.0, help="multiply env rewards before storing them (SAC is sensitive to reward scale)")
    p.add_argument("--eval-every", type=int, default=250_000)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--heldout-episodes", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=1_000_000)
    p.add_argument("--threads", type=int, default=8, help="torch threads in the main process")
    p.add_argument("--seed", type=int, default=31000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    p.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    args = p.parse_args()

    dev = torch.device(args.device); torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(args.threads)
    cfg = load_config(args.config); K = int(args.history); E = int(args.envs_per_worker)
    n = int(cfg.ants.n); layout = [list(map(float, q)) for q in cfg.ants.offsets] if getattr(cfg.ants, "offsets", None) is not None else None
    probe = AntSwarmEnv(config=cfg, seed=0); obs_dim = int(probe.obs_model.obs_dim); act_dim = int(probe.action_space.shape[-1]); probe.close()
    tag = "masac" + (f"_h{K}" if K > 1 else "") + ("_ind" if args.independent else "")
    out = Path(args.out) if args.out else Path("storage_local") / build_run_id(tag)
    out.mkdir(parents=True, exist_ok=True); logger.add(out / "train.log", level="INFO")
    tr = Tracker(name=out.name, group="masac", tags=["marl", "sac"], config=vars(args), enabled=args.wandb)

    actor = M.make_actor(obs_dim * K, act_dim, n, args.independent, init_log_std=args.init_log_std).to(dev)
    critic = TwinQ(n, obs_dim * K, act_dim).to(dev); target = TwinQ(n, obs_dim * K, act_dim).to(dev); target.load_state_dict(critic.state_dict())
    log_alpha = torch.full((1,), float(np.log(args.init_alpha)), requires_grad=True, device=dev); target_ent = -float(act_dim) * n
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr); opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr); opt_al = torch.optim.Adam([log_alpha], lr=args.lr)
    main_buf = Ring(args.buffer, n, obs_dim * K, act_dim); succ_buf = Ring(args.success_buffer, n, obs_dim * K, act_dim)
    logger.info(f"MASAC: obs_dim={obs_dim} act_dim={act_dim} n={n} independent={args.independent} workers={args.workers} "
                f"batch={args.batch} utd={args.utd} success_frac={args.success_frac} reward={cfg.env.reward_mode} "
                f"init_alpha={args.init_alpha} reward_scale={args.reward_scale}")
    if args.independent:
        logger.info(f"independent actors: {actor.describe()}")

    ctx = get_context("fork")
    pool = ctx.Pool(args.workers, initializer=M._w_init, initargs=(args.config, layout, args.seed, None, float(n), K, None, E, args.independent))
    steps_done, best, history, t0 = 0, -1.0, [], time.time()
    next_eval, next_ckpt = args.eval_every, (args.ckpt_every if args.ckpt_every > 0 else float("inf"))
    n_upd, n_succ_eps = 0, 0
    while steps_done < args.timesteps:
        drive = "random" if steps_done < args.start_steps else "actor"
        sd = {k: v.cpu() for k, v in actor.state_dict().items()}
        parts = [pt for lst in pool.map(_w_rollout_sac, [(sd, args.rollout_steps, 500, drive)] * args.workers) for pt in lst]
        new = 0
        for pt in parts:
            Rs_ = pt["R"] * float(args.reward_scale)
            main_buf.add(pt["X"], pt["A"], Rs_, pt["T"], pt["X2"]); new += len(pt["R"])
            m = pt["EP"] == 1
            if m.any():
                succ_buf.add(pt["X"][m], pt["A"][m], Rs_[m], pt["T"][m], pt["X2"][m])
        steps_done += new
        rets = [r for pt in parts for r in pt["rets"]]; succ = [s for pt in parts for s in pt["succ"]]; n_succ_eps += int(sum(succ))
        # ----- updates
        q_loss = a_loss = float("nan")
        if steps_done >= args.start_steps:
            k_s = int(args.batch * args.success_frac) if len(succ_buf) >= args.batch else 0
            for _ in range(int(args.utd * new)):
                X, A, R, T, X2 = main_buf.sample(args.batch - k_s, dev)
                if k_s:
                    Xs, As, Rs, Ts, X2s = succ_buf.sample(k_s, dev)
                    X, A, R, T, X2 = (torch.cat([X, Xs]), torch.cat([A, As]), torch.cat([R, Rs]), torch.cat([T, Ts]), torch.cat([X2, X2s]))
                alpha = log_alpha.exp().detach()
                with torch.no_grad():
                    a2, lp2 = policy_sample(actor, X2, args.independent)
                    q1t, q2t = target(X2, a2)
                    y = R + args.gamma * (1 - T) * (torch.min(q1t, q2t) - alpha * lp2)
                q1, q2 = critic(X, A); q_loss_t = nn.functional.mse_loss(q1, y) + nn.functional.mse_loss(q2, y)
                opt_c.zero_grad(); q_loss_t.backward(); opt_c.step()
                a_new, lp = policy_sample(actor, X, args.independent)
                q1n, q2n = critic(X, a_new); a_loss_t = (alpha * lp - torch.min(q1n, q2n)).mean()
                opt_a.zero_grad(); a_loss_t.backward(); opt_a.step()
                al_loss = -(log_alpha * (lp.detach() + target_ent)).mean()
                opt_al.zero_grad(); al_loss.backward(); opt_al.step()
                with torch.no_grad():
                    for pt_, pc_ in zip(target.parameters(), critic.parameters()): pt_.mul_(1 - args.tau).add_(args.tau * pc_)
                n_upd += 1
            q_loss, a_loss = float(q_loss_t), float(a_loss_t)
        ep_ret = float(np.mean(rets)) if rets else float("nan"); ep_succ = float(np.mean(succ)) if succ else float("nan")
        tr.log({"rollout/ep_return": ep_ret, "rollout/ep_success": ep_succ, "train/q_loss": q_loss, "train/actor_loss": a_loss,
                "train/alpha": float(log_alpha.exp()), "train/log_std": float(actor.log_std.mean()), "buffer/main": len(main_buf),
                "buffer/success": len(succ_buf), "buffer/success_episodes": n_succ_eps, "train/updates": n_upd}, step=steps_done)
        logger.info(f"[n={n}] {steps_done:>10} steps  ep_ret={ep_ret:.3f} ep_succ={ep_succ:.3f}  q={q_loss:.4f} pi={a_loss:.4f} "
                    f"alpha={float(log_alpha.exp()):.3f} log_std={float(actor.log_std.mean()):.2f}  succ_buf={len(succ_buf)} ({n_succ_eps} eps) upd={n_upd}  ({time.time()-t0:.0f}s)")
        if steps_done >= next_eval:
            next_eval += args.eval_every
            sr, md = M.evaluate(actor, args.config, layout, args.eval_episodes, args.seed + 50_000, K=K)
            history.append({"steps": steps_done, "stage_n": n, "sr": sr, "mean_dist": md})
            tr.log({"eval/success_rate_pct": sr, "eval/mean_distance_m": md}, step=steps_done)
            logger.info(f"  eval n={n}: SR={sr:.1f}%  mean={md:.4f}")
            if sr > best:
                best = sr; torch.save({"actor": actor.state_dict(), "obs_dim": obs_dim, "n": n, "layout": layout, "steps": steps_done, "independent": args.independent}, out / "best.pt")
                logger.info(f"  best so far ({sr:.1f}%) -> best.pt")
        if steps_done >= next_ckpt:
            next_ckpt += args.ckpt_every
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "log_alpha": log_alpha.detach(), "obs_dim": obs_dim, "n": n,
                        "layout": layout, "steps": steps_done, "best": best, "history": history, "independent": args.independent}, out / f"ckpt_{steps_done:09d}.pt")
            logger.info(f"  checkpoint -> ckpt_{steps_done:09d}.pt")
    pool.close(); pool.join()
    torch.save({"actor": actor.state_dict(), "obs_dim": obs_dim, "n": n, "layout": layout, "steps": steps_done, "independent": args.independent}, out / "final.pt")
    if (out / "best.pt").exists():
        actor.load_state_dict(torch.load(out / "best.pt", map_location=dev, weights_only=True)["actor"])
    ho_sr, ho_md = M.evaluate(actor, args.config, layout, int(args.heldout_episodes), args.seed + 90_000, K=K)
    logger.info(f"HELD-OUT ({args.heldout_episodes} eps): best actor SR={ho_sr:.1f}%  mean={ho_md:.4f}")
    tr.summary({"heldout/sr": ho_sr, "heldout/mean_dist": ho_md, "best/sr_during_training": best}); tr.finish()
    (out / "results.json").write_text(json.dumps({"args": vars(args), "history": history, "heldout": {"sr": ho_sr, "mean_dist": ho_md}}, indent=2) + "\n")
    print(f"\nMASAC done: best-during-training {best:.1f}%  held-out {ho_sr:.1f}%  (n={n})\n")


if __name__ == "__main__":
    main()
