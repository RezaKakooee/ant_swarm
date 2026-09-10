"""MARL v2 — what the cooperative-transport literature does, applied here.

Chapter 04 §10. Three zeros from decentralised RL (sparse, geodesic, geodesic+
curriculum) shared two departures from every successful paper in the field:
a decentralised critic (decPLM: "unstable due to non-stationarity") and an
observation that erased the load's local motion — the channel the field
coordinates through (Wang & Schwager; the ant model). This script fixes both
and adds the two other standard ingredients.

  A  centralised critic     one value from all ants' rows (training only);
                            the actor still sees only its own row
  B  contact-point velocity env.observe_contact_velocity (obs 27 -> 29),
                            absolute units (env.velocity_scale_ants: 1)
  C  warm start (optional)  distil the DAgger student into the actor, keep an
                            anchor to it during PPO  (--warm-start <pt>)
  D  parallel envs          --workers N processes, each with its own env
  E  team-size staging      --stages 2,3,5: actor carried over, critic rebuilt
                            (a curriculum over N, not over the maze)
  F  envs per worker        --envs-per-worker 5: each worker steps 5 independent
                            envs, so one worker step gives 5 actor rows for a
                            single ant, as one env step does for 5 ants (§12)
  G  periodic checkpoints   --ckpt-every 1000000: ckpt_<steps>.pt with actor,
                            critic and optimiser; --resume <ckpt> continues

Step counting: `steps` = worker steps = env steps per env summed over workers.
With --envs-per-worker 1 this is the env-step count of every earlier run. With
E envs per worker the env-step count is E x steps and the actor-row count is
n_ants x E x steps; both are logged under `samples/`.

Reward: geodesic_exit (chapter 01). No maze curriculum. Evaluation is always
the real task: fresh env, random start, random goal, deterministic actor.

    python scripts/rl/train_marl_v2.py --config configs/rl/marl_v2_5ants.yaml --workers 32
    python scripts/rl/train_marl_v2.py ... --warm-start storage_local/<dagger run>/shared_ant_policy.pt
    python scripts/rl/train_marl_v2.py ... --stages 2,3,5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import get_context
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

ACT_DIM = 2   # set at runtime from the env: 2 = (fx, fy); 3 = (fx, fy, spin) for one ant
STUDENT_OBS = 27


# --------------------------------------------------------------------------- #
# networks
# --------------------------------------------------------------------------- #
class Actor(nn.Module):
    """Shared per-ant actor: own row -> tanh-squashed Gaussian over [angle, force]."""

    def __init__(self, obs_dim, hidden=256, init_log_std=-1.0, act_dim=None):
        super().__init__()
        act_dim = int(act_dim or ACT_DIM)
        # the §9 student's body (3x256 ReLU): in the differential test it distilled to
        # 35% where the 2x256 tanh body reached 20% (and 5% with the same optimiser)
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def dist(self, obs):
        mu = self.mu(self.body(obs))
        return mu, torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def act(self, obs, deterministic=False):
        mu, d = self.dist(obs)
        raw = mu if deterministic else d.rsample()
        act = torch.tanh(raw)
        logp = d.log_prob(raw).sum(-1) - torch.log1p(-act.pow(2) + 1e-6).sum(-1)
        return act, logp, raw

    def evaluate(self, obs, raw):
        """Log-prob of the stored pre-tanh sample: exact, no atanh of a saturated action."""
        mu, d = self.dist(obs); act = torch.tanh(raw)
        logp = d.log_prob(raw).sum(-1) - torch.log1p(-act.pow(2) + 1e-6).sum(-1)
        return logp, d.entropy().sum(-1), torch.tanh(mu)


class CentralCritic(nn.Module):
    """Team value from ALL ants' rows concatenated. Training only."""

    def __init__(self, n_ants, obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_ants * obs_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, g):
        return self.net(g).squeeze(-1)


class Ring:
    """Per-ant ring of the last K actor rows, oldest first; flattened for the actor."""

    def __init__(self, n, dim, k):
        self.k, self.buf, self.primed = int(k), np.zeros((n, int(k), dim), np.float32), False

    def push(self, rows):
        if not self.primed:
            self.buf[:] = rows[:, None, :]; self.primed = True
        else:
            self.buf[:, :-1] = self.buf[:, 1:]; self.buf[:, -1] = rows
        return self.buf.reshape(len(self.buf), -1)


def to_env_action(a):
    """(fx, fy[, spin]) in [-1,1] -> env action [push_angle, force[, spin]]."""
    out = np.empty_like(a); out[..., 0] = np.arctan2(a[..., 1], a[..., 0])
    out[..., 1] = np.clip(np.linalg.norm(a[..., :2], axis=-1), 0.0, 1.0)
    if a.shape[-1] == 3:
        out[..., 2] = a[..., 2]
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# workers: one env (+ the student, for anchor labels) per process
# --------------------------------------------------------------------------- #
_W = {}


def _w_init(cfg_path, layout, seed, student_path, student_vel_div, history=1, sanity=None, envs_per_worker=1):
    torch.set_num_threads(1)
    # Pool passes the SAME initargs to every worker. Without this line all 30 workers
    # ran the same seed -> identical envs, spawns and (with deterministic collection)
    # identical rows: 30 copies of one stream. That is why the warm start memorised
    # its aggregate to MSE 0.00000 and never climbed, and why PPO saw 30 duplicated
    # trajectories per batch.
    from multiprocessing import current_process
    ident = current_process()._identity[0] if current_process()._identity else 0
    seed = int(seed) + 7919 * int(ident)
    _W["hist_k"] = int(history); _W["sanity"] = sanity
    cfg = load_config(cfg_path).copy()
    if layout is not None:
        cfg.ants.n = len(layout); cfg.ants.offsets = [list(map(float, p)) for p in layout]
    # E independent envs per worker (chapter 04 §12). Env e gets seed + 104729*e, so
    # the first env keeps the seed every earlier run used and no two envs share one.
    envs = []
    for e in range(int(envs_per_worker)):
        env = AntSwarmEnv(config=cfg, seed=seed + 104729 * e); env.reset(seed=seed + 104729 * e); envs.append(env)
    env = envs[0]
    _W.update(env=env, envs=envs, obs_dim=int(env.obs_model.obs_dim), n=int(cfg.ants.n), student=None)
    if student_path:
        # ORACLE labeller: the split controller (chapter 04 §2/§9). Unlike a distilled
        # student it is right at ANY state, which is what DAgger needs once the actor
        # drifts. It needs all arms (centralised) -- training time only.
        from dagger_shared_ant import Teacher
        _W.update(student=Teacher(student_path, cfg, torch.device("cpu")), hist=None, vel_div=1.0)


def _student_rows(obs):
    """29-D absolute-velocity rows -> the 27-D, N-normalised rows the student saw."""
    r = np.concatenate([obs[:, :9], obs[:, 9:11] / _W["vel_div"], obs[:, 13:]], axis=1)
    return r.astype(np.float32)


def _w_rollout(args):
    """One rollout of `steps` per env in this worker. Returns a LIST of streams
    (one dict per env) so that GAE in the main process never crosses env borders."""
    state_dict, steps, max_steps, drive = (list(args) + ["actor"])[:4]
    n, obs_dim, K = _W["n"], _W["obs_dim"], _W["hist_k"]
    act_dim = int(_W["env"].action_space.shape[-1])
    actor = Actor(obs_dim * K, act_dim=act_dim); actor.load_state_dict(state_dict); actor.eval()
    return [_w_rollout_env(env, actor, n, obs_dim, K, steps, max_steps, drive) for env in _W["envs"]]


def _w_rollout_env(env, actor, n, obs_dim, K, steps, max_steps, drive):
    ring = Ring(n, obs_dim, K)
    O, A, LP, R, D, G, S = [], [], [], [], [], [], []
    rets, succ, ep_ret, k = [], [], 0.0, 0
    obs = env.obs_model.observe(env.state)
    with torch.no_grad():
        for _ in range(steps):
            x = ring.push(obs)
            # DAgger states are collected by the DETERMINISTIC actor (as §9 does). Sampled
            # collection at std 0.135 pre-tanh -- 75% of the force scale -- left every
            # visited state off the deterministic actor's own trajectory (0% for 11 rounds).
            a, lp, raw = actor.act(torch.as_tensor(x, dtype=torch.float32), deterministic=(drive == "actor_det"))
            a_np, raw_np = a.numpy(), raw.numpy()
            if _W["student"] is not None:
                f = _W["student"].forces(env, obs)        # oracle: (n, 2) in push units
                S.append(np.clip(f, -1.0, 1.0).astype(np.float32))
                if drive == "student":
                    a_np = S[-1]; lp = torch.zeros(n); raw_np = np.arctanh(np.clip(a_np, -0.999, 0.999))
            _, rew, term, trunc, info = env.step(to_env_action(a_np)); k += 1
            if _W["sanity"] == "push":          # known-solvable: reward = mean |f| (max ~1.41)
                rew = float(np.linalg.norm(a_np[:, :2], axis=1).mean())
            done = bool(term or trunc or k >= max_steps)
            O.append(x.copy()); A.append(raw_np); LP.append(lp.numpy()); R.append(float(rew)); D.append(float(done))
            G.append(obs.reshape(-1).copy()); ep_ret += float(rew)
            if done:
                rets.append(ep_ret); succ.append(float(bool(info.get("is_success", False)))); ep_ret, k = 0.0, 0
                env.reset(); ring = Ring(n, obs_dim, K)
            obs = env.obs_model.observe(env.state)
    # A holds the PRE-tanh samples (exact PPO log-prob); S the student's force vectors
    return dict(O=np.array(O, np.float32), A=np.array(A, np.float32), LP=np.array(LP, np.float32),
                R=np.array(R, np.float32), D=np.array(D, np.float32), G=np.array(G, np.float32),
                S=(np.array(S, np.float32) if S else None), last_g=obs.reshape(-1).astype(np.float32),
                rets=rets, succ=succ)


# --------------------------------------------------------------------------- #
# evaluation on the real task (main process)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(actor, cfg_path, layout, episodes, seed, max_steps=500, K=1):
    cfg = load_config(cfg_path).copy()
    if layout is not None:
        cfg.ants.n = len(layout); cfg.ants.offsets = [list(map(float, p)) for p in layout]
    env = AntSwarmEnv(config=cfg, seed=seed); reach = float(env.layout.reach_radius)
    succ, dists = [], []
    for ep in range(episodes):
        env.reset(seed=seed + ep); info, done, k = {}, False, 0; ring = Ring(int(cfg.ants.n), int(env.obs_model.obs_dim), K)
        while not done and k < max_steps:
            obs = env.obs_model.observe(env.state)
            a, _, _ = actor.act(torch.as_tensor(ring.push(obs), dtype=torch.float32), deterministic=True)
            _, _, tm, tr, info = env.step(to_env_action(a.numpy())); k += 1; done = tm or tr
        d = float(info.get("object_distance", env.state.distance_to_goal())); dists.append(d)
        succ.append(bool(info.get("is_success", False) or d < reach))
    env.close()
    return float(np.mean(succ) * 100), float(np.mean(dists))


# --------------------------------------------------------------------------- #
def gae(R, D, V, last_v, gamma, lam):
    adv = np.zeros_like(R); g = 0.0
    for t in reversed(range(len(R))):
        nxt = last_v if t == len(R) - 1 else V[t + 1]
        delta = R[t] + gamma * nxt * (1 - D[t]) - V[t]
        g = delta + gamma * lam * (1 - D[t]) * g; adv[t] = g
    return adv, adv + V


def distil(actor, obs_rows, targets, epochs=40, bs=4096, lr=1e-3):
    """Warm start, the §9 way: refit body+mu FROM SCRATCH on the aggregate with a
    cosine schedule (log_std untouched). Returns the full-data MSE."""
    for m in list(actor.body) + [actor.mu]:
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    params = list(actor.body.parameters()) + list(actor.mu.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    X, Y = torch.as_tensor(obs_rows), torch.as_tensor(targets); n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for k in range(0, n - bs + 1, bs):
            i = perm[k:k + bs]
            loss = nn.functional.mse_loss(torch.tanh(actor.mu(actor.body(X[i]))), Y[i])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    with torch.no_grad():
        return nn.functional.mse_loss(torch.tanh(actor.mu(actor.body(X))), Y).item()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rl/marl_v2_5ants.yaml")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--timesteps", type=int, default=20_000_000, help="total env steps (all workers)")
    p.add_argument("--rollout-steps", type=int, default=512, help="per worker per iteration")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--minibatch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.002)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--init-log-std", type=float, default=-1.0)
    p.add_argument("--warm-start", default=None, help="single-ant BC .pt: the split-controller ORACLE labels the warm start and anchors PPO")
    p.add_argument("--history", type=int, default=1, help="frames of history for the actor")
    p.add_argument("--distil-rounds", type=int, default=6, help="warm start: DAgger rounds (actor drives, student labels)")
    p.add_argument("--distil-stop", type=float, default=10.0, help="stop distilling when actor SR is within this many points of the student")
    p.add_argument("--anchor-coef", type=float, default=0.5)
    p.add_argument("--stages", default=None, help="e.g. 2,3,5 -> team-size staging from config stage_layouts")
    p.add_argument("--init-actor", default=None, help="continue from a saved actor (best.pt/final.pt); critic starts fresh")
    p.add_argument("--stage-steps", type=int, default=4_000_000, help="max env steps per non-final stage")
    p.add_argument("--stage-sr", type=float, default=50.0, help="promote when eval SR >= this twice in a row")
    p.add_argument("--envs-per-worker", type=int, default=1,
                   help="independent envs stepped by each worker per worker step (§12: 5 gives a single ant the swarm's row count)")
    p.add_argument("--ckpt-every", type=int, default=1_000_000, help="save ckpt_<steps>.pt (actor+critic+optimiser) every this many steps; 0 = off")
    p.add_argument("--resume", default=None, help="continue from a ckpt_*.pt: actor, critic, optimiser, step count and eval history are restored")
    p.add_argument("--eval-every", type=int, default=250_000)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--heldout-episodes", type=int, default=100)
    p.add_argument("--sanity-reward", default=None, choices=[None, "push"],
                   help="replace the env reward with a known-solvable one to test the update")
    p.add_argument("--seed", type=int, default=30000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    p.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    args = p.parse_args()

    dev = torch.device(resolve_device(args.device)); torch.manual_seed(args.seed)
    cfg = load_config(args.config)
    K = int(args.history)
    tag = "marl_v2" + ("_warm" if args.warm_start else "") + ("_cont" if args.init_actor else "") + (f"_h{K}" if K > 1 else "") + (f"_stages{args.stages.replace(',', '-')}" if args.stages else "")
    resume = torch.load(args.resume, map_location=dev, weights_only=True) if args.resume else None
    if resume is not None and args.stages:
        raise SystemExit("--resume supports single-stage runs only")
    # a resumed run keeps writing into the folder it came from unless --out says otherwise
    out = Path(args.out) if args.out else (Path(args.resume).parent if resume is not None else Path("storage_local") / build_run_id(tag))
    out.mkdir(parents=True, exist_ok=True); logger.add(out / "train.log", level="INFO")
    tr = Tracker(name=out.name, group="marl_v2", tags=["marl", "v2"] + (["warm"] if args.warm_start else []) + (["staged"] if args.stages else []),
                 config=vars(args), enabled=args.wandb)

    # ----- stages: list of (n, layout)
    if args.stages:
        ns = [int(x) for x in args.stages.split(",")]
        layouts = [(n, [list(map(float, q)) for q in cfg.stage_layouts[n]]) for n in ns]
    else:
        n0 = int(cfg.ants.n); lay0 = [list(map(float, q)) for q in cfg.ants.offsets] if getattr(cfg.ants, "offsets", None) is not None else None
        layouts = [(n0, lay0)]
    probe = AntSwarmEnv(config=cfg, seed=0); obs_dim = int(probe.obs_model.obs_dim)
    global ACT_DIM
    ACT_DIM = int(probe.action_space.shape[-1]); probe.close()
    logger.info(f"MARL v2: obs_dim={obs_dim} act_dim={ACT_DIM} stages={[(n, len(l) if l else None) for n, l in layouts]} workers={args.workers} "
                f"envs/worker={args.envs_per_worker} warm={'yes' if args.warm_start else 'no'} device={dev}")

    actor = Actor(obs_dim * K, init_log_std=(-2.0 if args.warm_start else args.init_log_std), act_dim=ACT_DIM).to(dev)
    if args.init_actor:
        blob = torch.load(args.init_actor, map_location=dev, weights_only=True)
        actor.load_state_dict(blob["actor"])
        logger.info(f"actor initialised from {args.init_actor} (saved at {blob.get('steps')} steps, n={blob.get('n')})")
    ctx = get_context("fork")
    steps_done, best, history, t0 = 0, -1.0, [], time.time()
    final_n = layouts[-1][0]
    if resume is not None:
        actor.load_state_dict(resume["actor"])
        steps_done, best, history = int(resume["steps"]), float(resume.get("best", -1.0)), list(resume.get("history", []))
        logger.info(f"resumed from {args.resume} at {steps_done} steps (best so far {best:.1f}%)")
    E = int(args.envs_per_worker)

    for si, (n, layout) in enumerate(layouts):
        last_stage = si == len(layouts) - 1
        pool = ctx.Pool(args.workers, initializer=_w_init,
                        initargs=(args.config, layout, args.seed + 1000 * si + steps_done // 1000, args.warm_start, float(n), K, args.sanity_reward, E))
        critic = CentralCritic(n, obs_dim).to(dev)
        opt = torch.optim.Adam([{"params": actor.parameters(), "lr": args.lr},
                                {"params": critic.parameters(), "lr": args.lr}])
        if resume is not None:
            critic.load_state_dict(resume["critic"]); opt.load_state_dict(resume["opt"])
        logger.info(f"=== stage {si}: n={n} ===")

        if args.warm_start and si == 0:
            # DAgger distillation (chapter 04 §9 lesson): round 0 the STUDENT drives so the
            # states are its own; then the actor drives and the student labels. Refit on
            # everything each round. Stop when the actor is close to the student.
            st_sr = None
            X_all, Y_all = [], []
            for rnd in range(args.distil_rounds + 1):
                drive = "student" if rnd == 0 else "actor_det"   # round 0: the oracle drives; then the deterministic actor
                parts = [pt for lst in pool.map(_w_rollout, [(actor.state_dict(), 512, 500, drive)] * args.workers) for pt in lst]
                X_all.append(np.concatenate([pt["O"].reshape(-1, obs_dim * K) for pt in parts]))
                Y_all.append(np.concatenate([pt["S"].reshape(-1, ACT_DIM) for pt in parts]))
                if rnd == 0:
                    st_sr = float(100 * np.mean([x for pt in parts for x in pt["succ"]]))
                loss = distil(actor, np.concatenate(X_all), np.concatenate(Y_all), epochs=40)
                sr0, md0 = evaluate(actor, args.config, layout, args.eval_episodes, args.seed + 77_000, K=K)
                logger.info(f"distil round {rnd} ({drive} drove): {len(np.concatenate(X_all))} samples  MSE={loss:.5f}  "
                            f"actor SR={sr0:.1f}% mean={md0:.4f}   (oracle during collection: {st_sr:.1f}%)")
                tr.log({"warm/distil_mse": loss, "warm/actor_sr": sr0, "warm/student_sr": st_sr}, step=rnd)
                if rnd > 0 and sr0 >= st_sr - args.distil_stop:
                    logger.info("distillation close enough to the oracle -> start PPO"); break
        stage_start, promote_hits, next_eval = steps_done, 0, steps_done + args.eval_every
        next_ckpt = steps_done + args.ckpt_every if args.ckpt_every > 0 else float("inf")
        while steps_done < args.timesteps:
            if not last_stage and steps_done - stage_start >= args.stage_steps:
                logger.info(f"stage {si}: step budget reached -> promote"); break
            parts = [pt for lst in pool.map(_w_rollout, [(actor.state_dict(), args.rollout_steps, 500)] * args.workers) for pt in lst]
            # ----- advantages per worker stream, shared by the team
            Os, As, LPs, ADVs, RETs, Ss, Gs = [], [], [], [], [], [], []
            with torch.no_grad():
                for pt in parts:
                    V = critic(torch.as_tensor(pt["G"], device=dev)).cpu().numpy()
                    lv = critic(torch.as_tensor(pt["last_g"][None], device=dev)).cpu().numpy()[0]
                    adv, ret = gae(pt["R"], pt["D"], V, lv, args.gamma, args.lam)
                    T = len(adv)
                    Os.append(pt["O"].reshape(T * n, obs_dim * K)); As.append(pt["A"].reshape(T * n, ACT_DIM)); LPs.append(pt["LP"].reshape(T * n))
                    ADVs.append(np.repeat(adv, n)); RETs.append(ret); Gs.append(pt["G"])
                    if pt["S"] is not None: Ss.append(pt["S"].reshape(T * n, ACT_DIM))
            O = torch.as_tensor(np.concatenate(Os), device=dev); A = torch.as_tensor(np.concatenate(As), device=dev)
            LP = torch.as_tensor(np.concatenate(LPs), device=dev); ADV = torch.as_tensor(np.concatenate(ADVs), device=dev)
            RET = torch.as_tensor(np.concatenate(RETs), device=dev); G = torch.as_tensor(np.concatenate(Gs), device=dev)
            S = torch.as_tensor(np.concatenate(Ss), device=dev) if Ss else None
            ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
            # worker steps: env steps per env, summed over workers (= env steps when E == 1)
            steps_done += sum(len(pt["R"]) for pt in parts) // E
            # ----- PPO
            N_a, N_c = len(O), len(G)
            for _ in range(args.epochs):
                perm_a, perm_c = torch.randperm(N_a, device=dev), torch.randperm(N_c, device=dev)
                for k in range(0, N_a, args.minibatch):
                    i = perm_a[k:k + args.minibatch]
                    logp, ent, mu_t = actor.evaluate(O[i], A[i]); ratio = (logp - LP[i]).exp()
                    pg = -torch.min(ratio * ADV[i], ratio.clamp(1 - args.clip, 1 + args.clip) * ADV[i]).mean()
                    loss = pg - args.ent_coef * ent.mean()
                    if S is not None and args.anchor_coef > 0:
                        loss = loss + args.anchor_coef * nn.functional.mse_loss(mu_t, S[i])
                    j = perm_c[(k // n) % N_c:(k // n) % N_c + max(args.minibatch // n, 1)]
                    vloss = nn.functional.mse_loss(critic(G[j]), RET[j])
                    opt.zero_grad(); (loss + args.vf_coef * vloss).backward()
                    nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5); opt.step()
            rets = [r for pt in parts for r in pt["rets"]]; succ = [s for pt in parts for s in pt["succ"]]
            ep_ret, ep_succ = (float(np.mean(rets)) if rets else float("nan")), (float(np.mean(succ)) if succ else float("nan"))
            tr.log({"rollout/ep_return": ep_ret, "rollout/ep_success": ep_succ, "train/pg_loss": float(pg), "train/vf_loss": float(vloss),
                    "train/log_std": float(actor.log_std.mean()), "train/mean_force": float(torch.tanh(A).norm(dim=1).mean()), "stage/n": n,
                    "samples/env_steps": steps_done * E, "samples/actor_rows": steps_done * E * n}, step=steps_done)
            logger.info(f"[n={n}] {steps_done:>10} steps  ep_ret={ep_ret:.3f} ep_succ={ep_succ:.3f}  pg={float(pg):.4f} vf={float(vloss):.4f}  |f|={float(torch.tanh(A).norm(dim=1).mean()):.3f} log_std={float(actor.log_std.mean()):.2f}  ({time.time()-t0:.0f}s)")
            if steps_done >= next_eval:
                next_eval += args.eval_every
                sr, md = evaluate(actor, args.config, layout, args.eval_episodes, args.seed + 50_000, K=K)
                history.append({"steps": steps_done, "stage_n": n, "sr": sr, "mean_dist": md})
                tr.log({"eval/success_rate_pct": sr, "eval/mean_distance_m": md}, step=steps_done)
                logger.info(f"  eval n={n}: SR={sr:.1f}%  mean={md:.4f}")
                if n == final_n and sr > best:
                    best = sr; torch.save({"actor": actor.state_dict(), "obs_dim": obs_dim, "n": n, "layout": layout, "steps": steps_done}, out / "best.pt")
                    logger.info(f"  best so far ({sr:.1f}%) -> best.pt")
                if not last_stage:
                    promote_hits = promote_hits + 1 if sr >= args.stage_sr else 0
                    if promote_hits >= 2:
                        logger.info(f"stage {si}: SR>={args.stage_sr} twice -> promote"); break
            if steps_done >= next_ckpt:
                next_ckpt += args.ckpt_every
                torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "opt": opt.state_dict(),
                            "obs_dim": obs_dim, "n": n, "layout": layout, "steps": steps_done, "best": best, "history": history,
                            "envs_per_worker": E}, out / f"ckpt_{steps_done:09d}.pt")
                logger.info(f"  checkpoint -> ckpt_{steps_done:09d}.pt")
        pool.close(); pool.join()

    torch.save({"actor": actor.state_dict(), "obs_dim": obs_dim, "n": final_n, "layout": layouts[-1][1], "steps": steps_done}, out / "final.pt")
    if (out / "best.pt").exists():
        actor.load_state_dict(torch.load(out / "best.pt", map_location=dev, weights_only=True)["actor"])
    ho_sr, ho_md = evaluate(actor, args.config, layouts[-1][1], int(args.heldout_episodes), args.seed + 90_000, K=K)
    logger.info(f"HELD-OUT ({args.heldout_episodes} eps): best actor SR={ho_sr:.1f}%  mean={ho_md:.4f}")
    tr.summary({"heldout/sr": ho_sr, "heldout/mean_dist": ho_md, "best/sr_during_training": best}); tr.finish()
    (out / "results.json").write_text(json.dumps({"args": vars(args), "history": history, "heldout": {"sr": ho_sr, "mean_dist": ho_md}}, indent=2) + "\n")
    print(f"\nMARL v2 done: best-during-training {best:.1f}%  held-out {ho_sr:.1f}%  (n={final_n})\n")


if __name__ == "__main__":
    main()
