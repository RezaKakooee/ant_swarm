# ant_swarm

A 2-D RL environment where ant agent(s) rigidly attached to a T-shaped object
must push it through a narrow barrier gap to a goal — plus SB3 training scripts
(PPO / SAC) with curriculum learning, success-trajectory harvesting, and full
run reproducibility.

The scenario is a gym recreation of the cooperative-transport "piano-movers"
experiment from the Feinerman lab (Dreyer et al., *PNAS* 2025), in which
longhorn crazy ants — and, for comparison, human groups — maneuver a T-shaped
load through two narrow wall slits. The real system is inherently multi-agent;
this project deliberately starts with a single-agent controller (`ants.n: 1`,
with a spin action) as a tractable baseline, with the multi-agent force-based
mode (`ants.n >= 2`) as the end goal.

As an RL problem it is a **deceptive-exploration** benchmark: the straight-line
path to the goal is blocked by a gap narrower than the T's big cap, so the agent
must learn to rotate-while-translating ("thread") against the local reward
gradient. Design reasoning lives in
[notes/rl_design_notes.md](notes/rl_design_notes.md); the precise env spec in
[docs/ENV_DEFINITION.md](docs/ENV_DEFINITION.md).

## Layout

```
config.yaml            single source of truth for ALL parameters (no argparse)
ant_swarm/             the env package (gym/gymnasium, RL-library-agnostic)
  config.py            load config.yaml into a namespace
  geometry.py          LocalRect + SAT collision helpers
  layout.py            world bounds, barrier walls, wall "heads", goal
  tshape.py            the rigid T-shape: geometry, pose, collision, spawning
  state.py             mutable episode state + physics integration
  action.py            action space (kinematic | dynamic) → wrench
  observation.py       per-ant 25-float observation
  reward.py            sparse | shaped reward
  render.py            pure-NumPy RGB renderer (no display needed)
  ant_swarm.py         AntSwarmEnv composing the above + curriculum hooks
  snapshot.py          save_code(): snapshot code+config into each run dir
train_ppo.py           PPO training/eval  (settings: config.yaml `run:` + `ppo:`)
train_sac.py           SAC training/eval  (settings: config.yaml `run:` + `sac:`)
train_utils.py         SB3 callbacks: curriculum, success saving (SB3 kept out of the package)
random_agent.py        random-action baseline + GIFs
render_success.py      replay saved success JSONs → GIF (single or grid)
combine_successes.py   consolidate per-episode success files
interactive/           browser sandbox (interactive_t.html; web_config.js generated
                       from config.yaml by gen_web_config.py)
ops/                   SLURM scripts (sb_train.sh submits training)
storage_local/         run outputs: checkpoints, renders, successes, logs
```

## Quickstart

Dependencies: `numpy`, `pyyaml`, `gymnasium`, `stable-baselines3`, `matplotlib`
(GIFs), optional `wandb`. On the cluster these live in the `roboverse` conda env.

```bash
# sanity-check the env with random actions (writes GIFs under storage_local/)
python random_agent.py --episodes 3

# train — everything is configured in config.yaml, no CLI args
python train_ppo.py
python train_sac.py

# evaluate a checkpoint: set run.eval: true and run.eval_model: <ckpt.zip>
# in config.yaml, then run the same script

# replay a saved success trajectory to a GIF
python render_success.py storage_local/<run>/successes/success_*.json

# on the cluster (script + optional config variant as args)
sbatch ops/sb_train.sh train_sac
sbatch ops/sb_train.sh train_sac path/to/variant.yaml
```

Or use the env directly:

```python
from ant_swarm import AntSwarmEnv
env = AntSwarmEnv(seed=0)                # or gym.make("AntSwarmBarrier-v0")
obs, info = env.reset()                  # obs: (n_ants, 25)
obs, r, term, trunc, info = env.step(env.action_space.sample())
frame = env.render()                     # (H, W, 3) uint8
```

## How training is set up (current config)

* **Single agent** (`ants.n: 1`) in `dynamic` motion mode: action =
  `[push angle, magnitude, spin]`, real momentum + wall collision.
* **Sparse reward**, goal distance measured from the **big-cap centre**
  (`env.goal_track: big_cap`) so leading with the easy small end scores nothing.
* **Reverse curriculum**: gap pinned at the hard width; spawn starts *past* the
  barrier near the goal and moves back toward the full task as rolling success
  exceeds the threshold. Eval is always pinned at the real hard task; training
  stops early once the target is mastered (`curriculum.stop_*`).
* Every run dir (`storage_local/ant__<ts>__<jobid>__<script>__<single|multi>`)
  contains a `code/` snapshot of the package + config, checkpoints (+ best),
  policy GIFs, TensorBoard logs (mirrored to W&B), and deduplicated success
  trajectories stored as minimal replayable JSONs (init pose + actions).

After editing `config.yaml`, refresh the browser sandbox's copy with
`python interactive/gen_web_config.py`.
