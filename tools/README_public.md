# ant-piano-movers-rl

A reinforcement-learning environment for the **piano-movers problem that ants
solve**: a T-shaped load must be threaded through two narrow slits, big head
first, turned inside a short middle corridor, and taken out small head first.

It is a gym (Gymnasium) recreation of the cooperative-transport experiment in
**Dreyer et al., *PNAS* 2025**, where longhorn crazy ants — and, for comparison,
human groups — move a T-shaped load through a three-room maze.

📖 **Part one, one agent:** https://rezakakooee.github.io/ant-piano-movers-rl/
🐜 **Part two, the swarm:** https://rezakakooee.github.io/ant-swarm-rl/
🕹️ **Try the maze in your browser:** https://rezakakooee.github.io/ant-piano-movers-rl/sandbox/

---

## Why it is an interesting RL task

* The big head (0.175) does **not** fit the slit (0.150), so the load cannot be
  pushed straight through — it must be tilted and threaded.
* The corridor (0.211) is shorter than the load (0.359), so it cannot be freely
  turned in there either.
* At **no fixed angle** can the load cross the maze. Turning while moving is the
  only way through.

That makes it a **deceptive-exploration** benchmark: straight-line distance to
the goal points through the walls, so naive shaping actively punishes the
correct move. Sparse and shaped reward both fail here (0% after 14M steps); a
BFS-based route reward plus a curriculum taken from a real solution path solves
it in 247k steps.

## Install

```bash
git clone https://github.com/RezaKakooee/ant-piano-movers-rl.git
cd ant-piano-movers-rl
pip install -r requirements.txt          # pure Python, no physics engine
```

## Use the environment

```python
from ant_swarm import AntSwarmEnv

env = AntSwarmEnv(seed=0)                # or gym.make("AntSwarmBarrier-v0")
obs, info = env.reset()                  # (n_ants, 25) or (n_ants, 27) in dynamic mode
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
frame = env.render()                     # (H, W, 3) uint8, pure NumPy
```

Two motion modes: `kinematic` (direct control) and `dynamic` (forces, mass and
momentum — the mode closest to the ants). Rewards: `sparse`, `shaped`, or
`geodesic` (route distance from a precomputed BFS field).

## Train

```bash
# the geodesic field is not in git — build it once (~1 min)
python scripts/rl/gen_geodesic_field.py configs/rl/pnas_kin_geo_v2.yaml \
    --out storage_local/fields/pnas_gap015.npz --inflate 0.002 --dx 0.003 --dth 2

python scripts/rl/train_sac.py --config-name pnas_dyn_geo_v2
python scripts/rl/train_sac.py --config-name pnas_kin_geo_v2 run.wandb=false
```

Everything is configured in `configs/` (OmegaConf + Hydra), so any value can be
overridden on the command line: `sac.timesteps=5e6`, `env.reward_mode=sparse`.

### Many ants (part two)

```bash
# five ants, one shared network, PPO with a centralised critic (MAPPO)
python scripts/rl/train_marl_v2.py --config configs/rl/marl_v2_5ants.yaml --workers 8 --history 4

# one network per ant, with a little diversity in width and activation
python scripts/rl/train_marl_v2.py --config configs/rl/marl_v2_10ants.yaml --workers 8 --history 4 --independent-actors

# score any checkpoint on 200 fresh episodes
python scripts/rl/eval_marl_v2.py storage_local/<run>/final.pt --history 4

# multi-agent SAC with a second replay pool for successful episodes
python scripts/rl/train_masac.py --config configs/rl/marl_v2_5ants_nomap.yaml --workers 8 --history 4
```

`--workers` is the number of parallel copies of the maze; the runs in the post
used 30 on a 32-core node. Each ant sees only its own observation row.

## Layout

```
ant_swarm/     the environment package (RL-library agnostic)
configs/       yaml configs: rl / il / heuristic
scripts/rl/    SAC & PPO training, multi-agent trainers (train_marl_v2, train_masac), evaluation
scripts/il/    replay and render saved success trajectories
interactive/   browser sandbox (no install needed)
blogs/         the write-up and its project page
```

## Citing

The code is MIT licensed, so you are free to use it. If it helps your research,
**please cite the write-up** — that is the only thing I ask in return. GitHub's
"Cite this repository" button (from `CITATION.cff`) gives you the entry, or:

```bibtex
@misc{kakooee2026antswarm,
  author       = {Kakooee, Reza},
  title        = {Can an {AI} Agent Solve the Puzzle That Ants Solve?},
  year         = {2026},
  howpublished = {\url{https://rezakakooee.github.io/ant-piano-movers-rl/}},
  note         = {Code: \url{https://github.com/RezaKakooee/ant-piano-movers-rl}}
}
```

Experiment reproduced: Dreyer, T. et al., *Comparing cooperative geometric
puzzle solving in ants versus humans*, PNAS 2025, doi:10.1073/pnas.2414274121.

Code is MIT. The write-up, figures and videos in `blogs/` are
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — reuse them freely
with attribution.

## Contributing

Issues and pull requests are welcome — especially if you find a better method,
or make the agent cheat in a way I did not think of. Two things are open: a
swarm that transfers to a new set of attachment points, and learning the
maze without the geodesic teacher.
