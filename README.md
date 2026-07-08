# Ant Swarm

Ant Swarm is a toy Gym/Gymnasium reinforcement-learning environment for a
multi-agent transport task.

## Environment

- Ant-like agents are attached to a rigid T-shaped object.
- The agents push and rotate the object through narrow barrier gaps toward a goal.
- Custom 2-D rigid-body dynamics drive the task, while MuJoCo is used for rendering.

## Training

- Includes Stable-Baselines3 PPO and SAC training scripts.
- Supports shaped and sparse rewards, curriculum hooks, success rendering, and
  experiment scripts for local or cluster runs.

This repo is useful as a compact Actigon-style benchmark for deceptive
exploration, continuous actions, and multi-agent coordination.
