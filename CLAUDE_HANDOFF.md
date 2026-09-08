> **2026-09-05 correction:** A stale goal-reference bug corrupted the goal
> vectors in every cached transition. Fixing it and retraining the same BC
> architecture achieved **91/100 held-out successes**, without curriculum or a
> map at test time. See [CODEX_FINDINGS.md](CODEX_FINDINGS.md) for evidence,
> fixes, remaining failures, and reproducible evaluation.

# Ant-Swarm Barrier Navigation: Technical Handoff & Problem Diagnosis

---

## 1. Executive Summary & Objective

- **Task**: A single agent (ant) pushes and spins a passive 2D rigid T-shaped object through a maze with two consecutive narrow barrier slits to reach a target goal position in the goal room.
- **Strict User Constraint**: **NO Curriculum Learning** (no shrinking walls or moving starting points closer). Training must occur on the full maze using Imitation Learning and/or Reinforcement Learning.
- **The Dilemma**:
  - Open-loop replay of 34,777 expert demonstration trajectories (`storage_local/datasets/successes_v1/dataset.npz`) achieves **100.0% Success Rate (20/20)** with a mean final distance of `0.0525 m`.
  - However, all closed-loop neural network policies (Behavioral Cloning on 4.91M transitions, DAgger, HG-DAgger, Frame-Stacked BC, and Warm-Started SAC fine-tuning) achieve **0% to 5% closed-loop success rate** when deployed end-to-end.
  - The policy consistently solves open-room navigation and threads **Slit 1**, but consistently jams or wedges at the entrance/exit of **Slit 2**.

---

## 2. Environment & Physics Specifications

```text
World Bounds: [0, 1.65] m x [0, 0.72] m
Slit 1: x = 0.758 m, Gap Opening = 0.150 m (centered at y = 0.36 m)
Slit 2: x = 0.979 m, Gap Opening = 0.150 m (centered at y = 0.36 m)

T-Shape Load Geometry:
- Stem length: 0.3315 m
- Big cap width: 0.1748 m
- Small cap width: 0.0874 m
- Thickness: 0.027 m
- Mechanical Clearance through Slit (when vertical): 0.150 m - 0.1748 m (vertical alignment clearance = ~15 mm)

Physics & Friction:
- Linear Friction: 0.96 (high stiction)
- Angular Friction: 0.94
- Integration Substeps: 10 substeps per env step (dt = 0.01)

Observation Space (27-D):
- [0:2]   Attachment offset [x/stem, y/cap]
- [2:4]   Object center [x/W, y/H]
- [4:6]   Goal relative vector [(goal_x - obj_x)/W, (goal_y - obj_y)/H]
- [6:8]   Orientation [sin(theta), cos(theta)]
- [8]     Angular velocity [clipped obj.ang_vel / 0.05]
- [9:11]  Linear velocity [v_x, v_y]
- [11:27] 16 Normalized barrier distances (4 corners x 4 wall vertices)

Action Space (3-D Continuous):
- [0] Push Angle: [-pi, pi]
- [1] Push Force: [0, 1]
- [2] Spin Torque: [-1, 1]
```

---

## 3. Detailed Experimental Results Across All Paradigms

```text
+------------------------------------+--------------------+------------------+------------------+------------------+--------------------+
| Method                             | Dataset Size       | Slit 1 Status    | Slit 2 Status    | Mean Final Dist  | Best Episode Dist  |
+------------------------------------+--------------------+------------------+------------------+------------------+--------------------+
| Expert Demonstration Replay        | 20 episodes        | Cleared          | Cleared          | 0.0525 m         | 0.0210 m (100% SR) |
| Vanilla BC (Small subset)          | 500,000 steps      | Wedged           | Wedged           | 0.6025 m         | 0.3838 m (0% SR)   |
| Perturbation-Augmented BC          | 1,177,666 steps    | Wedged           | Wedged           | 0.6616 m         | 0.5884 m (0% SR)   |
| Standard DAgger (8 Iterations)     | 706,966 steps      | Cleared          | Wedged           | 0.3135 m         | 0.2185 m (0% SR)   |
| HG-DAgger (Intervention DAgger)    | 775,622 steps      | Cleared          | Wedged           | 0.3282 m         | 0.3980 m (0% SR)   |
| Full-Dataset BC (All 34,777 Demos) | 4,914,564 steps    | Cleared          | Near Exit        | 0.4358 m         | 0.0487 m (5% SR)   |
| FrameStack BC (History K=4, 108-D) | 4,914,564 steps    | Cleared          | Near Exit        | 0.4854 m         | 0.2607 m (0% SR)   |
| Warm-Started SAC (132k steps)      | Online RL + Pretr. | Cleared          | Wedged           | 0.4780 m         | 0.2615 m (0% SR)   |
+------------------------------------+--------------------+------------------+------------------+------------------+--------------------+
```

---

## 4. Forensic Diagnostics & Root Causes Identified

1. **Sub-Millimeter Contact Mechanics & Wall-Jamming (Stiction)**:
   - Vertical clearance is only **15 mm**. With high static friction (linear = 0.96, angular = 0.94), any momentary misalignment causes a corner of the T-shape to contact the slit boundary.
   - Pushing forward against a contacted wall face increases the normal force, locking the object in static friction.
2. **MSE Loss Over-Smoothing on Bimodal Contact Recovery**:
   - In contact states, the optimal recovery action is strictly bimodal: apply strong positive torque +1.0 (if top edge touched) or strong negative torque -1.0 (if bottom edge touched).
   - Supervised regression with MSE loss averages these opposing expert actions, outputting near-zero torque (~0), failing to un-wedge the load.
3. **Goal Vector Distraction**:
   - Observations contain the global relative vector `goal_d = (goal - obj_pos)`.
   - When the target goal is high in the goal room (`y = 0.58`), this vector pulls the policy upward against the barrier wall *before* the object has completely exited the vertical slit channel.
4. **SAC Exploration Noise Breakdown in Tight Spaces**:
   - In SAC fine-tuning, Gaussian policy noise injects random angular perturbations during rollout. In a 15 mm slit, this noise triggers wall-jamming before the agent ever experiences a successful terminal reward.

---

## 5. Key File Paths & Repository Structure

- **Expert Dataset**: `storage_local/datasets/successes_v1/dataset.npz` (34,777 trajectories, 4,914,568 transitions)
- **Precomputed Geodesic Field**: `storage_local/fields/pnas_gap015.npz` (BFS collision-free distance field)
- **Environment & Physics**: `ant_swarm/ant_swarm.py`, `ant_swarm/observation.py`, `ant_swarm/reward.py`
- **Training Scripts**:
  - Full Dataset BC: `scripts/il/train_full_dataset.py`
  - Frame-Stacked BC: `scripts/il/train_framestack_il.py`
  - Warm-Started SAC: `scripts/rl/train_sac.py` (Config: `configs/rl/gen_n_bc_transfer.yaml`)
- **Diagnostic Scripts**:
  - `scripts/forensic_action_discrepancy.py` (Step-by-step expert vs policy comparison)
  - `scripts/check_goal_overshoot.py` (Trajectory min-distance & goal-touch detector)

---

## 6. Questions & Guidance Requested from Claude

1. **Architecture & Action Representation**:
   - Would an Action-Chunking Transformer (ACT) or Diffusion Policy predicting trajectory chunks `[a_t, ..., a_{t+H}]` resolve the bimodal MSE over-smoothing issue better than single-step MLPs?
2. **Goal Conditioning vs Subgoal/Waypoint Tracking**:
   - How should we condition the policy so the global goal vector does not pull the object into the barrier wall during the intermediate slit traversal?
3. **Reinforcement Learning Formulation (No Curriculum)**:
   - For SAC / TD3 fine-tuning without curriculum, what exact exploration strategy, reward potential, or demonstration-seeding mechanism (e.g., SAC-fD, AWAC, or IQL) guarantees contact recovery learning through 15 mm clearances without noise-induced wall jamming?
