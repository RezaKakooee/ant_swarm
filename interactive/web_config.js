// AUTO-GENERATED from configs/rl/pnas_kin_geo_v2.yaml by gen_web_config.py — DO NOT EDIT.
// Re-run `python gen_web_config.py` after changing config.yaml
// (set ANT_SWARM_CONFIG=<variant.yaml> to use a sweep config).
const CONFIG = {
  "scene_scale": 1.0,
  "world": {
    "width": 1.65,
    "height": 0.72
  },
  "walls": {
    "x_columns": [
      0.758,
      0.979
    ],
    "length": 0.285,
    "thickness": 0.01,
    "render_extra": 0.2,
    "height": 0.08
  },
  "tshape": {
    "stem_len": 0.3315,
    "cap_big_len": 0.1748,
    "cap_small_len": 0.0874,
    "thickness": 0.027,
    "z": 0.02,
    "height": 0.04
  },
  "goal": {
    "pos": [
      1.2,
      0.36
    ],
    "reach_radius": 0.06
  },
  "spawn": {
    "x_range": [
      0.06,
      0.52
    ],
    "angle_range": [
      -3.14159265,
      3.14159265
    ],
    "margin": 0.06,
    "max_tries": 500
  },
  "ants": {
    "n": 1,
    "single_agent_spin": true,
    "radius": 0.005,
    "z": 0.04,
    "mass": 0.001
  },
  "motion": {
    "mode": "kinematic",
    "step_len": 0.01,
    "rot_step": 0.1
  },
  "physics": {
    "push_strength": 0.0005,
    "object_mass": 0.5,
    "object_inertia": 0.01,
    "linear_friction": 0.96,
    "angular_friction": 0.94,
    "substeps": 10,
    "spin_strength": null,
    "boundary_margin": 0.025,
    "restitution_wall": -0.2,
    "restitution_bound": -0.15
  },
  "env": {
    "max_steps": 500,
    "reward_mode": "geodesic",
    "reward_progress_coef": 0.1,
    "reward_success": 1.0,
    "goal_track": "big_cap",
    "geodesic_field": "storage_local/fields/pnas_gap015.npz",
    "reward_geodesic_coef": 1.0
  },
  "curriculum": {
    "enabled": true,
    "mode": "reverse",
    "success_threshold": 0.7,
    "window": 100,
    "max_steps_per_stage": 2000000,
    "stop_on_master": true,
    "stop_success": 0.9,
    "stop_window": 200,
    "start_wall_len": 0.245,
    "target_wall_len": 0.285,
    "step": 0.01,
    "reverse_wall_len": 0.285,
    "start_spawn_x": 1.3,
    "target_spawn_x": 0.3,
    "spawn_step": 0.07,
    "spawn_band": 0.05
  },
  "run": {
    "wandb": true,
    "render_freq": 500000,
    "init_from": null,
    "eval": false,
    "eval_model": null,
    "eval_episodes": 20,
    "save_successes": true,
    "dedup_successes": true,
    "dedup_tol": 0.05
  },
  "ppo": {
    "timesteps": 50000000,
    "n_envs": 8,
    "n_steps": 4096,
    "batch_size": 512,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "learning_rate": 0.0003,
    "ent_coef": 0.05,
    "ent_coef_final": 0.005,
    "use_sde": true,
    "sde_sample_freq": 16,
    "log_std_init": 0.5
  },
  "sac": {
    "timesteps": 15000000,
    "buffer_size": 1000000,
    "batch_size": 256,
    "learning_starts": 10000,
    "gamma": 0.99,
    "tau": 0.005,
    "learning_rate": 0.0003,
    "ent_coef": "auto"
  }
};
