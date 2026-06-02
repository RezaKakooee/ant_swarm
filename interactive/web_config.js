// AUTO-GENERATED from config.yaml by gen_web_config.py — DO NOT EDIT.
// Re-run `python gen_web_config.py` after changing config.yaml.
const CONFIG = {
  "scene_scale": 1.0,
  "world": {
    "width": 1.25,
    "height": 0.72
  },
  "walls": {
    "x_columns": [
      0.525,
      0.735
    ],
    "length": 0.285,
    "thickness": 0.02,
    "render_extra": 0.2,
    "height": 0.08
  },
  "tshape": {
    "stem_len": 0.265,
    "cap_big_len": 0.18,
    "cap_small_len": 0.09,
    "thickness": 0.02,
    "z": 0.02,
    "height": 0.04
  },
  "goal": {
    "pos": [
      1.05,
      0.36
    ],
    "reach_radius": 0.05
  },
  "spawn": {
    "x_range": [
      0.06,
      0.4
    ],
    "angle_range": [
      -1.5707963,
      1.5707963
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
    "mode": "dynamic",
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
    "reward_progress_coef": 0.1,
    "reward_success": 1.0
  },
  "curriculum": {
    "enabled": true,
    "start_wall_len": 0.205,
    "target_wall_len": 0.285,
    "step": 0.01,
    "success_threshold": 0.7,
    "window": 100,
    "max_steps_per_stage": 2000000,
    "stop_on_master": true,
    "stop_success": 0.9,
    "stop_window": 200
  },
  "ppo": {
    "n_steps": 4096,
    "batch_size": 512,
    "n_epochs": 100,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "learning_rate": 0.0003,
    "ent_coef": 0.05,
    "ent_coef_final": 0.005,
    "use_sde": true,
    "sde_sample_freq": 16,
    "log_std_init": 0.5
  }
};
