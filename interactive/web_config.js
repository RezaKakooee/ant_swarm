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
      0.52,
      0.74
    ],
    "length": 0.285,
    "thickness": 0.02,
    "render_extra": 0.2,
    "height": 0.08
  },
  "tshape": {
    "stem_len": 0.22,
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
    "n": 2,
    "radius": 0.005,
    "z": 0.04,
    "mass": 0.001
  },
  "physics": {
    "push_strength": 0.00025,
    "object_mass": 0.5,
    "object_inertia": 0.04,
    "linear_friction": 0.96,
    "angular_friction": 0.94,
    "substeps": 10,
    "boundary_margin": 0.025,
    "restitution_wall": -0.2,
    "restitution_bound": -0.15
  },
  "env": {
    "max_steps": 2000,
    "reward_progress_coef": 0.1,
    "reward_success": 1.0
  }
};
