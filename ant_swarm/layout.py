"""Static scene: world outline, inner barrier walls, and goal.

Built entirely from the config namespace.  Holds the wall collision AABBs and
rendering geometry; knows nothing about the T-shape or physics.
"""
from __future__ import annotations

import numpy as np


class Layout:
    """World bounds + inner barrier columns + goal."""

    def __init__(self, cfg):
        s = float(cfg.scene_scale)
        self.scene_scale = s

        self.world_size = (cfg.world.width * s, cfg.world.height * s)
        W, H = self.world_size

        self.wall_thickness = cfg.walls.thickness * s
        self.wall_len = cfg.walls.length * s
        self.wall_render_extra = cfg.walls.render_extra * s
        self.wall_height = cfg.walls.height * s

        hl = self.wall_len / 2
        upper_cy, lower_cy = H - hl, hl
        self.wall_positions = []
        for wx in cfg.walls.x_columns:
            self.wall_positions.append((wx * s, upper_cy))
            self.wall_positions.append((wx * s, lower_cy))

        ht = self.wall_thickness / 2
        self.walls_aabb = [
            (wx - ht, wx + ht, wy - hl, wy + hl)
            for wx, wy in self.wall_positions
        ]

        # "Heads" = the inner gap-facing tip of each wall segment (the corners
        # the T-shape must clear when threading the gap). One per segment.
        self.wall_heads = np.array(
            [(wx, (wy - hl) if wy > H / 2 else (wy + hl)) for wx, wy in self.wall_positions],
            dtype=np.float32,
        )

        self.goal = np.array(cfg.goal.pos, dtype=np.float32) * s
        self.reach_radius = cfg.goal.reach_radius * s
        self.world_diag = float(np.hypot(W, H))

    @property
    def gap(self) -> float:
        """Vertical opening between the upper and lower wall segments."""
        return self.world_size[1] - 2 * self.wall_len
