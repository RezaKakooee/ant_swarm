"""Rendering: produce an RGB frame from layout + physics state.

Entirely pure-NumPy, no MuJoCo / display dependency.

    from ant_swarm.render import Renderer
    renderer = Renderer(cfg, layout)
    frame = renderer.render(state)   # np.uint8 (H, W, 3)
"""
from __future__ import annotations

import math

import numpy as np


class Renderer:
    def __init__(self, cfg, layout, img_width: int = 650):
        self.layout = layout
        self.img_width = img_width
        W, H = layout.world_size
        self.img_h = int(img_width * H / W)
        self.ant_radius = cfg.ants.radius * float(cfg.scene_scale)

    # ------------------------------------------------------------------
    def render(self, state) -> np.ndarray:
        """Return an ``(H, W, 3)`` uint8 RGB image of the current state."""
        lay = self.layout
        obj = state.obj
        W, H = lay.world_size
        img_w, img_h = self.img_width, self.img_h
        img = np.full((img_h, img_w, 3), 0.72, dtype=np.float32)

        def to_px(xy):
            x, y = xy
            return int(x / W * (img_w - 1)), int((H - y) / H * (img_h - 1))

        self._draw_walls(img, lay, to_px, img_w, img_h)
        self._draw_goal(img, lay, to_px, img_h, img_w)
        self._draw_tshape(img, obj, W, H, img_w, img_h)
        self._draw_ants(img, state.ants, W, img_w, img_h, to_px)

        return np.clip(img * 255, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    def _draw_walls(self, img, lay, to_px, img_w, img_h):
        half_H = lay.world_size[1] / 2
        ht_w = lay.wall_thickness / 2
        hl = lay.wall_len / 2
        extra = lay.wall_render_extra
        for wx, wy in lay.wall_positions:
            if wy > half_H:
                y_inner, y_outer = wy - hl, wy + hl + extra
            else:
                y_inner, y_outer = wy + hl, wy - hl - extra
            y_lo, y_hi = min(y_inner, y_outer), max(y_inner, y_outer)
            x0, y0 = to_px((wx - ht_w, y_lo))
            x1, y1 = to_px((wx + ht_w, y_hi))
            xs0, xs1 = max(0, min(x0, x1)), min(img_w, max(x0, x1) + 1)
            ys0, ys1 = max(0, min(y0, y1)), min(img_h, max(y0, y1) + 1)
            img[ys0:ys1, xs0:xs1] = [0.38, 0.38, 0.42]

    def _draw_goal(self, img, lay, to_px, img_h, img_w):
        gx, gy = to_px(lay.goal)
        r = 4
        img[max(0, gy - r):min(img_h, gy + r + 1),
            max(0, gx - r):min(img_w, gx + r + 1)] = [0.18, 0.52, 0.18]

    def _draw_tshape(self, img, obj, W, H, img_w, img_h):
        yy, xx = np.mgrid[0:img_h, 0:img_w]
        xw = xx / (img_w - 1) * W
        yw = (img_h - 1 - yy) / (img_h - 1) * H
        cos_a, sin_a = math.cos(obj.angle), math.sin(obj.angle)
        lx = cos_a * (xw - obj.center[0]) + sin_a * (yw - obj.center[1])
        ly = -sin_a * (xw - obj.center[0]) + cos_a * (yw - obj.center[1])
        mask = np.zeros((img_h, img_w), dtype=bool)
        for rect in obj.rects:
            rcx, rcy = rect.center
            rhx, rhy = rect.half_size
            mask |= (np.abs(lx - rcx) <= rhx) & (np.abs(ly - rcy) <= rhy)
        img[mask] = [0.67, 0.08, 0.055]

    def _draw_ants(self, img, ants, W, img_w, img_h, to_px):
        r = max(4, int(self.ant_radius / W * img_w * 2))
        for p in ants:
            px, py = to_px(p)
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        iy, ix = py + dy, px + dx
                        if 0 <= iy < img_h and 0 <= ix < img_w:
                            img[iy, ix] = [1.0, 1.0, 1.0]
