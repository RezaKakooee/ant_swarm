"""Intrinsic (exploration) reward as a config-switchable env wrapper.

Two modes (``env.intrinsic.mode``):

  * ``count`` — visit counts over a discretised pose grid (x, y, theta);
    bonus = coef / sqrt(N(cell)). Transparent and cheap: the exact
    pseudo-count idea, on the same pose space our BFS field lives in.
  * ``rnd``  — Random Network Distillation: a frozen random net embeds the
    pose, a predictor is trained to match it; bonus = coef * normalised
    prediction error. Novel poses predict badly, so they pay more.

Both read the load's pose directly from the env state, so they are
independent of the observation format (flat or HER Dict) and of the
algorithm (SAC or PPO). The wrapper goes on the TRAINING env only —
evaluation must measure the true task reward.

The per-step bonus is added to the env reward and also reported in
``info["intrinsic_bonus"]`` so it can be logged or subtracted.
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np


class IntrinsicRewardWrapper(gym.Wrapper):
    def __init__(self, env, icfg):
        super().__init__(env)
        self.mode = str(getattr(icfg, "mode", "count"))
        self.coef = float(getattr(icfg, "coef", 0.01))
        if self.mode == "count":
            self.cell_xy = float(getattr(icfg, "cell_xy", 0.02))
            self.cell_th = math.radians(float(getattr(icfg, "cell_deg", 30.0)))
            self.counts: dict = {}
        elif self.mode == "rnd":
            import torch
            import torch.nn as nn
            embed = int(getattr(icfg, "embed_dim", 32))
            lr = float(getattr(icfg, "lr", 1e-3))
            def mlp():
                return nn.Sequential(nn.Linear(4, 64), nn.ReLU(),
                                     nn.Linear(64, 64), nn.ReLU(),
                                     nn.Linear(64, embed))
            self._torch = torch
            self.target = mlp()
            for p in self.target.parameters():
                p.requires_grad_(False)
            self.predictor = mlp()
            self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)
            self._err_mean, self._err_var, self._err_n = 0.0, 1.0, 1e-4
        else:
            raise ValueError(f"unknown intrinsic mode: {self.mode}")

    # ------------------------------------------------------------------
    def _pose(self):
        u = self.env.unwrapped
        c = u.state.object_center
        th = float(u.state.object_angle)
        return float(c[0]), float(c[1]), th

    def _bonus_count(self):
        x, y, th = self._pose()
        key = (int(x / self.cell_xy), int(y / self.cell_xy),
               int((th % (2 * math.pi)) / self.cell_th))
        n = self.counts.get(key, 0) + 1
        self.counts[key] = n
        return self.coef / math.sqrt(n)

    def _bonus_rnd(self):
        torch = self._torch
        x, y, th = self._pose()
        inp = torch.tensor([[x, y, math.sin(th), math.cos(th)]], dtype=torch.float32)
        pred = self.predictor(inp)
        with torch.no_grad():
            tgt = self.target(inp)
        err = ((pred - tgt) ** 2).mean()
        self.opt.zero_grad(); err.backward(); self.opt.step()
        e = float(err.detach())
        # running normalisation so coef means the same thing all run long
        self._err_n += 1
        d = e - self._err_mean
        self._err_mean += d / self._err_n
        self._err_var += (d * (e - self._err_mean) - self._err_var) / self._err_n
        return self.coef * e / (math.sqrt(max(self._err_var, 1e-8)) + 1e-8)

    # ------------------------------------------------------------------
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        bonus = self._bonus_count() if self.mode == "count" else self._bonus_rnd()
        info["intrinsic_bonus"] = bonus
        return obs, reward + bonus, terminated, truncated, info

    def stats(self) -> dict:
        if self.mode == "count":
            return {"cells_visited": len(self.counts)}
        return {"rnd_err_mean": self._err_mean}
