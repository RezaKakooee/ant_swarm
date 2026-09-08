"""Modular RL fine-tuning of a BC checkpoint, anchored to the frozen BC policy.

Chapter 02 showed sparse-reward RL never finds a single success from scratch.
Starting from the corrected BC policy (88-95%), roughly nine in ten rollouts
succeed, so the value function has signal from the first step. The remaining
failures all jam near slit 1, which is what this is meant to fix.

Two modes, selected by ``finetune.mode`` in the config. Nothing else changes.

    residual   the BC policy is FROZEN. The actor learns a bounded correction,
               ``a = clip(a_bc + scale * tanh(z))``. It cannot destroy what
               already works: at scale -> 0 it is exactly the BC policy.

    gaussian   an ordinary SAC actor, initialised by distilling the BC policy
               into it, then fine-tuned. More freedom, more risk of drift.

Both keep an anchor to the frozen reference:

    anchor_loss = mean((a_pi_mean - a_bc) ** 2)

For a Gaussian policy with fixed variance this is the KL to the reference up to
a constant factor, so ``finetune.anchor_coef`` behaves like a KL weight. It is
annealed from ``anchor_coef`` to ``anchor_final`` over ``anchor_anneal_steps``.

No geodesic field, no curriculum, no demonstrations beyond the BC checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import Actor, SACPolicy
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts" / "il") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "il"))
from train_chunked_bc import ChunkPolicy      # noqa: E402


# --------------------------------------------------------------------------- #
# the frozen reference
# --------------------------------------------------------------------------- #
class BCReference:
    """Frozen BC policy, exposed in SB3's normalised [-1, 1] action space."""

    def __init__(self, checkpoint: str | Path, obs_dim: int,
                 low: np.ndarray, high: np.ndarray, device):
        blob = th.load(checkpoint, map_location=device, weights_only=True)
        if blob.get("goal_observation_version") != 2:
            raise ValueError(
                f"{checkpoint} was not trained with repaired goal observations; "
                "retrain it before fine-tuning (see CODEX_FINDINGS.md)")
        self.net = ChunkPolicy(obs_dim, int(blob.get("horizon", 1))).to(device)
        self.net.load_state_dict(blob["state_dict"])
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.low = th.as_tensor(low, dtype=th.float32, device=device)
        self.high = th.as_tensor(high, dtype=th.float32, device=device)
        self.device = device

    def normalised(self, obs: th.Tensor) -> th.Tensor:
        """BC action mapped from env units into [-1, 1]."""
        with th.no_grad():
            a = self.net(obs)[:, 0, :]
        scaled = 2.0 * (a - self.low) / (self.high - self.low) - 1.0
        return scaled.clamp(-1.0, 1.0)


# --------------------------------------------------------------------------- #
# residual actor: frozen BC + bounded learned correction
# --------------------------------------------------------------------------- #
class ResidualActor(Actor):
    """``a = clip(a_bc + scale * tanh(z))`` in normalised action space."""

    def __init__(self, *args, reference: BCReference = None,
                 residual_scale: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference = reference
        self.residual_scale = float(residual_scale)

    def _base(self, obs: th.Tensor) -> th.Tensor:
        return self.reference.normalised(obs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        delta = self.action_dist.actions_from_params(
            mean_actions, log_std, deterministic=deterministic, **kwargs)
        return (self._base(obs) + self.residual_scale * delta).clamp(-1.0, 1.0)

    def action_log_prob(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        delta, log_prob = self.action_dist.log_prob_from_params(
            mean_actions, log_std, **kwargs)
        action = (self._base(obs) + self.residual_scale * delta).clamp(-1.0, 1.0)
        # the shift is a constant w.r.t. the sampled delta, and the scale is
        # constant, so the density changes only by a fixed log|Jacobian| term
        return action, log_prob

    def anchor_mean(self, obs: th.Tensor) -> th.Tensor:
        """Deterministic action, used by the anchor loss."""
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        delta = self.action_dist.actions_from_params(
            mean_actions, log_std, deterministic=True, **kwargs)
        return (self._base(obs) + self.residual_scale * delta).clamp(-1.0, 1.0)


class ResidualSACPolicy(SACPolicy):
    def __init__(self, *args, reference: BCReference = None,
                 residual_scale: float = 0.15, **kwargs):
        self._reference = reference
        self._residual_scale = residual_scale
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None) -> Actor:
        kw = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return ResidualActor(reference=self._reference,
                             residual_scale=self._residual_scale, **kw).to(self.device)


class AnchoredSACPolicy(SACPolicy):
    """Ordinary SAC policy; the anchor is applied by the algorithm, not here."""

    def make_actor(self, features_extractor=None) -> Actor:
        actor = super().make_actor(features_extractor)
        actor.anchor_mean = lambda obs: _tanh_mean(actor, obs)      # noqa: E731
        return actor


def _tanh_mean(actor: Actor, obs: th.Tensor) -> th.Tensor:
    mean_actions, log_std, kwargs = actor.get_action_dist_params(obs)
    return actor.action_dist.actions_from_params(
        mean_actions, log_std, deterministic=True, **kwargs)


# --------------------------------------------------------------------------- #
# SAC with an anchor to the frozen reference
# --------------------------------------------------------------------------- #
class AnchoredSAC(SAC):
    """SAC plus ``anchor_coef * MSE(pi_mean, pi_bc)`` on replay states."""

    def __init__(self, *args, reference: BCReference = None,
                 anchor_coef: float = 1.0, anchor_final: float = 0.1,
                 anchor_anneal_steps: int = 500_000,
                 critic_warmup_steps: int = 0, **kwargs):
        # With sparse reward the critic starts near-random. Letting the actor
        # follow it immediately is what destroyed the BC policy in the first
        # runs (94% -> 64%). Train the critic alone first.
        self.critic_warmup_steps = int(critic_warmup_steps)
        self.reference = reference
        self.anchor_coef = float(anchor_coef)
        self.anchor_final = float(anchor_final)
        self.anchor_anneal_steps = max(int(anchor_anneal_steps), 1)
        super().__init__(*args, **kwargs)

    def current_anchor_coef(self) -> float:
        frac = min(self.num_timesteps / self.anchor_anneal_steps, 1.0)
        return self.anchor_coef + frac * (self.anchor_final - self.anchor_coef)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        coef = self.current_anchor_coef()
        ent_losses, ent_coefs, critic_losses, actor_losses, anchors = [], [], [], [], []

        for _ in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef
                                  * (log_prob + self.target_entropy).detach()).mean()
                ent_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(float(ent_coef.item()))

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(data.next_observations)
                next_q = th.cat(self.critic_target(data.next_observations, next_actions), dim=1)
                next_q, _ = th.min(next_q, dim=1, keepdim=True)
                next_q = next_q - ent_coef * next_log_prob.reshape(-1, 1)
                target_q = data.rewards + (1 - data.dones) * self.gamma * next_q

            current_q = self.critic(data.observations, data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(q, target_q) for q in current_q)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self.num_timesteps < self.critic_warmup_steps:
                continue          # critic-only warm-up: the actor stays put

            q_pi = th.cat(self.critic(data.observations, actions_pi), dim=1)
            min_q, _ = th.min(q_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_q).mean()

            if coef > 0.0 and self.reference is not None:
                ref = self.reference.normalised(data.observations)
                anchor = F.mse_loss(self.actor.anchor_mean(data.observations), ref)
                anchors.append(anchor.item())
                actor_loss = actor_loss + coef * anchor

            actor_losses.append(actor_loss.item())
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            self._n_updates += 1
            if self._n_updates % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("finetune/anchor_coef", coef)
        if anchors:
            self.logger.record("finetune/anchor_mse", np.mean(anchors))
        if ent_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_losses))


# --------------------------------------------------------------------------- #
# distillation, used to start the gaussian mode from the BC policy
# --------------------------------------------------------------------------- #
def distill_into_actor(model: SAC, reference: BCReference, obs: np.ndarray,
                       steps: int = 2000, batch_size: int = 1024,
                       lr: float = 1e-3) -> float:
    """Fit the SAC actor's deterministic output to the BC policy."""
    device = model.device
    obs_t = th.as_tensor(obs, dtype=th.float32, device=device)
    opt = th.optim.Adam(model.actor.parameters(), lr=lr)
    n, last = len(obs_t), float("nan")
    for _ in range(steps):
        idx = th.randint(0, n, (min(batch_size, n),), device=device)
        batch = obs_t[idx]
        loss = F.mse_loss(_tanh_mean(model.actor, batch),
                          reference.normalised(batch))
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = loss.item()
    return last


# --------------------------------------------------------------------------- #
# the single switch point
# --------------------------------------------------------------------------- #
def build_finetune_model(cfg, env, reference: BCReference, tb_dir=None,
                         device: str = "auto") -> SAC:
    """Build the fine-tuning model from ``cfg.finetune``. This is the only
    place the mode is branched on; everything else is shared."""
    ft = cfg.finetune
    mode = str(ft.mode).lower()
    common: dict[str, Any] = dict(
        env=env,
        learning_rate=float(ft.learning_rate),
        buffer_size=int(ft.buffer_size),
        learning_starts=int(ft.learning_starts),
        batch_size=int(ft.batch_size),
        tau=float(ft.tau),
        gamma=float(ft.gamma),
        train_freq=int(ft.train_freq),
        gradient_steps=int(ft.gradient_steps),
        ent_coef=ft.ent_coef if isinstance(ft.ent_coef, str) else float(ft.ent_coef),
        tensorboard_log=str(tb_dir) if tb_dir else None,
        device=device,
        verbose=1,
        seed=int(ft.seed),
        reference=reference,
        anchor_coef=float(ft.anchor_coef),
        anchor_final=float(ft.anchor_final),
        anchor_anneal_steps=int(ft.anchor_anneal_steps),
        critic_warmup_steps=int(ft.get('critic_warmup_steps', 0)),
    )
    net_arch = list(ft.net_arch) if ft.get("net_arch") else [256, 256, 256]

    if mode == "residual":
        policy_kwargs = dict(net_arch=net_arch,
                             reference=reference,
                             residual_scale=float(ft.residual_scale))
        return AnchoredSAC(ResidualSACPolicy, policy_kwargs=policy_kwargs, **common)
    if mode == "gaussian":
        policy_kwargs = dict(net_arch=net_arch)
        return AnchoredSAC(AnchoredSACPolicy, policy_kwargs=policy_kwargs, **common)
    raise ValueError(f"unknown finetune.mode {ft.mode!r}; use 'residual' or 'gaussian'")
