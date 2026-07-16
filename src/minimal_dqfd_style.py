"""Project-local minimal DQfD-style extension for auditable smoke tests.

This is deliberately not a full reproduction of Hester et al. (2018): it
keeps demonstrations in a separate permanent tensor store and adds a
demonstration auxiliary update after each ordinary SB3 DQN update. It does not
implement prioritized replay or a mixed replay sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN


@dataclass(frozen=True)
class DemonstrationBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    n_step_returns: np.ndarray
    n_step_next_observations: np.ndarray
    n_step_dones: np.ndarray
    n_step_discounts: np.ndarray

    def validate(self) -> None:
        lengths = {
            len(self.observations), len(self.actions), len(self.rewards),
            len(self.next_observations), len(self.dones), len(self.n_step_returns),
            len(self.n_step_next_observations), len(self.n_step_dones),
            len(self.n_step_discounts),
        }
        if len(lengths) != 1 or next(iter(lengths)) <= 0:
            raise ValueError(f"Demonstration arrays have inconsistent lengths: {lengths}")
        arrays = (
            self.observations, self.rewards, self.next_observations,
            self.n_step_returns, self.n_step_next_observations,
            self.n_step_discounts,
        )
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("Demonstration contains non-finite values")


def large_margin_loss(q_values: th.Tensor, expert_actions: th.Tensor, margin: float) -> th.Tensor:
    """DQfD-style supervised margin loss for discrete actions."""

    expert_actions = expert_actions.long().reshape(-1)
    expert_q = q_values.gather(1, expert_actions[:, None]).squeeze(1)
    margins = th.full_like(q_values, float(margin))
    margins.scatter_(1, expert_actions[:, None], 0.0)
    return (th.max(q_values + margins, dim=1).values - expert_q).mean()


class MinimalDQfDStyle(DQN):
    """SB3 DQN plus permanent demonstration TD, n-step and margin updates."""

    def __init__(
        self,
        *args: Any,
        demo_batch_size: int = 32,
        demo_margin: float = 0.8,
        lambda_n_step: float = 1.0,
        lambda_margin: float = 1.0,
        lambda_l2: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.demo_batch_size = int(demo_batch_size)
        self.demo_margin = float(demo_margin)
        self.lambda_n_step = float(lambda_n_step)
        self.lambda_margin = float(lambda_margin)
        self.lambda_l2 = float(lambda_l2)
        self._demo: dict[str, th.Tensor] | None = None
        self._demo_rng = np.random.default_rng(int(getattr(self, "seed", 0) or 0) + 21020)
        self.demo_update_count = 0
        self.last_demo_metrics: dict[str, float] = {}

    def set_demonstrations(self, demonstration: DemonstrationBatch) -> None:
        demonstration.validate()
        self._demo = {
            "observations": th.as_tensor(demonstration.observations, dtype=th.float32, device=self.device),
            "actions": th.as_tensor(demonstration.actions, dtype=th.long, device=self.device).reshape(-1),
            "rewards": th.as_tensor(demonstration.rewards, dtype=th.float32, device=self.device).reshape(-1),
            "next_observations": th.as_tensor(demonstration.next_observations, dtype=th.float32, device=self.device),
            "dones": th.as_tensor(demonstration.dones, dtype=th.float32, device=self.device).reshape(-1),
            "n_step_returns": th.as_tensor(demonstration.n_step_returns, dtype=th.float32, device=self.device).reshape(-1),
            "n_step_next_observations": th.as_tensor(demonstration.n_step_next_observations, dtype=th.float32, device=self.device),
            "n_step_dones": th.as_tensor(demonstration.n_step_dones, dtype=th.float32, device=self.device).reshape(-1),
            "n_step_discounts": th.as_tensor(demonstration.n_step_discounts, dtype=th.float32, device=self.device).reshape(-1),
        }

    @property
    def demonstration_count(self) -> int:
        return 0 if self._demo is None else int(len(self._demo["actions"]))

    def _sample_demo_indices(self) -> th.Tensor:
        if self._demo is None:
            raise RuntimeError("Demonstrations have not been set")
        size = self.demonstration_count
        indices = self._demo_rng.integers(0, size, size=self.demo_batch_size)
        return th.as_tensor(indices, dtype=th.long, device=self.device)

    def _demo_loss(self, indices: th.Tensor) -> tuple[th.Tensor, dict[str, th.Tensor]]:
        if self._demo is None:
            raise RuntimeError("Demonstrations have not been set")
        d = {name: value[indices] for name, value in self._demo.items()}
        q_all = self.q_net(d["observations"])
        current_q = q_all.gather(1, d["actions"][:, None]).squeeze(1)
        with th.no_grad():
            next_q = self.q_net_target(d["next_observations"]).max(dim=1).values
            target_1 = d["rewards"] + (1.0 - d["dones"]) * self.gamma * next_q
            next_q_n = self.q_net_target(d["n_step_next_observations"]).max(dim=1).values
            target_n = d["n_step_returns"] + (1.0 - d["n_step_dones"]) * d["n_step_discounts"] * next_q_n
        loss_1 = F.smooth_l1_loss(current_q, target_1)
        loss_n = F.smooth_l1_loss(current_q, target_n)
        loss_margin = large_margin_loss(q_all, d["actions"], self.demo_margin)
        loss_l2 = sum(parameter.square().sum() for parameter in self.q_net.parameters())
        total = loss_1 + self.lambda_n_step * loss_n + self.lambda_margin * loss_margin + self.lambda_l2 * loss_l2
        return total, {
            "td_1": loss_1,
            "td_n": loss_n,
            "margin": loss_margin,
            "l2": loss_l2,
            "total": total,
        }

    def demonstration_update(self) -> dict[str, float]:
        indices = self._sample_demo_indices()
        total, parts = self._demo_loss(indices)
        self.policy.optimizer.zero_grad()
        total.backward()
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()
        self.demo_update_count += 1
        metrics = {name: float(value.detach().cpu()) for name, value in parts.items()}
        self.last_demo_metrics = metrics
        return metrics

    def pretrain_demonstrations(self, updates: int) -> list[dict[str, float]]:
        if self._demo is None:
            raise RuntimeError("Demonstrations have not been set")
        self.policy.set_training_mode(True)
        records: list[dict[str, float]] = []
        for update in range(1, int(updates) + 1):
            metrics = self.demonstration_update()
            records.append({"pretrain_update": float(update), **metrics})
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        return records

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        super().train(gradient_steps=gradient_steps, batch_size=batch_size)
        if self._demo is None:
            return
        metrics: list[dict[str, float]] = []
        for _ in range(int(gradient_steps)):
            metrics.append(self.demonstration_update())
        for key in ("td_1", "td_n", "margin", "total"):
            self.logger.record(f"train/demo_{key}", float(np.mean([item[key] for item in metrics])))

