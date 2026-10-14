"""Auditable replay and loss primitives for a literature-aligned DQfD adaptation.

This module contains no DSSAT calls and no training loop. It isolates the
mechanisms that must be correct before a DQfD-style online experiment is
allowed: permanent demonstrations, prioritized mixed sampling, importance
weights and decomposed losses.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ReplaySample:
    global_indices: np.ndarray
    probabilities: np.ndarray
    importance_weights: np.ndarray
    is_demonstration: np.ndarray
    data: dict[str, np.ndarray]


def arrays_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


class PrioritizedDemonstrationReplay:
    """Permanent demonstrations plus a replaceable agent ring buffer."""

    def __init__(
        self,
        demonstrations: Mapping[str, np.ndarray],
        *,
        agent_capacity: int,
        alpha: float = 0.4,
        epsilon_demo: float = 1.0,
        epsilon_agent: float = 0.001,
        seed: int = 0,
    ) -> None:
        if not demonstrations:
            raise ValueError("demonstrations cannot be empty")
        lengths = {len(value) for value in demonstrations.values()}
        if len(lengths) != 1 or next(iter(lengths)) <= 0:
            raise ValueError(f"inconsistent demonstration lengths: {lengths}")
        if agent_capacity <= 0:
            raise ValueError("agent_capacity must be positive")
        self.demonstrations = {
            name: np.asarray(value).copy() for name, value in demonstrations.items()
        }
        for value in self.demonstrations.values():
            value.setflags(write=False)
        self.agent_capacity = int(agent_capacity)
        self.alpha = float(alpha)
        self.epsilon_demo = float(epsilon_demo)
        self.epsilon_agent = float(epsilon_agent)
        self.rng = np.random.default_rng(seed)
        self.demo_priorities = np.full(self.demo_count, self.epsilon_demo, dtype=np.float64)
        self.agent_priorities = np.zeros(self.agent_capacity, dtype=np.float64)
        self.agent_data = {
            name: np.empty((self.agent_capacity, *value.shape[1:]), dtype=value.dtype)
            for name, value in self.demonstrations.items()
        }
        self.agent_size = 0
        self.agent_position = 0

    @property
    def demo_count(self) -> int:
        return len(next(iter(self.demonstrations.values())))

    @property
    def total_size(self) -> int:
        return self.demo_count + self.agent_size

    @property
    def demonstration_hash(self) -> str:
        return arrays_sha256(self.demonstrations)

    def add_agent(self, transition: Mapping[str, np.ndarray | float | int], td_error: float) -> int:
        missing = set(self.agent_data) - set(transition)
        if missing:
            raise KeyError(f"agent transition missing fields: {sorted(missing)}")
        position = self.agent_position
        for name, destination in self.agent_data.items():
            destination[position] = np.asarray(transition[name], dtype=destination.dtype)
        self.agent_priorities[position] = abs(float(td_error)) + self.epsilon_agent
        self.agent_position = (position + 1) % self.agent_capacity
        self.agent_size = min(self.agent_size + 1, self.agent_capacity)
        return self.demo_count + position

    def combined_priorities(self) -> np.ndarray:
        return np.concatenate([self.demo_priorities, self.agent_priorities[: self.agent_size]])

    def sampling_probabilities(self) -> np.ndarray:
        priorities = self.combined_priorities()
        if not np.isfinite(priorities).all() or np.any(priorities <= 0):
            raise ValueError("priorities must be finite and positive")
        powered = priorities ** self.alpha
        return powered / powered.sum()

    def _gather(self, global_indices: np.ndarray) -> dict[str, np.ndarray]:
        output: dict[str, list[np.ndarray]] = {name: [] for name in self.demonstrations}
        for index in global_indices:
            if index < self.demo_count:
                for name, values in self.demonstrations.items():
                    output[name].append(values[index])
            else:
                position = int(index - self.demo_count)
                if position >= self.agent_size:
                    raise IndexError(f"agent sample position {position} is not active")
                for name, values in self.agent_data.items():
                    output[name].append(values[position])
        return {name: np.asarray(values) for name, values in output.items()}

    def sample(self, batch_size: int, *, beta: float = 0.6, seed: int | None = None) -> ReplaySample:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        probabilities = self.sampling_probabilities()
        generator = np.random.default_rng(seed) if seed is not None else self.rng
        indices = generator.choice(self.total_size, size=int(batch_size), replace=True, p=probabilities)
        chosen_p = probabilities[indices]
        weights = (self.total_size * chosen_p) ** (-float(beta))
        weights = weights / weights.max()
        return ReplaySample(
            global_indices=indices.astype(np.int64),
            probabilities=chosen_p.astype(np.float64),
            importance_weights=weights.astype(np.float32),
            is_demonstration=(indices < self.demo_count),
            data=self._gather(indices),
        )

    def update_priorities(self, global_indices: Iterable[int], td_errors: Iterable[float]) -> None:
        for index, error in zip(global_indices, td_errors):
            index = int(index)
            if index < 0 or index >= self.total_size:
                raise IndexError(index)
            if index < self.demo_count:
                self.demo_priorities[index] = abs(float(error)) + self.epsilon_demo
            else:
                self.agent_priorities[index - self.demo_count] = abs(float(error)) + self.epsilon_agent


def dqfd_loss_components(
    *,
    q_values: torch.Tensor,
    actions: torch.Tensor,
    target_1_step: torch.Tensor,
    target_n_step: torch.Tensor,
    importance_weights: torch.Tensor,
    demonstration_mask: torch.Tensor,
    l2_parameters: Iterable[torch.Tensor],
    margin: float = 0.8,
    lambda_n_step: float = 1.0,
    lambda_margin: float = 1.0,
    lambda_l2: float = 1e-5,
) -> dict[str, torch.Tensor]:
    actions = actions.long().reshape(-1)
    target_1_step = target_1_step.reshape(-1)
    target_n_step = target_n_step.reshape(-1)
    importance_weights = importance_weights.reshape(-1)
    demonstration_mask = demonstration_mask.bool().reshape(-1)
    chosen_q = q_values.gather(1, actions[:, None]).squeeze(1)

    td_1_each = F.smooth_l1_loss(chosen_q, target_1_step, reduction="none")
    td_n_each = F.smooth_l1_loss(chosen_q, target_n_step, reduction="none")
    td_1_raw = (td_1_each * importance_weights).mean()
    td_n_raw = (td_n_each * importance_weights).mean()

    margins = torch.full_like(q_values, float(margin))
    margins.scatter_(1, actions[:, None], 0.0)
    margin_each = torch.max(q_values + margins, dim=1).values - chosen_q
    if demonstration_mask.any():
        margin_raw = (
            margin_each[demonstration_mask] * importance_weights[demonstration_mask]
        ).mean()
    else:
        margin_raw = q_values.sum() * 0.0

    l2_raw = sum(parameter.square().sum() for parameter in l2_parameters)
    td_1_weighted = td_1_raw
    td_n_weighted = float(lambda_n_step) * td_n_raw
    margin_weighted = float(lambda_margin) * margin_raw
    l2_weighted = float(lambda_l2) * l2_raw
    total = td_1_weighted + td_n_weighted + margin_weighted + l2_weighted
    return {
        "td_1_raw": td_1_raw,
        "td_n_raw": td_n_raw,
        "margin_raw": margin_raw,
        "l2_raw": l2_raw,
        "td_1_weighted": td_1_weighted,
        "td_n_weighted": td_n_weighted,
        "margin_weighted": margin_weighted,
        "l2_weighted": l2_weighted,
        "total": total,
    }

