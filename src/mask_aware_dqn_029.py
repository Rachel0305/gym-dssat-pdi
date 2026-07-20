"""Small project-local mask-aware DQN used by the preregistered 029 comparison.

This module deliberately does not patch Stable-Baselines3.  Dynamic action masks
are stored in replay and applied to exploration, greedy action selection, and the
target-network maximisation operation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class QNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


@dataclass(frozen=True)
class ReplayBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    masks: torch.Tensor
    next_masks: torch.Tensor


class MaskReplayBuffer:
    def __init__(self, capacity: int, observation_dim: int, action_dim: int) -> None:
        self.capacity = int(capacity)
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.masks = np.zeros((capacity, action_dim), dtype=bool)
        self.next_masks = np.zeros((capacity, action_dim), dtype=bool)
        self.position = 0
        self.size = 0

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        mask: np.ndarray,
        next_mask: np.ndarray,
    ) -> None:
        if not bool(np.asarray(mask, dtype=bool)[int(action)]):
            raise ValueError("Replay refuses an action that was masked at collection time")
        if not done and not np.asarray(next_mask, dtype=bool).any():
            raise ValueError("A non-terminal transition must have a valid next action")
        i = self.position
        self.observations[i] = np.asarray(observation, dtype=np.float32)
        self.next_observations[i] = np.asarray(next_observation, dtype=np.float32)
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.dones[i] = float(done)
        self.masks[i] = np.asarray(mask, dtype=bool)
        self.next_masks[i] = np.asarray(next_mask, dtype=bool)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator, device: torch.device) -> ReplayBatch:
        if self.size == 0:
            raise RuntimeError("Cannot sample an empty replay buffer")
        indices = rng.integers(0, self.size, size=int(batch_size))
        return ReplayBatch(
            observations=torch.as_tensor(self.observations[indices], device=device),
            actions=torch.as_tensor(self.actions[indices], device=device).long(),
            rewards=torch.as_tensor(self.rewards[indices], device=device),
            next_observations=torch.as_tensor(self.next_observations[indices], device=device),
            dones=torch.as_tensor(self.dones[indices], device=device),
            masks=torch.as_tensor(self.masks[indices], device=device),
            next_masks=torch.as_tensor(self.next_masks[indices], device=device),
        )


class MaskAwareDQN:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int = 9,
        learning_rate: float = 3e-4,
        gamma: float = 1.0,
        replay_capacity: int = 10_000,
        batch_size: int = 30,
        seed: int = 0,
        device: str = "cpu",
        gradient_clip_norm: float = 10.0,
    ) -> None:
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.online = QNetwork(self.observation_dim, self.action_dim).to(self.device)
        self.target = QNetwork(self.observation_dim, self.action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=float(learning_rate))
        self.replay = MaskReplayBuffer(replay_capacity, self.observation_dim, self.action_dim)
        self.optimizer_updates = 0
        self.target_updates = 0

    @staticmethod
    def epsilon(step: int, total_steps: int = 240, final: float = 0.05, fraction: float = 0.70) -> float:
        decay_steps = float(total_steps) * float(fraction)
        progress = min(max(float(step), 0.0) / decay_steps, 1.0)
        return float(1.0 + progress * (float(final) - 1.0))

    @staticmethod
    def _validated_mask(mask: np.ndarray, action_dim: int) -> np.ndarray:
        result = np.asarray(mask, dtype=bool)
        if result.shape != (action_dim,):
            raise ValueError(f"Expected mask shape {(action_dim,)}, got {result.shape}")
        if not result.any():
            raise ValueError("At least one action must be valid")
        return result

    def q_values(self, observation: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(observation, dtype=np.float32), device=self.device).unsqueeze(0)
            return self.online(tensor).cpu().numpy()[0]

    def select_action(self, observation: np.ndarray, mask: np.ndarray, step: int, deterministic: bool = False) -> int:
        valid_mask = self._validated_mask(mask, self.action_dim)
        valid_indices = np.flatnonzero(valid_mask)
        if not deterministic and self.rng.random() < self.epsilon(step):
            return int(self.rng.choice(valid_indices))
        q = self.q_values(observation)
        masked_q = np.where(valid_mask, q, -np.inf)
        return int(np.argmax(masked_q))

    def add_transition(self, **kwargs: Any) -> None:
        self.replay.add(**kwargs)

    def train_step(self) -> dict[str, float]:
        batch = self.replay.sample(self.batch_size, self.rng, self.device)
        chosen_q = self.online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_all = self.target(batch.next_observations)
            safe_next_masks = batch.next_masks.clone()
            safe_next_masks[batch.dones.bool(), 0] = True
            next_q = next_all.masked_fill(~safe_next_masks, -torch.inf).max(dim=1).values
            next_q = torch.where(batch.dones.bool(), torch.zeros_like(next_q), next_q)
            targets = batch.rewards + self.gamma * next_q
        loss = F.smooth_l1_loss(chosen_q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        preclip_norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.gradient_clip_norm)
        self.optimizer.step()
        self.optimizer_updates += 1
        return {
            "loss": float(loss.detach().cpu()),
            "mean_chosen_q": float(chosen_q.detach().mean().cpu()),
            "mean_target": float(targets.detach().mean().cpu()),
            "preclip_grad_norm": float(preclip_norm.detach().cpu()),
        }

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
        self.target_updates += 1

    @staticmethod
    def module_hash(module: nn.Module) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(module.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def save(self, path: Path, global_step: int) -> None:
        payload = {
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "gradient_clip_norm": self.gradient_clip_norm,
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_updates": self.optimizer_updates,
            "target_updates": self.target_updates,
            "global_step": int(global_step),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> tuple["MaskAwareDQN", int]:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        model = cls(
            observation_dim=int(payload["observation_dim"]),
            action_dim=int(payload["action_dim"]),
            gamma=float(payload["gamma"]),
            batch_size=int(payload["batch_size"]),
            gradient_clip_norm=float(payload["gradient_clip_norm"]),
            device=device,
        )
        model.online.load_state_dict(payload["online"])
        model.target.load_state_dict(payload["target"])
        model.optimizer.load_state_dict(payload["optimizer"])
        model.optimizer_updates = int(payload["optimizer_updates"])
        model.target_updates = int(payload["target_updates"])
        return model, int(payload["global_step"])
