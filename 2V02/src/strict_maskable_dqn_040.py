"""Strict mask-aware DQN for the 040 lowIC free-timing experiments.

This is a project-local DQN implementation, not a Stable-Baselines3 wrapper.
The action mask is used in all places where it matters:

- epsilon-random exploration samples only from legal actions;
- greedy action selection masks invalid Q values;
- replay stores current and next action masks;
- Bellman target maximisation uses the next-state legal action mask.

The class is intentionally small so the training scripts can audit every moving
part.  It is designed for discrete DSSAT water-nitrogen actions under dynamic
feasibility masks.
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
    def __init__(self, observation_dim: int, action_dim: int, net_arch: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = int(observation_dim)
        for width in net_arch:
            layers.append(nn.Linear(last, int(width)))
            layers.append(nn.ReLU())
            last = int(width)
        layers.append(nn.Linear(last, int(action_dim)))
        self.net = nn.Sequential(*layers)

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
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.observations = np.zeros((self.capacity, self.observation_dim), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, self.observation_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.masks = np.zeros((self.capacity, self.action_dim), dtype=bool)
        self.next_masks = np.zeros((self.capacity, self.action_dim), dtype=bool)
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
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        next_obs = np.asarray(next_observation, dtype=np.float32).reshape(-1)
        cur_mask = np.asarray(mask, dtype=bool)
        nxt_mask = np.asarray(next_mask, dtype=bool)
        if obs.shape != (self.observation_dim,):
            raise ValueError(f"Expected observation shape {(self.observation_dim,)}, got {obs.shape}")
        if next_obs.shape != (self.observation_dim,):
            raise ValueError(f"Expected next observation shape {(self.observation_dim,)}, got {next_obs.shape}")
        if cur_mask.shape != (self.action_dim,):
            raise ValueError(f"Expected mask shape {(self.action_dim,)}, got {cur_mask.shape}")
        if nxt_mask.shape != (self.action_dim,):
            raise ValueError(f"Expected next_mask shape {(self.action_dim,)}, got {nxt_mask.shape}")
        if not cur_mask.any():
            raise ValueError("Current mask has no valid action")
        if not bool(cur_mask[int(action)]):
            raise ValueError("Replay refuses an action that was masked at collection time")
        if not bool(done) and not nxt_mask.any():
            raise ValueError("Non-terminal transition has no valid next action")

        i = self.position
        self.observations[i] = obs
        self.next_observations[i] = next_obs
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.dones[i] = float(done)
        self.masks[i] = cur_mask
        self.next_masks[i] = nxt_mask
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator, device: torch.device) -> ReplayBatch:
        if self.size <= 0:
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


class StrictMaskableDQN:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        learning_rate: float = 1e-5,
        gamma: float = 0.99,
        buffer_size: int = 100_000,
        batch_size: int = 1024,
        net_arch: list[int] | None = None,
        weight_decay: float = 0.0,
        max_grad_norm: float = 10.0,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.0,
        exploration_fraction: float = 1.0,
        total_timesteps: int = 100_000,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.max_grad_norm = float(max_grad_norm)
        self.exploration_initial_eps = float(exploration_initial_eps)
        self.exploration_final_eps = float(exploration_final_eps)
        self.exploration_fraction = float(exploration_fraction)
        self.total_timesteps = int(total_timesteps)
        self.net_arch = [int(x) for x in (net_arch or [256, 256, 256])]
        self.device = torch.device(device)
        self.rng = np.random.default_rng(int(seed))
        torch.manual_seed(int(seed))

        self.online = QNetwork(self.observation_dim, self.action_dim, self.net_arch).to(self.device)
        self.target = QNetwork(self.observation_dim, self.action_dim, self.net_arch).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        self.replay = MaskReplayBuffer(buffer_size, self.observation_dim, self.action_dim)
        self.optimizer_updates = 0
        self.target_updates = 0

    def epsilon(self, step: int) -> float:
        decay_steps = max(float(self.total_timesteps) * float(self.exploration_fraction), 1.0)
        progress = min(max(float(step), 0.0) / decay_steps, 1.0)
        return float(self.exploration_initial_eps + progress * (self.exploration_final_eps - self.exploration_initial_eps))

    def _validated_mask(self, mask: np.ndarray) -> np.ndarray:
        result = np.asarray(mask, dtype=bool)
        if result.shape != (self.action_dim,):
            raise ValueError(f"Expected mask shape {(self.action_dim,)}, got {result.shape}")
        if not result.any():
            raise ValueError("At least one action must be valid")
        return result

    def q_values(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if obs.shape != (self.observation_dim,):
            raise ValueError(f"Expected observation shape {(self.observation_dim,)}, got {obs.shape}")
        with torch.no_grad():
            tensor = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            return self.online(tensor).cpu().numpy()[0]

    def select_action(self, observation: np.ndarray, mask: np.ndarray, step: int, deterministic: bool = False) -> int:
        valid_mask = self._validated_mask(mask)
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
        preclip_norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.max_grad_norm)
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
            "max_grad_norm": self.max_grad_norm,
            "exploration_initial_eps": self.exploration_initial_eps,
            "exploration_final_eps": self.exploration_final_eps,
            "exploration_fraction": self.exploration_fraction,
            "total_timesteps": self.total_timesteps,
            "net_arch": self.net_arch,
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_updates": self.optimizer_updates,
            "target_updates": self.target_updates,
            "global_step": int(global_step),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> tuple["StrictMaskableDQN", int]:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        model = cls(
            observation_dim=int(payload["observation_dim"]),
            action_dim=int(payload["action_dim"]),
            gamma=float(payload["gamma"]),
            batch_size=int(payload["batch_size"]),
            max_grad_norm=float(payload["max_grad_norm"]),
            exploration_initial_eps=float(payload["exploration_initial_eps"]),
            exploration_final_eps=float(payload["exploration_final_eps"]),
            exploration_fraction=float(payload["exploration_fraction"]),
            total_timesteps=int(payload["total_timesteps"]),
            net_arch=[int(x) for x in payload["net_arch"]],
            device=device,
        )
        model.online.load_state_dict(payload["online"])
        model.target.load_state_dict(payload["target"])
        model.optimizer.load_state_dict(payload["optimizer"])
        model.optimizer_updates = int(payload["optimizer_updates"])
        model.target_updates = int(payload["target_updates"])
        return model, int(payload["global_step"])

