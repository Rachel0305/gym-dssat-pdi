"""Auditable scalar wrapper for training rewards only.

This module does not define the economic objective.  It multiplies the reward
returned by an already constructed environment while retaining the raw value
and checking the baseline-relative component identity.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np


class AuditableTrainingRewardScaleWrapper(gym.Env):
    """Scale the complete reward and retain a bounded step-level audit trail."""

    metadata = {"render_modes": []}

    def __init__(self, env: gym.Env, scale: float, *, audit_capacity: int = 10000):
        super().__init__()
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("training reward scale must be finite and positive")
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.scale = float(scale)
        self.audit_records: deque[dict[str, float]] = deque(maxlen=int(audit_capacity))
        self.total_steps = 0
        self.max_scale_identity_error = 0.0
        self.max_component_identity_error = 0.0
        self.component_fields_missing = 0
        self.callback_boundary_uncommitted_steps: set[int] = set()

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        obs, raw_reward, terminated, truncated, info = self.env.step(action)
        raw = float(raw_reward)
        training = float(raw * self.scale)
        details = dict(info) if isinstance(info, dict) else {}
        required = ("yield_gain", "water_cost_term", "nitrogen_cost_term")
        if all(key in details for key in required):
            yield_gain = float(details["yield_gain"])
            water_cost = float(details["water_cost_term"])
            nitrogen_cost = float(details["nitrogen_cost_term"])
            scaled_by_components = float(
                self.scale * yield_gain
                - self.scale * water_cost
                - self.scale * nitrogen_cost
            )
            component_error = abs(training - scaled_by_components)
        else:
            yield_gain = water_cost = nitrogen_cost = scaled_by_components = np.nan
            component_error = np.nan
            self.component_fields_missing += 1
        scale_error = abs(training - raw * self.scale)
        self.max_scale_identity_error = max(self.max_scale_identity_error, scale_error)
        if np.isfinite(component_error):
            self.max_component_identity_error = max(
                self.max_component_identity_error, float(component_error)
            )
        self.total_steps += 1
        record = {
            "step": float(self.total_steps),
            "raw_reward": raw,
            "training_reward": training,
            "reward_scale": self.scale,
            "yield_gain": yield_gain,
            "water_cost_term": water_cost,
            "nitrogen_cost_term": nitrogen_cost,
            "scaled_component_sum": scaled_by_components,
            "scale_identity_error": scale_error,
            "component_identity_error": component_error,
        }
        self.audit_records.append(record)
        details.update(record)
        return obs, training, terminated, truncated, details

    def mark_callback_boundary_uncommitted_step(self) -> int:
        """Record the env step observed before a stop callback rejects replay insertion."""

        self.callback_boundary_uncommitted_steps.add(int(self.total_steps))
        return int(self.total_steps)

    def replay_consistency(self, replay_buffer: Any) -> dict[str, Any]:
        """Compare immediate rewards stored by SB3 with recent returned rewards."""

        if replay_buffer is None:
            return {"checked": False, "reason": "missing_replay_buffer"}
        if replay_buffer.full:
            stored = np.concatenate(
                (
                    replay_buffer.rewards[replay_buffer.pos :, 0],
                    replay_buffer.rewards[: replay_buffer.pos, 0],
                )
            )
        else:
            stored = replay_buffer.rewards[: replay_buffer.pos, 0]
        records = list(self.audit_records)
        committed_records = [
            row
            for row in records
            if int(row["step"]) not in self.callback_boundary_uncommitted_steps
        ]
        if not len(stored) or len(committed_records) < len(stored):
            return {
                "checked": False,
                "reason": "insufficient_committed_wrapper_records",
                "stored_count": int(len(stored)),
                "record_count": int(len(records)),
                "committed_record_count": int(len(committed_records)),
                "uncommitted_boundary_steps": sorted(self.callback_boundary_uncommitted_steps),
            }
        selected_records = committed_records[-len(stored) :]
        expected = np.asarray(
            [row["training_reward"] for row in selected_records], dtype=np.float64
        )
        stored64 = np.asarray(stored, dtype=np.float64).reshape(-1)
        error = np.abs(stored64 - expected)
        return {
            "checked": True,
            "stored_count": int(len(stored64)),
            "record_count": int(len(records)),
            "committed_record_count": int(len(committed_records)),
            "callback_boundary_uncommitted_final_step": bool(
                self.total_steps in self.callback_boundary_uncommitted_steps
            ),
            "uncommitted_boundary_steps": sorted(self.callback_boundary_uncommitted_steps),
            "max_abs_error": float(error.max()),
            "mean_abs_error": float(error.mean()),
            "all_finite": bool(np.isfinite(stored64).all() and np.isfinite(expected).all()),
        }

    def audit_snapshot(self, replay_buffer: Any | None = None) -> dict[str, Any]:
        records = list(self.audit_records)
        raw = np.asarray([row["raw_reward"] for row in records], dtype=float)
        training = np.asarray([row["training_reward"] for row in records], dtype=float)
        return {
            "enabled": True,
            "reward_scale": self.scale,
            "total_steps": int(self.total_steps),
            "retained_record_count": int(len(records)),
            "max_scale_identity_error": float(self.max_scale_identity_error),
            "max_component_identity_error": float(self.max_component_identity_error),
            "component_fields_missing": int(self.component_fields_missing),
            "uncommitted_boundary_steps": sorted(self.callback_boundary_uncommitted_steps),
            "raw_reward_min": float(raw.min()) if len(raw) else None,
            "raw_reward_max": float(raw.max()) if len(raw) else None,
            "training_reward_min": float(training.min()) if len(training) else None,
            "training_reward_max": float(training.max()) if len(training) else None,
            "all_rewards_finite": bool(np.isfinite(raw).all() and np.isfinite(training).all()),
            "replay_consistency": self.replay_consistency(replay_buffer),
        }

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name: str):
        return getattr(self.env, name)
