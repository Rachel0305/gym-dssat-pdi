from __future__ import annotations

from typing import Any

import numpy as np

from soft_stress_gate_wrapper_008_14 import SoftStressForecastGateDapStageActionWrapper


class JointSoftStressForecastGateDapStageActionWrapper(SoftStressForecastGateDapStageActionWrapper):
    """009 wrapper for stage-level water-nitrogen PPO.

    It keeps the 008 no-hard-minimum forecast gate and soft SWFAC penalty, then:
    - adds a soft NSTRES penalty because PPO now controls supplemental nitrogen;
    - credits back the nitrogen cost for fixed base nitrogen, so the effective
      nitrogen cost mainly acts on PPO-controlled supplemental nitrogen.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        n_cost = float(self.reward_config.get("nitrogen_cost", 0.0))
        stage_n = float(self.last_action_info.get("stage_action_anfer", 0.0))
        base_n = float(self.last_action_info.get("stage_nitrogen_base", 0.0))
        base_n_applied = min(stage_n, base_n)
        extra_n_applied = max(0.0, stage_n - base_n_applied)
        base_n_cost_credit = n_cost * base_n_applied
        reward = float(reward) + base_n_cost_credit

        nsoft = self.config.get("soft_nstres_reward", {})
        nstres_penalty = 0.0
        nstres_sum = 0.0
        nstres_days = 0
        if bool(nsoft.get("enabled", False)):
            threshold = float(nsoft.get("nstres_threshold", 0.05))
            nstres_values = np.asarray([row.get("nstres", 0.0) for row in self.last_stage_records], dtype=float)
            excess = np.maximum(nstres_values - threshold, 0.0)
            nstres_days = int((nstres_values > threshold).sum())
            nstres_sum = float(excess.sum())
            nstres_penalty = (
                float(nsoft.get("nstres_excess_cost", 0.0)) * nstres_sum
                + float(nsoft.get("nstres_day_cost", 0.0)) * nstres_days
            )
            reward = float(reward) - nstres_penalty

        self.last_action_info.update(
            {
                "base_n_cost_credit": float(base_n_cost_credit),
                "ppo_extra_n_applied": float(extra_n_applied),
                "ppo_extra_n_cost_charged": float(n_cost * extra_n_applied),
                "soft_nstres_excess_sum": float(nstres_sum),
                "soft_nstres_stress_days": int(nstres_days),
                "soft_nstres_penalty": float(nstres_penalty),
                "reward_after_joint_soft_stress": float(reward),
            }
        )
        return obs, float(reward), terminated, truncated, info

