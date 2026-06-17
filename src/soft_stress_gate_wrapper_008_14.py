from __future__ import annotations

from typing import Any

import numpy as np

from stage_action_wrapper import DapStage
from stress_aware_stage_action_wrapper import ForecastStressGateDapStageActionWrapper


class SoftStressForecastGateDapStageActionWrapper(ForecastStressGateDapStageActionWrapper):
    """Forecast gate without hard minimum irrigation, plus soft SWFAC penalty.

    This wrapper is intentionally scoped to 008_14. It keeps the useful
    forecast/stress gate audit trail from 008_09, but removes the hard
    `max(PPO_action, min_irrigation)` behavior that made PPO passive in 008_12.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        soft = self.config.get("soft_stress_reward", {})
        if bool(soft.get("enabled", False)):
            threshold = float(soft.get("swfac_threshold", 0.05))
            swfac_values = np.asarray([row.get("swfac", 0.0) for row in self.last_stage_records], dtype=float)
            excess = np.maximum(swfac_values - threshold, 0.0)
            stress_days = int((swfac_values > threshold).sum())
            stress_sum = float(excess.sum())
            penalty = (
                float(soft.get("swfac_excess_cost", 0.0)) * stress_sum
                + float(soft.get("swfac_day_cost", 0.0)) * stress_days
            )
            reward = float(reward) - penalty
            self.last_action_info.update(
                {
                    "soft_swfac_threshold": threshold,
                    "soft_swfac_excess_sum": stress_sum,
                    "soft_swfac_stress_days": stress_days,
                    "soft_swfac_penalty": penalty,
                    "reward_after_soft_stress": float(reward),
                }
            )
        return obs, float(reward), terminated, truncated, info

    def _stage_action_to_real(self, action, stage: DapStage) -> tuple[float, float, float]:
        requested_irrigation, requested_nitrogen, extra_nitrogen = super(
            ForecastStressGateDapStageActionWrapper, self
        )._stage_action_to_real(action, stage)
        irrigation_before_gate = float(requested_irrigation)
        dap = self._dap_from_obs(self.last_obs_dict)
        swfac_now = self._scalar_obs("swfac", 0.0)
        future_rain = self._future_rain(dap)
        rain_threshold = self.stage_rain_thresholds.get(stage.stage_id, 0.0)
        swfac_trigger = bool(swfac_now > self.swfac_threshold)
        forecast_trigger = bool(np.isfinite(future_rain) and future_rain < rain_threshold)
        allowed = True
        reason = "forecast_gate_disabled"

        if self.forecast_gate_enabled:
            if stage.stage_id in self.always_block_stages:
                allowed = False
                reason = f"{stage.stage_id}_blocked"
            else:
                allowed = swfac_trigger or forecast_trigger
                triggers = []
                if swfac_trigger:
                    triggers.append("swfac")
                if forecast_trigger:
                    triggers.append("forecast")
                reason = f"{stage.stage_id}_allowed_by_{'+'.join(triggers)}" if allowed else f"{stage.stage_id}_blocked_no_stress_or_dry_forecast"
            if not allowed:
                requested_irrigation = 0.0

        self._last_gate_info = {
            "forecast_stress_gate_enabled": self.forecast_gate_enabled,
            "forecast_gate_reason": reason,
            "irrigation_allowed_by_forecast_gate": bool(allowed),
            "gate_swfac_at_decision": float(swfac_now),
            "gate_future_rain_lookahead_days": self.lookahead_days,
            "gate_future_rain": float(future_rain) if np.isfinite(future_rain) else np.nan,
            "gate_future_rain_threshold": float(rain_threshold),
            "gate_swfac_trigger": swfac_trigger,
            "gate_forecast_trigger": forecast_trigger,
            "gate_min_irrigation_applied": 0.0,
            "irrigation_before_gate": irrigation_before_gate,
            "irrigation_after_gate": float(requested_irrigation),
            "irrigation_blocked_by_gate": bool(irrigation_before_gate > 1e-9 and not allowed),
            "hard_minimum_gate_removed": True,
        }
        return float(requested_irrigation), float(requested_nitrogen), float(extra_nitrogen)

