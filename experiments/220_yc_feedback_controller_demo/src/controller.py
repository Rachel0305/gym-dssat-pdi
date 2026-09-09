"""Pure daily feedback-controller logic for experiment 220.

The controller deliberately knows nothing about DSSAT, gym, or optimization.
This makes threshold/action semantics unit-testable before any simulator run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass(frozen=True)
class ControllerParams:
    n_threshold: float
    n_dose: int
    n_min_interval_days: int
    n_season_budget: int
    n_start_dap: int
    n_stop_dap: int
    water_threshold: float
    irrigation_dose: int
    irrigation_min_interval_days: int
    irrigation_season_budget: int
    irrigation_start_dap: int
    irrigation_stop_dap: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionDecision:
    dap: int
    swfac: float
    nstres: float
    water_window_ok: bool
    n_window_ok: bool
    water_condition: bool
    n_condition: bool
    water_interval_ok: bool
    n_interval_ok: bool
    water_budget_ok: bool
    n_budget_ok: bool
    irrigation_action_mm: float
    nitrogen_action_kg_ha: float
    cumulative_irrigation_mm: float
    cumulative_controller_n_kg_ha: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


class DailyStateFeedbackController:
    """Independent water and N threshold controllers.

    The comparison is ``stress >= threshold`` because this project exposes
    post-processed stress indices where larger values indicate stronger stress.
    Water and N keep separate last-event and budget states.
    """

    def __init__(self, params: ControllerParams):
        self.params = params
        self.reset()

    def reset(self) -> None:
        self.last_irrigation_dap: int | None = None
        self.last_n_dap: int | None = None
        self.cumulative_irrigation = 0.0
        self.cumulative_controller_n = 0.0

    @staticmethod
    def _interval_ok(dap: int, last_dap: int | None, minimum: int) -> bool:
        return last_dap is None or int(dap) - int(last_dap) >= int(minimum)

    def decide(self, dap: int, swfac: float, nstres: float) -> ActionDecision:
        dap = int(dap)
        swfac = float(swfac)
        nstres = float(nstres)
        water_window_ok = self.params.irrigation_start_dap <= dap <= self.params.irrigation_stop_dap
        n_window_ok = self.params.n_start_dap <= dap <= self.params.n_stop_dap
        water_condition = _finite(swfac) and swfac >= float(self.params.water_threshold)
        n_condition = _finite(nstres) and nstres >= float(self.params.n_threshold)
        water_interval_ok = self._interval_ok(dap, self.last_irrigation_dap, self.params.irrigation_min_interval_days)
        n_interval_ok = self._interval_ok(dap, self.last_n_dap, self.params.n_min_interval_days)
        water_budget_ok = self.cumulative_irrigation + float(self.params.irrigation_dose) <= float(self.params.irrigation_season_budget) + 1e-9
        n_budget_ok = self.cumulative_controller_n + float(self.params.n_dose) <= float(self.params.n_season_budget) + 1e-9

        irrigation = float(self.params.irrigation_dose) if water_window_ok and water_condition and water_interval_ok and water_budget_ok else 0.0
        nitrogen = float(self.params.n_dose) if n_window_ok and n_condition and n_interval_ok and n_budget_ok else 0.0
        if irrigation > 0:
            self.cumulative_irrigation += irrigation
            self.last_irrigation_dap = dap
        if nitrogen > 0:
            self.cumulative_controller_n += nitrogen
            self.last_n_dap = dap
        return ActionDecision(
            dap=dap,
            swfac=swfac,
            nstres=nstres,
            water_window_ok=water_window_ok,
            n_window_ok=n_window_ok,
            water_condition=water_condition,
            n_condition=n_condition,
            water_interval_ok=water_interval_ok,
            n_interval_ok=n_interval_ok,
            water_budget_ok=water_budget_ok,
            n_budget_ok=n_budget_ok,
            irrigation_action_mm=irrigation,
            nitrogen_action_kg_ha=nitrogen,
            cumulative_irrigation_mm=self.cumulative_irrigation,
            cumulative_controller_n_kg_ha=self.cumulative_controller_n,
        )
