"""Validated constraint metadata for water-nitrogen management."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ConstraintSpec:
    """Seasonal budgets, per-operation caps, and the shared action window."""

    irrigation_budget: float
    nitrogen_budget: float
    daily_irrigation_cap: float
    daily_nitrogen_cap: float
    min_interval_days: int
    decision_window_start_dap: int
    decision_window_end_dap: int
    interval_semantics: str = "shared_water_nitrogen_last_operation_dap"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML serializable representation."""

        return asdict(self)


def build_constraint_spec(config: dict[str, Any]) -> ConstraintSpec:
    """Construct constraints without changing the frozen wrapper semantics."""

    values = config.get("constraints", config)
    return ConstraintSpec(
        irrigation_budget=float(values["irrigation_budget"]),
        nitrogen_budget=float(values["nitrogen_budget"]),
        daily_irrigation_cap=float(values["daily_irrigation_cap"]),
        daily_nitrogen_cap=float(values["daily_nitrogen_cap"]),
        min_interval_days=int(values["min_interval_days"]),
        decision_window_start_dap=int(values["decision_window_start_dap"]),
        decision_window_end_dap=int(values["decision_window_end_dap"]),
    )

