"""Frozen reward definition and a compatibility wrapper.

This module centralizes the existing reward exactly as implemented in the
validated 015_12/020_12 line. It intentionally does not retune coefficients or
introduce leaching penalties.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RewardSpec:
    """Local-null terminal yield gain minus water and nitrogen costs."""

    reward_type: str
    irrigation_cost: float
    nitrogen_cost: float
    yield_gain_coefficient: float
    leaching_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""

        return asdict(self)


def build_reward_spec(config: dict[str, Any]) -> RewardSpec:
    """Build the frozen reward metadata from configuration."""

    values = config.get("reward", config)
    return RewardSpec(
        reward_type=str(values["type"]),
        irrigation_cost=float(values["irrigation_cost"]),
        nitrogen_cost=float(values["nitrogen_cost"]),
        yield_gain_coefficient=float(values.get("yield_gain_coefficient", 1.0)),
        leaching_cost=float(values.get("leaching_cost", 0.0)),
    )


def local_null_terminal_reward(
    final_yield: float,
    null_yield: float,
    irrigation: float,
    nitrogen: float,
    terminal: bool,
    spec: RewardSpec,
) -> tuple[float, dict[str, float]]:
    """Compute the frozen reward and expose auditable components."""

    yield_gain = (
        max(0.0, float(final_yield) - float(null_yield))
        * spec.yield_gain_coefficient
        if terminal
        else 0.0
    )
    water_cost = spec.irrigation_cost * float(irrigation)
    nitrogen_cost = spec.nitrogen_cost * float(nitrogen)
    reward = yield_gain - water_cost - nitrogen_cost
    return float(reward), {
        "yield_gain": float(yield_gain),
        "water_cost_term": float(water_cost),
        "nitrogen_cost_term": float(nitrogen_cost),
        "baseline_relative_reward": float(reward),
    }

