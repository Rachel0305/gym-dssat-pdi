"""Evaluation adapters and the unified daily/season schemas."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .environment_adapter import PreparedCase, load_prepared_case


LOGGER = logging.getLogger(__name__)

DAILY_FIELDS = [
    "experiment_id",
    "config_hash",
    "station_code",
    "year",
    "seed",
    "scenario",
    "checkpoint",
    "dap",
    "date",
    "istage",
    "vstage",
    "topwt",
    "grnwt",
    "xlai",
    "swfac",
    "nstres",
    "totir",
    "cumulative_irrigation",
    "cumulative_nitrogen",
    "action_irrigation",
    "action_nitrogen",
    "raw_action",
    "executed_action",
    "reward",
    "terminal_reward",
    "final_yield",
]

SEASON_FIELDS = [
    "experiment_id",
    "config_hash",
    "station_code",
    "year",
    "seed",
    "scenario",
    "checkpoint",
    "yield_kg_ha",
    "biomass_kg_ha",
    "irrigation_mm",
    "nitrogen_kg_ha",
    "reward_total",
    "number_of_irrigation_events",
    "number_of_nitrogen_events",
    "water_budget_use_ratio",
    "nitrogen_budget_use_ratio",
    "et_mm",
    "nitrogen_uptake_kg_ha",
    "WP_ET_kg_m3",
    "IWP_gross_kg_m3",
    "PFP_N_kg_kg",
    "NUtE_kg_kg",
    "yield_per_irrigation",
    "yield_per_nitrogen",
]


def _series(frame: pd.DataFrame, names: list[str], default: Any = np.nan) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series(default, index=frame.index)


def _safe_ratio(numerator: float, denominator: float, factor: float = 1.0) -> float:
    if not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator * factor)


def standardize_daily(
    frame: pd.DataFrame,
    *,
    experiment_id: str,
    config_hash: str,
    station_code: str,
    year: int,
    seed: int,
    checkpoint: int,
    final_yield: float,
    scenario: str = "dqn",
) -> pd.DataFrame:
    """Map legacy DQN trajectory columns into the benchmark schema."""

    output = pd.DataFrame(index=frame.index)
    output["experiment_id"] = experiment_id
    output["config_hash"] = config_hash
    output["station_code"] = station_code
    output["year"] = int(year)
    output["seed"] = int(seed)
    output["scenario"] = scenario
    output["checkpoint"] = int(checkpoint)
    output["dap"] = pd.to_numeric(_series(frame, ["dap", "operation_dap"]), errors="coerce")
    output["date"] = _series(frame, ["date"], pd.NA)
    for name in ("istage", "vstage", "topwt", "grnwt", "xlai", "swfac", "nstres", "totir"):
        output[name] = pd.to_numeric(_series(frame, [name]), errors="coerce")
    output["action_irrigation"] = pd.to_numeric(
        _series(frame, ["irrigation_mm", "action_irrigation", "executed_irrigation"]), errors="coerce"
    ).fillna(0.0)
    output["action_nitrogen"] = pd.to_numeric(
        _series(frame, ["fertilizer_kg_ha", "action_nitrogen", "executed_nitrogen"]), errors="coerce"
    ).fillna(0.0)
    output["cumulative_irrigation"] = output["action_irrigation"].cumsum()
    output["cumulative_nitrogen"] = output["action_nitrogen"].cumsum()
    output["raw_action"] = _series(frame, ["action_index", "raw_action"], pd.NA).astype("string")
    output["executed_action"] = [
        json.dumps({"amir": float(i), "anfer": float(n)}, ensure_ascii=False)
        for i, n in zip(output["action_irrigation"], output["action_nitrogen"])
    ]
    output["reward"] = pd.to_numeric(_series(frame, ["reward"]), errors="coerce").fillna(0.0)
    output["terminal_reward"] = pd.to_numeric(
        _series(frame, ["yield_gain", "terminal_reward"]), errors="coerce"
    ).fillna(0.0)
    output["final_yield"] = float(final_yield)
    return output.reindex(columns=DAILY_FIELDS)


def standardize_season(
    summary: dict[str, Any],
    daily: pd.DataFrame,
    *,
    experiment_id: str,
    config_hash: str,
    station_code: str,
    year: int,
    seed: int,
    checkpoint: int,
    config: dict[str, Any],
    scenario: str = "dqn",
) -> pd.DataFrame:
    """Create a one-row season summary and apply explicit zero-input NA rules."""

    yield_value = float(summary.get("final_grain_kg_ha", summary.get("yield_kg_ha", np.nan)))
    biomass = float(summary.get("final_biomass_kg_ha", summary.get("biomass_kg_ha", np.nan)))
    irrigation = float(summary.get("action_irrigation_total_mm", summary.get("irrigation_mm", 0.0)))
    nitrogen = float(summary.get("action_nitrogen_total_kg_ha", summary.get("nitrogen_kg_ha", 0.0)))
    reward_total = float(summary.get("total_reward", pd.to_numeric(daily.get("reward"), errors="coerce").sum()))
    actual_et = float(summary.get("et_mm", summary.get("actual_et_mm", np.nan)))
    crop_n = float(summary.get("nitrogen_uptake_kg_ha", summary.get("crop_n_uptake_kg_ha", np.nan)))
    constraints = config["constraints"]
    row = {
        "experiment_id": experiment_id,
        "config_hash": config_hash,
        "station_code": station_code,
        "year": int(year),
        "seed": int(seed),
        "scenario": scenario,
        "checkpoint": int(checkpoint),
        "yield_kg_ha": yield_value,
        "biomass_kg_ha": biomass,
        "irrigation_mm": irrigation,
        "nitrogen_kg_ha": nitrogen,
        "reward_total": reward_total,
        "number_of_irrigation_events": int((daily["action_irrigation"] > 0).sum()),
        "number_of_nitrogen_events": int((daily["action_nitrogen"] > 0).sum()),
        "water_budget_use_ratio": _safe_ratio(irrigation, float(constraints["irrigation_budget"])),
        "nitrogen_budget_use_ratio": _safe_ratio(nitrogen, float(constraints["nitrogen_budget"])),
        "et_mm": actual_et,
        "nitrogen_uptake_kg_ha": crop_n,
        "WP_ET_kg_m3": _safe_ratio(yield_value, actual_et, 0.1),
        "IWP_gross_kg_m3": _safe_ratio(yield_value, irrigation, 0.1),
        "PFP_N_kg_kg": _safe_ratio(yield_value, nitrogen),
        "NUtE_kg_kg": _safe_ratio(yield_value, crop_n),
        "yield_per_irrigation": _safe_ratio(yield_value, irrigation),
        "yield_per_nitrogen": _safe_ratio(yield_value, nitrogen),
    }
    return pd.DataFrame([row], columns=SEASON_FIELDS)


def evaluate_model_file(
    config: dict[str, Any],
    *,
    run_dir: Path,
    model_path: Path,
    checkpoint: int,
    experiment_id: str,
    config_hash: str,
    year: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate an existing model using the same-input prepared case."""

    root = Path(config.get("_project_root", Path(__file__).resolve().parents[1]))
    for value in (root, root / "src"):
        if str(value) not in sys.path:
            sys.path.insert(0, str(value))
    import run_yc_fq_frozen_nstep_cross_site_020_12 as legacy
    from frozen_nstep_dqn_config_020_11 import apply_environment_constants
    from stable_baselines3 import DQN
    import run_yc2014_linked_dqn_5k_multiseed_013_07 as shared

    apply_environment_constants(shared)
    prepared: PreparedCase = load_prepared_case(config, year, run_dir)
    spec = legacy.SiteSpec(
        code=prepared.station_code,
        station=prepared.station_name,
        year=prepared.year,
        treatment=prepared.treatment,
        input_root=prepared.dqn_filex.parent,
        mzx_name=prepared.dqn_filex.name,
        weather_name="",
        soil_id="",
    )
    null_summary_path = run_dir / "null_evaluation" / "null_summary.csv"
    if not null_summary_path.exists():
        raise FileNotFoundError(f"Local null result is missing: {null_summary_path}")
    null_yield = float(pd.read_csv(null_summary_path).iloc[0]["final_grain_kg_ha"])
    env = legacy.make_train_env(prepared.dqn_env_args, null_yield)
    model = DQN.load(str(model_path), env=env)
    env.close()
    evaluation_dir = run_dir / "evaluations" / f"checkpoint_{int(checkpoint)}"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    raw_daily, raw_summary, _audit = legacy.evaluate_checkpoint(
        model, spec, prepared.dqn_env_args, null_yield, int(checkpoint), evaluation_dir
    )
    final_yield = float(raw_summary["final_grain_kg_ha"])
    daily = standardize_daily(
        raw_daily,
        experiment_id=experiment_id,
        config_hash=config_hash,
        station_code=prepared.station_code,
        year=year,
        seed=seed,
        checkpoint=checkpoint,
        final_yield=final_yield,
    )
    season = standardize_season(
        raw_summary,
        daily,
        experiment_id=experiment_id,
        config_hash=config_hash,
        station_code=prepared.station_code,
        year=year,
        seed=seed,
        checkpoint=checkpoint,
        config=config,
    )
    return daily, season
