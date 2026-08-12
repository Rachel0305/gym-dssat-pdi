"""Shared stress-response reward branch for 058/059.

This branch is intentionally narrower than the earlier weather-forecast branch:

* no weather features;
* no observation-space change;
* no action-grid change;
* no first-pass DAP90 nitrogen safety override.

It adds local reward shaping on top of the inherited 046_10/053_00 environment
so PPO can learn the process logic behind the strong NSTRES-threshold auto-N
baseline: respond when stress is present, avoid resource application when no
stress is present, and reduce persistent stress days.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
import run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36 as base04036


SMOKE_TIMESTEPS = 2_000
SMOKE_CHECKPOINTS = [1_000, 2_000]
EXPECTED_IRRIGATION_LEVELS = [0.0, 15.0, 30.0, 45.0]
EXPECTED_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
INPUT_PROFILES = {
    "originIC": ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013",
    "lowIC": ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def output_root(cfg: dict[str, Any]) -> Path:
    return ROOT / "benchmark_results" / f"{cfg['task_id']}_{cfg['task_name']}"


def task_prefix(cfg: dict[str, Any]) -> str:
    return str(cfg["task_id"])


def prepare_config(cfg: dict[str, Any], smoke: bool) -> dict[str, Any]:
    planned = json.loads(json.dumps(cfg))
    if smoke:
        planned["task_name"] = f"{planned['task_name']}_smoke2k"
        planned["training"] = {"total_timesteps": SMOKE_TIMESTEPS, "checkpoint_steps": SMOKE_CHECKPOINTS}
    return planned


def validate_config(cfg: dict[str, Any], expected_task_id: str | None = None) -> dict[str, Any]:
    if expected_task_id is not None and str(cfg.get("task_id")) != expected_task_id:
        raise ValueError(f"Expected task_id={expected_task_id}, got {cfg.get('task_id')}.")
    if cfg.get("input_profile") not in INPUT_PROFILES:
        raise ValueError(f"input_profile must be one of {list(INPUT_PROFILES)}.")
    actions = cfg.get("actions", {})
    irrigation = list(map(float, actions.get("irrigation_levels_mm", [])))
    nitrogen = list(map(float, actions.get("nitrogen_levels_kg_ha", [])))
    if irrigation != EXPECTED_IRRIGATION_LEVELS or nitrogen != EXPECTED_NITROGEN_LEVELS:
        raise ValueError("Stress-response branch must keep the accepted 16-action grid.")
    obs = cfg.get("observation_contract", {})
    if obs.get("base") != "046_02_raw_observation":
        raise ValueError("Stress-response branch must inherit the 046_02 raw observation base.")
    if bool(obs.get("normalization_enabled", True)) or bool(obs.get("weather_forecast_enabled", True)):
        raise ValueError("Stress-response branch must not add normalization or weather features.")
    reward_cfg = cfg.get("stress_response_reward", {})
    if not bool(reward_cfg.get("enabled", False)):
        raise ValueError("stress_response_reward.enabled must be true.")
    return cfg


class StressResponseRewardWrapper(base04036.LateIrrigationReserveMaskWrapper):
    """Add SWFAC/NSTRES response shaping to the inherited PPO wrapper."""

    active_reward_config: dict[str, Any] = {}

    def step(self, action):
        prev_obs = dict(getattr(self, "last_obs_dict", {}) or {})
        prev_dap_raw = _finite_float(prev_obs.get("dap", 1.0), 1.0)
        prev_dap = int(round(prev_dap_raw)) if prev_dap_raw > 0 else 1
        prev_swfac = _finite_float(prev_obs.get("swfac", 0.0), 0.0)
        prev_nstres = _finite_float(prev_obs.get("nstres", 0.0), 0.0)

        obs, reward, terminated, truncated, info = super().step(action)
        cfg = dict(self.active_reward_config or {})
        if not bool(cfg.get("enabled", False)):
            return obs, reward, terminated, truncated, info

        cur_obs = dict(getattr(self, "last_obs_dict", {}) or {})
        cur_swfac = _finite_float(cur_obs.get("swfac", prev_swfac), prev_swfac)
        cur_nstres = _finite_float(cur_obs.get("nstres", prev_nstres), prev_nstres)
        safe_i = _finite_float(self.last_action_info.get("safe_action_amir", 0.0), 0.0)
        safe_n = _finite_float(self.last_action_info.get("safe_action_anfer", 0.0), 0.0)

        reward_scale = float(cfg.get("reward_scale", self.config.get("reward", {}).get("reward_scale", 1.0)))
        water_thr = float(cfg.get("water_stress_threshold", 0.05))
        n_thr = float(cfg.get("nitrogen_stress_threshold", 0.05))
        water_start = int(cfg.get("water_response_start_dap", 31))
        water_end = int(cfg.get("water_response_end_dap", 150))
        n_start = int(cfg.get("nitrogen_response_start_dap", 31))
        n_end = int(cfg.get("nitrogen_response_end_dap", 90))

        components: dict[str, float] = {}
        unscaled = 0.0

        water_window = water_start <= prev_dap <= water_end
        n_window = n_start <= prev_dap <= n_end
        water_stressed = bool(prev_swfac >= water_thr)
        n_stressed = bool(prev_nstres >= n_thr)

        if water_window and water_stressed and safe_i > 1e-9:
            bonus = float(cfg.get("water_response_bonus_unscaled", 0.0)) * min(safe_i / 45.0, 1.0)
            unscaled += bonus
            components["stress_water_response_bonus_unscaled"] = bonus
        if water_window and water_stressed and safe_i <= 1e-9:
            penalty = float(cfg.get("missed_water_stress_penalty_unscaled", 0.0)) * max(prev_swfac - water_thr, 0.0)
            unscaled -= penalty
            components["missed_water_stress_penalty_unscaled"] = -penalty
        if water_window and (not water_stressed) and safe_i > 1e-9:
            penalty = float(cfg.get("low_stress_irrigation_penalty_per_mm_unscaled", 0.0)) * safe_i
            unscaled -= penalty
            components["low_stress_irrigation_penalty_unscaled"] = -penalty

        if n_window and n_stressed and safe_n > 1e-9:
            bonus = float(cfg.get("nitrogen_response_bonus_unscaled", 0.0)) * min(safe_n / 40.0, 1.0)
            unscaled += bonus
            components["stress_nitrogen_response_bonus_unscaled"] = bonus
        if n_window and n_stressed and safe_n <= 1e-9:
            penalty = float(cfg.get("missed_nitrogen_stress_penalty_unscaled", 0.0)) * max(prev_nstres - n_thr, 0.0)
            unscaled -= penalty
            components["missed_nitrogen_stress_penalty_unscaled"] = -penalty
        if n_window and (not n_stressed) and safe_n > 1e-9:
            penalty = float(cfg.get("low_stress_nitrogen_penalty_per_kg_unscaled", 0.0)) * safe_n
            unscaled -= penalty
            components["low_stress_nitrogen_penalty_unscaled"] = -penalty

        post_water = float(cfg.get("post_step_water_stress_penalty_coef_unscaled", 0.0)) * max(cur_swfac - water_thr, 0.0)
        post_n = float(cfg.get("post_step_nitrogen_stress_penalty_coef_unscaled", 0.0)) * max(cur_nstres - n_thr, 0.0)
        if post_water > 0:
            unscaled -= post_water
            components["post_step_water_stress_penalty_unscaled"] = -post_water
        if post_n > 0:
            unscaled -= post_n
            components["post_step_nitrogen_stress_penalty_unscaled"] = -post_n
        if prev_dap == 1 and (safe_i > 1e-9 or safe_n > 1e-9):
            penalty = float(cfg.get("dap1_positive_action_penalty_unscaled", 0.0))
            unscaled -= penalty
            components["dap1_positive_action_penalty_unscaled"] = -penalty

        shaped = unscaled * reward_scale
        shaped_reward = float(reward) + shaped
        self.last_action_info.update(
            {
                "stress_response_reward_enabled": True,
                "stress_response_prev_dap": prev_dap,
                "stress_response_prev_swfac": float(prev_swfac),
                "stress_response_prev_nstres": float(prev_nstres),
                "stress_response_cur_swfac": float(cur_swfac),
                "stress_response_cur_nstres": float(cur_nstres),
                "stress_response_water_threshold": water_thr,
                "stress_response_nitrogen_threshold": n_thr,
                "stress_response_water_window_active": bool(water_window),
                "stress_response_nitrogen_window_active": bool(n_window),
                "stress_response_water_stressed_before": bool(water_stressed),
                "stress_response_nitrogen_stressed_before": bool(n_stressed),
                "stress_response_reward_unscaled": float(unscaled),
                "stress_response_reward_scaled": float(shaped),
                "reward_before_stress_response": float(reward),
                "reward_after_stress_response": float(shaped_reward),
                **components,
            }
        )
        return obs, shaped_reward, terminated, truncated, info


def split_for_station(station: str) -> pd.DataFrame:
    split = engine.base03222.load_split().copy()
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    return split[split["station_code"].astype(str).eq(station)].sort_values("year").reset_index(drop=True)


def preflight(cfg: dict[str, Any], prompt: Path) -> dict[str, Any]:
    station = str(cfg["station_code"])
    site = str(cfg.get("site", station))
    input_root = INPUT_PROFILES[str(cfg["input_profile"])]
    source_mzx_name = "CNSY1201.MZX" if site == "SY" else f"CN{site}0801.MZX"
    source_mzx = input_root / site / source_mzx_name
    split = split_for_station(station)
    train_actual = sorted(split.loc[split["split"].eq("train"), "year"].astype(int).tolist())
    validation_actual = sorted(split.loc[split["split"].eq("validation"), "year"].astype(int).tolist())
    expected_train = list(map(int, cfg["scope"]["train_years"]))
    expected_validation = list(map(int, cfg["scope"]["validation_years"]))
    issues: list[str] = []
    if not input_root.exists():
        issues.append(f"input root missing: {rel(input_root)}")
    if not source_mzx.exists():
        issues.append(f"source MZX missing: {rel(source_mzx)}")
    if train_actual != expected_train:
        issues.append(f"train years mismatch: config={expected_train}, engine={train_actual}")
    if validation_actual != expected_validation:
        issues.append(f"validation years mismatch: config={expected_validation}, engine={validation_actual}")
    if not prompt.exists():
        issues.append(f"prompt missing: {rel(prompt)}")
    return {
        "task": f"{cfg['task_id']}_{cfg['task_name']}",
        "station_code": station,
        "site": site,
        "input_profile": cfg["input_profile"],
        "resolved_input_root": rel(input_root),
        "source_mzx": rel(source_mzx),
        "config_train_years": expected_train,
        "config_validation_years": expected_validation,
        "engine_train_years": train_actual,
        "engine_validation_years": validation_actual,
        "reference_run": cfg["reference_run"],
        "observation_contract": cfg["observation_contract"],
        "stress_response_reward": cfg["stress_response_reward"],
        "irrigation_levels_mm": cfg["actions"]["irrigation_levels_mm"],
        "nitrogen_levels_kg_ha": cfg["actions"]["nitrogen_levels_kg_ha"],
        "combined_action_count": len(cfg["actions"]["irrigation_levels_mm"]) * len(cfg["actions"]["nitrogen_levels_kg_ha"]),
        "issues": issues,
        "next_step_allowed": not issues,
    }


def make_stress_response_env_factory(base_make_env: Callable, exp_cfg: dict[str, Any]) -> Callable:
    def make_env(runtime_config: dict[str, Any], env_config: dict[str, Any], station: str, year: int, seed: int, run_tag: str, evaluation: bool = False):
        old = StressResponseRewardWrapper.active_reward_config
        StressResponseRewardWrapper.active_reward_config = dict(exp_cfg["stress_response_reward"])
        try:
            base = base_make_env(runtime_config, env_config, station, year, seed, run_tag, evaluation=evaluation)
            wrapped = StressResponseRewardWrapper(base.env, base.config)
            wrapped.grid = base.grid
            return wrapped
        finally:
            StressResponseRewardWrapper.active_reward_config = old

    return make_env


def copy_clean_names(out: Path, prefix: str) -> dict[str, str]:
    mapping = {
        "evaluation/042_10_training_checkpoint_inventory.csv": f"evaluation/{prefix}_training_checkpoint_inventory.csv",
        "evaluation/042_10_checkpoint_validation_summary.csv": f"evaluation/{prefix}_checkpoint_validation_summary.csv",
        "evaluation/042_10_validation_summary_by_station_checkpoint.csv": f"evaluation/{prefix}_validation_summary_by_station_checkpoint.csv",
        "042_10_result.json": f"{prefix}_engine_result.json",
    }
    copied: dict[str, str] = {}
    for source_rel, target_rel in mapping.items():
        source, target = out / source_rel, out / target_rel
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied[target_rel] = source_rel
    return copied


def _on_grid(values: pd.Series, allowed: list[float]) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return values.map(lambda value: bool(np.isclose(float(value), allowed, atol=1e-6).any()))


def audit_actions_and_reward(out: Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    prefix = task_prefix(cfg)
    summary_path = out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing validation summary: {summary_path}")
    summary = pd.read_csv(summary_path, keep_default_na=False)
    rows: list[dict[str, Any]] = []
    required_cols = [
        "stress_response_reward_unscaled",
        "stress_response_reward_scaled",
        "stress_response_prev_swfac",
        "stress_response_prev_nstres",
        "stress_response_water_stressed_before",
        "stress_response_nitrogen_stressed_before",
    ]
    for record in summary.itertuples(index=False):
        daily_path = ROOT / str(getattr(record, "daily_csv_path"))
        if not daily_path.exists():
            rows.append({"checkpoint_step": int(record.checkpoint_step), "year": int(record.year), "daily_exists": False})
            continue
        daily = pd.read_csv(daily_path)
        irrigation = pd.to_numeric(daily.get("safe_action_amir"), errors="coerce").fillna(0.0)
        nitrogen = pd.to_numeric(daily.get("safe_action_anfer"), errors="coerce").fillna(0.0)
        daps = pd.to_numeric(daily.get("dap"), errors="coerce")
        positive = (irrigation > 1e-9) | (nitrogen > 1e-9)
        present = [col for col in required_cols if col in daily.columns]
        reward_unscaled = pd.to_numeric(daily.get("stress_response_reward_unscaled", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        rows.append(
            {
                "checkpoint_step": int(record.checkpoint_step),
                "year": int(record.year),
                "daily_exists": True,
                "row_count": int(len(daily)),
                "off_grid_irrigation_rows": int((~_on_grid(irrigation, EXPECTED_IRRIGATION_LEVELS)).sum()),
                "off_grid_nitrogen_rows": int((~_on_grid(nitrogen, EXPECTED_NITROGEN_LEVELS)).sum()),
                "positive_action_rows": int(positive.sum()),
                "positive_action_rows_after_dap1": int((positive & (daps > 1)).sum()),
                "stress_reward_required_column_count": len(required_cols),
                "stress_reward_present_column_count": len(present),
                "stress_reward_nonzero_days": int((np.abs(reward_unscaled) > 1e-12).sum()),
                "stress_reward_sum_unscaled": float(reward_unscaled.sum()),
                "water_stress_action_days": int(((pd.to_numeric(daily.get("stress_response_prev_swfac", pd.Series(dtype=float)), errors="coerce") >= 0.05) & (irrigation > 0)).sum()) if "stress_response_prev_swfac" in daily.columns else 0,
                "nitrogen_stress_action_days": int(((pd.to_numeric(daily.get("stress_response_prev_nstres", pd.Series(dtype=float)), errors="coerce") >= 0.05) & (nitrogen > 0)).sum()) if "stress_response_prev_nstres" in daily.columns else 0,
            }
        )
    audit = pd.DataFrame(rows).sort_values(["checkpoint_step", "year"]).reset_index(drop=True)
    final_checkpoint = int(max(cfg["training"]["checkpoint_steps"]))
    final = audit[audit["checkpoint_step"].eq(final_checkpoint)].copy()
    gate = {
        "final_checkpoint": final_checkpoint,
        "expected_validation_rows": len(cfg["scope"]["validation_years"]),
        "observed_validation_rows": int(len(final)),
        "all_daily_files_exist": bool(len(final) and final["daily_exists"].all()),
        "all_actions_on_declared_grid": bool(len(final) and final["off_grid_irrigation_rows"].sum() == 0 and final["off_grid_nitrogen_rows"].sum() == 0),
        "positive_actions_present": bool(len(final) and final["positive_action_rows"].sum() > 0),
        "not_all_positive_actions_at_dap1": bool(len(final) and final["positive_action_rows_after_dap1"].sum() > 0),
        "stress_reward_columns_present": bool(len(final) and final["stress_reward_present_column_count"].min() == len(required_cols)),
        "stress_reward_active": bool(len(final) and final["stress_reward_nonzero_days"].sum() > 0),
    }
    gate["next_step_allowed"] = bool(
        gate["observed_validation_rows"] == gate["expected_validation_rows"]
        and all(value for key, value in gate.items() if key not in {"final_checkpoint", "expected_validation_rows", "observed_validation_rows", "next_step_allowed"})
    )
    return audit, gate


def verify_completed_smoke(base_cfg: dict[str, Any]) -> dict[str, Any]:
    smoke_cfg = prepare_config(base_cfg, smoke=True)
    smoke_out = output_root(smoke_cfg)
    result_path = smoke_out / f"{task_prefix(smoke_cfg)}_smoke_result.json"
    manifest_path = smoke_out / f"{task_prefix(smoke_cfg)}_run_manifest.json"
    checks: dict[str, Any] = {
        "smoke_result_exists": result_path.exists(),
        "smoke_manifest_exists": manifest_path.exists(),
    }
    if not all(checks.values()):
        checks["next_step_allowed"] = False
        return checks
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    smoke_config = manifest.get("config", {})
    gate = result.get("smoke_gate", {})
    checks.update(
        {
            "smoke_status_completed": manifest.get("status") == "completed_smoke",
            "smoke_gate_passed": gate.get("next_step_allowed") is True,
            "smoke_timesteps_match": smoke_config.get("training", {}).get("total_timesteps") == SMOKE_TIMESTEPS,
            "smoke_checkpoints_match": smoke_config.get("training", {}).get("checkpoint_steps") == SMOKE_CHECKPOINTS,
            "smoke_input_profile_match": smoke_config.get("input_profile") == base_cfg.get("input_profile"),
            "smoke_station_match": smoke_config.get("station_code") == base_cfg.get("station_code"),
            "smoke_actions_match": smoke_config.get("actions") == base_cfg.get("actions"),
            "smoke_reward_match": smoke_config.get("stress_response_reward") == base_cfg.get("stress_response_reward"),
            "smoke_output_root": rel(smoke_out),
        }
    )
    checks["next_step_allowed"] = all(value for key, value in checks.items() if key not in {"smoke_output_root", "next_step_allowed"})
    return checks


def write_manifest(out: Path, cfg_path: Path, prompt: Path, cfg: dict[str, Any], pf: dict[str, Any], status: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    config_dir = out / "configs"
    config_dir.mkdir(exist_ok=True)
    shutil.copy2(cfg_path, config_dir / cfg_path.name)
    if prompt.exists():
        shutil.copy2(prompt, config_dir / prompt.name)
    path = out / f"{task_prefix(cfg)}_run_manifest.json"
    path.write_text(json.dumps({"status": status, "config": cfg, "preflight": pf}, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_record(cfg: dict[str, Any], pf: dict[str, Any], copied: dict[str, str], audit: pd.DataFrame, gate: dict[str, Any], phase: str, smoke_verification: dict[str, Any] | None = None) -> Path:
    doc = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
    lines = [
        f"# {cfg['task_id']} {cfg['station_code']} stress-response reward MaskablePPO record",
        "",
        "## Design",
        "",
        f"- Reference run: `{cfg['reference_run']}`.",
        "- Controlled change: add local SWFAC/NSTRES response reward shaping.",
        "- Unchanged: weather features, observation contract, action grid, train/validation years, seed, and first-pass safety constraints.",
        f"- Reward config: `{cfg['stress_response_reward']}`.",
        f"- Output root: `{rel(output_root(cfg))}`.",
        "",
        f"## {phase} gate",
        "",
    ]
    for key, value in gate.items():
        lines.append(f"- `{key}`: `{value}`")
    if smoke_verification is not None:
        lines.extend(["", "## Formal smoke verification", ""])
        for key, value in smoke_verification.items():
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Year-level action and reward audit", "", audit.to_markdown(index=False), "", "## Normalized outputs", ""])
    for target, source in copied.items():
        lines.append(f"- `{target}` <- `{source}`")
    lines.append("")
    doc.write_text("\n".join(lines), encoding="utf-8")
    return doc


def run_experiment(cfg_path: Path, prompt: Path, expected_task_id: str, dry_run: bool, smoke: bool, formal: bool) -> dict[str, Any]:
    base_cfg = validate_config(read_json(cfg_path), expected_task_id=expected_task_id)
    cfg = prepare_config(base_cfg, smoke)
    pf = preflight(cfg, prompt)
    out = output_root(cfg)
    smoke_verification: dict[str, Any] | None = None
    if formal:
        smoke_verification = verify_completed_smoke(base_cfg)
        if not smoke_verification["next_step_allowed"]:
            pf["issues"].append(f"{expected_task_id} completed smoke verification did not pass")
            pf["next_step_allowed"] = False
    if dry_run:
        result = {**pf, "mode": "dry_run", "output_root": rel(out)}
        if smoke_verification is not None:
            result["smoke_verification"] = smoke_verification
        return result
    if not smoke and not formal:
        raise ValueError("Use --smoke for 2K validation or --formal for authorized 100K training.")
    if not pf["next_step_allowed"]:
        raise RuntimeError("Preflight failed: " + "; ".join(pf["issues"]))
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment output: {rel(out)}")

    old = {key: getattr(engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_site_names = dict(engine.base03222.SITE_NAMES)
    old_make_env = engine.base03222.base.make_env
    old_wrapper = engine.base03222.base.StressAwareDiscreteWrapper
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    make_env = make_stress_response_env_factory(old_make_env, cfg)
    old_active = StressResponseRewardWrapper.active_reward_config
    try:
        StressResponseRewardWrapper.active_reward_config = dict(cfg["stress_response_reward"])
        engine.TASK_ID = str(cfg["task_id"])
        engine.TASK_NAME = str(cfg["task_name"])
        engine.BASE_OUT = out
        engine.BASE_DOC = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
        engine.PROMPT = prompt
        engine.LOWIC_INPUT_ROOT = INPUT_PROFILES[str(cfg["input_profile"])]
        engine.STATION = str(cfg["station_code"])
        engine.SITES = [str(cfg["station_code"])]
        engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
        engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
        engine.base03222.SITE_NAMES[str(cfg["station_code"])] = str(cfg.get("site", cfg["station_code"]))
        engine.base03222.base.make_env = make_env
        engine.base03222.base.StressAwareDiscreteWrapper = StressResponseRewardWrapper
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = INPUT_PROFILES[str(cfg["input_profile"])]
        engine.run_training(int(cfg["training"]["total_timesteps"]), list(map(int, cfg["training"]["checkpoint_steps"])), suffix="")
    finally:
        StressResponseRewardWrapper.active_reward_config = old_active
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root
        engine.base03222.base.make_env = old_make_env
        engine.base03222.base.StressAwareDiscreteWrapper = old_wrapper
        for key, value in old.items():
            setattr(engine, key, value)
        engine.base03222.SITE_NAMES.clear()
        engine.base03222.SITE_NAMES.update(old_site_names)

    copied = copy_clean_names(out, task_prefix(cfg))
    audit, gate = audit_actions_and_reward(out, cfg)
    phase = "smoke" if smoke else "formal"
    audit_path = out / "audits" / f"{task_prefix(cfg)}_{phase}_action_stress_reward_audit.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    manifest = write_manifest(out, cfg_path, prompt, cfg, pf, f"completed_{phase}")
    record = write_record(cfg, pf, copied, audit, gate, "2K smoke" if smoke else "100K formal", smoke_verification)
    result = {
        **pf,
        "mode": f"{phase}_train_and_validate",
        "output_root": rel(out),
        "manifest": rel(manifest),
        "record_md": rel(record),
        "action_stress_reward_audit_path": rel(audit_path),
        "action_stress_reward_gate": gate,
    }
    if smoke:
        result["smoke_gate"] = gate
        result["next_step_allowed"] = gate["next_step_allowed"]
        result_path = out / f"{task_prefix(cfg)}_smoke_result.json"
    else:
        result["smoke_verification"] = smoke_verification
        result_path = out / f"{task_prefix(cfg)}_formal_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def cli(default_config: Path, default_prompt: Path, expected_task_id: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--dry-run", action="store_true")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--smoke", action="store_true")
    phase.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    print(json.dumps(run_experiment(cfg_path, default_prompt, expected_task_id, args.dry_run, args.smoke, args.formal), indent=2, ensure_ascii=False))
