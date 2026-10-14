"""152E3: matched E2 dual-branch no-forecast control.

The 12-dimensional weather branch is retained for architecture matching, but
every weather value is fixed at zero. Reward, action grid, masks, and DSSAT
inputs remain inherited from the accepted E2/no-forecast engine.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import forecast_engineered_observation_056_057 as forecast
import run_all_year_direct_action_safe_ppo as direct_ppo
from run_141E1_sya_actionable_weather_encoding_smoke2k import patch_contract
from run_142E2_sya_dual_branch_weather_reader_smoke2k import patched_maskableppo


CONFIG = ROOT / "configs/152E3_sya_originIC_dual_branch_noforecast_control_5k_seed0.json"
PROMPT = ROOT / "prompts/2026-08-16_sya_E3_dual_branch_noforecast_control_5k.md"
EXPECTED_TASK_ID = "152E3"
STATION = "SYA"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def e3_validate(cfg: dict[str, Any], expected_task_id: str | None = None) -> dict[str, Any]:
    if expected_task_id is not None and str(cfg.get("task_id")) != expected_task_id:
        raise ValueError(f"Expected task_id={expected_task_id}, got {cfg.get('task_id')}")
    if cfg.get("input_profile") != "originIC" or cfg.get("station_code") != STATION:
        raise ValueError("E3 must remain SYA originIC.")
    if cfg.get("actions", {}).get("irrigation_levels_mm") != [0.0, 15.0, 30.0, 45.0]:
        raise ValueError("E3 irrigation grid mismatch.")
    if cfg.get("actions", {}).get("nitrogen_levels_kg_ha") != [0.0, 40.0, 80.0, 120.0]:
        raise ValueError("E3 nitrogen grid mismatch.")
    obs = cfg.get("observation_contract", {})
    if obs.get("base") != "046_02_raw_observation" or bool(obs.get("normalization_enabled")):
        raise ValueError("E3 base observation contract mismatch.")
    if bool(obs.get("weather_forecast_enabled")) or obs.get("weather_forecast_mode") != "matched_architecture_zero_weather_control":
        raise ValueError("E3 must have no weather signal.")
    arch = cfg.get("policy_architecture", {})
    expected_arch = {
        "base_observation_dim": 25,
        "weather_observation_dim": 12,
        "base_latent_dim": 64,
        "weather_latent_dim": 32,
        "combined_features_dim": 96,
        "downstream_net_arch": [64, 64],
    }
    for key, value in expected_arch.items():
        if arch.get(key) != value:
            raise ValueError(f"E3 architecture mismatch at {key}: {arch.get(key)}")
    if cfg.get("training") != {"total_timesteps": 5000, "checkpoint_steps": [2000, 5000]}:
        raise ValueError("E3 training contract mismatch.")
    return cfg


class ZeroWeatherDualBranchWrapper(gym.Env):
    """Append twelve constant-zero values to preserve E2 observation width."""

    def __init__(self, env: gym.Env):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})
        base_space = env.observation_space
        low = np.asarray(base_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(base_space.high, dtype=np.float32).reshape(-1)
        self.base_observation_dim = int(len(low))
        self.weather_observation_dim = 12
        self.enhanced_observation_dim = self.base_observation_dim + self.weather_observation_dim
        self.observation_space = spaces.Box(
            low=np.concatenate([low, np.full(self.weather_observation_dim, -5.0, dtype=np.float32)]),
            high=np.concatenate([high, np.full(self.weather_observation_dim, 5.0, dtype=np.float32)]),
            dtype=np.float32,
        )
        self.last_raw_forecast_features = {name: 0.0 for name in forecast.RAW_FORECAST_FEATURE_NAMES}
        self.last_norm_forecast_features = {name: 0.0 for name in forecast.FORECAST_FEATURE_NAMES}
        self.last_raw_enhanced_obs: np.ndarray | None = None
        # The shared observation smoke gate probes this optional attribute.
        # E3 deliberately has no forecast date/signal, so expose a stable
        # null value instead of forwarding the lookup to the DSSAT env.
        self.last_observation_date = None

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        if len(flat) != self.base_observation_dim:
            raise RuntimeError(f"Expected base obs len {self.base_observation_dim}, got {len(flat)}")
        out = np.concatenate([flat, np.zeros(self.weather_observation_dim, dtype=np.float32)]).astype(np.float32)
        if not np.isfinite(out).all():
            raise RuntimeError("E3 observation contains NaN or Inf.")
        self.last_raw_enhanced_obs = out
        return out

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return self._augment(obs), dict(info or {})

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, dict(info or {})

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()

    @property
    def last_action_info(self) -> dict[str, Any]:
        info = dict(getattr(self.env, "last_action_info", {}) or {})
        for key, value in self.last_raw_forecast_features.items():
            info[f"forecast_raw_{key}"] = 0.0
        for key, value in self.last_norm_forecast_features.items():
            info[f"forecast_obs_{key}"] = 0.0
        info.update({
            "forecast_observation_enabled_056_057": False,
            "forecast_mode_056_057": "matched_architecture_zero_weather_control",
            "forecast_signal_all_zero": True,
            "forecast_base_observation_dim_056_057": self.base_observation_dim,
            "forecast_enhanced_observation_dim_056_057": self.enhanced_observation_dim,
        })
        return info

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)


def make_zero_weather_factory(base_make_env):
    def make_env(runtime_config, env_config, station, year, seed, run_tag, evaluation=False):
        env = base_make_env(runtime_config, env_config, station, year, seed, run_tag, evaluation=evaluation)
        return ZeroWeatherDualBranchWrapper(env)

    return make_env


def e3_calendar_audit(cfg: dict[str, Any], out: Path) -> Path:
    rows = []
    for year in list(map(int, cfg["scope"]["validation_years"][:2])):
        for dap in [1, 15, 30, 60, 90]:
            rows.append({
                "year": year,
                "dap": dap,
                "weather_signal_present": False,
                "weather_branch_values_all_zero": True,
                **{name: 0.0 for name in forecast.RAW_FORECAST_FEATURE_NAMES},
                **{name: 0.0 for name in forecast.FORECAST_FEATURE_NAMES},
            })
    path = out / "audits" / f"{cfg['task_id']}_zero_weather_calendar_audit.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _on_grid(series: pd.Series, allowed: list[float]) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return values.map(lambda value: bool(np.isclose(float(value), allowed, atol=1e-6).any()))


def e3_audit(out: Path, cfg: dict[str, Any]):
    prefix = str(cfg["task_id"])
    summary_path = out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path, keep_default_na=False)
    required = [f"forecast_obs_{name}" for name in forecast.FORECAST_FEATURE_NAMES] + [f"forecast_raw_{name}" for name in forecast.RAW_FORECAST_FEATURE_NAMES]
    rows = []
    for rec in summary.itertuples(index=False):
        daily_path = ROOT / str(getattr(rec, "daily_csv_path"))
        if not daily_path.exists():
            rows.append({"checkpoint_step": int(rec.checkpoint_step), "year": int(rec.year), "daily_exists": False})
            continue
        daily = pd.read_csv(daily_path)
        irr = pd.to_numeric(daily.get("safe_action_amir"), errors="coerce").fillna(0.0)
        nit = pd.to_numeric(daily.get("safe_action_anfer"), errors="coerce").fillna(0.0)
        dap = pd.to_numeric(daily.get("dap"), errors="coerce")
        zero_ok = all(col in daily.columns and np.isclose(pd.to_numeric(daily[col], errors="coerce").fillna(0.0), 0.0, atol=1e-9).all() for col in required)
        positive = (irr > 1e-9) | (nit > 1e-9)
        rows.append({
            "checkpoint_step": int(rec.checkpoint_step),
            "year": int(rec.year),
            "daily_exists": True,
            "row_count": int(len(daily)),
            "off_grid_irrigation_rows": int((~_on_grid(irr, [0.0, 15.0, 30.0, 45.0])).sum()),
            "off_grid_nitrogen_rows": int((~_on_grid(nit, [0.0, 40.0, 80.0, 120.0])).sum()),
            "positive_action_rows": int(positive.sum()),
            "positive_action_rows_after_dap1": int((positive & (dap > 1)).sum()),
            "forecast_required_column_count": len(required),
            "forecast_present_column_count": int(sum(col in daily.columns for col in required)),
            "zero_weather_columns_all_zero": bool(zero_ok),
        })
    audit = pd.DataFrame(rows).sort_values(["checkpoint_step", "year"]).reset_index(drop=True)
    final_step = int(max(cfg["training"]["checkpoint_steps"]))
    final = audit[audit["checkpoint_step"].eq(final_step)]
    gate = {
        "final_checkpoint": final_step,
        "expected_validation_rows": len(cfg["scope"]["validation_years"]),
        "observed_validation_rows": int(len(final)),
        "all_daily_files_exist": bool(len(final) and final["daily_exists"].all()),
        "all_actions_on_declared_grid": bool(len(final) and final["off_grid_irrigation_rows"].sum() == 0 and final["off_grid_nitrogen_rows"].sum() == 0),
        "positive_actions_present": bool(len(final) and final["positive_action_rows"].sum() > 0),
        "not_all_positive_actions_at_dap1": bool(len(final) and final["positive_action_rows_after_dap1"].sum() > 0),
        "zero_weather_columns_present_and_all_zero": bool(len(final) and final["forecast_present_column_count"].min() == len(required) and final["zero_weather_columns_all_zero"].all()),
    }
    gate["next_step_allowed"] = bool(
        gate["observed_validation_rows"] == gate["expected_validation_rows"]
        and all(value for key, value in gate.items() if key not in {"final_checkpoint", "expected_validation_rows", "observed_validation_rows", "next_step_allowed"})
    )
    return audit, gate


def e3_write_record(cfg, pf, copied, audit, gate, phase, obs_smoke, calendar_audit, smoke_verification=None):
    out = forecast.output_root(cfg)
    doc = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
    lines = [
        f"# {cfg['task_id']} SYA originIC dual-branch 无天气信号对照记录",
        "",
        "## 设计",
        "",
        "- 保留 E2 dual-branch 结构：25 维基础分支 + 12 维天气分支，96 维 latent。",
        "- 12 个天气分支输入固定为 0，不读取任何天气数据。",
        "- reward、动作网格、安全约束、输入、年份和 PPO 设置保持不变。",
        f"- 当前阶段：`{phase}`；输出目录：`{rel(out)}`。",
        "",
        "## gate",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in gate.items())
    lines.extend([
        "",
        "## observation / zero-weather audit",
        "",
        audit.to_markdown(index=False),
        "",
        "## 解释边界",
        "",
        "- 这是 E2 的 matched-architecture no-forecast 对照，不是最终 forecast 优势证明。",
        "- 若 E2 相对本对照仍有优势，才说明天气信号可能带来额外价值。",
        "",
    ])
    if smoke_verification is not None:
        lines.extend(["## smoke verification", "", *(f"- `{k}`: `{v}`" for k, v in smoke_verification.items()), ""])
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines), encoding="utf-8")
    return doc


def run_phase(dry_run: bool, smoke: bool, formal: bool) -> dict[str, Any]:
    # E1 patch supplies the 12-feature names used by the saved E2 extractor;
    # the E3 wrapper never calls its weather feature functions.
    patch_contract()
    originals = {
        "validate": forecast.validate_forecast_config,
        "factory": forecast.make_forecast_env_factory,
        "calendar": forecast.forecast_calendar_audit,
        "audit": forecast.audit_actions_and_forecast_columns,
        "record": forecast.write_record,
    }
    forecast.validate_forecast_config = e3_validate
    forecast.make_forecast_env_factory = lambda base_make_env, _cfg: make_zero_weather_factory(base_make_env)
    forecast.forecast_calendar_audit = e3_calendar_audit
    forecast.audit_actions_and_forecast_columns = e3_audit
    forecast.write_record = e3_write_record
    try:
        with patched_maskableppo():
            result = forecast.run_forecast_experiment(CONFIG, PROMPT, EXPECTED_TASK_ID, dry_run, smoke, formal)
    finally:
        forecast.validate_forecast_config = originals["validate"]
        forecast.make_forecast_env_factory = originals["factory"]
        forecast.forecast_calendar_audit = originals["calendar"]
        forecast.audit_actions_and_forecast_columns = originals["audit"]
        forecast.write_record = originals["record"]
    result["e3_control"] = {
        "changed_factor": "weather_signal_zeroed_only_with_E2_dual_branch_architecture",
        "reward_changed": False,
        "action_evaluation_changed": False,
        "next_step_allowed": result.get("next_step_allowed", result.get("action_forecast_gate", {}).get("next_step_allowed", False)),
    }
    if smoke and result.get("output_root"):
        out = ROOT / str(result["output_root"])
        (out / "152E3_control_audit.json").write_text(json.dumps(result["e3_control"], ensure_ascii=False, indent=2), encoding="utf-8")
    if formal and result.get("output_root"):
        out = ROOT / str(result["output_root"])
        (out / "152E3_5k_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--dry-run", action="store_true")
    phase.add_argument("--smoke", action="store_true")
    phase.add_argument("--train5k", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        run_phase(True, False, False)
    elif args.smoke:
        run_phase(False, True, False)
    else:
        run_phase(False, False, True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
