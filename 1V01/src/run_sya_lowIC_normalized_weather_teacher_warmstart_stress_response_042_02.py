from __future__ import annotations

import argparse
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

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo_041_04 as base04104
import run_sya_lowIC_teacher_warmstart_maskableppo_041_03 as base04103
import run_sya_lowIC_ppo_yield_guardrail_v3_040_40 as base04040


TASK_ID = "042_02"
TASK_NAME = "sya_lowIC_normalized_weather_teacher_warmstart_stress_response"
BASE_OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
BASE_DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

LOWIC_INPUT_ROOT = base04040.LOWIC_INPUT_ROOT
STATION = "SYA"
SITE = "SY"
SEED = 0
DEFAULT_TIMESTEPS = 100_000
DEFAULT_CHECKPOINTS = [25_000, 50_000, 75_000, 100_000]

BASE_OBS_NAMES = [
    "cumsumfert",
    "dap",
    "dtt",
    "ep",
    "grnwt",
    "istage",
    "nstres",
    "rtdep",
    "srad",
    "sw_1",
    "sw_2",
    "sw_3",
    "sw_4",
    "sw_5",
    "sw_6",
    "sw_7",
    "sw_8",
    "sw_9",
    "swfac",
    "tmax",
    "topwt",
    "totir",
    "vstage",
    "wtdep",
    "xlai",
]
WEATHER_FEATURE_NAMES = [
    "rain_today_mm",
    "tmin_today_c",
    "rain_past7_mm",
    "rain_future7_mm",
    "tmean_future7_c",
]
ENHANCED_OBS_NAMES = BASE_OBS_NAMES + WEATHER_FEATURE_NAMES

# Fixed physical scales. These are intentionally not fitted from validation
# performance. Values are broad agronomic/DSSAT ranges used only to avoid raw
# mixed units entering the policy network.
SCALE = np.array(
    [
        300.0,  # cumsumfert kg/ha
        160.0,  # dap
        2500.0,  # accumulated thermal time proxy
        10.0,  # daily plant transpiration/evap proxy
        15000.0,  # grain weight kg/ha
        10.0,  # istage
        1.0,  # nstres
        200.0,  # root depth cm
        35.0,  # srad
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,  # soil water layers
        1.0,  # swfac
        45.0,  # tmax
        25000.0,  # topwt kg/ha
        300.0,  # totir mm
        20.0,  # vstage
        200.0,  # water table/depth proxy
        8.0,  # xlai
        100.0,  # rain today mm
        40.0,  # tmin C, clipped broad
        200.0,  # past 7 day rain
        200.0,  # future 7 day rain
        40.0,  # future 7 day mean temperature
    ],
    dtype=np.float32,
)

CLIP_LOW = np.full(len(ENHANCED_OBS_NAMES), -5.0, dtype=np.float32)
CLIP_HIGH = np.full(len(ENHANCED_OBS_NAMES), 5.0, dtype=np.float32)

NSTRES_THRESHOLD = 0.05
NSTRES_PENALTY_COEF = 50.0

ORIGINAL_04103_TASK_ID = base04103.TASK_ID
ORIGINAL_04103_TASK_NAME = base04103.TASK_NAME
ORIGINAL_04103_BASE_OUT = base04103.BASE_OUT
ORIGINAL_04103_BASE_DOC = base04103.BASE_DOC
ORIGINAL_04103_PROMPT = base04103.PROMPT
ORIGINAL_04103_OUT_FOR_SUFFIX = base04103.out_for_suffix
ORIGINAL_04103_DOC_FOR_SUFFIX = base04103.doc_for_suffix
ORIGINAL_04103_PRETRAIN = base04103.pretrain_policy_bc
ORIGINAL_04040_MAKE_ENV = base04040.make_env_with_yield_guardrail
ORIGINAL_04040_LOAD_CONFIG = base04040.load_config


def out_for_suffix(suffix: str) -> Path:
    return ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}_{suffix}" if suffix else BASE_OUT


def doc_for_suffix(suffix: str) -> Path:
    return ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_{suffix}_record.md" if suffix else BASE_DOC


def parse_checkpoint_steps(raw: str | None, timesteps: int) -> list[int]:
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return [x for x in DEFAULT_CHECKPOINTS if x <= int(timesteps)]


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def lowic_authoritative_audit_status() -> dict[str, Any]:
    issue_path = (
        ROOT
        / "benchmark_results"
        / "039_00_low_initial_soil_water_nitrogen_input_audit"
        / "tables"
        / "039_00_authoritative_template_issues.csv"
    )
    direct_path = (
        ROOT
        / "benchmark_results"
        / "039_00_low_initial_soil_water_nitrogen_input_audit"
        / "tables"
        / "039_00_authoritative_template_comparison.csv"
    )
    issues = _read_csv_or_empty(issue_path)
    direct = _read_csv_or_empty(direct_path)
    return {
        "issue_path": issue_path.relative_to(ROOT).as_posix(),
        "comparison_path": direct_path.relative_to(ROOT).as_posix(),
        "issue_count": int(len(issues)),
        "pass": bool(issue_path.exists() and len(issues) == 0),
        "comparison_rows": int(len(direct)),
    }


def load_config() -> dict[str, Any]:
    cfg = ORIGINAL_04040_LOAD_CONFIG()
    cfg["observation_normalization_04202"] = {
        "enabled": True,
        "base_observation_names": BASE_OBS_NAMES,
        "weather_feature_names": WEATHER_FEATURE_NAMES,
        "scale": {name: float(value) for name, value in zip(ENHANCED_OBS_NAMES, SCALE)},
        "clip_low": -5.0,
        "clip_high": 5.0,
        "fitted_from_validation_results": False,
    }
    cfg["weather_forecast_observation"] = {
        "enabled": True,
        "feature_names": WEATHER_FEATURE_NAMES,
        "past_window_days": 7,
        "future_window_days": 7,
        "future_weather_assumption": "perfect historical weather forecast",
        "normalized_by_04202": True,
    }
    cfg["reward"]["nstres_guardrail_process_penalty_04202"] = {
        "enabled": True,
        "threshold": NSTRES_THRESHOLD,
        "coef": NSTRES_PENALTY_COEF,
        "formula": "coef * max(0, nstres_after_step - threshold) * reward_scale",
        "reason": "symmetric nitrogen-stress process response to the existing SWFAC guardrail; fixed before training.",
    }
    return cfg


def _weather_table(config: dict[str, Any], station: str) -> pd.DataFrame:
    weather = direct_ppo.weather_for_daily(config)
    weather = weather[weather["station_code"].astype(str).eq(station)].copy()
    weather["date"] = pd.to_datetime(weather["date"])
    for col in ["rain", "tmin", "tmax"]:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")
    return weather.sort_values("date").reset_index(drop=True)


class NormalizedWeatherStressRewardWrapper(gym.Env):
    """Append weather/forecast features, normalize all inputs, and add NSTRES penalty."""

    def __init__(self, env: gym.Env, config: dict[str, Any], env_config: dict[str, Any], station: str, year: int):
        super().__init__()
        self.env = env
        self.config = config
        self.env_config = env_config
        self.station = str(station)
        self.year = int(year)
        self.weather = _weather_table(config, self.station)
        self.planting = pd.Timestamp(direct_ppo.find_year(env_config, self.station, self.year)["planting_date"])
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})
        base_space = env.observation_space
        self.base_observation_dim = int(np.asarray(base_space.low).reshape(-1).shape[0])
        if self.base_observation_dim != len(BASE_OBS_NAMES):
            raise RuntimeError(f"Expected base observation dim {len(BASE_OBS_NAMES)}, got {self.base_observation_dim}")
        self.enhanced_observation_dim = len(ENHANCED_OBS_NAMES)
        self.observation_space = spaces.Box(low=CLIP_LOW.copy(), high=CLIP_HIGH.copy(), dtype=np.float32)
        self.last_weather_features: dict[str, float] = {}
        self.last_raw_enhanced_obs: np.ndarray | None = None
        self.last_normalized_obs: np.ndarray | None = None

    def _date_from_obs(self, obs: np.ndarray, info: dict | None = None) -> pd.Timestamp:
        latest = direct_ppo.latest_observation_dict(self.env, obs, info or {})
        dap_raw = direct_ppo.scalar(latest.get("dap", np.nan), np.nan)
        if np.isfinite(dap_raw) and dap_raw > 0:
            return self.planting + pd.Timedelta(days=max(int(round(dap_raw)) - 1, 0))
        return self.planting

    def _features_for_date(self, date: pd.Timestamp) -> np.ndarray:
        w = self.weather
        today = w[w["date"].eq(date)]
        if today.empty:
            rain_today = 0.0
            tmin_today = 0.0
        else:
            rain_today = float(today["rain"].iloc[0]) if pd.notna(today["rain"].iloc[0]) else 0.0
            tmin_today = float(today["tmin"].iloc[0]) if pd.notna(today["tmin"].iloc[0]) else 0.0
        past_start = date - pd.Timedelta(days=6)
        future_end = date + pd.Timedelta(days=6)
        past = w[(w["date"] >= past_start) & (w["date"] <= date)]
        future = w[(w["date"] >= date) & (w["date"] <= future_end)]
        rain_past7 = float(pd.to_numeric(past["rain"], errors="coerce").fillna(0).sum()) if not past.empty else 0.0
        rain_future7 = float(pd.to_numeric(future["rain"], errors="coerce").fillna(0).sum()) if not future.empty else 0.0
        if not future.empty:
            tmean_daily = (pd.to_numeric(future["tmax"], errors="coerce") + pd.to_numeric(future["tmin"], errors="coerce")) / 2.0
            tmean_future7 = float(tmean_daily.mean()) if tmean_daily.notna().any() else 0.0
        else:
            tmean_future7 = 0.0
        vals = np.array([rain_today, tmin_today, rain_past7, rain_future7, tmean_future7], dtype=np.float32)
        self.last_weather_features = {name: float(value) for name, value in zip(WEATHER_FEATURE_NAMES, vals)}
        return vals

    def _normalize(self, raw_enhanced: np.ndarray) -> np.ndarray:
        safe = np.nan_to_num(np.asarray(raw_enhanced, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        normalized = safe / SCALE
        return np.clip(normalized, CLIP_LOW, CLIP_HIGH).astype(np.float32)

    def _augment_and_normalize(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        if flat.shape[0] != len(BASE_OBS_NAMES):
            raise RuntimeError(f"Expected raw obs len {len(BASE_OBS_NAMES)}, got {flat.shape[0]}")
        date = self._date_from_obs(flat, info)
        raw_enhanced = np.concatenate([flat, self._features_for_date(date)]).astype(np.float32)
        normalized = self._normalize(raw_enhanced)
        self.last_raw_enhanced_obs = raw_enhanced
        self.last_normalized_obs = normalized
        return normalized

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        info = dict(info or {})
        return self._augment_and_normalize(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cur_nstres = direct_ppo.scalar(getattr(self.env, "last_obs_dict", {}).get("nstres", 0.0), 0.0)
        nstres_excess = max(float(cur_nstres) - NSTRES_THRESHOLD, 0.0) if np.isfinite(cur_nstres) else 0.0
        reward_scale = float(self.config.get("reward", {}).get("reward_scale", 1.0))
        penalty_unscaled = NSTRES_PENALTY_COEF * nstres_excess
        penalty_scaled = penalty_unscaled * reward_scale
        new_reward = float(reward) - penalty_scaled
        if hasattr(self.env, "last_action_info"):
            self.env.last_action_info.update(
                {
                    "nstres_guardrail_04202_enabled": True,
                    "nstres_guardrail_04202_threshold": NSTRES_THRESHOLD,
                    "nstres_guardrail_04202_coef": NSTRES_PENALTY_COEF,
                    "nstres_after_step": float(cur_nstres) if np.isfinite(cur_nstres) else np.nan,
                    "nstres_guardrail_04202_excess": float(nstres_excess),
                    "nstres_guardrail_04202_penalty_unscaled": float(penalty_unscaled),
                    "nstres_guardrail_04202_penalty_scaled": float(penalty_scaled),
                    "reward_before_nstres_guardrail_04202": float(reward),
                    "reward_after_nstres_guardrail_04202": float(new_reward),
                }
            )
        info = dict(info or {})
        return self._augment_and_normalize(obs, info), new_reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()

    @property
    def last_action_info(self) -> dict[str, Any]:
        base = dict(getattr(self.env, "last_action_info", {}))
        base.update({f"obs_{k}": v for k, v in self.last_weather_features.items()})
        base["observation_normalization_04202_enabled"] = True
        base["base_observation_dim"] = self.base_observation_dim
        base["enhanced_observation_dim"] = self.enhanced_observation_dim
        return base

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name: str):
        return getattr(self.env, name)


def make_env_04202(config: dict, env_config: dict, station: str, year: int, seed: int, run_tag: str, evaluation: bool = False):
    env = ORIGINAL_04040_MAKE_ENV(config, env_config, station, year, seed, run_tag, evaluation=evaluation)
    return NormalizedWeatherStressRewardWrapper(env, config, env_config, station, int(year))


def configure_base_module() -> None:
    base04104.configure_base_module()
    base04103.TASK_ID = TASK_ID
    base04103.TASK_NAME = TASK_NAME
    base04103.BASE_OUT = BASE_OUT
    base04103.BASE_DOC = BASE_DOC
    base04103.PROMPT = PROMPT
    base04103.out_for_suffix = out_for_suffix
    base04103.doc_for_suffix = doc_for_suffix
    base04103.DEFAULT_TIMESTEPS = DEFAULT_TIMESTEPS
    base04103.DEFAULT_CHECKPOINTS = DEFAULT_CHECKPOINTS
    base04103.pretrain_policy_bc = base04104.pretrain_policy_bc_balanced
    base04040.load_config = load_config
    base04040.make_env_with_yield_guardrail = make_env_04202


def restore_base_module() -> None:
    base04103.TASK_ID = ORIGINAL_04103_TASK_ID
    base04103.TASK_NAME = ORIGINAL_04103_TASK_NAME
    base04103.BASE_OUT = ORIGINAL_04103_BASE_OUT
    base04103.BASE_DOC = ORIGINAL_04103_BASE_DOC
    base04103.PROMPT = ORIGINAL_04103_PROMPT
    base04103.out_for_suffix = ORIGINAL_04103_OUT_FOR_SUFFIX
    base04103.doc_for_suffix = ORIGINAL_04103_DOC_FOR_SUFFIX
    base04103.pretrain_policy_bc = ORIGINAL_04103_PRETRAIN
    base04040.make_env_with_yield_guardrail = ORIGINAL_04040_MAKE_ENV
    base04040.load_config = ORIGINAL_04040_LOAD_CONFIG


def normalization_smoke_audit(config: dict[str, Any], env_config: dict[str, Any], years: list[int], out: Path) -> pd.DataFrame:
    rows = []
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_INPUT_ROOT
    try:
        for year in years:
            env = make_env_04202(config, env_config, STATION, int(year), SEED, f"{STATION}_{year}_042_02_obs_audit", evaluation=True)
            try:
                obs, info = env.reset()
                raw = env.last_raw_enhanced_obs
                norm = env.last_normalized_obs
                if raw is None or norm is None:
                    raise RuntimeError("Wrapper did not record raw/normalized observation.")
                for name, raw_value, norm_value, scale in zip(ENHANCED_OBS_NAMES, raw, norm, SCALE):
                    rows.append(
                        {
                            "station_code": STATION,
                            "year": int(year),
                            "feature": name,
                            "raw_value_at_reset": float(raw_value),
                            "scale": float(scale),
                            "normalized_value_at_reset": float(norm_value),
                        }
                    )
                rows.append(
                    {
                        "station_code": STATION,
                        "year": int(year),
                        "feature": "__shape__",
                        "raw_value_at_reset": float(len(raw)),
                        "scale": np.nan,
                        "normalized_value_at_reset": float(len(obs)),
                    }
                )
            finally:
                env.close()
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root
    audit = pd.DataFrame(rows)
    audit_dir = out / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_dir / "042_02_normalized_observation_smoke_audit.csv", index=False, encoding="utf-8-sig")
    return audit


def dry_run(timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    configure_base_module()
    try:
        out = out_for_suffix(suffix)
        out.mkdir(parents=True, exist_ok=True)
        (out / "audits").mkdir(parents=True, exist_ok=True)
        selected = base04103.load_selected_teachers()
        config, env_config = base04103.build_config_and_env_config(base04103.YEARS)
        audit = normalization_smoke_audit(config, env_config, [2014, 2018, 2023], out)
        shape_rows = audit[audit["feature"].eq("__shape__")]
        lowic_status = lowic_authoritative_audit_status()
        result = {
            "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
            "mode": "dry_run",
            "lowIC_input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
            "lowIC_input_root_exists": LOWIC_INPUT_ROOT.exists(),
            "lowIC_authoritative_audit": lowic_status,
            "teacher_selected_exists": base04103.TEACHER_SELECTED.exists(),
            "selected_years": selected["year"].astype(int).tolist(),
            "teacher_tier_counts": selected["teacher_tier"].value_counts().to_dict(),
            "base_observation_dim": len(BASE_OBS_NAMES),
            "enhanced_observation_dim": len(ENHANCED_OBS_NAMES),
            "shape_audit_pass": bool(
                len(shape_rows) == 3
                and (shape_rows["raw_value_at_reset"].astype(int) == len(ENHANCED_OBS_NAMES)).all()
                and (shape_rows["normalized_value_at_reset"].astype(int) == len(ENHANCED_OBS_NAMES)).all()
            ),
            "normalization_scale_count": int(len(SCALE)),
            "total_timesteps": int(timesteps),
            "checkpoint_steps": checkpoint_steps,
            "bc": {
                "epochs": base04104.BC_EPOCHS,
                "batch_size": base04104.BC_BATCH_SIZE,
                "lr": base04104.BC_LR,
                "balanced_nonzero_fraction": base04104.BALANCED_NONZERO_FRACTION,
            },
            "reward_added": config["reward"]["nstres_guardrail_process_penalty_04202"],
            "action_safety": config["action_safety"],
            "discrete_actions": config["discrete_actions"],
            "next_step_allowed": bool(
                PROMPT.exists()
                and LOWIC_INPUT_ROOT.exists()
                and lowic_status["pass"]
                and base04103.TEACHER_SELECTED.exists()
            ),
            "audit_csv": (out / "audits" / "042_02_normalized_observation_smoke_audit.csv").relative_to(ROOT).as_posix(),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        restore_base_module()


def run_training(timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    configure_base_module()
    try:
        lowic_status = lowic_authoritative_audit_status()
        if not lowic_status["pass"]:
            raise RuntimeError(f"lowIC authoritative audit is not clean: {lowic_status}")
        base04103.run_training(timesteps, checkpoint_steps, suffix)
        out = out_for_suffix(suffix)
        config_path = out / "configs" / "042_02_config_patch.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(load_config(), indent=2, ensure_ascii=False), encoding="utf-8")
    finally:
        restore_base_module()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--checkpoint-steps", type=str, default=None)
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()
    checkpoint_steps = parse_checkpoint_steps(args.checkpoint_steps, args.timesteps)
    if args.dry_run:
        dry_run(args.timesteps, checkpoint_steps, args.suffix)
    else:
        run_training(args.timesteps, checkpoint_steps, args.suffix)


if __name__ == "__main__":
    main()
