"""043_02: SYA lowIC binary-timing MaskablePPO with normalized weather forecast inputs.

This keeps the current binary-timing PPO line from 042_10:

    irrigation: [0, 45] mm
    nitrogen:   [0, 80] kg/ha

and adds the observation/reward components from 042_02, but deliberately does
not use teacher warm-start. The aim is a clean test of whether explicit
weather/perfect-forecast inputs plus fixed physical normalization improve
policy responsiveness.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_sya_lowIC_binary_timing_maskableppo_042_10 as base04210


ORIGINAL_SHUTIL_COPY2 = shutil.copy2
ORIGINAL_BASE_MAKE_ENV = base03222.base.make_env

TASK_ID = "043_02"
TASK_NAME = "sya_lowIC_binary_timing_forecast_normalized_maskableppo"
BASE_OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
BASE_DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

LOWIC_INPUT_ROOT = base04210.LOWIC_INPUT_ROOT
STATION = base04210.STATION
SITES = [STATION]
DEFAULT_TOTAL_TIMESTEPS = 100_000
DEFAULT_CHECKPOINT_STEPS = [25_000, 50_000, 75_000, 100_000]

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

# Fixed physical scales. These are not fitted from validation results.
SCALE = np.array(
    [
        300.0,  # cumsumfert kg/ha
        160.0,  # dap
        2500.0,  # accumulated thermal time proxy
        10.0,  # daily ET/transpiration proxy
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
        25000.0,  # topwt
        300.0,  # total irrigation
        20.0,  # vstage
        200.0,  # water-table/root proxy
        8.0,  # xlai
        100.0,  # rain today
        40.0,  # tmin
        200.0,  # past7 rain
        200.0,  # future7 rain
        40.0,  # future7 mean temp
    ],
    dtype=np.float32,
)
CLIP_LOW = np.full(len(ENHANCED_OBS_NAMES), -5.0, dtype=np.float32)
CLIP_HIGH = np.full(len(ENHANCED_OBS_NAMES), 5.0, dtype=np.float32)

NSTRES_THRESHOLD = 0.05
NSTRES_PENALTY_COEF = 50.0

RENAMES = {
    "configs/032_22_half_split_selection.csv": "configs/043_02_half_split_selection.csv",
    "evaluation/032_22_training_checkpoint_inventory_partial.csv": "evaluation/043_02_training_checkpoint_inventory_partial.csv",
    "evaluation/032_22_checkpoint_validation_summary_partial.csv": "evaluation/043_02_checkpoint_validation_summary_partial.csv",
    "evaluation/032_22_training_checkpoint_inventory.csv": "evaluation/043_02_training_checkpoint_inventory.csv",
    "evaluation/032_22_checkpoint_validation_summary.csv": "evaluation/043_02_checkpoint_validation_summary.csv",
    "evaluation/032_22_validation_summary_by_station_checkpoint.csv": "evaluation/043_02_validation_summary_by_station_checkpoint.csv",
    "logs/032_22_training_year_reset_counts.csv": "logs/043_02_training_year_reset_counts.csv",
    "032_22_result.json": "043_02_result.json",
}


def safe_copy2(src: str | Path, dst: str | Path, *args, **kwargs) -> str:
    try:
        return str(ORIGINAL_SHUTIL_COPY2(src, dst, *args, **kwargs))
    except PermissionError:
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst_path)
        return str(dst_path)


def out_for_suffix(suffix: str) -> Path:
    return ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}_{suffix}" if suffix else BASE_OUT


def doc_for_suffix(suffix: str) -> Path:
    return ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_{suffix}_record.md" if suffix else BASE_DOC


def parse_checkpoint_steps(raw: str | None, total_timesteps: int) -> list[int]:
    if not raw:
        return [x for x in DEFAULT_CHECKPOINT_STEPS if x <= total_timesteps]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def load_config() -> dict[str, Any]:
    cfg = base04210.load_config()
    cfg["observation_normalization_04302"] = {
        "enabled": True,
        "base_observation_names": BASE_OBS_NAMES,
        "weather_feature_names": WEATHER_FEATURE_NAMES,
        "scale": {name: float(value) for name, value in zip(ENHANCED_OBS_NAMES, SCALE)},
        "clip_low": -5.0,
        "clip_high": 5.0,
        "fitted_from_validation_results": False,
    }
    cfg["weather_forecast_observation_04302"] = {
        "enabled": True,
        "feature_names": WEATHER_FEATURE_NAMES,
        "past_window_days": 7,
        "future_window_days": 7,
        "future_weather_assumption": "perfect historical weather forecast",
        "normalized_by_04302": True,
    }
    cfg["reward"]["nstres_guardrail_process_penalty_04302"] = {
        "enabled": True,
        "threshold": NSTRES_THRESHOLD,
        "coef": NSTRES_PENALTY_COEF,
        "formula": "coef * max(0, nstres_after_step - threshold) * reward_scale",
        "source": "inherited concept from 042_02; fixed before 043_02 training",
    }
    cfg["teacher_warmstart_04302"] = {
        "enabled": False,
        "reason": "clean attribution to weather/forecast observation and normalization",
    }
    return cfg


def _weather_table(config: dict[str, Any], station: str) -> pd.DataFrame:
    weather = direct_ppo.weather_for_daily(config)
    weather = weather[weather["station_code"].astype(str).eq(station)].copy()
    weather["date"] = pd.to_datetime(weather["date"])
    for col in ["rain", "tmin", "tmax"]:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")
    return weather.sort_values("date").reset_index(drop=True)


class NormalizedForecastObservationWrapper(gym.Env):
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
            return self.planting + pd.Timedelta(days=int(round(float(dap_raw))) - 1)
        return self.planting

    def _features_for_date(self, date: pd.Timestamp) -> np.ndarray:
        cur = self.weather[self.weather["date"].eq(date)]
        rain_today = float(cur["rain"].iloc[0]) if not cur.empty and pd.notna(cur["rain"].iloc[0]) else 0.0
        tmin_today = float(cur["tmin"].iloc[0]) if not cur.empty and pd.notna(cur["tmin"].iloc[0]) else 0.0
        past = self.weather[(self.weather["date"] >= date - pd.Timedelta(days=6)) & (self.weather["date"] <= date)]
        future = self.weather[(self.weather["date"] >= date) & (self.weather["date"] <= date + pd.Timedelta(days=6))]
        rain_past7 = float(past["rain"].sum(skipna=True)) if not past.empty else 0.0
        rain_future7 = float(future["rain"].sum(skipna=True)) if not future.empty else 0.0
        if not future.empty:
            tmean_series = (future["tmax"] + future["tmin"]) / 2.0
            tmean_future7 = float(tmean_series.mean(skipna=True)) if tmean_series.notna().any() else 0.0
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
                    "nstres_guardrail_04302_enabled": True,
                    "nstres_guardrail_04302_threshold": NSTRES_THRESHOLD,
                    "nstres_guardrail_04302_coef": NSTRES_PENALTY_COEF,
                    "nstres_after_step": float(cur_nstres) if np.isfinite(cur_nstres) else np.nan,
                    "nstres_guardrail_04302_excess": float(nstres_excess),
                    "nstres_guardrail_04302_penalty_unscaled": float(penalty_unscaled),
                    "nstres_guardrail_04302_penalty_scaled": float(penalty_scaled),
                    "reward_before_nstres_guardrail_04302": float(reward),
                    "reward_after_nstres_guardrail_04302": float(new_reward),
                }
            )
        return self._augment_and_normalize(obs, info), new_reward, terminated, truncated, dict(info or {})

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()

    @property
    def last_action_info(self) -> dict[str, Any]:
        base_info = getattr(self.env, "last_action_info", {})
        info = dict(base_info) if isinstance(base_info, dict) else {}
        for key, value in self.last_weather_features.items():
            info[f"obs_{key}"] = float(value)
        info["base_observation_dim_04302"] = self.base_observation_dim
        info["enhanced_observation_dim_04302"] = self.enhanced_observation_dim
        info["observation_normalization_04302_enabled"] = True
        return info

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)


def make_env_04302(
    config: dict[str, Any],
    env_config: dict[str, Any],
    station: str,
    year: int,
    seed: int,
    run_tag: str,
    evaluation: bool = False,
):
    env = ORIGINAL_BASE_MAKE_ENV(config, env_config, station, year, seed, run_tag, evaluation=evaluation)
    return NormalizedForecastObservationWrapper(env, config, env_config, station, year)


def patch_base_module(out: Path, doc: Path, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    base04210.patch_base_module(out, doc, total_timesteps, checkpoint_steps)
    base03222.PROMPT = PROMPT
    base03222.load_config = load_config
    base03222.base.make_env = make_env_04302
    base03222.shutil.copy2 = safe_copy2


def copy_with_task_names(out: Path, suffix: str) -> None:
    for old_rel, new_rel in RENAMES.items():
        old = out / old_rel
        if suffix:
            new_rel = new_rel.replace("043_02_", f"043_02_{suffix}_")
        new = out / new_rel
        if old.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            safe_copy2(old, new)
    generic_result = out / "032_22_result.json"
    result_path = out / ("043_02_result.json" if not suffix else f"043_02_{suffix}_result.json")
    if generic_result.exists():
        result = json.loads(generic_result.read_text(encoding="utf-8"))
        result.update(
            {
                "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
                "input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
                "station_scope": SITES,
                "binary_timing_action_levels": {
                    "irrigation_levels": base04210.BINARY_IRRIGATION_LEVELS,
                    "nitrogen_levels": base04210.BINARY_NITROGEN_LEVELS,
                    "combined_action_count": len(base04210.BINARY_IRRIGATION_LEVELS)
                    * len(base04210.BINARY_NITROGEN_LEVELS),
                },
                "enhanced_observation_dim": len(ENHANCED_OBS_NAMES),
                "weather_forecast_features": WEATHER_FEATURE_NAMES,
                "teacher_warmstart_enabled": False,
            }
        )
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    lines = [
        "| " + " | ".join(map(str, work.columns)) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for row in work.to_numpy().tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def observation_smoke_audit(out: Path) -> tuple[bool, Path]:
    config = load_config()
    split = base03222.load_split()
    selection = base03222.build_selection(split)
    env_config = direct_ppo.build_env_config(config, selection)
    years = [2005, 2014, 2023]
    rows: list[dict[str, Any]] = []
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_INPUT_ROOT
    try:
        for year in years:
            env = make_env_04302(config, env_config, STATION, int(year), int(config["seed"]), f"043_02_smoke_{year}", evaluation=True)
            try:
                obs, info = env.reset()
                raw = env.last_raw_enhanced_obs
                norm = env.last_normalized_obs
                masks = env.action_masks()
                rows.append(
                    {
                        "station_code": STATION,
                        "year": int(year),
                        "obs_dim": int(np.asarray(obs).reshape(-1).shape[0]),
                        "raw_dim": int(len(raw)) if raw is not None else -1,
                        "norm_dim": int(len(norm)) if norm is not None else -1,
                        "base_observation_dim": env.base_observation_dim,
                        "enhanced_observation_dim": env.enhanced_observation_dim,
                        "mask_dim": int(np.asarray(masks).reshape(-1).shape[0]),
                        "valid_action_count_at_reset": int(np.asarray(masks).astype(bool).sum()),
                        "rain_today_mm": env.last_weather_features.get("rain_today_mm", np.nan),
                        "tmin_today_c": env.last_weather_features.get("tmin_today_c", np.nan),
                        "rain_past7_mm": env.last_weather_features.get("rain_past7_mm", np.nan),
                        "rain_future7_mm": env.last_weather_features.get("rain_future7_mm", np.nan),
                        "tmean_future7_c": env.last_weather_features.get("tmean_future7_c", np.nan),
                    }
                )
            finally:
                env.close()
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root
    audit = pd.DataFrame(rows)
    path = out / "audits" / "043_02_observation_smoke_audit.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(path, index=False, encoding="utf-8-sig")
    pass_flag = (
        len(audit) == len(years)
        and (audit["obs_dim"].astype(int) == len(ENHANCED_OBS_NAMES)).all()
        and (audit["raw_dim"].astype(int) == len(ENHANCED_OBS_NAMES)).all()
        and (audit["norm_dim"].astype(int) == len(ENHANCED_OBS_NAMES)).all()
        and (audit["mask_dim"].astype(int) == len(base04210.BINARY_IRRIGATION_LEVELS) * len(base04210.BINARY_NITROGEN_LEVELS)).all()
        and audit[WEATHER_FEATURE_NAMES].notna().all().all()
    )
    return bool(pass_flag), path


def write_clean_record(out: Path, doc: Path, suffix: str, total_timesteps: int, checkpoint_steps: list[int], smoke_csv: Path) -> None:
    prefix = "043_02" if not suffix else f"043_02_{suffix}"
    split = read_csv_or_empty(out / "configs" / f"{prefix}_half_split_selection.csv")
    train = read_csv_or_empty(out / "evaluation" / f"{prefix}_training_checkpoint_inventory.csv")
    eval_df = read_csv_or_empty(out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv")
    by_station = read_csv_or_empty(out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv")
    smoke = read_csv_or_empty(smoke_csv)
    split_show = split[[c for c in ["station_code", "site", "year", "split"] if c in split.columns]] if not split.empty else split

    lines = [
        f"# {prefix} SYA lowIC 天气预报归一化 binary-timing MaskablePPO 记录",
        "",
        "## 本轮结论边界",
        "",
        "- 本轮是在 042_10 binary-timing PPO 基础上加入天气/完美预报输入和固定物理尺度归一化。",
        "- 不使用 teacher warm-start，因此不能把结果解释为 teacher 带来的提升。",
        "- reward 继承主线，并加入 042_02 已使用过的氮胁迫过程惩罚；本轮不扫描系数。",
        f"- 训练步数：`{total_timesteps}`；checkpoint：`{', '.join(map(str, checkpoint_steps))}`。",
        "",
        "## 固定配置",
        "",
        f"- lowIC 输入目录：`{LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix()}`",
        f"- 灌溉档位：`{base04210.BINARY_IRRIGATION_LEVELS}`",
        f"- 施氮档位：`{base04210.BINARY_NITROGEN_LEVELS}`",
        f"- observation：基础 25 维 + 天气/预报 5 维 = `{len(ENHANCED_OBS_NAMES)}` 维。",
        f"- 天气/预报变量：`{WEATHER_FEATURE_NAMES}`",
        "- 安全约束：继承 040_36，包括 7 天间隔、季节上限、DAP90 后禁氮、晚期灌溉保留 mask。",
        "",
        "## Observation smoke 审计",
        "",
        md_table(smoke, 20),
        "",
        "## 年份划分",
        "",
        md_table(split_show, 120),
        "",
        "## 训练 checkpoint 库存",
        "",
        md_table(train, 80),
        "",
        "## 验证集按 checkpoint 汇总",
        "",
        md_table(by_station, 80),
        "",
        "## 逐年验证结果",
        "",
        md_table(eval_df, 240),
        "",
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines), encoding="utf-8")


def dry_run(total_timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    out = out_for_suffix(suffix)
    doc = doc_for_suffix(suffix)
    patch_base_module(out, doc, total_timesteps, checkpoint_steps)
    cfg = load_config()
    split = base03222.load_split().copy()
    split = split[split["station_code"].eq(STATION)].sort_values(["year"]).reset_index(drop=True)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
        "mode": "dry_run",
        "station": STATION,
        "lowIC_input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "lowIC_input_root_exists": LOWIC_INPUT_ROOT.exists(),
        "prompt_exists": PROMPT.exists(),
        "split_years": split[["station_code", "site", "year", "split"]].to_dict(orient="records"),
        "total_timesteps": int(total_timesteps),
        "checkpoint_steps": checkpoint_steps,
        "discrete_actions": cfg["discrete_actions"],
        "action_safety": cfg["action_safety"],
        "reward": cfg["reward"],
        "observation_normalization_04302": cfg["observation_normalization_04302"],
        "weather_forecast_observation_04302": cfg["weather_forecast_observation_04302"],
        "enhanced_observation_dim": len(ENHANCED_OBS_NAMES),
        "teacher_warmstart_enabled": False,
        "next_step_allowed": bool(LOWIC_INPUT_ROOT.exists() and PROMPT.exists() and not split.empty),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_training(total_timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    out = out_for_suffix(suffix)
    doc = doc_for_suffix(suffix)
    patch_base_module(out, doc, total_timesteps, checkpoint_steps)
    out.mkdir(parents=True, exist_ok=True)
    smoke_pass, smoke_csv = observation_smoke_audit(out)
    if not smoke_pass:
        raise RuntimeError(f"043_02 observation smoke failed: {smoke_csv.relative_to(ROOT).as_posix()}")

    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_INPUT_ROOT
    try:
        base03222.main()
        copy_with_task_names(out, suffix)
        base04210.add_04210_summaries(out, suffix.replace("043_02_", "") if suffix.startswith("043_02_") else suffix)
        # The 042_10 helper writes 042_10 names; keep our canonical 043_02 copies as primary.
        write_clean_record(out, doc, suffix, total_timesteps, checkpoint_steps, smoke_csv)
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root
        base03222.base.make_env = ORIGINAL_BASE_MAKE_ENV

    prefix = "043_02" if not suffix else f"043_02_{suffix}"
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
        "algorithm": "MaskablePPO",
        "record_md": doc.relative_to(ROOT).as_posix(),
        "observation_smoke_csv": smoke_csv.relative_to(ROOT).as_posix(),
        "train_inventory": (out / "evaluation" / f"{prefix}_training_checkpoint_inventory.csv").relative_to(ROOT).as_posix(),
        "validation_summary": (out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv").relative_to(ROOT).as_posix(),
        "by_checkpoint": (out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv").relative_to(ROOT).as_posix(),
        "total_timesteps": int(total_timesteps),
        "checkpoint_steps": checkpoint_steps,
        "input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "binary_irrigation_levels": base04210.BINARY_IRRIGATION_LEVELS,
        "binary_nitrogen_levels": base04210.BINARY_NITROGEN_LEVELS,
        "enhanced_observation_dim": len(ENHANCED_OBS_NAMES),
        "teacher_warmstart_enabled": False,
    }
    result_path = out / (f"{prefix}_wrapper_result.json")
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
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
