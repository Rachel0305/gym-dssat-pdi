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
import run_sya_lowIC_ppo_yield_guardrail_v3_040_40 as base04040
import run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36 as base04036
import run_sya_lowIC_ppo_early_irrigation_reserve_penalty_040_10 as util04010


TASK_ID = "042_00"
TASK_NAME = "sya_lowIC_weather_forecast_observation_maskableppo"
BASE_OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
BASE_DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

LOWIC_INPUT_ROOT = base04040.LOWIC_INPUT_ROOT
STATION = "SYA"
SITE = "SY"
SITES = [STATION]
SEED = 0
DEFAULT_TOTAL_TIMESTEPS = 100_000
DEFAULT_CHECKPOINT_STEPS = [25_000, 50_000, 75_000, 100_000]
WEATHER_FEATURE_NAMES = [
    "rain_today_mm",
    "tmin_today_c",
    "rain_past7_mm",
    "rain_future7_mm",
    "tmean_future7_c",
]

ORIGINAL_COPY2 = shutil.copy2
ORIGINAL_BASE_MAKE_ENV = base03222.base.make_env
ORIGINAL_LOAD_CONFIG = base03222.load_config

RENAMES = {
    "configs/032_22_half_split_selection.csv": "configs/042_00_half_split_selection.csv",
    "evaluation/032_22_training_checkpoint_inventory_partial.csv": "evaluation/042_00_training_checkpoint_inventory_partial.csv",
    "evaluation/032_22_checkpoint_validation_summary_partial.csv": "evaluation/042_00_checkpoint_validation_summary_partial.csv",
    "evaluation/032_22_training_checkpoint_inventory.csv": "evaluation/042_00_training_checkpoint_inventory.csv",
    "evaluation/032_22_checkpoint_validation_summary.csv": "evaluation/042_00_checkpoint_validation_summary.csv",
    "evaluation/032_22_validation_summary_by_station_checkpoint.csv": "evaluation/042_00_validation_summary_by_station_checkpoint.csv",
    "logs/032_22_training_year_reset_counts.csv": "logs/042_00_training_year_reset_counts.csv",
    "032_22_result.json": "042_00_result.json",
}


def safe_copy2(src: str | Path, dst: str | Path, *args, **kwargs) -> str:
    try:
        return str(ORIGINAL_COPY2(src, dst, *args, **kwargs))
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
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return [x for x in DEFAULT_CHECKPOINT_STEPS if x <= total_timesteps]


def ensure_dirs(out: Path) -> None:
    for rel in ["configs", "evaluation", "logs", "models", "daily_outputs", "audits"]:
        (out / rel).mkdir(parents=True, exist_ok=True)
    BASE_DOC.parent.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    cfg = base04040.load_config()
    cfg["weather_forecast_observation"] = {
        "enabled": True,
        "feature_names": WEATHER_FEATURE_NAMES,
        "past_window_days": 7,
        "future_window_days": 7,
        "future_weather_assumption": "perfect historical weather forecast",
        "raw_physical_units": True,
        "reward_changed": False,
    }
    return cfg


def _weather_table(config: dict[str, Any], station: str) -> pd.DataFrame:
    weather = direct_ppo.weather_for_daily(config)
    weather = weather[weather["station_code"].astype(str).eq(station)].copy()
    weather["date"] = pd.to_datetime(weather["date"])
    for col in ["rain", "tmin", "tmax"]:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")
    return weather.sort_values("date").reset_index(drop=True)


class WeatherForecastObservationWrapper(gym.Env):
    """Append same-day and 7-day historical/perfect-forecast weather features."""

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
        low = np.concatenate([np.asarray(base_space.low, dtype=np.float32).reshape(-1), np.full(len(WEATHER_FEATURE_NAMES), -np.inf, dtype=np.float32)])
        high = np.concatenate([np.asarray(base_space.high, dtype=np.float32).reshape(-1), np.full(len(WEATHER_FEATURE_NAMES), np.inf, dtype=np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.last_weather_features: dict[str, float] = {}
        self.base_observation_dim = int(np.asarray(base_space.low).reshape(-1).shape[0])
        self.enhanced_observation_dim = self.base_observation_dim + len(WEATHER_FEATURE_NAMES)

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
            tmin_today = np.nan
        else:
            rain_today = float(today["rain"].iloc[0]) if pd.notna(today["rain"].iloc[0]) else 0.0
            tmin_today = float(today["tmin"].iloc[0]) if pd.notna(today["tmin"].iloc[0]) else np.nan
        past_start = date - pd.Timedelta(days=6)
        future_end = date + pd.Timedelta(days=6)
        past = w[(w["date"] >= past_start) & (w["date"] <= date)]
        future = w[(w["date"] >= date) & (w["date"] <= future_end)]
        rain_past7 = float(pd.to_numeric(past["rain"], errors="coerce").fillna(0).sum()) if not past.empty else 0.0
        rain_future7 = float(pd.to_numeric(future["rain"], errors="coerce").fillna(0).sum()) if not future.empty else 0.0
        if not future.empty:
            tmean_daily = (pd.to_numeric(future["tmax"], errors="coerce") + pd.to_numeric(future["tmin"], errors="coerce")) / 2.0
            tmean_future7 = float(tmean_daily.mean()) if tmean_daily.notna().any() else np.nan
        else:
            tmean_future7 = np.nan
        vals = np.array(
            [
                rain_today,
                0.0 if not np.isfinite(tmin_today) else tmin_today,
                rain_past7,
                rain_future7,
                0.0 if not np.isfinite(tmean_future7) else tmean_future7,
            ],
            dtype=np.float32,
        )
        self.last_weather_features = {name: float(value) for name, value in zip(WEATHER_FEATURE_NAMES, vals)}
        return vals

    def _augment(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        date = self._date_from_obs(flat, info)
        return np.concatenate([flat, self._features_for_date(date)]).astype(np.float32)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        info = dict(info or {})
        return self._augment(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})
        return self._augment(obs, info), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()

    @property
    def last_action_info(self) -> dict[str, Any]:
        base = dict(getattr(self.env, "last_action_info", {}))
        base.update({f"obs_{k}": v for k, v in self.last_weather_features.items()})
        base["weather_forecast_observation_enabled"] = True
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


def make_env_with_weather_forecast(
    config: dict,
    env_config: dict,
    station: str,
    year: int,
    seed: int,
    run_tag: str,
    evaluation: bool = False,
):
    env = base04040.make_env_with_yield_guardrail(config, env_config, station, year, seed, run_tag, evaluation=evaluation)
    return WeatherForecastObservationWrapper(env, config, env_config, station, int(year))


def patch_base_module(out: Path, doc: Path, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    base03222.OUT = out
    base03222.DOC = doc
    base03222.PROMPT = PROMPT
    base03222.SITES = list(SITES)
    base03222.TOTAL_TIMESTEPS = int(total_timesteps)
    base03222.CHECKPOINT_STEPS = [int(x) for x in checkpoint_steps]
    base03222.summarize_by_station = util04010.summarize_by_station_safe
    base03222.load_config = load_config
    base03222.base.make_env = make_env_with_weather_forecast
    base03222.shutil.copy2 = safe_copy2


def restore_base_module() -> None:
    base03222.base.make_env = ORIGINAL_BASE_MAKE_ENV
    base03222.load_config = ORIGINAL_LOAD_CONFIG


def copy_with_task_names(out: Path, suffix: str) -> None:
    for old_rel, new_rel in RENAMES.items():
        old = out / old_rel
        if suffix:
            new_rel = new_rel.replace("042_00_", f"042_00_{suffix}_")
        new = out / new_rel
        if old.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            safe_copy2(old, new)
    generic_result = out / "032_22_result.json"
    result_path = out / ("042_00_result.json" if not suffix else f"042_00_{suffix}_result.json")
    if generic_result.exists():
        result = json.loads(generic_result.read_text(encoding="utf-8"))
        result.update(
            {
                "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
                "input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
                "weather_forecast_observation": {
                    "enabled": True,
                    "feature_names": WEATHER_FEATURE_NAMES,
                    "base_observation_dim": 25,
                    "enhanced_observation_dim": 30,
                    "future_weather_assumption": "perfect historical weather forecast",
                },
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


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
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
    if len(df) > max_rows:
        lines.append(f"\n仅显示前 {max_rows} 行，共 {len(df)} 行。")
    return "\n".join(lines)


def lowic_authoritative_audit_status() -> dict[str, Any]:
    path = ROOT / "benchmark_results" / "039_00_low_initial_soil_water_nitrogen_input_audit" / "tables" / "039_00_authoritative_template_issues.csv"
    if not path.exists():
        return {"exists": False, "issue_count": None, "status": "missing_039_00_authoritative_issue_table"}
    if path.stat().st_size == 0:
        return {"exists": True, "issue_count": 0, "status": "pass", "note": "empty issue file means no authoritative issues"}
    try:
        df = pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return {"exists": True, "issue_count": 0, "status": "pass", "note": "empty issue file means no authoritative issues"}
    return {"exists": True, "issue_count": int(len(df)), "status": "pass" if len(df) == 0 else "has_issues"}


def build_selection() -> pd.DataFrame:
    split = base03222.load_split().copy()
    split = split[split["station_code"].eq(STATION)].sort_values(["year"]).reset_index(drop=True)
    pool = pd.read_csv(base03222.POOL, keep_default_na=False)
    pool["year"] = pd.to_numeric(pool["year"], errors="coerce").astype(int)
    selected = pool.merge(split[["station_code", "year", "split"]], on=["station_code", "year"], how="inner")
    selected["selected_for_train"] = selected["split"].eq("train")
    selected["selected_for_eval"] = True
    selected["selection_reason"] = "042_00_sya_lowIC_weather_forecast_observation"
    return selected.sort_values(["station_code", "year"]).reset_index(drop=True)


def observation_smoke_audit(config: dict[str, Any], env_config: dict[str, Any], out: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year in [2005, 2014, 2023]:
        env = make_env_with_weather_forecast(config, env_config, STATION, year, SEED, f"{STATION}_{year}_042_00_obs_audit", evaluation=True)
        try:
            obs, info = env.reset()
            action_mask = env.action_masks()
            last_info = env.last_action_info
            rows.append(
                {
                    "station_code": STATION,
                    "year": int(year),
                    "base_observation_dim": int(last_info.get("base_observation_dim", -1)),
                    "enhanced_observation_dim": int(last_info.get("enhanced_observation_dim", -1)),
                    "obs_len": int(np.asarray(obs).reshape(-1).shape[0]),
                    "legal_action_count": int(np.asarray(action_mask, dtype=bool).sum()),
                    **{f"feature_{k}": float(last_info.get(f"obs_{k}", np.nan)) for k in WEATHER_FEATURE_NAMES},
                }
            )
        finally:
            env.close()
    df = pd.DataFrame(rows)
    path = out / "audits" / "042_00_observation_smoke_audit.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    ok = bool(
        len(df) == 3
        and (df["base_observation_dim"].astype(int) == 25).all()
        and (df["enhanced_observation_dim"].astype(int) == 30).all()
        and (df["obs_len"].astype(int) == 30).all()
        and df[[f"feature_{x}" for x in WEATHER_FEATURE_NAMES]].notna().all().all()
    )
    return df, {"path": path.relative_to(ROOT).as_posix(), "pass": ok}


def write_clean_record(out: Path, doc: Path, suffix: str, total_timesteps: int, checkpoint_steps: list[int], obs_audit: pd.DataFrame | None = None) -> None:
    prefix = "042_00" if not suffix else f"042_00_{suffix}"
    split = read_csv_or_empty(out / "configs" / f"{prefix}_half_split_selection.csv")
    train = read_csv_or_empty(out / "evaluation" / f"{prefix}_training_checkpoint_inventory.csv")
    eval_df = read_csv_or_empty(out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv")
    by_station = read_csv_or_empty(out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv")
    audit = obs_audit if obs_audit is not None else read_csv_or_empty(out / "audits" / "042_00_observation_smoke_audit.csv")
    lines = [
        f"# {prefix} SYA lowIC 天气预报增强 observation MaskablePPO 记录",
        "",
        "## 结论边界",
        "",
        "- 本任务改变 agent observation 信息结构：新增当天降雨、当天 Tmin、近 7 天降雨、未来 7 天降雨、未来 7 天平均温度。",
        "- 未来天气使用历史天气构造，属于 perfect weather forecast 场景。",
        "- reward、动作档位和安全约束沿用 040_40；本任务不是 reward 调参。",
        f"- 训练步数：`{total_timesteps}`；checkpoint：`{', '.join(map(str, checkpoint_steps))}`。",
        "",
        "## 输入路径和数据安全",
        "",
        f"- lowIC 输入目录：`{LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix()}`",
        f"- lowIC 输入目录存在：`{LOWIC_INPUT_ROOT.exists()}`",
        f"- 039_00 authoritative lowIC 审计状态：`{lowic_authoritative_audit_status()}`",
        "",
        "## observation smoke audit",
        "",
        md_table(audit, 20),
        "",
        "## 年份划分",
        "",
        md_table(split[[c for c in ['station_code','site','year','split'] if c in split.columns]], 80),
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
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run(total_timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    out = out_for_suffix(suffix)
    doc = doc_for_suffix(suffix)
    ensure_dirs(out)
    patch_base_module(out, doc, total_timesteps, checkpoint_steps)
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_INPUT_ROOT
    try:
        config = load_config()
        selection = build_selection()
        env_config = direct_ppo.build_env_config(config, selection)
        obs_audit, obs_status = observation_smoke_audit(config, env_config, out)
        lowic_status = lowic_authoritative_audit_status()
        result = {
            "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
            "mode": "dry_run",
            "station": STATION,
            "lowIC_input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
            "lowIC_input_root_exists": LOWIC_INPUT_ROOT.exists(),
            "lowIC_authoritative_audit": lowic_status,
            "prompt_exists": PROMPT.exists(),
            "split_years": selection[[c for c in ["station_code", "site", "year", "split"] if c in selection.columns]].to_dict(orient="records"),
            "total_timesteps": int(total_timesteps),
            "checkpoint_steps": checkpoint_steps,
            "weather_forecast_observation": config["weather_forecast_observation"],
            "observation_smoke_audit": obs_status,
            "next_step_allowed": bool(
                LOWIC_INPUT_ROOT.exists()
                and PROMPT.exists()
                and lowic_status.get("status") == "pass"
                and obs_status.get("pass")
            ),
        }
        write_clean_record(out, doc, suffix, total_timesteps, checkpoint_steps, obs_audit)
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root
        restore_base_module()
    print(json.dumps(result, indent=2, ensure_ascii=False))


def add_task_summaries(out: Path, suffix: str) -> None:
    prefix = "042_00" if not suffix else f"042_00_{suffix}"
    eval_path = out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv"
    if not eval_path.exists():
        return
    eval_df = pd.read_csv(eval_path, keep_default_na=False)
    by_path = out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv"
    if by_path.exists() and not eval_df.empty:
        by_station = util04010.summarize_by_station_safe(eval_df)
        by_station.to_csv(by_path, index=False, encoding="utf-8-sig")


def run_training(total_timesteps: int, checkpoint_steps: list[int], suffix: str) -> None:
    out = out_for_suffix(suffix)
    doc = doc_for_suffix(suffix)
    ensure_dirs(out)
    patch_base_module(out, doc, total_timesteps, checkpoint_steps)
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_INPUT_ROOT
    obs_audit = pd.DataFrame()
    try:
        config = load_config()
        selection = build_selection()
        env_config = direct_ppo.build_env_config(config, selection)
        obs_audit, obs_status = observation_smoke_audit(config, env_config, out)
        lowic_status = lowic_authoritative_audit_status()
        if not (LOWIC_INPUT_ROOT.exists() and lowic_status.get("status") == "pass" and obs_status.get("pass")):
            raise RuntimeError(f"Preflight failed: lowIC={lowic_status}, obs={obs_status}")
        base03222.main()
        copy_with_task_names(out, suffix)
        add_task_summaries(out, suffix)
        write_clean_record(out, doc, suffix, total_timesteps, checkpoint_steps, obs_audit)
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root
        restore_base_module()

    prefix = "042_00" if not suffix else f"042_00_{suffix}"
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}" + (f"_{suffix}" if suffix else ""),
        "algorithm": "MaskablePPO",
        "record_md": doc.relative_to(ROOT).as_posix(),
        "train_inventory": (out / "evaluation" / f"{prefix}_training_checkpoint_inventory.csv").relative_to(ROOT).as_posix(),
        "validation_summary": (out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv").relative_to(ROOT).as_posix(),
        "by_checkpoint": (out / "evaluation" / f"{prefix}_validation_summary_by_station_checkpoint.csv").relative_to(ROOT).as_posix(),
        "total_timesteps": int(total_timesteps),
        "checkpoint_steps": checkpoint_steps,
        "input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "weather_forecast_observation": {
            "enabled": True,
            "feature_names": WEATHER_FEATURE_NAMES,
            "base_observation_dim": 25,
            "enhanced_observation_dim": 30,
        },
    }
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
