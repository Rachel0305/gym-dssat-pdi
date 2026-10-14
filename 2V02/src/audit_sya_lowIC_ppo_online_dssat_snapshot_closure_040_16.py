from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base032
import run_lc_multiyear_free_timing_ppo_smoke_032_10 as lc_multi
import run_sya_lowIC_ppo_early_irrigation_hard_reserve_mask_040_13 as run04013
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo
from build_relaxed_success_five_scenario_daily_evidence_027_05 import parse_management_events


TASK = "040_16_sya_lowIC_ppo_online_dssat_snapshot_closure_audit"
OUT = ROOT / "benchmark_results" / TASK
TAB_DIR = OUT / "tables"
SNAP_DIR = OUT / "snapshots"
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

LOWIC_INPUT_ROOT = run04013.LOWIC_INPUT_ROOT
STATION = "SYA"
YEARS = [2014, 2017]
CHECKPOINT = 25_000
SEED = 0
MAX_STEPS = 260
EVAL04013 = ROOT / "benchmark_results" / "040_13_sya_lowIC_ppo_early_irrigation_hard_reserve_mask" / "evaluation" / "040_13_checkpoint_validation_summary.csv"


def ensure_dirs() -> None:
    for path in [OUT, TAB_DIR, SNAP_DIR, OUT / "configs", OUT / "daily_outputs"]:
        path.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def scalar(value: Any, default: float = np.nan) -> float:
    try:
        arr = np.asarray(value).flatten()
        if len(arr) == 0:
            return default
        out = float(arr[0])
        return out if np.isfinite(out) else default
    except Exception:
        return default


def load_env_config() -> tuple[dict[str, Any], dict[str, Any]]:
    base03222.OUT = OUT
    base03222.DOC = DOC
    base03222.SITES = [STATION]
    base03222.SEED = SEED
    base03222.TOTAL_TIMESTEPS = 100_000
    base03222.CHECKPOINT_STEPS = [CHECKPOINT]
    base03222.base.StressAwareDiscreteWrapper = run04013.EarlyIrrigationHardReserveMaskWrapper
    lc_multi.OUT = OUT
    lc_multi.STATION = STATION
    lc_multi.SEED = SEED
    config = base03222.load_config()
    config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    config["runtime"]["max_steps"] = MAX_STEPS
    split = base03222.load_split()
    split = split[split["station_code"].astype(str).eq(STATION)].copy()
    selection = base03222.build_selection(split)
    env_config = direct_ppo.build_env_config(config, selection)
    env_config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    env_config["runtime"]["max_steps"] = MAX_STEPS
    selection.to_csv(OUT / "configs" / "040_16_selection.csv", index=False, encoding="utf-8-sig")
    direct_ppo.write_yaml(env_config, OUT / "configs" / "040_16_resolved_env_config.yaml")
    return config, env_config


def summary_metrics(snapshot: Path, final_yield: float) -> dict[str, Any]:
    rows = siteppo.parse_summary_out(snapshot / "Summary.OUT")
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        hwam = siteppo.num(row, "HWAM")
        if hwam is not None:
            candidates.append((abs(float(hwam) - float(final_yield)), -idx, row))
    if not candidates:
        return {"summary_hwam": np.nan, "summary_ircm": np.nan, "summary_nicm": np.nan, "summary_row_count": len(rows)}
    _, neg_idx, row = min(candidates, key=lambda item: (item[0], item[1]))
    return {
        "summary_hwam": float(siteppo.num(row, "HWAM") or np.nan),
        "summary_ircm": float(siteppo.num(row, "IRCM") or 0.0),
        "summary_nicm": float(siteppo.num(row, "NICM") or 0.0),
        "summary_row_index": int(-neg_idx),
        "summary_row_count": len(rows),
    }


def run_one(config: dict[str, Any], env_config: dict[str, Any], model_path: Path, year: int) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    model = MaskablePPO.load(str(model_path), device="cpu")
    weather = direct_ppo.weather_for_daily(config)
    env = base032.make_env(config, env_config, STATION, int(year), SEED, f"{STATION}_{year}_040_16_online_snapshot", evaluation=True)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, STATION, int(year))["planting_date"])
        while not done and step_count < MAX_STEPS:
            latest = base032.latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = base032.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(STATION)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            rows.append(
                {
                    "station_code": STATION,
                    "year": int(year),
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": scalar(wrow.get("rain"), np.nan),
                    "tmax": scalar(wrow.get("tmax"), np.nan),
                    "tmin": scalar(wrow.get("tmin"), np.nan),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "topwt": scalar(latest.get("topwt")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "reward": float(reward),
                    **dict(getattr(env, "last_action_info", {})),
                    "done": done,
                }
            )
            step_count += 1
        daily = pd.DataFrame(rows)
        snapshot_tmp = siteppo.snapshot_from_env(env)
        snapshot = SNAP_DIR / STATION / str(year) / "online_replay"
        if snapshot.exists():
            shutil.rmtree(snapshot)
        shutil.copytree(snapshot_tmp, snapshot)
    finally:
        env.close()
    if daily.empty:
        raise RuntimeError(f"{STATION}{year} online replay produced empty daily")
    final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
    total_i = float(pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    total_n = float(pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    metrics = summary_metrics(snapshot, final_y)
    events = parse_management_events(snapshot / "MgmtEvent.OUT")
    event_i = float(events[events["operation"].eq("irrigation")]["amount"].sum()) if not events.empty else 0.0
    event_n = float(events[events["operation"].eq("fertilizer")]["amount"].sum()) if not events.empty else 0.0
    out = {
        "station_code": STATION,
        "year": int(year),
        "checkpoint_step": CHECKPOINT,
        "daily_final_grnwt": final_y,
        "daily_total_i": total_i,
        "daily_total_n": total_n,
        "mgmt_event_i": event_i,
        "mgmt_event_n": event_n,
        "snapshot_path": str(snapshot.relative_to(ROOT)).replace("\\", "/"),
        **metrics,
    }
    out["daily_minus_summary_hwam"] = out["daily_final_grnwt"] - out["summary_hwam"]
    out["daily_i_minus_summary_ircm"] = out["daily_total_i"] - out["summary_ircm"]
    out["daily_n_minus_summary_nicm"] = out["daily_total_n"] - out["summary_nicm"]
    out["event_i_minus_summary_ircm"] = event_i - out["summary_ircm"]
    out["event_n_minus_summary_nicm"] = event_n - out["summary_nicm"]
    return daily, out, events


def write_record(comparison: pd.DataFrame, summaries: pd.DataFrame, failures: pd.DataFrame) -> None:
    max_abs_y = pd.to_numeric(comparison.get("daily_minus_summary_hwam", pd.Series(dtype=float)), errors="coerce").abs().max() if not comparison.empty else np.nan
    max_abs_i = pd.to_numeric(comparison.get("daily_i_minus_summary_ircm", pd.Series(dtype=float)), errors="coerce").abs().max() if not comparison.empty else np.nan
    max_abs_n = pd.to_numeric(comparison.get("daily_n_minus_summary_nicm", pd.Series(dtype=float)), errors="coerce").abs().max() if not comparison.empty else np.nan
    if len(failures):
        branch = "failed_execution"
    elif pd.notna(max_abs_y) and max_abs_y <= 1e-3 and max_abs_i <= 1e-3 and max_abs_n <= 1e-3:
        branch = "A_online_daily_summary_management_closed"
    elif pd.notna(max_abs_y) and max_abs_y > 1e-3:
        branch = "B_online_daily_summary_yield_not_closed"
    else:
        branch = "C_management_or_partial_closure_issue"
    lines = [
        "# 040_16 SYA lowIC PPO 在线评估与 DSSAT 原始输出闭合审计记录",
        "",
        f"分支判定：`{branch}`",
        "",
        "## 关键结论",
        "",
        "- 本任务不训练模型，只重放 040_13 ckpt25k。",
        "- 目标是检查 online daily 终值、在线动作总量、DSSAT Summary.OUT、MgmtEvent.OUT 是否在同一次运行中闭合。",
        "",
        "## 040_16 在线重放 vs 同次 DSSAT Summary",
        "",
        md_table(comparison, 50),
        "",
        "## 与 040_13 已发布评估表对比",
        "",
        md_table(summaries, 50),
        "",
        "## 失败记录",
        "",
        md_table(failures, 20),
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    config, env_config = load_env_config()
    published = pd.read_csv(EVAL04013, keep_default_na=False)
    published = published[
        published["station_code"].astype(str).eq(STATION)
        & published["checkpoint_step"].astype(int).eq(CHECKPOINT)
        & published["year"].astype(int).isin(YEARS)
    ].copy()
    if published.empty:
        raise RuntimeError("找不到 040_13 ckpt25k published rows")
    model_path = ROOT / str(published.iloc[0]["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_INPUT_ROOT
    daily_frames = []
    comp_rows = []
    event_frames = []
    failures = []
    try:
        for year in YEARS:
            try:
                daily, comp, events = run_one(config, env_config, model_path, int(year))
                daily_frames.append(daily)
                comp_rows.append(comp)
                if not events.empty:
                    events = events.copy()
                    events.insert(0, "year", int(year))
                    event_frames.append(events)
            except Exception:
                failures.append({"year": int(year), "traceback": traceback.format_exc()[-5000:]})
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root

    daily_all = pd.concat(daily_frames, ignore_index=True, sort=False) if daily_frames else pd.DataFrame()
    comparison = pd.DataFrame(comp_rows)
    events_all = pd.concat(event_frames, ignore_index=True, sort=False) if event_frames else pd.DataFrame()
    failures_df = pd.DataFrame(failures)
    pub_cols = ["year", "final_grnwt", "total_irrigation", "total_n", "daily_csv_path", "model_path"]
    summaries = published[pub_cols].merge(comparison, on="year", how="left", suffixes=("_04013", "_04016"))
    daily_all.to_csv(TAB_DIR / "040_16_online_replay_daily.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(TAB_DIR / "040_16_online_summary_closure.csv", index=False, encoding="utf-8-sig")
    events_all.to_csv(TAB_DIR / "040_16_mgmt_events.csv", index=False, encoding="utf-8-sig")
    summaries.to_csv(TAB_DIR / "040_16_compare_to_04013_published.csv", index=False, encoding="utf-8-sig")
    failures_df.to_csv(TAB_DIR / "040_16_failures.csv", index=False, encoding="utf-8-sig")
    write_record(comparison, summaries, failures_df)
    result = {
        "task": TASK,
        "failed_runs": int(len(failures_df)),
        "comparison_csv": str((TAB_DIR / "040_16_online_summary_closure.csv").relative_to(ROOT)).replace("\\", "/"),
        "compare_to_04013_csv": str((TAB_DIR / "040_16_compare_to_04013_published.csv").relative_to(ROOT)).replace("\\", "/"),
        "events_csv": str((TAB_DIR / "040_16_mgmt_events.csv").relative_to(ROOT)).replace("\\", "/"),
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

