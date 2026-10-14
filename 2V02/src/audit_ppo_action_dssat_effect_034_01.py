from __future__ import annotations

import json
import math
import re
import shutil
import sys
import time
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

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch_ppo
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as ppo_env
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline_034
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo
from ppo_action_safety import normalize_action
from ppo_safe_rendering import check_rendered_input, ensure_project_on_path


TASK_ID = "034_01"
OUT = ROOT / "benchmark_results" / "034_01_ppo_action_dssat_effect_audit"
DOC = ROOT / "docs" / "034_01_ppo_action_dssat_effect_audit_record.md"
PROMPT = ROOT / "prompts" / "034_01_ppo_action_dssat_effect_audit.md"
SOURCE_SUMMARY = ROOT / "benchmark_results" / "033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun" / "evaluation" / "032_22_checkpoint_validation_summary.csv"
SOURCE_SPLIT = ROOT / "benchmark_results" / "033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun" / "configs" / "033_04_available_weather_half_split_years.csv"
MAX_STEPS = 260
SEED = 0


def ensure_dirs() -> None:
    for rel in ["configs", "evaluation", "snapshots", "rendered_inputs", "logs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def scalar(value: Any, default: float = np.nan) -> float:
    try:
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            if len(value) == 0:
                return default
            value = np.asarray(value).flatten()[0]
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def latest_observation_dict(env: Any, obs: Any, info: Any | None = None) -> dict[str, Any]:
    history = getattr(env.unwrapped, "history", {})
    if isinstance(history, dict):
        observations = history.get("observation", [])
        if observations and isinstance(observations[-1], dict):
            return dict(observations[-1])
    if isinstance(info, dict) and isinstance(info.get("observation"), dict):
        return dict(info["observation"])
    if isinstance(obs, dict):
        return dict(obs)
    return {}


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


def select_cases() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_SUMMARY, keep_default_na=False)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["checkpoint_step"] = pd.to_numeric(df["checkpoint_step"], errors="coerce").astype(int)
    df["total_irrigation"] = pd.to_numeric(df["total_irrigation"], errors="coerce").fillna(0.0)
    df["total_n"] = pd.to_numeric(df["total_n"], errors="coerce").fillna(0.0)
    ok = df[
        df["run_status"].astype(str).str.startswith("ok")
        & ((df["total_irrigation"] > 0) | (df["total_n"] > 0))
        & df["model_path"].astype(str).ne("")
    ].copy()
    rows: list[dict[str, Any]] = []
    for station, group in ok.sort_values(["station_code", "year", "checkpoint_step"]).groupby("station_code"):
        row = group.iloc[0].to_dict()
        model_path = ROOT / str(row["model_path"])
        row["model_exists"] = bool(model_path.exists())
        row["selection_reason"] = "first_033_04_ok_validation_row_with_nonzero_safe_action_per_station"
        rows.append(row)
    return pd.DataFrame(rows)


def load_config_and_env_config() -> tuple[dict[str, Any], dict[str, Any]]:
    config = batch_ppo.load_config()
    config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    split = pd.read_csv(SOURCE_SPLIT, keep_default_na=False)
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    selection = batch_ppo.build_selection(split)
    direct_ppo.OUTPUT_ROOT = OUT
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "034_01_resolved_env_config.yaml")
    shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)
    return config, env_config


def zero_action(env: Any) -> np.ndarray:
    return normalize_action(env.formator.action_names, env.formator.action_space_dict, {"amir": 0.0, "anfer": 0.0})


def metrics_from_env_snapshot(env: Any, scenario_dir: Path, final_yield: float) -> dict[str, Any]:
    tmp = siteppo.snapshot_from_env(env)
    if scenario_dir.exists():
        shutil.rmtree(scenario_dir)
    shutil.copytree(tmp, scenario_dir)
    return baseline_034.metrics_from_snapshot(scenario_dir, final_yield)


def run_external_case(
    config: dict[str, Any],
    env_config: dict[str, Any],
    case: pd.Series,
    *,
    scenario: str,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    station = str(case["station_code"])
    year = int(case["year"])
    checkpoint_step = int(case.get("checkpoint_step", 0) or 0)
    model_path = ROOT / str(case["model_path"])
    env = ppo_env.make_env(config, env_config, station, year, SEED, f"{station}_{year}_{TASK_ID}_{scenario}", evaluation=True)
    weather = direct_ppo.weather_for_daily(config)
    render_checks: list[dict[str, Any]] = []
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import get_action_masks

        model = MaskablePPO.load(str(model_path), device="cpu") if scenario == "ppo_external_replay" else None
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, station, year)["planting_date"])
        records: list[dict[str, Any]] = []
        while not done and step_count < MAX_STEPS:
            latest_pre = latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest_pre.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            if scenario == "ppo_external_replay":
                mask = get_action_masks(env)
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            else:
                action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            wrow = baseline_034.weather_row(weather, station, date)
            action_info = dict(getattr(env, "last_action_info", {}))
            records.append(
                {
                    "station_code": station,
                    "year": year,
                    "checkpoint_step": checkpoint_step,
                    "scenario": scenario,
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": scalar(wrow.get("rain"), np.nan),
                    "tmax": scalar(wrow.get("tmax"), np.nan),
                    "tmin": scalar(wrow.get("tmin"), np.nan),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "reward": float(reward),
                    **action_info,
                    "done": done,
                }
            )
            step_count += 1
        daily = pd.DataFrame(records)
        if daily.empty or not bool(daily["done"].iloc[-1]):
            raise RuntimeError(f"{station}{year} {scenario} 未在 {MAX_STEPS} 步内结束")
        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
        snapshot = OUT / "snapshots" / station / str(year) / scenario
        metrics = metrics_from_env_snapshot(env, snapshot, final_y)
        safe_i = float(pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        safe_n = float(pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        summary = {
            "station_code": station,
            "year": year,
            "checkpoint_step": checkpoint_step,
            "scenario": scenario,
            "model_path": str(model_path.relative_to(ROOT)) if model_path.exists() else "",
            "safe_action_irrigation_sum_mm": safe_i,
            "safe_action_n_sum_kg_ha": safe_n,
            "summary_irrigation_mm": float(metrics["actual_irrigation_mm"]),
            "summary_n_kg_ha": float(metrics["actual_nitrogen_kg_ha"]),
            "grain_yield_kg_ha": final_y,
            "max_swfac": float(pd.to_numeric(daily["swfac"], errors="coerce").max()),
            "max_nstres": float(pd.to_numeric(daily["nstres"], errors="coerce").max()),
            "snapshot_path": str(snapshot.relative_to(ROOT)).replace("\\", "/"),
            **metrics,
        }
        return daily, summary, render_checks
    finally:
        env.close()


def schedule_from_daily(daily: pd.DataFrame) -> dict[int, dict[str, float]]:
    schedule: dict[int, dict[str, float]] = {}
    for _, row in daily.iterrows():
        dap = int(row["dap"])
        i = scalar(row.get("safe_action_amir"), 0.0)
        n = scalar(row.get("safe_action_anfer"), 0.0)
        if i > 1e-9 or n > 1e-9:
            schedule[dap] = {"amir": float(i), "anfer": float(n)}
    return schedule


def run_static_same_schedule(
    config: dict[str, Any],
    env_config: dict[str, Any],
    case: pd.Series,
    schedule: dict[int, dict[str, float]],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    station = str(case["station_code"])
    year = int(case["year"])
    checkpoint_step = int(case.get("checkpoint_step", 0) or 0)
    scenario = "ppo_static_mzx_same_schedule"
    env, env_args = baseline_034.make_base_env(
        env_config,
        station,
        year,
        f"{station}_{year}_{TASK_ID}_{scenario}",
        auto_management=False,
        static_schedule=schedule,
    )
    weather = direct_ppo.weather_for_daily(config)
    render_checks = check_rendered_input(Path(env_args["fileX_template_path"]))
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, station, year)["planting_date"])
        records: list[dict[str, Any]] = []
        while not done and step_count < MAX_STEPS:
            latest_pre = latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest_pre.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            requested = schedule.get(dap, {"amir": 0.0, "anfer": 0.0})
            obs, reward, terminated, truncated, info = env.step(zero_action(env))
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            wrow = baseline_034.weather_row(weather, station, date)
            records.append(
                {
                    "station_code": station,
                    "year": year,
                    "checkpoint_step": checkpoint_step,
                    "scenario": scenario,
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": scalar(wrow.get("rain"), np.nan),
                    "tmax": scalar(wrow.get("tmax"), np.nan),
                    "tmin": scalar(wrow.get("tmin"), np.nan),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "safe_action_amir": float(requested.get("amir", 0.0)),
                    "safe_action_anfer": float(requested.get("anfer", 0.0)),
                    "reward": float(reward),
                    "done": done,
                }
            )
            step_count += 1
        daily = pd.DataFrame(records)
        if daily.empty or not bool(daily["done"].iloc[-1]):
            raise RuntimeError(f"{station}{year} {scenario} 未在 {MAX_STEPS} 步内结束")
        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
        snapshot = OUT / "snapshots" / station / str(year) / scenario
        metrics = metrics_from_env_snapshot(env, snapshot, final_y)
        safe_i = float(pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0).sum())
        safe_n = float(pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0).sum())
        summary = {
            "station_code": station,
            "year": year,
            "checkpoint_step": checkpoint_step,
            "scenario": scenario,
            "model_path": str((ROOT / str(case["model_path"])).relative_to(ROOT)),
            "safe_action_irrigation_sum_mm": safe_i,
            "safe_action_n_sum_kg_ha": safe_n,
            "summary_irrigation_mm": float(metrics["actual_irrigation_mm"]),
            "summary_n_kg_ha": float(metrics["actual_nitrogen_kg_ha"]),
            "grain_yield_kg_ha": final_y,
            "max_swfac": float(pd.to_numeric(daily["swfac"], errors="coerce").max()),
            "max_nstres": float(pd.to_numeric(daily["nstres"], errors="coerce").max()),
            "snapshot_path": str(snapshot.relative_to(ROOT)).replace("\\", "/"),
            **metrics,
        }
        return daily, summary, render_checks
    finally:
        env.close()


def build_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (station, year, checkpoint), group in summary.groupby(["station_code", "year", "checkpoint_step"]):
        wide = {str(row["scenario"]): row for _, row in group.iterrows()}
        null = wide.get("null_external_noop")
        ext = wide.get("ppo_external_replay")
        static = wide.get("ppo_static_mzx_same_schedule")
        if null is None or ext is None or static is None:
            continue
        rows.append(
            {
                "station_code": station,
                "year": int(year),
                "checkpoint_step": int(checkpoint),
                "ppo_safe_i": float(ext["safe_action_irrigation_sum_mm"]),
                "ppo_safe_n": float(ext["safe_action_n_sum_kg_ha"]),
                "external_summary_i": float(ext["summary_irrigation_mm"]),
                "external_summary_n": float(ext["summary_n_kg_ha"]),
                "static_summary_i": float(static["summary_irrigation_mm"]),
                "static_summary_n": float(static["summary_n_kg_ha"]),
                "null_yield": float(null["grain_yield_kg_ha"]),
                "external_yield": float(ext["grain_yield_kg_ha"]),
                "static_yield": float(static["grain_yield_kg_ha"]),
                "external_minus_null_yield": float(ext["grain_yield_kg_ha"]) - float(null["grain_yield_kg_ha"]),
                "static_minus_null_yield": float(static["grain_yield_kg_ha"]) - float(null["grain_yield_kg_ha"]),
                "external_summary_matches_safe": abs(float(ext["summary_irrigation_mm"]) - float(ext["safe_action_irrigation_sum_mm"])) < 1e-6
                and abs(float(ext["summary_n_kg_ha"]) - float(ext["safe_action_n_sum_kg_ha"])) < 1e-6,
                "static_summary_matches_safe": abs(float(static["summary_irrigation_mm"]) - float(ext["safe_action_irrigation_sum_mm"])) < 1e-6
                and abs(float(static["summary_n_kg_ha"]) - float(ext["safe_action_n_sum_kg_ha"])) < 1e-6,
                "external_behaves_like_null": abs(float(ext["grain_yield_kg_ha"]) - float(null["grain_yield_kg_ha"])) < 1e-6
                and abs(float(ext["summary_irrigation_mm"]) - float(null["summary_irrigation_mm"])) < 1e-6
                and abs(float(ext["summary_n_kg_ha"]) - float(null["summary_n_kg_ha"])) < 1e-6,
                "static_changes_dssat": abs(float(static["grain_yield_kg_ha"]) - float(null["grain_yield_kg_ha"])) > 1e-6
                or abs(float(static["summary_irrigation_mm"]) - float(null["summary_irrigation_mm"])) > 1e-6
                or abs(float(static["summary_n_kg_ha"]) - float(null["summary_n_kg_ha"])) > 1e-6,
                "external_summary_zero_despite_safe_action": (float(ext["safe_action_irrigation_sum_mm"]) + float(ext["safe_action_n_sum_kg_ha"]) > 0)
                and abs(float(ext["summary_irrigation_mm"])) < 1e-6
                and abs(float(ext["summary_n_kg_ha"])) < 1e-6,
            }
        )
    comp = pd.DataFrame(rows)
    if not comp.empty:
        comp["diagnosis"] = np.where(
            (comp["ppo_safe_i"] + comp["ppo_safe_n"] > 0)
            & (~comp["external_summary_matches_safe"])
            & (comp["external_behaves_like_null"]),
            "external_action_not_applied_to_dssat",
            np.where(
                comp["external_summary_matches_safe"] & comp["static_summary_matches_safe"],
                "external_action_applied",
                "mixed_or_requires_followup",
            ),
        )
    return comp


def write_record(cases: pd.DataFrame, summary: pd.DataFrame, comparison: pd.DataFrame, failures: pd.DataFrame, elapsed: float) -> None:
    lines = [
        "# 034_01 PPO 动作是否真实进入 DSSAT 的三角审计记录",
        "",
        "## 结论先说",
        "",
    ]
    if not comparison.empty:
        diag_counts = comparison.groupby("diagnosis").size().reset_index(name="n")
        lines.extend([md_table(diag_counts, 20), ""])
    else:
        lines.extend(["无可判定对照。", ""])
    lines.extend(
        [
            "## 固定边界",
            "",
            "- 不重新训练 PPO；只加载 033_04 已有 checkpoint。",
            "- PPO 外部回放沿用 033_04 的 `StressAwareDiscreteWrapper` + action safety 路径。",
            "- 静态回放使用同一 PPO safe-action 序列写入 `.MZX` 的 `@I/@F` 管理表，step 阶段只发送 no-op。",
            "- PPO 代码层动作累计值与 DSSAT `Summary.OUT` 实际执行值分开报告。",
            "- 本任务优先判定 PPO 外部 action 是否落地；静态 MZX 同序列回放只作为辅助参照。若多事件静态回放未完全等于 safe-action 总量，另立后续静态多事件格式修复，不影响外部 action 是否进入 DSSAT 的判定。",
            "",
            "## 案例选择",
            "",
            md_table(cases[["station_code", "year", "checkpoint_step", "total_irrigation", "total_n", "action_sequence", "model_path", "model_exists"]], 20),
            "",
            "## 三角审计 summary",
            "",
            md_table(summary[["station_code", "year", "checkpoint_step", "scenario", "safe_action_irrigation_sum_mm", "safe_action_n_sum_kg_ha", "summary_irrigation_mm", "summary_n_kg_ha", "grain_yield_kg_ha", "max_swfac", "max_nstres"]], 80),
            "",
            "## 对照判定",
            "",
            md_table(comparison, 80),
            "",
            "## 失败记录",
            "",
            md_table(failures, 50),
            "",
            "## 输出文件",
            "",
            "- `benchmark_results/034_01_ppo_action_dssat_effect_audit/evaluation/034_01_case_selection.csv`",
            "- `benchmark_results/034_01_ppo_action_dssat_effect_audit/evaluation/034_01_summary.csv`",
            "- `benchmark_results/034_01_ppo_action_dssat_effect_audit/evaluation/034_01_daily.csv`",
            "- `benchmark_results/034_01_ppo_action_dssat_effect_audit/evaluation/034_01_comparison.csv`",
            "",
            f"耗时：{elapsed:.1f} 秒",
        ]
    )
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    ensure_dirs()
    ensure_project_on_path()
    config, env_config = load_config_and_env_config()
    cases = select_cases()
    cases.to_csv(OUT / "evaluation" / "034_01_case_selection.csv", index=False, encoding="utf-8-sig")
    summaries: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    render_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for _, case in cases.iterrows():
        station = str(case["station_code"])
        year = int(case["year"])
        checkpoint = int(case["checkpoint_step"])
        try:
            null_daily, null_summary, null_render = run_external_case(config, env_config, case, scenario="null_external_noop")
            ppo_daily, ppo_summary, ppo_render = run_external_case(config, env_config, case, scenario="ppo_external_replay")
            schedule = schedule_from_daily(ppo_daily)
            static_daily, static_summary, static_render = run_static_same_schedule(config, env_config, case, schedule)
            daily_frames.extend([null_daily, ppo_daily, static_daily])
            summaries.extend([null_summary, ppo_summary, static_summary])
            for item in null_render + ppo_render + static_render:
                out = dict(item)
                out.update({"station_code": station, "year": year, "checkpoint_step": checkpoint})
                render_rows.append(out)
        except Exception:
            failures.append(
                {
                    "station_code": station,
                    "year": year,
                    "checkpoint_step": checkpoint,
                    "status": "failed",
                    "traceback": traceback.format_exc()[-5000:],
                }
            )
    summary = pd.DataFrame(summaries)
    daily = pd.concat(daily_frames, ignore_index=True, sort=False) if daily_frames else pd.DataFrame()
    render = pd.DataFrame(render_rows)
    failures_df = pd.DataFrame(failures)
    comparison = build_comparison(summary) if not summary.empty else pd.DataFrame()
    summary.to_csv(OUT / "evaluation" / "034_01_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT / "evaluation" / "034_01_daily.csv", index=False, encoding="utf-8-sig")
    render.to_csv(OUT / "evaluation" / "034_01_render_check.csv", index=False, encoding="utf-8-sig")
    failures_df.to_csv(OUT / "evaluation" / "034_01_failures.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "evaluation" / "034_01_comparison.csv", index=False, encoding="utf-8-sig")
    write_record(cases, summary, comparison, failures_df, time.time() - start)
    result = {
        "task": TASK_ID,
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "case_selection": str((OUT / "evaluation" / "034_01_case_selection.csv").relative_to(ROOT)).replace("\\", "/"),
        "summary": str((OUT / "evaluation" / "034_01_summary.csv").relative_to(ROOT)).replace("\\", "/"),
        "comparison": str((OUT / "evaluation" / "034_01_comparison.csv").relative_to(ROOT)).replace("\\", "/"),
        "n_cases": int(len(cases)),
        "n_failures": int(len(failures_df)),
    }
    (OUT / "034_01_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not comparison.empty:
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
