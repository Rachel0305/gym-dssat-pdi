from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calculate_five_site_wue_nue_from_summary_019_10 import num, parse_summary_out, select_matching_row
from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table

import ppo_safe_rendering
import run_sya_lowIC_ppo_yield_guardrail_v3_040_40 as ppo04040


TASK_ID = "041_00"
TASK_NAME = "sya_lowIC_teacher_candidate_search"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

STATION = "SYA"
SITE = "SY"
LOWIC_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
BASELINE_SUMMARY = (
    ROOT
    / "benchmark_results"
    / "040_42_sya_lowIC_04040_ckpt100k_validation_five_scenario_metric_bars"
    / "tables"
    / "040_42_sya_lowIC_04040_ckpt100k_five_scenario_metric_summary.csv"
)
SCENARIO_POOL = (
    ROOT
    / "Leave_One_experiments"
    / "all_year_weather_calibration_validation"
    / "scenario_pool"
    / "all_year_weather_scenario_pool.csv"
)
DEFAULT_YEARS = list(range(2014, 2024))


WATER_SCHEDULES: dict[str, dict[int, float]] = {
    "W225_ppo_like": {1: 45.0, 8: 30.0, 31: 45.0, 38: 30.0, 61: 45.0, 91: 30.0},
    "W240_ppo_plus_late": {1: 45.0, 8: 30.0, 31: 45.0, 38: 30.0, 61: 45.0, 91: 45.0},
    "W240_late_balanced": {1: 30.0, 30: 45.0, 60: 45.0, 90: 45.0, 105: 45.0, 120: 30.0},
    "W240_mid_late": {30: 45.0, 45: 45.0, 60: 45.0, 75: 30.0, 95: 45.0, 110: 30.0},
    "W195_conservative": {30: 45.0, 60: 45.0, 90: 45.0, 105: 30.0, 120: 30.0},
    "W150_late_saving": {45: 30.0, 75: 45.0, 95: 45.0, 115: 30.0},
}

N_SCHEDULES: dict[str, dict[int, float]] = {
    "N000_none": {},
    "N160_two80": {1: 80.0, 43: 80.0},
    "N200_early_mid": {1: 80.0, 43: 120.0},
    "N200_mid_late": {43: 80.0, 61: 120.0},
    "N240_three80": {1: 80.0, 43: 80.0, 61: 80.0},
    "N240_two120": {1: 120.0, 43: 120.0},
}

FOUR_BASELINES = {"null", "recorded_farmer_template", "dssat_auto", "official_extension_expert"}


def ensure_dirs() -> None:
    for rel in ["tables", "runs", "configs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
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


def validate_schedule(water: dict[int, float], nitrogen: dict[int, float]) -> None:
    if any(v not in {30.0, 45.0} for v in water.values()):
        raise ValueError(f"Illegal irrigation level in {water}")
    if any(v not in {80.0, 120.0} for v in nitrogen.values()):
        raise ValueError(f"Illegal nitrogen level in {nitrogen}")
    if sum(water.values()) > 240.0 + 1e-9:
        raise ValueError(f"Irrigation cap exceeded: {sum(water.values())}")
    if sum(nitrogen.values()) > 250.0 + 1e-9:
        raise ValueError(f"N cap exceeded: {sum(nitrogen.values())}")
    if sum(v for d, v in water.items() if d <= 90) > 195.0 + 1e-9:
        raise ValueError(f"Pre-DAP90 irrigation reserve cap exceeded: {water}")
    if any(d < 1 or d > 120 for d in water):
        raise ValueError(f"Irrigation DAP outside 1-120: {water}")
    if any(d < 1 or d > 90 for d in nitrogen):
        raise ValueError(f"N DAP outside 1-90: {nitrogen}")
    w_daps = sorted(water)
    n_daps = sorted(nitrogen)
    if any(b - a < 7 for a, b in zip(w_daps, w_daps[1:])):
        raise ValueError(f"Irrigation interval <7 d: {water}")
    if any(b - a < 7 for a, b in zip(n_daps, n_daps[1:])):
        raise ValueError(f"N interval <7 d: {nitrogen}")


def build_grid() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for w_name, water in WATER_SCHEDULES.items():
        for n_name, nitrogen in N_SCHEDULES.items():
            validate_schedule(water, nitrogen)
            rows.append(
                {
                    "candidate_id": f"{w_name}__{n_name}",
                    "water_schedule_id": w_name,
                    "n_schedule_id": n_name,
                    "water_schedule_json": json.dumps(water, sort_keys=True),
                    "n_schedule_json": json.dumps(nitrogen, sort_keys=True),
                    "requested_irrigation_mm": float(sum(water.values())),
                    "requested_nitrogen_kg_ha": float(sum(nitrogen.values())),
                    "pre_dap90_irrigation_mm": float(sum(v for d, v in water.items() if d <= 90)),
                }
            )
    return pd.DataFrame(rows)


def load_baseline_thresholds() -> pd.DataFrame:
    if not BASELINE_SUMMARY.exists():
        raise FileNotFoundError(BASELINE_SUMMARY)
    df = pd.read_csv(BASELINE_SUMMARY, keep_default_na=False)
    df = df[df["station_code"].eq(STATION) & df["scenario"].isin(FOUR_BASELINES)].copy()
    numeric = [
        "grain_yield_kg_ha",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "actual_irrigation_mm",
        "actual_nitrogen_kg_ha",
        "summary_irrigation_total",
        "summary_nitrogen_total",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    expert = df[df["scenario"].eq("official_extension_expert")][
        ["year", "grain_yield_kg_ha", "summary_irrigation_total", "summary_nitrogen_total"]
    ].rename(
        columns={
            "grain_yield_kg_ha": "expert_yield_kg_ha",
            "summary_irrigation_total": "expert_irrigation_mm",
            "summary_nitrogen_total": "expert_nitrogen_kg_ha",
        }
    )
    grouped = (
        df.groupby("year", as_index=False)
        .agg(
            four_max_yield_kg_ha=("grain_yield_kg_ha", "max"),
            four_max_wp_et_kg_m3=("WP_ET_kg_m3", "max"),
            four_max_pfp_n_kg_kg=("PFP_N_kg_kg", "max"),
        )
        .merge(expert, on="year", how="left")
    )
    return grouped


def build_selection_for_years(years: list[int]) -> pd.DataFrame:
    if not SCENARIO_POOL.exists():
        raise FileNotFoundError(SCENARIO_POOL)
    pool = pd.read_csv(SCENARIO_POOL, keep_default_na=False)
    pool["year"] = pd.to_numeric(pool["year"], errors="coerce").astype(int)
    selected = pool[pool["station_code"].eq(STATION) & pool["year"].isin([int(y) for y in years])].copy()
    if selected.empty:
        raise RuntimeError(f"No scenario pool rows for {STATION} years={years}")
    selected["selected_for_train"] = False
    selected["selected_for_eval"] = True
    selected["selection_reason"] = "041_00_sya_lowIC_teacher_candidate_search"
    return selected.sort_values(["station_code", "year"]).reset_index(drop=True)


def combined_schedule(water: dict[int, float], nitrogen: dict[int, float]) -> dict[int, dict[str, float]]:
    schedule: dict[int, dict[str, float]] = {}
    for dap, amount in water.items():
        schedule.setdefault(int(dap), {"amir": 0.0, "anfer": 0.0})["amir"] += float(amount)
    for dap, amount in nitrogen.items():
        schedule.setdefault(int(dap), {"amir": 0.0, "anfer": 0.0})["anfer"] += float(amount)
    return dict(sorted(schedule.items()))


def action_index_for(env: Any, amir: float, anfer: float) -> int:
    grid = getattr(env, "grid", None)
    if grid is None:
        raise RuntimeError("Wrapped env does not expose discrete action grid")
    matches = [
        idx
        for idx, raw in enumerate(grid)
        if abs(float(raw.get("amir", 0.0)) - float(amir)) < 1e-9
        and abs(float(raw.get("anfer", 0.0)) - float(anfer)) < 1e-9
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Cannot map action amir={amir}, anfer={anfer}; matches={matches}")
    return int(matches[0])


def copy_snapshot(env: Any, destination: Path) -> None:
    tmp = getattr(env.unwrapped, "_tmp_folder", None)
    if not tmp or not Path(tmp).exists():
        raise RuntimeError("PDI temporary output directory unavailable")
    if destination.exists():
        return
    shutil.copytree(Path(tmp), destination)


def run_one_candidate(year: int, candidate: pd.Series, thresholds: dict[str, float], config: dict, env_config: dict) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    run_id = f"{STATION}_{year}_{candidate_id}"
    run_dir = OUT / "runs" / run_id
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)

    water = {int(k): float(v) for k, v in json.loads(candidate["water_schedule_json"]).items()}
    nitrogen = {int(k): float(v) for k, v in json.loads(candidate["n_schedule_json"]).items()}
    schedule = combined_schedule(water, nitrogen)
    (run_dir / "schedule.json").write_text(json.dumps(schedule, indent=2, ensure_ascii=False), encoding="utf-8")

    env = None
    daily_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    fired: set[int] = set()
    status = "ok"
    notes = ""
    try:
        env = ppo04040.make_env_with_yield_guardrail(
            config,
            env_config,
            STATION,
            int(year),
            0,
            f"{run_id}_041_00_eval",
            evaluation=True,
        )
        obs, info = env.reset()
        for step in range(420):
            before = latest_observation_dict(env, obs, info)
            dap = int(round(scalar(before.get("dap", step + 1), step + 1)))
            requested = schedule[dap] if dap in schedule and dap not in fired else {"amir": 0.0, "anfer": 0.0}
            if dap in schedule:
                fired.add(dap)
            action_idx = action_index_for(env, requested["amir"], requested["anfer"])
            obs, reward, terminated, truncated, info = env.step(action_idx)
            latest = latest_observation_dict(env, obs, info)
            last = dict(getattr(env, "last_action_info", {}))
            row = {
                "station_code": STATION,
                "site": SITE,
                "year": int(year),
                "candidate_id": candidate_id,
                "step": int(step),
                "dap_before_action": int(dap),
                "yrdoy": scalar(latest.get("yrdoy")),
                "dap_after_step": scalar(latest.get("dap")),
                "rain": scalar(latest.get("rain")),
                "tmax": scalar(latest.get("tmax")),
                "tmin": scalar(latest.get("tmin")),
                "grnwt": scalar(latest.get("grnwt")),
                "topwt": scalar(latest.get("topwt")),
                "swfac": scalar(latest.get("swfac")),
                "nstres": scalar(latest.get("nstres")),
                "requested_irrigation_mm_action": float(requested["amir"]),
                "requested_nitrogen_kg_ha_action": float(requested["anfer"]),
                "safe_irrigation_mm_action": float(last.get("safe_action_amir", np.nan)),
                "safe_nitrogen_kg_ha_action": float(last.get("safe_action_anfer", np.nan)),
                "mask_forced_noop": bool(last.get("mask_forced_noop", False)),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
            daily_rows.append(row)
            if row["requested_irrigation_mm_action"] > 0 or row["requested_nitrogen_kg_ha_action"] > 0:
                action_rows.append(row.copy())
            if terminated or truncated:
                break
    except Exception:
        status = "failed"
        notes = traceback.format_exc()
    finally:
        if env is not None:
            try:
                copy_snapshot(env, run_dir / "pdi_tmp_snapshot_eval")
            finally:
                env.close()

    daily = pd.DataFrame(daily_rows)
    actions = pd.DataFrame(action_rows)
    daily.to_csv(run_dir / "daily_values.csv", index=False)
    actions.to_csv(run_dir / "requested_actions.csv", index=False)

    result: dict[str, Any] = {
        "station_code": STATION,
        "site": SITE,
        "year": int(year),
        "candidate_id": candidate_id,
        "water_schedule_id": str(candidate["water_schedule_id"]),
        "n_schedule_id": str(candidate["n_schedule_id"]),
        "water_schedule_json": str(candidate["water_schedule_json"]),
        "n_schedule_json": str(candidate["n_schedule_json"]),
        "requested_irrigation_mm": float(actions["requested_irrigation_mm_action"].sum()) if not actions.empty else 0.0,
        "requested_nitrogen_kg_ha": float(actions["requested_nitrogen_kg_ha_action"].sum()) if not actions.empty else 0.0,
        "safe_irrigation_mm": float(actions["safe_irrigation_mm_action"].sum()) if not actions.empty else 0.0,
        "safe_nitrogen_kg_ha": float(actions["safe_nitrogen_kg_ha_action"].sum()) if not actions.empty else 0.0,
        "missing_schedule_daps": json.dumps(sorted(set(schedule) - fired)),
        "mask_forced_noop_count": int(daily["mask_forced_noop"].sum()) if "mask_forced_noop" in daily else 0,
        "max_swfac": float(pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce").max()) if not daily.empty else np.nan,
        "max_nstres": float(pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce").max()) if not daily.empty else np.nan,
        "run_status": status,
        "notes": notes[-2500:] if notes else "",
        "run_dir": run_dir.relative_to(ROOT).as_posix(),
    }
    if status == "ok":
        snapshot = run_dir / "pdi_tmp_snapshot_eval"
        plantgro = parse_dssat_table(snapshot / "PlantGro.OUT")
        plantgro.to_csv(run_dir / "plantgro_parsed.csv", index=False)
        gwad = pd.to_numeric(plantgro["GWAD"], errors="coerce").dropna()
        cwad = pd.to_numeric(plantgro["CWAD"], errors="coerce").dropna()
        final_gwad = float(gwad.iloc[-1]) if not gwad.empty else np.nan
        final_cwad = float(cwad.iloc[-1]) if not cwad.empty else np.nan
        summary_rows = parse_summary_out(snapshot / "Summary.OUT")
        srow, match_score, row_index = select_matching_row(
            summary_rows,
            final_gwad,
            result["safe_irrigation_mm"],
            result["safe_nitrogen_kg_ha"],
        )
        ircm, nicm, nucm, etcp = num(srow, "IRCM"), num(srow, "NICM"), num(srow, "NUCM"), num(srow, "ETCP")
        ypem, ypnam = num(srow, "YPEM"), num(srow, "YPNAM")
        result.update(
            {
                "grain_yield_kg_ha": final_gwad,
                "biomass_kg_ha": final_cwad,
                "summary_irrigation_total": ircm,
                "summary_nitrogen_total": nicm,
                "nitrogen_uptake_kg_ha": nucm,
                "etcp_mm": etcp,
                "WP_ET_kg_m3": ypem * 0.1 if ypem is not None and ypem >= 0 else np.nan,
                "PFP_N_kg_kg": ypnam if nicm and nicm > 0 and ypnam is not None and ypnam >= 0 else np.nan,
                "summary_match_score": match_score,
                "summary_row_index": row_index,
            }
        )
    else:
        result.update(
            {
                "grain_yield_kg_ha": np.nan,
                "biomass_kg_ha": np.nan,
                "summary_irrigation_total": np.nan,
                "summary_nitrogen_total": np.nan,
                "nitrogen_uptake_kg_ha": np.nan,
                "etcp_mm": np.nan,
                "WP_ET_kg_m3": np.nan,
                "PFP_N_kg_kg": np.nan,
                "summary_match_score": np.nan,
                "summary_row_index": np.nan,
            }
        )

    result.update(
        {
            "gap_yield_vs_four_max": result["grain_yield_kg_ha"] - thresholds["four_max_yield_kg_ha"],
            "gap_wp_et_vs_four_max": result["WP_ET_kg_m3"] - thresholds["four_max_wp_et_kg_m3"],
            "gap_pfp_n_vs_four_max": result["PFP_N_kg_kg"] - thresholds["four_max_pfp_n_kg_kg"],
            "gap_yield_vs_expert": result["grain_yield_kg_ha"] - thresholds["expert_yield_kg_ha"],
            "delta_i_vs_expert": result["summary_irrigation_total"] - thresholds["expert_irrigation_mm"],
            "delta_n_vs_expert": result["summary_nitrogen_total"] - thresholds["expert_nitrogen_kg_ha"],
        }
    )
    result["yield_win_vs_four_max"] = bool(result["gap_yield_vs_four_max"] > 0)
    result["wp_et_win_vs_four_max"] = bool(result["gap_wp_et_vs_four_max"] > 0)
    result["pfp_n_win_vs_four_max"] = bool(result["gap_pfp_n_vs_four_max"] > 0)
    result["any_metric_win_vs_four_max"] = bool(
        result["yield_win_vs_four_max"] or result["wp_et_win_vs_four_max"] or result["pfp_n_win_vs_four_max"]
    )
    result["expert_98_yield_and_resource_saving"] = bool(
        result["grain_yield_kg_ha"] >= 0.98 * thresholds["expert_yield_kg_ha"]
        and (result["delta_i_vs_expert"] < 0 or result["delta_n_vs_expert"] < 0)
    )
    result["teacher_candidate"] = bool(
        result["run_status"] == "ok"
        and not json.loads(result["missing_schedule_daps"])
        and result["mask_forced_noop_count"] == 0
        and (result["any_metric_win_vs_four_max"] or result["expert_98_yield_and_resource_saving"])
    )

    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    return result


def write_record(result: dict[str, Any], summary: pd.DataFrame, teachers: pd.DataFrame) -> None:
    top_cols = [
        "year",
        "candidate_id",
        "grain_yield_kg_ha",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "summary_irrigation_total",
        "summary_nitrogen_total",
        "gap_yield_vs_four_max",
        "gap_wp_et_vs_four_max",
        "gap_pfp_n_vs_four_max",
        "teacher_candidate",
    ]
    top = summary.sort_values(
        ["teacher_candidate", "any_metric_win_vs_four_max", "grain_yield_kg_ha"],
        ascending=[False, False, False],
    )[top_cols]
    lines = [
        f"# {TASK_ID} SYA lowIC teacher 候选轨迹搜索记录",
        "",
        "## 结论",
        "",
        f"- 分支：`{result['branch']}`",
        f"- 年份数：{len(result['years'])}",
        f"- 每年候选数：{result['candidate_count_per_year']}",
        f"- 总 DSSAT 候选调用数：{result['candidate_runs_attempted']}",
        f"- 成功调用数：{result['successful_runs']}",
        f"- teacher 候选数：{result['teacher_candidate_count']}",
        "",
        "## 关键边界",
        "",
        "- 本任务只在 lowIC 输入条件下重新搜索 teacher 候选，不复用 originIC teacher。",
        "- 本任务不训练 PPO/DQN/SAC。",
        "- 候选动作边界与 040_40 PPO 可执行动作保持一致。",
        "- 本任务不根据结果现场扩展候选网格。",
        "",
        "## teacher 候选预览",
        "",
        md_table(teachers[top_cols] if not teachers.empty else teachers, max_rows=40),
        "",
        "## 总候选预览",
        "",
        md_table(top, max_rows=60),
        "",
        "## 输出文件",
        "",
        f"- 候选总表：`{result['outputs']['candidate_summary']}`",
        f"- teacher 候选表：`{result['outputs']['teacher_candidates']}`",
        f"- 动作事件表：`{result['outputs']['requested_actions']}`",
        f"- 候选网格：`{result['outputs']['candidate_grid']}`",
        f"- JSON 结果：`{result['outputs']['result_json']}`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run(years: list[int], max_candidates: int | None) -> None:
    ensure_dirs()
    grid = build_grid()
    if max_candidates is not None:
        grid = grid.head(int(max_candidates)).copy()
    thresholds = load_baseline_thresholds()
    thresholds = thresholds[thresholds["year"].isin(years)].copy()
    selection = build_selection_for_years(years)
    grid.to_csv(OUT / "tables" / "041_00_candidate_grid.csv", index=False)
    preview = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "mode": "dry_run",
        "station": STATION,
        "years": years,
        "lowIC_input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "lowIC_input_root_exists": LOWIC_INPUT_ROOT.exists(),
        "baseline_summary": BASELINE_SUMMARY.relative_to(ROOT).as_posix(),
        "baseline_summary_exists": BASELINE_SUMMARY.exists(),
        "candidate_count_per_year": int(len(grid)),
        "total_candidate_runs": int(len(grid) * len(years)),
        "scenario_pool_rows": int(len(selection)),
        "water_schedules": list(WATER_SCHEDULES),
        "n_schedules": list(N_SCHEDULES),
        "threshold_year_count": int(len(thresholds)),
        "next_step_allowed": bool(LOWIC_INPUT_ROOT.exists() and BASELINE_SUMMARY.exists() and len(thresholds) == len(years)),
    }
    print(json.dumps(preview, indent=2, ensure_ascii=False))


def run_search(years: list[int], max_candidates: int | None) -> None:
    ensure_dirs()
    grid = build_grid()
    if max_candidates is not None:
        grid = grid.head(int(max_candidates)).copy()
    grid.to_csv(OUT / "tables" / "041_00_candidate_grid.csv", index=False)
    thresholds_df = load_baseline_thresholds()
    thresholds_df = thresholds_df[thresholds_df["year"].isin(years)].copy()
    if len(thresholds_df) != len(years):
        raise RuntimeError(f"Missing baseline thresholds: expected {years}, got {thresholds_df['year'].tolist()}")
    thresholds_by_year = {int(row.year): row._asdict() for row in thresholds_df.itertuples(index=False)}
    selection = build_selection_for_years(years)
    config = ppo04040.load_config()
    env_config = ppo04040.base03222.direct_ppo.build_env_config(config, selection)
    (OUT / "configs" / "041_00_env_config.json").write_text(
        json.dumps(env_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_INPUT_ROOT
    rows: list[dict[str, Any]] = []
    try:
        total = len(years) * len(grid)
        counter = 0
        for year in years:
            for _, candidate in grid.iterrows():
                counter += 1
                print(f"[041_00] {counter}/{total} {STATION}{year} {candidate['candidate_id']}", flush=True)
                rows.append(run_one_candidate(int(year), candidate, thresholds_by_year[int(year)], config, env_config))
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root

    summary = pd.DataFrame(rows)
    summary_path = OUT / "tables" / "041_00_candidate_summary.csv"
    summary.to_csv(summary_path, index=False)
    teachers = summary[summary["teacher_candidate"].astype(bool)].copy()
    teachers_path = OUT / "tables" / "041_00_teacher_candidates.csv"
    teachers.to_csv(teachers_path, index=False)

    action_frames = []
    for run_dir in summary["run_dir"].dropna().astype(str):
        path = ROOT / run_dir / "requested_actions.csv"
        if path.exists() and path.stat().st_size > 0:
            try:
                action_frames.append(pd.read_csv(path, keep_default_na=False))
            except pd.errors.EmptyDataError:
                # Some DSSAT seasons can terminate before any scheduled action is
                # fired, leaving a zero-column CSV.  This is not a failed
                # candidate; the summary already records missing_schedule_daps.
                continue
    actions = pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame()
    actions_path = OUT / "tables" / "041_00_requested_actions.csv"
    actions.to_csv(actions_path, index=False)

    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": "A_teacher_candidates_found" if len(teachers) > 0 else "C_no_teacher_candidates_found",
        "training_run": False,
        "station": STATION,
        "site": SITE,
        "years": years,
        "lowIC_input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "candidate_count_per_year": int(len(grid)),
        "candidate_runs_attempted": int(len(summary)),
        "successful_runs": int(summary["run_status"].eq("ok").sum()),
        "failed_runs": int(summary["run_status"].ne("ok").sum()),
        "teacher_candidate_count": int(len(teachers)),
        "outputs": {
            "candidate_summary": summary_path.relative_to(ROOT).as_posix(),
            "teacher_candidates": teachers_path.relative_to(ROOT).as_posix(),
            "requested_actions": actions_path.relative_to(ROOT).as_posix(),
            "candidate_grid": (OUT / "tables" / "041_00_candidate_grid.csv").relative_to(ROOT).as_posix(),
            "result_json": (OUT / "041_00_result.json").relative_to(ROOT).as_posix(),
            "record_md": DOC.relative_to(ROOT).as_posix(),
        },
    }
    (OUT / "041_00_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_record(result, summary, teachers)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def parse_years(raw: str) -> list[int]:
    years: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = [int(x) for x in part.split("-", 1)]
            years.extend(range(a, b + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--years", default="2014-2023")
    parser.add_argument("--max-candidates", type=int, default=None, help="Optional smoke limiter; omit for the preregistered full grid.")
    args = parser.parse_args()
    years = parse_years(args.years)
    if args.dry_run:
        dry_run(years, args.max_candidates)
    else:
        run_search(years, args.max_candidates)


if __name__ == "__main__":
    main()
