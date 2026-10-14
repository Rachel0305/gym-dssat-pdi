from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base032
import run_sya_lowIC_ppo_irrigation_reallocation_counterfactual_040_07 as replay04007
import build_sya_lowIC_04036_ckpt100k_validation_five_scenario_metric_bars_040_38 as base04038
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo


TASK_ID = "042_17"
TASK_NAME = "sya_lowIC_2017_04212_vs_04216_irrigation_counterfactual"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
TAB_DIR = OUT / "tables"
DAILY_DIR = OUT / "daily_outputs"
SNAP_DIR = OUT / "snapshots"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

STATION = "SYA"
SITE = "SY"
YEAR = 2017
SEED = 0
LOWIC_INPUT_ROOT = base04038.LOWIC_INPUT_ROOT

DAILY_04212 = (
    ROOT
    / "benchmark_results"
    / "042_12_sya_lowIC_binary_timing_25k_candidate_audit"
    / "daily_outputs"
    / "SYA_2017_04036_ckpt100k_replay_daily.csv"
)
DAILY_04216 = (
    ROOT
    / "benchmark_results"
    / "042_16_sya_lowIC_ternary_irrigation_binary_nitrogen_maskableppo"
    / "daily_outputs"
    / "SYA"
    / "SYA_2017_seed0_ckpt25000_daily.csv"
)


def ensure_dirs() -> None:
    for path in [OUT, TAB_DIR, DAILY_DIR, SNAP_DIR, OUT / "configs"]:
        path.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)


def scalar(x: Any, default: float = np.nan) -> float:
    return base032.scalar(x, default=default)


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
    return "\n".join(lines)


def load_config_and_env_config() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    old_out = base04038.OUT
    old_tab = base04038.TAB_DIR
    old_daily = base04038.DAILY_DIR
    old_snap = base04038.SNAP_DIR
    try:
        base04038.OUT = OUT
        base04038.TAB_DIR = TAB_DIR
        base04038.DAILY_DIR = DAILY_DIR
        base04038.SNAP_DIR = SNAP_DIR
        config, env_config, selection = base04038.load_config_and_env_config()
    finally:
        base04038.OUT = old_out
        base04038.TAB_DIR = old_tab
        base04038.DAILY_DIR = old_daily
        base04038.SNAP_DIR = old_snap
    return config, env_config, selection


def actions_by_dap(path: Path) -> dict[int, tuple[float, float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return replay04007.original_actions_by_dap(path.relative_to(ROOT).as_posix())


def split_i_n(schedule: dict[int, tuple[float, float]]) -> tuple[dict[int, float], dict[int, float]]:
    i = {dap: float(v[0]) for dap, v in schedule.items() if abs(float(v[0])) > 1e-9}
    n = {dap: float(v[1]) for dap, v in schedule.items() if abs(float(v[1])) > 1e-9}
    return i, n


def combine(i_sched: dict[int, float], n_sched: dict[int, float]) -> dict[int, tuple[float, float]]:
    daps = sorted(set(i_sched) | set(n_sched))
    return {dap: (float(i_sched.get(dap, 0.0)), float(n_sched.get(dap, 0.0))) for dap in daps}


def action_sequence(schedule: dict[int, tuple[float, float]]) -> str:
    return "; ".join(f"DAP{dap} I{i:g}/N{n:g}" for dap, (i, n) in sorted(schedule.items()) if abs(i) > 1e-9 or abs(n) > 1e-9)


def build_schedules() -> tuple[pd.DataFrame, dict[str, dict[int, tuple[float, float]]]]:
    s12 = actions_by_dap(DAILY_04212)
    s16 = actions_by_dap(DAILY_04216)
    i12, n12 = split_i_n(s12)
    i16, n16 = split_i_n(s16)

    schedules: dict[str, dict[int, tuple[float, float]]] = {
        "baseline_04212_full": s12,
        "candidate_04216_full": s16,
        "04216_irrigation_04212_nitrogen": combine(i16, n12),
        "04212_irrigation_04216_nitrogen": combine(i12, n16),
    }

    for dap in sorted(i16):
        work = dict(i16)
        removed = work.pop(dap)
        schedules[f"04216_without_I_DAP{dap}"] = combine(work, n16)
    for dap in sorted(n16):
        work = dict(n16)
        removed = work.pop(dap)
        schedules[f"04216_without_N_DAP{dap}"] = combine(i16, work)

    rows: list[dict[str, Any]] = []
    for branch, schedule in schedules.items():
        rows.append(
            {
                "branch": branch,
                "planned_total_irrigation": sum(i for i, _n in schedule.values()),
                "planned_total_n": sum(n for _i, n in schedule.values()),
                "irrigation_events": sum(1 for i, _n in schedule.values() if i > 0),
                "nitrogen_events": sum(1 for _i, n in schedule.values() if n > 0),
                "action_sequence": action_sequence(schedule),
            }
        )
    return pd.DataFrame(rows), schedules


def stress_summary(daily: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in ["swfac", "nstres"]:
        s = pd.to_numeric(daily.get(col, pd.Series(dtype=float)), errors="coerce")
        out[f"max_{col}"] = float(s.max()) if len(s) else np.nan
        for threshold in [0.001, 0.01, 0.05]:
            out[f"{col}_days_gt_{str(threshold).replace('.', 'p')}"] = int((s > threshold).sum())
    return out


def replay_schedule(
    config: dict[str, Any],
    env_config: dict[str, Any],
    branch: str,
    schedule: dict[int, tuple[float, float]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    weather = direct_ppo.weather_for_daily(config)
    records: list[dict[str, Any]] = []
    old_input = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_out = direct_ppo.OUTPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_INPUT_ROOT
    direct_ppo.OUTPUT_ROOT = OUT
    env = None
    try:
        env = base032.make_env(config, env_config, STATION, YEAR, SEED, f"{STATION}_{YEAR}_{TASK_ID}_{branch}", evaluation=True)
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, STATION, YEAR)["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = base032.latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            i, n = schedule.get(dap, (0.0, 0.0))
            action = replay04007.action_index_for(config, i, n)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = base032.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(STATION)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "station_code": STATION,
                    "site": SITE,
                    "year": YEAR,
                    "branch": branch,
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": scalar(wrow.get("rain"), np.nan),
                    "srad": scalar(wrow.get("srad"), np.nan),
                    "tmax": scalar(wrow.get("tmax"), np.nan),
                    "tmin": scalar(wrow.get("tmin"), np.nan),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "topwt": scalar(latest.get("topwt")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "safe_action_amir": float(i),
                    "safe_action_anfer": float(n),
                    "reward": float(reward),
                    **dict(getattr(env, "last_action_info", {})),
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1

        daily = pd.DataFrame(records)
        daily_path = DAILY_DIR / f"{TASK_ID}_SYA_{YEAR}_{branch}_daily.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").ffill().iloc[-1])
        total_i = float(pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0).sum())
        total_n = float(pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0).sum())
        snapshot_src = siteppo.snapshot_from_env(env)
        snapshot = SNAP_DIR / branch
        if snapshot.exists():
            shutil.rmtree(snapshot)
        shutil.copytree(snapshot_src, snapshot)
        metrics = siteppo.strict_metrics_from_snapshot(snapshot, final_y, total_i, total_n)
        summary = {
            "station_code": STATION,
            "site": SITE,
            "year": YEAR,
            "branch": branch,
            "grain_yield_kg_ha": final_y,
            "biomass_kg_ha": float(pd.to_numeric(daily["topwt"], errors="coerce").ffill().iloc[-1]),
            "actual_irrigation_mm": total_i,
            "actual_nitrogen_kg_ha": total_n,
            "requested_irrigation_mm": sum(i for i, _n in schedule.values()),
            "requested_nitrogen_kg_ha": sum(n for _i, n in schedule.values()),
            "irrigation_event_count": sum(1 for i, _n in schedule.values() if i > 0),
            "nitrogen_event_count": sum(1 for _i, n in schedule.values() if n > 0),
            "action_sequence": action_sequence(schedule),
            "daily_csv_path": daily_path.relative_to(ROOT).as_posix(),
            "snapshot_path": snapshot.relative_to(ROOT).as_posix(),
            **metrics,
            **stress_summary(daily),
        }
        return daily, summary
    finally:
        if env is not None:
            env.close()
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input
        direct_ppo.OUTPUT_ROOT = old_out


def add_delta_columns(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    base = out[out["branch"].eq("baseline_04212_full")].iloc[0]
    cand = out[out["branch"].eq("candidate_04216_full")].iloc[0]
    for ref_name, ref in [("vs_04212", base), ("vs_04216", cand)]:
        out[f"delta_yield_{ref_name}"] = pd.to_numeric(out["grain_yield_kg_ha"], errors="coerce") - float(ref["grain_yield_kg_ha"])
        out[f"delta_swfac_days_gt_0p05_{ref_name}"] = pd.to_numeric(out["swfac_days_gt_0p05"], errors="coerce") - int(ref["swfac_days_gt_0p05"])
        out[f"delta_nstres_days_gt_0p05_{ref_name}"] = pd.to_numeric(out["nstres_days_gt_0p05"], errors="coerce") - int(ref["nstres_days_gt_0p05"])
        out[f"delta_irrigation_{ref_name}"] = pd.to_numeric(out["actual_irrigation_mm"], errors="coerce") - float(ref["actual_irrigation_mm"])
        out[f"delta_n_{ref_name}"] = pd.to_numeric(out["actual_nitrogen_kg_ha"], errors="coerce") - float(ref["actual_nitrogen_kg_ha"])
    return out


def write_record(plan: pd.DataFrame, summary: pd.DataFrame | None) -> None:
    lines = [
        "# 042_17 SYA lowIC 2017：042_12 vs 042_16 灌溉差异反事实审计记录",
        "",
        "## 任务性质",
        "",
        "- 不训练 PPO，不改 reward，不改输入数据。",
        "- 只把 042_12 和 042_16 的 2017 固定动作 schedule 拿出来做 DSSAT 反事实重放。",
        "- 目的：定位 042_16 改善 2017 的原因到底主要来自灌溉、施氮，还是两者共同作用。",
        "",
        "## 动作计划",
        "",
        md_table(plan, 80),
    ]
    if summary is not None and not summary.empty:
        key_cols = [
            "branch",
            "grain_yield_kg_ha",
            "actual_irrigation_mm",
            "actual_nitrogen_kg_ha",
            "WP_ET_kg_m3",
            "PFP_N_kg_kg",
            "swfac_days_gt_0p05",
            "nstres_days_gt_0p05",
            "delta_yield_vs_04212",
            "delta_yield_vs_04216",
            "delta_swfac_days_gt_0p05_vs_04212",
            "delta_swfac_days_gt_0p05_vs_04216",
        ]
        show = summary[[c for c in key_cols if c in summary.columns]]
        lines.extend(["", "## 反事实结果", "", md_table(show, 80)])
        lines.extend(["", "## 初步判读", ""])
        water = summary[summary["branch"].eq("04216_irrigation_04212_nitrogen")]
        n = summary[summary["branch"].eq("04212_irrigation_04216_nitrogen")]
        full = summary[summary["branch"].eq("candidate_04216_full")]
        if not water.empty and not n.empty and not full.empty:
            lines.append(
                f"- 只替换灌溉分支相对 042_12 的产量变化：{float(water.iloc[0]['delta_yield_vs_04212']):.2f} kg/ha。"
            )
            lines.append(
                f"- 只替换施氮分支相对 042_12 的产量变化：{float(n.iloc[0]['delta_yield_vs_04212']):.2f} kg/ha。"
            )
            lines.append(
                f"- 完整 042_16 相对 042_12 的产量变化：{float(full.iloc[0]['delta_yield_vs_04212']):.2f} kg/ha。"
            )
        lines.extend(
            [
                "",
                "## 边界",
                "",
                "- 本任务只解释 2017 年机制，不自动推广到所有年份。",
                "- 若要把发现写进下一轮训练约束或 reward，必须另开任务预注册。",
            ]
        )
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dry() -> None:
    ensure_dirs()
    plan, _schedules = build_schedules()
    plan.to_csv(TAB_DIR / f"{TASK_ID}_schedule_plan_dry_run.csv", index=False, encoding="utf-8-sig")
    write_record(plan, None)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "mode": "dry_run",
        "source_daily_04212_exists": DAILY_04212.exists(),
        "source_daily_04216_exists": DAILY_04216.exists(),
        "lowIC_input_root_exists": LOWIC_INPUT_ROOT.exists(),
        "branch_count": int(len(plan)),
        "plan_csv": (TAB_DIR / f"{TASK_ID}_schedule_plan_dry_run.csv").relative_to(ROOT).as_posix(),
        "record_md": DOC.relative_to(ROOT).as_posix(),
        "next_step_allowed": bool(DAILY_04212.exists() and DAILY_04216.exists() and LOWIC_INPUT_ROOT.exists()),
    }
    (OUT / f"{TASK_ID}_dry_run_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_full() -> None:
    ensure_dirs()
    if not PROMPT.exists():
        raise FileNotFoundError(PROMPT)
    if not DAILY_04212.exists() or not DAILY_04216.exists():
        raise FileNotFoundError("Missing 042_12 or 042_16 daily CSV for 2017.")
    if not LOWIC_INPUT_ROOT.exists():
        raise FileNotFoundError(LOWIC_INPUT_ROOT)

    config, env_config, selection = load_config_and_env_config()
    direct_ppo.write_yaml(env_config, OUT / "configs" / f"{TASK_ID}_resolved_env_config.yaml")
    selection.to_csv(OUT / "configs" / f"{TASK_ID}_selection.csv", index=False, encoding="utf-8-sig")

    plan, schedules = build_schedules()
    plan.to_csv(TAB_DIR / f"{TASK_ID}_schedule_plan.csv", index=False, encoding="utf-8-sig")
    daily_frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for branch, schedule in schedules.items():
        daily, summary = replay_schedule(config, env_config, branch, schedule)
        daily_frames.append(daily)
        rows.append(summary)

    summary = add_delta_columns(pd.DataFrame(rows))
    summary.to_csv(TAB_DIR / f"{TASK_ID}_counterfactual_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily_frames, ignore_index=True).to_csv(TAB_DIR / f"{TASK_ID}_counterfactual_daily.csv", index=False, encoding="utf-8-sig")
    write_record(plan, summary)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": "completed",
        "year": YEAR,
        "branch_count": int(len(summary)),
        "summary_csv": (TAB_DIR / f"{TASK_ID}_counterfactual_summary.csv").relative_to(ROOT).as_posix(),
        "daily_csv": (TAB_DIR / f"{TASK_ID}_counterfactual_daily.csv").relative_to(ROOT).as_posix(),
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / f"{TASK_ID}_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        run_dry()
    else:
        run_full()


if __name__ == "__main__":
    main()

