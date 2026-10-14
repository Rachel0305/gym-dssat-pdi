from __future__ import annotations

import argparse
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
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline034
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo


TASK = "037_07_static_level1_four_baseline_rebuild"
OUT = ROOT / "benchmark_results" / TASK
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

STATIONS = baseline034.STATIONS
STATION_TO_SITE = baseline034.STATION_TO_SITE
SCENARIOS = baseline034.SCENARIOS
SEED = baseline034.SEED
MAX_STEPS = baseline034.MAX_STEPS
IRR_TOL = 0.6
N_TOL = 1.2


def ensure_dirs() -> None:
    for rel in ["configs", "evaluation", "snapshots", "logs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
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
    lines = [
        "| " + " | ".join(map(str, work.columns)) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for row in work.to_numpy().tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def yyddd_from_dap(planting_date: str, dap: int) -> str:
    planting = pd.Timestamp(planting_date)
    date = planting + pd.Timedelta(days=max(int(dap) - 1, 0))
    return f"{date.year % 100:02d}{date.dayofyear:03d}"


def replace_static_application_rows_level1(text: str, schedule: dict[int, dict[str, float]], planting_date: str, year: int) -> str:
    """Render static DSSAT application rows with one selected management level.

    The first numeric column in @I IDATE and @F FDATE rows is the management
    level ID, not an event sequence number. All events for one selected
    scenario therefore need the same leading level ID (=1). This is the
    central correction relative to 034_00.
    """
    irrigation_rows: list[str] = []
    fertilizer_rows: list[str] = []
    for dap, action in sorted(schedule.items()):
        date = yyddd_from_dap(planting_date, int(dap))
        irrigation = float(action.get("amir", 0.0))
        nitrogen = float(action.get("anfer", 0.0))
        if irrigation > 1e-9:
            irrigation_rows.append(f" 1 {date} IR001 {irrigation:5.1f}")
        if nitrogen > 1e-9:
            fertilizer_rows.append(f" 1 {date} FE005 AP002     5 {nitrogen:6.1f}   -99   -99   -99   -99   -99 {year}")
    if not irrigation_rows:
        irrigation_rows = [f" 1 {yyddd_from_dap(planting_date, 1)} IR001     0"]
    if not fertilizer_rows:
        fertilizer_rows = [f" 1 {yyddd_from_dap(planting_date, 1)} FE005 AP002     5     0   -99   -99   -99   -99   -99 {year}"]

    lines = text.splitlines()
    output: list[str] = []
    mode: str | None = None
    inserted_i = False
    inserted_f = False
    for line in lines:
        stripped = line.strip()
        if line.startswith("@I IDATE"):
            output.append(line)
            output.extend(irrigation_rows)
            mode = "I"
            inserted_i = True
            continue
        if line.startswith("@F FDATE"):
            output.append(line)
            output.extend(fertilizer_rows)
            mode = "F"
            inserted_f = True
            continue
        if mode is not None:
            if stripped.startswith("@") or stripped.startswith("*"):
                mode = None
            elif stripped == "":
                continue
            elif re.match(r"^\s*\d+\s+\d{5}\b", line):
                continue
        output.append(line)
    if not inserted_i:
        raise ValueError("渲染模板中缺少 @I IDATE 行，无法写入静态灌溉表")
    if not inserted_f:
        raise ValueError("渲染模板中缺少 @F FDATE 行，无法写入静态施肥表")
    return "\n".join(output) + "\n"


def configure_baseline034_globals() -> None:
    baseline034.TASK_ID = "037_07"
    baseline034.OUT = OUT
    baseline034.DOC = DOC
    baseline034.PROMPT = PROMPT
    baseline034.replace_static_application_rows = replace_static_application_rows_level1  # type: ignore[assignment]


def load_run_config(selected: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    run_config = direct_ppo.load_yaml(baseline034.BASE_CONFIG)
    run_config = json.loads(json.dumps(run_config))
    run_config["seed"] = SEED
    run_config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    run_config["runtime"]["max_steps"] = MAX_STEPS
    env_config = baseline034.build_env_config(run_config, selected)
    return run_config, env_config


def is_data_line(line: str) -> bool:
    return bool(re.match(r"^\s*\d+\s+", line))


def to_float(text: str) -> float:
    try:
        return float(text)
    except Exception:
        return 0.0


def parse_inp_events(path: Path) -> dict[str, float | int]:
    lines = path.read_text(errors="ignore").splitlines()
    section = ""
    i_amounts: list[float] = []
    n_amounts: list[float] = []
    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("*IRRIGATION"):
            section = "I"
            continue
        if upper.startswith("*FERTILIZERS"):
            section = "F"
            continue
        if stripped.startswith("*"):
            section = ""
            continue
        if not section or stripped.startswith("@") or not is_data_line(line):
            continue
        parts = stripped.split()
        if section == "I" and len(parts) >= 3:
            amt = to_float(parts[2])
            if amt > 1e-9:
                i_amounts.append(amt)
        if section == "F" and len(parts) >= 5:
            amt = to_float(parts[4])
            if amt > 1e-9:
                n_amounts.append(amt)
    return {
        "inp_i_events": len(i_amounts),
        "inp_i_total": sum(i_amounts),
        "inp_n_events": len(n_amounts),
        "inp_n_total": sum(n_amounts),
    }


def parse_mgmt_events(snapshot: Path) -> dict[str, float | int]:
    path = snapshot / "MgmtEvent.OUT"
    i_events: set[tuple[str, float]] = set()
    n_events: set[tuple[str, float]] = set()
    if not path.exists():
        return {"mgmt_i_events": math.nan, "mgmt_i_total": math.nan, "mgmt_n_events": math.nan, "mgmt_n_total": math.nan}
    for line in path.read_text(errors="ignore").splitlines():
        if "Irrigation" in line:
            m = re.search(r"Irrigation\s+([-+]?\d+(?:\.\d+)?)", line)
            if m:
                amt = float(m.group(1))
                if amt > 1e-9:
                    i_events.add((line.split("Irrigation", 1)[0].strip(), round(amt, 6)))
        if "Fertilizer" in line:
            m = re.search(r"Fertilizer\s+([-+]?\d+(?:\.\d+)?)", line)
            if m:
                amt = float(m.group(1))
                if amt > 1e-9:
                    n_events.add((line.split("Fertilizer", 1)[0].strip(), round(amt, 6)))
    i_amounts = [v for _, v in sorted(i_events)]
    n_amounts = [v for _, v in sorted(n_events)]
    return {"mgmt_i_events": len(i_amounts), "mgmt_i_total": sum(i_amounts), "mgmt_n_events": len(n_amounts), "mgmt_n_total": sum(n_amounts)}


def executable_schedule(schedule: dict[int, dict[str, float]], last_dap: int) -> tuple[dict[int, dict[str, float]], list[int]]:
    exe = {int(d): a for d, a in schedule.items() if int(d) <= int(last_dap)}
    post = [int(d) for d in schedule if int(d) > int(last_dap)]
    return exe, sorted(post)


def planned_counts(schedule: dict[int, dict[str, float]]) -> dict[str, float | int]:
    i_vals = [float(a.get("amir", 0.0)) for a in schedule.values() if float(a.get("amir", 0.0)) > 1e-9]
    n_vals = [float(a.get("anfer", 0.0)) for a in schedule.values() if float(a.get("anfer", 0.0)) > 1e-9]
    return {"planned_i_events": len(i_vals), "planned_i_total": sum(i_vals), "planned_n_events": len(n_vals), "planned_n_total": sum(n_vals)}


def audit_row(summary: dict[str, Any], schedule: dict[int, dict[str, float]]) -> dict[str, Any]:
    snapshot = ROOT / str(summary["snapshot_path"])
    scenario = str(summary["scenario"])
    last_dap = int(summary["season_last_dap"]) if "season_last_dap" in summary else int(pd.to_numeric(summary.get("season_last_dap", 999), errors="coerce") or 999)
    exe, post = executable_schedule(schedule, last_dap)
    post_schedule = {int(d): a for d, a in schedule.items() if int(d) > int(last_dap)}
    post_counts = planned_counts(post_schedule)
    row: dict[str, Any] = {
        "station_code": summary["station_code"],
        "site": summary["site"],
        "year": summary["year"],
        "scenario": scenario,
        "season_last_dap": last_dap,
        "post_harvest_planned_daps": ";".join(map(str, post)),
        "snapshot_path": summary["snapshot_path"],
    }
    row.update(planned_counts(exe))
    row["post_harvest_i_events"] = post_counts["planned_i_events"]
    row["post_harvest_i_total"] = post_counts["planned_i_total"]
    row["post_harvest_n_events"] = post_counts["planned_n_events"]
    row["post_harvest_n_total"] = post_counts["planned_n_total"]
    row.update(parse_inp_events(snapshot / "DSSAT48.INP") if (snapshot / "DSSAT48.INP").exists() else {})
    row.update(parse_mgmt_events(snapshot))
    row["summary_i_total"] = float(summary.get("actual_irrigation_mm", math.nan))
    row["summary_n_total"] = float(summary.get("actual_nitrogen_kg_ha", math.nan))
    issues: list[str] = []
    planned_notes: list[str] = []
    if scenario in {"recorded_farmer_template", "official_extension_expert"}:
        i_event_gap = int(row.get("inp_i_events", 0)) - int(row.get("mgmt_i_events", -1))
        i_total_gap = float(row.get("inp_i_total", 0.0)) - float(row.get("mgmt_i_total", math.nan))
        n_event_gap = int(row.get("inp_n_events", 0)) - int(row.get("mgmt_n_events", -1))
        n_total_gap = float(row.get("inp_n_total", 0.0)) - float(row.get("mgmt_n_total", math.nan))
        i_gap_is_post_harvest = (
            i_event_gap == int(row.get("post_harvest_i_events", 0))
            and abs(i_total_gap - float(row.get("post_harvest_i_total", 0.0))) <= IRR_TOL
        )
        n_gap_is_post_harvest = (
            n_event_gap == int(row.get("post_harvest_n_events", 0))
            and abs(n_total_gap - float(row.get("post_harvest_n_total", 0.0))) <= N_TOL
        )
        if int(row.get("inp_i_events", 0)) != int(row.get("mgmt_i_events", -1)) and not i_gap_is_post_harvest:
            issues.append("irrigation_event_count_inp_vs_mgmt")
        if abs(float(row.get("inp_i_total", 0.0)) - float(row.get("mgmt_i_total", math.nan))) > IRR_TOL and not i_gap_is_post_harvest:
            issues.append("irrigation_total_inp_vs_mgmt")
        if abs(float(row.get("mgmt_i_total", 0.0)) - float(row.get("summary_i_total", math.nan))) > IRR_TOL:
            issues.append("irrigation_total_mgmt_vs_summary")
        if int(row.get("inp_n_events", 0)) != int(row.get("mgmt_n_events", -1)) and not n_gap_is_post_harvest:
            issues.append("nitrogen_event_count_inp_vs_mgmt")
        if abs(float(row.get("inp_n_total", 0.0)) - float(row.get("mgmt_n_total", math.nan))) > N_TOL and not n_gap_is_post_harvest:
            issues.append("nitrogen_total_inp_vs_mgmt")
        if abs(float(row.get("mgmt_n_total", 0.0)) - float(row.get("summary_n_total", math.nan))) > N_TOL:
            issues.append("nitrogen_total_mgmt_vs_summary")
        # Planned rows are source provenance, but DSSAT may round fertilizer N
        # by material code and may include post-harvest rows in DSSAT48.INP
        # that are not executed. Keep these as notes, not hard failures.
        if int(row.get("planned_i_events", 0)) != int(row.get("inp_i_events", -1)):
            planned_notes.append("irrigation_event_count_planned_vs_inp")
        if abs(float(row.get("planned_i_total", 0.0)) - float(row.get("inp_i_total", math.nan))) > IRR_TOL:
            planned_notes.append("irrigation_total_planned_vs_inp")
        if int(row.get("planned_n_events", 0)) != int(row.get("inp_n_events", -1)):
            planned_notes.append("nitrogen_event_count_planned_vs_inp")
        if abs(float(row.get("planned_n_total", 0.0)) - float(row.get("inp_n_total", math.nan))) > N_TOL:
            planned_notes.append("nitrogen_total_planned_vs_inp")
    row["status"] = "ok" if not issues else "event_chain_issue"
    row["issues"] = ";".join(issues)
    row["planned_vs_inp_notes"] = ";".join(planned_notes)
    return row


def run(mode: str) -> dict[str, Any]:
    configure_baseline034_globals()
    ensure_dirs()
    if baseline034.BASE_CONFIG.exists():
        shutil.copy2(baseline034.BASE_CONFIG, OUT / "configs" / baseline034.BASE_CONFIG.name)
    selected = baseline034.load_selection(mode)
    selected.to_csv(OUT / "configs" / f"037_07_{mode}_selected_station_years.csv", index=False, encoding="utf-8-sig")
    run_config, env_config = load_run_config(selected)
    direct_ppo.write_yaml(env_config, OUT / "configs" / f"037_07_{mode}_resolved_env_config.yaml")
    recorded_templates = baseline034.recorded_template_schedules()
    weather = direct_ppo.weather_for_daily(run_config)

    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    start = time.time()

    for row in selected.itertuples(index=False):
        station = str(row.station_code)
        year = int(row.year)
        site = STATION_TO_SITE[station]
        for scenario in SCENARIOS:
            print(f"[037_07] {mode} {station}{year} {scenario}", flush=True)
            try:
                daily, summary, _ = baseline034.evaluate_scenario(run_config, env_config, station, year, scenario, recorded_templates)
                # Add season_last_dap for audit and downstream filtering.
                summary["season_last_dap"] = int(pd.to_numeric(daily["dap"], errors="coerce").max())
                schedule = {} if scenario == "dssat_auto" else baseline034.schedule_for(scenario, site, recorded_templates)
                audit = audit_row(summary, schedule)
                daily_frames.append(daily)
                summary_rows.append(summary)
                audit_rows.append(audit)
                status = "ok" if audit["status"] == "ok" else "event_chain_issue"
                manifest_rows.append({"station_code": station, "site": site, "year": year, "scenario": scenario, "status": status, "details": audit.get("issues", "")})
                pd.DataFrame(summary_rows).to_csv(OUT / "evaluation" / f"037_07_{mode}_baseline_summary_partial.csv", index=False, encoding="utf-8-sig")
            except Exception as exc:
                manifest_rows.append({"station_code": station, "site": site, "year": year, "scenario": scenario, "status": "failed", "details": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})

    summary_df = pd.DataFrame(summary_rows)
    daily_df = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    audit_df = pd.DataFrame(audit_rows)
    manifest_df = pd.DataFrame(manifest_rows)
    failures_df = manifest_df[manifest_df["status"].ne("ok")].copy() if not manifest_df.empty else pd.DataFrame()

    summary_df.to_csv(OUT / "evaluation" / f"037_07_{mode}_baseline_summary.csv", index=False, encoding="utf-8-sig")
    daily_df.to_csv(OUT / "evaluation" / f"037_07_{mode}_baseline_daily.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(OUT / "evaluation" / f"037_07_{mode}_management_event_audit.csv", index=False, encoding="utf-8-sig")
    manifest_df.to_csv(OUT / "evaluation" / f"037_07_{mode}_coverage_manifest.csv", index=False, encoding="utf-8-sig")
    failures_df.to_csv(OUT / "evaluation" / f"037_07_{mode}_failures.csv", index=False, encoding="utf-8-sig")

    elapsed = time.time() - start
    coverage = manifest_df.groupby(["scenario", "status"]).size().reset_index(name="n") if not manifest_df.empty else pd.DataFrame()
    by_station = pd.DataFrame()
    if not summary_df.empty:
        tmp = summary_df.copy()
        for col in ["grain_yield_kg_ha", "actual_irrigation_mm", "actual_nitrogen_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]:
            if col in tmp:
                tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        by_station = tmp.groupby(["station_code", "scenario"]).agg(
            n=("year", "count"),
            mean_yield=("grain_yield_kg_ha", "mean"),
            mean_irrigation=("actual_irrigation_mm", "mean"),
            mean_nitrogen=("actual_nitrogen_kg_ha", "mean"),
            mean_wp_et=("WP_ET_kg_m3", "mean"),
            mean_pfp_n=("PFP_N_kg_kg", "mean"),
        ).reset_index()
    lines = [
        "# 037_07 静态 level-1 四基线重建记录",
        "",
        f"- 模式：`{mode}`",
        f"- 耗时：{elapsed:.1f} 秒",
        f"- 成功情景：{len(summary_df)} / {len(selected) * len(SCENARIOS)}",
        f"- 非 ok 记录：{len(failures_df)}",
        "",
        "## 本轮修复",
        "",
        "旧 034_00 把 `@I IDATE` / `@F FDATE` 行首编号误写成事件序号。DSSAT 将该列解释为 management level，因此只会把 treatment 选中的 level 1 事件写入 `DSSAT48.INP`。本轮将同一情景内所有静态应用事件统一写为 level 1，以复现多次 recorded/expert 管理事件。",
        "",
        "## 覆盖状态",
        "",
        md_table(coverage, 80),
        "",
        "## 非 ok 记录",
        "",
        md_table(failures_df[["station_code", "year", "scenario", "status", "details"]] if not failures_df.empty else failures_df, 120),
        "",
        "## 站点均值预览",
        "",
        md_table(by_station, 120),
        "",
        "## 输出文件",
        "",
        f"- summary: `benchmark_results/{TASK}/evaluation/037_07_{mode}_baseline_summary.csv`",
        f"- daily: `benchmark_results/{TASK}/evaluation/037_07_{mode}_baseline_daily.csv`",
        f"- audit: `benchmark_results/{TASK}/evaluation/037_07_{mode}_management_event_audit.csv`",
        f"- manifest: `benchmark_results/{TASK}/evaluation/037_07_{mode}_coverage_manifest.csv`",
        f"- failures: `benchmark_results/{TASK}/evaluation/037_07_{mode}_failures.csv`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {
        "task": TASK,
        "mode": mode,
        "selected_station_years": int(len(selected)),
        "expected_runs": int(len(selected) * len(SCENARIOS)),
        "successful_runs": int(len(summary_df)),
        "non_ok_runs": int(len(failures_df)),
        "elapsed_seconds": elapsed,
        "output_root": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / f"037_07_{mode}_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if not failures_df.empty:
        print(failures_df[["station_code", "year", "scenario", "status", "details"]].to_string(index=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
