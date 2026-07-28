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
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline034
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo
from ppo_action_safety import normalize_action


TASK = "037_06_lc2019_linked_four_baseline_smoke"
OUT = ROOT / "benchmark_results" / TASK
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"
BASE_CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"

STATION = "LCA"
SITE = "LC"
YEAR = 2019
SEED = 0
MAX_STEPS = 260
SCENARIOS = ["null", "recorded_farmer_template", "official_extension_expert", "dssat_auto"]
IRR_TOL = 0.5
N_TOL = 1.0


def ensure_dirs() -> None:
    for rel in ["configs", "evaluation", "snapshots", "logs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "无记录"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def build_lc2019_env_config() -> tuple[dict[str, Any], dict[str, Any]]:
    config = direct_ppo.load_yaml(BASE_CONFIG)
    config = json.loads(json.dumps(config))
    config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    config["runtime"]["max_steps"] = MAX_STEPS
    config["seed"] = SEED
    pool = pd.read_csv(POOL, keep_default_na=False)
    pool["year"] = pd.to_numeric(pool["year"], errors="coerce").astype(int)
    selected = pool[(pool["station_code"].astype(str).eq(STATION)) & (pool["year"].eq(YEAR))].copy()
    if len(selected) != 1:
        raise RuntimeError(f"需要唯一 {STATION}{YEAR} scenario_pool 行，实际 {len(selected)}")
    selected["selected_for_train"] = False
    selected["selected_for_eval"] = True
    selected["selection_reason"] = TASK
    env_config = direct_ppo.build_env_config(config, selected)
    env_config["paths"]["output_root"] = config["paths"]["output_root"]
    env_config["runtime"]["max_steps"] = MAX_STEPS
    direct_ppo.write_yaml(env_config, OUT / "configs" / "037_06_resolved_env_config.yaml")
    selected.to_csv(OUT / "configs" / "037_06_selected_lc2019.csv", index=False, encoding="utf-8-sig")
    return config, env_config


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def latest_obs(env: Any, obs: Any, info: Any | None = None) -> dict[str, Any]:
    return direct_ppo.latest_observation_dict(env, obs, info)


def exact_action(env: Any, amir: float, anfer: float) -> np.ndarray:
    return normalize_action(env.formator.action_names, env.formator.action_space_dict, {"amir": float(amir), "anfer": float(anfer)})


def zero_action(env: Any) -> np.ndarray:
    return exact_action(env, 0.0, 0.0)


def merged_schedule(schedule: dict[int, dict[str, float]]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for dap, action in sorted(schedule.items()):
        key = int(dap)
        out.setdefault(key, {"amir": 0.0, "anfer": 0.0})
        out[key]["amir"] += float(action.get("amir", 0.0))
        out[key]["anfer"] += float(action.get("anfer", 0.0))
    return out


def schedule_for_scenario(scenario: str) -> dict[int, dict[str, float]]:
    if scenario == "null" or scenario == "dssat_auto":
        return {}
    recorded = baseline034.recorded_template_schedules()
    return merged_schedule(baseline034.schedule_for(scenario, SITE, recorded))


def parse_mgmt_events(snapshot: Path) -> dict[str, Any]:
    path = snapshot / "MgmtEvent.OUT"
    i_events: set[tuple[str, float]] = set()
    n_events: set[tuple[str, float]] = set()
    if not path.exists():
        return {
            "mgmt_irrigation_event_count": math.nan,
            "mgmt_irrigation_total_mm": math.nan,
            "mgmt_n_event_count": math.nan,
            "mgmt_n_total_kg_ha": math.nan,
        }
    for line in path.read_text(errors="ignore").splitlines():
        if "Irrigation" in line:
            m = re.search(r"Irrigation\s+([-+]?\d+(?:\.\d+)?)", line)
            if m:
                amount = float(m.group(1))
                if amount > 1e-9:
                    i_events.add((line.split("Irrigation", 1)[0].strip(), round(amount, 6)))
        if "Fertilizer" in line:
            m = re.search(r"Fertilizer\s+([-+]?\d+(?:\.\d+)?)", line)
            if m:
                amount = float(m.group(1))
                if amount > 1e-9:
                    n_events.add((line.split("Fertilizer", 1)[0].strip(), round(amount, 6)))
    i_amounts = [v for _, v in sorted(i_events)]
    n_amounts = [v for _, v in sorted(n_events)]
    return {
        "mgmt_irrigation_event_count": len(i_amounts),
        "mgmt_irrigation_total_mm": sum(i_amounts),
        "mgmt_n_event_count": len(n_amounts),
        "mgmt_n_total_kg_ha": sum(n_amounts),
    }


def metrics_from_snapshot(snapshot: Path, final_yield: float) -> dict[str, Any]:
    return baseline034.metrics_from_snapshot(snapshot, final_yield)


def make_env(config: dict[str, Any], env_config: dict[str, Any], scenario: str):
    if scenario == "dssat_auto":
        return baseline034.make_base_env(
            env_config,
            STATION,
            YEAR,
            f"{STATION}_{YEAR}_{TASK}_{scenario}",
            auto_management=True,
            static_schedule=None,
        )[0]
    return direct_ppo.make_base_env(env_config, STATION, YEAR, SEED, f"{STATION}_{YEAR}_{TASK}_{scenario}", evaluation=True)


def run_scenario(config: dict[str, Any], env_config: dict[str, Any], scenario: str) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    schedule = schedule_for_scenario(scenario)
    env = make_env(config, env_config, scenario)
    weather = direct_ppo.weather_for_daily(config)
    rows: list[dict[str, Any]] = []
    sent_i = 0.0
    sent_n = 0.0
    applied_daps: set[int] = set()
    try:
        obs, info = env.reset()
        done = False
        step = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, STATION, YEAR)["planting_date"])
        while not done and step < MAX_STEPS:
            pre = latest_obs(env, obs, info)
            dap_raw = scalar(pre.get("dap", step + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step + 1
            planned = schedule.get(dap, {"amir": 0.0, "anfer": 0.0})
            if scenario in {"null", "dssat_auto"}:
                action = zero_action(env)
                send_i, send_n = 0.0, 0.0
            else:
                # Gym/DSSAT can expose duplicate early DAP rows. A fixed plan
                # should be applied once per DAP, not once per repeated row.
                if dap in applied_daps:
                    send_i, send_n = 0.0, 0.0
                else:
                    send_i = float(planned.get("amir", 0.0))
                    send_n = float(planned.get("anfer", 0.0))
                    if send_i > 1e-9 or send_n > 1e-9:
                        applied_daps.add(dap)
                action = exact_action(env, send_i, send_n)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = latest_obs(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            wrow = baseline034.weather_row(weather, STATION, date)
            sent_i += send_i
            sent_n += send_n
            rows.append(
                {
                    "station_code": STATION,
                    "site": SITE,
                    "year": YEAR,
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
                    "planned_irrigation_mm": float(planned.get("amir", 0.0)),
                    "planned_n_kg_ha": float(planned.get("anfer", 0.0)),
                    "sent_irrigation_mm": send_i,
                    "sent_n_kg_ha": send_n,
                    "reward": float(reward),
                    "done": done,
                }
            )
            step += 1
        daily = pd.DataFrame(rows)
        if daily.empty or not bool(daily["done"].iloc[-1]):
            raise RuntimeError(f"{scenario} 未在 {MAX_STEPS} 步内结束")
        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
        final_b = float(pd.to_numeric(daily["topwt"], errors="coerce").iloc[-1])
        snapshot_tmp = siteppo.snapshot_from_env(env)
        snapshot = OUT / "snapshots" / STATION / str(YEAR) / scenario
        if snapshot.exists():
            shutil.rmtree(snapshot)
        shutil.copytree(snapshot_tmp, snapshot)
        metrics = metrics_from_snapshot(snapshot, final_y)
        mgmt = parse_mgmt_events(snapshot)
        last_dap = int(pd.to_numeric(daily["dap"], errors="coerce").max())
        executable_schedule = {dap: action for dap, action in schedule.items() if int(dap) <= last_dap}
        post_harvest_schedule = {dap: action for dap, action in schedule.items() if int(dap) > last_dap}
        planned_i_events = sum(1 for action in executable_schedule.values() if float(action.get("amir", 0.0)) > 1e-9)
        planned_n_events = sum(1 for action in executable_schedule.values() if float(action.get("anfer", 0.0)) > 1e-9)
        summary = {
            "station_code": STATION,
            "site": SITE,
            "year": YEAR,
            "scenario": scenario,
            "grain_yield_kg_ha": final_y,
            "biomass_kg_ha": final_b,
            "planned_irrigation_event_count": planned_i_events,
            "season_last_dap": last_dap,
            "post_harvest_planned_daps": ",".join(map(str, sorted(post_harvest_schedule))),
            "planned_irrigation_total_mm": sum(float(a.get("amir", 0.0)) for a in executable_schedule.values()),
            "planned_n_event_count": planned_n_events,
            "planned_n_total_kg_ha": sum(float(a.get("anfer", 0.0)) for a in executable_schedule.values()),
            "sent_irrigation_total_mm": sent_i,
            "sent_n_total_kg_ha": sent_n,
            "summary_irrigation_mm": float(metrics["actual_irrigation_mm"]),
            "summary_n_kg_ha": float(metrics["actual_nitrogen_kg_ha"]),
            "max_water_stress": float(pd.to_numeric(daily["swfac"], errors="coerce").max()),
            "max_nitrogen_stress": float(pd.to_numeric(daily["nstres"], errors="coerce").max()),
            "WP_ET_kg_m3": float(metrics["WP_ET_kg_m3"]),
            "PFP_N_kg_kg": float(metrics["PFP_N_kg_kg"]) if not pd.isna(metrics["PFP_N_kg_kg"]) else math.nan,
            "snapshot_path": str(snapshot.relative_to(ROOT)).replace("\\", "/"),
            **mgmt,
        }
        audit = {
            "scenario": scenario,
            "manual_schedule": scenario not in {"null", "dssat_auto"},
            "planned_i_events": planned_i_events,
            "mgmt_i_events": mgmt["mgmt_irrigation_event_count"],
            "season_last_dap": last_dap,
            "post_harvest_planned_daps": summary["post_harvest_planned_daps"],
            "planned_i_total": summary["planned_irrigation_total_mm"],
            "sent_i_total": sent_i,
            "mgmt_i_total": mgmt["mgmt_irrigation_total_mm"],
            "summary_i_total": summary["summary_irrigation_mm"],
            "planned_n_events": planned_n_events,
            "mgmt_n_events": mgmt["mgmt_n_event_count"],
            "planned_n_total": summary["planned_n_total_kg_ha"],
            "sent_n_total": sent_n,
            "mgmt_n_total": mgmt["mgmt_n_total_kg_ha"],
            "summary_n_total": summary["summary_n_kg_ha"],
        }
        if scenario in {"null", "dssat_auto"}:
            audit["status"] = "ok_auto_or_null_recorded"
        else:
            ok = (
                planned_i_events == mgmt["mgmt_irrigation_event_count"]
                and planned_n_events == mgmt["mgmt_n_event_count"]
                and abs(summary["planned_irrigation_total_mm"] - mgmt["mgmt_irrigation_total_mm"]) <= IRR_TOL
                and abs(summary["planned_n_total_kg_ha"] - mgmt["mgmt_n_total_kg_ha"]) <= N_TOL
                and abs(sent_i - mgmt["mgmt_irrigation_total_mm"]) <= IRR_TOL
                and abs(sent_n - mgmt["mgmt_n_total_kg_ha"]) <= N_TOL
                and abs(float(metrics["actual_irrigation_mm"]) - mgmt["mgmt_irrigation_total_mm"]) <= IRR_TOL
                and abs(float(metrics["actual_nitrogen_kg_ha"]) - mgmt["mgmt_n_total_kg_ha"]) <= N_TOL
            )
            audit["status"] = "ok_linked_manual_baseline" if ok else "event_chain_issue"
        return daily, summary, audit
    finally:
        env.close()


def write_record(summary: pd.DataFrame, audit: pd.DataFrame, failures: pd.DataFrame, elapsed: float) -> None:
    lines = [
        "# 037_06：LC2019 linked-action 四基线可信烟测记录",
        "",
        "## 结论先说",
        "",
        f"- 成功情景数：{len(summary)} / {len(SCENARIOS)}",
        f"- 手工基线链路通过数：{int(audit['status'].eq('ok_linked_manual_baseline').sum()) if not audit.empty else 0} / 2",
        f"- 失败数：{len(failures)}",
        f"- 耗时：{elapsed:.1f} 秒",
        "",
        "本任务验证一种可信基线重建方式：手工基线不再依赖静态 `fileX.MZX` 管理表，而是通过 linked management step action 按计划日期精确发送水肥动作。",
        "",
        "## 管理事件链路审计",
        "",
        md_table(audit, 20),
        "",
        "## 四情景结果",
        "",
        md_table(
            summary[
                [
                    "scenario",
                    "grain_yield_kg_ha",
                    "summary_irrigation_mm",
                    "summary_n_kg_ha",
                    "WP_ET_kg_m3",
                    "PFP_N_kg_kg",
                    "max_water_stress",
                    "max_nitrogen_stress",
                    "planned_irrigation_event_count",
                    "planned_n_event_count",
                    "mgmt_irrigation_event_count",
                    "mgmt_n_event_count",
                    "snapshot_path",
                ]
            ]
            if not summary.empty
            else summary,
            20,
        ),
        "",
        "## 失败记录",
        "",
        md_table(failures, 20),
        "",
        "## 输出文件",
        "",
        "- summary：`benchmark_results/037_06_lc2019_linked_four_baseline_smoke/evaluation/037_06_lc2019_four_baseline_summary.csv`",
        "- daily：`benchmark_results/037_06_lc2019_linked_four_baseline_smoke/evaluation/037_06_lc2019_four_baseline_daily.csv`",
        "- audit：`benchmark_results/037_06_lc2019_linked_four_baseline_smoke/evaluation/037_06_lc2019_management_event_audit.csv`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    start = time.time()
    ensure_dirs()
    config, env_config = build_lc2019_env_config()
    if PROMPT.exists():
        shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)
    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        print(f"[037_06] {STATION}{YEAR} {scenario}", flush=True)
        try:
            daily, summary, audit = run_scenario(config, env_config, scenario)
            daily_frames.append(daily)
            summary_rows.append(summary)
            audit_rows.append(audit)
        except Exception as exc:
            failures.append({"scenario": scenario, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]})
    daily_all = pd.concat(daily_frames, ignore_index=True, sort=False) if daily_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    audit = pd.DataFrame(audit_rows)
    failures_df = pd.DataFrame(failures)
    daily_all.to_csv(OUT / "evaluation" / "037_06_lc2019_four_baseline_daily.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "evaluation" / "037_06_lc2019_four_baseline_summary.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(OUT / "evaluation" / "037_06_lc2019_management_event_audit.csv", index=False, encoding="utf-8-sig")
    failures_df.to_csv(OUT / "evaluation" / "037_06_lc2019_failures.csv", index=False, encoding="utf-8-sig")
    write_record(summary, audit, failures_df, time.time() - start)
    result = {
        "task": TASK,
        "successful_scenarios": int(len(summary)),
        "failed_scenarios": int(len(failures_df)),
        "manual_baseline_pass": int(audit["status"].eq("ok_linked_manual_baseline").sum()) if not audit.empty else 0,
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "summary_csv": str((OUT / "evaluation" / "037_06_lc2019_four_baseline_summary.csv").relative_to(ROOT)).replace("\\", "/"),
        "audit_csv": str((OUT / "evaluation" / "037_06_lc2019_management_event_audit.csv").relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "037_06_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not audit.empty:
        print(audit.to_string(index=False), flush=True)
    if not summary.empty:
        print(summary[["scenario", "grain_yield_kg_ha", "summary_irrigation_mm", "summary_n_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]].to_string(index=False), flush=True)
    if not failures_df.empty:
        print(failures_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
