#!/usr/bin/env python3
"""Run only the six official-expert seasons missing after 028_06."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_extension_expert_baseline_018_03 as expert
from run_fq_yc_new_cultivar_forward_screening_013_01 import (
    INPUT_ROOT as MULTISITE_INPUT_ROOT,
    SITE_CONFIG,
    set_management_for_treatment,
)


OUT = ROOT / "benchmark_results" / "028_07_missing_official_expert_six_season_completion"
DOC = ROOT / "docs" / "2026-07-18_028_07_missing_official_expert_six_season_completion.md"
CASES = [
    {"site": "YC", "station": "Yucheng", "year": 2008, "region": "huanghuai_fenwei_summer_maize"},
    *(
        {"site": "FQ", "station": "Fengqiu", "year": year, "region": "huanghuai_fenwei_summer_maize"}
        for year in (2013, 2014, 2019, 2020, 2023)
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def copy_aux(input_root: Path, input_dir: Path, exclude: str) -> None:
    for source in input_root.iterdir():
        if source.is_file() and source.name != exclude:
            shutil.copyfile(source, input_dir / source.name)


def prepare_yc2008(case: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    cfg = SITE_CONFIG["YC"]
    trno = int(cfg["treatments"][2008])
    source_root = MULTISITE_INPUT_ROOT / "YC"
    source = (source_root / cfg["mzx"]).read_text(encoding="latin-1", errors="ignore")
    # External management is the only active management path for this new copy.
    text = set_management_for_treatment(source, trno, "L", "L")
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=False)
    filex = input_dir / "CNYC2008_official_extension_expert_028_07.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")
    copy_aux(source_root, input_dir, cfg["mzx"])
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": trno,
        "auxiliary_file_paths": expert.copy_aux_from_input(input_dir),
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, ensure_ascii=False, indent=2), encoding="utf-8")
    return env_args


def prepare_fq(case: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    return expert.prepare_fq(case, run_dir)


def prepare_case(case: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    return prepare_yc2008(case, run_dir) if case["site"] == "YC" else prepare_fq(case, run_dir)


def smoke_row(case: dict[str, Any], run_dir: Path, env_args: dict[str, Any]) -> dict[str, Any]:
    filex = Path(env_args["fileX_template_path"])
    text = filex.read_text(encoding="latin-1", errors="ignore")
    weather_token = f"CN{case['site']}{case['year'] % 100:02d}01"
    linked_lines = [line for line in text.splitlines() if line.strip().startswith(str(env_args["experiment_number"]))]
    return {
        "site": case["site"],
        "year": case["year"],
        "experiment_number": env_args["experiment_number"],
        "filex": str(filex.relative_to(ROOT)).replace("\\", "/"),
        "filex_sha256": sha256(filex),
        "weather_token": weather_token,
        "weather_token_present": weather_token in text,
        "linked_management_marker_present": any(" L " in f" {line} " for line in linked_lines),
        "status": "pass" if weather_token in text else "fail_weather_reference",
    }


def build_schedule() -> pd.DataFrame:
    region = expert.build_region_schedule()
    chunks = []
    for case in CASES:
        frame = region[region["region"].eq(case["region"])].copy()
        frame.insert(0, "site", case["site"])
        frame.insert(1, "station", case["station"])
        frame.insert(2, "year", case["year"])
        chunks.append(frame)
    return pd.concat(chunks, ignore_index=True)


def write_doc(smoke: pd.DataFrame, summaries: pd.DataFrame, mode: str) -> None:
    ok = int((summaries.get("status", pd.Series(dtype=str)) == "ok").sum()) if not summaries.empty else 0
    lines = [
        "# 028_07 缺失 official expert 六季补齐记录",
        "",
        f"状态：`{'completed' if mode in {'run', 'reconcile'} and ok == 6 else 'smoke_completed' if mode == 'smoke' else 'partial'}`",
        "",
        "## 范围与纪律",
        "",
        "- 只包含 YC2008 与 FQ2013/2014/2019/2020/2023 的 official expert。",
        "- null、recorded、DSSAT auto 复用 028_06 已验证快照，没有重跑。",
        "- RL 训练 0 次。",
        "- 使用 018_03 已冻结华北黄淮/汾渭夏玉米固定 DAP 调度。",
        "",
        "## Input smoke",
        "",
        "|site|year|experiment|weather token|weather check|status|",
        "|---|---:|---:|---|---|---|",
    ]
    for _, row in smoke.iterrows():
        lines.append(f"|{row.site}|{int(row.year)}|{int(row.experiment_number)}|{row.weather_token}|{row.weather_token_present}|{row.status}|")
    lines += ["", "## DSSAT 结果", "", "|site|year|status|yield kg/ha|irrigation mm|N kg/ha|error|", "|---|---:|---|---:|---:|---:|---|"]
    for _, row in summaries.iterrows():
        lines.append(f"|{row.get('site','')}|{row.get('year','')}|{row.get('status','')}|{row.get('final_gwad','')}|{row.get('event_irrigation_total','')}|{row.get('event_fertilizer_total','')}|{row.get('error','')}|")
    lines += [
        "",
        "## 失败与边界",
        "",
        "- 输入 smoke 不通过时不得启动 DSSAT。",
        "- 第一次结果检查错误地把全季计划总量与实际执行总量作严格浮点相等比较；FQ2013/2023 在 DAP100 前终止，且 MgmtEvent.OUT 有输出舍入。该错误判定已保留，修正只重算检查结果，没有重跑 DSSAT。",
        "- 该任务只补齐比较基线，不构成 RL 成功或失败判定。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reconcile_existing() -> None:
    summary_path = OUT / "028_07_official_expert_summary.csv"
    actions_path = OUT / "028_07_official_expert_actions.csv"
    smoke_path = OUT / "028_07_run_input_audit.csv"
    if not all(path.is_file() for path in (summary_path, actions_path, smoke_path)):
        raise FileNotFoundError("Existing run outputs required for reconciliation are incomplete")
    summary = pd.read_csv(summary_path)
    actions = pd.read_csv(actions_path)
    failed_copy = OUT / "failed_attempt_1_schedule_checker_summary.csv"
    if not failed_copy.exists():
        shutil.copyfile(summary_path, failed_copy)
    for idx, row in summary.iterrows():
        subset = actions[(actions["site"] == row["site"]) & (actions["year"] == int(row["year"]))]
        executed_i = float(subset["irrigation_mm_action"].sum())
        executed_n = float(subset["fertilizer_kg_ha_action"].sum())
        event_i = float(row["event_irrigation_total"])
        event_n = float(row["event_fertilizer_total"])
        # MgmtEvent formats water to 0.1 mm and fertilizer to whole kg/ha.
        event_match = abs(event_i - executed_i) <= 0.051 and abs(event_n - executed_n) <= 0.501
        full_i = float(row["expected_irrigation_total"])
        full_n = float(row["expected_fertilizer_total"])
        summary.loc[idx, "executed_irrigation_total"] = executed_i
        summary.loc[idx, "executed_fertilizer_total"] = executed_n
        summary.loc[idx, "event_matches_executed_actions"] = event_match
        summary.loc[idx, "unfired_planned_irrigation_after_termination"] = full_i - executed_i
        summary.loc[idx, "unfired_planned_fertilizer_after_termination"] = full_n - executed_n
        summary.loc[idx, "schedule_total_match"] = event_match
        summary.loc[idx, "status"] = "ok" if event_match else "failed_event_action_mismatch"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    smoke = pd.read_csv(smoke_path)
    write_doc(smoke, summary, "reconcile")
    print(f"028_07 reconciliation completed: {(summary['status'] == 'ok').sum()}/{len(summary)} cases ok; DSSAT reruns=0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "run", "reconcile"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "reconcile":
        reconcile_existing()
        return
    schedule = build_schedule()
    schedule.to_csv(OUT / "028_07_official_expert_schedule.csv", index=False, encoding="utf-8-sig")

    smoke_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    action_frames: list[pd.DataFrame] = []
    for case in CASES:
        run_dir = OUT / "runs" / f"{case['site']}{case['year']}" / "official_extension_expert"
        if args.mode == "run" and run_dir.exists():
            # A successful smoke directory is immutable evidence; copy it into a run-specific directory.
            run_dir = OUT / "runs_completed" / f"{case['site']}{case['year']}" / "official_extension_expert"
        env_args = prepare_case(case, run_dir)
        smoke = smoke_row(case, run_dir, env_args)
        smoke_rows.append(smoke)
        if smoke["status"] != "pass":
            raise RuntimeError(f"Input smoke failed: {smoke}")
        if args.mode == "run":
            case_schedule = schedule[(schedule["site"] == case["site"]) & (schedule["year"] == case["year"])]
            try:
                daily, actions, summary = expert.run_fixed_schedule(case, env_args, case_schedule, run_dir)
                expected_actions = expert.split_irrigation_events(case_schedule)
                expected_i = sum(float(v["amir"]) for v in expected_actions.values())
                expected_n = sum(float(v["anfer"]) for v in expected_actions.values())
                summary["expected_irrigation_total"] = expected_i
                summary["expected_fertilizer_total"] = expected_n
                # Validate the events actually fired before termination. DSSAT rounds
                # MgmtEvent water to 0.1 mm and fertilizer to whole kg/ha.
                action_i = float(daily["irrigation_mm_action"].sum())
                action_n = float(daily["fertilizer_kg_ha_action"].sum())
                summary["executed_irrigation_total"] = action_i
                summary["executed_fertilizer_total"] = action_n
                summary["unfired_planned_irrigation_after_termination"] = expected_i - action_i
                summary["unfired_planned_fertilizer_after_termination"] = expected_n - action_n
                summary["event_matches_executed_actions"] = abs(summary["event_irrigation_total"] - action_i) <= 0.051 and abs(summary["event_fertilizer_total"] - action_n) <= 0.501
                summary["schedule_total_match"] = summary["event_matches_executed_actions"]
                if not summary["schedule_total_match"]:
                    summary["status"] = "failed_schedule_total_mismatch"
                daily_frames.append(daily)
                action_frames.append(actions)
                summaries.append(summary)
                print(f"OK {case['site']}{case['year']} GWAD={summary['final_gwad']} I={summary['event_irrigation_total']} N={summary['event_fertilizer_total']}")
            except Exception as exc:
                summaries.append({"site": case["site"], "year": case["year"], "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                raise

    smoke_df = pd.DataFrame(smoke_rows)
    summary_df = pd.DataFrame(summaries)
    smoke_df.to_csv(OUT / f"028_07_{args.mode}_input_audit.csv", index=False, encoding="utf-8-sig")
    if args.mode == "run":
        summary_df.to_csv(OUT / "028_07_official_expert_summary.csv", index=False, encoding="utf-8-sig")
        pd.concat(daily_frames, ignore_index=True).to_csv(OUT / "028_07_official_expert_daily.csv", index=False, encoding="utf-8-sig")
        pd.concat(action_frames, ignore_index=True).to_csv(OUT / "028_07_official_expert_actions.csv", index=False, encoding="utf-8-sig")
    write_doc(smoke_df, summary_df, args.mode)
    print(f"028_07 {args.mode} completed: {len(smoke_df)} input cases, {len(summary_df)} DSSAT seasons")


if __name__ == "__main__":
    main()
