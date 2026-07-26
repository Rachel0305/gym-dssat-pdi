from __future__ import annotations

import json
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

import audit_ppo_action_dssat_effect_034_01 as audit03401
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch_ppo


TASK_ID = "034_03"
OUT = ROOT / "benchmark_results" / "034_03_dynamic_rl_linked_management_fix_smoke"
DOC = ROOT / "docs" / "034_03_dynamic_rl_linked_management_fix_smoke_record.md"
PROMPT = ROOT / "prompts" / "034_03_dynamic_rl_linked_management_fix_smoke.md"
SOURCE_SPLIT = ROOT / "benchmark_results" / "033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun" / "configs" / "033_04_available_weather_half_split_years.csv"


def ensure_dirs() -> None:
    for rel in ["configs", "evaluation", "snapshots", "logs", "rendered_inputs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


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


def build_config() -> tuple[dict[str, Any], dict[str, Any]]:
    config = batch_ppo.load_config()
    config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    split = pd.read_csv(SOURCE_SPLIT, keep_default_na=False)
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    selection = batch_ppo.build_selection(split)
    direct_ppo.OUTPUT_ROOT = OUT
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "034_03_resolved_env_config.yaml")
    shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)
    return config, env_config


def overview_modes(snapshot: Path) -> str:
    path = snapshot / "OVERVIEW.OUT"
    if not path.exists():
        return ""
    lines = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "MANAGEMENT OPT" in line:
            lines.append(line.strip())
    return " | ".join(lines[:3])


def run_case(config: dict[str, Any], env_config: dict[str, Any], case: pd.Series, scenario: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    daily, summary, _render = audit03401.run_external_case(config, env_config, case, scenario=scenario)
    station = str(case["station_code"])
    year = int(case["year"])
    target_snapshot = OUT / "snapshots" / station / str(year) / scenario
    original_snapshot = ROOT / summary["snapshot_path"]
    if original_snapshot.resolve() != target_snapshot.resolve():
        if target_snapshot.exists():
            shutil.rmtree(target_snapshot)
        shutil.copytree(original_snapshot, target_snapshot)
    summary["snapshot_path"] = str(target_snapshot.relative_to(ROOT)).replace("\\", "/")
    summary["overview_management_opt"] = overview_modes(target_snapshot)
    return daily, summary


def compare(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (station, year, checkpoint), group in summary.groupby(["station_code", "year", "checkpoint_step"]):
        wide = {row["scenario"]: row for _, row in group.iterrows()}
        null = wide.get("null_external_noop")
        ppo = wide.get("ppo_external_replay")
        if null is None or ppo is None:
            continue
        safe_i = float(ppo["safe_action_irrigation_sum_mm"])
        safe_n = float(ppo["safe_action_n_sum_kg_ha"])
        sum_i = float(ppo["summary_irrigation_mm"])
        sum_n = float(ppo["summary_n_kg_ha"])
        null_zero = abs(float(null["summary_irrigation_mm"])) < 1e-6 and abs(float(null["summary_n_kg_ha"])) < 1e-6
        ppo_matches = abs(safe_i - sum_i) < 1e-6 and abs(safe_n - sum_n) < 1e-6
        linked = "IRRIG   :L" in str(ppo.get("overview_management_opt", "")) and "FERT :L" in str(ppo.get("overview_management_opt", ""))
        rows.append(
            {
                "station_code": station,
                "year": int(year),
                "checkpoint_step": int(checkpoint),
                "ppo_safe_i": safe_i,
                "ppo_safe_n": safe_n,
                "ppo_summary_i": sum_i,
                "ppo_summary_n": sum_n,
                "null_summary_i": float(null["summary_irrigation_mm"]),
                "null_summary_n": float(null["summary_n_kg_ha"]),
                "null_zero": null_zero,
                "ppo_summary_matches_safe": ppo_matches,
                "overview_is_linked": linked,
                "pass": bool(null_zero and ppo_matches and linked and (safe_i + safe_n > 0)),
            }
        )
    return pd.DataFrame(rows)


def write_record(summary: pd.DataFrame, comp: pd.DataFrame, failures: pd.DataFrame, elapsed: float) -> None:
    pass_count = int(comp["pass"].sum()) if not comp.empty and "pass" in comp else 0
    total = int(len(comp))
    lines = [
        "# 034_03 动态 RL linked management 修复 smoke 记录",
        "",
        "## 结论先说",
        "",
        f"- 通过数：{pass_count}/{total}。",
        "- 本任务不训练，只验证修复后动态 PPO 外部动作是否真正进入 DSSAT。",
        "",
        "## Summary",
        "",
        md_table(summary[[
            "station_code",
            "year",
            "checkpoint_step",
            "scenario",
            "safe_action_irrigation_sum_mm",
            "safe_action_n_sum_kg_ha",
            "summary_irrigation_mm",
            "summary_n_kg_ha",
            "grain_yield_kg_ha",
            "overview_management_opt",
        ]] if not summary.empty else summary, 40),
        "",
        "## 判定表",
        "",
        md_table(comp, 40),
        "",
        "## 失败记录",
        "",
        md_table(failures, 20),
        "",
        "## 输出文件",
        "",
        "- `benchmark_results/034_03_dynamic_rl_linked_management_fix_smoke/evaluation/034_03_summary.csv`",
        "- `benchmark_results/034_03_dynamic_rl_linked_management_fix_smoke/evaluation/034_03_comparison.csv`",
        "",
        f"耗时：{elapsed:.1f} 秒",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    ensure_dirs()
    audit03401.OUT = OUT
    audit03401.TASK_ID = TASK_ID
    config, env_config = build_config()
    cases = audit03401.select_cases()
    cases.to_csv(OUT / "evaluation" / "034_03_case_selection.csv", index=False, encoding="utf-8-sig")
    summaries = []
    daily_frames = []
    failures = []
    for _, case in cases.iterrows():
        for scenario in ["null_external_noop", "ppo_external_replay"]:
            try:
                daily, summary = run_case(config, env_config, case, scenario)
                daily_frames.append(daily)
                summaries.append(summary)
            except Exception:
                failures.append(
                    {
                        "station_code": str(case["station_code"]),
                        "year": int(case["year"]),
                        "checkpoint_step": int(case["checkpoint_step"]),
                        "scenario": scenario,
                        "traceback": traceback.format_exc()[-5000:],
                    }
                )
    summary = pd.DataFrame(summaries)
    daily = pd.concat(daily_frames, ignore_index=True, sort=False) if daily_frames else pd.DataFrame()
    failures_df = pd.DataFrame(failures)
    comp = compare(summary) if not summary.empty else pd.DataFrame()
    summary.to_csv(OUT / "evaluation" / "034_03_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT / "evaluation" / "034_03_daily.csv", index=False, encoding="utf-8-sig")
    failures_df.to_csv(OUT / "evaluation" / "034_03_failures.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(OUT / "evaluation" / "034_03_comparison.csv", index=False, encoding="utf-8-sig")
    write_record(summary, comp, failures_df, time.time() - start)
    result = {
        "task": TASK_ID,
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "pass_count": int(comp["pass"].sum()) if not comp.empty and "pass" in comp else 0,
        "total": int(len(comp)),
        "failures": int(len(failures_df)),
    }
    (OUT / "034_03_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not comp.empty:
        print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
