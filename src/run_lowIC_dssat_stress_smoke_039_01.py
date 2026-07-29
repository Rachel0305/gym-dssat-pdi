"""039_01 DSSAT smoke for lowered initial soil water and mineral nitrogen.

Run original-IC and lowIC inputs for a small fixed site-year/scenario set to
verify whether the manually lowered initial conditions actually create stronger
DSSAT water/nitrogen stress signals.

This script does not train RL models and does not modify input data.  It only
temporarily redirects ppo_safe_rendering.MULTISITE_INPUT_ROOT during each
forward run.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline034
import run_static_level1_four_baseline_rebuild_037_07 as baseline037


TASK = "039_01_lowIC_dssat_stress_smoke"
OUT = ROOT / "benchmark_results" / TASK
PROMPT = ROOT / "prompts" / f"{TASK}.md"
DOC = OUT / f"{TASK}_record.md"
BASE_CONFIG = baseline034.BASE_CONFIG
ORIGINAL_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
LOWIC_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
SCENARIOS = ["null", "official_extension_expert"]
DEFAULT_STATION = "FQA"
DEFAULT_YEAR = 2014
MAX_STEPS = 260
SEED = 0


def ensure_dirs() -> None:
    for rel in ["configs", "tables", "figures", "snapshots", "logs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
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


def load_single_selection(station: str, year: int) -> pd.DataFrame:
    pool = pd.read_csv(baseline034.POOL, keep_default_na=False)
    pool["year"] = pd.to_numeric(pool["year"], errors="coerce").astype(int)
    row = pool[(pool["station_code"].astype(str).eq(station)) & (pool["year"].eq(int(year)))].copy()
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one scenario-pool row for {station}{year}, found {len(row)}")
    row["split"] = "smoke"
    row["selected_for_train"] = False
    row["selected_for_eval"] = True
    row["selection_reason"] = "039_01_lowIC_dssat_stress_smoke"
    return row.reset_index(drop=True)


def load_run_config(selected: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    run_config = direct_ppo.load_yaml(BASE_CONFIG)
    # Deep-copy YAML-safe structures.
    run_config = json.loads(json.dumps(run_config))
    run_config["seed"] = SEED
    run_config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    run_config["runtime"]["max_steps"] = MAX_STEPS
    env_config = baseline034.build_env_config(run_config, selected)
    env_config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    env_config["runtime"]["max_steps"] = MAX_STEPS
    env_config["seed"] = SEED
    return run_config, env_config


def configure_baseline_globals() -> None:
    """Redirect baseline helpers into the 039_01 output folder."""
    baseline034.TASK_ID = "039_01"
    baseline034.OUT = OUT
    baseline034.DOC = DOC
    baseline034.PROMPT = PROMPT
    baseline034.MAX_STEPS = MAX_STEPS
    # Reuse the 037_07 level-1 static management correction.  Without this,
    # multiple static expert/recorded events can be silently ignored by DSSAT.
    baseline034.replace_static_application_rows = baseline037.replace_static_application_rows_level1  # type: ignore[assignment]


def run_one_condition(
    condition: str,
    input_root: Path,
    run_config: dict[str, Any],
    env_config: dict[str, Any],
    station: str,
    year: int,
    scenario: str,
    recorded_templates: dict[str, dict[int, dict[str, float]]],
) -> tuple[pd.DataFrame | None, dict[str, Any], str]:
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = input_root
    status = "ok"
    try:
        tag_scenario = f"{condition}_{scenario}"
        daily, summary, checks = baseline034.evaluate_scenario(
            run_config,
            env_config,
            station,
            int(year),
            scenario,
            recorded_templates,
        )
        daily = daily.copy()
        daily.insert(0, "condition", condition)
        summary = dict(summary)
        summary["condition"] = condition
        summary["input_root"] = str(input_root.relative_to(ROOT)).replace("\\", "/")
        summary["render_checks_pass"] = all(x.get("status") == "pass" for x in checks)

        # Move snapshot to condition-specific folder because baseline helper only
        # keys snapshots by station/year/scenario.
        old_snapshot = OUT / "snapshots" / station / str(year) / scenario
        new_snapshot = OUT / "snapshots" / condition / station / str(year) / scenario
        if old_snapshot.exists():
            if new_snapshot.exists():
                shutil.rmtree(new_snapshot)
            new_snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_snapshot), str(new_snapshot))
            summary["snapshot_path"] = str(new_snapshot.relative_to(ROOT)).replace("\\", "/")
        summary["run_status"] = "ok"
        return daily, summary, ""
    except Exception:
        status = "failed"
        summary = {
            "condition": condition,
            "station_code": station,
            "site": baseline034.STATION_TO_SITE.get(station, ""),
            "year": int(year),
            "scenario": scenario,
            "input_root": str(input_root.relative_to(ROOT)).replace("\\", "/"),
            "run_status": status,
        }
        return None, summary, traceback.format_exc()
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root


def summarize_condition_effect(summary: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    pivot = summary[summary["run_status"].eq("ok")].copy()
    if pivot.empty:
        return {"branch": "C_run_failed", "reason": "no successful runs"}
    def value(condition: str, scenario: str, col: str) -> float:
        sub = pivot[pivot["condition"].eq(condition) & pivot["scenario"].eq(scenario)]
        if len(sub) != 1 or col not in sub:
            return math.nan
        return float(pd.to_numeric(sub[col], errors="coerce").iloc[0])

    for col in ["grain_yield_kg_ha", "actual_irrigation_mm", "actual_nitrogen_kg_ha", "max_water_stress", "max_nitrogen_stress", "WP_ET_kg_m3", "PFP_N_kg_kg"]:
        result[f"lowIC_null_minus_original_null_{col}"] = value("lowIC", "null", col) - value("original", "null", col)
        result[f"lowIC_expert_minus_lowIC_null_{col}"] = value("lowIC", "official_extension_expert", col) - value("lowIC", "null", col)

    stress_enhanced = (
        result.get("lowIC_null_minus_original_null_max_water_stress", math.nan) > 1e-9
        or result.get("lowIC_null_minus_original_null_max_nitrogen_stress", math.nan) > 1e-9
    )
    expert_helps = (
        result.get("lowIC_expert_minus_lowIC_null_grain_yield_kg_ha", math.nan) > 1e-9
        or result.get("lowIC_expert_minus_lowIC_null_max_water_stress", math.nan) < -1e-9
        or result.get("lowIC_expert_minus_lowIC_null_max_nitrogen_stress", math.nan) < -1e-9
    )
    if not stress_enhanced:
        result["branch"] = "B_lowIC_did_not_strengthen_stress"
    elif not expert_helps:
        result["branch"] = "B_management_did_not_alleviate_lowIC_stress"
    else:
        result["branch"] = "A_lowIC_smoke_pass"
    return result


def plot_stress(daily: pd.DataFrame, station: str, year: int) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig.suptitle(f"039_01 {station}{year}: original IC vs lowIC DSSAT stress smoke", fontsize=13, fontweight="bold")
    styles = {
        ("original", "null"): ("#555555", "-"),
        ("original", "official_extension_expert"): ("#6A51A3", "--"),
        ("lowIC", "null"): ("#D95F02", "-"),
        ("lowIC", "official_extension_expert"): ("#1B9E77", "--"),
    }
    labels = {
        ("original", "null"): "original-null",
        ("original", "official_extension_expert"): "original-expert",
        ("lowIC", "null"): "lowIC-null",
        ("lowIC", "official_extension_expert"): "lowIC-expert",
    }
    for (cond, scenario), group in daily.groupby(["condition", "scenario"], sort=False):
        color, ls = styles.get((cond, scenario), ("#000000", "-"))
        label = labels.get((cond, scenario), f"{cond}-{scenario}")
        group = group.sort_values("dap")
        axes[0, 0].plot(group["dap"], pd.to_numeric(group["swfac"], errors="coerce"), label=label, color=color, linestyle=ls)
        axes[0, 1].plot(group["dap"], pd.to_numeric(group["nstres"], errors="coerce"), label=label, color=color, linestyle=ls)
        axes[1, 0].plot(group["dap"], pd.to_numeric(group["grnwt"], errors="coerce"), label=label, color=color, linestyle=ls)
        axes[1, 1].plot(group["dap"], pd.to_numeric(group["topwt"], errors="coerce"), label=label, color=color, linestyle=ls)
    axes[0, 0].set_title("Water stress index")
    axes[0, 0].set_ylabel("WSPD/SWFAC (0=no stress)")
    axes[0, 1].set_title("Nitrogen stress index")
    axes[0, 1].set_ylabel("NSTD/NSTRES (0=no stress)")
    axes[1, 0].set_title("Grain weight")
    axes[1, 0].set_ylabel("GRNWT kg/ha")
    axes[1, 1].set_title("Biomass")
    axes[1, 1].set_ylabel("TOPWT kg/ha")
    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("DAP")
    axes[0, 1].legend(fontsize=8, loc="best")
    out = OUT / "figures" / f"039_01_{station}{year}_original_vs_lowIC_stress.png"
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_record(summary: pd.DataFrame, effect: dict[str, Any], failures: pd.DataFrame, fig_path: Path | None) -> None:
    lines = [
        "# 039_01 lowIC 初始土壤水氮减半 DSSAT 胁迫响应 smoke 记录",
        "",
        "## 任务边界",
        "",
        "- 本任务只运行 DSSAT 前向模拟；",
        "- 不训练 PPO/DQN；",
        "- 不修改原始输入或 lowIC 输入；",
        "- 通过临时重定向 `ppo_safe_rendering.MULTISITE_INPUT_ROOT` 分别读取 original 与 lowIC 输入。",
        "",
        "## 结果分支",
        "",
        f"- 分支：`{effect.get('branch', 'unknown')}`",
        "",
        "## Summary",
        "",
        md_table(summary, 80),
        "",
        "## lowIC 相对 original / expert 相对 null 的差异",
        "",
        md_table(pd.DataFrame([effect]), 20),
        "",
        "## 失败记录",
        "",
        md_table(failures, 40),
        "",
        "## 图件",
        "",
        f"- stress figure: `{fig_path.relative_to(ROOT).as_posix() if fig_path else ''}`",
        "",
        "## 文件",
        "",
        f"- daily: `benchmark_results/{TASK}/tables/039_01_smoke_daily.csv`",
        f"- summary: `benchmark_results/{TASK}/tables/039_01_smoke_summary.csv`",
        f"- effect: `benchmark_results/{TASK}/tables/039_01_condition_effect.csv`",
        f"- result json: `benchmark_results/{TASK}/039_01_result.json`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", default=DEFAULT_STATION, help="Station code, e.g. FQA/HLA/LCA/SYA/YCA")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    args = parser.parse_args()

    start = time.time()
    ensure_dirs()
    configure_baseline_globals()
    selected = load_single_selection(args.station, args.year)
    selected.to_csv(OUT / "configs" / "039_01_selected_site_year.csv", index=False, encoding="utf-8-sig")
    run_config, env_config = load_run_config(selected)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "039_01_resolved_env_config.yaml")

    recorded_templates = baseline034.recorded_template_schedules()
    daily_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for condition, input_root in [("original", ORIGINAL_INPUT_ROOT), ("lowIC", LOWIC_INPUT_ROOT)]:
        for scenario in SCENARIOS:
            print(f"[039_01] {condition} {args.station}{args.year} {scenario}", flush=True)
            daily, summary, tb = run_one_condition(condition, input_root, run_config, env_config, args.station, args.year, scenario, recorded_templates)
            summaries.append(summary)
            if daily is not None:
                daily_parts.append(daily)
            if tb:
                failures.append(
                    {
                        "condition": condition,
                        "station_code": args.station,
                        "year": args.year,
                        "scenario": scenario,
                        "traceback": tb[-3000:],
                    }
                )

    summary_df = pd.DataFrame(summaries)
    daily_df = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    failures_df = pd.DataFrame(failures)
    effect = summarize_condition_effect(summary_df)
    effect["station_code"] = args.station
    effect["year"] = int(args.year)
    effect["elapsed_seconds"] = time.time() - start

    summary_df.to_csv(OUT / "tables" / "039_01_smoke_summary.csv", index=False, encoding="utf-8-sig")
    daily_df.to_csv(OUT / "tables" / "039_01_smoke_daily.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([effect]).to_csv(OUT / "tables" / "039_01_condition_effect.csv", index=False, encoding="utf-8-sig")
    failures_df.to_csv(OUT / "tables" / "039_01_failures.csv", index=False, encoding="utf-8-sig")

    fig_path = plot_stress(daily_df, args.station, int(args.year)) if not daily_df.empty else None
    result = {
        "task": TASK,
        "station_code": args.station,
        "year": int(args.year),
        "branch": effect.get("branch", "unknown"),
        "successful_runs": int(summary_df["run_status"].eq("ok").sum()) if "run_status" in summary_df else 0,
        "failed_runs": int(len(failures_df)),
        "output_root": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "figure": str(fig_path.relative_to(ROOT)).replace("\\", "/") if fig_path else "",
        "elapsed_seconds": effect["elapsed_seconds"],
    }
    (OUT / "039_01_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_record(summary_df, effect, failures_df, fig_path)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0 if result["branch"] == "A_lowIC_smoke_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
