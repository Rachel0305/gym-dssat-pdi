"""040_21: rebuild SYA four baselines under the current lowIC input root.

This task is intentionally not a training task.  It rebuilds null, recorded
farmer template, official expert, and DSSAT auto baselines under the same
lowIC source used by the 040_20 PPO rework, so later PPO comparisons are
input-consistent.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

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


TASK = "040_21_sya_lowIC_four_baseline_rebuild"
OUT = ROOT / "benchmark_results" / TASK
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"
LOWIC_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
STATION = "SYA"
SITE = "SY"
SCENARIOS = baseline034.SCENARIOS
MAX_STEPS = 260
SEED = 0

LOWIC_MARKERS = [" 1    10   .14", " 1    10  0.14"]
ORIGIN_MARKERS = [" 1    10   .27", " 1    10  0.27"]


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_dirs() -> None:
    for rel_dir in ["configs", "evaluation", "snapshots", "logs"]:
        (OUT / rel_dir).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)
    if baseline034.BASE_CONFIG.exists():
        shutil.copy2(baseline034.BASE_CONFIG, OUT / "configs" / baseline034.BASE_CONFIG.name)


def classify_template_family(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if any(marker in text for marker in LOWIC_MARKERS):
        return "LOWIC"
    if any(marker in text for marker in ORIGIN_MARKERS):
        return "ORIGIN"
    return "UNKNOWN"


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
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


def configure_imported_modules() -> None:
    """Point trusted baseline helpers at the 040_21 lowIC output/input."""
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_ROOT
    baseline034.MULTISITE_INPUT_ROOT = LOWIC_ROOT
    baseline034.TASK_ID = "040_21"
    baseline034.OUT = OUT
    baseline034.DOC = DOC
    baseline034.PROMPT = PROMPT
    baseline034.MAX_STEPS = MAX_STEPS
    baseline034.replace_static_application_rows = baseline037.replace_static_application_rows_level1  # type: ignore[assignment]


def load_selected_years(mode: str, year: int | None) -> pd.DataFrame:
    split = pd.read_csv(baseline034.SPLIT_CSV, keep_default_na=False)
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    selected = split[split["station_code"].astype(str).eq(STATION)].copy()
    selected = selected[selected["year"].ge(2000)].copy()
    if year is not None:
        selected = selected[selected["year"].eq(int(year))].copy()
    elif mode == "smoke":
        selected = selected[selected["year"].eq(2014)].copy()
    else:
        selected = selected[selected["split"].astype(str).eq("validation")].copy()
    selected = selected.sort_values("year").reset_index(drop=True)
    if selected.empty:
        raise RuntimeError(f"没有选中任何 {STATION} 年份：mode={mode}, year={year}")
    return selected


def load_configs(selected: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    run_config = direct_ppo.load_yaml(baseline034.BASE_CONFIG)
    run_config = json.loads(json.dumps(run_config))
    run_config["seed"] = SEED
    run_config["paths"]["output_root"] = rel(OUT)
    run_config["runtime"]["max_steps"] = MAX_STEPS
    env_config = baseline034.build_env_config(run_config, selected)
    env_config["paths"]["output_root"] = rel(OUT)
    env_config["runtime"]["max_steps"] = MAX_STEPS
    env_config["seed"] = SEED
    return run_config, env_config


def source_preflight() -> dict[str, Any]:
    mzx = LOWIC_ROOT / SITE / "CNSY1201.MZX"
    issues: list[str] = []
    if not LOWIC_ROOT.exists():
        issues.append(f"lowIC input root missing: {rel(LOWIC_ROOT)}")
    if not mzx.exists():
        issues.append(f"SYA lowIC MZX missing: {rel(mzx)}")
    family = classify_template_family(mzx) if mzx.exists() else "MISSING"
    if family != "LOWIC":
        issues.append(f"source MZX family is {family}, expected LOWIC: {rel(mzx)}")
    return {
        "input_root": rel(LOWIC_ROOT),
        "source_mzx": rel(mzx),
        "source_family": family,
        "status": "pass" if not issues else "fail",
        "issues": "; ".join(issues),
    }


def write_record(
    mode: str,
    selected: pd.DataFrame,
    summary: pd.DataFrame,
    manifest: pd.DataFrame,
    audit: pd.DataFrame,
    render: pd.DataFrame,
    source_check: dict[str, Any],
    elapsed: float,
) -> None:
    coverage = manifest.groupby(["scenario", "status"]).size().reset_index(name="n") if not manifest.empty else pd.DataFrame()
    family_counts = pd.DataFrame()
    if not render.empty and "template_family_checked" in render.columns:
        family_rows = render[render["check_item"].astype(str).eq("rendered_template_ic_family")].copy()
        if not family_rows.empty:
            family_counts = family_rows.groupby("template_family_checked").size().reset_index(name="n")
    by_scenario = pd.DataFrame()
    if not summary.empty:
        tmp = summary.copy()
        for col in [
            "grain_yield_kg_ha",
            "actual_irrigation_mm",
            "actual_nitrogen_kg_ha",
            "WP_ET_kg_m3",
            "PFP_N_kg_kg",
            "max_water_stress",
            "max_nitrogen_stress",
        ]:
            if col in tmp:
                tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        by_scenario = tmp.groupby("scenario", as_index=False).agg(
            n=("year", "count"),
            mean_yield=("grain_yield_kg_ha", "mean"),
            mean_irrigation=("actual_irrigation_mm", "mean"),
            mean_nitrogen=("actual_nitrogen_kg_ha", "mean"),
            mean_wp_et=("WP_ET_kg_m3", "mean"),
            mean_pfp_n=("PFP_N_kg_kg", "mean"),
            max_water_stress=("max_water_stress", "max"),
            max_nitrogen_stress=("max_nitrogen_stress", "max"),
        )

    lines = [
        "# 040_21 SYA lowIC 四情景基线重建记录",
        "",
        "## 结论边界",
        "",
        "- 本任务只重建四情景基线，不训练 PPO/DQN。",
        "- 目的：给 040_20 lowIC PPO 结果提供同源、同条件的 null / recorded farmer template / official expert / DSSAT auto 对照。",
        "- 如果 rendered template 不是 LOWIC，本任务结果不得用于 lowIC 比较。",
        "",
        "## 运行配置",
        "",
        f"- mode：`{mode}`",
        f"- 输入源：`{source_check['input_root']}`",
        f"- 源 MZX：`{source_check['source_mzx']}`",
        f"- 源 MZX 家族：`{source_check['source_family']}`",
        f"- 源检查状态：`{source_check['status']}`",
        f"- 耗时：{elapsed:.1f} 秒",
        "",
        "## 年份范围",
        "",
        md_table(selected[["station_code", "site", "year", "split"]], 80),
        "",
        "## 覆盖状态",
        "",
        md_table(coverage, 80),
        "",
        "## rendered template IC 家族",
        "",
        md_table(family_counts, 20),
        "",
        "## 情景均值预览",
        "",
        md_table(by_scenario, 80),
        "",
        "## 非 ok 记录",
        "",
        md_table(manifest[manifest["status"].ne("ok")].copy(), 120),
        "",
        "## 管理事件审计非 ok 记录",
        "",
        md_table(audit[audit["status"].ne("ok")].copy() if "status" in audit else pd.DataFrame(), 120),
        "",
        "## 输出文件",
        "",
        f"- summary：`{rel(OUT / 'evaluation' / '040_21_baseline_summary.csv')}`",
        f"- daily：`{rel(OUT / 'evaluation' / '040_21_baseline_daily.csv')}`",
        f"- management audit：`{rel(OUT / 'evaluation' / '040_21_management_event_audit.csv')}`",
        f"- manifest：`{rel(OUT / 'evaluation' / '040_21_coverage_manifest.csv')}`",
        f"- render check：`{rel(OUT / 'evaluation' / '040_21_render_check.csv')}`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(mode: str, year: int | None) -> dict[str, Any]:
    ensure_dirs()
    configure_imported_modules()
    source_check = source_preflight()
    if source_check["status"] != "pass":
        raise RuntimeError(f"040_21 源输入检查失败：{source_check['issues']}")

    selected = load_selected_years(mode, year)
    selected.to_csv(OUT / "configs" / "040_21_selected_years.csv", index=False, encoding="utf-8-sig")
    run_config, env_config = load_configs(selected)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "040_21_resolved_env_config.yaml")
    recorded_templates = baseline034.recorded_template_schedules()

    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    render_rows: list[dict[str, Any]] = []
    start = time.time()

    for row in selected.itertuples(index=False):
        station = str(row.station_code)
        site = str(row.site)
        run_year = int(row.year)
        for scenario in SCENARIOS:
            print(f"[040_21] {mode} {station}{run_year} {scenario}", flush=True)
            try:
                daily, summary, checks = baseline034.evaluate_scenario(
                    run_config,
                    env_config,
                    station,
                    run_year,
                    scenario,
                    recorded_templates,
                )
                summary["season_last_dap"] = int(pd.to_numeric(daily["dap"], errors="coerce").max())
                summary["source_status"] = "generated_040_21_lowIC_four_baseline"
                summary["source_input_root"] = rel(LOWIC_ROOT)
                schedule = {} if scenario == "dssat_auto" else baseline034.schedule_for(scenario, site, recorded_templates)
                audit = baseline037.audit_row(summary, schedule)

                snapshot = ROOT / str(summary["snapshot_path"])
                template_files = sorted((OUT / "rendered_inputs" / station / str(run_year)).rglob("*.jinja2"))
                matching_templates = [p for p in template_files if scenario in p.name or scenario in str(p.parent)]
                if not matching_templates:
                    matching_templates = template_files[-1:]
                families = [classify_template_family(p) for p in matching_templates]
                family_counter = Counter(families)
                if family_counter.get("ORIGIN", 0) or family_counter.get("UNKNOWN", 0):
                    raise RuntimeError(f"rendered template family not clean LOWIC: {dict(family_counter)}")

                for check in checks:
                    render_rows.append(
                        {
                            "station_code": station,
                            "site": site,
                            "year": run_year,
                            "scenario": scenario,
                            **check,
                            "template_family_checked": ";".join(sorted(set(families))),
                        }
                    )
                for p, fam in zip(matching_templates, families):
                    render_rows.append(
                        {
                            "station_code": station,
                            "site": site,
                            "year": run_year,
                            "scenario": scenario,
                            "check_item": "rendered_template_ic_family",
                            "status": "pass" if fam == "LOWIC" else "fail",
                            "details": rel(p),
                            "template_family_checked": fam,
                        }
                    )

                daily_frames.append(daily)
                summary_rows.append(summary)
                audit_rows.append(audit)
                status = "ok" if audit.get("status") == "ok" else "event_chain_issue"
                manifest_rows.append({"station_code": station, "site": site, "year": run_year, "scenario": scenario, "status": status, "details": audit.get("issues", "")})
                pd.DataFrame(summary_rows).to_csv(OUT / "evaluation" / "040_21_baseline_summary_partial.csv", index=False, encoding="utf-8-sig")
            except Exception as exc:
                manifest_rows.append(
                    {
                        "station_code": station,
                        "site": site,
                        "year": run_year,
                        "scenario": scenario,
                        "status": "failed",
                        "details": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-4000:],
                    }
                )

    elapsed = time.time() - start
    summary_df = pd.DataFrame(summary_rows)
    daily_df = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    audit_df = pd.DataFrame(audit_rows)
    manifest_df = pd.DataFrame(manifest_rows)
    render_df = pd.DataFrame(render_rows)

    summary_df.to_csv(OUT / "evaluation" / "040_21_baseline_summary.csv", index=False, encoding="utf-8-sig")
    daily_df.to_csv(OUT / "evaluation" / "040_21_baseline_daily.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(OUT / "evaluation" / "040_21_management_event_audit.csv", index=False, encoding="utf-8-sig")
    manifest_df.to_csv(OUT / "evaluation" / "040_21_coverage_manifest.csv", index=False, encoding="utf-8-sig")
    render_df.to_csv(OUT / "evaluation" / "040_21_render_check.csv", index=False, encoding="utf-8-sig")

    write_record(mode, selected, summary_df, manifest_df, audit_df, render_df, source_check, elapsed)
    result = {
        "task": TASK,
        "mode": mode,
        "selected_years": int(len(selected)),
        "expected_runs": int(len(selected) * len(SCENARIOS)),
        "successful_runs": int(len(summary_df)),
        "non_ok_runs": int((manifest_df["status"] != "ok").sum()) if not manifest_df.empty and "status" in manifest_df else 0,
        "output_root": rel(OUT),
        "record_md": rel(DOC),
        "elapsed_seconds": elapsed,
    }
    (OUT / "040_21_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if result["non_ok_runs"]:
        print(manifest_df[manifest_df["status"].ne("ok")].to_string(index=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--year", type=int, default=None, help="可选：只跑某一年，例如 2014")
    args = parser.parse_args()
    run(args.mode, args.year)


if __name__ == "__main__":
    main()
