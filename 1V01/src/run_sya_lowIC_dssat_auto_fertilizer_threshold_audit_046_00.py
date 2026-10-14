"""046_00: SYA lowIC DSSAT native auto-fertilizer threshold audit.

This is a no-training diagnostic.  It reuses the trusted 040_21 lowIC baseline
pipeline, but monkey-patches the generated DSSAT automatic nitrogen block for a
small set of pre-registered NMTHR/NAMNT/NCODE variants.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_sya_lowIC_four_baseline_rebuild_040_21 as base04021  # noqa: E402
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline034  # noqa: E402


TASK = "046_00_sya_lowIC_dssat_auto_fertilizer_threshold_audit"
OUT = ROOT / "benchmark_results" / TASK
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"
YEARS_DEFAULT = [2014, 2017, 2022]

VARIANTS: dict[str, dict[str, Any]] = {
    "base_nmthr50_amt25_fe001": {"nmdep": 30, "nmthr": 50, "namnt": 25, "ncode": "FE001", "naoff": "GS000"},
    "low_nmthr10_amt25_fe001": {"nmdep": 30, "nmthr": 10, "namnt": 25, "ncode": "FE001", "naoff": "GS000"},
    "very_low_nmthr01_amt25_fe001": {"nmdep": 30, "nmthr": 1, "namnt": 25, "ncode": "FE001", "naoff": "GS000"},
    "high_nmthr99_amt50_fe001": {"nmdep": 30, "nmthr": 99, "namnt": 50, "ncode": "FE001", "naoff": "GS000"},
    "high_nmthr99_amt50_fe005": {"nmdep": 30, "nmthr": 99, "namnt": 50, "ncode": "FE005", "naoff": "GS000"},
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def ensure_dirs() -> None:
    for sub in ["configs", "evaluation", "daily_outputs"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_空表_"
    show = df.head(max_rows).copy()
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"| ... | 仅显示前 {max_rows} 行，共 {len(df)} 行 |")
    return "\n".join(lines)


def automatic_management_block_for_variant(year: int, cfg: dict[str, Any]) -> str:
    yy001 = f"{int(year) % 100:02d}001"
    return (
        "\n@  AUTOMATIC MANAGEMENT\n"
        "@N PLANTING    PFRST PLAST PH2OL PH2OU PH2OD PSTMX PSTMN\n"
        f" 1 PL          {yy001} {yy001}    40   100    30    40    10\n"
        "@N IRRIGATION  IMDEP ITHRL ITHRU IROFF IMETH IRAMT IREFF\n"
        " 1 IR             30    50   100 GS000 IR001    10     1\n"
        "@N NITROGEN    NMDEP NMTHR NAMNT NCODE NAOFF\n"
        f" 1 NI             {int(cfg['nmdep']):2d} {int(cfg['nmthr']):5d} {int(cfg['namnt']):5d} {cfg['ncode']} {cfg['naoff']}\n"
        "@N RESIDUES    RIPCN RTIME RIDEP\n"
        " 1 RE            100     1    20\n"
        "@N HARVEST     HFRST HLAST HPCNP HPCNR\n"
        " 1 HA              0 01001   100     0\n"
    )


def patch_existing_auto_nitrogen(text: str, cfg: dict[str, Any]) -> str:
    """Patch all treatment-level automatic nitrogen rows in a rendered MZX."""
    pattern = re.compile(r"(?m)^(\s*\d+\s+NI\s+)\d+(\s+)\d+(\s+)\d+(\s+)\S+(\s+)\S+\s*$")

    def repl(match: re.Match[str]) -> str:
        return (
            f"{match.group(1)}{int(cfg['nmdep']):>10d}"
            f"{match.group(2)}{int(cfg['nmthr']):>5d}"
            f"{match.group(3)}{int(cfg['namnt']):>5d}"
            f"{match.group(4)}{cfg['ncode']}"
            f"{match.group(5)}{cfg['naoff']}"
        )

    new, n = pattern.subn(repl, text)
    if n < 1:
        raise RuntimeError("未找到可修改的 @N NITROGEN / NI 行")
    return new


def run_variant(years: list[int], variant: str, cfg: dict[str, Any]) -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]]:
    base04021.OUT = OUT / f"variant_{variant}"
    base04021.DOC = DOC
    base04021.PROMPT = PROMPT
    baseline034.OUT = base04021.OUT
    baseline034.DOC = DOC
    baseline034.PROMPT = PROMPT
    baseline034.TASK_ID = f"046_00_{variant}"
    baseline034.SCENARIOS = ["dssat_auto"]
    baseline034.automatic_management_block = lambda y: automatic_management_block_for_variant(y, cfg)  # type: ignore[assignment]
    original_set_auto = baseline034.set_auto_treatment_one

    def set_auto_with_variant(text: str, year: int) -> str:
        text2 = original_set_auto(text, year)
        return patch_existing_auto_nitrogen(text2, cfg)

    baseline034.set_auto_treatment_one = set_auto_with_variant  # type: ignore[assignment]
    base04021.configure_imported_modules()
    baseline034.TASK_ID = f"046_00_{variant}"
    baseline034.automatic_management_block = lambda y: automatic_management_block_for_variant(y, cfg)  # type: ignore[assignment]
    baseline034.set_auto_treatment_one = set_auto_with_variant  # type: ignore[assignment]

    selected = base04021.load_selected_years("all", None)
    selected = selected[selected["year"].isin(years)].copy()
    run_config, env_config = base04021.load_configs(selected)
    recorded_templates = baseline034.recorded_template_schedules()
    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        station = str(row.station_code)
        year = int(row.year)
        print(f"[046_00] {variant} {station}{year}", flush=True)
        try:
            daily, summary, _checks = baseline034.evaluate_scenario(
                run_config, env_config, station, year, "dssat_auto", recorded_templates
            )
            daily["auto_variant"] = variant
            daily["nmthr"] = cfg["nmthr"]
            daily["namnt"] = cfg["namnt"]
            daily["ncode"] = cfg["ncode"]
            summary["auto_variant"] = variant
            summary["nmthr"] = cfg["nmthr"]
            summary["namnt"] = cfg["namnt"]
            summary["ncode"] = cfg["ncode"]
            daily_frames.append(daily)
            summary_rows.append(summary)
        except Exception as exc:
            failures.append({"auto_variant": variant, "station_code": station, "year": year, "error": str(exc)})
    return daily_frames, summary_rows, failures


def write_record(summary: pd.DataFrame, failures: pd.DataFrame, years: list[int], elapsed: float) -> str:
    if summary.empty:
        branch = "E_no_successful_runs"
    elif (pd.to_numeric(summary.get("actual_nitrogen_kg_ha", 0), errors="coerce").fillna(0) > 0).any():
        branch = "A_auto_n_triggered_by_parameter_variant"
    else:
        branch = "B_native_auto_n_not_triggered_by_threshold_variants"
    compact_cols = [
        "auto_variant",
        "year",
        "grain_yield_kg_ha",
        "actual_irrigation_mm",
        "actual_nitrogen_kg_ha",
        "WP_ET_kg_m3",
        "PFP_N_kg_grain_per_kg_N",
        "max_nstd",
        "nmthr",
        "namnt",
        "ncode",
    ]
    compact = summary[[c for c in compact_cols if c in summary.columns]].copy() if not summary.empty else pd.DataFrame()
    lines = [
        "# 046_00 SYA lowIC DSSAT auto 施肥阈值复核记录",
        "",
        f"- 分支：`{branch}`",
        "- 性质：零训练 DSSAT baseline 参数复核。",
        f"- 年份：{years}",
        f"- 输入源：`{rel(base04021.LOWIC_ROOT)}`",
        f"- 耗时：{elapsed:.1f} s",
        "",
        "## 结果摘要",
        "",
        md_table(compact, 80),
        "",
        "## 失败记录",
        "",
        md_table(failures, 80),
        "",
        "## 输出",
        "",
        f"- summary: `{rel(OUT / 'evaluation' / '046_00_auto_n_threshold_summary.csv')}`",
        f"- daily: `{rel(OUT / 'evaluation' / '046_00_auto_n_threshold_daily.csv')}`",
        f"- failures: `{rel(OUT / 'evaluation' / '046_00_auto_n_threshold_failures.csv')}`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return branch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default=",".join(str(y) for y in YEARS_DEFAULT), help="逗号分隔年份")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    years = [int(x.strip()) for x in str(args.years).split(",") if x.strip()]
    ensure_dirs()
    if args.dry_run:
        print(json.dumps({"task": TASK, "mode": "dry_run", "years": years, "variants": VARIANTS, "next_step_allowed": True}, indent=2, ensure_ascii=False))
        return
    start = time.time()
    all_daily: list[pd.DataFrame] = []
    all_summary: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    for variant, cfg in VARIANTS.items():
        daily_frames, summary_rows, failures = run_variant(years, variant, cfg)
        all_daily.extend(daily_frames)
        all_summary.extend(summary_rows)
        all_failures.extend(failures)
    daily_df = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    summary_df = pd.DataFrame(all_summary)
    failures_df = pd.DataFrame(all_failures)
    daily_df.to_csv(OUT / "evaluation" / "046_00_auto_n_threshold_daily.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT / "evaluation" / "046_00_auto_n_threshold_summary.csv", index=False, encoding="utf-8-sig")
    failures_df.to_csv(OUT / "evaluation" / "046_00_auto_n_threshold_failures.csv", index=False, encoding="utf-8-sig")
    branch = write_record(summary_df, failures_df, years, time.time() - start)
    result = {
        "task": TASK,
        "branch": branch,
        "years": years,
        "successful_runs": int(len(summary_df)),
        "failed_runs": int(len(failures_df)),
        "summary": rel(OUT / "evaluation" / "046_00_auto_n_threshold_summary.csv"),
        "daily": rel(OUT / "evaluation" / "046_00_auto_n_threshold_daily.csv"),
        "record_md": rel(DOC),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
