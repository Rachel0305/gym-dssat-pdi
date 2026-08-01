from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_sya_lowIC_teacher_candidate_search_041_00 as base04100
import ppo_safe_rendering


TASK_ID = "041_01"
TASK_NAME = "sya_lowIC_teacher_targeted_refinement"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

TARGET_YEARS = [2015, 2017, 2019, 2022, 2023]

N_REFINEMENT: dict[str, dict[int, float]] = {
    "N160_two80": {1: 80.0, 43: 80.0},
    "N200_early_mid": {1: 80.0, 43: 120.0},
    "N200_mid_late": {43: 80.0, 61: 120.0},
    "N240_three80": {1: 80.0, 43: 80.0, 61: 80.0},
    "N240_two120": {1: 120.0, 43: 120.0},
}

WP_ET_WATER_REFINEMENT: dict[str, dict[int, float]] = {
    "W210_drop_dap8": {1: 45.0, 31: 45.0, 38: 30.0, 61: 45.0, 91: 45.0},
    "W195_no_dap8_late_save": {1: 45.0, 31: 45.0, 61: 45.0, 91: 30.0, 110: 30.0},
    "W210_late_balanced_no120": {1: 30.0, 30: 45.0, 60: 45.0, 90: 45.0, 105: 45.0},
    "W195_late_balanced_save": {1: 30.0, 30: 45.0, 60: 45.0, 90: 45.0, 105: 30.0},
    "W180_mid_late_saving": {30: 45.0, 60: 45.0, 90: 45.0, 105: 45.0},
    "W210_mid_late_saving": {30: 45.0, 60: 45.0, 75: 30.0, 95: 45.0, 110: 45.0},
}

YIELD_2017_WATER_REFINEMENT: dict[str, dict[int, float]] = {
    "W240_bridge_15_35_55_75": {15: 45.0, 35: 45.0, 55: 45.0, 75: 45.0, 95: 30.0, 115: 30.0},
    "W240_early_mid_bridge": {1: 45.0, 22: 45.0, 45: 45.0, 68: 45.0, 95: 30.0, 115: 30.0},
    "W240_stress_bridge": {30: 45.0, 50: 45.0, 70: 45.0, 90: 45.0, 105: 30.0, 120: 30.0},
    "W240_pre195_balanced": {1: 30.0, 30: 45.0, 50: 45.0, 70: 45.0, 90: 30.0, 105: 45.0},
    "W240_shifted_mid_late": {8: 45.0, 30: 45.0, 52: 45.0, 74: 45.0, 96: 30.0, 118: 30.0},
    "W225_less_early_more_late": {30: 45.0, 52: 45.0, 74: 45.0, 96: 45.0, 118: 45.0},
}


def ensure_dirs() -> None:
    for rel in ["tables", "runs", "configs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
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


def build_refinement_grid(years: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in years:
        water_set = YIELD_2017_WATER_REFINEMENT if int(year) == 2017 else WP_ET_WATER_REFINEMENT
        for w_name, water in water_set.items():
            for n_name, nitrogen in N_REFINEMENT.items():
                base04100.validate_schedule(water, nitrogen)
                rows.append(
                    {
                        "year": int(year),
                        "candidate_id": f"{w_name}__{n_name}",
                        "water_schedule_id": w_name,
                        "n_schedule_id": n_name,
                        "water_schedule_json": json.dumps(water, sort_keys=True),
                        "n_schedule_json": json.dumps(nitrogen, sort_keys=True),
                        "requested_irrigation_mm": float(sum(water.values())),
                        "requested_nitrogen_kg_ha": float(sum(nitrogen.values())),
                        "pre_dap90_irrigation_mm": float(sum(v for d, v in water.items() if d <= 90)),
                        "refinement_reason": "2017_yield_and_water_stress_gap" if int(year) == 2017 else "wp_et_gap",
                    }
                )
    return pd.DataFrame(rows)


def write_record(result: dict[str, Any], summary: pd.DataFrame, all3: pd.DataFrame) -> None:
    cols = [
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
        "max_swfac",
        "max_nstres",
        "all3_win",
    ]
    top = summary.sort_values(
        ["year", "all3_win", "win_count", "gap_yield_vs_four_max", "gap_wp_et_vs_four_max", "gap_pfp_n_vs_four_max"],
        ascending=[True, False, False, False, False, False],
    ).groupby("year").head(5)
    by_year = (
        summary.groupby("year", as_index=False)
        .agg(
            candidates=("candidate_id", "count"),
            all3_candidates=("all3_win", "sum"),
            best_yield_gap=("gap_yield_vs_four_max", "max"),
            best_wp_gap=("gap_wp_et_vs_four_max", "max"),
            best_pfp_gap=("gap_pfp_n_vs_four_max", "max"),
            best_win_count=("win_count", "max"),
        )
        .sort_values("year")
    )
    lines = [
        f"# {TASK_ID} SYA lowIC 缺口年份 teacher 定向补搜记录",
        "",
        "## 结论",
        "",
        f"- 分支：`{result['branch']}`",
        f"- 目标年份：{', '.join(map(str, result['years']))}",
        f"- 候选总数：{result['candidate_runs_attempted']}",
        f"- 成功运行：{result['successful_runs']}",
        f"- 三项全超 teacher 数：{result['all3_teacher_count']}",
        f"- 补出三项全超的年份：{', '.join(map(str, result['all3_years'])) if result['all3_years'] else '无'}",
        "",
        "## 分年份摘要",
        "",
        md_table(by_year, max_rows=20),
        "",
        "## 三项全超候选",
        "",
        md_table(all3[cols] if not all3.empty else all3, max_rows=80),
        "",
        "## 每年 Top 5 候选",
        "",
        md_table(top[cols], max_rows=80),
        "",
        "## 输出文件",
        "",
        f"- 候选总表：`{result['outputs']['candidate_summary']}`",
        f"- 三项全超 teacher：`{result['outputs']['all3_teacher_candidates']}`",
        f"- 动作事件表：`{result['outputs']['requested_actions']}`",
        f"- refinement 网格：`{result['outputs']['refinement_grid']}`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run(years: list[int], max_candidates: int | None) -> None:
    ensure_dirs()
    grid = build_refinement_grid(years)
    if max_candidates is not None:
        grid = grid.head(int(max_candidates)).copy()
    thresholds = base04100.load_baseline_thresholds()
    thresholds = thresholds[thresholds["year"].isin(years)].copy()
    grid.to_csv(OUT / "tables" / "041_01_refinement_grid.csv", index=False)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "mode": "dry_run",
        "station": base04100.STATION,
        "years": years,
        "lowIC_input_root": base04100.LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "lowIC_input_root_exists": base04100.LOWIC_INPUT_ROOT.exists(),
        "baseline_summary_exists": base04100.BASELINE_SUMMARY.exists(),
        "candidate_count_total": int(len(grid)),
        "candidate_count_by_year": grid.groupby("year")["candidate_id"].count().to_dict(),
        "threshold_year_count": int(len(thresholds)),
        "next_step_allowed": bool(base04100.LOWIC_INPUT_ROOT.exists() and base04100.BASELINE_SUMMARY.exists() and len(thresholds) == len(years)),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_search(years: list[int], max_candidates: int | None) -> None:
    ensure_dirs()
    base04100.OUT = OUT
    base04100.DOC = DOC
    grid = build_refinement_grid(years)
    if max_candidates is not None:
        grid = grid.head(int(max_candidates)).copy()
    grid_path = OUT / "tables" / "041_01_refinement_grid.csv"
    grid.to_csv(grid_path, index=False)

    thresholds_df = base04100.load_baseline_thresholds()
    thresholds_df = thresholds_df[thresholds_df["year"].isin(years)].copy()
    if len(thresholds_df) != len(years):
        raise RuntimeError(f"Missing baseline thresholds: expected {years}, got {thresholds_df['year'].tolist()}")
    thresholds_by_year = {int(row.year): row._asdict() for row in thresholds_df.itertuples(index=False)}
    selection = base04100.build_selection_for_years(years)
    config = base04100.ppo04040.load_config()
    env_config = base04100.ppo04040.base03222.direct_ppo.build_env_config(config, selection)
    (OUT / "configs" / "041_01_env_config.json").write_text(json.dumps(env_config, indent=2, ensure_ascii=False), encoding="utf-8")

    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = base04100.LOWIC_INPUT_ROOT
    rows: list[dict[str, Any]] = []
    try:
        total = len(grid)
        for counter, (_, candidate) in enumerate(grid.iterrows(), start=1):
            year = int(candidate["year"])
            print(f"[041_01] {counter}/{total} {base04100.STATION}{year} {candidate['candidate_id']}", flush=True)
            rows.append(base04100.run_one_candidate(year, candidate, thresholds_by_year[year], config, env_config))
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root

    summary = pd.DataFrame(rows)
    for col in ["gap_yield_vs_four_max", "gap_wp_et_vs_four_max", "gap_pfp_n_vs_four_max"]:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    summary["win_count"] = (
        (summary["gap_yield_vs_four_max"] > 0).astype(int)
        + (summary["gap_wp_et_vs_four_max"] > 0).astype(int)
        + (summary["gap_pfp_n_vs_four_max"] > 0).astype(int)
    )
    summary["all3_win"] = summary["win_count"].eq(3)
    summary_path = OUT / "tables" / "041_01_candidate_summary.csv"
    summary.to_csv(summary_path, index=False)
    all3 = summary[summary["all3_win"]].copy()
    all3_path = OUT / "tables" / "041_01_all3_teacher_candidates.csv"
    all3.to_csv(all3_path, index=False)

    action_frames = []
    for run_dir in summary["run_dir"].dropna().astype(str):
        path = ROOT / run_dir / "requested_actions.csv"
        if path.exists() and path.stat().st_size > 0:
            try:
                action_frames.append(pd.read_csv(path, keep_default_na=False))
            except pd.errors.EmptyDataError:
                continue
    actions = pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame()
    actions_path = OUT / "tables" / "041_01_requested_actions.csv"
    actions.to_csv(actions_path, index=False)

    all3_years = sorted(all3["year"].astype(int).unique().tolist()) if not all3.empty else []
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": "A_all3_teacher_refined" if all3_years else "C_no_all3_teacher_in_refinement_grid",
        "training_run": False,
        "station": base04100.STATION,
        "site": base04100.SITE,
        "years": years,
        "lowIC_input_root": base04100.LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "candidate_runs_attempted": int(len(summary)),
        "successful_runs": int(summary["run_status"].eq("ok").sum()),
        "failed_runs": int(summary["run_status"].ne("ok").sum()),
        "all3_teacher_count": int(len(all3)),
        "all3_years": all3_years,
        "outputs": {
            "candidate_summary": summary_path.relative_to(ROOT).as_posix(),
            "all3_teacher_candidates": all3_path.relative_to(ROOT).as_posix(),
            "requested_actions": actions_path.relative_to(ROOT).as_posix(),
            "refinement_grid": grid_path.relative_to(ROOT).as_posix(),
            "result_json": (OUT / "041_01_result.json").relative_to(ROOT).as_posix(),
            "record_md": DOC.relative_to(ROOT).as_posix(),
        },
    }
    (OUT / "041_01_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_record(result, summary, all3)
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
    return [y for y in sorted(set(years)) if y in TARGET_YEARS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--years", default=",".join(map(str, TARGET_YEARS)))
    parser.add_argument("--max-candidates", type=int, default=None)
    args = parser.parse_args()
    years = parse_years(args.years)
    if not years:
        raise RuntimeError("No target years selected")
    if args.dry_run:
        dry_run(years, args.max_candidates)
    else:
        run_search(years, args.max_candidates)


if __name__ == "__main__":
    main()
