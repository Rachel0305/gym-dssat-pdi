from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "041_02"
TASK_NAME = "sya_lowIC_layered_teacher_imitation_dataset"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"

SOURCE_04100 = (
    ROOT
    / "benchmark_results"
    / "041_00_sya_lowIC_teacher_candidate_search"
    / "tables"
    / "041_00_candidate_summary.csv"
)
SOURCE_04101 = (
    ROOT
    / "benchmark_results"
    / "041_01_sya_lowIC_teacher_targeted_refinement"
    / "tables"
    / "041_01_candidate_summary.csv"
)
YEARS = list(range(2014, 2024))

ACTION_GRID = [
    {"action_index": idx, "irrigation_mm": float(i), "nitrogen_kg_ha": float(n)}
    for idx, (i, n) in enumerate((i, n) for i in [0.0, 30.0, 45.0] for n in [0.0, 80.0, 120.0])
]


def ensure_dirs() -> None:
    for rel in ["tables", "configs"]:
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


def read_source(path: Path, source_task: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, keep_default_na=False)
    df["source_task"] = source_task
    return df


def prepare_candidates() -> pd.DataFrame:
    frames = [read_source(SOURCE_04100, "041_00")]
    if SOURCE_04101.exists():
        frames.append(read_source(SOURCE_04101, "041_01"))
    df = pd.concat(frames, ignore_index=True)
    numeric_cols = [
        "year",
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
        "mask_forced_noop_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year"] = df["year"].astype(int)
    df["yield_win"] = df["gap_yield_vs_four_max"] > 0
    df["wp_et_win"] = df["gap_wp_et_vs_four_max"] > 0
    df["pfp_n_win"] = df["gap_pfp_n_vs_four_max"] > 0
    df["win_count"] = df[["yield_win", "wp_et_win", "pfp_n_win"]].sum(axis=1)
    df["all3_win"] = df["win_count"].eq(3)
    df["yield_deficit_scaled"] = np.maximum(-df["gap_yield_vs_four_max"], 0.0) / 1000.0
    df["wp_et_deficit_scaled"] = np.maximum(-df["gap_wp_et_vs_four_max"], 0.0) / 0.1
    df["pfp_n_deficit_scaled"] = np.maximum(-df["gap_pfp_n_vs_four_max"], 0.0) / 10.0
    df["deficit_sum_scaled"] = df[["yield_deficit_scaled", "wp_et_deficit_scaled", "pfp_n_deficit_scaled"]].sum(axis=1)
    df = df[df["run_status"].astype(str).eq("ok")].copy()
    df = df[pd.to_numeric(df.get("mask_forced_noop_count", 0), errors="coerce").fillna(0).eq(0)].copy()
    return df


def select_teachers(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        year_df = candidates[candidates["year"].eq(year)].copy()
        if year_df.empty:
            raise RuntimeError(f"No candidate rows for {year}")
        strong = year_df[year_df["all3_win"]].copy()
        if not strong.empty:
            selected = strong.sort_values(
                [
                    "gap_yield_vs_four_max",
                    "gap_wp_et_vs_four_max",
                    "gap_pfp_n_vs_four_max",
                    "summary_irrigation_total",
                    "summary_nitrogen_total",
                ],
                ascending=[False, False, False, True, True],
            ).iloc[0].copy()
            selected["teacher_tier"] = "strong_all3"
            selected["teacher_base_weight"] = 1.0
        else:
            selected = year_df.sort_values(
                [
                    "win_count",
                    "deficit_sum_scaled",
                    "gap_yield_vs_four_max",
                    "gap_wp_et_vs_four_max",
                    "gap_pfp_n_vs_four_max",
                    "summary_irrigation_total",
                    "summary_nitrogen_total",
                ],
                ascending=[False, True, False, False, False, True, True],
            ).iloc[0].copy()
            selected["teacher_tier"] = "near_miss"
            selected["teacher_base_weight"] = 0.5
        selected["selected_for_imitation"] = True
        rows.append(selected)
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["teacher_rank_note"] = np.where(
        out["teacher_tier"].eq("strong_all3"),
        "三项全超，作为强 teacher",
        "未三项全超，作为弱 teacher/near-miss 辅助信号",
    )
    return out


def action_index(irrigation: float, nitrogen: float) -> int:
    for item in ACTION_GRID:
        if abs(float(item["irrigation_mm"]) - float(irrigation)) < 1e-9 and abs(float(item["nitrogen_kg_ha"]) - float(nitrogen)) < 1e-9:
            return int(item["action_index"])
    raise ValueError(f"Action not in grid: irrigation={irrigation}, nitrogen={nitrogen}")


def build_imitation_rows(selected: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for row in selected.itertuples(index=False):
        run_dir = ROOT / str(row.run_dir)
        daily_path = run_dir / "daily_values.csv"
        if not daily_path.exists():
            raise FileNotFoundError(daily_path)
        daily = pd.read_csv(daily_path, keep_default_na=False)
        daily["source_task"] = str(row.source_task)
        daily["teacher_tier"] = str(row.teacher_tier)
        daily["teacher_base_weight"] = float(row.teacher_base_weight)
        daily["teacher_candidate_id"] = str(row.candidate_id)
        daily["teacher_run_dir"] = str(row.run_dir)
        for col in ["requested_irrigation_mm_action", "requested_nitrogen_kg_ha_action"]:
            daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0.0)
        daily["teacher_action_index"] = [
            action_index(i, n)
            for i, n in zip(daily["requested_irrigation_mm_action"], daily["requested_nitrogen_kg_ha_action"])
        ]
        daily["is_nonzero_action_day"] = (
            (daily["requested_irrigation_mm_action"].abs() > 1e-9)
            | (daily["requested_nitrogen_kg_ha_action"].abs() > 1e-9)
        )
        daily["sample_weight"] = np.where(
            daily["teacher_tier"].eq("strong_all3"),
            np.where(daily["is_nonzero_action_day"], 1.0, 0.2),
            np.where(daily["is_nonzero_action_day"], 0.5, 0.1),
        )
        frames.append(daily)
    return pd.concat(frames, ignore_index=True)


def write_record(result: dict[str, Any], selected: pd.DataFrame, daily: pd.DataFrame, action_dist: pd.DataFrame) -> None:
    selected_cols = [
        "year",
        "source_task",
        "teacher_tier",
        "candidate_id",
        "grain_yield_kg_ha",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "summary_irrigation_total",
        "summary_nitrogen_total",
        "gap_yield_vs_four_max",
        "gap_wp_et_vs_four_max",
        "gap_pfp_n_vs_four_max",
        "win_count",
        "deficit_sum_scaled",
    ]
    lines = [
        f"# {TASK_ID} SYA lowIC 分层 teacher imitation 数据集记录",
        "",
        "## 结论",
        "",
        f"- 分支：`{result['branch']}`",
        f"- 选中年份数：{result['selected_year_count']}",
        f"- strong teacher 年份数：{result['strong_teacher_year_count']}",
        f"- near-miss teacher 年份数：{result['near_miss_teacher_year_count']}",
        f"- 每日 imitation 样本数：{result['daily_sample_count']}",
        f"- 非零动作样本数：{result['nonzero_action_sample_count']}",
        "",
        "## 选中 teacher",
        "",
        md_table(selected[selected_cols], max_rows=20),
        "",
        "## 动作分布",
        "",
        md_table(action_dist, max_rows=30),
        "",
        "## 说明",
        "",
        "- 本任务不训练 PPO，只构建 warm-start 数据集。",
        "- strong teacher 是三项全超；near-miss teacher 是较弱辅助信号，不能当作三项全优真值。",
        "- 后续 041_03 若训练，必须保留 strong/near-miss 分层权重，不能混成同等标签。",
        "",
        "## 输出文件",
        "",
        f"- 选中 teacher：`{result['outputs']['selected_teachers']}`",
        f"- 每日 imitation 数据：`{result['outputs']['imitation_daily_dataset']}`",
        f"- 非零动作日：`{result['outputs']['imitation_action_days']}`",
        f"- 动作分布：`{result['outputs']['action_distribution']}`",
        f"- JSON 结果：`{result['outputs']['result_json']}`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    candidates = prepare_candidates()
    selected = select_teachers(candidates)
    if len(selected) != len(YEARS):
        raise RuntimeError(f"Expected {len(YEARS)} selected teachers, got {len(selected)}")
    daily = build_imitation_rows(selected)
    action_days = daily[daily["is_nonzero_action_day"]].copy()
    action_dist = (
        daily.groupby(["teacher_tier", "teacher_action_index", "requested_irrigation_mm_action", "requested_nitrogen_kg_ha_action"], as_index=False)
        .agg(samples=("teacher_action_index", "count"), weighted_samples=("sample_weight", "sum"))
        .sort_values(["teacher_tier", "teacher_action_index"])
    )

    selected_path = OUT / "tables" / "041_02_selected_teacher_trajectories.csv"
    daily_path = OUT / "tables" / "041_02_imitation_daily_dataset.csv"
    action_days_path = OUT / "tables" / "041_02_imitation_action_days.csv"
    action_dist_path = OUT / "tables" / "041_02_action_distribution.csv"
    action_grid_path = OUT / "configs" / "041_02_action_grid.csv"
    selected.to_csv(selected_path, index=False)
    daily.to_csv(daily_path, index=False)
    action_days.to_csv(action_days_path, index=False)
    action_dist.to_csv(action_dist_path, index=False)
    pd.DataFrame(ACTION_GRID).to_csv(action_grid_path, index=False)

    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": "A_layered_teacher_dataset_ready",
        "training_run": False,
        "selected_year_count": int(len(selected)),
        "strong_teacher_year_count": int(selected["teacher_tier"].eq("strong_all3").sum()),
        "near_miss_teacher_year_count": int(selected["teacher_tier"].eq("near_miss").sum()),
        "daily_sample_count": int(len(daily)),
        "nonzero_action_sample_count": int(len(action_days)),
        "outputs": {
            "selected_teachers": selected_path.relative_to(ROOT).as_posix(),
            "imitation_daily_dataset": daily_path.relative_to(ROOT).as_posix(),
            "imitation_action_days": action_days_path.relative_to(ROOT).as_posix(),
            "action_distribution": action_dist_path.relative_to(ROOT).as_posix(),
            "action_grid": action_grid_path.relative_to(ROOT).as_posix(),
            "result_json": (OUT / "041_02_result.json").relative_to(ROOT).as_posix(),
            "record_md": DOC.relative_to(ROOT).as_posix(),
        },
    }
    (OUT / "041_02_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_record(result, selected, daily, action_dist)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
