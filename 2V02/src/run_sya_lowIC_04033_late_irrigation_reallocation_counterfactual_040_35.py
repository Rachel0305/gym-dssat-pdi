from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_sya_lowIC_ppo_i240_swfac_guardrail_coarse_actions_040_33 as base04033
import run_sya_lowIC_ppo_irrigation_reallocation_counterfactual_040_07 as replay04007


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "040_35"
TASK_NAME = "sya_lowIC_04033_late_irrigation_reallocation_counterfactual"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

LOWIC_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
PPO04033_EVAL = (
    ROOT
    / "benchmark_results"
    / "040_33_sya_lowIC_ppo_i240_swfac_guardrail_coarse_actions"
    / "evaluation"
    / "040_33_checkpoint_validation_summary.csv"
)

STATION = "SYA"
SEED = 0
PPO_CHECKPOINT = 25_000
YEARS = [2014, 2017, 2022]
MOVE_TARGET_DAPS = [96, 103, 110]
MOVE_SOURCE_MIN_DAP = 31
MOVE_SOURCE_MAX_DAP = 90
MOVE_AMOUNT = 45.0
STRESS_THRESHOLD = 0.05


def ensure_dirs() -> None:
    for rel in ["configs", "tables", "daily_outputs", "snapshots"]:
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


def patch_replay_module() -> None:
    replay04007.TASK_ID = TASK_ID
    replay04007.TASK_NAME = TASK_NAME
    replay04007.OUT = OUT
    replay04007.DOC = DOC
    replay04007.PROMPT = PROMPT
    replay04007.LOWIC_INPUT_ROOT = LOWIC_INPUT_ROOT
    replay04007.STATION = STATION
    replay04007.SEED = SEED
    replay04007.PPO_CHECKPOINT = PPO_CHECKPOINT
    replay04007.YEARS = YEARS
    replay04007.STRESS_THRESHOLD = STRESS_THRESHOLD


def load_config_and_env_config() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    config = base04033.load_config()
    split = base03222.load_split()
    split = split[split["station_code"].eq(STATION)].copy()
    selection = base03222.build_selection(split)
    config["runtime"]["smoke_station"] = STATION
    config["paths"]["output_root"] = OUT.relative_to(ROOT).as_posix()

    old_out = direct_ppo.OUTPUT_ROOT
    old_input = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    try:
        direct_ppo.OUTPUT_ROOT = OUT
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_INPUT_ROOT
        env_config = direct_ppo.build_env_config(config, selection)
    finally:
        direct_ppo.OUTPUT_ROOT = old_out
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input
    return config, env_config, selection


def load_ppo_rows() -> pd.DataFrame:
    df = pd.read_csv(PPO04033_EVAL, keep_default_na=False)
    rows = df[
        (df["station_code"].eq(STATION))
        & (pd.to_numeric(df["checkpoint_step"], errors="coerce").astype("Int64").eq(PPO_CHECKPOINT))
        & (pd.to_numeric(df["year"], errors="coerce").astype("Int64").isin(YEARS))
    ].copy()
    if len(rows) != len(YEARS):
        raise RuntimeError(f"缺少 040_33 checkpoint {PPO_CHECKPOINT} 的年份行：expected {len(YEARS)}, got {len(rows)}")
    return rows.sort_values("year").reset_index(drop=True)


def choose_source_irrigation_dap(actions: dict[int, tuple[float, float]]) -> int:
    candidates = [
        dap
        for dap, (irrigation, _n) in actions.items()
        if MOVE_SOURCE_MIN_DAP <= dap <= MOVE_SOURCE_MAX_DAP and abs(irrigation - MOVE_AMOUNT) < 1e-9
    ]
    if not candidates:
        raise RuntimeError(f"没有找到 DAP{MOVE_SOURCE_MIN_DAP}-{MOVE_SOURCE_MAX_DAP} 内 {MOVE_AMOUNT:g} mm 灌溉事件，不能做后移反事实。")
    return max(candidates)


def move_one_irrigation(
    actions: dict[int, tuple[float, float]],
    source_dap: int,
    target_dap: int,
    config: dict[str, Any],
) -> tuple[dict[int, tuple[float, float]], dict[str, Any]]:
    out = dict(actions)
    old_i, old_n = out[source_dap]
    if abs(old_i - MOVE_AMOUNT) > 1e-9:
        raise RuntimeError(f"DAP{source_dap} 不是 {MOVE_AMOUNT:g} mm 灌溉事件：I={old_i}")

    if old_n > 0:
        out[source_dap] = (0.0, old_n)
    else:
        out.pop(source_dap, None)

    min_gap = int(config["action_safety"]["min_days_between_irrigation"])
    dap_min, dap_max = config["action_safety"]["irrigation_allowed_dap_range"]
    chosen_dap: int | None = None
    for dap in range(max(int(target_dap), int(dap_min)), int(dap_max) + 1):
        existing_irrigation_daps = sorted(d for d, (i, _n) in out.items() if i > 1e-9)
        if any(abs(dap - old) < min_gap for old in existing_irrigation_daps):
            continue
        current_i, current_n = out.get(dap, (0.0, 0.0))
        if current_i + MOVE_AMOUNT > max(base04033.IRRIGATION_LEVELS) + 1e-9:
            continue
        out[dap] = (current_i + MOVE_AMOUNT, current_n)
        chosen_dap = dap
        break

    if chosen_dap is None:
        raise RuntimeError(f"无法从 DAP{target_dap} 起找到合法后移日期。")

    edit = {
        "source_dap": source_dap,
        "target_requested_dap": target_dap,
        "target_actual_dap": chosen_dap,
        "moved_irrigation_mm": MOVE_AMOUNT,
        "source_had_n_kg_ha": old_n,
    }
    return out, edit


def action_sequence(actions: dict[int, tuple[float, float]]) -> str:
    return "; ".join(
        f"DAP{dap} I{i:g}/N{n:g}"
        for dap, (i, n) in sorted(actions.items())
        if i > 1e-9 or n > 1e-9
    )


def build_plan(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[int, str], dict[int, tuple[float, float]]]]:
    plan_rows: list[dict[str, Any]] = []
    edit_rows: list[dict[str, Any]] = []
    schedules: dict[tuple[int, str], dict[int, tuple[float, float]]] = {}

    for _, row in load_ppo_rows().iterrows():
        year = int(row["year"])
        original = replay04007.original_actions_by_dap(str(row["daily_csv_path"]))
        source_dap = choose_source_irrigation_dap(original)
        branch_schedules: list[tuple[str, dict[int, tuple[float, float]], dict[str, Any] | None]] = [
            ("original_replay", original, None)
        ]
        for target in MOVE_TARGET_DAPS:
            moved, edit = move_one_irrigation(original, source_dap, target, config)
            branch = f"move_DAP{source_dap}_to_{target}"
            branch_schedules.append((branch, moved, edit))
            edit_rows.append({"station_code": STATION, "year": year, "branch": branch, **edit})

        for branch, actions, edit in branch_schedules:
            schedules[(year, branch)] = actions
            plan_rows.append(
                {
                    "station_code": STATION,
                    "year": year,
                    "branch": branch,
                    "ppo_source": "040_33 checkpoint 25k",
                    "source_move_dap": source_dap if branch != "original_replay" else "",
                    "target_requested_dap": "" if edit is None else edit["target_requested_dap"],
                    "target_actual_dap": "" if edit is None else edit["target_actual_dap"],
                    "planned_total_irrigation": sum(i for i, _n in actions.values()),
                    "planned_total_n": sum(n for _i, n in actions.values()),
                    "irrigation_dap1_30": sum(i for dap, (i, _n) in actions.items() if dap <= 30),
                    "irrigation_dap31_60": sum(i for dap, (i, _n) in actions.items() if 31 <= dap <= 60),
                    "irrigation_dap61_90": sum(i for dap, (i, _n) in actions.items() if 61 <= dap <= 90),
                    "irrigation_dap91_plus": sum(i for dap, (i, _n) in actions.items() if dap >= 91),
                    "action_sequence": action_sequence(actions),
                }
            )

    return pd.DataFrame(plan_rows), pd.DataFrame(edit_rows), schedules


def write_record(plan: pd.DataFrame, edits: pd.DataFrame, summary: pd.DataFrame | None = None) -> None:
    lines = [
        "# 040_35 SYA lowIC 040_33 后期灌溉重分配反事实记录",
        "",
        "## 先说结论",
        "",
    ]
    if summary is None or summary.empty:
        lines.append("- 已生成预注册动作计划，尚未运行 DSSAT。")
    else:
        best2017 = summary[(summary["year"].eq(2017)) & (~summary["branch"].eq("original_replay"))].copy()
        if not best2017.empty:
            best2017 = best2017.sort_values(["delta_yield_vs_original_replay", "delta_swfac_days_gt_0p05_vs_original_replay"], ascending=[False, True])
            row = best2017.iloc[0]
            lines.append(
                f"- 2017 最优后移分支为 `{row['branch']}`：产量变化 {float(row['delta_yield_vs_original_replay']):.2f} kg/ha，"
                f"SWFAC>0.05 天数变化 {int(row['delta_swfac_days_gt_0p05_vs_original_replay'])} 天。"
            )
        lines.append("- 本任务未训练模型，只是固定日程 DSSAT 反事实，结果只能作为下一轮 PPO 约束/reward 的依据。")

    lines.extend(
        [
            "",
            "## 预注册动作计划",
            "",
            md_table(plan),
            "",
            "## 动作编辑明细",
            "",
            md_table(edits),
        ]
    )
    if summary is not None and not summary.empty:
        lines.extend(["", "## DSSAT 反事实结果", "", md_table(summary)])
    lines.extend(
        [
            "",
            "## 判读边界",
            "",
            "- 如果后移分支改善 2017 的产量和后期水分胁迫，说明 040_33 的主要问题可能是水分预算过早用完。",
            "- 如果后移分支不改善，则不应继续围绕简单后移灌溉去调 PPO。",
        ]
    )
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def value_from_row(row: pd.Series, *names: str, default: float = np.nan) -> float:
    for name in names:
        if name in row.index:
            value = pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0]
            return float(value) if pd.notna(value) else default
    return default


def run_dry() -> None:
    ensure_dirs()
    patch_replay_module()
    config, _env_config, _selection = load_config_and_env_config()
    plan, edits, _schedules = build_plan(config)
    plan.to_csv(OUT / "tables" / f"{TASK_ID}_action_plan_dry_run.csv", index=False, encoding="utf-8-sig")
    edits.to_csv(OUT / "tables" / f"{TASK_ID}_action_edits_dry_run.csv", index=False, encoding="utf-8-sig")
    write_record(plan, edits)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "mode": "dry_run",
        "next_step_allowed": True,
        "action_plan": (OUT / "tables" / f"{TASK_ID}_action_plan_dry_run.csv").relative_to(ROOT).as_posix(),
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / f"{TASK_ID}_dry_run_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_full() -> None:
    ensure_dirs()
    patch_replay_module()
    if not PROMPT.exists():
        raise FileNotFoundError(PROMPT)
    if not LOWIC_INPUT_ROOT.exists():
        raise FileNotFoundError(LOWIC_INPUT_ROOT)
    if not PPO04033_EVAL.exists():
        raise FileNotFoundError(PPO04033_EVAL)

    config, env_config, selection = load_config_and_env_config()
    shutil.copyfile(PROMPT, OUT / "configs" / PROMPT.name)
    direct_ppo.write_yaml(env_config, OUT / "configs" / f"{TASK_ID}_resolved_env_config.yaml")
    selection.to_csv(OUT / "configs" / f"{TASK_ID}_selection.csv", index=False, encoding="utf-8-sig")

    plan, edits, schedules = build_plan(config)
    plan.to_csv(OUT / "tables" / f"{TASK_ID}_action_plan.csv", index=False, encoding="utf-8-sig")
    edits.to_csv(OUT / "tables" / f"{TASK_ID}_action_edits.csv", index=False, encoding="utf-8-sig")

    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for year in YEARS:
        branches = [key[1] for key in schedules if key[0] == year]
        for branch in branches:
            daily, summary = replay04007.replay_fixed_schedule(config, env_config, year, branch, schedules[(year, branch)])
            daily_frames.append(daily)
            summary_rows.append(summary)

    summary = pd.DataFrame(summary_rows)
    delta_rows: list[dict[str, Any]] = []
    for year, group in summary.groupby("year"):
        orig = group[group["branch"].eq("original_replay")].iloc[0]
        for _, row in group.iterrows():
            out = row.to_dict()
            out["delta_yield_vs_original_replay"] = float(row["final_grnwt"]) - float(orig["final_grnwt"])
            out["delta_swfac_days_gt_0p05_vs_original_replay"] = int(row["swfac_days_gt_0p05"]) - int(orig["swfac_days_gt_0p05"])
            out["delta_max_swfac_vs_original_replay"] = float(row["max_swfac"]) - float(orig["max_swfac"])
            out["delta_total_irrigation_vs_original_replay"] = value_from_row(row, "total_irrigation") - value_from_row(orig, "total_irrigation")
            out["delta_total_n_vs_original_replay"] = value_from_row(row, "total_nitrogen", "total_n") - value_from_row(orig, "total_nitrogen", "total_n")
            delta_rows.append(out)
    summary = pd.DataFrame(delta_rows)
    summary.to_csv(OUT / "tables" / f"{TASK_ID}_counterfactual_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily_frames, ignore_index=True).to_csv(OUT / "tables" / f"{TASK_ID}_counterfactual_daily.csv", index=False, encoding="utf-8-sig")
    write_record(plan, edits, summary)

    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": "completed",
        "years": YEARS,
        "summary_csv": (OUT / "tables" / f"{TASK_ID}_counterfactual_summary.csv").relative_to(ROOT).as_posix(),
        "daily_csv": (OUT / "tables" / f"{TASK_ID}_counterfactual_daily.csv").relative_to(ROOT).as_posix(),
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / f"{TASK_ID}_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="040_35 SYA lowIC 040_33 late irrigation reallocation counterfactual")
    parser.add_argument("--dry-run", action="store_true", help="只写动作计划，不运行 DSSAT。")
    args = parser.parse_args()
    if args.dry_run:
        run_dry()
    else:
        run_full()


if __name__ == "__main__":
    main()
