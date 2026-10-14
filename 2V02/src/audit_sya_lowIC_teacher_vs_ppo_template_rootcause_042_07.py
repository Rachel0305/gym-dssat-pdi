"""042_07: Audit whether PPO template-like behavior comes from teacher data.

This script is intentionally offline-only:
- no training
- no DSSAT call
- no input-data modification

It compares the 041_02 teacher imitation trajectories with the 042_04/042_05
PPO checkpoint action-diversity audits.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "042_07"
TASK_NAME = "sya_lowIC_teacher_vs_ppo_template_rootcause_audit"
OUTPUT_ROOT = Path("benchmark_results") / f"{TASK_ID}_{TASK_NAME}"
TABLE_DIR = OUTPUT_ROOT / "tables"
DOC_PATH = Path("docs") / f"{TASK_ID}_{TASK_NAME}_record.md"

TEACHER_DIR = Path("benchmark_results/041_02_sya_lowIC_layered_teacher_imitation_dataset/tables")
PPO04204_DIR = Path("benchmark_results/042_04_sya_lowIC_04202_weather_responsiveness_audit/tables")
PPO04205_DIR = Path("benchmark_results/042_05_sya_lowIC_04202_checkpoint_weather_response_guardrail/tables")


@dataclass
class AuditResult:
    task: str
    branch: str
    next_step_allowed: bool
    teacher_years: int | None
    teacher_unique_action_signatures: int | None
    teacher_sd_total_irrigation: float | None
    teacher_sd_total_nitrogen: float | None
    teacher_mean_irrigation_event_dap_sd: float | None
    teacher_mean_nitrogen_event_dap_sd: float | None
    ppo75_unique_action_signatures: int | None
    ppo75_sd_total_irrigation: float | None
    ppo75_sd_total_nitrogen: float | None
    ppo75_mean_irrigation_event_dap_sd: float | None
    ppo75_mean_nitrogen_event_dap_sd: float | None
    record_md: str
    outputs: dict[str, str]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input missing: {path}")
    return pd.read_csv(path)


def event_signature(events: pd.DataFrame) -> str:
    if events.empty:
        return "NO_NONZERO_ACTION"
    parts: list[str] = []
    for row in events.sort_values(["dap_before_action", "step"]).itertuples(index=False):
        dap = getattr(row, "dap_before_action")
        irr = getattr(row, "requested_irrigation_mm_action")
        n = getattr(row, "requested_nitrogen_kg_ha_action")
        parts.append(f"DAP{int(round(float(dap)))}:I{float(irr):g}/N{float(n):g}")
    return "; ".join(parts)


def summarize_event_positions(events: pd.DataFrame, action_type: str) -> pd.DataFrame:
    if action_type == "irrigation":
        sub = events[events["requested_irrigation_mm_action"] > 0].copy()
        amount_col = "requested_irrigation_mm_action"
    elif action_type == "nitrogen":
        sub = events[events["requested_nitrogen_kg_ha_action"] > 0].copy()
        amount_col = "requested_nitrogen_kg_ha_action"
    else:
        raise ValueError(action_type)

    rows: list[dict[str, Any]] = []
    if sub.empty:
        return pd.DataFrame(columns=[
            "source", "action_type", "event_order", "n_years_with_event",
            "mean_dap", "sd_dap", "min_dap", "max_dap", "mean_amount", "sd_amount",
        ])

    sub = sub.sort_values(["year", "dap_before_action", "step"]).copy()
    sub["event_order"] = sub.groupby("year").cumcount() + 1
    for order, g in sub.groupby("event_order"):
        rows.append(
            {
                "source": "teacher_041_02",
                "action_type": action_type,
                "event_order": int(order),
                "n_years_with_event": int(g["year"].nunique()),
                "mean_dap": float(g["dap_before_action"].mean()),
                "sd_dap": float(g["dap_before_action"].std(ddof=0)),
                "min_dap": float(g["dap_before_action"].min()),
                "max_dap": float(g["dap_before_action"].max()),
                "mean_amount": float(g[amount_col].mean()),
                "sd_amount": float(g[amount_col].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def build_teacher_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    selected = read_required_csv(TEACHER_DIR / "041_02_selected_teacher_trajectories.csv")
    events = read_required_csv(TEACHER_DIR / "041_02_imitation_action_days.csv")

    year_rows: list[dict[str, Any]] = []
    for year, g in events.groupby("year"):
        nonzero = g[
            (g["requested_irrigation_mm_action"] > 0)
            | (g["requested_nitrogen_kg_ha_action"] > 0)
        ].copy()
        irr = nonzero[nonzero["requested_irrigation_mm_action"] > 0]
        nit = nonzero[nonzero["requested_nitrogen_kg_ha_action"] > 0]
        year_rows.append(
            {
                "year": int(year),
                "teacher_action_signature": event_signature(nonzero),
                "teacher_irrigation_signature": event_signature(irr),
                "teacher_nitrogen_signature": event_signature(nit),
                "teacher_irrigation_events": int(len(irr)),
                "teacher_nitrogen_events": int(len(nit)),
                "teacher_total_irrigation": float(g["requested_irrigation_mm_action"].sum()),
                "teacher_total_nitrogen": float(g["requested_nitrogen_kg_ha_action"].sum()),
            }
        )
    teacher_by_year = pd.DataFrame(year_rows)

    keep_cols = [
        "year", "candidate_id", "water_schedule_id", "n_schedule_id",
        "grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "teacher_tier",
    ]
    teacher_by_year = teacher_by_year.merge(selected[keep_cols], on="year", how="left")

    event_pos = pd.concat(
        [
            summarize_event_positions(events, "irrigation"),
            summarize_event_positions(events, "nitrogen"),
        ],
        ignore_index=True,
    )

    irr_pos = event_pos[event_pos["action_type"] == "irrigation"]
    nit_pos = event_pos[event_pos["action_type"] == "nitrogen"]
    stats = {
        "teacher_years": int(teacher_by_year["year"].nunique()),
        "teacher_unique_action_signatures": int(teacher_by_year["teacher_action_signature"].nunique()),
        "teacher_unique_irrigation_signatures": int(teacher_by_year["teacher_irrigation_signature"].nunique()),
        "teacher_unique_nitrogen_signatures": int(teacher_by_year["teacher_nitrogen_signature"].nunique()),
        "teacher_sd_total_irrigation": float(teacher_by_year["teacher_total_irrigation"].std(ddof=0)),
        "teacher_sd_total_nitrogen": float(teacher_by_year["teacher_total_nitrogen"].std(ddof=0)),
        "teacher_mean_irrigation_event_dap_sd": float(irr_pos["sd_dap"].mean()) if not irr_pos.empty else 0.0,
        "teacher_mean_nitrogen_event_dap_sd": float(nit_pos["sd_dap"].mean()) if not nit_pos.empty else 0.0,
    }
    return teacher_by_year, event_pos, stats


def build_ppo75_stats() -> dict[str, float | int | None]:
    guard = read_required_csv(PPO04205_DIR / "042_05_checkpoint_response_guardrail_summary.csv")
    row = guard[guard["checkpoint_step"] == 75000]
    if row.empty:
        return {
            "ppo75_unique_action_signatures": None,
            "ppo75_sd_total_irrigation": None,
            "ppo75_sd_total_nitrogen": None,
            "ppo75_mean_irrigation_event_dap_sd": None,
            "ppo75_mean_nitrogen_event_dap_sd": None,
        }
    r = row.iloc[0]
    return {
        "ppo75_unique_action_signatures": int(r["unique_nonzero_action_signatures"]),
        "ppo75_sd_total_irrigation": float(r["sd_total_irrigation"]),
        "ppo75_sd_total_nitrogen": float(r["sd_total_nitrogen"]),
        "ppo75_mean_irrigation_event_dap_sd": float(r["mean_irrigation_event_dap_sd"]),
        "ppo75_mean_nitrogen_event_dap_sd": float(r["mean_nitrogen_event_dap_sd"]),
    }


def decide_branch(teacher: dict[str, float | int], ppo: dict[str, float | int | None]) -> tuple[str, bool]:
    missing_ppo = any(v is None for v in ppo.values())
    if missing_ppo:
        return "C_insufficient_existing_evidence", False

    teacher_template_like = (
        int(teacher["teacher_unique_action_signatures"]) <= 1
        or (
            float(teacher["teacher_sd_total_irrigation"]) < 1e-9
            and float(teacher["teacher_sd_total_nitrogen"]) < 1e-9
        )
    )
    ppo_template_like = (
        int(ppo["ppo75_unique_action_signatures"] or 0) <= 1
        and float(ppo["ppo75_sd_total_irrigation"] or 0) < 1e-9
        and float(ppo["ppo75_sd_total_nitrogen"] or 0) < 1e-9
    )
    if teacher_template_like:
        return "A_teacher_template_like", False
    if ppo_template_like:
        return "B_ppo_collapsed_diverse_teacher", True
    return "C_insufficient_existing_evidence", False


def write_record(result: AuditResult, teacher_by_year: pd.DataFrame, comparison: pd.DataFrame) -> None:
    branch_explain = {
        "A_teacher_template_like": "teacher 数据本身模板化，下一步应先重建 teacher，不应直接改 PPO。",
        "B_ppo_collapsed_diverse_teacher": "teacher 跨年有差异，但 PPO 75K 压成模板。下一步应修改训练目标/BC 约束，使 PPO 保留或学习天气响应性。",
        "C_insufficient_existing_evidence": "现有证据不足，不能进入训练修改。",
    }
    lines = [
        f"# {TASK_ID} SYA lowIC teacher-vs-PPO 模板化根因审计",
        "",
        "## 结论",
        "",
        f"- 分支：`{result.branch}`",
        f"- 解释：{branch_explain.get(result.branch, '')}",
        f"- next_step_allowed：`{str(result.next_step_allowed).lower()}`",
        "",
        "## 关键数字",
        "",
        f"- teacher 年份数：{result.teacher_years}",
        f"- teacher 非零动作签名唯一数：{result.teacher_unique_action_signatures}",
        f"- teacher 总灌溉标准差：{result.teacher_sd_total_irrigation:.3f}",
        f"- teacher 总施氮标准差：{result.teacher_sd_total_nitrogen:.3f}",
        f"- teacher 灌溉事件 DAP 平均标准差：{result.teacher_mean_irrigation_event_dap_sd:.3f}",
        f"- teacher 施肥事件 DAP 平均标准差：{result.teacher_mean_nitrogen_event_dap_sd:.3f}",
        f"- PPO 75K 非零动作签名唯一数：{result.ppo75_unique_action_signatures}",
        f"- PPO 75K 总灌溉标准差：{result.ppo75_sd_total_irrigation:.3f}",
        f"- PPO 75K 总施氮标准差：{result.ppo75_sd_total_nitrogen:.3f}",
        f"- PPO 75K 灌溉事件 DAP 平均标准差：{result.ppo75_mean_irrigation_event_dap_sd:.3f}",
        f"- PPO 75K 施肥事件 DAP 平均标准差：{result.ppo75_mean_nitrogen_event_dap_sd:.3f}",
        "",
        "## 解释边界",
        "",
        "- 本任务只做离线审计，不训练模型，不调用 DSSAT。",
        "- 本任务不能证明具体哪一个 loss 项导致模板化，只能排除“teacher 本身完全模板化”这个解释。",
        "- 若后续进入训练修改，应作为新任务单独预注册，不应把本任务当作调参结果。",
        "",
        "## 输出文件",
        "",
    ]
    for key, path in result.outputs.items():
        lines.append(f"- {key}: `{path}`")
    lines += [
        "",
        "## teacher 年度概览（前 10 行）",
        "",
        teacher_by_year[
            [
                "year", "candidate_id", "teacher_total_irrigation", "teacher_total_nitrogen",
                "teacher_irrigation_events", "teacher_nitrogen_events", "teacher_tier",
            ]
        ].to_markdown(index=False),
        "",
        "## teacher vs PPO 75K 对比",
        "",
        comparison.to_markdown(index=False),
        "",
    ]
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    try:
        teacher_by_year, teacher_event_pos, teacher_stats = build_teacher_tables()
        ppo75_stats = build_ppo75_stats()
        branch, next_allowed = decide_branch(teacher_stats, ppo75_stats)
    except Exception as exc:
        outputs: dict[str, str] = {}
        result = AuditResult(
            task=f"{TASK_ID}_{TASK_NAME}",
            branch="C_insufficient_existing_evidence",
            next_step_allowed=False,
            teacher_years=None,
            teacher_unique_action_signatures=None,
            teacher_sd_total_irrigation=None,
            teacher_sd_total_nitrogen=None,
            teacher_mean_irrigation_event_dap_sd=None,
            teacher_mean_nitrogen_event_dap_sd=None,
            ppo75_unique_action_signatures=None,
            ppo75_sd_total_irrigation=None,
            ppo75_sd_total_nitrogen=None,
            ppo75_mean_irrigation_event_dap_sd=None,
            ppo75_mean_nitrogen_event_dap_sd=None,
            record_md=str(DOC_PATH),
            outputs=outputs,
        )
        DOC_PATH.write_text(f"# {TASK_ID} failed\n\n{type(exc).__name__}: {exc}\n", encoding="utf-8")
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return

    teacher_by_year_path = TABLE_DIR / f"{TASK_ID}_teacher_action_diversity_by_year.csv"
    teacher_event_pos_path = TABLE_DIR / f"{TASK_ID}_teacher_event_position_variability.csv"
    comparison_path = TABLE_DIR / f"{TASK_ID}_teacher_vs_ppo75k_template_metrics.csv"
    result_path = OUTPUT_ROOT / f"{TASK_ID}_result.json"

    teacher_by_year.to_csv(teacher_by_year_path, index=False, encoding="utf-8-sig")
    teacher_event_pos.to_csv(teacher_event_pos_path, index=False, encoding="utf-8-sig")

    comparison = pd.DataFrame(
        [
            {
                "source": "teacher_041_02",
                "unique_nonzero_action_signatures": teacher_stats["teacher_unique_action_signatures"],
                "sd_total_irrigation": teacher_stats["teacher_sd_total_irrigation"],
                "sd_total_nitrogen": teacher_stats["teacher_sd_total_nitrogen"],
                "mean_irrigation_event_dap_sd": teacher_stats["teacher_mean_irrigation_event_dap_sd"],
                "mean_nitrogen_event_dap_sd": teacher_stats["teacher_mean_nitrogen_event_dap_sd"],
            },
            {
                "source": "ppo_042_02_checkpoint_75k",
                "unique_nonzero_action_signatures": ppo75_stats["ppo75_unique_action_signatures"],
                "sd_total_irrigation": ppo75_stats["ppo75_sd_total_irrigation"],
                "sd_total_nitrogen": ppo75_stats["ppo75_sd_total_nitrogen"],
                "mean_irrigation_event_dap_sd": ppo75_stats["ppo75_mean_irrigation_event_dap_sd"],
                "mean_nitrogen_event_dap_sd": ppo75_stats["ppo75_mean_nitrogen_event_dap_sd"],
            },
        ]
    )
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    outputs = {
        "teacher_action_diversity_by_year": str(teacher_by_year_path),
        "teacher_event_position_variability": str(teacher_event_pos_path),
        "teacher_vs_ppo75k_template_metrics": str(comparison_path),
        "result_json": str(result_path),
    }
    result = AuditResult(
        task=f"{TASK_ID}_{TASK_NAME}",
        branch=branch,
        next_step_allowed=next_allowed,
        teacher_years=int(teacher_stats["teacher_years"]),
        teacher_unique_action_signatures=int(teacher_stats["teacher_unique_action_signatures"]),
        teacher_sd_total_irrigation=float(teacher_stats["teacher_sd_total_irrigation"]),
        teacher_sd_total_nitrogen=float(teacher_stats["teacher_sd_total_nitrogen"]),
        teacher_mean_irrigation_event_dap_sd=float(teacher_stats["teacher_mean_irrigation_event_dap_sd"]),
        teacher_mean_nitrogen_event_dap_sd=float(teacher_stats["teacher_mean_nitrogen_event_dap_sd"]),
        ppo75_unique_action_signatures=int(ppo75_stats["ppo75_unique_action_signatures"]),
        ppo75_sd_total_irrigation=float(ppo75_stats["ppo75_sd_total_irrigation"]),
        ppo75_sd_total_nitrogen=float(ppo75_stats["ppo75_sd_total_nitrogen"]),
        ppo75_mean_irrigation_event_dap_sd=float(ppo75_stats["ppo75_mean_irrigation_event_dap_sd"]),
        ppo75_mean_nitrogen_event_dap_sd=float(ppo75_stats["ppo75_mean_nitrogen_event_dap_sd"]),
        record_md=str(DOC_PATH),
        outputs=outputs,
    )
    write_record(result, teacher_by_year, comparison)
    result_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
