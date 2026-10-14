from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "042_06"
TASK_NAME = "sya_lowIC_weather_response_checkpoint_selection"
SRC = ROOT / "benchmark_results" / "042_05_sya_lowIC_04202_checkpoint_weather_response_guardrail"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
TABLES = OUT / "tables"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

GUARDRAIL = SRC / "tables" / "042_05_checkpoint_response_guardrail_summary.csv"
ACTION_SEQUENCE = SRC / "tables" / "042_05_action_sequence_by_year_checkpoint.csv"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size <= 0:
        raise ValueError(f"Empty input table: {path}")
    return pd.read_csv(path, keep_default_na=False)


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
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


def annotate_passes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric(
        out,
        [
            "checkpoint_step",
            "any_metric_win_years",
            "all3_win_years",
            "unique_nonzero_action_signatures",
            "mean_irrigation_event_dap_sd",
            "irrigation_future7_rain_abs_std_diff",
            "mean_nitrogen_event_dap_sd",
            "nitrogen_nstres_abs_std_diff",
            "mean_yield",
            "mean_wp_et",
            "mean_pfp_n",
        ],
    )
    out["endpoint_pass_04206"] = out["any_metric_win_years"].eq(10) & out["all3_win_years"].ge(5)
    out["non_template_pass_04206"] = out["unique_nonzero_action_signatures"].ge(3)
    out["irrigation_response_pass_04206"] = (
        out["mean_irrigation_event_dap_sd"].ge(3)
        | out["irrigation_future7_rain_abs_std_diff"].ge(0.2)
    )
    out["nitrogen_response_pass_04206"] = (
        out["mean_nitrogen_event_dap_sd"].ge(3)
        | out["nitrogen_nstres_abs_std_diff"].ge(0.2)
    )
    out["selected_candidate_pass_04206"] = (
        out["endpoint_pass_04206"]
        & out["non_template_pass_04206"]
        & out["irrigation_response_pass_04206"]
        & out["nitrogen_response_pass_04206"]
    )
    return out


def failure_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    if not to_bool(row["endpoint_pass_04206"]):
        reasons.append("终点指标未通过")
    if not to_bool(row["non_template_pass_04206"]):
        reasons.append("非零动作模板化")
    if not to_bool(row["irrigation_response_pass_04206"]):
        reasons.append("灌溉天气响应不足")
    if not to_bool(row["nitrogen_response_pass_04206"]):
        reasons.append("施肥胁迫响应不足")
    return "；".join(reasons) if reasons else "通过"


def choose_candidate(table: pd.DataFrame) -> pd.DataFrame:
    passed = table[table["selected_candidate_pass_04206"]].copy()
    if passed.empty:
        return passed
    return passed.sort_values(
        [
            "all3_win_years",
            "any_metric_win_years",
            "unique_nonzero_action_signatures",
            "checkpoint_step",
        ],
        ascending=[False, False, False, True],
    ).head(1)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if not PROMPT.exists():
        raise FileNotFoundError(PROMPT)

    guard = read_csv(GUARDRAIL)
    _seq = read_csv(ACTION_SEQUENCE)
    table = annotate_passes(guard)
    table["failure_reason_04206"] = table.apply(failure_reasons, axis=1)

    selected = choose_candidate(table)
    if selected.empty:
        branch = "C_no_checkpoint_selected_under_weather_response_guardrail"
        selected_checkpoint = None
    else:
        branch = "A_checkpoint_selected_under_weather_response_guardrail"
        selected_checkpoint = int(selected["checkpoint_step"].iloc[0])

    selection_cols = [
        "checkpoint_step",
        "any_metric_win_years",
        "all3_win_years",
        "mean_yield",
        "mean_wp_et",
        "mean_pfp_n",
        "unique_nonzero_action_signatures",
        "mean_irrigation_event_dap_sd",
        "irrigation_future7_rain_abs_std_diff",
        "mean_nitrogen_event_dap_sd",
        "nitrogen_nstres_abs_std_diff",
        "endpoint_pass_04206",
        "non_template_pass_04206",
        "irrigation_response_pass_04206",
        "nitrogen_response_pass_04206",
        "selected_candidate_pass_04206",
        "failure_reason_04206",
    ]
    selection_table = table[selection_cols].sort_values("checkpoint_step")
    failure_table = selection_table[~selection_table["selected_candidate_pass_04206"]].copy()

    selection_path = TABLES / "042_06_checkpoint_selection_table.csv"
    failure_path = TABLES / "042_06_checkpoint_failure_reasons.csv"
    selection_table.to_csv(selection_path, index=False, encoding="utf-8-sig")
    failure_table.to_csv(failure_path, index=False, encoding="utf-8-sig")

    lines = [
        "# 042_06 SYA lowIC 天气响应性 checkpoint 选择审计记录",
        "",
        "## 任务性质",
        "",
        "- 只读取 042_05 guardrail 表和 action sequence 表。",
        "- 不训练 PPO，不运行 DSSAT，不修改 checkpoint。",
        "- 固定沿用 042_05 的天气响应性门槛，不现场降低标准。",
        "",
        "## 选择规则",
        "",
        "checkpoint 必须同时满足：",
        "",
        "1. `any_metric_win_years == 10` 且 `all3_win_years >= 5`；",
        "2. `unique_nonzero_action_signatures >= 3`；",
        "3. `mean_irrigation_event_dap_sd >= 3` 或 `irrigation_future7_rain_abs_std_diff >= 0.2`；",
        "4. `mean_nitrogen_event_dap_sd >= 3` 或 `nitrogen_nstres_abs_std_diff >= 0.2`。",
        "",
        "## 选择结果表",
        "",
        md_table(selection_table, max_rows=20),
        "",
        "## 判定",
        "",
        f"- 分支：`{branch}`",
        f"- 选中 checkpoint：`{selected_checkpoint}`",
    ]
    if selected.empty:
        lines.extend(
            [
                "- 结论：当前 042_02 的所有 checkpoint 中，没有一个可以同时作为“高指标 + 非模板化天气响应”的最终候选。",
                "- 75K 仍可作为阶段性“指标型候选”保留，但不能被表述为已经学会天气自适应管理。",
                "- 下一步应修改训练目标、teacher 数据或模型选择机制，让天气/土壤响应性进入学习过程，而不是继续硬挑 checkpoint。",
            ]
        )
    else:
        lines.extend(
            [
                "- 结论：存在满足响应性 guardrail 的候选，可进入下一步五情景图和反事实合理性检查。",
            ]
        )
    lines.extend(
        [
            "",
            "## 输出",
            "",
            f"- selection table: `{selection_path.relative_to(ROOT)}`",
            f"- failure reasons: `{failure_path.relative_to(ROOT)}`",
        ]
    )
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": branch,
        "selected_checkpoint": selected_checkpoint,
        "candidate_count": int(table["selected_candidate_pass_04206"].sum()),
        "outputs": {
            "record_md": str(DOC.relative_to(ROOT)),
            "selection_table": str(selection_path.relative_to(ROOT)),
            "failure_reasons": str(failure_path.relative_to(ROOT)),
        },
    }
    (OUT / "042_06_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
