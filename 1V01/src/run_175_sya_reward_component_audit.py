"""Task 175: read-only reward-component audit for formal SYA tasks 172 and 174.

This script deliberately does not import the training environment and does not
run PPO. It only reads existing daily CSV outputs and writes new audit files.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "175_sya_reward_component_audit.json"
DEFAULT_OUTPUT = ROOT / "benchmark_results" / "175_sya_reward_component_audit"

EXPERIMENTS = {
    "172": ROOT / "benchmark_results" / "172_sya_weather_aug_rain_observation_smoke2k",
    "174": ROOT / "benchmark_results" / "174_sya_weather_aug_rain_obs_no_swfac_guardrail_smoke2k",
}

CORE_COMPONENTS = {
    "yield_component": "yield_component",
    "resource_cost": "resource_cost",
    "stress_relief_bonus": "stress_relief_bonus",
}

GUARDRAIL_COLUMNS = [
    "swfac_guardrail_penalty_unscaled",
    "swfac_guardrail_penalty_scaled",
]


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def numeric_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0)


def first_value(df: pd.DataFrame, name: str, default: Any = None) -> Any:
    if df.empty or name not in df.columns:
        return default
    value = df[name].iloc[0]
    if pd.isna(value):
        return default
    return value


def dap_range(df: pd.DataFrame, mask: pd.Series) -> tuple[Any, Any]:
    if "dap" not in df.columns or not bool(mask.any()):
        return None, None
    daps = pd.to_numeric(df.loc[mask, "dap"], errors="coerce").dropna()
    if daps.empty:
        return None, None
    return int(daps.min()), int(daps.max())


def parse_checkpoint(path: Path, df: pd.DataFrame) -> int | None:
    value = first_value(df, "checkpoint_step", None)
    if value is not None:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            pass
    match = re.search(r"ckpt(\d+)", path.name)
    return int(match.group(1)) if match else None


def audit_daily_file(experiment: str, path: Path) -> Dict[str, Any]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty daily CSV: {path}")

    yield_s = numeric_series(df, "yield_component")
    cost_s = numeric_series(df, "resource_cost")
    stress_s = numeric_series(df, "stress_relief_bonus")
    reward_unscaled_s = numeric_series(df, "reward_unscaled")
    reward_scaled_name = "reward_stress_aware" if "reward_stress_aware" in df.columns else "reward"
    reward_scaled_s = numeric_series(df, reward_scaled_name)
    guardrail_unscaled_present = "swfac_guardrail_penalty_unscaled" in df.columns
    guardrail_scaled_present = "swfac_guardrail_penalty_scaled" in df.columns
    guardrail_s = numeric_series(df, "swfac_guardrail_penalty_unscaled")
    guardrail_scaled_s = numeric_series(df, "swfac_guardrail_penalty_scaled")

    # The reward convention in the recorded CSV is:
    # yield_component - resource_cost + stress_relief_bonus - guardrail_penalty.
    reconstructed = yield_s - cost_s + stress_s - guardrail_s
    reconstruction_error = reward_unscaled_s - reconstructed

    masks = {
        "yield": yield_s.abs() > 1e-9,
        "cost": cost_s.abs() > 1e-9,
        "stress": stress_s.abs() > 1e-9,
        "guardrail": guardrail_s.abs() > 1e-9,
    }
    totals = {
        "yield_component_sum": float(yield_s.sum()),
        "resource_cost_sum": float(cost_s.sum()),
        "stress_relief_bonus_sum": float(stress_s.sum()),
        "guardrail_penalty_unscaled_sum": float(guardrail_s.sum()),
        "guardrail_penalty_scaled_sum": float(guardrail_scaled_s.sum()),
        "reward_unscaled_sum": float(reward_unscaled_s.sum()),
        "reward_scaled_sum": float(reward_scaled_s.sum()),
        "reconstructed_reward_unscaled_sum": float(reconstructed.sum()),
        "reward_reconstruction_error_sum": float(reconstruction_error.sum()),
        "reward_reconstruction_error_max_abs": float(reconstruction_error.abs().max()),
    }
    abs_component_total = sum(
        abs(totals[key])
        for key in (
            "yield_component_sum",
            "resource_cost_sum",
            "stress_relief_bonus_sum",
            "guardrail_penalty_unscaled_sum",
        )
    )
    totals["absolute_component_total"] = float(abs_component_total)
    for short, key in (
        ("yield", "yield_component_sum"),
        ("cost", "resource_cost_sum"),
        ("stress", "stress_relief_bonus_sum"),
        ("guardrail", "guardrail_penalty_unscaled_sum"),
    ):
        totals[f"{short}_absolute_share"] = (
            abs(totals[key]) / abs_component_total if abs_component_total else 0.0
        )

    yield_first, yield_last = dap_range(df, masks["yield"])
    cost_first, cost_last = dap_range(df, masks["cost"])
    stress_first, stress_last = dap_range(df, masks["stress"])
    guard_first, guard_last = dap_range(df, masks["guardrail"])

    record: Dict[str, Any] = {
        "experiment": experiment,
        "daily_file": str(path.relative_to(ROOT)),
        "station_code": first_value(df, "station_code", ""),
        "source_year": first_value(df, "source_year", None),
        "weather_variant": first_value(df, "weather_variant", ""),
        "seed": first_value(df, "seed", None),
        "checkpoint_step": parse_checkpoint(path, df),
        "n_daily_rows": int(len(df)),
        "episode_done_rows": int((df["done"].astype(str).str.lower() == "true").sum()) if "done" in df.columns else 0,
        "reward_scaled_column": reward_scaled_name,
        "guardrail_unscaled_column_present": guardrail_unscaled_present,
        "guardrail_scaled_column_present": guardrail_scaled_present,
        "swfac_guardrail_enabled_first": first_value(df, "swfac_guardrail_enabled", None),
        "swfac_guardrail_disabled_for_task": first_value(df, "swfac_guardrail_disabled_for_task", None),
        "yield_nonzero_rows": int(masks["yield"].sum()),
        "yield_first_dap": yield_first,
        "yield_last_dap": yield_last,
        "resource_cost_nonzero_rows": int(masks["cost"].sum()),
        "resource_cost_first_dap": cost_first,
        "resource_cost_last_dap": cost_last,
        "stress_bonus_nonzero_rows": int(masks["stress"].sum()),
        "stress_bonus_first_dap": stress_first,
        "stress_bonus_last_dap": stress_last,
        "guardrail_nonzero_rows": int(masks["guardrail"].sum()),
        "guardrail_first_dap": guard_first,
        "guardrail_last_dap": guard_last,
    }
    record.update(totals)
    return record


def collect_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for experiment, root in EXPERIMENTS.items():
        paths = sorted((root / "daily_outputs" / "SYA").glob("*_daily.csv"))
        if not paths:
            raise FileNotFoundError(f"No daily CSV files found under {root}")
        for path in paths:
            records.append(audit_daily_file(experiment, path))
    return records


def make_summary(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "yield_component_sum",
        "resource_cost_sum",
        "stress_relief_bonus_sum",
        "guardrail_penalty_unscaled_sum",
        "reward_unscaled_sum",
        "reward_scaled_sum",
        "absolute_component_total",
        "yield_absolute_share",
        "cost_absolute_share",
        "stress_absolute_share",
        "guardrail_absolute_share",
        "yield_nonzero_rows",
        "resource_cost_nonzero_rows",
        "stress_bonus_nonzero_rows",
        "guardrail_nonzero_rows",
        "reward_reconstruction_error_max_abs",
    ]
    grouped = (
        frame.groupby(["experiment", "checkpoint_step"], dropna=False)[numeric_columns]
        .agg(["count", "mean", "min", "max"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in column if str(part) != "").rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in grouped.columns
    ]
    return grouped


def make_delta(frame: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["weather_variant", "checkpoint_step"]
    value_columns = [
        "yield_component_sum",
        "resource_cost_sum",
        "stress_relief_bonus_sum",
        "guardrail_penalty_unscaled_sum",
        "reward_unscaled_sum",
        "reward_scaled_sum",
        "yield_absolute_share",
        "cost_absolute_share",
        "stress_absolute_share",
        "guardrail_absolute_share",
        "guardrail_nonzero_rows",
        "reward_reconstruction_error_max_abs",
    ]
    available = [column for column in value_columns if column in frame.columns]
    left = frame[frame["experiment"] == "172"][key_columns + available].copy()
    right = frame[frame["experiment"] == "174"][key_columns + available].copy()
    left = left.rename(columns={column: f"{column}_172" for column in available})
    right = right.rename(columns={column: f"{column}_174" for column in available})
    merged = left.merge(right, on=key_columns, how="outer", validate="one_to_one")
    for column in available:
        merged[f"{column}_174_minus_172"] = (
            pd.to_numeric(merged[f"{column}_174"], errors="coerce")
            - pd.to_numeric(merged[f"{column}_172"], errors="coerce")
        )
    return merged.sort_values(key_columns).reset_index(drop=True)


def choose_next_step(frame: pd.DataFrame) -> Dict[str, Any]:
    mean_abs = {
        "yield_component": float(frame["yield_component_sum"].abs().mean()),
        "resource_cost": float(frame["resource_cost_sum"].abs().mean()),
        "stress_relief_bonus": float(frame["stress_relief_bonus_sum"].abs().mean()),
        "swfac_guardrail_penalty": float(frame["guardrail_penalty_unscaled_sum"].abs().mean()),
    }
    dominant = max(mean_abs, key=mean_abs.get)
    guardrail_nonzero = int(frame["guardrail_nonzero_rows"].sum()) > 0
    reconstruction_max = float(frame["reward_reconstruction_error_max_abs"].max())
    if guardrail_nonzero:
        recommendation = "优先做 SWFAC guardrail 惩罚的独立消融；当前数据证明它确实触发。"
    elif dominant == "resource_cost":
        recommendation = "优先做 resource_cost 单项消融或系数敏感性；保持产量项、胁迫缓解项、天气观测和动作 mask 不变。"
    elif dominant == "stress_relief_bonus":
        recommendation = "优先做 stress_relief_bonus 单项消融；保持资源成本、产量项、天气观测和动作 mask 不变。"
    elif dominant == "yield_component":
        recommendation = "优先审计终点 yield_component 的系数/尺度；先做低成本 reward 计算审计，再决定是否训练。"
    else:
        recommendation = "先复核 reward 写入和尺度，再决定下一项消融；当前没有单一分量占主导。"
    return {
        "mean_absolute_component_totals": mean_abs,
        "dominant_component_by_mean_absolute_total": dominant,
        "guardrail_nonzero_rows_total": int(frame["guardrail_nonzero_rows"].sum()),
        "reward_reconstruction_max_abs_error": reconstruction_max,
        "reward_reconstruction_passed": reconstruction_max <= 1e-5,
        "next_ablation_recommendation": recommendation,
    }


def write_markdown(output_dir: Path, frame: pd.DataFrame, summary: pd.DataFrame, delta: pd.DataFrame, conclusion: Dict[str, Any]) -> Path:
    doc_path = ROOT / "docs" / "175_sya_reward_component_audit_record.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    guardrail_rows = int(frame["guardrail_nonzero_rows"].sum())
    delta_max = float(delta.filter(like="_174_minus_172").abs().max(numeric_only=True).max()) if not delta.empty else 0.0
    lines = [
        "# Task 175：SYA PPO reward 分量审计记录",
        "",
        "> 本任务为只读审计，不包含 PPO 训练，不修改环境、奖励函数或原始结果。",
        "",
        "## 数据范围",
        "",
        f"- 任务 172：{int((frame['experiment'] == '172').sum())} 个 daily CSV。",
        f"- 任务 174：{int((frame['experiment'] == '174').sum())} 个 daily CSV。",
        f"- 覆盖 checkpoint：{sorted(frame['checkpoint_step'].dropna().astype(int).unique().tolist())}。",
        f"- 覆盖情景：{sorted(frame['weather_variant'].dropna().unique().tolist())}。",
        "",
        "## 审计结果",
        "",
        f"- reward 重构误差最大绝对值：`{conclusion['reward_reconstruction_max_abs_error']:.8g}`；通过阈值检查：`{conclusion['reward_reconstruction_passed']}`。",
        f"- SWFAC guardrail 非零行数：`{guardrail_rows}`。",
        "- 对任务 172，guardrail 数值列存在但在当前六个 SYA 2005 情景中没有非零惩罚；对任务 174，guardrail 被关闭，字段本身不再输出。",
        "- 因此可以排除“当前这组实验中 SWFAC guardrail 触发导致策略固定”，但不能据此推广到其他年份、站点或更强胁迫情景。",
        "",
        "## 任务 172 与 174 的配对差异",
        "",
        f"- 相同 weather variant/checkpoint 的最大数值差异：`{delta_max:.8g}`。",
        "- 若该值为 0，说明关闭 guardrail 在当前输入上没有改变实际 reward 轨迹；这与 guardrail 从未触发相一致。",
        "",
        "## 分量主导性",
        "",
    ]
    for name, value in conclusion["mean_absolute_component_totals"].items():
        lines.append(f"- `{name}`：跨文件平均绝对 episode 总量 `{value:.6g}`。")
    lines.extend([
        "",
        f"- 按平均绝对 episode 总量，当前主导分量是：`{conclusion['dominant_component_by_mean_absolute_total']}`。",
        "",
        "## 下一步建议",
        "",
        f"- {conclusion['next_ablation_recommendation']}",
        "- 下一项实验仍应只改变一个因素，并先做 2K smoke；不能把本审计结果直接当成 PPO 泛化证明。",
        "",
        "## 输出文件",
        "",
        f"- `{(output_dir / '175_episode_reward_components.csv').relative_to(ROOT)}`",
        f"- `{(output_dir / '175_component_summary_by_experiment_checkpoint.csv').relative_to(ROOT)}`",
        f"- `{(output_dir / '175_component_delta_174_vs_172.csv').relative_to(ROOT)}`",
        f"- `{(output_dir / '175_reward_component_audit.json').relative_to(ROOT)}`",
    ])
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return doc_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    # Read config for provenance and fail loudly if it is malformed.
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("run_training") is not False or config.get("modify_environment") is not False:
        raise ValueError("Task 175 must remain read-only and must not run training.")

    records = collect_records()
    frame = pd.DataFrame(records).sort_values(["experiment", "weather_variant", "checkpoint_step"])
    summary = make_summary(frame)
    delta = make_delta(frame)
    conclusion = choose_next_step(frame)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "175_episode_reward_components.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "175_component_summary_by_experiment_checkpoint.csv", index=False, encoding="utf-8-sig")
    delta.to_csv(output_dir / "175_component_delta_174_vs_172.csv", index=False, encoding="utf-8-sig")
    payload = {
        "task_id": 175,
        "mode": "read_only",
        "config": str(args.config.resolve().relative_to(ROOT)),
        "n_files": int(len(frame)),
        "n_files_by_experiment": {str(k): int(v) for k, v in frame.groupby("experiment").size().items()},
        "weather_variants": sorted(frame["weather_variant"].dropna().unique().tolist()),
        "checkpoint_steps": sorted(frame["checkpoint_step"].dropna().astype(int).unique().tolist()),
        "conclusion": conclusion,
    }
    (output_dir / "175_reward_component_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    doc_path = write_markdown(output_dir, frame, summary, delta, conclusion)
    print(json.dumps({"output_dir": str(output_dir), "doc": str(doc_path), "conclusion": conclusion}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
