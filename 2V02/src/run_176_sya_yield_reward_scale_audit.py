"""Task 176: read-only audit of terminal yield reward coefficient and scale."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "176_sya_yield_reward_scale_audit.json"
DEFAULT_OUTPUT = ROOT / "benchmark_results" / "176_sya_yield_reward_scale_audit"
REWARD_CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
REWARD_CODE = ROOT / "src" / "run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py"
EXPERIMENTS = {
    "172": ROOT / "benchmark_results" / "172_sya_weather_aug_rain_observation_smoke2k",
    "174": ROOT / "benchmark_results" / "174_sya_weather_aug_rain_obs_no_swfac_guardrail_smoke2k",
}


def scalar(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def first(df: pd.DataFrame, column: str, default: Any = None) -> Any:
    if df.empty or column not in df.columns:
        return default
    value = df[column].iloc[0]
    return default if pd.isna(value) else value


def parse_yaml_reward_values(path: Path) -> Dict[str, float]:
    text = path.read_text(encoding="utf-8")
    values: Dict[str, float] = {}
    for key in [
        "yield_coef",
        "nitrogen_cost",
        "water_cost",
        "water_stress_relief_coef",
        "nitrogen_stress_relief_coef",
        "reward_scale",
    ]:
        match = re.search(rf"^\s*{re.escape(key)}:\s*([-+]?\d+(?:\.\d+)?)\s*$", text, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"Could not find {key} in {path}")
        values[key] = float(match.group(1))
    return values


def source_line_numbers(path: Path, patterns: List[str]) -> Dict[str, int | None]:
    lines = path.read_text(encoding="utf-8").splitlines()
    result: Dict[str, int | None] = {}
    for pattern in patterns:
        result[pattern] = next((index for index, line in enumerate(lines, start=1) if pattern in line), None)
    return result


def parse_checkpoint(path: Path, df: pd.DataFrame) -> int | None:
    if "checkpoint_step" in df.columns:
        try:
            return int(float(df["checkpoint_step"].iloc[0]))
        except (TypeError, ValueError):
            pass
    match = re.search(r"ckpt(\d+)", path.name)
    return int(match.group(1)) if match else None


def audit_file(experiment: str, path: Path, cfg: Dict[str, float]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty daily file: {path}")

    yield_s = numeric(df, "yield_component")
    grnwt_s = numeric(df, "grnwt")
    unscaled_s = numeric(df, "reward_unscaled")
    scaled_name = "reward_stress_aware" if "reward_stress_aware" in df.columns else "reward"
    scaled_s = numeric(df, scaled_name)
    scale_s = numeric(df, "reward_scale")
    done_mask = df["done"].astype(str).str.strip().str.lower().eq("true") if "done" in df.columns else pd.Series(False, index=df.index)
    yield_mask = yield_s.abs() > 1e-9
    nonterminal_yield_mask = yield_mask & ~done_mask
    implied_coef = pd.Series(float("nan"), index=df.index, dtype=float)
    valid_grain = yield_mask & (grnwt_s.abs() > 1e-9)
    implied_coef.loc[valid_grain] = yield_s.loc[valid_grain] / grnwt_s.loc[valid_grain]
    coef_error = (implied_coef - cfg["yield_coef"]).abs()
    scale_error = (scaled_s - unscaled_s * scale_s).abs()
    source_scale_error = (scale_s - cfg["reward_scale"]).abs()

    def dap_bounds(mask: pd.Series) -> tuple[Any, Any]:
        if "dap" not in df.columns or not bool(mask.any()):
            return None, None
        daps = pd.to_numeric(df.loc[mask, "dap"], errors="coerce").dropna()
        return (int(daps.min()), int(daps.max())) if not daps.empty else (None, None)

    yield_first, yield_last = dap_bounds(yield_mask)
    reward_first, reward_last = dap_bounds(unscaled_s.abs() > 1e-9)
    row = {
        "experiment": experiment,
        "daily_file": str(path.relative_to(ROOT)),
        "station_code": first(df, "station_code", ""),
        "weather_variant": first(df, "weather_variant", ""),
        "source_year": first(df, "source_year", None),
        "checkpoint_step": parse_checkpoint(path, df),
        "n_rows": int(len(df)),
        "done_rows": int(done_mask.sum()),
        "yield_nonzero_rows": int(yield_mask.sum()),
        "yield_nonzero_nonterminal_rows": int(nonterminal_yield_mask.sum()),
        "yield_first_dap": yield_first,
        "yield_last_dap": yield_last,
        "reward_nonzero_first_dap": reward_first,
        "reward_nonzero_last_dap": reward_last,
        "terminal_yield_component_sum": float(yield_s.sum()),
        "terminal_grnwt_at_yield_rows_sum": float(grnwt_s.loc[yield_mask].sum()),
        "yield_coef_expected": cfg["yield_coef"],
        "yield_coef_implied_mean": float(implied_coef.loc[valid_grain].mean()) if bool(valid_grain.any()) else None,
        "yield_coef_implied_min": float(implied_coef.loc[valid_grain].min()) if bool(valid_grain.any()) else None,
        "yield_coef_implied_max": float(implied_coef.loc[valid_grain].max()) if bool(valid_grain.any()) else None,
        "yield_coef_max_abs_error": float(coef_error.loc[valid_grain].max()) if bool(valid_grain.any()) else None,
        "reward_scale_expected": cfg["reward_scale"],
        "reward_scale_observed_min": float(scale_s.min()),
        "reward_scale_observed_max": float(scale_s.max()),
        "reward_scale_source_max_abs_error": float(source_scale_error.max()),
        "reward_scale_formula_max_abs_error": float(scale_error.max()),
        "yield_component_sum": float(yield_s.sum()),
        "reward_unscaled_sum": float(unscaled_s.sum()),
        "reward_scaled_sum": float(scaled_s.sum()),
        "positive_unscaled_reward_rows": int((unscaled_s > 1e-9).sum()),
        "negative_unscaled_reward_rows": int((unscaled_s < -1e-9).sum()),
    }
    detail = {
        "experiment": experiment,
        "daily_file": str(path.relative_to(ROOT)),
        "checkpoint_step": parse_checkpoint(path, df),
        "weather_variant": first(df, "weather_variant", ""),
        "yield_nonzero_dap": pd.to_numeric(df.loc[yield_mask, "dap"], errors="coerce").dropna().astype(int).tolist()
        if "dap" in df.columns
        else [],
        "done_dap": pd.to_numeric(df.loc[done_mask, "dap"], errors="coerce").dropna().astype(int).tolist()
        if "dap" in df.columns
        else [],
        "yield_coef_implied_values": implied_coef.loc[valid_grain].round(12).tolist(),
        "reward_scale_values": sorted({round(float(x), 12) for x in scale_s.tolist()}),
        "scale_formula_error_values": scale_error.round(12).tolist(),
    }
    return row, detail


def write_doc(output: Path, records: pd.DataFrame, conclusion: Dict[str, Any], provenance: Dict[str, Any]) -> Path:
    doc = ROOT / "docs" / "176_sya_yield_reward_scale_audit_record.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Task 176：SYA 终点产量奖励系数与尺度审计记录",
        "",
        "> 只读审计；没有重新训练 PPO，也没有修改 reward、环境或原始输出。",
        "",
        "## 运行时 provenance",
        "",
        f"- reward 配置：`{provenance['reward_config_path']}`。",
        f"- reward 代码：`{provenance['reward_code_path']}`。",
        f"- 172/174 共审计 `{len(records)}` 个 daily 文件。",
        f"- 有效 `yield_coef`：`{provenance['reward_values']['yield_coef']}`。",
        f"- 有效 `reward_scale`：`{provenance['reward_values']['reward_scale']}`。",
        "",
        "## 结果",
        "",
        f"- 终点产量项非终止行非零数：`{conclusion['yield_nonzero_nonterminal_rows_total']}`。",
        f"- 所有文件的终点产量系数重构最大误差：`{conclusion['yield_coef_max_abs_error']:.8g}`。",
        f"- 所有文件的 reward scale 公式重构最大误差：`{conclusion['reward_scale_formula_max_abs_error']:.8g}`。",
        f"- 所有文件的 reward scale 运行时漂移最大误差：`{conclusion['reward_scale_source_max_abs_error']:.8g}`。",
        f"- 终点产量项是否只出现在 done 行：`{conclusion['terminal_yield_only']}`。",
        "",
        "## 解释边界",
        "",
        "- 如果终点项只在 done 行出现，说明当前 reward 对产量的反馈是稀疏的，可能增加信用分配难度。",
        "- 这不是 PPO 策略坍缩的充分证明；它只说明 reward 的产量反馈发生得晚。还需要结合 action diversity、优势估计和动作反事实检查。",
        "- `reward_scale=0.001` 是全体 reward 的统一缩放；在理想实现中它不会改变最优策略排序，但会改变数值梯度/优势的量级，因此仍应记录。",
        "",
        "## 下一步",
        "",
        f"- {conclusion['next_step']}",
        "- 本任务没有启动新的 PPO 训练。",
        "",
        "## 输出",
        "",
        f"- `{(output / '176_runtime_reward_config.json').relative_to(ROOT)}`",
        f"- `{(output / '176_yield_timing_by_episode.csv').relative_to(ROOT)}`",
        f"- `{(output / '176_reward_scale_consistency.csv').relative_to(ROOT)}`",
        f"- `{(output / '176_sya_yield_reward_scale_audit.json').relative_to(ROOT)}`",
    ]
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    task_cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if any(task_cfg.get(key) is not False for key in ["run_training", "modify_reward", "modify_environment"]):
        raise ValueError("Task 176 must remain read-only and must not train or modify reward/environment.")

    reward_values = parse_yaml_reward_values(REWARD_CONFIG)
    source_lines = source_line_numbers(
        REWARD_CODE,
        [
            "final_yield_component = 0.0",
            "final_yield_component = float(reward_cfg[\"yield_coef\"])",
            "unscaled_reward = final_yield_component - resource_cost + stress_relief_bonus",
            "reward_scale = float(reward_cfg.get(\"reward_scale\", 1.0))",
            "reward = unscaled_reward * reward_scale",
        ],
    )
    provenance = {
        "reward_config_path": str(REWARD_CONFIG.relative_to(ROOT)),
        "reward_code_path": str(REWARD_CODE.relative_to(ROOT)),
        "reward_values": reward_values,
        "reward_source_line_numbers": source_lines,
        "inheritance_note": "172/174 -> 168/040_36 -> 040_26 -> base03222.CONFIG -> config_032_00; 174 only disables SWFAC guardrail.",
    }

    rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    for experiment, root in EXPERIMENTS.items():
        paths = sorted((root / "daily_outputs" / "SYA").glob("*_daily.csv"))
        if not paths:
            raise FileNotFoundError(f"No daily CSV files found under {root}")
        for path in paths:
            row, detail = audit_file(experiment, path, reward_values)
            rows.append(row)
            details.append(detail)

    frame = pd.DataFrame(rows).sort_values(["experiment", "weather_variant", "checkpoint_step"])
    scale_frame = frame[
        [
            "experiment",
            "daily_file",
            "weather_variant",
            "checkpoint_step",
            "yield_coef_expected",
            "yield_coef_implied_mean",
            "yield_coef_max_abs_error",
            "reward_scale_expected",
            "reward_scale_observed_min",
            "reward_scale_observed_max",
            "reward_scale_source_max_abs_error",
            "reward_scale_formula_max_abs_error",
        ]
    ].copy()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "176_yield_timing_by_episode.csv", index=False, encoding="utf-8-sig")
    scale_frame.to_csv(output / "176_reward_scale_consistency.csv", index=False, encoding="utf-8-sig")

    max_coef_error = float(frame["yield_coef_max_abs_error"].dropna().max())
    max_scale_error = float(frame["reward_scale_formula_max_abs_error"].max())
    max_source_scale_error = float(frame["reward_scale_source_max_abs_error"].max())
    nonterminal_rows = int(frame["yield_nonzero_nonterminal_rows"].sum())
    terminal_only = nonterminal_rows == 0 and bool((frame["yield_nonzero_rows"] > 0).all())
    conclusion = {
        "n_files": int(len(frame)),
        "n_files_by_experiment": {str(k): int(v) for k, v in frame.groupby("experiment").size().items()},
        "yield_nonzero_nonterminal_rows_total": nonterminal_rows,
        "terminal_yield_only": terminal_only,
        "yield_coef_expected": reward_values["yield_coef"],
        "yield_coef_max_abs_error": max_coef_error,
        "reward_scale_expected": reward_values["reward_scale"],
        "reward_scale_formula_max_abs_error": max_scale_error,
        "reward_scale_source_max_abs_error": max_source_scale_error,
        "yield_coefficient_consistency_passed": max_coef_error <= 1e-9,
        "reward_scale_consistency_passed": max_scale_error <= 1e-12 and max_source_scale_error <= 1e-12,
        "next_step": "若要验证稀疏终点奖励是否参与策略固定，下一项应是只改变终点产量奖励时序/信用分配方式的 2K 消融；不要把 reward_scale 统一缩放本身当成原因。",
    }
    (output / "176_runtime_reward_config.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "176_sya_yield_reward_scale_audit.json").write_text(
        json.dumps({"task_id": 176, "mode": "read_only", "conclusion": conclusion, "details": details}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    doc = write_doc(output, frame, conclusion, provenance)
    print(json.dumps({"output_dir": str(output), "doc": str(doc), "conclusion": conclusion}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
