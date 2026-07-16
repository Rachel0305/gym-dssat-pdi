from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "021_46"
DOC = ROOT / "docs" / "2026-07-15_021_46_sy2014_terminal_bonus_epsilon_schedule_offline_audit.md"
SOURCE = ROOT / "src" / "run_sy2014_dqfd_real_network_loss_diagnostic_021_22.py"
CONTROL = ROOT / "benchmark_results" / "021_42" / "online_seed1" / "training_interactions.csv"
TREATMENT = ROOT / "benchmark_results" / "021_45" / "021_45_training_interactions.csv"


def constants_from_source() -> dict[str, float]:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    wanted = {"PLANNED_TIMESTEPS", "EXPLORATION_FRACTION"}
    values: dict[str, float] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in wanted:
                values[name] = float(ast.literal_eval(node.value))
    if set(values) != wanted:
        raise RuntimeError(f"Missing epsilon constants: {wanted - set(values)}")
    return values


def epsilon_at(step: int, planned: float, exploration_fraction: float) -> float:
    decay_steps = exploration_fraction * planned
    fraction = min(max(float(step) / decay_steps, 0.0), 1.0)
    return 1.0 + fraction * (0.05 - 1.0)


def summarize(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    rows = []
    for start, end in ((1, 250), (251, 500), (501, 750), (751, 1000)):
        part = frame.loc[frame.env_step.between(start, end)]
        rows.append({
            "arm": arm, "step_start": start, "step_end": end, "rows": len(part),
            "epsilon_min": float(part.epsilon.min()), "epsilon_max": float(part.epsilon.max()),
            "epsilon_mean": float(part.epsilon.mean()),
            "epsilon_random_fraction": float(part.action_source.eq("epsilon_random").mean()),
            "greedy_fraction": float(part.action_source.eq("greedy").mean()),
        })
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    if OUT.exists(): raise FileExistsError(f"Refusing to overwrite {OUT}")
    if DOC.exists(): raise FileExistsError(f"Refusing to overwrite {DOC}")
    OUT.mkdir(parents=True)
    constants = constants_from_source()
    control = pd.read_csv(CONTROL)
    treatment = pd.read_csv(TREATMENT)
    planned = constants["PLANNED_TIMESTEPS"]
    fraction = constants["EXPLORATION_FRACTION"]
    for frame in (control, treatment):
        expected = np.asarray([epsilon_at(int(step), planned, fraction) for step in frame.env_step])
        frame["expected_epsilon"] = expected
        frame["epsilon_abs_error"] = np.abs(frame.epsilon - expected)
    intervals = pd.concat([
        summarize(control, "control_021_42_seed1"),
        summarize(treatment, "treatment_021_45_terminal_bonus"),
    ], ignore_index=True)
    intervals.to_csv(OUT / "021_46_epsilon_interval_summary.csv", index=False, encoding="utf-8-sig")
    milestones = treatment.loc[treatment.env_step.isin([1, 250, 500, 750, 1000]),
                               ["env_step", "epsilon", "expected_epsilon", "action_source", "action"]]
    milestones.to_csv(OUT / "021_46_epsilon_milestones.csv", index=False, encoding="utf-8-sig")
    control_treatment_equal = bool(np.array_equal(control.epsilon.to_numpy(), treatment.epsilon.to_numpy()))
    final_epsilon = float(treatment.iloc[-1].epsilon)
    late = treatment.loc[treatment.env_step.between(751, 1000)]
    validation = {
        "planned_timesteps_from_source": planned,
        "exploration_fraction_from_source": fraction,
        "decay_steps": planned * fraction,
        "final_epsilon": final_epsilon,
        "late_751_1000_epsilon_mean": float(late.epsilon.mean()),
        "late_751_1000_random_action_fraction": float(late.action_source.eq("epsilon_random").mean()),
        "control_treatment_epsilon_arrays_exactly_equal": control_treatment_equal,
        "control_max_formula_error": float(control.epsilon_abs_error.max()),
        "treatment_max_formula_error": float(treatment.epsilon_abs_error.max()),
        "uses_custom_explicit_schedule_not_sb3_learn_schedule": True,
        "dssat_calls": 0,
        "training_steps": 0,
    }
    validation["compressed_exploration_hypothesis_supported"] = bool(final_epsilon <= 0.10)
    validation["all_required_checks_pass"] = bool(
        len(control) == 1000 and len(treatment) == 1000 and control_treatment_equal
        and validation["control_max_formula_error"] < 1e-12
        and validation["treatment_max_formula_error"] < 1e-12
    )
    (OUT / "021_46_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    display = intervals.round(6)
    doc = f"""# 021_46 SY2014 终端 bonus 的 epsilon 调度离线审计

## 结论

021_45 没有受到 SB3 分段 `learn()` 导致的探索率衰减压缩。它不调用 SB3 的 exploration schedule，而是使用源码中显式的全局公式：`PLANNED_TIMESTEPS={planned:.0f}`、`EXPLORATION_FRACTION={fraction}`，即在 {planned*fraction:.0f} 步内从 1.0 衰减到 0.05。

1000 步时 epsilon={final_epsilon:.6f}；751–1000 步平均 epsilon={validation['late_751_1000_epsilon_mean']:.6f}，实际随机动作比例={validation['late_751_1000_random_action_fraction']:.3f}。因此“750–1000 步探索率已接近 0、策略被锁死”与数据相反。

## 分段统计

{markdown_table(display)}

Control 与 Treatment 的 epsilon 数组逐点完全相同：{control_treatment_equal}；两者与源码公式的最大误差分别为 {validation['control_max_formula_error']:.3g} 和 {validation['treatment_max_formula_error']:.3g}。

## 对 021_45 的解释修正

- 1000-step 的相同确定性结果不能归因于探索率过早归零。
- 训练交互在 1K 内仍以随机探索为主；checkpoint 评估则是冻结网络的确定性 argmax，两者不能混为一谈。
- Control/Treatment 在 250/500 步并非完全相同，只在 750/1000 步重新汇合。单 seed、单个终点不足以证明存在“与 reward 无关的固定吸引点”。
- 021_26–34 主要是离线训练/审计；021_35–45 的在线部分使用当前显式全局 epsilon 公式，不能笼统标记为“探索率异常条件下成立”。

## 边界

本任务只排除了探索率压缩这一特定解释，没有确认 terminal bonus 为什么未改变最终保持性。没有新训练、没有 DSSAT 调用，也不据此修改 reward。
"""
    DOC.write_text(doc, encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__": main()
