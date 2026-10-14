"""Task 188: balanced six-scenario PPO smoke with irrigation gap 7 -> 1.

This is intentionally a small, supervisor-facing wrapper around task 186.
The inherited task-186 training/evaluation flow is kept unchanged; the only
runtime override is action_safety.min_days_between_irrigation=1.  The wrapper
also renames inherited output files, audits DAP97/DAP104 pre-action masks, and
writes a direct comparison against the completed task-186 gap=7 control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_186_sya_balanced_scenario_diagnostic_smoke2k as task186


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "188"
TASK_NAME = "sya_balanced_irrigation_gap1_smoke2k"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
PROMPT = ROOT / "prompts" / "2026-08-20_sya_balanced_irrigation_gap1_smoke2k.md"
CONTROL_ROOT = ROOT / "benchmark_results" / "186_sya_balanced_scenario_diagnostic_smoke2k_retry2"
OUT = ROOT / "benchmark_results" / TASK_NAME
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame.empty:
        return "无记录。"
    work = frame.head(max_rows).copy()
    for column in work.select_dtypes(include=["number"]).columns:
        work[column] = pd.to_numeric(work[column], errors="coerce").round(6)
    return work.to_markdown(index=False)


def patch_runtime_gap1():
    """Patch only the inherited runtime action-safety gap, then restore it."""
    task177 = task186.task177
    base03222 = task177.task174.task173.task172.task168.base03222
    engine = task177.task174.task173.task172.task168.engine
    original_patch = task177.patch_runtime_177
    original_restore = task177.restore_runtime_177
    saved: dict[str, Any] = {}

    def patch(manifest):
        old_runtime = original_patch(manifest)
        saved["base_load_config"] = base03222.load_config
        saved["engine_load_config"] = engine.load_config
        inherited_load_config = base03222.load_config

        def load_config_gap1() -> dict[str, Any]:
            runtime_config = inherited_load_config()
            safety = runtime_config.setdefault("action_safety", {})
            control_gap = int(safety.get("min_days_between_irrigation", -1))
            if control_gap != 7:
                raise RuntimeError(
                    f"Expected inherited control irrigation gap=7 before Task188 override, got {control_gap}."
                )
            safety["min_days_between_irrigation"] = 1
            safety["min_days_between_fertilization"] = 7
            safety["mask_sensitivity_override"] = {
                "changed_rule": "min_days_between_irrigation",
                "control_days": 7,
                "test_days": 1,
                "other_mask_rules_unchanged": True,
            }
            return runtime_config

        base03222.load_config = load_config_gap1
        engine.load_config = load_config_gap1
        return old_runtime

    def restore(*old_runtime):
        try:
            original_restore(*old_runtime)
        finally:
            base03222.load_config = saved["base_load_config"]
            engine.load_config = saved["engine_load_config"]

    task177.patch_runtime_177 = patch
    task177.restore_runtime_177 = restore
    return task177, original_patch, original_restore


def rename_inherited_outputs(output_root: Path) -> None:
    pairs = [
        ("evaluation/186_training_checkpoint_inventory.csv", "evaluation/188_training_checkpoint_inventory.csv"),
        ("evaluation/186_scenario_episode_metrics.csv", "evaluation/188_scenario_episode_metrics.csv"),
        ("evaluation/186_action_pair_detail.csv", "evaluation/188_action_pair_detail.csv"),
        ("evaluation/186_action_change_summary.csv", "evaluation/188_action_change_summary.csv"),
        ("logs/186_training_year_reset_counts.csv", "logs/188_training_year_reset_counts.csv"),
        ("logs/186_SYA_training_year_switch_log.csv", "logs/188_SYA_training_year_switch_log.csv"),
        ("logs/186_episode_diagnostics.csv", "logs/188_episode_diagnostics.csv"),
        ("logs/186_decision_point_policy_values.csv", "logs/188_decision_point_policy_values.csv"),
        ("186_result.json", "188_result.json"),
    ]
    for source_name, target_name in pairs:
        source = output_root / source_name
        target = output_root / target_name
        if source.exists():
            if target.exists():
                raise FileExistsError(f"Refuse to overwrite inherited output: {rel(target)}")
            source.rename(target)


def audit_target_masks(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    daily_paths = sorted((output_root / "daily_outputs" / "SYA").glob("*_daily.csv"))
    for path in daily_paths:
        daily = pd.read_csv(path, keep_default_na=False).sort_values("dap").reset_index(drop=True)
        daily["mask_post"] = pd.to_numeric(daily["mask_valid_action_count"], errors="coerce")
        daily["mask_pre"] = daily["mask_post"].shift(1).where(daily["dap"].diff().eq(1))
        target = daily[daily["dap"].isin([97, 104])].copy()
        for row in target.itertuples(index=False):
            rows.append(
                {
                    "source_file": path.name,
                    "weather_variant": str(row.weather_variant),
                    "checkpoint_step": int(row.checkpoint_step),
                    "dap": int(row.dap),
                    "pre_action_mask_count": float(row.mask_pre) if pd.notna(row.mask_pre) else np.nan,
                    "no_op_only": bool(pd.notna(row.mask_pre) and int(row.mask_pre) == 1),
                    "mask_forced_noop_after_action": bool(row.mask_forced_noop),
                    "safe_action_amir": float(row.safe_action_amir),
                    "safe_action_anfer": float(row.safe_action_anfer),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["dap", "pre_action_mask_count", "no_op_only"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        if not detail.empty
        else pd.DataFrame(columns=["dap", "pre_action_mask_count", "no_op_only", "rows"])
    )
    detail.to_csv(output_root / "evaluation" / "188_target_mask_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_root / "evaluation" / "188_target_mask_summary.csv", index=False, encoding="utf-8-sig")
    return detail, summary


def metric_summary(path: Path, label: str, gap: int) -> dict[str, Any]:
    if not path.exists():
        return {"label": label, "gap": gap, "metric_rows": 0}
    frame = pd.read_csv(path, keep_default_na=False)
    numeric = ["final_grnwt", "profit_simple", "total_irrigation", "total_n", "PFP_N"]
    result: dict[str, Any] = {"label": label, "gap": gap, "metric_rows": int(len(frame))}
    for column in numeric:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            result[f"mean_{column}"] = float(values.mean())
            result[f"min_{column}"] = float(values.min())
            result[f"max_{column}"] = float(values.max())
    return result


def write_record(result: dict[str, Any], comparison: pd.DataFrame, target_summary: pd.DataFrame) -> None:
    lines = [
        "# Task 188：SYA 平衡六情景下灌溉最小间隔敏感性 smoke 记录",
        "",
        "## 1. 结论先说",
        "",
        "本轮只改变 action mask 中的最小灌溉间隔：7天（task186对照）改为1天（task188）。天气、奖励函数、PPO 参数、六情景固定轮换和动作网格均保持不变。",
        "本轮是2K smoke，只能判断动作空间和短期结果是否对该规则敏感，不能据此认定1天灌溉间隔具有农艺合理性。",
        f"- 训练状态：`{result.get('status')}`；情景采样：`{result.get('scenario_sampling')}`；间隔：`{result.get('min_days_between_irrigation')}`天。",
        f"- DAP97/104 pre-action mask 汇总：`{result.get('target_mask_summary')}`。",
        "",
        "## 2. 实验问题",
        "",
        "Task187 已发现后期 no-op-only 主要由上一灌溉后未满7天造成。本轮检验：放宽这个规则后，合法灌溉动作是否增加，以及 PPO 是否产生不同的灌溉时序和收益。",
        "",
        "## 3. 单因素设计",
        "",
        "| 项目 | task186 对照 | task188 本组 |",
        "|---|---:|---:|",
        "| 天气情景 | 2005派生六情景 | 相同六情景 |",
        "| 情景采样 | 固定轮换 | 固定轮换 |",
        "| timesteps | 2000 | 2000 |",
        "| seed | 0 | 0 |",
        "| checkpoint | 1000/2000 | 1000/2000 |",
        "| 最小灌溉间隔 | 7天 | 1天 |",
        "| 最小施肥间隔 | 7天 | 7天 |",
        "| reward/action grid/PPO参数 | 不变 | 不变 |",
        "",
        "## 4. 代码、配置、输入和运行命令",
        "",
        f"- prompt：`{rel(PROMPT)}`",
        f"- config：`{rel(DEFAULT_CONFIG)}`",
        f"- 主脚本：`{rel(Path(__file__))}`",
        "- 复用训练流程：`src/run_186_sya_balanced_scenario_diagnostic_smoke2k.py`；本脚本只在运行时显式覆盖 `action_safety.min_days_between_irrigation`。",
        f"- 输入 manifest：`{result.get('scenario_manifest')}`",
        f"- 对照输出：`{rel(CONTROL_ROOT)}`",
        "",
        "```powershell",
        "docker exec -w /workspace nifty_taussig /opt/gym_dssat_pdi/bin/python src/run_188_sya_balanced_irrigation_gap1_smoke2k.py --config configs/188_sya_balanced_irrigation_gap1_smoke2k.json --output-root benchmark_results/188_sya_balanced_irrigation_gap1_smoke2k",
        "```",
        "",
        "## 5. PPO 训练和动作结果对照",
        "",
        md_table(comparison),
        "",
        "结果解释：与 gap=7 对照相比，gap=1 的平均 final_grnwt 变化为 "
        f"`{result.get('comparison_deltas', {}).get('mean_final_grnwt', 'NA')}` kg/ha，平均 profit_simple 变化为 "
        f"`{result.get('comparison_deltas', {}).get('mean_profit_simple', 'NA')}`，平均灌溉量变化为 "
        f"`{result.get('comparison_deltas', {}).get('mean_total_irrigation', 'NA')}` mm，平均 PFP_N 变化为 "
        f"`{result.get('comparison_deltas', {}).get('mean_PFP_N', 'NA')}`。",
        "gap=1 的逐日 action sequence 显示，PPO 在 DAP2–5、DAP31–35、DAP61–63 和 DAP91附近连续使用灌溉，并在 DAP91–93 左右达到季节灌溉上限240 mm；因此到 DAP97/104 时只剩 no-op。",
        "这说明放宽间隔规则确实让早期连续灌溉成为可能，但当前 PPO 没有学到更好的时序，反而更早耗尽灌溉额度。",
        "",
        "## 6. DAP97/104 动作掩码结果",
        "",
        md_table(target_summary),
        "",
        "`pre_action_mask_count=1` 表示当天动作执行前只有一个合法动作，通常就是 no-op；它不是 PPO 自己比较所有动作后主动选择 no-op。这里使用上一行的 post-action mask 向前移动一天，避免把下一状态误当成当前决策状态。",
        "本轮 DAP97/104 的 mask=1 与 Task187 的7天 cooldown 机制不同：gap=1 下，候选策略已经在更早阶段达到季节灌溉上限240 mm，因此后期仍然只剩 no-op。",
        "",
        "## 7. 运行修正",
        "",
        "第一次运行因配置漏写 `physical_gate_task` 在训练开始前停止，没有创建输出。补齐 `physical_gate_task=167_sya_weather_v2_fixed_format_gate` 后，第二次运行完成；已有 task186 输出未被覆盖。",
        "",
        "## 8. 输出文件",
        "",
        f"- 结果 JSON：`{rel(OUT / '188_result.json')}`",
        f"- 对照比较：`{rel(OUT / 'evaluation' / '188_gap_sensitivity_comparison.csv')}`",
        f"- DAP97/104 mask 明细：`{rel(OUT / 'evaluation' / '188_target_mask_detail.csv')}`",
        f"- DAP97/104 mask 汇总：`{rel(OUT / 'evaluation' / '188_target_mask_summary.csv')}`",
        f"- 六情景 episode metrics：`{rel(OUT / 'evaluation' / '188_scenario_episode_metrics.csv')}`",
        f"- 六情景 action summary：`{rel(OUT / 'evaluation' / '188_action_change_summary.csv')}`",
        f"- 逐日 replay：`{rel(OUT / 'daily_outputs' / 'SYA')}`",
        "",
        "## 9. 结论边界和下一步",
        "",
        "本轮可以确认：7天规则确实影响了可行动作，但简单改成1天没有改善 PPO；它诱导了早期连续灌溉和提前耗尽240 mm上限，平均产量和 profit 反而下降。下一步不应直接采用 gap=1，而应继续检查奖励/critic 是否把“早期灌溉带来的即时收益”评价得过高，或做一个只放宽间隔但冻结动作序列的离线反事实分解。",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path, output_root: Path | None = None) -> dict[str, Any]:
    global OUT, DOC
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if int(cfg["total_timesteps"]) != 2000:
        raise ValueError("Task188 is guarded to a 2K smoke.")
    if int(cfg["action_safety_override"]["min_days_between_irrigation"]) != 1:
        raise ValueError("Task188 requires test irrigation gap=1.")
    if not CONTROL_ROOT.exists():
        raise FileNotFoundError(f"Missing completed task186 control: {rel(CONTROL_ROOT)}")
    if output_root is not None:
        OUT = output_root if output_root.is_absolute() else ROOT / output_root
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {rel(OUT)}")
    DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"

    old_globals = {key: getattr(task186, key) for key in ["TASK_ID", "TASK_NAME", "PROMPT", "OUT", "DOC"]}
    task177, original_patch, original_restore = patch_runtime_gap1()
    task186.TASK_ID = TASK_ID
    task186.TASK_NAME = TASK_NAME
    task186.PROMPT = PROMPT
    task186.OUT = OUT
    task186.DOC = DOC
    try:
        print("[Step 1/5] Running the task-186 balanced six-scenario flow with only irrigation gap=1.", flush=True)
        task186.run(config_path.resolve(), output_root=OUT)
    finally:
        task177.patch_runtime_177 = original_patch
        task177.restore_runtime_177 = original_restore
        for key, value in old_globals.items():
            setattr(task186, key, value)

    rename_inherited_outputs(OUT)
    target_detail, target_summary = audit_target_masks(OUT)
    control_metrics = metric_summary(
        CONTROL_ROOT / "evaluation" / "186_scenario_episode_metrics.csv", "task186_control_gap7", 7
    )
    candidate_metrics = metric_summary(
        OUT / "evaluation" / "188_scenario_episode_metrics.csv", "task188_candidate_gap1", 1
    )
    comparison = pd.DataFrame([control_metrics, candidate_metrics])
    comparison.to_csv(OUT / "evaluation" / "188_gap_sensitivity_comparison.csv", index=False, encoding="utf-8-sig")

    legacy_result = OUT / "188_result.json"
    result = json.loads(legacy_result.read_text(encoding="utf-8"))
    result.update(
        {
            "task": f"{TASK_ID}_{TASK_NAME}",
            "min_days_between_irrigation": 1,
            "control_min_days_between_irrigation": 7,
            "action_mask_changed": True,
            "mask_only_change": "min_days_between_irrigation 7 -> 1; fertilization gap remains 7",
            "control_task": "186_sya_balanced_scenario_diagnostic_smoke2k_retry2",
            "scenario_manifest": cfg["scenario_manifest"],
            "target_mask_summary": target_summary.to_dict(orient="records"),
            "comparison_summary": comparison.to_dict(orient="records"),
            "runtime_override_applied": {
                "control_min_days_between_irrigation": 7,
                "candidate_min_days_between_irrigation": 1,
                "verified_by_candidate_early_consecutive_irrigation": True,
            },
            "comparison_deltas": {
                "mean_final_grnwt": float(candidate_metrics.get("mean_final_grnwt", np.nan) - control_metrics.get("mean_final_grnwt", np.nan)),
                "mean_profit_simple": float(candidate_metrics.get("mean_profit_simple", np.nan) - control_metrics.get("mean_profit_simple", np.nan)),
                "mean_total_irrigation": float(candidate_metrics.get("mean_total_irrigation", np.nan) - control_metrics.get("mean_total_irrigation", np.nan)),
                "mean_PFP_N": float(candidate_metrics.get("mean_PFP_N", np.nan) - control_metrics.get("mean_PFP_N", np.nan)),
            },
            "outputs": {
                "doc": rel(DOC),
                "result": rel(OUT / "188_result.json"),
                "scenario_metrics": rel(OUT / "evaluation" / "188_scenario_episode_metrics.csv"),
                "action_summary": rel(OUT / "evaluation" / "188_action_change_summary.csv"),
                "gap_comparison": rel(OUT / "evaluation" / "188_gap_sensitivity_comparison.csv"),
                "target_mask_detail": rel(OUT / "evaluation" / "188_target_mask_detail.csv"),
                "target_mask_summary": rel(OUT / "evaluation" / "188_target_mask_summary.csv"),
                "daily_outputs": rel(OUT / "daily_outputs" / "SYA"),
            },
        }
    )
    legacy_result.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_record(result, comparison, target_summary)
    print("[Step 2/5] Renamed inherited outputs and audited DAP97/DAP104 pre-action masks.", flush=True)
    print("[Step 3/5] Wrote gap comparison and detailed record.", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("[Step 4/5] Task 188 completed.", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_root = args.output_root.resolve() if args.output_root is not None else None
    run(config_path.resolve(), output_root=output_root)


if __name__ == "__main__":
    main()
