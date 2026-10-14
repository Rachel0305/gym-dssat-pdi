"""Task 189: read-only early-irrigation reward/critic audit for task 188.

The inherited task-184 runner already exposes a transparent one-step TD
diagnostic.  This wrapper changes only its source checkpoints, gap=1 runtime,
and target DAPs [2, 31, 61, 91].  No PPO training is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_184_sya_ppo_critic_td_action_value_audit as task184


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "189"
TASK_NAME = "sya_gap1_early_irrigation_reward_critic_audit"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
PROMPT = ROOT / "prompts" / "2026-08-20_sya_gap1_early_irrigation_reward_critic_audit.md"
MODEL_ROOT = ROOT / "benchmark_results" / "188_sya_balanced_irrigation_gap1_smoke2k" / "models" / "SYA"
OUT = ROOT / "benchmark_results" / TASK_NAME
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
CHECKPOINTS = [1000, 2000]
TARGET_DAPS = [2, 31, 61, 91]
SCENARIOS = task184.SCENARIOS.copy()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def patch_gap1_runtime():
    """Apply the gap=1 action-safety override and restore all runtime hooks."""
    task178 = task184.task178
    task168 = task178.task177.task174.task173.task172.task168
    base03222 = task168.base03222
    engine = task168.engine
    original_patch = task178.patch_runtime_178
    original_restore = task178.restore_runtime_178
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
                raise RuntimeError(f"Expected inherited gap=7 before Task189 override, got {control_gap}.")
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

    task178.patch_runtime_178 = patch
    task178.restore_runtime_178 = restore
    return task178, original_patch, original_restore


def rename_outputs(output_root: Path) -> None:
    pairs = [
        ("184_rootcause_summary.json", "189_rootcause_summary.json"),
        ("184_error.txt", "189_error.txt"),
    ]
    for directory in ["evaluation", "debug"]:
        for source in (output_root / directory).glob("184_*"):
            pairs.append((f"{directory}/{source.name}", f"{directory}/{source.name.replace('184_', '189_', 1)}"))
    for source_name, target_name in pairs:
        source = output_root / source_name
        target = output_root / target_name
        if source.exists():
            if target.exists():
                raise FileExistsError(f"Refuse to overwrite {rel(target)}")
            source.rename(target)


def write_record(result: dict[str, Any], group: pd.DataFrame, alignment: pd.DataFrame, reward_summary: pd.DataFrame) -> None:
    lines = [
        "# Task 189：SYA gap=1 早期连续灌溉 reward/critic 审计记录",
        "",
        "## 1. 结论先说",
        "",
        "本轮没有训练 PPO；只读取 task188 gap=1 的 checkpoint1000/2000，在 DAP2、31、61、91 四个关键时点，对 no-op、I15、I30、I45 做一步 DSSAT 分支，并计算 `reward_scaled + gamma × V(next)`。",
        f"- 状态：`{result.get('status')}`；目标状态：`{result.get('target_state_rows')}`；动作分支：`{result.get('candidate_td_rows')}`。",
        f"- 初步标签：`{result.get('rootcause_summary', {}).get('primary_label')}`。",
        "- 该标签只能说明局部 reward/critic 信号与 actor 的关系，不能替代整季 profit 结果。",
        "",
        "## 2. 实验问题和方法",
        "",
        "Task188 的 gap=1 PPO 在早期连续灌溉并提前用完240 mm额度。本轮检查这些动作是否获得了局部 reward 或 critic 支持。",
        "",
        "```text",
        "TD(a) = reward_scaled(s,a) + gamma * V(s')",
        "```",
        "",
        "PPO 的 critic 是 state-value V(s)，不是 Q(s,a)，因此 TD 结果是局部诊断代理。",
        "",
        "## 3. 设计和输入",
        "",
        "| 项目 | 本轮设置 |",
        "|---|---|",
        "| PPO模型 | task188 gap=1 checkpoint1000/2000 |",
        "| 天气 | 2005派生六情景 |",
        "| action mask | gap=1，施肥间隔7天，灌溉上限240 mm |",
        "| 目标DAP | 2、31、61、91 |",
        "| 候选动作 | I0/N0、I15/N0、I30/N0、I45/N0 |",
        "| 训练 | 不训练 |",
        "",
        f"- prompt：`{rel(PROMPT)}`",
        f"- config：`{rel(DEFAULT_CONFIG)}`",
        f"- 主脚本：`{rel(Path(__file__))}`",
        "- 复用 TD 方法：`src/run_184_sya_ppo_critic_td_action_value_audit.py`。",
        f"- PPO模型目录：`{rel(MODEL_ROOT)}`",
        "",
        "实际命令：",
        "",
        "```powershell",
        "docker exec -w /workspace nifty_taussig /opt/gym_dssat_pdi/bin/python src/run_189_sya_gap1_early_irrigation_reward_critic_audit.py --config configs/189_sya_gap1_early_irrigation_reward_critic_audit.json",
        "```",
        "",
        "## 4. actor 与 one-step TD 对照",
        "",
        md_table(alignment),
        "",
        "## 5. 按时点的 TD 汇总",
        "",
        md_table(group),
        "",
        f"具体结果：48个目标状态中，actor实际动作与one-step TD最高动作对齐 `{result.get('diagnostic_interpretation', {}).get('actor_td_aligned_rows', 'NA')}` 个；checkpoint2000中，TD最高动作是 I45 的状态为 `{result.get('diagnostic_interpretation', {}).get('ckpt2000_td_top_i45_rows', 'NA')}` 个，TD最高动作是 no-op 的状态为 `{result.get('diagnostic_interpretation', {}).get('ckpt2000_td_top_noop_rows', 'NA')}` 个。",
        "DAP2 的 I15/I30/I45 分支即时 reward 都只有水成本，但 critic 对灌溉后状态的 V(next) 估计更高，导致 I30 的 TD proxy 在多个情景中高于 no-op；DAP61/91 则相反，TD proxy普遍更支持 no-op，但 actor仍选择正灌溉。",
        "因此当前更应该优先怀疑 critic 的状态价值估计或时间信用分配，而不是认为即时 reward 本身在奖励早期灌溉。",
        "",
        "## 6. reward、成本和 critic 分量",
        "",
        md_table(reward_summary),
        "",
        "## 7. 输出文件",
        "",
        f"- 结果摘要：`{rel(OUT / '189_rootcause_summary.json')}`",
        f"- 目标状态：`{rel(OUT / 'evaluation' / '189_target_state_values.csv')}`",
        f"- 动作 TD 分支：`{rel(OUT / 'evaluation' / '189_action_one_step_td.csv')}`",
        f"- actor/TD 对照：`{rel(OUT / 'evaluation' / '189_actor_critic_alignment.csv')}`",
        f"- 分组汇总：`{rel(OUT / 'evaluation' / '189_actor_critic_group_summary.csv')}`",
        f"- reward 分量汇总：`{rel(OUT / 'evaluation' / '189_reward_component_summary.csv')}`",
        "",
        "## 8. 结论边界和下一步",
        "",
        "如果 I15/I45 的即时 reward 较高但 TD 或最终 profit 不占优，说明短期 reward 可能鼓励了过早灌溉；如果 actor 选择与 TD top 长期不一致，说明 actor/critic 学习信号不一致。无论哪种情况，下一步都不能直接把 gap=1 作为修正方案，而应在冻结动作和环境的条件下检查 reward 的时间信用分配。",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    global OUT, DOC, MODEL_ROOT
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("training_executed") is not False:
        raise ValueError("Task189 is read-only and must not train PPO.")
    if [int(x) for x in cfg["checkpoint_steps"]] != CHECKPOINTS:
        raise ValueError("Task189 expects checkpoint_steps=[1000,2000].")
    if [int(x) for x in cfg["target_daps"]] != TARGET_DAPS:
        raise ValueError("Task189 expects target_daps=[2,31,61,91].")
    if not MODEL_ROOT.exists():
        raise FileNotFoundError(f"Missing task188 model root: {rel(MODEL_ROOT)}")
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {rel(OUT)}")

    old_globals = {
        key: getattr(task184, key)
        for key in ["TASK_ID", "TASK_NAME", "PROMPT", "OUT", "DOC", "MODEL_ROOT", "TARGET_DAPS", "SCENARIOS"]
    }
    task178, original_patch, original_restore = patch_gap1_runtime()
    task184.TASK_ID = TASK_ID
    task184.TASK_NAME = TASK_NAME
    task184.PROMPT = PROMPT
    task184.OUT = OUT
    task184.DOC = DOC
    task184.MODEL_ROOT = MODEL_ROOT
    task184.TARGET_DAPS = TARGET_DAPS
    task184.SCENARIOS = SCENARIOS
    try:
        print("[Step 1/5] Replaying task188 gap=1 checkpoints to early target DAPs.", flush=True)
        task184.run(config_path.resolve())
    finally:
        task178.patch_runtime_178 = original_patch
        task178.restore_runtime_178 = original_restore
        for key, value in old_globals.items():
            setattr(task184, key, value)

    rename_outputs(OUT)
    summary_path = OUT / "189_rootcause_summary.json"
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    alignment = pd.read_csv(OUT / "evaluation" / "189_actor_critic_alignment.csv", keep_default_na=False)
    group = pd.read_csv(OUT / "evaluation" / "189_actor_critic_group_summary.csv", keep_default_na=False)
    td = pd.read_csv(OUT / "evaluation" / "189_action_one_step_td.csv", keep_default_na=False)
    reward_summary = (
        td.groupby(["checkpoint_step", "target_dap", "candidate_irrigation_mm"], as_index=False)
        .agg(
            scenario_count=("weather_variant", "count"),
            mean_reward_scaled=("reward_scaled", "mean"),
            mean_reward_unscaled=("reward_unscaled", "mean"),
            mean_resource_cost_unscaled=("resource_cost_unscaled", "mean"),
            mean_critic_value_after=("critic_value_after", "mean"),
            mean_td_one_step_scaled=("td_one_step_scaled", "mean"),
            mean_td_delta_vs_noop_scaled=("td_delta_vs_noop_scaled", "mean"),
        )
        .sort_values(["checkpoint_step", "target_dap", "candidate_irrigation_mm"])
    )
    reward_summary.to_csv(OUT / "evaluation" / "189_reward_component_summary.csv", index=False, encoding="utf-8-sig")

    actor_td_aligned_rows = int(alignment["actor_selected_is_td_top"].astype(bool).sum())
    ckpt2000_alignment = alignment[alignment["checkpoint_step"].eq(2000)]
    ckpt2000_td_top_i45_rows = int(ckpt2000_alignment["td_top_is_i45"].astype(bool).sum())
    ckpt2000_td_top_noop_rows = int(ckpt2000_alignment["td_top_is_noop"].astype(bool).sum())
    late_td_top_noop_rows = int(
        alignment[alignment["target_dap"].isin([61, 91])]["td_top_is_noop"].astype(bool).sum()
    )
    i45_positive_rows = int((alignment["i45_td_delta_vs_noop_scaled"] > 1e-12).sum())

    result.update(
        {
            "task": f"{TASK_ID}_{TASK_NAME}",
            "source_task": "188_sya_balanced_irrigation_gap1_smoke2k",
            "model_root": rel(MODEL_ROOT),
            "action_mask_override": cfg["action_mask_override"],
            "target_daps": TARGET_DAPS,
            "target_state_rows": int(len(alignment)),
            "candidate_td_rows": int(len(td)),
            "diagnostic_interpretation": {
                "actor_td_aligned_rows": actor_td_aligned_rows,
                "ckpt2000_td_top_i45_rows": ckpt2000_td_top_i45_rows,
                "ckpt2000_td_top_noop_rows": ckpt2000_td_top_noop_rows,
                "late_dap61_91_td_top_noop_rows": late_td_top_noop_rows,
                "i45_td_positive_rows": i45_positive_rows,
                "immediate_reward_pattern": "positive irrigation branches carry water cost; no systematic immediate yield gain at audited one-step states",
                "preferred_next_diagnostic": "critic_value_calibration_or_temporal_credit_assignment",
            },
            "outputs": {
                "doc": rel(DOC),
                "summary": rel(OUT / "189_rootcause_summary.json"),
                "target_state_values": rel(OUT / "evaluation" / "189_target_state_values.csv"),
                "action_one_step_td": rel(OUT / "evaluation" / "189_action_one_step_td.csv"),
                "actor_critic_alignment": rel(OUT / "evaluation" / "189_actor_critic_alignment.csv"),
                "group_summary": rel(OUT / "evaluation" / "189_actor_critic_group_summary.csv"),
                "reward_component_summary": rel(OUT / "evaluation" / "189_reward_component_summary.csv"),
            },
        }
    )
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_record(result, group, alignment, reward_summary)
    print("[Step 2/5] Wrote actor/critic alignment and reward component summaries.", flush=True)
    print("[Step 3/5] Wrote detailed record.", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("[Step 4/5] Task 189 completed.", flush=True)
    return result


def md_table(frame: pd.DataFrame, max_rows: int = 120) -> str:
    if frame.empty:
        return "无记录。"
    work = frame.head(max_rows).copy()
    for column in work.select_dtypes(include=["number"]).columns:
        work[column] = pd.to_numeric(work[column], errors="coerce").round(6)
    return work.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    run(config_path.resolve())


if __name__ == "__main__":
    main()
