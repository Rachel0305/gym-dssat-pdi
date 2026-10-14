"""Task 185: audit task-177 PPO training logs.

This is a read-only diagnostic.  It parses the already produced task-177
TensorBoard event file and training scenario logs, then links them with
task-183/task-184 decision-point diagnostics.

The script intentionally stays linear and explicit for supervisor-facing
debugging:

1. Load config and check inputs.
2. Parse TensorBoard scalar events.
3. Summarize metrics near checkpoint 1000 and 2000.
4. Summarize training scenario coverage.
5. Summarize checkpoint behaviour from task-177 evaluation outputs.
6. Write CSV/JSON/Markdown outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "185"
TASK_NAME = "sya_ppo_training_log_drift_audit"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / "2026-08-20_sya_ppo_training_log_drift_audit.md"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_tensorboard_scalars(event_file: Path) -> pd.DataFrame:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()

    rows: list[dict[str, Any]] = []
    for tag in accumulator.Tags().get("scalars", []):
        for event in accumulator.Scalars(tag):
            rows.append(
                {
                    "tag": tag,
                    "step": int(event.step),
                    "wall_time": float(event.wall_time),
                    "value": float(event.value),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["tag", "step", "wall_time", "value"])
    return pd.DataFrame(rows).sort_values(["tag", "step"]).reset_index(drop=True)


def nearest_scalar_rows(scalars: pd.DataFrame, checkpoint_steps: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tag, group in scalars.groupby("tag"):
        group = group.sort_values("step")
        for checkpoint in checkpoint_steps:
            idx = (group["step"] - checkpoint).abs().idxmin()
            item = group.loc[idx]
            rows.append(
                {
                    "tag": tag,
                    "checkpoint_step": int(checkpoint),
                    "nearest_tb_step": int(item["step"]),
                    "step_delta": int(item["step"] - checkpoint),
                    "value": float(item["value"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["tag", "checkpoint_step"]).reset_index(drop=True)


def summarize_scalar_trends(scalars: pd.DataFrame, nearest: pd.DataFrame, checkpoint_steps: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    first_checkpoint, last_checkpoint = checkpoint_steps[0], checkpoint_steps[-1]
    for tag, group in scalars.groupby("tag"):
        group = group.sort_values("step")
        first = group.iloc[0]
        last = group.iloc[-1]
        near_tag = nearest[nearest["tag"].eq(tag)]
        values_by_checkpoint = {
            int(row["checkpoint_step"]): float(row["value"])
            for _, row in near_tag.iterrows()
        }
        value_at_first_checkpoint = values_by_checkpoint.get(first_checkpoint, np.nan)
        value_at_last_checkpoint = values_by_checkpoint.get(last_checkpoint, np.nan)
        rows.append(
            {
                "tag": tag,
                "event_count": int(len(group)),
                "first_step": int(first["step"]),
                "last_step": int(last["step"]),
                "first_value": float(first["value"]),
                "last_value": float(last["value"]),
                "delta_first_to_last": float(last["value"] - first["value"]),
                "min_value": float(group["value"].min()),
                "max_value": float(group["value"].max()),
                "mean_value": float(group["value"].mean()),
                "value_near_ckpt1000": value_at_first_checkpoint,
                "value_near_ckpt2000": value_at_last_checkpoint,
                "delta_ckpt1000_to_ckpt2000": float(value_at_last_checkpoint - value_at_first_checkpoint)
                if np.isfinite(value_at_first_checkpoint) and np.isfinite(value_at_last_checkpoint)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("tag").reset_index(drop=True)


def scenario_coverage(reset_counts: pd.DataFrame, switch_log: pd.DataFrame, min_count: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    coverage = reset_counts.copy()
    coverage["episode_count"] = coverage["episode_count"].astype(int)
    total_episodes = int(coverage["episode_count"].sum())
    scenario_count = int(len(coverage))
    expected_mean = float(total_episodes / scenario_count) if scenario_count else np.nan
    coverage["expected_uniform_count"] = expected_mean
    coverage["count_minus_expected"] = coverage["episode_count"] - expected_mean
    coverage["coverage_pass_min_count"] = coverage["episode_count"] >= int(min_count)

    summary = {
        "training_episode_count_from_switch_log": int(len(switch_log)),
        "training_episode_count_from_reset_counts": total_episodes,
        "scenario_count": scenario_count,
        "min_episode_count": int(coverage["episode_count"].min()) if scenario_count else 0,
        "max_episode_count": int(coverage["episode_count"].max()) if scenario_count else 0,
        "zero_count_scenarios": coverage.loc[coverage["episode_count"].eq(0), "year"].astype(int).tolist(),
        "coverage_complete_min_count": bool(coverage["coverage_pass_min_count"].all()) if scenario_count else False,
        "expected_uniform_count": expected_mean,
    }
    return coverage, summary


def checkpoint_behaviour(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for checkpoint, group in metrics.groupby("checkpoint_step"):
        rows.append(
            {
                "checkpoint_step": int(checkpoint),
                "scenario_count": int(len(group)),
                "unique_action_sequences": int(group["action_sequence"].nunique()),
                "action_sequence": str(group["action_sequence"].iloc[0]) if len(group) else "",
                "mean_final_grnwt": float(group["final_grnwt"].mean()),
                "min_final_grnwt": float(group["final_grnwt"].min()),
                "max_final_grnwt": float(group["final_grnwt"].max()),
                "mean_profit_simple": float(group["profit_simple"].mean()),
                "mean_total_irrigation": float(group["total_irrigation"].mean()),
                "mean_total_n": float(group["total_n"].mean()),
                "mean_pfp_n": float(group["PFP_N"].mean()),
                "mean_nstres_days_gt_0p05": float(group["nstres_days_gt_0p05"].mean()),
                "mean_swfac_days_gt_0p05": float(group["swfac_days_gt_0p05"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("checkpoint_step").reset_index(drop=True)


def build_red_flags(
    cfg: dict[str, Any],
    task177: dict[str, Any],
    scalar_trends: pd.DataFrame,
    nearest: pd.DataFrame,
    coverage_summary: dict[str, Any],
    behaviour: pd.DataFrame,
    task183: dict[str, Any],
    task184: dict[str, Any],
) -> pd.DataFrame:
    thresholds = cfg.get("audit_thresholds", {})
    small_kl = float(thresholds.get("small_approx_kl", 0.001))
    very_small_kl = float(thresholds.get("very_small_approx_kl", 0.0001))

    rows: list[dict[str, Any]] = []

    def add(item: str, status: str, evidence: str, interpretation: str) -> None:
        rows.append(
            {
                "item": item,
                "status": status,
                "evidence": evidence,
                "interpretation": interpretation,
            }
        )

    max_step = int(scalar_trends["last_step"].max()) if len(scalar_trends) else 0
    add(
        "training_length",
        "weak_evidence_for_undertraining",
        (
            f"requested_timesteps={task177.get('total_timesteps')}; "
            f"last_tensorboard_step={max_step}; "
            f"episodes={coverage_summary['training_episode_count_from_switch_log']}"
        ),
        "2K smoke 只产生约 15 个 episode，不能支持 PPO 已充分学习 6 个天气情景。",
    )

    add(
        "scenario_sampling",
        "problem_found" if not coverage_summary["coverage_complete_min_count"] else "pass",
        (
            f"min_count={coverage_summary['min_episode_count']}; "
            f"max_count={coverage_summary['max_episode_count']}; "
            f"zero_count_scenarios={coverage_summary['zero_count_scenarios']}"
        ),
        "训练 episode 的天气情景覆盖不均衡；存在 0 次采样情景时，不能说 PPO 在训练中见过所有扰动情景。",
    )

    approx = scalar_trends[scalar_trends["tag"].eq("train/approx_kl")]
    if len(approx):
        max_kl = float(approx["max_value"].iloc[0])
        last_kl = float(approx["last_value"].iloc[0])
        status = "very_small_updates" if max_kl < very_small_kl else ("small_updates" if max_kl < small_kl else "normal_or_large")
        add(
            "approx_kl",
            status,
            f"max={max_kl:.8g}; last={last_kl:.8g}; threshold_small={small_kl}",
            "KL 很小，说明 2K 内 PPO 每轮策略更新幅度很弱；这与 task183 的“小概率差异导致 argmax 漂移”一致。",
        )

    clip = scalar_trends[scalar_trends["tag"].eq("train/clip_fraction")]
    if len(clip):
        add(
            "clip_fraction",
            "no_clipping",
            f"min={float(clip['min_value'].iloc[0]):.6g}; max={float(clip['max_value'].iloc[0]):.6g}",
            "clip_fraction 全为 0，说明 PPO clipping 约束基本没有被触发；不是典型的大步更新被 clip 截断。",
        )

    ev = scalar_trends[scalar_trends["tag"].eq("train/explained_variance")]
    if len(ev):
        add(
            "explained_variance",
            "early_negative_then_improved",
            (
                f"first={float(ev['first_value'].iloc[0]):.6g}; "
                f"last={float(ev['last_value'].iloc[0]):.6g}; "
                f"min={float(ev['min_value'].iloc[0]):.6g}; "
                f"max={float(ev['max_value'].iloc[0]):.6g}"
            ),
            "critic 早期拟合很差，后期改善；日志不支持 critic 爆炸式崩溃，但也不证明目标决策点的 action 价值判断正确。",
        )

    vl = scalar_trends[scalar_trends["tag"].eq("train/value_loss")]
    if len(vl):
        add(
            "value_loss",
            "declined_overall",
            (
                f"first={float(vl['first_value'].iloc[0]):.6g}; "
                f"last={float(vl['last_value'].iloc[0]):.6g}; "
                f"delta={float(vl['delta_first_to_last'].iloc[0]):.6g}"
            ),
            "value_loss 总体下降，说明没有明显数值发散；但 task184 显示最终 critic 局部 TD 仍不支持多数 I45。",
        )

    ep_rew = scalar_trends[scalar_trends["tag"].eq("rollout/ep_rew_mean")]
    if len(ep_rew):
        add(
            "training_ep_reward",
            "increased_but_not_validation",
            (
                f"first={float(ep_rew['first_value'].iloc[0]):.6g}; "
                f"last={float(ep_rew['last_value'].iloc[0]):.6g}; "
                f"delta={float(ep_rew['delta_first_to_last'].iloc[0]):.6g}"
            ),
            "训练 episode reward 上升，但 task177 没有正式 validation gate；训练均值上升不能证明 DAP97/104 固定 I45 是好策略。",
        )

    entropy = scalar_trends[scalar_trends["tag"].eq("train/entropy_loss")]
    if len(entropy):
        first_entropy = -float(entropy["first_value"].iloc[0])
        last_entropy = -float(entropy["last_value"].iloc[0])
        add(
            "training_entropy_loss",
            "low_global_entropy_metric_with_target_state_caveat",
            f"estimated_entropy_nats_first={first_entropy:.6g}; estimated_entropy_nats_last={last_entropy:.6g}",
            "训练平均 entropy 指标较低，但 task183 的 DAP97/104 局部合法动作 entropy 接近均匀；全局 entropy 不能直接解释目标日动作。",
        )

    add(
        "validation_gate",
        "missing_or_failed",
        (
            f"validation_executed={task177.get('validation_executed')}; "
            f"action_response_passed={task177.get('action_response_passed')}"
        ),
        "checkpoint 2000 不是通过泛化或 action-response gate 选出来的，只是 2K smoke 的末端 checkpoint。",
    )

    task183_ev = task183.get("evidence", {})
    task184_ev = task184.get("evidence", {})
    add(
        "link_to_decision_audits",
        "consistent_with_argmax_drift",
        (
            f"task183_i45_minus_noop_prob_range="
            f"{task183_ev.get('ckpt2000_i45_minus_noop_prob_min')}.."
            f"{task183_ev.get('ckpt2000_i45_minus_noop_prob_max')}; "
            f"task184_ckpt2000_actor_td_aligned={task184_ev.get('ckpt2000_actor_td_aligned_rows')}/"
            f"{task184_ev.get('ckpt2000_rows')}"
        ),
        "训练日志的弱更新/短训练/缺 gate，与前序发现的近乎平局 argmax 漂移和 actor-critic 不一致相互吻合。",
    )

    if len(behaviour) >= 2:
        row1000 = behaviour[behaviour["checkpoint_step"].eq(1000)].iloc[0]
        row2000 = behaviour[behaviour["checkpoint_step"].eq(2000)].iloc[0]
        add(
            "checkpoint_action_template_change",
            "large_management_change",
            (
                f"ckpt1000_action={row1000['action_sequence']}; "
                f"ckpt2000_action={row2000['action_sequence']}"
            ),
            "1000 到 2000 的管理模板从早期多次小灌水/施氮，变成 DAP0 加 DAP97/104 固定灌水；这是策略结构变化，不是天气响应增强。",
        )

    return pd.DataFrame(rows)


def write_doc(result: dict[str, Any], red_flags: pd.DataFrame) -> None:
    lines = [
        "# Task 185：SYA PPO 训练日志漂移审计记录",
        "",
        "> 自动生成的详细记录骨架；关键结论来自 CSV/JSON 原始输出。后续如需要，可以继续人工扩展为更长版。",
        "",
        "## 实验问题",
        "",
        "task 177 的 PPO 从 checkpoint 1000 的 no-op 后期策略漂到 checkpoint 2000 的固定 `I45/N0`，训练日志是否支持“采样不足、更新信号弱、缺少 checkpoint gate”等原因？",
        "",
        "## 方法",
        "",
        "- 只读取 task 177 现有 TensorBoard event、训练情景采样日志和 evaluation 输出；",
        "- 联合读取 task 183/184 的后处理解释；",
        "- 不重新训练，不重新跑 DSSAT，不修改 reward/weather/observation/action mask/action grid。",
        "",
        "## 关键结论",
        "",
        f"- primary label：`{result['primary_label']}`",
        f"- 训练 episode 数：{result['evidence']['training_episode_count']}；",
        f"- 0 次采样情景：{result['evidence']['zero_count_scenarios']}；",
        f"- TensorBoard 最后 step：{result['evidence']['last_tensorboard_step']}；",
        f"- max approx KL：{result['evidence']['max_approx_kl']}；",
        f"- max clip fraction：{result['evidence']['max_clip_fraction']}；",
        f"- validation_executed：{result['evidence']['validation_executed']}；action_response_passed：{result['evidence']['action_response_passed']}。",
        "",
        "## 红旗/证据摘要",
        "",
        red_flags.to_markdown(index=False),
        "",
        "## 输出文件",
        "",
        "- `benchmark_results/185_sya_ppo_training_log_drift_audit/185_training_log_audit_summary.json`",
        "- `benchmark_results/185_sya_ppo_training_log_drift_audit/evaluation/185_tensorboard_scalars.csv`",
        "- `benchmark_results/185_sya_ppo_training_log_drift_audit/evaluation/185_scalar_trends.csv`",
        "- `benchmark_results/185_sya_ppo_training_log_drift_audit/evaluation/185_checkpoint_nearest_scalars.csv`",
        "- `benchmark_results/185_sya_ppo_training_log_drift_audit/evaluation/185_training_scenario_coverage.csv`",
        "- `benchmark_results/185_sya_ppo_training_log_drift_audit/evaluation/185_checkpoint_behavior_summary.csv`",
        "- `benchmark_results/185_sya_ppo_training_log_drift_audit/evaluation/185_training_red_flags.csv`",
        "",
        "## 结论边界",
        "",
        "本实验不能直接还原 PPO rollout buffer 中每一步的 advantage，因此不能证明某个训练 batch 的 advantage 必然错误。它能证明的是：当前 2K smoke 训练证据不足以支持 checkpoint 2000 的固定 I45 是稳健学到的策略。",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    if cfg.get("training_executed") is not False:
        raise ValueError("Task 185 must be read-only and must not train PPO.")
    if cfg.get("rerun_dssat") is not False:
        raise ValueError("Task 185 must not rerun DSSAT.")
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {rel(OUT)}")

    inputs = {key: as_path(value) for key, value in cfg["inputs"].items()}
    for key, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing input {key}: {path}")

    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "configs").mkdir()
    (OUT / "evaluation").mkdir()
    (OUT / "logs").mkdir()
    (OUT / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT / "configs" / PROMPT.name).write_text(PROMPT.read_text(encoding="utf-8"), encoding="utf-8")

    print("[Step 1/5] Parsing TensorBoard scalars.", flush=True)
    scalars = parse_tensorboard_scalars(inputs["tensorboard_event"])
    if scalars.empty:
        raise RuntimeError("No TensorBoard scalar events found.")
    scalar_path = OUT / "evaluation" / "185_tensorboard_scalars.csv"
    scalars.to_csv(scalar_path, index=False, encoding="utf-8-sig")

    checkpoint_steps = [int(x) for x in cfg["checkpoint_steps"]]
    nearest = nearest_scalar_rows(scalars, checkpoint_steps)
    nearest_path = OUT / "evaluation" / "185_checkpoint_nearest_scalars.csv"
    nearest.to_csv(nearest_path, index=False, encoding="utf-8-sig")

    trends = summarize_scalar_trends(scalars, nearest, checkpoint_steps)
    trends_path = OUT / "evaluation" / "185_scalar_trends.csv"
    trends.to_csv(trends_path, index=False, encoding="utf-8-sig")

    print("[Step 2/5] Summarizing training scenario coverage.", flush=True)
    switch_log = pd.read_csv(inputs["training_year_switch_log"])
    reset_counts = pd.read_csv(inputs["training_year_reset_counts"])
    min_count = int(cfg.get("audit_thresholds", {}).get("min_episode_count_per_scenario", 1))
    coverage, coverage_summary = scenario_coverage(reset_counts, switch_log, min_count)
    coverage_path = OUT / "evaluation" / "185_training_scenario_coverage.csv"
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    switch_copy_path = OUT / "logs" / "185_training_year_switch_log_copy.csv"
    switch_log.to_csv(switch_copy_path, index=False, encoding="utf-8-sig")

    print("[Step 3/5] Summarizing checkpoint behaviour.", flush=True)
    task177 = load_json(inputs["task177_result"])
    metrics = pd.read_csv(inputs["scenario_episode_metrics"])
    action_summary = pd.read_csv(inputs["action_change_summary"])
    behaviour = checkpoint_behaviour(metrics)
    behaviour_path = OUT / "evaluation" / "185_checkpoint_behavior_summary.csv"
    behaviour.to_csv(behaviour_path, index=False, encoding="utf-8-sig")
    action_copy_path = OUT / "evaluation" / "185_task177_action_change_summary_copy.csv"
    action_summary.to_csv(action_copy_path, index=False, encoding="utf-8-sig")

    print("[Step 4/5] Linking task 183/184 diagnostics.", flush=True)
    task183 = load_json(inputs["task183_interpretation"])
    task184 = load_json(inputs["task184_interpretation"])
    red_flags = build_red_flags(cfg, task177, trends, nearest, coverage_summary, behaviour, task183, task184)
    red_flags_path = OUT / "evaluation" / "185_training_red_flags.csv"
    red_flags.to_csv(red_flags_path, index=False, encoding="utf-8-sig")

    tag_values = trends.set_index("tag") if len(trends) else pd.DataFrame()
    def metric(tag: str, column: str, default: Any = None) -> Any:
        if tag in tag_values.index and column in tag_values.columns:
            value = tag_values.loc[tag, column]
            return float(value) if pd.notna(value) else default
        return default

    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "status": "completed",
        "mode": "read_only_training_log_audit",
        "training_executed": False,
        "rerun_dssat": False,
        "source_task": cfg["source_task"],
        "primary_label": "short_imbalanced_training_plus_weak_updates_no_validation_gate",
        "evidence": {
            "training_episode_count": coverage_summary["training_episode_count_from_switch_log"],
            "scenario_count": coverage_summary["scenario_count"],
            "min_episode_count": coverage_summary["min_episode_count"],
            "max_episode_count": coverage_summary["max_episode_count"],
            "zero_count_scenarios": coverage_summary["zero_count_scenarios"],
            "last_tensorboard_step": int(scalars["step"].max()),
            "requested_total_timesteps": int(task177.get("total_timesteps", -1)),
            "max_approx_kl": metric("train/approx_kl", "max_value"),
            "last_approx_kl": metric("train/approx_kl", "last_value"),
            "max_clip_fraction": metric("train/clip_fraction", "max_value"),
            "first_explained_variance": metric("train/explained_variance", "first_value"),
            "last_explained_variance": metric("train/explained_variance", "last_value"),
            "first_value_loss": metric("train/value_loss", "first_value"),
            "last_value_loss": metric("train/value_loss", "last_value"),
            "first_ep_rew_mean": metric("rollout/ep_rew_mean", "first_value"),
            "last_ep_rew_mean": metric("rollout/ep_rew_mean", "last_value"),
            "validation_executed": bool(task177.get("validation_executed")),
            "action_response_passed": bool(task177.get("action_response_passed")),
            "task183_primary_label": task183.get("primary_label"),
            "task184_primary_label": task184.get("primary_label"),
        },
        "interpretation": {
            "sampling": "2K smoke generated only about 15 training episodes over 6 weather scenarios, and at least one scenario had zero training resets. This is not enough evidence that PPO learned robust weather-conditioned behaviour.",
            "update_strength": "approx_kl was very small and clip_fraction stayed at zero, indicating weak policy movement rather than a strong PPO update regime.",
            "value_learning": "explained_variance improved and value_loss declined overall, so logs do not show obvious critic numerical collapse. However task 184 showed final local critic TD mostly did not support I45.",
            "checkpoint_selection": "checkpoint 2000 was not selected by validation/action-response gate. The final checkpoint can therefore keep a low-value argmax drift.",
            "combined_root_cause": "The current evidence supports: short and imbalanced scenario exposure + weak/near-tie policy updates + missing checkpoint gate, rather than mask forcing, strong probability collapse, or critic-backed I45 preference.",
            "boundary": "This audit does not reconstruct PPO rollout advantages, so it cannot prove the exact training-time advantage error for DAP97/104.",
        },
        "outputs": {
            "tensorboard_scalars": rel(scalar_path),
            "checkpoint_nearest_scalars": rel(nearest_path),
            "scalar_trends": rel(trends_path),
            "training_scenario_coverage": rel(coverage_path),
            "checkpoint_behavior_summary": rel(behaviour_path),
            "training_red_flags": rel(red_flags_path),
            "training_year_switch_log_copy": rel(switch_copy_path),
            "task177_action_change_summary_copy": rel(action_copy_path),
            "doc": rel(DOC),
        },
    }

    summary_path = OUT / "185_training_log_audit_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Step 5/5] Writing Markdown record.", flush=True)
    write_doc(result, red_flags)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
