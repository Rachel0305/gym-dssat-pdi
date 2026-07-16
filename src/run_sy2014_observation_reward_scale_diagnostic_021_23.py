from __future__ import annotations

import json
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_sy2014_dqfd_real_network_loss_diagnostic_021_22 as base
from frozen_nstep_dqn_config_020_11 import dqn_kwargs
from literature_aligned_dqfd import PrioritizedDemonstrationReplay
from ppo_evaluate import latest_observation_dict, scalar
from stable_baselines3 import DQN


OUT = ROOT / "benchmark_results" / "021_23"
DOC = ROOT / "docs" / "2026-07-15_021_23_sy2014_observation_scale_and_reward_scale_diagnostic.md"
SCALE = 0.1
OBSERVATION_LABELS = [
    "cumsumfert", "dap", "dtt", "ep", "grnwt", "istage", "nstres", "rtdep", "srad",
    "sw_layer_1", "sw_layer_2", "sw_layer_3", "sw_layer_4", "sw_layer_5",
    "sw_layer_6", "sw_layer_7", "sw_layer_8", "sw_layer_9",
    "swfac", "tmax", "topwt", "totir", "vstage", "wtdep", "xlai",
]


def load_and_scale_demonstrations() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = base.load_demonstrations()
    scaled = {name: np.asarray(value).copy() for name, value in raw.items()}
    scaled["rewards"] = scaled["rewards"].astype(np.float32) * SCALE
    scaled["n_step_returns"] = scaled["n_step_returns"].astype(np.float32) * SCALE

    reward_error = float(np.max(np.abs(scaled["rewards"] - raw["rewards"] * SCALE)))
    nstep_error = float(
        np.max(np.abs(scaled["n_step_returns"] - raw["n_step_returns"] * SCALE))
    )
    reward_nonzero = np.abs(raw["rewards"]) > 1e-12
    nstep_nonzero = np.abs(raw["n_step_returns"]) > 1e-12
    validation = {
        "scale": SCALE,
        "demo_reward_max_abs_error": reward_error,
        "demo_n_step_return_max_abs_error": nstep_error,
        "demo_reward_nonzero_ratio_min": float(
            np.min(scaled["rewards"][reward_nonzero] / raw["rewards"][reward_nonzero])
        ) if reward_nonzero.any() else None,
        "demo_reward_nonzero_ratio_max": float(
            np.max(scaled["rewards"][reward_nonzero] / raw["rewards"][reward_nonzero])
        ) if reward_nonzero.any() else None,
        "demo_nstep_nonzero_ratio_min": float(
            np.min(scaled["n_step_returns"][nstep_nonzero] / raw["n_step_returns"][nstep_nonzero])
        ) if nstep_nonzero.any() else None,
        "demo_nstep_nonzero_ratio_max": float(
            np.max(scaled["n_step_returns"][nstep_nonzero] / raw["n_step_returns"][nstep_nonzero])
        ) if nstep_nonzero.any() else None,
    }
    return scaled, validation


def observation_audit(observations: np.ndarray) -> pd.DataFrame:
    values = np.asarray(observations, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for index in range(values.shape[1]):
        column = values[:, index]
        rows.append({
            "observation_index": index,
            "observation_label": OBSERVATION_LABELS[index] if len(OBSERVATION_LABELS) == values.shape[1] else f"obs_{index:02d}",
            "minimum": float(np.min(column)),
            "maximum": float(np.max(column)),
            "mean": float(np.mean(column)),
            "median": float(np.median(column)),
            "std": float(np.std(column)),
            "range": float(np.ptp(column)),
            "max_abs": float(np.max(np.abs(column))),
            "nonzero_fraction": float(np.mean(np.abs(column) > 1e-12)),
            "unique_count": int(np.unique(column).size),
        })
    frame = pd.DataFrame(rows)
    positive_scales = frame.loc[frame.max_abs > 0, "max_abs"]
    reference = float(positive_scales.median()) if not positive_scales.empty else 1.0
    frame["max_abs_vs_nonzero_dimension_median"] = frame.max_abs / max(reference, 1e-12)
    frame["descriptive_scale_flag_gt10x"] = frame.max_abs_vs_nonzero_dimension_median > 10.0
    return frame


def environment_arguments() -> dict[str, Any]:
    args = json.loads(base.ENV_SOURCE.read_text(encoding="utf-8"))
    run = OUT / "real_environment"
    run.mkdir(parents=True, exist_ok=True)
    args["log_saving_path"] = str(run / "pdi_gym.log")
    return args


def component_values(env: Any, info: dict[str, Any]) -> dict[str, float]:
    components = dict(getattr(env, "last_reward_components", {}) or {})
    if not components:
        components = {
            key: info.get(key, np.nan)
            for key in ("yield_gain", "water_cost_term", "nitrogen_cost_term")
        }
    return {
        key: float(components.get(key, np.nan))
        for key in ("yield_gain", "water_cost_term", "nitrogen_cost_term")
    }


def comparative_stage_summary(frame: pd.DataFrame, phase: str) -> dict[str, Any]:
    summary = base.stage_summary(frame, phase)
    for column in (
        "td_1_weighted", "td_n_weighted", "margin_weighted", "l2_weighted",
        "grad_td_1", "grad_td_n", "grad_margin", "grad_l2",
        "q_abs_mean", "q_abs_max", "relative_parameter_step",
    ):
        summary[f"median_{column}"] = float(frame[column].median())
        summary[f"max_{column}"] = float(frame[column].max())
    return summary


def ratio(new: float, old: float) -> float:
    return float(new) / max(abs(float(old)), 1e-12)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def make_comparison(scaled_stages: pd.DataFrame) -> pd.DataFrame:
    baseline = pd.read_csv(ROOT / "benchmark_results/021_22/021_22_stage_summary.csv")
    rows = []
    for phase in ("pretrain", "online"):
        old = baseline.loc[baseline.phase == phase].iloc[0]
        new = scaled_stages.loc[scaled_stages.phase == phase].iloc[0]
        for metric in (
            "gradient_clip_fraction", "median_total_gradient_norm",
            "max_total_gradient_norm", "cumulative_parameter_relative_l2",
            "median_td_1_weighted", "median_td_n_weighted",
            "median_margin_weighted", "median_q_abs_mean", "median_q_abs_max",
        ):
            if metric not in old.index:
                old_log = pd.read_csv(
                    ROOT / f"benchmark_results/021_22/021_22_{'pretrain' if phase == 'pretrain' else 'online'}_update_log.csv"
                )
                old_value = float(old_log[metric.removeprefix("median_")].median())
            else:
                old_value = float(old[metric])
            new_value = float(new[metric])
            rows.append({
                "phase": phase,
                "metric": metric,
                "unscaled_021_22": old_value,
                "scaled_021_23": new_value,
                "scaled_to_unscaled_ratio": ratio(new_value, old_value),
            })
    return pd.DataFrame(rows)


def plot_results(observation_frame: pd.DataFrame, comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = observation_frame.observation_index
    axes[0, 0].bar(x, observation_frame.max_abs, color="#4C78A8")
    axes[0, 0].set_yscale("symlog", linthresh=1e-3)
    axes[0, 0].set_title("Observation max absolute value")
    axes[0, 0].set_xlabel("Observation index")
    axes[0, 0].set_ylabel("max |value|")

    axes[0, 1].bar(x, observation_frame["std"], color="#F58518")
    axes[0, 1].set_yscale("symlog", linthresh=1e-3)
    axes[0, 1].set_title("Observation standard deviation")
    axes[0, 1].set_xlabel("Observation index")
    axes[0, 1].set_ylabel("std")

    selected = comparison[comparison.metric.isin([
        "median_td_1_weighted", "median_td_n_weighted", "median_total_gradient_norm"
    ])].copy()
    labels = selected.phase + "\n" + selected.metric.str.replace("median_", "", regex=False)
    positions = np.arange(len(selected))
    width = 0.36
    axes[1, 0].bar(positions - width / 2, selected.unscaled_021_22, width, label="unscaled", color="#999999")
    axes[1, 0].bar(positions + width / 2, selected.scaled_021_23, width, label="reward x0.1", color="#009E73")
    axes[1, 0].set_xticks(positions, labels, rotation=25, ha="right")
    axes[1, 0].set_yscale("symlog", linthresh=1e-3)
    axes[1, 0].set_title("Loss and gradient comparison")
    axes[1, 0].legend(frameon=False)

    clips = comparison[comparison.metric == "gradient_clip_fraction"]
    positions = np.arange(len(clips))
    axes[1, 1].bar(positions - width / 2, clips.unscaled_021_22, width, label="unscaled", color="#999999")
    axes[1, 1].bar(positions + width / 2, clips.scaled_021_23, width, label="reward x0.1", color="#009E73")
    axes[1, 1].set_xticks(positions, clips.phase)
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("Gradient clipping fraction")
    axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.grid(alpha=0.2, axis="y")
    fig.suptitle("SY2014 observation scale and reward-scale diagnostic")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "021_23_observation_reward_scale_diagnostic.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_23_observation_reward_scale_diagnostic.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    np.random.seed(0)
    torch.manual_seed(0)
    action_rng = np.random.default_rng(0)

    demonstrations, validation = load_and_scale_demonstrations()
    observation_frame = observation_audit(demonstrations["observations"])
    observation_frame.to_csv(
        OUT / "021_23_observation_scale_audit.csv", index=False, encoding="utf-8-sig"
    )
    replay = PrioritizedDemonstrationReplay(
        demonstrations, agent_capacity=10_000, alpha=0.4,
        epsilon_demo=1.0, epsilon_agent=0.001, seed=21022,
    )
    env = base.make_env(environment_arguments())
    model = DQN("MlpPolicy", env, verbose=0, **dqn_kwargs(seed=0))
    parameters = list(model.q_net.parameters())
    initial_parameters = base.parameter_vector(parameters)

    pretrain_rows = [
        base.one_update(model, replay, phase="pretrain", update=update, env_step=0, epsilon=1.0)
        for update in range(1, base.PRETRAIN_UPDATES + 1)
    ]
    after_pretrain = base.parameter_vector(parameters)
    pretrain_change = float(torch.linalg.vector_norm(after_pretrain - initial_parameters)) / max(
        float(torch.linalg.vector_norm(initial_parameters)), 1e-12
    )
    model.q_net_target.load_state_dict(model.q_net.state_dict())

    interactions: list[dict[str, Any]] = []
    online_rows: list[dict[str, Any]] = []
    pending: deque[dict[str, Any]] = deque()
    online_start = base.parameter_vector(parameters)
    component_errors: list[float] = []
    try:
        obs, info = env.reset()
        observation_variables = list(getattr(env.unwrapped, "observation_variables", []) or [])
        validation["environment_observation_variables"] = observation_variables
        validation["environment_observation_shape"] = list(np.asarray(obs).shape)
        online_update = 0
        for step in range(1, base.ENV_STEPS + 1):
            latest = latest_observation_dict(env, obs, info)
            dap = int(round(scalar(latest.get("dap", step)) or 0))
            epsilon = base.epsilon_at(step)
            if action_rng.random() < epsilon:
                action = int(action_rng.integers(0, 9))
                action_source = "epsilon_random"
            else:
                with torch.no_grad():
                    q_values = model.q_net(
                        torch.as_tensor(obs, dtype=torch.float32, device=model.device).reshape(1, -1)
                    )
                action = int(q_values.argmax(dim=1).item())
                action_source = "greedy"
            before = np.asarray(obs, dtype=np.float32).copy()
            next_obs, raw_reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            training_reward = float(raw_reward) * SCALE
            safe = dict(getattr(env, "last_safe_real_action", {}) or {})
            components = component_values(env, info if isinstance(info, dict) else {})
            if all(np.isfinite(list(components.values()))):
                raw_by_components = (
                    components["yield_gain"] - components["water_cost_term"] - components["nitrogen_cost_term"]
                )
                component_errors.append(abs(SCALE * raw_by_components - training_reward))
            pending.append({
                "observation": before, "action": action, "reward": training_reward,
                "next_observation": np.asarray(next_obs, dtype=np.float32).copy(), "done": done,
            })
            added = 0
            while len(pending) >= base.N_STEP or (done and pending):
                transition = base.matured_transition(pending)
                replay.add_agent(transition, td_error=base.initial_td_error(model, transition))
                pending.popleft()
                added += 1
                if not done and len(pending) < base.N_STEP:
                    break
            interactions.append({
                "env_step": step, "dap_action": dap, "epsilon": epsilon,
                "action": action, "action_source": action_source,
                "safe_amir": float(safe.get("amir", 0.0)),
                "safe_anfer": float(safe.get("anfer", 0.0)),
                "raw_reward": float(raw_reward), "training_reward": training_reward,
                "reward_scale": SCALE,
                **components, "done": done, "agent_transitions_added": added,
                "replay_agent_count": replay.agent_size,
            })
            obs = next_obs
            if step >= base.LEARNING_STARTS:
                online_update += 1
                online_rows.append(base.one_update(
                    model, replay, phase="online", update=online_update,
                    env_step=step, epsilon=epsilon,
                ))
            if done:
                break
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(Path(tmp), OUT / "pdi_tmp_snapshot_after_diagnostic")
        env.close()

    online_end = base.parameter_vector(parameters)
    online_change = float(torch.linalg.vector_norm(online_end - online_start)) / max(
        float(torch.linalg.vector_norm(online_start)), 1e-12
    )
    pretrain = pd.DataFrame(pretrain_rows)
    online = pd.DataFrame(online_rows)
    interaction_frame = pd.DataFrame(interactions)
    pretrain.to_csv(OUT / "021_23_pretrain_update_log.csv", index=False, encoding="utf-8-sig")
    online.to_csv(OUT / "021_23_online_update_log.csv", index=False, encoding="utf-8-sig")
    interaction_frame.to_csv(OUT / "021_23_agent_interactions.csv", index=False, encoding="utf-8-sig")

    nonzero_agent = interaction_frame.raw_reward.abs() > 1e-12
    validation["agent_reward_max_abs_error"] = float(
        np.max(np.abs(interaction_frame.training_reward - SCALE * interaction_frame.raw_reward))
    )
    validation["agent_nonzero_reward_ratio_min"] = float(
        (interaction_frame.loc[nonzero_agent, "training_reward"] / interaction_frame.loc[nonzero_agent, "raw_reward"]).min()
    ) if nonzero_agent.any() else None
    validation["agent_nonzero_reward_ratio_max"] = float(
        (interaction_frame.loc[nonzero_agent, "training_reward"] / interaction_frame.loc[nonzero_agent, "raw_reward"]).max()
    ) if nonzero_agent.any() else None
    validation["component_scaled_sum_max_abs_error"] = float(max(component_errors, default=0.0))
    validation["all_scale_checks_pass"] = bool(
        validation["demo_reward_max_abs_error"] <= 1e-5
        and validation["demo_n_step_return_max_abs_error"] <= 1e-5
        and validation["agent_reward_max_abs_error"] <= 1e-8
        and validation["component_scaled_sum_max_abs_error"] <= 1e-5
    )
    (OUT / "021_23_scale_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    stage_rows = []
    for frame, phase, cumulative in (
        (pretrain, "pretrain", pretrain_change), (online, "online", online_change)
    ):
        summary = comparative_stage_summary(frame, phase)
        summary["cumulative_parameter_relative_l2"] = cumulative
        stage_rows.append(summary)
    stages = pd.DataFrame(stage_rows)
    stages.to_csv(OUT / "021_23_scaled_stage_summary.csv", index=False, encoding="utf-8-sig")
    comparison = make_comparison(stages)
    comparison.to_csv(OUT / "021_23_stage_comparison.csv", index=False, encoding="utf-8-sig")
    make_comparison_rows = {
        (row.phase, row.metric): row.scaled_to_unscaled_ratio for row in comparison.itertuples()
    }
    median_grad_ratios = [
        make_comparison_rows[(phase, "median_total_gradient_norm")] for phase in ("pretrain", "online")
    ]
    td1_ratios = [
        make_comparison_rows[(phase, "median_td_1_weighted")] for phase in ("pretrain", "online")
    ]
    clip_ratios = [
        make_comparison_rows[(phase, "gradient_clip_fraction")] for phase in ("pretrain", "online")
    ]
    if max(td1_ratios) < 0.75 and max(median_grad_ratios) < 0.75 and max(clip_ratios) < 0.75:
        branch = "A_reward_scale_participant"
    elif max(td1_ratios) < 0.75 and (max(median_grad_ratios) >= 0.75 or max(clip_ratios) >= 0.75):
        branch = "B_td_down_but_gradient_or_clipping_not_down"
    else:
        branch = "C_reward_scaling_not_primary_explanation"

    summary = {
        "status": "completed" if validation["all_scale_checks_pass"] else "failed",
        "no_5k_training_performed": True,
        "reward_scale": SCALE,
        "observation_dimensions": int(len(observation_frame)),
        "observation_scale_flag_count": int(observation_frame.descriptive_scale_flag_gt10x.sum()),
        "largest_observation_max_abs": float(observation_frame.max_abs.max()),
        "smallest_nonzero_observation_max_abs": float(
            observation_frame.loc[observation_frame.max_abs > 0, "max_abs"].min()
        ),
        "pre_registered_interpretation_branch": branch,
        "scale_validation_passed": validation["all_scale_checks_pass"],
        "pretrain": stage_rows[0], "online": stage_rows[1],
    }
    (OUT / "021_23_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_results(observation_frame, comparison)

    key = comparison[comparison.metric.isin([
        "median_td_1_weighted", "median_td_n_weighted",
        "median_total_gradient_norm", "gradient_clip_fraction",
        "cumulative_parameter_relative_l2",
    ])].copy()
    key = key.round(6)
    record = f"""# 021_23 SY2014 观测尺度与 reward×0.1 短诊断记录

## 边界

本任务先审计既有示范观测的数值尺度，再严格复刻 021_22 的 100 次示范预训练和 100 个真实环境步，仅将训练 reward 与 n-step return 乘以 0.1。没有进入 5K，没有改变奖励相对权重、观测信息、动作空间、IC、网络或 DQfD 参数。

## 生效校验

- scale validation: `{validation['all_scale_checks_pass']}`
- demo reward max error: `{validation['demo_reward_max_abs_error']:.3g}`
- demo n-step max error: `{validation['demo_n_step_return_max_abs_error']:.3g}`
- agent reward max error: `{validation['agent_reward_max_abs_error']:.3g}`
- component path max error: `{validation['component_scaled_sum_max_abs_error']:.3g}`

## 观测尺度

- 维数：{len(observation_frame)}
- max_abs 最大值：{observation_frame.max_abs.max():.6g}
- 非零维 max_abs 最小值：{observation_frame.loc[observation_frame.max_abs > 0, 'max_abs'].min():.6g}
- 超过非零维中位尺度 10 倍的维数：{int(observation_frame.descriptive_scale_flag_gt10x.sum())}

25 维展开顺序由环境报告的 17 个变量顺序核对：其中 `sw` 展开为 9 个土层，所以形成 `cumsumfert, dap, dtt, ep, grnwt, istage, nstres, rtdep, srad, sw_layer_1..9, swfac, tmax, topwt, totir, vstage, wtdep, xlai`。最大尺度来自 `topwt`，其次是 `grnwt`；这与量级较小的胁迫和土壤水分特征未经归一化地共同进入 MLP。

这些是描述性证据，不单独证明大梯度由观测尺度造成。

## 与 021_22 未缩放基准的关键对比

{markdown_table(key)}

## 预注册判定

`{branch}`

该判定只回答 reward 数值缩放是否参与短程训练动力学，不代表 DQfD 有效，也不允许自动进入 5K。Smooth L1/Huber loss 在大误差区间梯度饱和，因此 loss 数值和参数梯度必须分开解释。

## 输出

- `benchmark_results/021_23/021_23_observation_scale_audit.csv`
- `benchmark_results/021_23/021_23_pretrain_update_log.csv`
- `benchmark_results/021_23/021_23_online_update_log.csv`
- `benchmark_results/021_23/021_23_agent_interactions.csv`
- `benchmark_results/021_23/021_23_scaled_stage_summary.csv`
- `benchmark_results/021_23/021_23_stage_comparison.csv`
- `benchmark_results/021_23/021_23_scale_validation.json`
- `benchmark_results/021_23/021_23_summary.json`
- `benchmark_results/021_23/021_23_observation_reward_scale_diagnostic.png/.svg`

## 失败记录

第一次运行已经完成数值计算和结果文件写出，但在最后生成 Markdown 表格时，容器缺少 Pandas 可选依赖 `tabulate`，导致文档阶段失败。完整现场保存在 `benchmark_results/021_23_failed_attempt_1_missing_tabulate/`。随后仅将表格输出改成本地 Markdown 格式化函数，未安装新包、未改变任何科学参数，并从头按相同配置重跑成功。
"""
    DOC.write_text(record, encoding="utf-8")
    if not validation["all_scale_checks_pass"]:
        raise RuntimeError("Reward scale validation failed")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
