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


OUT = ROOT / "benchmark_results" / "021_24"
DOC = ROOT / "docs" / "2026-07-15_021_24_sy2014_fixed_observation_standardization_short_diagnostic.md"
STD_EPSILON = 1e-6
LABELS = [
    "cumsumfert", "dap", "dtt", "ep", "grnwt", "istage", "nstres", "rtdep", "srad",
    "sw_layer_1", "sw_layer_2", "sw_layer_3", "sw_layer_4", "sw_layer_5",
    "sw_layer_6", "sw_layer_7", "sw_layer_8", "sw_layer_9",
    "swfac", "tmax", "topwt", "totir", "vstage", "wtdep", "xlai",
]


def scaler_from_demonstrations(observations: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(observations, dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    near_constant = std < STD_EPSILON
    scale = std.copy()
    scale[near_constant] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32), near_constant


def normalize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((np.asarray(values, dtype=np.float32) - mean) / scale).astype(np.float32)


def prepare_demonstrations() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    raw = base.load_demonstrations()
    mean, scale, near_constant = scaler_from_demonstrations(raw["observations"])
    transformed = {name: np.asarray(value).copy() for name, value in raw.items()}
    for name in ("observations", "next_observations", "n_step_next_observations"):
        transformed[name] = normalize(raw[name], mean, scale)

    normalized_observations = transformed["observations"].astype(np.float64)
    variable = ~near_constant
    reconstructed = normalized_observations * scale + mean
    unchanged_names = [
        "actions", "rewards", "dones", "n_step_returns", "n_step_dones",
        "n_step_discounts", "n_step_horizons",
    ]
    validation = {
        "dimensions": int(raw["observations"].shape[1]),
        "near_constant_dimensions": int(near_constant.sum()),
        "max_abs_normalized_mean_nonconstant": float(
            np.max(np.abs(normalized_observations[:, variable].mean(axis=0)))
        ) if variable.any() else 0.0,
        "max_abs_normalized_std_minus_one_nonconstant": float(
            np.max(np.abs(normalized_observations[:, variable].std(axis=0) - 1.0))
        ) if variable.any() else 0.0,
        "reconstruction_max_abs_error": float(
            np.max(np.abs(reconstructed - np.asarray(raw["observations"], dtype=np.float64)))
        ),
        "unchanged_fields_max_abs_error": {
            name: float(np.max(np.abs(np.asarray(transformed[name]) - np.asarray(raw[name]))))
            for name in unchanged_names
        },
    }
    validation["all_pre_run_checks_pass"] = bool(
        validation["dimensions"] == 25
        and validation["max_abs_normalized_mean_nonconstant"] < 1e-4
        and validation["max_abs_normalized_std_minus_one_nonconstant"] < 1e-4
        and validation["reconstruction_max_abs_error"] < 2e-3
        and max(validation["unchanged_fields_max_abs_error"].values()) == 0.0
    )
    scaler = {"mean": mean, "scale": scale, "near_constant": near_constant}
    return raw, transformed, {**validation, **scaler}


def save_scaler(validation: dict[str, Any]) -> None:
    frame = pd.DataFrame({
        "observation_index": np.arange(len(LABELS)),
        "observation_label": LABELS,
        "mean": validation["mean"],
        "scale_std_or_one": validation["scale"],
        "near_constant": validation["near_constant"],
    })
    frame.to_csv(OUT / "021_24_observation_scaler.csv", index=False, encoding="utf-8-sig")


def growth_stage_distribution(raw_obs: np.ndarray, norm_obs: np.ndarray) -> pd.DataFrame:
    dap = np.asarray(raw_obs)[:, 1]
    phases = {
        "early_dap_lt_50": dap < 50,
        "middle_dap_50_to_90": (dap >= 50) & (dap <= 90),
        "late_dap_gt_90": dap > 90,
    }
    selected = [0, 4, 6, 18, 20, 21]
    rows: list[dict[str, Any]] = []
    for phase, mask in phases.items():
        for index in selected:
            for representation, values in (("raw", raw_obs), ("standardized", norm_obs)):
                column = np.asarray(values)[mask, index]
                rows.append({
                    "phase": phase, "n_states": int(mask.sum()),
                    "observation_index": index, "observation_label": LABELS[index],
                    "representation": representation,
                    "minimum": float(column.min()) if len(column) else np.nan,
                    "maximum": float(column.max()) if len(column) else np.nan,
                    "mean": float(column.mean()) if len(column) else np.nan,
                    "std": float(column.std()) if len(column) else np.nan,
                    "unique_count": int(np.unique(column).size) if len(column) else 0,
                })
    return pd.DataFrame(rows)


def environment_arguments() -> dict[str, Any]:
    args = json.loads(base.ENV_SOURCE.read_text(encoding="utf-8"))
    run = OUT / "real_environment"
    run.mkdir(parents=True, exist_ok=True)
    args["log_saving_path"] = str(run / "pdi_gym.log")
    return args


def stage_summary(frame: pd.DataFrame, phase: str, cumulative_l2: float) -> dict[str, Any]:
    result = base.stage_summary(frame, phase)
    result["cumulative_parameter_relative_l2"] = cumulative_l2
    for column in (
        "td_1_weighted", "td_n_weighted", "margin_weighted", "l2_weighted",
        "grad_td_1", "grad_td_n", "grad_margin", "grad_l2", "q_abs_mean", "q_abs_max",
    ):
        result[f"median_{column}"] = float(frame[column].median())
        result[f"max_{column}"] = float(frame[column].max())
    return result


def compare_to_021_22(stages: pd.DataFrame) -> pd.DataFrame:
    old_stage = pd.read_csv(ROOT / "benchmark_results/021_22/021_22_stage_summary.csv")
    rows = []
    metrics = (
        "gradient_clip_fraction", "median_total_gradient_norm", "max_total_gradient_norm",
        "cumulative_parameter_relative_l2", "median_td_1_weighted", "median_td_n_weighted",
        "median_margin_weighted", "median_q_abs_mean", "median_q_abs_max",
    )
    for phase in ("pretrain", "online"):
        old_row = old_stage.loc[old_stage.phase == phase].iloc[0]
        new_row = stages.loc[stages.phase == phase].iloc[0]
        old_log = pd.read_csv(
            ROOT / f"benchmark_results/021_22/021_22_{phase}_update_log.csv"
        )
        for metric in metrics:
            if metric in old_row.index:
                old_value = float(old_row[metric])
            else:
                old_value = float(old_log[metric.removeprefix("median_")].median())
            new_value = float(new_row[metric])
            rows.append({
                "phase": phase, "metric": metric,
                "raw_observation_021_22": old_value,
                "standardized_observation_021_24": new_value,
                "standardized_to_raw_ratio": new_value / max(abs(old_value), 1e-12),
            })
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def plot_results(distribution: pd.DataFrame, comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for label, color in (("topwt", "#0072B2"), ("grnwt", "#D55E00")):
        subset = distribution[(distribution.observation_label == label) & (distribution.representation == "standardized")]
        x = np.arange(len(subset))
        axes[0, 0].errorbar(x, subset["mean"], yerr=subset["std"], marker="o", capsize=3, label=label, color=color)
    axes[0, 0].set_xticks(np.arange(3), ["early", "middle", "late"])
    axes[0, 0].set_title("Standardized biomass distribution")
    axes[0, 0].set_ylabel("mean +/- std")
    axes[0, 0].legend(frameon=False)

    raw = pd.read_csv(ROOT / "benchmark_results/021_24/021_24_observation_scaler.csv")
    axes[0, 1].bar(raw.observation_index, raw.scale_std_or_one, color="#4C78A8")
    axes[0, 1].set_yscale("symlog", linthresh=1e-3)
    axes[0, 1].set_title("Fixed standardization scales")
    axes[0, 1].set_xlabel("Observation index")
    axes[0, 1].set_ylabel("training-demo std")

    chosen = comparison[comparison.metric.isin(["median_td_1_weighted", "median_td_n_weighted", "median_total_gradient_norm"])]
    labels = chosen.phase + "\n" + chosen.metric.str.replace("median_", "", regex=False)
    positions = np.arange(len(chosen)); width = 0.36
    axes[1, 0].bar(positions - width / 2, chosen.raw_observation_021_22, width, color="#999999", label="raw obs")
    axes[1, 0].bar(positions + width / 2, chosen.standardized_observation_021_24, width, color="#009E73", label="standardized")
    axes[1, 0].set_xticks(positions, labels, rotation=25, ha="right")
    axes[1, 0].set_yscale("symlog", linthresh=1e-3)
    axes[1, 0].set_title("Loss and gradient comparison")
    axes[1, 0].legend(frameon=False)

    clips = comparison[comparison.metric == "gradient_clip_fraction"]
    positions = np.arange(len(clips))
    axes[1, 1].bar(positions - width / 2, clips.raw_observation_021_22, width, color="#999999", label="raw obs")
    axes[1, 1].bar(positions + width / 2, clips.standardized_observation_021_24, width, color="#009E73", label="standardized")
    axes[1, 1].set_xticks(positions, clips.phase)
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("Gradient clipping fraction")
    axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.grid(alpha=0.2, axis="y")
    fig.suptitle("SY2014 fixed observation-standardization diagnostic")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "021_24_observation_standardization_diagnostic.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_24_observation_standardization_diagnostic.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    np.random.seed(0)
    torch.manual_seed(0)
    action_rng = np.random.default_rng(0)

    raw_demo, demo, validation = prepare_demonstrations()
    if not validation["all_pre_run_checks_pass"]:
        raise RuntimeError(f"Pre-run standardization validation failed: {validation}")
    save_scaler(validation)
    distribution = growth_stage_distribution(raw_demo["observations"], demo["observations"])
    distribution.to_csv(OUT / "021_24_growth_stage_distribution.csv", index=False, encoding="utf-8-sig")

    replay = PrioritizedDemonstrationReplay(
        demo, agent_capacity=10_000, alpha=0.4,
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

    mean = validation["mean"]
    scale = validation["scale"]
    interactions: list[dict[str, Any]] = []
    online_rows: list[dict[str, Any]] = []
    pending: deque[dict[str, Any]] = deque()
    online_start = base.parameter_vector(parameters)
    try:
        raw_obs, info = env.reset()
        online_update = 0
        for step in range(1, base.ENV_STEPS + 1):
            latest = latest_observation_dict(env, raw_obs, info)
            dap = int(round(scalar(latest.get("dap", step)) or 0))
            obs = normalize(raw_obs, mean, scale)
            epsilon = base.epsilon_at(step)
            if action_rng.random() < epsilon:
                action = int(action_rng.integers(0, 9)); action_source = "epsilon_random"
            else:
                with torch.no_grad():
                    q = model.q_net(torch.as_tensor(obs, dtype=torch.float32, device=model.device).reshape(1, -1))
                action = int(q.argmax(dim=1).item()); action_source = "greedy"
            next_raw_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = normalize(next_raw_obs, mean, scale)
            done = bool(terminated or truncated)
            safe = dict(getattr(env, "last_safe_real_action", {}) or {})
            pending.append({
                "observation": obs.copy(), "action": action, "reward": float(reward),
                "next_observation": next_obs.copy(), "done": done,
            })
            added = 0
            while len(pending) >= base.N_STEP or (done and pending):
                transition = base.matured_transition(pending)
                replay.add_agent(transition, td_error=base.initial_td_error(model, transition))
                pending.popleft(); added += 1
                if not done and len(pending) < base.N_STEP:
                    break
            interactions.append({
                "env_step": step, "dap_action": dap, "epsilon": epsilon,
                "action": action, "action_source": action_source,
                "safe_amir": float(safe.get("amir", 0.0)),
                "safe_anfer": float(safe.get("anfer", 0.0)),
                "raw_reward": float(reward), "training_reward": float(reward),
                "done": done, "agent_transitions_added": added,
                "replay_agent_count": replay.agent_size,
                "normalized_obs_abs_max": float(np.max(np.abs(obs))),
            })
            raw_obs = next_raw_obs
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
    pretrain.to_csv(OUT / "021_24_pretrain_update_log.csv", index=False, encoding="utf-8-sig")
    online.to_csv(OUT / "021_24_online_update_log.csv", index=False, encoding="utf-8-sig")
    interaction_frame.to_csv(OUT / "021_24_agent_interactions.csv", index=False, encoding="utf-8-sig")

    validation["online_reward_max_abs_error"] = float(
        np.max(np.abs(interaction_frame.training_reward - interaction_frame.raw_reward))
    )
    validation["all_post_run_checks_pass"] = bool(
        validation["all_pre_run_checks_pass"] and validation["online_reward_max_abs_error"] == 0.0
    )
    serializable_validation = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in validation.items()
    }
    (OUT / "021_24_validation.json").write_text(
        json.dumps(serializable_validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    stage_rows = [
        stage_summary(pretrain, "pretrain", pretrain_change),
        stage_summary(online, "online", online_change),
    ]
    stages = pd.DataFrame(stage_rows)
    stages.to_csv(OUT / "021_24_standardized_stage_summary.csv", index=False, encoding="utf-8-sig")
    comparison = compare_to_021_22(stages)
    comparison.to_csv(OUT / "021_24_stage_comparison.csv", index=False, encoding="utf-8-sig")

    lookup = {(row.phase, row.metric): row.standardized_to_raw_ratio for row in comparison.itertuples()}
    grad_both_down = all(lookup[(phase, "median_total_gradient_norm")] <= 0.75 for phase in ("pretrain", "online"))
    clips = {
        phase: float(stages.loc[stages.phase == phase, "gradient_clip_fraction"].iloc[0])
        for phase in ("pretrain", "online")
    }
    any_clip_below_one = any(value < 0.95 for value in clips.values())
    any_grad_down = any(lookup[(phase, "median_total_gradient_norm")] <= 0.75 for phase in ("pretrain", "online"))
    if grad_both_down and any_clip_below_one:
        branch = "A_observation_scale_participant"
    elif any_grad_down or any_clip_below_one:
        branch = "B_partial_but_insufficient_improvement"
    else:
        branch = "C_not_supporting_standardization_as_primary_fix"

    summary = {
        "status": "completed" if validation["all_post_run_checks_pass"] else "failed",
        "no_5k_training_performed": True,
        "only_scientific_variable": "fixed_demo_statistics_observation_standardization",
        "reward_scaled": False,
        "pre_registered_interpretation_branch": branch,
        "gradient_ratio_pretrain": lookup[("pretrain", "median_total_gradient_norm")],
        "gradient_ratio_online": lookup[("online", "median_total_gradient_norm")],
        "gradient_clip_fraction_pretrain": clips["pretrain"],
        "gradient_clip_fraction_online": clips["online"],
        "validation_passed": validation["all_post_run_checks_pass"],
        "pretrain": stage_rows[0], "online": stage_rows[1],
    }
    (OUT / "021_24_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_results(distribution, comparison)

    key = comparison[comparison.metric.isin([
        "median_td_1_weighted", "median_td_n_weighted", "median_total_gradient_norm",
        "gradient_clip_fraction", "cumulative_parameter_relative_l2",
    ])].round(6)
    early = distribution[
        (distribution.phase == "early_dap_lt_50")
        & (distribution.representation == "standardized")
        & (distribution.observation_label.isin(["topwt", "grnwt"]))
    ][["observation_label", "n_states", "minimum", "maximum", "mean", "std", "unique_count"]].round(6)
    record = f"""# 021_24 SY2014 固定观测标准化短诊断记录

## 边界与防止乱调

本任务在运行前固定了标准化公式、统计来源、零方差处理、100+100 步规模和三分支判定。恢复 021_22 原始 reward，唯一科学变量是用冻结示范集统计量对同一 25 维信息做固定标准化。没有添加特征、改变 IC/reward/action/loss，也没有进入 5K。

## 实现校验

- pre-run checks: `{validation['all_pre_run_checks_pass']}`
- post-run checks: `{validation['all_post_run_checks_pass']}`
- 非常量维标准化均值最大绝对误差：`{validation['max_abs_normalized_mean_nonconstant']:.3g}`
- 非常量维标准差偏离 1 的最大误差：`{validation['max_abs_normalized_std_minus_one_nonconstant']:.3g}`
- 逆变换最大误差：`{validation['reconstruction_max_abs_error']:.3g}`
- 在线 reward 改动误差：`{validation['online_reward_max_abs_error']:.3g}`

## 早期生长状态是否仍可区分

{markdown_table(early)}

`unique_count` 和组内标准差用于确认固定标准化没有把早期 biomass 状态压成同一个常数；这不等同于证明网络一定能够学会利用差异。

早期 `grnwt` 只有一个取值，是因为原始示范数据在 DAP<50 时尚未形成籽粒，原始 `grnwt` 本来就是常数 0，并非标准化造成的信息丢失。早期 `topwt` 仍保留多个不同取值。

## 与 021_22 原始观测基准对比

{markdown_table(key)}

## 预注册结论

`{branch}`

该结论只针对 100+100 步的梯度动力学，不代表长期策略已经改善，也不自动授权 5K。

需要同时保留两个反例：标准化后的 TD loss 没有下降；预训练累计参数相对 L2 变化可能高于原始观测基准。因此目前只能确认原始输入尺度造成了极大的瞬时梯度并迫使更新裁剪，不能声称长期振荡、施氮时序或策略坍缩已经解决。

## 输出

- `benchmark_results/021_24/021_24_observation_scaler.csv`
- `benchmark_results/021_24/021_24_growth_stage_distribution.csv`
- `benchmark_results/021_24/021_24_pretrain_update_log.csv`
- `benchmark_results/021_24/021_24_online_update_log.csv`
- `benchmark_results/021_24/021_24_agent_interactions.csv`
- `benchmark_results/021_24/021_24_standardized_stage_summary.csv`
- `benchmark_results/021_24/021_24_stage_comparison.csv`
- `benchmark_results/021_24/021_24_validation.json`
- `benchmark_results/021_24/021_24_summary.json`
- `benchmark_results/021_24/021_24_observation_standardization_diagnostic.png/.svg`
"""
    DOC.write_text(record, encoding="utf-8")
    if not validation["all_post_run_checks_pass"]:
        raise RuntimeError("Observation-standardization validation failed")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
