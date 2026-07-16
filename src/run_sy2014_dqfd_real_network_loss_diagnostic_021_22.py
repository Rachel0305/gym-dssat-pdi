from __future__ import annotations

import json
import math
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

from frozen_nstep_dqn_config_020_11 import apply_environment_constants, dqn_kwargs
from literature_aligned_dqfd import PrioritizedDemonstrationReplay, dqfd_loss_components
from ppo_evaluate import latest_observation_dict, scalar
import run_yc2014_linked_dqn_5k_multiseed_013_07 as shared
import run_yc_fq_frozen_nstep_cross_site_020_12 as legacy
from stable_baselines3 import DQN


OUT = ROOT / "benchmark_results" / "021_22"
DOC = ROOT / "docs" / "2026-07-15_021_22_sy2014_dqfd_real_network_loss_scale_diagnostic.md"
ENV_SOURCE = ROOT / "benchmark_results/021_14/021_14_sy2014_reward_scale_seed1_25k_retry__sy_2014_seed1/dqn_env_args.json"
DEMO_SOURCE = ROOT / "benchmark_results/021_20/021_20_demonstration_transitions.npz"
NULL_YIELD = 5408.0
GAMMA = 0.99
N_STEP = 5
PRETRAIN_UPDATES = 100
ENV_STEPS = 100
LEARNING_STARTS = 50
BATCH_SIZE = 32
PLANNED_TIMESTEPS = 50_000
EXPLORATION_FRACTION = 0.35
MAX_GRAD_NORM = 10.0


def env_args() -> dict[str, Any]:
    args = json.loads(ENV_SOURCE.read_text(encoding="utf-8"))
    run = OUT / "real_environment"
    run.mkdir(parents=True, exist_ok=True)
    args["log_saving_path"] = str(run / "pdi_gym.log")
    return args


def make_env(args: dict[str, Any]):
    apply_environment_constants(shared)
    return legacy.make_train_env(args, NULL_YIELD)


def load_demonstrations() -> dict[str, np.ndarray]:
    source = np.load(DEMO_SOURCE)
    return {name: source[name] for name in source.files}


def tensor(value: np.ndarray, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


def gradient_norm(loss: torch.Tensor, parameters: list[torch.Tensor]) -> float:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    squared = torch.zeros((), dtype=torch.float32, device=loss.device)
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().square().sum()
    return float(torch.sqrt(squared).cpu())


def parameter_vector(parameters: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1).cpu() for parameter in parameters])


def epsilon_at(step: int) -> float:
    decay_steps = EXPLORATION_FRACTION * PLANNED_TIMESTEPS
    fraction = min(max(float(step) / decay_steps, 0.0), 1.0)
    return 1.0 + fraction * (0.05 - 1.0)


def compute_targets(model: DQN, data: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    device = model.device
    rewards = tensor(data["rewards"], dtype=torch.float32, device=device).reshape(-1)
    dones = tensor(data["dones"], dtype=torch.float32, device=device).reshape(-1)
    next_obs = tensor(data["next_observations"], dtype=torch.float32, device=device)
    n_returns = tensor(data["n_step_returns"], dtype=torch.float32, device=device).reshape(-1)
    n_dones = tensor(data["n_step_dones"], dtype=torch.float32, device=device).reshape(-1)
    n_discounts = tensor(data["n_step_discounts"], dtype=torch.float32, device=device).reshape(-1)
    n_next = tensor(data["n_step_next_observations"], dtype=torch.float32, device=device)
    with torch.no_grad():
        next_action = model.q_net(next_obs).argmax(dim=1)
        target_next = model.q_net_target(next_obs).gather(1, next_action[:, None]).squeeze(1)
        target_1 = rewards + (1.0 - dones) * GAMMA * target_next
        n_next_action = model.q_net(n_next).argmax(dim=1)
        target_n_next = model.q_net_target(n_next).gather(1, n_next_action[:, None]).squeeze(1)
        target_n = n_returns + (1.0 - n_dones) * n_discounts * target_n_next
    return target_1, target_n


def one_update(
    model: DQN,
    replay: PrioritizedDemonstrationReplay,
    *,
    phase: str,
    update: int,
    env_step: int,
    epsilon: float,
) -> dict[str, Any]:
    sample = replay.sample(BATCH_SIZE, beta=0.6)
    device = model.device
    observations = tensor(sample.data["observations"], dtype=torch.float32, device=device)
    actions = tensor(sample.data["actions"], dtype=torch.long, device=device).reshape(-1)
    weights = tensor(sample.importance_weights, dtype=torch.float32, device=device)
    demo_mask = tensor(sample.is_demonstration, dtype=torch.bool, device=device)
    target_1, target_n = compute_targets(model, sample.data)
    q_values = model.q_net(observations)
    parameters = list(model.q_net.parameters())
    losses = dqfd_loss_components(
        q_values=q_values,
        actions=actions,
        target_1_step=target_1,
        target_n_step=target_n,
        importance_weights=weights,
        demonstration_mask=demo_mask,
        l2_parameters=parameters,
        margin=0.8,
        lambda_n_step=1.0,
        lambda_margin=1.0,
        lambda_l2=1e-5,
    )
    gradient_components = {
        name: gradient_norm(losses[name], parameters)
        for name in ("td_1_weighted", "td_n_weighted", "margin_weighted", "l2_weighted")
    }
    before = parameter_vector(parameters)
    model.policy.optimizer.zero_grad()
    losses["total"].backward()
    total_gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM))
    model.policy.optimizer.step()
    after = parameter_vector(parameters)
    denominator = max(float(torch.linalg.vector_norm(before)), 1e-12)
    relative_parameter_step = float(torch.linalg.vector_norm(after - before)) / denominator

    with torch.no_grad():
        chosen_q = q_values.gather(1, actions[:, None]).squeeze(1)
        td_errors = torch.abs(target_1 - chosen_q).detach().cpu().numpy()
    updates: dict[int, float] = {}
    for index, error in zip(sample.global_indices, td_errors):
        updates[int(index)] = max(updates.get(int(index), 0.0), float(error))
    replay.update_priorities(updates.keys(), updates.values())

    values = {name: float(value.detach().cpu()) for name, value in losses.items()}
    td_reference = max(abs(values["td_1_weighted"]), 1e-12)
    all_finite = bool(
        all(np.isfinite(value) for value in values.values())
        and all(np.isfinite(value) for value in gradient_components.values())
        and np.isfinite(total_gradient_norm)
        and np.isfinite(relative_parameter_step)
        and torch.isfinite(q_values).all().item()
    )
    return {
        "phase": phase,
        "update": int(update),
        "env_step": int(env_step),
        "epsilon": float(epsilon),
        "sample_demo_count": int(sample.is_demonstration.sum()),
        "sample_agent_count": int((~sample.is_demonstration).sum()),
        "replay_demo_count": replay.demo_count,
        "replay_agent_count": replay.agent_size,
        **values,
        "td_n_to_td1": values["td_n_weighted"] / td_reference,
        "margin_to_td1": values["margin_weighted"] / td_reference,
        "l2_to_td1": values["l2_weighted"] / td_reference,
        "grad_td_1": gradient_components["td_1_weighted"],
        "grad_td_n": gradient_components["td_n_weighted"],
        "grad_margin": gradient_components["margin_weighted"],
        "grad_l2": gradient_components["l2_weighted"],
        "grad_total_before_clip": total_gradient_norm,
        "gradient_would_clip": total_gradient_norm > MAX_GRAD_NORM,
        "relative_parameter_step": relative_parameter_step,
        "q_abs_mean": float(q_values.detach().abs().mean().cpu()),
        "q_abs_max": float(q_values.detach().abs().max().cpu()),
        "all_finite": all_finite,
    }


def matured_transition(pending: deque[dict[str, Any]]) -> dict[str, np.ndarray | float | int]:
    first = pending[0]
    value = 0.0
    last = first
    horizon = 0
    for offset, item in enumerate(pending):
        if offset >= N_STEP:
            break
        value += (GAMMA ** offset) * float(item["reward"])
        last = item
        horizon += 1
        if bool(item["done"]):
            break
    return {
        "observations": first["observation"],
        "actions": int(first["action"]),
        "rewards": float(first["reward"]),
        "next_observations": first["next_observation"],
        "dones": float(first["done"]),
        "n_step_returns": float(value),
        "n_step_next_observations": last["next_observation"],
        "n_step_dones": float(last["done"]),
        "n_step_discounts": float(GAMMA ** horizon),
        "n_step_horizons": int(horizon),
    }


def initial_td_error(model: DQN, transition: dict[str, Any]) -> float:
    data = {
        "rewards": np.asarray([transition["rewards"]], dtype=np.float32),
        "dones": np.asarray([transition["dones"]], dtype=np.float32),
        "next_observations": np.asarray([transition["next_observations"]], dtype=np.float32),
        "n_step_returns": np.asarray([transition["n_step_returns"]], dtype=np.float32),
        "n_step_dones": np.asarray([transition["n_step_dones"]], dtype=np.float32),
        "n_step_discounts": np.asarray([transition["n_step_discounts"]], dtype=np.float32),
        "n_step_next_observations": np.asarray([transition["n_step_next_observations"]], dtype=np.float32),
    }
    target, _ = compute_targets(model, data)
    obs = torch.as_tensor(
        np.asarray(transition["observations"])[None], dtype=torch.float32, device=model.device
    )
    action = int(transition["actions"])
    with torch.no_grad():
        current = model.q_net(obs)[0, action]
    return float(torch.abs(target[0] - current).cpu())


def stage_summary(frame: pd.DataFrame, phase: str) -> dict[str, Any]:
    ratios = {}
    for name in ("td_n_to_td1", "margin_to_td1", "l2_to_td1"):
        ratios[f"median_{name}"] = float(frame[name].median())
        ratios[f"max_{name}"] = float(frame[name].max())
    return {
        "phase": phase,
        "updates": int(len(frame)),
        "all_finite": bool(frame.all_finite.all()),
        "gradient_clip_fraction": float(frame.gradient_would_clip.mean()),
        "median_total_gradient_norm": float(frame.grad_total_before_clip.median()),
        "max_total_gradient_norm": float(frame.grad_total_before_clip.max()),
        "cumulative_relative_parameter_step_sum": float(frame.relative_parameter_step.sum()),
        "median_sample_demo_fraction": float((frame.sample_demo_count / BATCH_SIZE).median()),
        **ratios,
    }


def plot_logs(pretrain: pd.DataFrame, online: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    loss_columns = ["td_1_weighted", "td_n_weighted", "margin_weighted", "l2_weighted"]
    grad_columns = ["grad_td_1", "grad_td_n", "grad_margin", "grad_l2", "grad_total_before_clip"]
    colors = {"td_1_weighted": "#222222", "td_n_weighted": "#0072B2", "margin_weighted": "#D55E00", "l2_weighted": "#009E73"}
    grad_colors = {"grad_td_1": "#222222", "grad_td_n": "#0072B2", "grad_margin": "#D55E00", "grad_l2": "#009E73", "grad_total_before_clip": "#CC79A7"}
    for row, (frame, title) in enumerate(((pretrain, "Demonstration pretraining"), (online, "Online diagnostic"))):
        for column in loss_columns:
            axes[row, 0].plot(frame["update"], frame[column], linewidth=1.2, color=colors[column], label=column)
        for column in grad_columns:
            axes[row, 1].plot(frame["update"], frame[column], linewidth=1.2, color=grad_colors[column], label=column)
        axes[row, 0].set_ylabel(f"{title}\nweighted loss")
        axes[row, 1].set_ylabel("gradient norm")
        for ax in axes[row]:
            ax.set_yscale("symlog", linthresh=1e-5)
            ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Update")
    axes[1, 1].set_xlabel("Update")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0, 1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("SY2014 literature-aligned DQfD loss and gradient diagnostic")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "021_22_loss_gradient_diagnostic.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_22_loss_gradient_diagnostic.svg", bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    np.random.seed(0)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    demonstrations = load_demonstrations()
    replay = PrioritizedDemonstrationReplay(
        demonstrations, agent_capacity=10_000, alpha=0.4,
        epsilon_demo=1.0, epsilon_agent=0.001, seed=21022,
    )
    args = env_args()
    env = make_env(args)
    kwargs = dqn_kwargs(seed=0)
    model = DQN("MlpPolicy", env, verbose=0, **kwargs)
    parameters = list(model.q_net.parameters())
    initial_parameters = parameter_vector(parameters)
    pretrain_rows = []
    for update in range(1, PRETRAIN_UPDATES + 1):
        pretrain_rows.append(one_update(
            model, replay, phase="pretrain", update=update, env_step=0, epsilon=1.0
        ))
    after_pretrain_parameters = parameter_vector(parameters)
    pretrain_cumulative_change = float(torch.linalg.vector_norm(after_pretrain_parameters - initial_parameters)) / max(
        float(torch.linalg.vector_norm(initial_parameters)), 1e-12
    )
    model.q_net_target.load_state_dict(model.q_net.state_dict())

    interactions = []
    online_rows = []
    pending: deque[dict[str, Any]] = deque()
    online_start_parameters = parameter_vector(parameters)
    try:
        obs, info = env.reset()
        online_update = 0
        for step in range(1, ENV_STEPS + 1):
            latest = latest_observation_dict(env, obs, info)
            dap = int(round(scalar(latest.get("dap", step)) or 0))
            epsilon = epsilon_at(step)
            if rng.random() < epsilon:
                action = int(rng.integers(0, 9))
                action_source = "epsilon_random"
            else:
                with torch.no_grad():
                    q = model.q_net(torch.as_tensor(obs, dtype=torch.float32, device=model.device).reshape(1, -1))
                action = int(q.argmax(dim=1).item())
                action_source = "greedy"
            before = np.asarray(obs, dtype=np.float32).copy()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            safe = dict(getattr(env, "last_safe_real_action", {}) or {})
            pending.append({
                "observation": before,
                "action": action,
                "reward": float(reward),
                "next_observation": np.asarray(next_obs, dtype=np.float32).copy(),
                "done": done,
            })
            added = 0
            while len(pending) >= N_STEP or (done and pending):
                transition = matured_transition(pending)
                error = initial_td_error(model, transition)
                replay.add_agent(transition, td_error=error)
                pending.popleft()
                added += 1
                if not done and len(pending) < N_STEP:
                    break
            interactions.append({
                "env_step": step, "dap_action": dap, "epsilon": epsilon,
                "action": action, "action_source": action_source,
                "safe_amir": float(safe.get("amir", 0.0)),
                "safe_anfer": float(safe.get("anfer", 0.0)),
                "reward": float(reward), "done": done,
                "agent_transitions_added": added,
                "replay_agent_count": replay.agent_size,
            })
            obs = next_obs
            if step >= LEARNING_STARTS:
                online_update += 1
                online_rows.append(one_update(
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

    online_end_parameters = parameter_vector(parameters)
    online_cumulative_change = float(torch.linalg.vector_norm(online_end_parameters - online_start_parameters)) / max(
        float(torch.linalg.vector_norm(online_start_parameters)), 1e-12
    )
    pretrain = pd.DataFrame(pretrain_rows)
    online = pd.DataFrame(online_rows)
    interaction_frame = pd.DataFrame(interactions)
    pretrain.to_csv(OUT / "021_22_pretrain_update_log.csv", index=False, encoding="utf-8-sig")
    online.to_csv(OUT / "021_22_online_update_log.csv", index=False, encoding="utf-8-sig")
    interaction_frame.to_csv(OUT / "021_22_agent_interactions.csv", index=False, encoding="utf-8-sig")

    pretrain_summary = stage_summary(pretrain, "pretrain")
    online_summary = stage_summary(online, "online")
    pretrain_summary["cumulative_parameter_relative_l2"] = pretrain_cumulative_change
    online_summary["cumulative_parameter_relative_l2"] = online_cumulative_change
    stage_frame = pd.DataFrame([pretrain_summary, online_summary])
    stage_frame.to_csv(OUT / "021_22_stage_summary.csv", index=False, encoding="utf-8-sig")
    component_alert = bool(any(
        value > 100
        for summary in (pretrain_summary, online_summary)
        for key, value in summary.items()
        if key.startswith("median_") and key.endswith("_to_td1")
    ))
    clipping_alert = bool(
        pretrain_summary["gradient_clip_fraction"] > 0.5
        or online_summary["gradient_clip_fraction"] > 0.5
    )
    pretraining_shift_alert = bool(
        pretrain_cumulative_change > 10 * max(online_cumulative_change, 1e-12)
    )
    all_finite = bool(pretrain_summary["all_finite"] and online_summary["all_finite"])
    summary = {
        "status": "completed" if all_finite else "failed",
        "no_5k_training_performed": True,
        "pretrain_updates": len(pretrain),
        "environment_steps": len(interaction_frame),
        "online_updates": len(online),
        "pretrain_cumulative_parameter_relative_l2": pretrain_cumulative_change,
        "online_cumulative_parameter_relative_l2": online_cumulative_change,
        "component_dominance_alert": component_alert,
        "frequent_gradient_clipping_alert": clipping_alert,
        "pretraining_shift_alert": pretraining_shift_alert,
        "all_values_finite": all_finite,
        "pretrain": pretrain_summary,
        "online": online_summary,
    }
    (OUT / "021_22_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_logs(pretrain, online)

    display = stage_frame[[
        "phase", "updates", "gradient_clip_fraction", "median_total_gradient_norm",
        "max_total_gradient_norm", "median_td_n_to_td1", "median_margin_to_td1",
        "median_l2_to_td1", "median_sample_demo_fraction", "cumulative_parameter_relative_l2",
    ]].copy().round(6)
    record = f"""# 021_22 SY2014 文献对齐 DQfD 真实网络损失与梯度诊断记录

## 边界

本轮使用真实SY2014 IC=2环境和真实SB3 DQN网络，执行100次示范预训练与{len(interaction_frame)}个环境交互步；在线更新从step50开始，共{len(online)}次。没有完成生长季、没有5K训练、没有评价产量或选择checkpoint，也没有调整任何预注册参数。

## 阶段汇总

{markdown_table(display)}

## 诊断警报

- component_dominance_alert: `{component_alert}`
- frequent_gradient_clipping_alert: `{clipping_alert}`
- pretraining_shift_alert: `{pretraining_shift_alert}`
- all_values_finite: `{all_finite}`

这些警报按运行前规则计算，只用于决定下一任务，不在本轮现场调整lambda、priority参数或更新频率。

## 解释边界

1. 预训练和在线阶段均保存了每次更新的四项raw/weighted loss、分量梯度、total gradient、参数步长和Q量级，可检查趋势而非只看终点。
2. 100步时epsilon仍接近1，因此agent经验主要来自探索动作；这是一项损失链路诊断，不代表可用策略。
3. 若损失和梯度有限，只能说明工程链路可运行；是否允许5K仍需结合警报和完整曲线另立任务判断。
4. 本轮结果不能用于声称DQfD有效或无效。

## 失败记录

第一次运行完成了数值计算和CSV/JSON写出，但在最终绘图时因Pandas列名`update`与DataFrame方法同名，`frame.update`被解释为方法而非列，触发横纵坐标长度错误。完整失败现场保存在`benchmark_results/021_22_failed_attempt_1_plot_attribute_collision/`。随后只把绘图访问改为`frame["update"]`并从头运行；没有修改任何科学参数、输入或诊断逻辑。

## 输出

- `benchmark_results/021_22/021_22_pretrain_update_log.csv`
- `benchmark_results/021_22/021_22_online_update_log.csv`
- `benchmark_results/021_22/021_22_agent_interactions.csv`
- `benchmark_results/021_22/021_22_stage_summary.csv`
- `benchmark_results/021_22/021_22_loss_gradient_diagnostic.png/.svg`
- `benchmark_results/021_22/021_22_summary.json`

## 状态

`{summary['status']}`：真实网络短诊断已完成；没有自动进入5K。
"""
    DOC.write_text(record, encoding="utf-8")
    if not all_finite:
        raise RuntimeError("Non-finite value detected; see 021_22 logs")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
