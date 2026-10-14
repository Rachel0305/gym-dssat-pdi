from __future__ import annotations

import hashlib
import json
import sys
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
import run_sy2014_event_balanced_sampling_ab_021_32 as eventbase
import run_sy2014_fixed_observation_standardization_diagnostic_021_24 as normbase
import run_sy2014_minimal_dqfd_style_5k_ab_021_20 as oldab
import run_sy2014_stratified_demo_sampling_ab_021_30 as exp
from calculate_five_site_wue_nue_from_summary_019_10 import num, parse_summary_out, select_matching_row
from frozen_nstep_dqn_config_020_11 import ACTION_TABLE_9
from literature_aligned_dqfd import PrioritizedDemonstrationReplay, ReplaySample
from ppo_evaluate import latest_observation_dict, scalar
from run_sy2014_demo_return_to_go_ab_021_27 import full_return_to_go
from stable_baselines3 import DQN


OUT = ROOT / "benchmark_results" / "021_34"
DOC = ROOT / "docs" / "2026-07-15_021_34_sy2014_event_balanced_frozen_policy_forward_evaluation.md"
REFERENCE_CURVE = ROOT / "benchmark_results" / "021_32" / "021_32_learning_curve.csv"
BASELINE_SOURCE = ROOT / "benchmark_results" / "021_18" / "021_18_baseline_metrics.csv"
ORACLE_SOURCE = ROOT / "benchmark_results" / "021_18" / "021_18_candidate_summary.csv"
MILESTONES = [0, 10, 25, 50, 100, 250, 500]
BETA = 0.6
EVAL_LIMIT = 420


def parameter_hash(model: DQN) -> str:
    digest = hashlib.sha256()
    for tensor in model.q_net.state_dict().values():
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_model() -> tuple[DQN, PrioritizedDemonstrationReplay, dict[str, Any], np.ndarray, np.ndarray]:
    raw, standardized, validation = normbase.prepare_demonstrations()
    if not validation["all_pre_run_checks_pass"]:
        raise RuntimeError("Frozen observation-standardization validation failed")
    demonstrations = full_return_to_go(standardized)
    observations = demonstrations["observations"]
    actions = np.asarray(demonstrations["actions"], dtype=np.int64).reshape(-1)
    raw_observations = np.asarray(raw["observations"])
    nonzero_indices = np.where(actions != 0)[0]
    noop_indices = np.where(actions == 0)[0]

    np.random.seed(0)
    torch.manual_seed(0)
    env = exp.OfflineShapeEnv(observations.shape[1])
    model = DQN("MlpPolicy", env, verbose=0, **exp.dqn_kwargs(seed=0))
    replay = PrioritizedDemonstrationReplay(
        demonstrations, agent_capacity=1, alpha=0.4,
        epsilon_demo=1.0, epsilon_agent=0.001, seed=21022,
    )
    rng = np.random.default_rng(21022)
    state = {"update": 0}
    original_sampler = exp.stratified_sample

    def event_balanced_sampler(replay_object, action_values, generator):
        probabilities = replay_object.sampling_probabilities()
        noop_conditional = probabilities[noop_indices] / probabilities[noop_indices].sum()
        chosen_noop = generator.choice(noop_indices, size=16, replace=True, p=noop_conditional)
        counts = np.full(len(nonzero_indices), 3, dtype=np.int64)
        counts[(state["update"] - 1) % len(nonzero_indices)] += 1
        chosen_nonzero = np.repeat(nonzero_indices, counts)
        chosen = np.concatenate([chosen_noop, chosen_nonzero]).astype(np.int64)
        generator.shuffle(chosen)
        conditional = np.empty(len(chosen), dtype=np.float64)
        lookup = {int(index): float(value) for index, value in zip(noop_indices, noop_conditional)}
        for position, index in enumerate(chosen):
            conditional[position] = lookup[int(index)] if actions[index] == 0 else 0.2
        group_sizes = np.where(actions[chosen] == 0, len(noop_indices), len(nonzero_indices))
        importance = (group_sizes * conditional) ** (-BETA)
        importance /= importance.max()
        data = {name: np.asarray(values)[chosen] for name, values in replay_object.demonstrations.items()}
        return ReplaySample(
            global_indices=chosen,
            probabilities=0.5 * conditional,
            importance_weights=importance.astype(np.float32),
            is_demonstration=np.ones(len(chosen), dtype=bool),
            data=data,
        )

    curves: list[dict[str, Any]] = []
    summary, _events = exp.evaluate_demonstrations(
        model, observations, actions, raw_observations, 0, pd.DataFrame()
    )
    curves.append(summary)
    updates: list[dict[str, Any]] = []
    previous = 0
    exp.stratified_sample = event_balanced_sampler
    try:
        for milestone in MILESTONES[1:]:
            interval = []
            for update in range(previous + 1, milestone + 1):
                state["update"] = update
                row = exp.update_model(
                    model, replay, actions, rng,
                    arm="treatment_event_balanced", update=update,
                )
                interval.append(row)
                updates.append(row)
            summary, _events = exp.evaluate_demonstrations(
                model, observations, actions, raw_observations,
                milestone, pd.DataFrame(interval),
            )
            curves.append(summary)
            previous = milestone
    finally:
        exp.stratified_sample = original_sampler
        env.close()
    return model, replay, validation, pd.DataFrame(curves), pd.DataFrame(updates)


def env_args() -> dict[str, Any]:
    args = json.loads(base.ENV_SOURCE.read_text(encoding="utf-8"))
    run = OUT / "forward_environment"
    run.mkdir(parents=True, exist_ok=True)
    args["log_saving_path"] = str(run / "pdi_gym.log")
    return args


def frozen_forward(model: DQN, mean: np.ndarray, scale: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    env = base.make_env(env_args())
    rows: list[dict[str, Any]] = []
    snapshot_dir = OUT / "forward_environment" / "pdi_tmp_snapshot_eval"
    try:
        raw_obs, info = env.reset()
        for step in range(EVAL_LIMIT):
            before = latest_observation_dict(env, raw_obs, info)
            dap = int(round(scalar(before.get("dap", step)) or 0))
            transformed = normbase.normalize(raw_obs, mean, scale)
            with torch.no_grad():
                q = model.q_net(torch.as_tensor(
                    transformed, dtype=torch.float32, device=model.device
                ).reshape(1, -1)).detach().cpu().numpy().reshape(-1)
            action = int(np.argmax(q))
            request = ACTION_TABLE_9[action]
            raw_obs, reward, terminated, truncated, info = env.step(action)
            safe = dict(getattr(env, "last_safe_real_action", {}) or {})
            row = {
                "step": step,
                "dap_action": dap,
                "action_index": action,
                "requested_irrigation_mm": float(request["amir"]),
                "requested_nitrogen_kg_ha": float(request["anfer"]),
                "safe_irrigation_mm": float(safe.get("amir", 0.0)),
                "safe_nitrogen_kg_ha": float(safe.get("anfer", 0.0)),
                "reward": float(reward),
                "rain": scalar(before.get("rain", np.nan)),
                "swfac": scalar(before.get("swfac", np.nan)),
                "nstres": scalar(before.get("nstres", np.nan)),
                "topwt": scalar(before.get("topwt", np.nan)),
                "grnwt": scalar(before.get("grnwt", np.nan)),
                "q_min": float(np.min(q)),
                "q_max": float(np.max(q)),
                "q_argmax": action,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
            row.update({f"q_action_{index}": float(value) for index, value in enumerate(q)})
            rows.append(row)
            if terminated or truncated:
                break
    finally:
        oldab.snapshot(env, snapshot_dir)
        env.close()
    frame = pd.DataFrame(rows)
    gwad, cwad = oldab.final_yield(snapshot_dir)
    expected_i = float(frame.safe_irrigation_mm.sum())
    expected_n = float(frame.safe_nitrogen_kg_ha.sum())
    summary_rows = parse_summary_out(snapshot_dir / "Summary.OUT")
    matched, match_score, row_index = select_matching_row(summary_rows, gwad, expected_i, expected_n)
    ircm, nicm = num(matched, "IRCM"), num(matched, "NICM")
    nucm, etcp = num(matched, "NUCM"), num(matched, "ETCP")
    ypem, ypim = num(matched, "YPEM"), num(matched, "YPIM")
    ypnam, ypnum = num(matched, "YPNAM"), num(matched, "YPNUM")
    result = {
        "scenario": "event_balanced_500_frozen",
        "yield_kg_ha": gwad,
        "biomass_kg_ha": cwad,
        "irrigation_mm": ircm,
        "nitrogen_kg_ha": nicm,
        "safe_irrigation_sum_mm": expected_i,
        "safe_nitrogen_sum_kg_ha": expected_n,
        "nitrogen_uptake_kg_ha": nucm,
        "ETCP_mm": etcp,
        "WP_ET_kg_m3": ypem * 0.1 if ypem is not None and ypem >= 0 else np.nan,
        "IWP_gross_kg_m3": ypim * 0.1 if ypim is not None and ypim >= 0 and ircm and ircm > 0 else np.nan,
        "PFP_N_kg_kg": ypnam if ypnam is not None and ypnam >= 0 and nicm and nicm > 0 else np.nan,
        "NUtE_kg_kg": ypnum if ypnum is not None and ypnum >= 0 and nucm and nucm > 0 else np.nan,
        "PNB_N": nucm / nicm if nucm is not None and nicm and nicm > 0 else np.nan,
        "late_n_after_dap90_kg_ha": float(frame.loc[frame.dap_action > 90, "safe_nitrogen_kg_ha"].sum()),
        "nonzero_operation_count": int((frame.safe_irrigation_mm.gt(0) | frame.safe_nitrogen_kg_ha.gt(0)).sum()),
        "reward_total": float(frame.reward.sum()),
        "summary_match_score": match_score,
        "summary_row_index": row_index,
        "terminated_or_truncated": bool(frame.iloc[-1].terminated or frame.iloc[-1].truncated),
        "q_values_finite": bool(np.isfinite(frame.filter(regex=r"^q_action_").to_numpy()).all()),
    }
    return frame, result


def comparison_table(policy: dict[str, Any]) -> pd.DataFrame:
    # Preserve the literal scenario label "null" instead of treating it as NaN.
    baseline = pd.read_csv(BASELINE_SOURCE, keep_default_na=False).rename(columns={
        "irrigation_mm_summary": "irrigation_mm",
        "nitrogen_kg_ha_summary": "nitrogen_kg_ha",
    })
    columns = [
        "scenario", "yield_kg_ha", "irrigation_mm", "nitrogen_kg_ha",
        "nitrogen_uptake_kg_ha", "ETCP_mm", "WP_ET_kg_m3",
        "IWP_gross_kg_m3", "PFP_N_kg_kg", "NUtE_kg_kg", "PNB_N",
    ]
    baseline = baseline.reindex(columns=columns)
    oracle = pd.read_csv(ORACLE_SOURCE)
    oracle = oracle.loc[oracle.scenario.eq("R_I75_no_early_mid_N200")].iloc[0]
    oracle_row = {
        "scenario": "deterministic_oracle_021_18",
        "yield_kg_ha": oracle.final_gwad,
        "irrigation_mm": oracle.summary_irrigation_total,
        "nitrogen_kg_ha": oracle.summary_nitrogen_total,
        "nitrogen_uptake_kg_ha": oracle.nitrogen_uptake_kg_ha,
        "ETCP_mm": oracle.etcp_mm,
        "WP_ET_kg_m3": oracle.WP_ET_kg_m3,
        "IWP_gross_kg_m3": oracle.IWP_gross_kg_m3,
        "PFP_N_kg_kg": oracle.PFP_N_kg_kg,
        "NUtE_kg_kg": oracle.NUtE_kg_kg,
        "PNB_N": oracle.PNB_N,
    }
    policy_row = {column: policy.get(column, np.nan) for column in columns}
    return pd.concat([baseline, pd.DataFrame([oracle_row, policy_row])], ignore_index=True)


def plot_results(daily: pd.DataFrame, comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    operations = daily[(daily.safe_irrigation_mm > 0) | (daily.safe_nitrogen_kg_ha > 0)]
    axes[0, 0].bar(operations.dap_action - 0.8, operations.safe_irrigation_mm, 1.6, color="#0072B2", label="Irrigation")
    axes[0, 0].bar(operations.dap_action + 0.8, operations.safe_nitrogen_kg_ha, 1.6, color="#D55E00", label="Nitrogen")
    axes[0, 0].set_ylabel("Applied amount"); axes[0, 0].legend(frameon=False)
    axes[0, 1].step(daily.dap_action, daily.safe_irrigation_mm.cumsum(), where="post", color="#0072B2", label="Cumulative I")
    axes[0, 1].step(daily.dap_action, daily.safe_nitrogen_kg_ha.cumsum(), where="post", color="#D55E00", label="Cumulative N")
    axes[0, 1].set_ylabel("Cumulative input"); axes[0, 1].legend(frameon=False)
    x = np.arange(len(comparison))
    axes[1, 0].bar(x, comparison.yield_kg_ha, color="#777777")
    axes[1, 0].set_xticks(x, comparison.scenario, rotation=35, ha="right")
    axes[1, 0].set_ylabel("Yield (kg/ha)")
    width = 0.36
    axes[1, 1].bar(x - width / 2, comparison.irrigation_mm.fillna(0), width, color="#0072B2", label="Irrigation")
    axes[1, 1].bar(x + width / 2, comparison.nitrogen_kg_ha.fillna(0), width, color="#D55E00", label="Nitrogen")
    axes[1, 1].set_xticks(x, comparison.scenario, rotation=35, ha="right")
    axes[1, 1].set_ylabel("Seasonal input"); axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.grid(alpha=0.2); ax.set_xlabel("DAP" if ax in axes[0, :] else "Scenario")
    fig.suptitle("SY2014 frozen 500-update event-balanced policy forward evaluation")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "021_34_frozen_policy_forward.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_34_frozen_policy_forward.svg", bbox_inches="tight")
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
    if DOC.exists():
        raise FileExistsError(f"Refusing to overwrite {DOC}")
    OUT.mkdir(parents=True)
    model, replay, validation, curve, updates = build_model()
    curve.to_csv(OUT / "021_34_offline_reproduction_curve.csv", index=False, encoding="utf-8-sig")
    updates.to_csv(OUT / "021_34_offline_update_log.csv", index=False, encoding="utf-8-sig")

    reference = pd.read_csv(REFERENCE_CURVE)
    reference = reference[(reference.arm == "treatment_event_balanced") & (reference.milestone_updates <= 500)]
    reference = reference.sort_values("milestone_updates")
    current = curve.sort_values("milestone_updates")
    columns = ["noop_accuracy", "nonzero_action_recall", "positive_expert_margin_fraction_nonzero", "mean_expert_margin_nonzero"]
    errors = {column: float(np.max(np.abs(current[column].to_numpy() - reference[column].to_numpy()))) for column in columns}

    model_path = OUT / "event_balanced_500_frozen_model.zip"
    model.save(str(model_path))
    hash_before = parameter_hash(model)
    agent_size_before = replay.agent_size
    daily, policy = frozen_forward(model, validation["mean"], validation["scale"])
    hash_after = parameter_hash(model)
    agent_size_after = replay.agent_size
    daily.to_csv(OUT / "021_34_frozen_policy_daily.csv", index=False, encoding="utf-8-sig")
    comparison = comparison_table(policy)
    comparison.to_csv(OUT / "021_34_baseline_comparison.csv", index=False, encoding="utf-8-sig")

    if policy["yield_kg_ha"] >= 11077 and policy["irrigation_mm"] <= 266 and policy["nitrogen_kg_ha"] <= 300 and policy["late_n_after_dap90_kg_ha"] == 0:
        branch = "A"
    elif policy["yield_kg_ha"] >= 9613 and policy["irrigation_mm"] <= 266 and policy["nitrogen_kg_ha"] <= 300:
        branch = "B"
    else:
        branch = "C"
    policy["preregistered_branch"] = branch
    (OUT / "021_34_frozen_policy_summary.json").write_text(json.dumps(policy, indent=2, allow_nan=True), encoding="utf-8")

    checks = {
        "offline_500_max_abs_errors": errors,
        "offline_500_reproduced": bool(max(errors.values()) < 1e-6),
        "parameter_hash_before_forward": hash_before,
        "parameter_hash_after_forward": hash_after,
        "parameters_unchanged": hash_before == hash_after,
        "replay_agent_size_before": agent_size_before,
        "replay_agent_size_after": agent_size_after,
        "replay_unchanged_and_demo_only": bool(agent_size_before == 0 and agent_size_after == 0),
        "q_values_finite": policy["q_values_finite"],
        "season_terminated_or_truncated": policy["terminated_or_truncated"],
        "resource_totals_within_wrapper_budgets": bool(policy["safe_irrigation_sum_mm"] <= 120 and policy["safe_nitrogen_sum_kg_ha"] <= 300),
        "no_optimizer_or_online_learning": True,
        "all_required_checks_pass": False,
    }
    checks["all_required_checks_pass"] = bool(
        checks["offline_500_reproduced"] and checks["parameters_unchanged"]
        and checks["replay_unchanged_and_demo_only"] and checks["q_values_finite"]
        and checks["season_terminated_or_truncated"]
        and checks["resource_totals_within_wrapper_budgets"]
    )
    (OUT / "021_34_validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    summary = {
        "status": "completed" if checks["all_required_checks_pass"] else "failed_validation",
        "branch": branch,
        "policy": policy,
        "validation": checks,
        "interpretation": {
            "A": "冻结策略达到 official expert 产量，资源不超基准且无 DAP>90 晚期施氮；可另立在线 A/B。",
            "B": "冻结策略达到 recorded 产量且资源受控，但未通过全部 A 门槛；离线引导有价值但尚非正式策略。",
            "C": "冻结自由策略未达到 recorded 门槛或出现无意义资源行为；不进入在线训练。",
        }[branch],
        "online_training_started": False,
    }
    (OUT / "021_34_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    plot_results(daily, comparison)

    display = comparison[["scenario", "yield_kg_ha", "irrigation_mm", "nitrogen_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]].round(3)
    doc = f"""# 021_34 SY2014 event-balanced 冻结策略前向评估记录

## 目的与边界

重建 021_32 Treatment 到 500 次离线更新，随后冻结模型，仅进行一个 SY2014 DSSAT/PDI 生长季的确定性前向评估。无 epsilon、无在线梯度、无 replay 写入；reward、IC、动作空间、预算和 DSSAT 输入均未修改。

## 复现与防污染检查

- 021_32 500-update 指标最大误差：`{max(errors.values()):.3e}`；复现通过：`{checks['offline_500_reproduced']}`。
- 前向前后参数哈希一致：`{checks['parameters_unchanged']}`。
- replay agent_size 前/后：`{agent_size_before}/{agent_size_after}`。
- Q 有限、季节正常终止、资源在 wrapper 预算内：`{checks['q_values_finite']}/{checks['season_terminated_or_truncated']}/{checks['resource_totals_within_wrapper_budgets']}`。
- 在线训练：未启动。

## 冻结策略结果

- 产量：{policy['yield_kg_ha']:.1f} kg/ha；生物量：{policy['biomass_kg_ha']:.1f} kg/ha。
- 灌溉：{policy['irrigation_mm']:.1f} mm；施氮：{policy['nitrogen_kg_ha']:.1f} kg/ha。
- DAP>90 施氮：{policy['late_n_after_dap90_kg_ha']:.1f} kg/ha；非零操作次数：{policy['nonzero_operation_count']}。
- 预注册分支：**{branch}**。

## 与既有基准比较

{markdown_table(display)}

## 结论

{summary['interpretation']}

这只是单个冻结模型、单季确定性前向结果，不代表跨 seed 稳定性，也不自动授权在线 5K。

## 输出

- `benchmark_results/021_34/021_34_frozen_policy_daily.csv`
- `benchmark_results/021_34/021_34_frozen_policy_summary.json`
- `benchmark_results/021_34/021_34_baseline_comparison.csv`
- `benchmark_results/021_34/021_34_validation.json`
- `benchmark_results/021_34/021_34_frozen_policy_forward.png/.svg`
- 本地模型 `event_balanced_500_frozen_model.zip`（不纳入 Git）

## 失败尝试

第一次完整运行的数值与最终结果一致，但 Pandas 默认把情景名字符串 `null` 解析为缺失值，导致比较表和图的首个标签为空。该次输出完整保存在 `benchmark_results/021_34_failed_attempt_1_null_label_parsed_as_na/`；修复为 `keep_default_na=False` 后原样重跑，未改变任何实验变量。
"""
    DOC.write_text(doc, encoding="utf-8")
    print(json.dumps({"status": summary["status"], "branch": branch, "policy": policy, "validation": checks}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
