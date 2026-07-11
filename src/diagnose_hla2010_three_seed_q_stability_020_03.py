from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_hla_baseline_relative_dqn_checkpoint_015_12 as hla
import run_hla_unified_dqn_long_train_015_09 as base
from ppo_evaluate import latest_observation_dict, scalar
from run_hla_official_reward_restart_smoke import install_official_reward_module


OUT = (
    ROOT / "DSSAT_auto_validation" / "HLA_2004"
    / "hla2010_three_seed_q_diagnostic_020_03"
)
DOC = ROOT / "docs" / "2026-07-10_020_03_hla2010_three_seed_q_stability_diagnostic_record.md"
COMPARISON_020_02 = (
    ROOT / "DSSAT_auto_validation" / "HLA_2004"
    / "hla2010_seed2_stability_020_02"
    / "020_02_seed0_seed1_seed2_fixed_rule_comparison.csv"
)

RUNS = {
    0: ROOT / "DSSAT_auto_validation" / "HLA_2004"
    / "hla_baseline_relative_dqn_checkpoint_015_12" / "2010"
    / "baseline_relative_seed0_50000steps",
    1: ROOT / "DSSAT_auto_validation" / "HLA_2004"
    / "hla_baseline_relative_dqn_checkpoint_015_12" / "2010"
    / "baseline_relative_seed1_50000steps",
    2: ROOT / "DSSAT_auto_validation" / "HLA_2004"
    / "hla2010_seed2_stability_020_02" / "2010"
    / "baseline_relative_seed2_50000steps",
}


def q_values(model: Any, obs: Any) -> np.ndarray:
    import torch

    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        values = model.policy.q_net(obs_tensor)
    return values.detach().cpu().numpy().reshape(-1)


def model_path(seed: int, step: int) -> Path:
    return RUNS[seed] / "models" / f"dqn_baseline_relative_checkpoint_{step}.zip"


def load_selected_models() -> tuple[pd.DataFrame, dict[int, Any]]:
    from stable_baselines3 import DQN

    selected = pd.read_csv(COMPARISON_020_02).sort_values("seed")
    models = {}
    for row in selected.itertuples(index=False):
        seed = int(row.seed)
        path = model_path(seed, int(row.checkpoint_step))
        if not path.exists():
            raise FileNotFoundError(path)
        models[seed] = DQN.load(str(path), device="cpu", print_system_info=False)
    return selected, models


def action_description(index: int) -> tuple[float, float]:
    action = base.ACTION_TABLE_9[int(index)]
    return float(action["amir"]), float(action["anfer"])


def run_common_state_probes(models: dict[int, Any]) -> pd.DataFrame:
    reference_env_args = json.loads((RUNS[0] / "env_args.json").read_text(encoding="utf-8"))
    null_baseline = hla.get_null_baseline_yield(2010)
    frames = []
    for reference_seed, reference_model in models.items():
        env_args = dict(reference_env_args)
        env_args["log_saving_path"] = str(OUT / f"reference_seed{reference_seed}.log")
        env = hla.make_env(env_args, null_baseline)
        rows = []
        try:
            obs, info = env.reset()
            for step in range(260):
                latest_before = latest_observation_dict(env, obs, info)
                dap_before = scalar(latest_before.get("dap", np.nan))
                reference_action, _ = reference_model.predict(obs, deterministic=True)
                reference_action = int(np.asarray(reference_action).item())
                probe_values = {}
                for probe_seed, probe_model in models.items():
                    q = q_values(probe_model, obs)
                    order = np.argsort(q)
                    best = int(order[-1])
                    second = int(order[-2])
                    gap = float(q[best] - q[second])
                    normalized_gap = gap / max(abs(float(q[best])), 1.0e-8)
                    probe_values[probe_seed] = (q, best, second, gap, normalized_gap)

                obs_after, reward, terminated, truncated, info_after = env.step(reference_action)
                latest_after = latest_observation_dict(env, obs_after, info_after)
                safe_i = float(env.last_safe_real_action.get("amir", 0.0))
                safe_n = float(env.last_safe_real_action.get("anfer", 0.0))
                for probe_seed, (q, best, second, gap, normalized_gap) in probe_values.items():
                    raw_i, raw_n = action_description(best)
                    row = {
                        "reference_seed": reference_seed,
                        "probe_seed": probe_seed,
                        "step": step,
                        "dap_before": dap_before,
                        "reference_action": reference_action,
                        "probe_best_action": best,
                        "probe_second_action": second,
                        "action_agrees_with_reference": best == reference_action,
                        "probe_best_raw_irrigation": raw_i,
                        "probe_best_raw_nitrogen": raw_n,
                        "q_best": float(q[best]),
                        "q_second": float(q[second]),
                        "q_gap_best_second": gap,
                        "q_gap_normalized": normalized_gap,
                        "reference_safe_irrigation": safe_i,
                        "reference_safe_nitrogen": safe_n,
                        "reference_used_irrigation": float(env.used_irrigation),
                        "reference_used_nitrogen": float(env.used_nitrogen),
                        "reference_reward": float(reward),
                        "grnwt_after": scalar(latest_after.get("grnwt")),
                        "topwt_after": scalar(latest_after.get("topwt")),
                        "swfac_after": scalar(latest_after.get("swfac")),
                        "nstres_after": scalar(latest_after.get("nstres")),
                        "done": bool(terminated or truncated),
                    }
                    for action_index, q_value in enumerate(q):
                        row[f"q_action_{action_index}"] = float(q_value)
                    rows.append(row)
                obs, info = obs_after, info_after
                if terminated or truncated:
                    break
        finally:
            env.close()
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def summarize_own_trajectories(probes: pd.DataFrame) -> pd.DataFrame:
    own = probes[probes["reference_seed"].eq(probes["probe_seed"])].copy()
    rows = []
    for seed, group in own.groupby("reference_seed"):
        action_counts = Counter(group["probe_best_action"].astype(int))
        rows.append(
            {
                "seed": int(seed),
                "steps": len(group),
                "final_grnwt": float(group["grnwt_after"].dropna().iloc[-1]),
                "irrigation_total": float(group["reference_safe_irrigation"].sum()),
                "nitrogen_total": float(group["reference_safe_nitrogen"].sum()),
                "total_reward": float(group["reference_reward"].sum()),
                "mean_q_gap": float(group["q_gap_best_second"].mean()),
                "median_q_gap": float(group["q_gap_best_second"].median()),
                "mean_normalized_q_gap": float(group["q_gap_normalized"].mean()),
                "small_normalized_gap_rate_lt_1pct": float((group["q_gap_normalized"].abs() < 0.01).mean()),
                "action_counts": json.dumps(dict(sorted(action_counts.items())), ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows).sort_values("seed")


def summarize_common_state_agreement(probes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (reference_seed, probe_seed), group in probes.groupby(["reference_seed", "probe_seed"]):
        rows.append(
            {
                "reference_seed": int(reference_seed),
                "probe_seed": int(probe_seed),
                "states": len(group),
                "action_agreement_rate": float(group["action_agrees_with_reference"].mean()),
                "mean_normalized_q_gap": float(group["q_gap_normalized"].mean()),
                "small_normalized_gap_rate_lt_1pct": float((group["q_gap_normalized"].abs() < 0.01).mean()),
                "predicted_irrigation_action_rate": float((group["probe_best_raw_irrigation"] > 0).mean()),
                "predicted_nitrogen_action_rate": float((group["probe_best_raw_nitrogen"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["reference_seed", "probe_seed"])


def checkpoint_evolution(selected: pd.DataFrame) -> pd.DataFrame:
    from stable_baselines3 import DQN

    frames = []
    for seed, run_dir in RUNS.items():
        summary = pd.read_csv(run_dir / "checkpoint_summary.csv")
        chosen_step = int(selected[selected["seed"].eq(seed)].iloc[0]["checkpoint_step"])
        metadata = []
        for step in summary["checkpoint_step"].astype(int):
            model = DQN.load(str(model_path(seed, step)), device="cpu", print_system_info=False)
            metadata.append(
                {
                    "checkpoint_step": step,
                    "model_num_timesteps": int(model.num_timesteps),
                    "model_total_timesteps_schedule": int(getattr(model, "_total_timesteps", -1)),
                    "exploration_rate": float(model.exploration_rate),
                }
            )
        metadata_df = pd.DataFrame(metadata)
        summary = summary.merge(metadata_df, on="checkpoint_step", how="left")
        summary.insert(0, "seed", seed)
        summary["selected_by_fixed_rule"] = summary["checkpoint_step"].astype(int).eq(chosen_step)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True).sort_values(["seed", "checkpoint_step"])


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in df.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.4f}".rstrip("0").rstrip("."))
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_record(own: pd.DataFrame, agreement: pd.DataFrame, evolution: pd.DataFrame) -> None:
    selected_evolution = evolution[evolution["selected_by_fixed_rule"]].copy()
    selected_view = selected_evolution[
        [
            "seed", "checkpoint_step", "final_grain_kg_ha",
            "action_irrigation_total", "action_fertilizer_total",
            "total_reward", "exploration_rate",
        ]
    ]
    cross_only = agreement[agreement["reference_seed"].ne(agreement["probe_seed"])]
    seed1_on_seed2 = float(
        agreement[(agreement["reference_seed"].eq(2)) & (agreement["probe_seed"].eq(1))]
        .iloc[0]["action_agreement_rate"]
    )
    seed2_on_seed1 = float(
        agreement[(agreement["reference_seed"].eq(1)) & (agreement["probe_seed"].eq(2))]
        .iloc[0]["action_agreement_rate"]
    )
    seed0_cross = agreement[
        agreement["reference_seed"].ne(agreement["probe_seed"])
        & ((agreement["reference_seed"].eq(0)) | (agreement["probe_seed"].eq(0)))
    ]["action_agreement_rate"].mean()
    lines = [
        "# 020_03 HLA2010 三 seed Q 值与策略稳定性诊断记录",
        "",
        "## 历史边界",
        "",
        "旧项目做过不同动作空间或不同奖励线路的Q值诊断；本轮首次对015_12当前统一9动作、baseline-relative reward的三个正式seed进行同状态诊断，不重新训练。",
        "",
        "## 固定规则选定模型",
        "",
        markdown_table(selected_view),
        "",
        "## 各模型自身确定性轨迹",
        "",
        markdown_table(own),
        "",
        "## 相同状态上的跨模型动作一致率",
        "",
        markdown_table(agreement),
        "",
        "## 机制判读",
        "",
        f"- seed1 与 seed2 在对方轨迹的相同状态上动作一致率分别为 {seed1_on_seed2:.3f} 和 {seed2_on_seed1:.3f}，说明二者复现的是同一个低灌溉策略族，而不是两个互不相关的失败。",
        f"- seed0 与另外两个模型在相同状态上的平均动作一致率仅 {seed0_cross:.3f}，说明 seed0 形成了另一套动作价值排序。",
        "- seed0 自身轨迹主要把 action2（I30/N0）排在首位；seed1、seed2主要在 action0（no-op）和 action1（I15/N0）之间选择，经过预算和7天间隔约束后分别形成 I120 与 I60。",
        "- 三个模型当前 exploration_rate 都是0.05，且评估采用 deterministic=True，因此最终差异不是评估时随机探索造成，而是Q网络学到的动作排序不同。",
        "- seed1/2的最高奖励仍低于seed0，说明当前reward本身更偏好seed0策略；问题更接近训练没有稳定到达同一个高奖励解，而不是目标函数把I60判为更优。",
        "- 部分状态最佳与次佳Q值间隔较小，但总体上不能把分叉简单归因于所有状态Q值都接近；主要证据是两类稳定动作排序。",
        "",
        "## 结果边界",
        "",
        f"- 三个选定模型的 exploration_rate 范围：{selected_evolution['exploration_rate'].min():.4f}–{selected_evolution['exploration_rate'].max():.4f}。",
        f"- 不同模型在相同状态上的平均动作一致率：{cross_only['action_agreement_rate'].mean():.3f}。",
        "- Q值绝对大小不能跨模型直接比较；本记录仅比较相同状态下的动作排序、动作一致率和模型内部Q间隔。",
        "- 本轮是机制诊断，不修改reward、IC、动作空间或训练参数。",
        "",
        "## 输出",
        "",
        "- `020_03_common_state_q_values_daily.csv`：三条参考轨迹上三个模型的逐状态9动作Q值。",
        "- `020_03_own_trajectory_summary.csv`：各模型自身策略结果与Q间隔。",
        "- `020_03_common_state_action_agreement.csv`：相同状态下动作一致率。",
        "- `020_03_three_seed_checkpoint_evolution.csv`：全部30个checkpoint的行为和探索率。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    required = [COMPARISON_020_02] + [path / "checkpoint_summary.csv" for path in RUNS.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prior evidence:\n" + "\n".join(missing))

    OUT.mkdir(parents=True, exist_ok=True)
    base.configure_shared_settings()
    install_official_reward_module()
    selected, models = load_selected_models()
    probes = run_common_state_probes(models)
    own = summarize_own_trajectories(probes)
    agreement = summarize_common_state_agreement(probes)
    evolution = checkpoint_evolution(selected)

    probes.to_csv(OUT / "020_03_common_state_q_values_daily.csv", index=False, encoding="utf-8-sig")
    own.to_csv(OUT / "020_03_own_trajectory_summary.csv", index=False, encoding="utf-8-sig")
    agreement.to_csv(OUT / "020_03_common_state_action_agreement.csv", index=False, encoding="utf-8-sig")
    evolution.to_csv(OUT / "020_03_three_seed_checkpoint_evolution.csv", index=False, encoding="utf-8-sig")
    write_record(own, agreement, evolution)
    print(own.to_string(index=False))
    print(agreement.to_string(index=False))


if __name__ == "__main__":
    main()
