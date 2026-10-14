"""Read-only correction for Task192 retry1 nitrogen/profit aggregation.

The retry1 DSSAT trajectories are valid.  Its original export read cumulative
nitrogen from the wrong runtime attribute, so raw-profit fields were NaN.
This script never calls DSSAT or PPO; it reconstructs applied N from the saved
safe-action sequences, writes only new correction files, and recomputes ranks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "192_sya_critic_fullseason_counterfactual_audit_retry1"
CORRECTION = OUT / "correction"
DOC = ROOT / "docs" / "192_sya_critic_fullseason_counterfactual_audit_corrected_record.md"
KEY = ["checkpoint_step", "scenario_key", "weather_variant", "target_dap"]
EVENT = re.compile(r"DAP\d+\s+I([0-9.]+)/N([0-9.]+)")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def total_n(sequence: object) -> float:
    return float(sum(float(match.group(2)) for match in EVENT.finditer(str(sequence))))


def total_i(sequence: object) -> float:
    return float(sum(float(match.group(1)) for match in EVENT.finditer(str(sequence))))


def rank_tables(branches: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = branches[branches["dssat_success"].astype(bool)].copy()
    target_index = targets.set_index(KEY)
    rows = []
    for values, group in ok.groupby(KEY, sort=True):
        profit = group.sort_values(["remaining_profit_simple_corrected", "candidate_action_index"], ascending=[False, True])
        reward = group.sort_values(["remaining_reward_return_scaled", "candidate_action_index"], ascending=[False, True])
        td = group.sort_values(["one_step_td_scaled", "candidate_action_index"], ascending=[False, True])
        true_top, reward_top, td_top = profit.iloc[0], reward.iloc[0], td.iloc[0]
        state = target_index.loc[values]
        rows.append({
            **dict(zip(KEY, values)),
            "successful_candidate_count": int(len(group)),
            "true_profit_top_action": int(true_top.candidate_action_index),
            "true_profit_top_pair": str(true_top.candidate_pair),
            "true_profit_top_remaining_profit": float(true_top.remaining_profit_simple_corrected),
            "reward_return_top_action": int(reward_top.candidate_action_index),
            "td_top_action": int(td_top.candidate_action_index),
            "td_top_pair": str(td_top.candidate_pair),
            "td_top_matches_true_profit_top": bool(int(td_top.candidate_action_index) == int(true_top.candidate_action_index)),
            "reward_return_top_matches_true_profit_top": bool(int(reward_top.candidate_action_index) == int(true_top.candidate_action_index)),
            "td_top_matches_reward_return_top": bool(int(td_top.candidate_action_index) == int(reward_top.candidate_action_index)),
            "actor_selected_action": int(state.actor_selected_action),
            "actor_selected_pair": str(state.actor_selected_pair),
            "actor_matches_true_profit_top": bool(int(state.actor_selected_action) == int(true_top.candidate_action_index)),
            "actor_matches_td_top": bool(int(state.actor_selected_action) == int(td_top.candidate_action_index)),
        })
    aligned = pd.DataFrame(rows)
    calibration = aligned[[*KEY, "successful_candidate_count", "td_top_action", "td_top_pair", "true_profit_top_action", "true_profit_top_pair", "true_profit_top_remaining_profit", "td_top_matches_true_profit_top", "reward_return_top_action", "reward_return_top_matches_true_profit_top", "td_top_matches_reward_return_top"]]
    actor = aligned[[*KEY, "actor_selected_action", "actor_selected_pair", "td_top_action", "true_profit_top_action", "actor_matches_true_profit_top", "actor_matches_td_top"]]
    return calibration, actor


def main() -> None:
    branches_path = OUT / "evaluation" / "192_fullseason_counterfactual.csv"
    targets_path = OUT / "evaluation" / "192_target_state_values.csv"
    if CORRECTION.exists() and any(CORRECTION.iterdir()):
        raise FileExistsError(f"Refuse to overwrite non-empty {rel(CORRECTION)}")
    branches = pd.read_csv(branches_path, keep_default_na=False)
    targets = pd.read_csv(targets_path, keep_default_na=False)
    if len(branches) != 192 or len(targets) != 48:
        raise ValueError("Task192 retry1 input is incomplete; correction cannot proceed.")
    CORRECTION.mkdir(exist_ok=True)
    prefix_n = targets[[*KEY, "prefix_nonzero_action_sequence", "cumulative_irrigation_before", "grnwt"]].copy()
    prefix_n["prefix_n_from_safe_actions"] = prefix_n["prefix_nonzero_action_sequence"].map(total_n)
    prefix_n["prefix_i_from_safe_actions"] = prefix_n["prefix_nonzero_action_sequence"].map(total_i)
    corrected = branches.merge(prefix_n[[*KEY, "prefix_n_from_safe_actions", "prefix_i_from_safe_actions", "cumulative_irrigation_before", "grnwt"]], on=KEY, how="left", validate="many_to_one")
    corrected["future_n_from_safe_actions"] = corrected["future_nonzero_action_sequence"].map(total_n)
    corrected["future_i_from_safe_actions"] = corrected["future_nonzero_action_sequence"].map(total_i)
    corrected["total_n_corrected"] = corrected["prefix_n_from_safe_actions"] + corrected["future_n_from_safe_actions"]
    corrected["remaining_n_corrected"] = corrected["future_n_from_safe_actions"]
    corrected["total_irrigation"] = pd.to_numeric(corrected["total_irrigation"], errors="coerce")
    corrected["cumulative_irrigation_before"] = pd.to_numeric(corrected["cumulative_irrigation_before"], errors="coerce")
    corrected["final_grnwt"] = pd.to_numeric(corrected["final_grnwt"], errors="coerce")
    corrected["grnwt"] = pd.to_numeric(corrected["grnwt"], errors="coerce")
    corrected["remaining_irrigation_corrected"] = corrected["total_irrigation"] - corrected["cumulative_irrigation_before"]
    corrected["remaining_profit_simple_corrected"] = (corrected["final_grnwt"] - corrected["grnwt"]) - 1.1 * corrected["remaining_irrigation_corrected"] - 1.58 * corrected["remaining_n_corrected"]
    corrected["full_season_profit_simple_corrected"] = corrected["final_grnwt"] - 1.1 * corrected["total_irrigation"] - 1.58 * corrected["total_n_corrected"]
    corrected["PFP_N_corrected"] = corrected["final_grnwt"] / corrected["total_n_corrected"].where(corrected["total_n_corrected"].gt(0))
    calibration, actor = rank_tables(corrected, targets)
    success = corrected["dssat_success"].astype(bool)
    illegal = corrected["error"].eq("candidate_illegal_under_target_mask")
    actual_fail = (~success) & (~illegal)
    pivot = calibration.pivot_table(index=["scenario_key", "weather_variant", "target_dap"], columns="checkpoint_step", values="td_top_matches_true_profit_top", aggfunc="first")
    same_td_errors = int((pivot.eq(False).all(axis=1)).sum())
    td_profit = float(calibration["td_top_matches_true_profit_top"].mean())
    actor_profit = float(actor["actor_matches_true_profit_top"].mean())
    reward_profit = float(calibration["reward_return_top_matches_true_profit_top"].mean())
    td_reward = float(calibration["td_top_matches_reward_return_top"].mean())
    if td_profit < 0.5 and same_td_errors > 0:
        label = "supports_critic_or_temporal_credit_miscalibration"
        conclusion = "TD与真实剩余profit在多数状态不一致，且50K/100K有共同误判；支持critic/时间信用问题。"
    elif actor_profit < 0.5:
        label = "supports_actor_policy_misalignment"
        conclusion = "TD对真实剩余profit的排序总体更接近，但actor很少选择真实最佳候选动作；主要证据指向actor策略与长期收益不一致，而不是训练步数不足或单纯critic误判。"
    else:
        label = "mixed_or_inconclusive"
        conclusion = "TD、actor与真实剩余profit没有形成单一的稳定失配模式。"
    corrected.to_csv(CORRECTION / "192_fullseason_counterfactual_n_profit_corrected.csv", index=False, encoding="utf-8-sig")
    calibration.to_csv(CORRECTION / "192_critic_calibration_summary_corrected.csv", index=False, encoding="utf-8-sig")
    actor.to_csv(CORRECTION / "192_actor_vs_true_best_action_corrected.csv", index=False, encoding="utf-8-sig")
    result = {
        "task": "192_sya_critic_fullseason_counterfactual_audit",
        "status": "completed_with_read_only_n_profit_correction",
        "correction_scope": "No PPO/DSSAT rerun. Nitrogen and raw profit reconstructed from saved safe-action sequences.",
        "input_branch_records": 192, "input_target_states": 48,
        "dssat_executed_success_count": int(success.sum()), "candidate_illegal_not_executed_count": int(illegal.sum()), "dssat_runtime_failure_count": int(actual_fail.sum()),
        "td_top_matches_true_profit_top_rate": td_profit, "actor_matches_true_profit_top_rate": actor_profit,
        "reward_return_top_matches_true_profit_top_rate": reward_profit, "td_top_matches_reward_return_top_rate": td_reward,
        "same_td_misclassification_50k_100k_states": same_td_errors, "primary_label": label, "conclusion": conclusion,
        "outputs": {"corrected_branches": rel(CORRECTION / "192_fullseason_counterfactual_n_profit_corrected.csv"), "calibration": rel(CORRECTION / "192_critic_calibration_summary_corrected.csv"), "actor": rel(CORRECTION / "192_actor_vs_true_best_action_corrected.csv")},
    }
    (CORRECTION / "192_result_n_profit_correction.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    DOC.write_text("\n".join([
        "# Task192：critic—整季收益反事实审计（N/profit校正记录）", "",
        "## 结论", "", result["conclusion"], "",
        "## 为什么需要校正", "", "retry1的180条DSSAT轨迹均完成，但原导出错误地从`safety_state.cumulative_nitrogen`读取N；实际运行时N累计量位于`last_action_info.season_cumulative_n`。原始raw profit因此为NaN，不能用于排序。", "",
        "本校正不运行DSSAT、不重新训练PPO：从每条已保存的安全动作序列解析实际N，重新计算remaining/full-season raw profit和PFP_N。原始retry1输出保留未改。", "",
        "## 完整性", "", f"- 目标状态：48；候选记录：192。", f"- DSSAT实际成功：{int(success.sum())}；候选动作因mask不合法而未执行：{int(illegal.sum())}（均为DAP2的I45/N0）；DSSAT运行错误：{int(actual_fail.sum())}。", f"- TD top—真实profit top：{td_profit:.1%}；actor—真实profit top：{actor_profit:.1%}。", f"- 50K与100K共有TD误判状态：{same_td_errors}。", "",
        "## 证据边界", "", "候选动作后仍由当前冻结PPO继续控制，因此这衡量的是当前策略闭环下的局部长期后果，不是重新优化后的全局最优控制。", "",
        "## 文件", "", f"- 校正结果：`{rel(CORRECTION / '192_result_n_profit_correction.json')}`", f"- 分支表：`{rel(CORRECTION / '192_fullseason_counterfactual_n_profit_corrected.csv')}`", f"- critic校准表：`{rel(CORRECTION / '192_critic_calibration_summary_corrected.csv')}`", f"- actor对照表：`{rel(CORRECTION / '192_actor_vs_true_best_action_corrected.csv')}`", "",
    ]) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
