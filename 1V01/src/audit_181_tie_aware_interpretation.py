"""Tie-aware, read-only interpretation audit for completed task 181.

This script does not run DSSAT or PPO.  It reads the 96 candidate-horizon
rows, makes equal GRNWT outcomes explicit as ties, and writes an addendum
without replacing any completed task-181 result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "benchmark_results" / "181_sya_late_stage_short_horizon_counterfactual_corrected"
SOURCE = RESULT_ROOT / "evaluation" / "181_candidate_horizon_metrics.csv"
TIE_TABLE = RESULT_ROOT / "evaluation" / "181_tie_aware_rankings.csv"
ADDENDUM = RESULT_ROOT / "181_interpretation_addendum.json"
DOC = ROOT / "docs" / "181_sya_late_stage_short_horizon_counterfactual_corrected_interpretation_addendum.md"


def signature(group: pd.DataFrame, value_column: str) -> str:
    ordered = group.sort_values([value_column, "candidate_action_index"], ascending=[False, True])
    levels: list[list[int]] = []
    values: list[float] = []
    for row in ordered.itertuples(index=False):
        value = float(getattr(row, value_column))
        action = int(row.candidate_action_index)
        if not values or not np.isclose(value, values[-1], rtol=0.0, atol=1e-9):
            values.append(value)
            levels.append([action])
        else:
            levels[-1].append(action)
    return ">".join("=".join(str(action) for action in level) for level in levels)


if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)
if TIE_TABLE.exists() or ADDENDUM.exists() or DOC.exists():
    raise FileExistsError("Refuse to overwrite an existing task-181 interpretation addendum.")

candidate = pd.read_csv(SOURCE)
if len(candidate) != 96:
    raise ValueError(f"Expected 96 candidate-horizon rows, found {len(candidate)}")

rows: list[dict[str, object]] = []
group_columns = ["weather_variant", "target_dap", "horizon_days"]
for (variant, target_dap, horizon), group in candidate.groupby(group_columns, sort=True):
    reward_order = group.sort_values(
        ["cumulative_reward_unscaled", "candidate_action_index"], ascending=[False, True]
    )
    rows.append(
        {
            "weather_variant": str(variant),
            "target_dap": int(target_dap),
            "horizon_days": int(horizon),
            "ppo_action_index_at_target": int(group["ppo_action_index_at_target"].iloc[0]),
            "best_reward_action_index": int(reward_order.iloc[0]["candidate_action_index"]),
            "best_reward_irrigation_mm": float(reward_order.iloc[0]["candidate_irrigation_mm"]),
            "reward_ranking_tie_aware": signature(group, "cumulative_reward_unscaled"),
            "grnwt_ranking_tie_aware": signature(group, "grnwt_gain"),
            "within_group_grnwt_spread": float(group["grnwt_gain"].max() - group["grnwt_gain"].min()),
            "max_abs_marginal_grnwt_vs_noop": float(group["marginal_grnwt_vs_noop"].abs().max()),
            "max_abs_stress_delta_vs_noop": float(
                max(
                    group["end_swfac_delta_vs_noop"].abs().max(),
                    group["end_nstres_delta_vs_noop"].abs().max(),
                )
            ),
        }
    )

tie_frame = pd.DataFrame(rows)
nonzero_stress = candidate[
    candidate["end_swfac_delta_vs_noop"].abs().gt(1e-12)
    | candidate["end_nstres_delta_vs_noop"].abs().gt(1e-12)
].copy()

weather_growth_ranges = (
    candidate.groupby(["target_dap", "horizon_days"], sort=True)
    .agg(
        min_grnwt_gain=("grnwt_gain", "min"),
        max_grnwt_gain=("grnwt_gain", "max"),
        min_cumulative_rain=("cumulative_rain", "min"),
        max_cumulative_rain=("cumulative_rain", "max"),
    )
    .reset_index()
)

summary = {
    "task": "181_sya_late_stage_short_horizon_counterfactual_corrected_interpretation_addendum",
    "mode": "read_only_tie_aware_audit",
    "status": "completed",
    "source_rows": int(len(candidate)),
    "scenario_target_horizon_groups": int(len(tie_frame)),
    "all_ppo_target_actions_are_noop": bool(tie_frame["ppo_action_index_at_target"].eq(0).all()),
    "all_best_reward_actions_are_noop": bool(tie_frame["best_reward_action_index"].eq(0).all()),
    "all_reward_rankings": sorted(tie_frame["reward_ranking_tie_aware"].unique().tolist()),
    "all_grnwt_rankings": sorted(tie_frame["grnwt_ranking_tie_aware"].unique().tolist()),
    "max_within_group_grnwt_spread": float(tie_frame["within_group_grnwt_spread"].max()),
    "max_abs_marginal_grnwt_vs_noop": float(tie_frame["max_abs_marginal_grnwt_vs_noop"].max()),
    "nonzero_stress_delta_rows": int(len(nonzero_stress)),
    "max_abs_stress_delta_vs_noop": float(tie_frame["max_abs_stress_delta_vs_noop"].max()),
    "diagnosis": "weather_changes_crop_growth_level_but_not_the_7_or_14_day_marginal_value_or_ordering_of_target_irrigation",
    "interpretation": [
        "Within every weather-target-horizon group, I0/I15/I30/I45 produced identical GRNWT gain.",
        "The reward order I0>I15>I30>I45 is entirely the irrigation-cost order because yield gain is tied.",
        "Only two I45 branches changed NSTRES slightly at DAP111/day14; neither changed GRNWT.",
        "This short-horizon result does not prove irrigation has no whole-season effect.",
    ],
    "weather_growth_ranges": weather_growth_ranges.to_dict(orient="records"),
    "nonzero_stress_cases": nonzero_stress[
        [
            "weather_variant",
            "target_dap",
            "horizon_days",
            "candidate_action_index",
            "candidate_irrigation_mm",
            "end_nstres_delta_vs_noop",
            "marginal_grnwt_vs_noop",
        ]
    ].to_dict(orient="records"),
}

tie_frame.to_csv(TIE_TABLE, index=False, encoding="utf-8-sig")
ADDENDUM.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
DOC.write_text(
    "\n".join(
        [
            "# Task 181 tie-aware 结果解读补充",
            "",
            "- 该审计只读取已完成的 96 条 candidate-horizon 记录，没有运行 DSSAT 或 PPO。",
            "- 24 个情景×目标DAP×horizon 组合中，PPO 目标动作和 reward 最优动作均为 no-op。",
            "- reward 排序均为 `0>4>8>12`，对应灌溉成本递增。",
            "- GRNWT 的正确并列表示均为 `0=4=8=12`；原始 action-index 排名不应解释为产量优劣。",
            "- 同一组合内最大 GRNWT 差异为 0，天气改变了总体生长量，但没有改变目标日灌溉的 7/14 天边际产量价值。",
            "- 仅 early_wet 和 mid_wet 的 DAP111、14天、I45 分支出现轻微 NSTRES 增加，GRNWT 仍未改变。",
            "- 结论边界：不能由 14 天结果推断整季灌溉无效。",
        ]
    )
    + "\n",
    encoding="utf-8",
)

print(json.dumps(summary, ensure_ascii=False, indent=2))
