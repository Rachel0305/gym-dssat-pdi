#!/usr/bin/env python3
"""Build the 028_03 no-training evidence/gap registry."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "028_03_screened_year_rl_evidence_gap"
DOC = ROOT / "docs" / "2026-07-18_028_03_screened_year_rl_evidence_gap.md"


ROWS = [
    # site, year, tier, role, selection basis
    ("HLA", 2007, "A", "screened_candidate", "014_02_null_auto_response_screen"),
    ("HLA", 2010, "A", "training_anchor", "014_02_null_auto_response_screen"),
    ("HLA", 2015, "B", "screened_candidate", "014_02_null_auto_response_screen"),
    ("HLA", 2016, "B", "screened_candidate", "014_02_null_auto_response_screen"),
    ("HLA", 2022, "B", "screened_candidate", "014_02_null_auto_response_screen"),
    ("YC", 2008, "A", "secondary_candidate", "013_01_forward_optimization_space_screen"),
    ("YC", 2014, "A", "training_anchor", "013_01_forward_optimization_space_screen"),
    ("FQ", 2013, "B", "screened_candidate", "014_01_all_year_optimization_space_screen"),
    ("FQ", 2014, "B", "screened_candidate", "014_01_all_year_optimization_space_screen"),
    ("FQ", 2016, "B", "training_anchor", "014_01_all_year_optimization_space_screen"),
    ("FQ", 2019, "B", "screened_candidate", "014_01_all_year_optimization_space_screen"),
    ("FQ", 2020, "B", "screened_candidate", "014_01_all_year_optimization_space_screen"),
    ("FQ", 2023, "B", "screened_candidate", "014_01_all_year_optimization_space_screen"),
    ("LC", 2010, "A", "training_anchor", "017_11_fixed_input_baseline_screen"),
    ("SY", 2012, "A", "fixed_weight_validation_year", "026_07_authoritative_crossyear_protocol"),
    ("SY", 2014, "A", "training_anchor", "current_authoritative_treatment_and_026_07"),
    ("SY", 2015, "A", "fixed_weight_validation_year", "026_07_authoritative_crossyear_protocol"),
]


CURRENT = {
    ("HLA", 2010): dict(algorithm="stage_maskable_ppo", seeds="0,1,2", rl_status="trained_three_seed", baseline="complete_current", endpoint="complete", daily="complete_selected_seed", figure="complete_selected_seed", any_winner="yes_selected_seed_PFP_N", next="report_existing_and_keep_other_seeds_in_table"),
    ("YC", 2014): dict(algorithm="stage_maskable_ppo", seeds="0,1,2", rl_status="trained_three_seed", baseline="complete_current", endpoint="complete", daily="missing_for_current_ppo", figure="missing_for_current_ppo", any_winner="2_of_3_PFP_N", next="reuse_models_and_snapshots_then_complete_daily_figures"),
    ("FQ", 2016): dict(algorithm="stage_maskable_ppo", seeds="0", rl_status="seed0_stopped_by_old_primary", baseline="complete_current", endpoint="complete", daily="missing_for_current_ppo", figure="historical_dqn_only", any_winner="0_of_1", next="do_not_repeat_seed0; advisor_rule_requires_new_preregistered_decision_before_more_seeds"),
    ("LC", 2010): dict(algorithm="stage_maskable_ppo", seeds="0", rl_status="seed0_stopped_by_old_primary", baseline="complete_current", endpoint="complete", daily="missing_for_current_ppo", figure="historical_dqn_only", any_winner="1_of_1_PFP_N", next="reuse_seed0_and_complete_current_ppo_daily_figures"),
    ("SY", 2012): dict(algorithm="stage_maskable_ppo", seeds="0,1,2", rl_status="frozen_sy2014_transfer", baseline="complete_current", endpoint="complete", daily="snapshots_exist", figure="needs_02705_style_confirmation", any_winner="needs_advisor_rule_summary", next="no_training; reuse_frozen_evaluations"),
    ("SY", 2014): dict(algorithm="stage_maskable_ppo", seeds="0,1,2", rl_status="trained_three_seed", baseline="complete_current", endpoint="complete", daily="complete_selected_seed", figure="complete_selected_seed", any_winner="3_of_3_at_least_one_comparable_metric", next="report_existing"),
    ("SY", 2015): dict(algorithm="stage_maskable_ppo", seeds="0,1,2", rl_status="frozen_sy2014_transfer", baseline="complete_current", endpoint="complete", daily="snapshots_exist", figure="needs_02705_style_confirmation", any_winner="needs_advisor_rule_summary", next="no_training; reuse_frozen_evaluations"),
}


def exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    registry = []
    for site, year, tier, role, basis in ROWS:
        current = CURRENT.get((site, year), {})
        row = {
            "site": site,
            "year": year,
            "evidence_tier": tier,
            "candidate_role": role,
            "selection_basis": basis,
            "baseline_status": current.get("baseline", "missing_current_four_baselines"),
            "rl_algorithm": current.get("algorithm", "none_current_stage_ppo"),
            "seeds_available": current.get("seeds", "none"),
            "rl_status": current.get("rl_status", "not_run_current_stage_ppo"),
            "endpoint_status": current.get("endpoint", "missing"),
            "daily_status": current.get("daily", "missing"),
            "figure_status": current.get("figure", "missing"),
            "advisor_any_metric_status": current.get("any_winner", "not_evaluated"),
            "next_action": current.get("next", "complete_baselines_then_decide_training_or_fixed_transfer"),
        }
        registry.append(row)

    csv_path = OUT / "028_03_screened_year_evidence_gap.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(registry[0]))
        writer.writeheader()
        writer.writerows(registry)

    summary = {
        "status": "completed_audit_only",
        "candidate_year_count": len(registry),
        "current_four_baseline_complete": sum(r["baseline_status"] == "complete_current" for r in registry),
        "current_stage_ppo_present": sum(r["rl_algorithm"] == "stage_maskable_ppo" for r in registry),
        "current_stage_ppo_daily_or_snapshot_present": sum(r["daily_status"] in {"complete_selected_seed", "snapshots_exist"} for r in registry),
        "training_or_dssat_calls": 0,
        "scope_note": "Tier is provenance; selection_basis is optimization-screen evidence.",
    }
    (OUT / "028_03_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 028_03 已筛选年份 RL 证据缺口审计记录",
        "",
        "## 结论",
        "",
        f"共登记 **{len(registry)}** 个首批站点—年份。本轮为零训练、零 DSSAT 文件审计。",
        f"其中当前四基线完整 {summary['current_four_baseline_complete']} 个，已有当前阶段型 MaskablePPO 证据 {summary['current_stage_ppo_present']} 个。",
        "Tier 只表示输入来源；是否来自历史优化空间筛选由 `selection_basis` 独立记录。",
        "",
        "## 逐年缺口矩阵",
        "",
        "|站点|年份|Tier|角色|筛选依据|四基线|RL|seed|日值|图|导师单项领先状态|下一步|",
        "|---|---:|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in registry:
        lines.append("|{site}|{year}|{evidence_tier}|{candidate_role}|{selection_basis}|{baseline_status}|{rl_algorithm}|{seeds_available}|{daily_status}|{figure_status}|{advisor_any_metric_status}|{next_action}|".format(**r))
    lines += [
        "",
        "## 判据边界",
        "",
        "- 至少一项严格第一是硬标签；另外两项只报告差值和百分比差。",
        "- 导师尚未定义‘接近’容差，本任务不自行设阈值。",
        "- 历史 DQN 图表保留，但受 021_05 训练协议问题影响的结果不冒充当前统一框架。",
        "- 下一步优先补当前模型已有但日值/图缺失的 YC2014、LC2010，以及复用 SY2012/2015 snapshot；不重复训练。",
        "- 其余年份先补四基线和 provenance，再决定固定权重迁移或本地训练。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("028_03 screened-year evidence gap audit completed")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
