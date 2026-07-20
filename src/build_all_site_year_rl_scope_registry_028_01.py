#!/usr/bin/env python3
"""Build the read-only 028_01 station-year scope and evidence registry.

This script does not import gym-DSSAT, start Docker, run DSSAT, or train a model.
It inventories weather files and combines them with explicitly documented
station-year evidence.  Missing evidence remains missing; it is never inferred
from the mere presence of a WTH file.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "028_01_all_site_year_scope_registry"
DOC = ROOT / "docs" / "2026-07-18_028_01_all_site_year_scope_and_feasibility_registry_audit.md"

SITE_DIR = {"HLA": "HL", "SY": "SY", "YC": "YC", "FQ": "FQ", "LC": "LC"}
SITE_CODE = {"HLA": "HL", "SY": "SY", "YC": "YC", "FQ": "FQ", "LC": "LC"}
INPUT_BASE = Path("DSSAT_auto_validation/multisite_new_cultivar_inputs_013")
MZX_SOURCE = {
    "HLA": [
        "DSSAT_auto_validation/multisite_new_cultivar_inputs_013/HL/CNHL0701_corrected_IC123.MZX",
        "DSSAT_auto_validation/multisite_new_cultivar_inputs_013/HL/CNHL10N1.MZX",
    ],
    "SY": ["DSSAT_auto_validation/multisite_new_cultivar_inputs_013/SY/CNSY1201.MZX"],
    "YC": ["DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC/CNYC0801.MZX"],
    "FQ": ["DSSAT_auto_validation/multisite_new_cultivar_inputs_013/FQ/CNFQ0801.MZX"],
    "LC": ["DSSAT_auto_validation/multisite_new_cultivar_inputs_013/LC/CNLC0801.MZX"],
}

COMMON_SOURCES = {
    "HLA": [
        "prompts/020_11_hla_five_scenario_expert_completion_and_nstep_freeze.md",
        "benchmark_results/027_03/027_03_result.json",
        "docs/2026-07-17_027_03_hla2010_stage_maskable_ppo_three_seed_replication.md",
    ],
    "SY": [
        "prompts/026_06_sy_all_authoritative_years_frozen_stage_ppo_validation.md",
        "benchmark_results/026_07_attempt2/026_07_result.json",
        "docs/2026-07-17_026_07_sy_icdat_aligned_all_years_frozen_stage_ppo_validation.md",
    ],
    "YC": [
        "benchmark_results/027_07_site_specific_stage_maskable_ppo_attempt2/027_07_result.json",
        "docs/2026-07-17_027_07_yc_fq_lc_site_specific_stage_maskable_ppo_attempt2.md",
        "docs/2026-07-05_016_11_yc2014_cross_year_transfer_success_plots_record_fixed.md",
    ],
    "FQ": [
        "benchmark_results/027_07_site_specific_stage_maskable_ppo_attempt2/027_07_result.json",
        "docs/2026-06-30_014_01_fq_all_year_screen_and_dqn_transfer_record.md",
        "docs/2026-07-02_016_05_deterministic_oracle_upper_bound_scan_record.md",
    ],
    "LC": [
        "benchmark_results/027_07_site_specific_stage_maskable_ppo_attempt2/027_07_result.json",
        "docs/2026-07-07_017_11_lc_fixed_input_year_screening_record.md",
    ],
}

HASH_SOURCES = sorted(
    {
        "prompts/028_00_five_station_all_optimizable_year_strict_rl_framework.md",
        "prompts/028_01_all_site_year_scope_and_feasibility_registry_audit.md",
        "docs/2026-07-13_021_05_dqn_training_protocol_correction.md",
        *[p for paths in MZX_SOURCE.values() for p in paths],
        *[p for paths in COMMON_SOURCES.values() for p in paths],
    }
)


def fact(
    tier: str,
    treatment: str,
    provenance: str,
    baseline: str,
    optimization: str,
    latest_algorithm: str = "none",
    same_year: str = "not_run",
    crossyear: str = "not_run",
    strict_status: str = "not_evaluated_under_028",
    eligibility: str = "needs_028_baseline_and_feasibility_audit",
    notes: str = "",
) -> dict[str, str]:
    return {
        "evidence_tier": tier,
        "treatment_status": treatment,
        "input_provenance_status": provenance,
        "four_baseline_status": baseline,
        "strict_feasibility_status": "not_certified_under_028",
        "optimization_space_status": optimization,
        "latest_rl_algorithm": latest_algorithm,
        "latest_same_year_training_status": same_year,
        "latest_fixed_weight_crossyear_status": crossyear,
        "strict_all_four_three_metric_status": strict_status,
        "next_phase_eligibility": eligibility,
        "notes": notes,
    }


FACTS: dict[tuple[str, int], dict[str, str]] = {}


def add(site: str, years: list[int], **kwargs: str) -> None:
    for year in years:
        FACTS[(site, year)] = fact(**kwargs)


# HLA: preserve the distinction between original multi-treatment inputs and
# HLA2010-style prepared/shifted inputs used in later formal adapters.
add(
    "HLA", [2007],
    tier="A_authoritative_treatment",
    treatment="original_treatment_and_separate_prepared_variant",
    provenance="needs_variant_lock_before_028",
    baseline="complete_historical_for_prepared_variant",
    optimization="historical_candidate_needs_strict_recertification",
    same_year="latest_stage_ppo_not_run_for_this_year",
    eligibility="blocked_until_input_variant_locked",
    notes="Original 2007 treatment exists, but later formal HLA set used a separate HLA2010-style prepared variant.",
)
add(
    "HLA", [2009, 2011],
    tier="A_authoritative_treatment",
    treatment="original_multitreatment_mzx",
    provenance="documented_not_revalidated_under_028",
    baseline="not_revalidated_as_four_complete_baselines",
    optimization="needs_028_feasibility",
    eligibility="needs_four_baselines_and_feasibility",
)
add(
    "HLA", [2010],
    tier="A_authoritative_treatment",
    treatment="formal_single_year_anchor",
    provenance="current_formal_anchor",
    baseline="complete_current",
    optimization="historical_candidate_needs_strict_recertification",
    latest_algorithm="stage_maskable_ppo",
    same_year="three_seeds_run_old_primary_1_of_3",
    crossyear="not_started_by_preregistered_stop",
    eligibility="needs_028_strict_feasibility",
    notes="0/3 selected models passed the separate recorded comparison.",
)
add(
    "HLA", [2015, 2016, 2022],
    tier="B_derived_weather_replay",
    treatment="hla2010_style_prepared_year_input",
    provenance="formal_prepared_adapter_not_original_treatment",
    baseline="complete_historical_prepared_input",
    optimization="historical_candidate_needs_strict_recertification",
    same_year="latest_stage_ppo_not_run",
    crossyear="latest_stage_ppo_not_started",
    eligibility="tier_b_needs_strict_feasibility",
)

# SY: the only station with current fixed-weight, zero-training PPO cross-year
# evidence, still judged by the older local-primary rule.
add(
    "SY", [2012, 2014, 2015],
    tier="A_authoritative_treatment",
    treatment="original_multitreatment_mzx",
    provenance="current_authoritative_with_approved_runtime_icdat_alignment",
    baseline="complete_current",
    optimization="old_local_primary_evidence_needs_028_strict_recertification",
    latest_algorithm="stage_maskable_ppo",
    same_year="see_year_specific_note",
    crossyear="fixed_sy2014_models_zero_training_complete",
    eligibility="needs_028_strict_feasibility",
    notes="Existing pass counts use auto+expert local primary, not the 028 all-four strict target.",
)
FACTS[("SY", 2012)]["latest_same_year_training_status"] = "not_retrained_frozen_sy2014_models_evaluated"
FACTS[("SY", 2012)]["notes"] += " Local primary 3/3, but all three PPO yields are below recorded yield."
FACTS[("SY", 2014)]["latest_same_year_training_status"] = "three_seeds_trained_old_primary_3_of_3"
FACTS[("SY", 2015)]["latest_same_year_training_status"] = "not_retrained_frozen_sy2014_models_evaluated"
FACTS[("SY", 2015)]["notes"] += " Local primary 2/3."

# YC authoritative treatments and old derived weather-replay candidates.
add(
    "YC", [2008],
    tier="A_authoritative_treatment",
    treatment="original_multitreatment_mzx",
    provenance="documented_not_revalidated_under_028",
    baseline="historical_evidence_needs_four_baseline_recompute",
    optimization="needs_028_feasibility",
    eligibility="needs_four_baselines_and_feasibility",
)
add(
    "YC", [2014],
    tier="A_authoritative_treatment",
    treatment="original_multitreatment_mzx",
    provenance="current_formal_anchor",
    baseline="complete_current",
    optimization="historical_candidate_needs_strict_recertification",
    latest_algorithm="stage_maskable_ppo",
    same_year="three_seeds_run_old_primary_2_of_3",
    crossyear="not_started",
    eligibility="needs_028_strict_feasibility",
    notes="Two seeds win PFP_N among positive-N baselines, but yield and WP_ET do not strictly exceed the all-four maxima.",
)
add(
    "YC", [2006, 2009, 2015, 2018],
    tier="B_derived_weather_replay",
    treatment="weather_shifted_replay_from_old_dqn_workflow",
    provenance="derived_not_authoritative_treatment",
    baseline="partial_historical_not_028_complete",
    optimization="old_dqn_candidate_needs_strict_recertification",
    latest_algorithm="historical_dqn_provisional",
    same_year="latest_stage_ppo_not_run",
    crossyear="latest_stage_ppo_not_run",
    eligibility="tier_b_needs_four_baselines_and_feasibility",
)

# FQ original treatments and derived years selected by the old all-year screen.
add(
    "FQ", [2007, 2008, 2010],
    tier="A_authoritative_treatment",
    treatment="original_multitreatment_mzx",
    provenance="documented_original_treatment_needs_028_revalidation",
    baseline="partial_historical_not_028_complete",
    optimization="year_sensitive_2007_known_low_space_others_need_028",
    eligibility="needs_four_baselines_and_feasibility",
    notes="2007/2008 were made runnable after SDATE/ICDAT repairs; original files must remain unchanged.",
)
add(
    "FQ", [2013, 2014, 2019, 2020, 2023],
    tier="B_derived_weather_replay",
    treatment="fq2008_template_shifted_to_target_weather",
    provenance="derived_candidate_from_014_01",
    baseline="partial_historical_not_028_complete",
    optimization="old_screen_candidate_needs_strict_recertification",
    latest_algorithm="historical_dqn_provisional",
    same_year="latest_stage_ppo_not_run",
    crossyear="latest_stage_ppo_not_run",
    eligibility="tier_b_needs_four_baselines_and_feasibility",
)
add(
    "FQ", [2016],
    tier="B_derived_weather_replay",
    treatment="fq2008_template_shifted_to_2016_weather",
    provenance="current_formal_derived_anchor",
    baseline="complete_current",
    optimization="old_candidate_current_stage_ppo_failed_old_primary",
    latest_algorithm="stage_maskable_ppo",
    same_year="seed0_run_old_primary_failed_then_stopped",
    crossyear="not_started",
    strict_status="not_passed_by_current_candidate_and_not_evaluated_as_full_search",
    eligibility="tier_b_needs_028_strict_feasibility_before_more_rl",
)

# LC original treatments.  Only 2010 currently has clear optimization-space
# evidence; the other years remain useful low-space/negative controls.
add(
    "LC", [2008, 2009, 2011],
    tier="A_authoritative_treatment",
    treatment="original_multitreatment_mzx_with_runtime_adapter",
    provenance="documented_soil_and_ic_adapter_required",
    baseline="partial_historical_not_028_complete",
    optimization="historical_low_or_no_management_gain",
    eligibility="needs_four_baselines_and_feasibility",
)
add(
    "LC", [2010],
    tier="A_authoritative_treatment",
    treatment="original_multitreatment_mzx_with_runtime_adapter",
    provenance="current_formal_anchor_with_adapter",
    baseline="complete_current",
    optimization="clear_historical_candidate_needs_strict_recertification",
    latest_algorithm="stage_maskable_ppo",
    same_year="seed0_run_old_primary_failed_then_stopped",
    crossyear="not_started",
    strict_status="not_passed_by_current_candidate_and_not_evaluated_as_full_search",
    eligibility="needs_028_strict_feasibility_before_more_rl",
    notes="Current seed0 PFP_N exceeds positive-N baselines, but WP_ET is 0.07 below the all-four maximum.",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def year_from_wth(name: str, station_code: str) -> int | None:
    match = re.fullmatch(rf"CN{station_code}(\d{{2}})[A-Z0-9]{{2}}\.WTH", name, flags=re.IGNORECASE)
    if not match:
        return None
    yy = int(match.group(1))
    return 1900 + yy if yy >= 80 else 2000 + yy


def discover_weather() -> dict[tuple[str, int], list[str]]:
    found: dict[tuple[str, int], list[str]] = {}
    for site, dirname in SITE_DIR.items():
        folder = ROOT / INPUT_BASE / dirname
        for path in sorted(folder.glob("*.WTH")):
            year = year_from_wth(path.name, SITE_CODE[site])
            if year is None:
                continue
            found.setdefault((site, year), []).append(path.relative_to(ROOT).as_posix())
    return found


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    weather = discover_weather()
    keys = sorted(set(weather) | set(FACTS), key=lambda x: (x[0], x[1]))

    rows: list[dict[str, Any]] = []
    for site, year in keys:
        weather_paths = weather.get((site, year), [])
        base = fact(
            tier="C_weather_only",
            treatment="no_registered_treatment_for_this_year",
            provenance="wth_only_not_authoritative",
            baseline="not_run_as_complete_four_baselines",
            optimization="not_assessed",
            eligibility="excluded_until_authoritative_treatment_and_ic_exist",
            notes="WTH presence alone does not make a formal site-year.",
        )
        base.update(FACTS.get((site, year), {}))
        sources = [*weather_paths, *MZX_SOURCE.get(site, []), *COMMON_SOURCES.get(site, [])]
        sources = list(dict.fromkeys(sources))
        rows.append(
            {
                "site": site,
                "year": year,
                **base,
                "weather_status": "present" if weather_paths else "missing_from_current_input_directory",
                "source_paths": ";".join(sources),
            }
        )

    registry_fields = [
        "site", "year", "evidence_tier", "treatment_status", "weather_status",
        "input_provenance_status", "four_baseline_status", "strict_feasibility_status",
        "optimization_space_status", "latest_rl_algorithm", "latest_same_year_training_status",
        "latest_fixed_weight_crossyear_status", "strict_all_four_three_metric_status",
        "next_phase_eligibility", "source_paths", "notes",
    ]
    write_csv(OUT / "028_01_station_year_registry.csv", rows, registry_fields)

    cross_rows = [
        {
            "site": "SY", "training_anchor": 2014, "target_years": "2012;2014;2015",
            "latest_algorithm": "stage_maskable_ppo", "fixed_weight_zero_training": True,
            "old_local_primary_pass": "2012=3/3;2014=3/3;2015=2/3",
            "strict_028_all_four_three_metric": "not_evaluated;SY2012 yield fails recorded for 3/3",
            "status": "completed_only_under_old_local_primary",
            "source": "benchmark_results/026_07_attempt2/026_07_result.json",
        },
        {
            "site": "HLA", "training_anchor": 2010, "target_years": "not_started",
            "latest_algorithm": "stage_maskable_ppo", "fixed_weight_zero_training": False,
            "old_local_primary_pass": "training anchor 1/3",
            "strict_028_all_four_three_metric": "not_evaluated",
            "status": "stopped_before_crossyear",
            "source": "benchmark_results/027_03/027_03_result.json",
        },
        {
            "site": "YC", "training_anchor": 2014, "target_years": "not_started",
            "latest_algorithm": "stage_maskable_ppo", "fixed_weight_zero_training": False,
            "old_local_primary_pass": "training anchor 2/3",
            "strict_028_all_four_three_metric": "not_evaluated",
            "status": "cross_year_started=false",
            "source": "benchmark_results/027_07_site_specific_stage_maskable_ppo_attempt2/YC/site_result.json",
        },
        {
            "site": "FQ", "training_anchor": 2016, "target_years": "not_started",
            "latest_algorithm": "stage_maskable_ppo", "fixed_weight_zero_training": False,
            "old_local_primary_pass": "training anchor seed0 failed",
            "strict_028_all_four_three_metric": "not_evaluated",
            "status": "stopped_before_crossyear",
            "source": "benchmark_results/027_07_site_specific_stage_maskable_ppo_attempt2/FQ/site_result.json",
        },
        {
            "site": "LC", "training_anchor": 2010, "target_years": "not_started",
            "latest_algorithm": "stage_maskable_ppo", "fixed_weight_zero_training": False,
            "old_local_primary_pass": "training anchor seed0 failed",
            "strict_028_all_four_three_metric": "not_evaluated",
            "status": "stopped_before_crossyear",
            "source": "benchmark_results/027_07_site_specific_stage_maskable_ppo_attempt2/LC/site_result.json",
        },
    ]
    cross_fields = list(cross_rows[0])
    write_csv(OUT / "028_01_current_crossyear_evidence.csv", cross_rows, cross_fields)

    hash_rows: list[dict[str, Any]] = []
    missing_sources: list[str] = []
    for rel in HASH_SOURCES:
        path = ROOT / rel
        if path.is_file():
            hash_rows.append({"path": rel, "exists": True, "bytes": path.stat().st_size, "sha256": sha256(path)})
        else:
            hash_rows.append({"path": rel, "exists": False, "bytes": "", "sha256": ""})
            missing_sources.append(rel)
    write_csv(OUT / "028_01_source_file_hashes.csv", hash_rows, ["path", "exists", "bytes", "sha256"])

    tier_counts = Counter(row["evidence_tier"] for row in rows)
    certified = [row for row in rows if row["strict_feasibility_status"] == "certified"]
    result = {
        "status": "completed" if not missing_sources else "partial",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_calls": 0,
        "dssat_calls": 0,
        "registry_row_count": len(rows),
        "tier_counts": dict(sorted(tier_counts.items())),
        "strict_feasibility_certified_count_under_028": len(certified),
        "latest_stage_ppo_fixed_weight_crossyear_sites": ["SY"],
        "all_five_sites_latest_ppo_crossyear_complete": False,
        "unified_strict_all_four_three_metric_config_exists": False,
        "next_training_allowed": False,
        "next_required_task": "028_02 four-baseline completion and deterministic strict-feasibility certification",
        "missing_hashed_sources": missing_sources,
        "important_scope_corrections": [
            "HLA 2007/2010/2015/2016/2022 is a formal prepared-adapter set, not five original multi-treatment years.",
            "FQ2016 is a derived FQ2008-template weather replay, not an original FQ treatment year.",
            "Only SY has current stage-PPO fixed-weight cross-year evaluation, and it used the older auto+expert local-primary rule.",
            "No site-year has yet been certified under the new all-four, three-metric 028 strict definition.",
        ],
    }
    (OUT / "028_01_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    tier_lines = "\n".join(f"- `{key}`: {value}" for key, value in sorted(tier_counts.items()))
    missing_text = "无" if not missing_sources else "、".join(f"`{p}`" for p in missing_sources)
    doc_text = f"""# 028_01 五站点全部年份范围与可行性证据注册表审计记录

## 结论先行

本任务完成了零训练、零 DSSAT 的站点—年份证据注册。当前**只有 SY** 做过最新阶段型 MaskablePPO 的固定权重同站跨年验证；HLA、YC、FQ、LC 均未完成。现有 SY 结果采用旧的 auto+expert local-primary 口径，不能当作已经同时超过 null、recorded、auto、official expert 的产量、WP_ET 与 PFP_N。

当前尚无任何站点—年份按 028 新定义完成“同阶段、同动作、同预算、同时严格超过四基线三指标”的可行性认证。因此本任务不允许直接进入全量调参或训练；下一步必须先完成 028_02 四基线复算和 deterministic strict-feasibility certification。

## 执行边界

- RL `learn()` 调用：0；
- DSSAT 调用：0；
- 原始输入修改：0；
- 旧结果覆盖：0；
- 只扫描 WTH 文件名、读取已有记录并哈希关键证据。

## 注册表规模

- 总行数：{len(rows)}；
{tier_lines}
- 028 严格可行性已认证：{len(certified)}；
- 关键哈希源缺失：{missing_text}。

Tier A 是原始/正式 treatment 或正式单年锚点；Tier B 是明确派生天气回放；Tier C 只有 WTH，不得因为天气文件存在就视为正式训练年份。

## 当前最新 PPO 跨年证据

| 站点 | 训练锚点 | 最新 PPO 同年状态 | 固定权重同站跨年 | 028 严格结论 |
|---|---:|---|---|---|
| SY | 2014 | 三 seed | 已完成 2012/2014/2015，旧 local-primary 为 3/3、3/3、2/3 | 未按四基线三指标严格认证；2012 三模型产量均低于 recorded |
| HLA | 2010 | 旧 primary 1/3 | 未启动 | 未认证 |
| YC | 2014 | 旧 primary 2/3 | 未启动 | 未认证 |
| FQ | 2016 | seed0 旧 primary 失败后停止 | 未启动 | 未认证；2016 为派生天气回放 |
| LC | 2010 | seed0 旧 primary 失败后停止 | 未启动 | 未认证；仅 PFP_N 有单项优势 |

## 重要 provenance 纠正

1. HLA 原始多 treatment MZX 的年份为 2007、2009、2011；HLA2010 是另一个正式单年锚点。后续正式 prepared-adapter 集中的 2015、2016、2022，以及所用 2007 变体，不能统称为原始权威 treatment。
2. FQ 原始 treatment 为 2007、2008、2010；FQ2016 及 2013、2014、2019、2020、2023 是由 FQ2008 风格模板结合目标年天气生成的派生候选。
3. YC 原始 treatment 为 2008、2014；2006、2009、2015、2018 是旧 DQN 天气回放证据。
4. LC 原始 treatment 为 2008、2009、2010、2011，但运行依赖 soil ID/IC 日期 adapter；只有 2010 目前有明确优化空间证据。
5. SY 原始正式 treatment 为 2012、2014、2015；其余 WTH-only 年份不进入正式结论。

## 输出文件

- `benchmark_results/028_01_all_site_year_scope_registry/028_01_station_year_registry.csv`
- `benchmark_results/028_01_all_site_year_scope_registry/028_01_current_crossyear_evidence.csv`
- `benchmark_results/028_01_all_site_year_scope_registry/028_01_source_file_hashes.csv`
- `benchmark_results/028_01_all_site_year_scope_registry/028_01_result.json`

## 下一步

执行 028_02：先对 Tier A 逐年补齐/复算四基线，再用与未来 RL 完全相同的阶段、动作、预算和 mask 做 deterministic strict-feasibility certification。Tier B 单列执行，不能与 Tier A 合并宣称。只有通过认证的年份才进入统一 reward 单测和小型全局调参。
"""
    DOC.write_text(doc_text, encoding="utf-8")

    print("028_01 All-site-year scope registry audit completed")
    print(f"status={result['status']}")
    print(f"registry_rows={len(rows)}")
    print(f"tier_counts={dict(sorted(tier_counts.items()))}")
    print(f"strict_feasibility_certified_under_028={len(certified)}")
    print("latest_stage_ppo_fixed_weight_crossyear_sites=SY")
    print("training_calls=0")
    print("dssat_calls=0")
    print("next_training_allowed=false")


if __name__ == "__main__":
    main()
