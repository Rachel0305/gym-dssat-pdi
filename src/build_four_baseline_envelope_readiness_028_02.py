#!/usr/bin/env python3
"""Recompute current four-baseline envelopes for the 028 strict RL target.

This is a read-only evidence transformation: no DSSAT and no RL training.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "028_02_four_baseline_envelope_readiness"
DOC = ROOT / "docs" / "2026-07-18_028_02_four_baseline_envelope_and_strict_feasibility_readiness.md"
REGISTRY = ROOT / "benchmark_results" / "028_01_all_site_year_scope_registry" / "028_01_station_year_registry.csv"

BASELINE_SOURCES = [
    ("HLA", 2010, "A_authoritative_treatment", "benchmark_results/027_01_attempt2/027_01_hla2010_four_baselines.csv"),
    ("SY", 2012, "A_authoritative_treatment", "benchmark_results/026_07_attempt2/026_07_sy2012_four_baselines.csv"),
    ("SY", 2014, "A_authoritative_treatment", "benchmark_results/026_07_attempt2/026_07_sy2014_four_baselines.csv"),
    ("SY", 2015, "A_authoritative_treatment", "benchmark_results/026_07_attempt2/026_07_sy2015_four_baselines.csv"),
    ("YC", 2014, "A_authoritative_treatment", "benchmark_results/027_07_site_specific_stage_maskable_ppo_attempt2/YC/readiness/four_baseline_fresh_rerun.csv"),
    ("FQ", 2016, "B_derived_weather_replay", "benchmark_results/027_07_site_specific_stage_maskable_ppo_attempt2/FQ/readiness/four_baseline_fresh_rerun.csv"),
    ("LC", 2010, "A_authoritative_treatment", "benchmark_results/027_07_site_specific_stage_maskable_ppo_attempt2/LC/readiness/four_baseline_fresh_rerun.csv"),
]

RL_SOURCES = [
    ("HLA", 2010, "benchmark_results/027_03/027_03_hla2010_three_seed_selected_summary.csv", "hla"),
    ("SY", None, "benchmark_results/026_07_attempt2/026_07_sy_all_years_frozen_ppo_summary.csv", "sy"),
    ("YC", 2014, "benchmark_results/027_07_site_specific_stage_maskable_ppo_attempt2/YC/selected_seed_summary.csv", "three_site"),
    ("FQ", 2016, "benchmark_results/027_07_site_specific_stage_maskable_ppo_attempt2/FQ/selected_seed_summary.csv", "three_site"),
    ("LC", 2010, "benchmark_results/027_07_site_specific_stage_maskable_ppo_attempt2/LC/selected_seed_summary.csv", "three_site"),
]

SCENARIO_MAP = {
    "null": "null",
    "recorded": "recorded",
    "recorded_farmer": "recorded",
    "dssat_auto": "auto",
    "extension_expert": "expert",
    "official_extension_expert": "expert",
}

DELTA_Y = 1.0
DELTA_WP = 0.01
DELTA_PFP = 0.1


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float:
    if value is None or str(value).strip() == "":
        return math.nan
    return float(value)


def first_float(row: dict[str, str], *names: str) -> float:
    for name in names:
        if name in row and str(row[name]).strip() != "":
            return float(row[name])
    return math.nan


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_baselines() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    envelopes: list[dict[str, Any]] = []
    for site, expected_year, tier, rel in BASELINE_SOURCES:
        path = ROOT / rel
        raw = read_csv(path)
        standardized: list[dict[str, Any]] = []
        errors: list[str] = []
        for row in raw:
            scenario_raw = row.get("scenario", "")
            scenario = SCENARIO_MAP.get(scenario_raw)
            if scenario is None:
                errors.append(f"unknown_scenario:{scenario_raw}")
                continue
            year = int(float(row.get("year", expected_year)))
            if year != expected_year:
                errors.append(f"year_mismatch:{year}")
            grain = first_float(row, "final_gwad", "final_yield")
            etcp = first_float(row, "etcp_mm")
            actual_n = first_float(row, "summary_nitrogen_total")
            actual_i = first_float(row, "summary_irrigation_total")
            wp = grain / (10.0 * etcp) if math.isfinite(grain) and math.isfinite(etcp) and etcp > 0 else math.nan
            pfp = grain / actual_n if math.isfinite(grain) and math.isfinite(actual_n) and actual_n > 0 else math.nan
            reported_wp = first_float(row, "WP_ET_kg_m3")
            reported_pfp = first_float(row, "PFP_N_kg_kg")
            entry = {
                "site": site, "year": expected_year, "evidence_tier": tier,
                "scenario": scenario, "scenario_raw": scenario_raw,
                "grain_yield_kg_ha": grain, "etcp_mm": etcp,
                "actual_irrigation_mm": actual_i, "actual_nitrogen_kg_ha": actual_n,
                "WP_ET_recomputed_kg_m3": wp, "PFP_N_recomputed_kg_kg": pfp,
                "WP_ET_reported": reported_wp, "PFP_N_reported": reported_pfp,
                "WP_report_abs_error": abs(wp - reported_wp) if math.isfinite(wp) and math.isfinite(reported_wp) else math.nan,
                "PFP_report_abs_error": abs(pfp - reported_pfp) if math.isfinite(pfp) and math.isfinite(reported_pfp) else math.nan,
                "source_path": rel,
            }
            standardized.append(entry)
            metrics.append(entry)

        scenarios = [row["scenario"] for row in standardized]
        if sorted(scenarios) != ["auto", "expert", "null", "recorded"]:
            errors.append(f"scenario_set:{sorted(scenarios)}")
        if len(scenarios) != len(set(scenarios)):
            errors.append("duplicate_canonical_scenario")
        if not all(math.isfinite(row["grain_yield_kg_ha"]) and row["grain_yield_kg_ha"] >= 0 for row in standardized):
            errors.append("invalid_yield")
        if not all(math.isfinite(row["WP_ET_recomputed_kg_m3"]) for row in standardized):
            errors.append("invalid_wp")
        positive_n = [row for row in standardized if math.isfinite(row["actual_nitrogen_kg_ha"]) and row["actual_nitrogen_kg_ha"] > 0]
        if not positive_n:
            errors.append("no_positive_n_baseline")

        valid = not errors
        if valid:
            y_winner = max(standardized, key=lambda row: row["grain_yield_kg_ha"])
            wp_winner = max(standardized, key=lambda row: row["WP_ET_recomputed_kg_m3"])
            pfp_winner = max(positive_n, key=lambda row: row["PFP_N_recomputed_kg_kg"])
            y_target = y_winner["grain_yield_kg_ha"]
            wp_target = wp_winner["WP_ET_recomputed_kg_m3"]
            pfp_target = pfp_winner["PFP_N_recomputed_kg_kg"]
        else:
            y_winner = wp_winner = pfp_winner = {"scenario": ""}
            y_target = wp_target = pfp_target = math.nan
        envelopes.append(
            {
                "site": site, "year": expected_year, "evidence_tier": tier,
                "baseline_envelope_valid": valid, "invalid_reasons": ";".join(errors),
                "Y_target_kg_ha": y_target, "Y_target_scenario": y_winner["scenario"],
                "WP_target_kg_m3": wp_target, "WP_target_scenario": wp_winner["scenario"],
                "PFP_target_kg_kg": pfp_target, "PFP_target_scenario": pfp_winner["scenario"],
                "strict_Y_min_kg_ha": y_target + DELTA_Y if valid else math.nan,
                "strict_WP_min_kg_m3": wp_target + DELTA_WP if valid else math.nan,
                "strict_PFP_min_kg_kg": pfp_target + DELTA_PFP if valid else math.nan,
                "source_path": rel,
            }
        )
    return metrics, envelopes


def selected_rl_rows(envelopes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets = {(row["site"], int(row["year"])): row for row in envelopes if row["baseline_envelope_valid"]}
    result: list[dict[str, Any]] = []
    for site, fixed_year, rel, schema in RL_SOURCES:
        for row in read_csv(ROOT / rel):
            year = int(float(row["year"])) if schema == "sy" else int(fixed_year)
            target = targets.get((site, year))
            if target is None:
                continue
            seed = int(float(row["seed"]))
            if schema == "hla":
                grain = as_float(row["selected_yield"])
                irrigation = as_float(row["selected_irrigation"])
                nitrogen = as_float(row["selected_nitrogen"])
                wp = as_float(row["selected_WP_ET"])
                pfp = as_float(row["selected_PFP_N"])
                checkpoint = row["selected_checkpoint"]
                model_hash = row["selected_model_sha256"]
            elif schema == "sy":
                grain = as_float(row["final_gwad"])
                irrigation = as_float(row["summary_irrigation_total"])
                nitrogen = as_float(row["summary_nitrogen_total"])
                etcp = as_float(row["etcp_mm"])
                wp = grain / (10.0 * etcp) if etcp > 0 else math.nan
                pfp = grain / nitrogen if nitrogen > 0 else math.nan
                checkpoint = "frozen_from_SY2014"
                model_hash = row["model_sha256"]
            else:
                grain = as_float(row["yield"])
                irrigation = as_float(row["irrigation"])
                nitrogen = as_float(row["nitrogen"])
                wp = as_float(row["WP_ET"])
                pfp = as_float(row["PFP_N"])
                checkpoint = row["selected_checkpoint"]
                model_hash = row["model_sha256"]

            y_gap = grain - target["Y_target_kg_ha"]
            wp_gap = wp - target["WP_target_kg_m3"]
            pfp_gap = pfp - target["PFP_target_kg_kg"] if math.isfinite(pfp) else math.nan
            y_pass = math.isfinite(grain) and grain >= target["strict_Y_min_kg_ha"]
            wp_pass = math.isfinite(wp) and wp >= target["strict_WP_min_kg_m3"]
            pfp_pass = nitrogen > 0 and math.isfinite(pfp) and pfp >= target["strict_PFP_min_kg_kg"]
            result.append(
                {
                    "site": site, "year": year, "seed": seed,
                    "checkpoint": checkpoint, "model_sha256": model_hash,
                    "grain_yield_kg_ha": grain, "actual_irrigation_mm": irrigation,
                    "actual_nitrogen_kg_ha": nitrogen, "WP_ET_kg_m3": wp,
                    "PFP_N_kg_kg": pfp, "Y_gap_vs_all_four_max": y_gap,
                    "WP_gap_vs_all_four_max": wp_gap,
                    "PFP_gap_vs_positive_n_all_four_max": pfp_gap,
                    "strict_yield_pass": y_pass, "strict_wp_pass": wp_pass,
                    "strict_pfp_pass": pfp_pass,
                    "strict_joint_pass": y_pass and wp_pass and pfp_pass,
                    "evaluation_type": "fixed_weight_crossyear" if schema == "sy" and year != 2014 else "same_year_selected_checkpoint",
                    "source_path": rel,
                }
            )
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics, envelopes = canonical_baselines()
    selected = selected_rl_rows(envelopes)

    metrics_fields = list(metrics[0])
    envelope_fields = list(envelopes[0])
    selected_fields = list(selected[0])
    write_csv(OUT / "028_02_baseline_metrics_recomputed.csv", metrics, metrics_fields)
    write_csv(OUT / "028_02_strict_target_envelopes.csv", envelopes, envelope_fields)
    write_csv(OUT / "028_02_current_selected_rl_strict_comparison.csv", selected, selected_fields)

    registry_rows = read_csv(REGISTRY)
    ready_a = {(row["site"], str(row["year"])) for row in envelopes if row["evidence_tier"] == "A_authoritative_treatment" and row["baseline_envelope_valid"]}
    missing_a = [
        row for row in registry_rows
        if row["evidence_tier"] == "A_authoritative_treatment" and (row["site"], row["year"]) not in ready_a
    ]
    missing_fields = [
        "site", "year", "evidence_tier", "treatment_status", "weather_status",
        "input_provenance_status", "four_baseline_status", "next_phase_eligibility",
        "source_paths", "notes",
    ]
    write_csv(OUT / "028_02_missing_tier_a_four_baselines.csv", missing_a, missing_fields)

    sources = sorted({str(REGISTRY.relative_to(ROOT)).replace("\\", "/"), *[rel for _, _, _, rel in BASELINE_SOURCES], *[rel for _, _, rel, _ in RL_SOURCES]})
    hash_rows = []
    for rel in sources:
        path = ROOT / rel
        hash_rows.append({"path": rel, "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else "", "sha256": sha256(path) if path.is_file() else ""})
    write_csv(OUT / "028_02_source_hashes.csv", hash_rows, ["path", "exists", "bytes", "sha256"])

    valid_envelopes = [row for row in envelopes if row["baseline_envelope_valid"]]
    strict_passes = [row for row in selected if row["strict_joint_pass"]]
    wp_report_errors = [row["WP_report_abs_error"] for row in metrics if math.isfinite(row["WP_report_abs_error"])]
    pfp_report_errors = [row["PFP_report_abs_error"] for row in metrics if math.isfinite(row["PFP_report_abs_error"])]
    max_wp_report_error = max(wp_report_errors, default=math.nan)
    max_pfp_report_error = max(pfp_report_errors, default=math.nan)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[(row["site"], int(row["year"]))].append(row)
    year_summary = []
    for (site, year), rows in sorted(grouped.items()):
        pass_count = sum(bool(row["strict_joint_pass"]) for row in rows)
        year_summary.append({"site": site, "year": year, "models_evaluated": len(rows), "strict_pass_count": pass_count, "formal_2_of_3_success": len(rows) == 3 and pass_count >= 2})

    result = {
        "status": "completed" if all(row["exists"] for row in hash_rows) else "partial",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_calls": 0,
        "dssat_calls": 0,
        "baseline_source_year_count": len(envelopes),
        "valid_baseline_envelope_count": len(valid_envelopes),
        "valid_tier_a_envelope_count": sum(row["evidence_tier"] == "A_authoritative_treatment" and row["baseline_envelope_valid"] for row in envelopes),
        "valid_tier_b_envelope_count": sum(row["evidence_tier"] == "B_derived_weather_replay" and row["baseline_envelope_valid"] for row in envelopes),
        "missing_tier_a_four_baseline_count": len(missing_a),
        "selected_rl_model_year_count": len(selected),
        "selected_rl_strict_joint_pass_count": len(strict_passes),
        "max_abs_difference_recomputed_vs_reported_WP_ET": max_wp_report_error,
        "max_abs_difference_recomputed_vs_reported_PFP_N": max_pfp_report_error,
        "current_year_summary": year_summary,
        "strict_feasibility_certified_count": 0,
        "next_training_allowed": False,
        "next_required_action": "fresh four-baseline completion for missing Tier A years, then deterministic strict-feasibility search for all valid envelopes",
    }
    (OUT / "028_02_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")

    year_lines = "\n".join(
        f"| {row['site']} | {row['year']} | {row['models_evaluated']} | {row['strict_pass_count']} | {row['formal_2_of_3_success']} |"
        for row in year_summary
    )
    missing_lines = "\n".join(f"- {row['site']}{row['year']}：{row['input_provenance_status']}；旧四基线状态 `{row['four_baseline_status']}`" for row in missing_a)
    doc = f"""# 028_02 四基线包络与严格可行性准备审计记录

## 结论先行

本任务复用了 {len(envelopes)} 个当前已有完整四基线源，并按 DSSAT 实际 Summary 施氮量重新计算 WP_ET、PFP_N 和四情景包络。其中 Tier A 有 {sum(row['evidence_tier'] == 'A_authoritative_treatment' and row['baseline_envelope_valid'] for row in envelopes)} 个，Tier B 有 {sum(row['evidence_tier'] == 'B_derived_weather_replay' and row['baseline_envelope_valid'] for row in envelopes)} 个。Tier A 尚有 {len(missing_a)} 个年份没有当前可直接复用的完整四基线。

对现有 {len(selected)} 个 PPO 模型—年份结果按 028 的四情景三指标严格门槛重判后，严格联合通过为 **{len(strict_passes)}**。这只是当前选中模型的重判，不是可行性搜索；严格通过为 0 也不能说明目标不存在。

## 执行边界

- RL训练：0；
- DSSAT运行：0；
- 旧结果修改：0；
- 指标统一为 `WP_ET=yield/(10×ETCP)`、`PFP_N=yield/Summary实际N`；
- N=0 的 PFP_N 为不可比较。
- 旧表报告值与统一重算值的最大绝对差：WP_ET={max_wp_report_error:.6f} kg/m³，PFP_N={max_pfp_report_error:.6f} kg/kg；因此本任务的门槛和胜负一律使用统一重算值，不使用旧表舍入值。

## 当前 selected PPO 严格重判

| 站点 | 年份 | 模型数 | 严格联合通过数 | 是否达到2/3 |
|---|---:|---:|---:|---|
{year_lines}

## Tier A 待补四基线

{missing_lines if missing_lines else '- 无'}

## 方法边界

1. 当前 selected PPO 的 028 严格通过数不等于 RL 框架最终能力；旧 reward 和选模协议并未针对四情景三指标联合目标设计。
2. 本任务尚未运行 deterministic feasibility，因此严格可优化年份数仍为 0 个“已认证”，不是 0 个“实际存在”。
3. FQ2016 虽有当前完整四基线，但属于 Tier B 派生天气回放，必须单列。

## 输出

- `benchmark_results/028_02_four_baseline_envelope_readiness/028_02_baseline_metrics_recomputed.csv`
- `benchmark_results/028_02_four_baseline_envelope_readiness/028_02_strict_target_envelopes.csv`
- `benchmark_results/028_02_four_baseline_envelope_readiness/028_02_current_selected_rl_strict_comparison.csv`
- `benchmark_results/028_02_four_baseline_envelope_readiness/028_02_missing_tier_a_four_baselines.csv`
- `benchmark_results/028_02_four_baseline_envelope_readiness/028_02_result.json`

## 下一步

先为缺失 Tier A 年份在新目录补齐四基线；随后对全部 valid envelope 年份使用相同阶段、动作、预算和 mask 做 deterministic strict-feasibility 搜索。完成前不启动全局 RL 调参。
"""
    DOC.write_text(doc, encoding="utf-8")

    print("028_02 Four-baseline envelope readiness completed")
    print(f"status={result['status']}")
    print(f"valid_envelopes={len(valid_envelopes)}")
    print(f"missing_tier_a_four_baselines={len(missing_a)}")
    print(f"selected_rl_strict_joint_pass={len(strict_passes)}/{len(selected)}")
    print("training_calls=0")
    print("dssat_calls=0")
    print("next_training_allowed=false")


if __name__ == "__main__":
    main()
