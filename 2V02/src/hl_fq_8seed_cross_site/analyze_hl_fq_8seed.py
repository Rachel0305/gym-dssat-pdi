"""Collect frozen HL/FQ 8-seed baseline outputs and make comparable figures.

This is a read-only postprocessor for completed 100K runs. It does not import
DSSAT, train a policy, rerun evaluation, or alter any configuration.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "results/hl_fq_8seed_cross_site"
FIGURE_ROOT = RESULT_ROOT / "figures"
YEARS = list(range(2014, 2024))
SEEDS = list(range(8))
EXPECTED_I_LEVELS = {0.0, 15.0, 30.0, 45.0}
EXPECTED_N_LEVELS = {0.0, 40.0, 80.0, 120.0}
COMMON_REWARD_FORMULA = "0.158 * yield_kg_ha - 1.1 * irrigation_mm - 1.58 * nitrogen_kg_ha"

SITE_SPECS: dict[str, dict[str, Any]] = {
    "HL": {
        "station_code": "HLA",
        "profile": "lowIC",
        "prefix": "HL",
        "soil_id": "HL99001200",
        "cultivar_id": "HY0006",
        "cultivar_name": "Haiyu No006",
        "train_years": list(range(2004, 2014)),
        "weather_root": "DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/HL",
        "mzx": "DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/HL/CNHL0701_corrected_IC123.MZX",
        "seed0_config": "configs/054_00_hla_lowIC_expanded_action_maskableppo.json",
        "seed0_output": "benchmark_results/054_00_hla_lowIC_expanded_action_maskableppo",
        "seed_config_pattern": "hl_seed{seed}.json",
    },
    "FQ": {
        "station_code": "FQA",
        "profile": "originIC",
        "prefix": "FQ",
        "soil_id": "FQ99001200",
        "cultivar_id": "FQ0985",
        "cultivar_name": "Zhengdan No985",
        "train_years": list(range(2005, 2014)),
        "weather_root": "DSSAT_auto_validation/multisite_new_cultivar_inputs_013/FQ",
        "mzx": "DSSAT_auto_validation/multisite_new_cultivar_inputs_013/FQ/CNFQ0801.MZX",
        "seed0_config": "configs/051_00_fqa_originIC_expanded_action_maskableppo.json",
        "seed0_output": "benchmark_results/051_00_fqa_originIC_expanded_action_maskableppo",
        "seed_config_pattern": "fq_seed{seed}.json",
    },
}

COLORS = {
    "policy": "#2A9D55",
    "irrigation": "#2878B5",
    "nitrogen": "#D9822B",
    "muted": "#666666",
    "yield": "#2A9D55",
    "common_reward": "#7E63B6",
    "pfp_n": "#C44E52",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: None if isinstance(value, float) and math.isnan(value) else value
                for key, value in row.items()
            })


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    def clean(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value

    path.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def number(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_signature(payload: Any) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def config_and_output(site: str, seed: int) -> tuple[Path, Path]:
    spec = SITE_SPECS[site]
    if seed == 0:
        config_path = ROOT / spec["seed0_config"]
        output_root = ROOT / spec["seed0_output"]
    else:
        config_path = ROOT / "configs/hl_fq_8seed_cross_site" / spec["seed_config_pattern"].format(seed=seed)
        config = read_json(config_path)
        output_root = ROOT / "benchmark_results" / f"{config['task_id']}_{config['task_name']}"
    return config_path, output_root


def daily_output_path(output_root: Path, station_code: str, year: int, seed: int) -> Path:
    folder = output_root / "daily_outputs" / station_code
    matches = sorted(folder.glob(f"{station_code}_{year}_seed{seed}_ckpt100000_daily.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one 100K daily file for {station_code} seed {seed} year {year}; found {len(matches)} under {folder}"
        )
    return matches[0]


def selected_summary(output_root: Path, station_code: str, seed: int) -> dict[int, dict[str, str]]:
    summary_path = output_root / "evaluation/032_22_checkpoint_validation_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    chosen: dict[int, dict[str, str]] = {}
    for row in read_csv(summary_path):
        if int(number(row.get("checkpoint_step"), -1)) != 100000:
            continue
        if str(row.get("station_code", "")).upper() != station_code:
            continue
        if int(number(row.get("seed"), -1)) != seed:
            continue
        year = int(number(row.get("year"), -1))
        if year in YEARS:
            if year in chosen:
                raise ValueError(f"Duplicate selected-checkpoint summary row: {station_code} seed {seed} year {year}")
            if row.get("run_status", "ok") != "ok":
                raise ValueError(f"Evaluation status not ok: {station_code} seed {seed} year {year}: {row.get('run_status')}")
            chosen[year] = row
    if sorted(chosen) != YEARS:
        raise ValueError(f"100K summary coverage incomplete for {station_code} seed {seed}: {sorted(chosen)}")
    return chosen


def rounded_action(value: Any) -> float:
    result = number(value, 0.0)
    result = round(result, 3)
    return 0.0 if abs(result) < 1e-8 else result


def extract_year(
    site: str,
    seed: int,
    year: int,
    summary_row: dict[str, str],
    daily_path: Path,
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    spec = SITE_SPECS[site]
    daily = read_csv(daily_path)
    if not daily:
        raise ValueError(f"Empty daily output: {daily_path}")
    events: list[dict[str, Any]] = []
    full_actions: list[list[Any]] = []
    for daily_row_index, row in enumerate(daily):
        dap = int(number(row.get("dap"), -1))
        i_mm = rounded_action(row.get("safe_action_amir", row.get("raw_action_amir", 0)))
        n_kg = rounded_action(row.get("safe_action_anfer", row.get("raw_action_anfer", 0)))
        action_index = int(number(row.get("discrete_action_index"), -1))
        full_actions.append([row.get("date", ""), dap, action_index, i_mm, n_kg])
        if i_mm > 0 or n_kg > 0:
            events.append({
                "site": site,
                "station_code": spec["station_code"],
                "seed": seed,
                "year": year,
                "daily_row_index": daily_row_index,
                "date": row.get("date", ""),
                "dap": dap,
                "irrigation_mm": i_mm,
                "nitrogen_kg_ha": n_kg,
                "discrete_action_index": action_index,
            })

    total_i = sum(float(event["irrigation_mm"]) for event in events)
    total_n = sum(float(event["nitrogen_kg_ha"]) for event in events)
    irrigation_count = sum(float(event["irrigation_mm"]) > 0 for event in events)
    nitrogen_count = sum(float(event["nitrogen_kg_ha"]) > 0 for event in events)
    summary_i = number(summary_row.get("total_irrigation"))
    summary_n = number(summary_row.get("total_n"))
    if abs(total_i - summary_i) > 0.1 or abs(total_n - summary_n) > 0.1:
        raise ValueError(
            f"Action trace does not close to episode summary for {site} seed {seed} year {year}: "
            f"I={total_i}/{summary_i}, N={total_n}/{summary_n}"
        )
    if irrigation_count != int(number(summary_row.get("irrigation_event_count"), -1)):
        raise ValueError(f"I event count mismatch for {site} seed {seed} year {year}")
    if nitrogen_count != int(number(summary_row.get("n_event_count"), -1)):
        raise ValueError(f"N event count mismatch for {site} seed {seed} year {year}")

    # The score follows the selected 100K validation row. Yield is compared
    # against the terminal daily record as a separate closure check.
    yield_kg = number(summary_row.get("final_grnwt"))
    terminal_yield = number(daily[-1].get("grnwt"))
    yield_close = math.isfinite(terminal_yield) and abs(yield_kg - terminal_yield) <= 0.5
    if not yield_close:
        raise ValueError(
            f"Terminal grain weight mismatch for {site} seed {seed} year {year}: {yield_kg} vs {terminal_yield}"
        )

    event_signature = [[int(e["dap"]), float(e["irrigation_mm"]), float(e["nitrogen_kg_ha"])] for e in events]
    full_sig = [[int(row[1]), int(row[2]), float(row[3]), float(row[4])] for row in full_actions]
    common_reward = 0.158 * yield_kg - 1.1 * total_i - 1.58 * total_n
    pfp_n = number(summary_row.get("PFP_N"), math.nan)
    if total_n <= 0:
        pfp_n = math.nan

    metric = {
        "site": site,
        "station_code": spec["station_code"],
        "seed": seed,
        "year": year,
        "checkpoint_step": 100000,
        "grain_yield_kg_ha": yield_kg,
        "total_irrigation_mm": total_i,
        "total_n_kg_ha": total_n,
        "PFP_N_kg_grain_per_kg_N": pfp_n,
        "NUE": None,
        "WUE": None,
        "WP_ET_kg_m3": None,
        "NUE_status": "N/A: exact plant N uptake is not present in selected validation artifacts",
        "WUE_status": "N/A: exact Summary.OUT/ETCP replay is not present; no inference from irrigation",
        "canonical_reward_return": number(summary_row.get("reward_stress_aware_sum")),
        "common_reward_formula": COMMON_REWARD_FORMULA,
        "common_reward": common_reward,
        "irrigation_event_count": irrigation_count,
        "nitrogen_event_count": nitrogen_count,
        "management_event_days": len(events),
        "validation_episode_days": int(number(summary_row.get("episode_length"), len(daily))),
        "summary_model_path": summary_row.get("model_path", ""),
        "summary_daily_path": summary_row.get("daily_csv_path", ""),
        "daily_output_path": daily_path.relative_to(ROOT).as_posix(),
        "terminal_yield_closure": yield_close,
        "water_n_closure": abs(total_i - summary_i) <= 0.1 and abs(total_n - summary_n) <= 0.1,
        "checkpoint_sha256": "",
        "canonical_config": "",
    }
    mask_flags = [row.get("i240_staged_irrigation_reserve_enabled", "") for row in daily]
    late_flags = [row.get("late_irrigation_reserve_mask_enabled", "") for row in daily]
    swfac_flags = [row.get("swfac_guardrail_enabled", "") for row in daily]
    swfac_thresholds = {round(number(row.get("swfac_guardrail_threshold"), math.nan), 6) for row in daily}
    sorted_event_daps = sorted({int(event["dap"]) for event in events})
    chronological_daps = [int(number(row.get("dap"), -1)) for row in daily]
    dap_reset_count = sum(current < previous for previous, current in zip(chronological_daps, chronological_daps[1:]))
    irrigation_daps = sorted({int(event["dap"]) for event in events if float(event["irrigation_mm"]) > 0})
    nitrogen_daps = sorted({int(event["dap"]) for event in events if float(event["nitrogen_kg_ha"]) > 0})
    gaps = [b - a for a, b in zip(sorted_event_daps, sorted_event_daps[1:])]
    irrigation_gaps = [b - a for a, b in zip(irrigation_daps, irrigation_daps[1:])]
    nitrogen_gaps = [b - a for a, b in zip(nitrogen_daps, nitrogen_daps[1:])]
    cumulative_fields = {
        "max_cumulative_irrigation_mm": max(number(row.get("season_cumulative_irrigation"), 0.0) for row in daily),
        "max_cumulative_n_kg_ha": max(number(row.get("season_cumulative_n"), 0.0) for row in daily),
        "max_cumulative_i_by_dap30_mm": max((number(row.get("season_cumulative_irrigation"), 0.0) for row in daily if int(number(row.get("dap"), -1)) <= 30), default=0.0),
        "max_cumulative_i_by_dap60_mm": max((number(row.get("season_cumulative_irrigation"), 0.0) for row in daily if int(number(row.get("dap"), -1)) <= 60), default=0.0),
        "max_cumulative_i_by_dap90_mm": max((number(row.get("season_cumulative_irrigation"), 0.0) for row in daily if int(number(row.get("dap"), -1)) <= 90), default=0.0),
        "minimum_any_operation_day_gap": min(gaps) if gaps else math.nan,
        "minimum_irrigation_interval_days": min(irrigation_gaps) if irrigation_gaps else math.nan,
        "minimum_fertilization_interval_days": min(nitrogen_gaps) if nitrogen_gaps else math.nan,
        "recorded_dap_reset_count": dap_reset_count,
        "action_grid_valid": all(float(e["irrigation_mm"]) in EXPECTED_I_LEVELS and float(e["nitrogen_kg_ha"]) in EXPECTED_N_LEVELS for e in events),
        "staged_mask_flag_consistent_true": bool(mask_flags) and all(str(v).lower() == "true" for v in mask_flags),
        "late_reserve_mask_flag_consistent_true": bool(late_flags) and all(str(v).lower() == "true" for v in late_flags),
        "swfac_guardrail_flag_consistent_true": bool(swfac_flags) and all(str(v).lower() == "true" for v in swfac_flags),
        "swfac_guardrail_thresholds": ";".join("N/A" if not math.isfinite(v) else f"{v:g}" for v in sorted(swfac_thresholds)),
    }
    mask_fields = {
        "irrigation_cap_dap30_pass": cumulative_fields["max_cumulative_i_by_dap30_mm"] <= 75.0 + 1e-6,
        "irrigation_cap_dap60_pass": cumulative_fields["max_cumulative_i_by_dap60_mm"] <= 150.0 + 1e-6,
        "irrigation_cap_dap90_pass": cumulative_fields["max_cumulative_i_by_dap90_mm"] <= 195.0 + 1e-6,
        "irrigation_season_cap_pass": cumulative_fields["max_cumulative_irrigation_mm"] <= 240.0 + 1e-6,
        "nitrogen_season_cap_pass": cumulative_fields["max_cumulative_n_kg_ha"] <= 250.0 + 1e-6,
        "minimum_irrigation_interval_pass": not irrigation_gaps or min(irrigation_gaps) >= 7,
        "minimum_fertilization_interval_pass": not nitrogen_gaps or min(nitrogen_gaps) >= 7,
    }
    mask_fields["minimum_management_interval_pass"] = (
        mask_fields["minimum_irrigation_interval_pass"] and mask_fields["minimum_fertilization_interval_pass"]
    )
    mask_audit = {
        "site": site, "station_code": spec["station_code"], "seed": seed, "year": year,
        **cumulative_fields, **mask_fields,
        "all_observed_mask_checks_pass": all(mask_fields.values()) and cumulative_fields["action_grid_valid"]
        and cumulative_fields["staged_mask_flag_consistent_true"] and cumulative_fields["late_reserve_mask_flag_consistent_true"]
        and cumulative_fields["swfac_guardrail_flag_consistent_true"] and swfac_thresholds == {0.05},
        "daily_output_path": daily_path.relative_to(ROOT).as_posix(),
    }
    signature = {
        "site": site,
        "station_code": spec["station_code"],
        "seed": seed,
        "year": year,
        "event_sequence": event_signature,
        "event_sequence_sha256": json_signature(event_signature),
        "full_daily_action_sequence": full_sig,
        "full_daily_action_sequence_sha256": json_signature(full_sig),
        "management_action_sequence_text": "; ".join(
            f"DAP{e['dap']} I{e['irrigation_mm']:g}/N{e['nitrogen_kg_ha']:g}" for e in events
        ) or "无正向水氮管理事件",
        "event_count": len(events),
        "daily_rows": len(daily),
    }
    return metric, events, {"sequence": signature, "mask_audit": mask_audit, "full_actions": full_actions}


def classify_years(signatures: list[str]) -> tuple[str, int, int]:
    counts = Counter(signatures)
    mode_count = max(counts.values(), default=0)
    if len(counts) <= 1:
        return "IDENTICAL_ACROSS_YEARS", mode_count, len(counts)
    if mode_count >= 7:
        return "MOSTLY_STABLE_WITH_VARIATION", mode_count, len(counts)
    return "YEAR_SPECIFIC_POLICY", mode_count, len(counts)


def jaccard(left: set[tuple[Any, ...]], right: set[tuple[Any, ...]]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def station_classification(
    site: str,
    by_seed_year: dict[tuple[str, int, int], list[list[Any]]],
    expected_seed_count: int = 8,
) -> dict[str, Any]:
    seed_signatures: dict[int, tuple[str, ...]] = {}
    available_seeds = sorted({seed for (entry_site, seed, _year) in by_seed_year if entry_site == site})
    for seed in available_seeds:
        seed_signatures[seed] = tuple(
            json.dumps(by_seed_year[(site, seed, year)], separators=(",", ":"), ensure_ascii=False)
            for year in YEARS
        )
    exact_counts = Counter(seed_signatures.values())
    dominant_exact_count = max(exact_counts.values(), default=0)
    similarities: list[float] = []
    similarity_matrix = [[1.0 if i == j else math.nan for j in SEEDS] for i in SEEDS]
    for idx, seed_a in enumerate(available_seeds):
        for seed_b in available_seeds[idx + 1 :]:
            annual = []
            for year in YEARS:
                a = {tuple(x) for x in by_seed_year[(site, seed_a, year)]}
                b = {tuple(x) for x in by_seed_year[(site, seed_b, year)]}
                annual.append(jaccard(a, b))
            pair_score = float(np.mean(annual))
            similarities.append(pair_score)
            similarity_matrix[seed_a][seed_b] = pair_score
            similarity_matrix[seed_b][seed_a] = pair_score
    median_similarity = float(statistics.median(similarities)) if similarities else math.nan
    if median_similarity >= 0.80:
        assessment = "SEED_STABLE"
    elif median_similarity <= 0.35:
        assessment = "SEED_SENSITIVE"
    else:
        assessment = "MIXED_SEED_BEHAVIOR"
    provisional_assessment = assessment
    if len(seed_signatures) < expected_seed_count:
        assessment = "INCOMPLETE_EIGHT_SEED_COVERAGE"
    return {
        "site": site,
        "seed_count": len(seed_signatures),
        "expected_seed_count": expected_seed_count,
        "coverage_status": f"{len(seed_signatures)}/{expected_seed_count}",
        "dominant_full_10y_policy_signature_seed_count": dominant_exact_count,
        "median_pairwise_yearly_event_jaccard": median_similarity,
        "pairwise_seed_similarity_matrix": similarity_matrix,
        "site_level_assessment": assessment,
        "provisional_assessment_from_completed_seeds": provisional_assessment if len(seed_signatures) < expected_seed_count else "",
        "classification_rule": "median pairwise per-year event-set Jaccard >=0.80 stable; <=0.35 sensitive; otherwise mixed",
    }


def plot_seed(site: str, seed: int, metrics: list[dict[str, Any]], events: list[dict[str, Any]], target: Path) -> list[dict[str, str]]:
    target.mkdir(parents=True, exist_ok=True)
    by_year = {int(row["year"]): row for row in metrics if row["site"] == site and int(row["seed"]) == seed}
    event_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event["site"] == site and int(event["seed"]) == seed:
            event_year[int(event["year"])].append(event)
    x = np.arange(len(YEARS))
    labels = [str(y) for y in YEARS]
    plotted: list[dict[str, str]] = []

    # Metrics: keep the YC panel order and mark metrics that have no exact source data.
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    for ax, key, ylabel, color in (
        (axes[0], "grain_yield_kg_ha", "Grain yield (kg/ha)", COLORS["yield"]),
        (axes[1], "WP_ET_kg_m3", "WP_ET / WUE (kg/m³)", COLORS["irrigation"]),
        (axes[2], "PFP_N_kg_grain_per_kg_N", "PFP_N (kg grain/kg N)", COLORS["pfp_n"]),
        (axes[3], "NUE", "NUE", COLORS["nitrogen"]),
    ):
        vals = [number(by_year[y].get(key)) for y in YEARS]
        if any(math.isfinite(v) for v in vals):
            ax.plot(x, vals, color=color, marker="o", lw=1.8, ms=4)
        else:
            reason = "N uptake not in selected outputs" if key == "NUE" else "requires exact Summary.OUT + ETCP replay"
            ax.text(0.5, 0.5, f"N/A: {reason}", transform=ax.transAxes, ha="center", va="center", color=COLORS["muted"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.24)
    axes[-1].set_xticks(x, labels)
    axes[-1].set_xlabel("Observed validation year")
    fig.suptitle(f"{site} | historical-weather MaskablePPO seed {seed} | 100K", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    name = f"seed{seed}_annual_metrics.png"
    fig.savefig(target / name, dpi=180)
    plt.close(fig)
    plotted.append({"site": site, "seed": str(seed), "file": (target / name).relative_to(ROOT).as_posix(), "description": "YC-style annual yield, WP_ET, PFP_N, NUE panels; unavailable metrics marked N/A"})

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, key, title, color in (
        (axes[0], "total_irrigation_mm", "Total irrigation (mm)", COLORS["irrigation"]),
        (axes[1], "total_n_kg_ha", "Total nitrogen (kg N/ha)", COLORS["nitrogen"]),
    ):
        ax.bar(x, [number(by_year[y][key], 0.0) for y in YEARS], color=color, width=0.72)
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.24)
    axes[-1].set_xticks(x, labels)
    axes[-1].set_xlabel("Observed validation year")
    fig.suptitle(f"{site} | annual management totals | seed {seed}", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    name = f"seed{seed}_annual_management.png"
    fig.savefig(target / name, dpi=180)
    plt.close(fig)
    plotted.append({"site": site, "seed": str(seed), "file": (target / name).relative_to(ROOT).as_posix(), "description": "Annual irrigation and nitrogen totals"})

    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
    common = [number(by_year[y].get("common_reward")) for y in YEARS]
    canonical = [number(by_year[y].get("canonical_reward_return")) for y in YEARS]
    axes[0].plot(x, common, color=COLORS["common_reward"], marker="o", lw=1.8)
    axes[0].set_ylabel("Common reward")
    axes[0].set_title(COMMON_REWARD_FORMULA, loc="left", fontsize=9)
    axes[0].grid(axis="y", alpha=0.24)
    axes[1].plot(x, canonical, color=COLORS["policy"], marker="o", lw=1.8)
    axes[1].set_ylabel("Canonical episode return")
    axes[1].set_xlabel("Observed validation year")
    axes[1].grid(axis="y", alpha=0.24)
    axes[1].set_xticks(x, labels)
    fig.suptitle(f"{site} | common and PPO-only reward | seed {seed}", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    name = f"seed{seed}_reward.png"
    fig.savefig(target / name, dpi=180)
    plt.close(fig)
    plotted.append({"site": site, "seed": str(seed), "file": (target / name).relative_to(ROOT).as_posix(), "description": "Common reward and canonical PPO return on separate axes"})

    trend_panels = (
        ("grain_yield_kg_ha", "Yield (kg/ha)", COLORS["yield"]),
        ("common_reward", "Common reward", COLORS["common_reward"]),
        ("total_irrigation_mm", "Irrigation (mm)", COLORS["irrigation"]),
        ("total_n_kg_ha", "N fertilizer (kg/ha)", COLORS["nitrogen"]),
        ("PFP_N_kg_grain_per_kg_N", "PFP_N", COLORS["pfp_n"]),
    )
    fig, axes = plt.subplots(len(trend_panels), 1, figsize=(14, 13), sharex=True)
    for ax, (key, ylabel, color) in zip(axes, trend_panels):
        ax.plot(x, [number(by_year[y].get(key)) for y in YEARS], color=color, marker="o", lw=1.7, ms=4)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
    axes[-1].set_xticks(x, labels)
    axes[-1].set_xlabel("Observed validation year")
    fig.suptitle(f"{site} | annual trends | seed {seed}", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    name = f"seed{seed}_annual_trends.png"
    fig.savefig(target / name, dpi=180)
    plt.close(fig)
    plotted.append({"site": site, "seed": str(seed), "file": (target / name).relative_to(ROOT).as_posix(), "description": "Annual yield, common reward, irrigation, N, and PFP_N trends"})

    year_pos = {year: i for i, year in enumerate(YEARS)}
    fig, ax = plt.subplots(figsize=(12, 7))
    for year in YEARS:
        for event in event_year.get(year, []):
            y = year_pos[year]
            if number(event["irrigation_mm"], 0.0) > 0:
                ax.scatter(event["dap"], y - 0.12, marker="o", s=42, color=COLORS["irrigation"], edgecolor="white", linewidth=0.45)
            if number(event["nitrogen_kg_ha"], 0.0) > 0:
                ax.scatter(event["dap"], y + 0.12, marker="s", s=44, color=COLORS["nitrogen"], edgecolor="white", linewidth=0.45)
    ax.set_yticks(range(len(YEARS)), YEARS)
    ax.invert_yaxis()
    ax.set_xlim(-5, 125)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90, 105, 120])
    ax.set_xlabel("Days after planting (DAP)")
    ax.set_ylabel("Observed validation year")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", color=COLORS["irrigation"], label="Irrigation event"),
        plt.Line2D([], [], marker="s", ls="", color=COLORS["nitrogen"], label="Nitrogen event"),
    ], frameon=False, ncol=2, loc="upper right")
    fig.suptitle(f"{site} | action calendar | seed {seed}", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    name = f"seed{seed}_action_calendar.png"
    fig.savefig(target / name, dpi=180)
    plt.close(fig)
    plotted.append({"site": site, "seed": str(seed), "file": (target / name).relative_to(ROOT).as_posix(), "description": "Event timing by DAP and validation year; circles=I, squares=N"})

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, amount_key, title, color, marker in (
        (axes[0], "irrigation_mm", "Irrigation amount (mm/event)", COLORS["irrigation"], "o"),
        (axes[1], "nitrogen_kg_ha", "Nitrogen amount (kg/ha/event)", COLORS["nitrogen"], "s"),
    ):
        for year in YEARS:
            for event in event_year.get(year, []):
                amount = number(event[amount_key], 0.0)
                if amount <= 0:
                    continue
                y = year_pos[year]
                ax.scatter(event["dap"], y, marker=marker, s=42, color=color, edgecolor="white", linewidth=0.45)
                ax.annotate(f"{amount:g}", (event["dap"], y), xytext=(3, -8 if amount_key == "irrigation_mm" else 4), textcoords="offset points", fontsize=7, color=color)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_yticks(range(len(YEARS)), YEARS)
        ax.invert_yaxis()
        ax.set_xlim(-5, 125)
        ax.set_xticks([0, 15, 30, 45, 60, 75, 90, 105, 120])
        ax.grid(axis="x", alpha=0.22)
    axes[-1].set_xlabel("Days after planting (DAP)")
    axes[0].set_ylabel("Observed validation year")
    axes[1].set_ylabel("Observed validation year")
    fig.suptitle(f"{site} | management doses by year | seed {seed}", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    name = f"seed{seed}_management_action_doses_by_year.png"
    fig.savefig(target / name, dpi=180)
    plt.close(fig)
    plotted.append({"site": site, "seed": str(seed), "file": (target / name).relative_to(ROOT).as_posix(), "description": "Management event dose and DAP for each validation year"})
    return plotted


def plot_site_summaries(seed_rows: list[dict[str, Any]], site_rows: list[dict[str, Any]], out: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for site in ("HL", "FQ"):
        frame = [r for r in seed_rows if r["site"] == site]
        fig, axes = plt.subplots(1, 4, figsize=(14, 5))
        for ax, key, label, color in (
            (axes[0], "mean_yield_kg_ha", "Yield (kg/ha)", COLORS["yield"]),
            (axes[1], "mean_irrigation_mm", "Irrigation (mm)", COLORS["irrigation"]),
            (axes[2], "mean_n_kg_ha", "N fertilizer (kg/ha)", COLORS["nitrogen"]),
            (axes[3], "mean_pfp_n", "PFP_N", COLORS["pfp_n"]),
        ):
            values = [number(r.get(key)) for r in frame]
            values = [value for value in values if math.isfinite(value)]
            if values:
                ax.boxplot(values, showfliers=True, widths=0.5)
                ax.scatter(np.ones(len(values)), values, color=color, s=28, zorder=3)
            ax.set_xticks([1], [site])
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=0.22)
        fig.suptitle(f"{site} | seed distribution of 10-year means (100K)", x=0.02, ha="left", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        name = f"{site.lower()}_seed_metric_distributions.png"
        fig.savefig(out / name, dpi=180)
        plt.close(fig)
        entries.append({"site": site, "seed": "all", "file": (out / name).relative_to(ROOT).as_posix(), "description": "Per-seed distribution of 10-year mean yield, irrigation, N and PFP_N"})

        class_order = ["YEAR_SPECIFIC_POLICY", "MOSTLY_STABLE_WITH_VARIATION", "IDENTICAL_ACROSS_YEARS"]
        class_color = {class_order[0]: "#2878B5", class_order[1]: "#D8A305", class_order[2]: "#2A9D55",
                       "NOT_EVALUATED_SMOKE_GATE_BLOCKED": "#777777"}
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(8)
        for idx, row in enumerate(sorted(frame, key=lambda r: int(r["seed"]))):
            cls = str(row["year_to_year_classification"])
            ax.bar(idx, 1, color=class_color.get(cls, "#777777"), width=0.74)
            label = "BLOCKED\nBY SMOKE" if cls == "NOT_EVALUATED_SMOKE_GATE_BLOCKED" else cls.replace("_", "\n")
            ax.text(idx, 0.5, label, ha="center", va="center", fontsize=7, color="white", fontweight="bold")
        ax.set_xticks(x, [str(i) for i in range(8)])
        ax.set_ylim(0, 1.05)
        ax.set_yticks([])
        ax.set_xlabel("PPO seed")
        ax.set_title(f"{site} | year-to-year management classification by seed", loc="left", fontweight="bold")
        fig.tight_layout()
        name = f"{site.lower()}_seed_policy_classifications.png"
        fig.savefig(out / name, dpi=180)
        plt.close(fig)
        entries.append({"site": site, "seed": "all", "file": (out / name).relative_to(ROOT).as_posix(), "description": "Seed-wise A/B/C year-to-year management classification"})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, site in zip(axes, ("HL", "FQ")):
        matrix = np.eye(8)
        lookup = {(r["site"], int(r["seed"])): r for r in []}
        # Similarity values are added by the caller in the site summary JSON;
        # this compact comparison shows per-seed classification counts.
        counts = Counter(r["year_to_year_classification"] for r in seed_rows if r["site"] == site)
        labels2 = class_order
        vals = [counts[label] for label in labels2]
        ax.bar(range(3), vals, color=[class_color[k] for k in labels2])
        ax.set_xticks(range(3), ["A", "B", "C"])
        ax.set_ylim(0, 8)
        ax.set_ylabel("Number of seeds")
        ax.set_title(f"{site} | A/B/C counts", loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Year-to-year policy stability by site", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    name = "hl_fq_year_to_year_classification_counts.png"
    fig.savefig(out / name, dpi=180)
    plt.close(fig)
    entries.append({"site": "HL/FQ", "seed": "all", "file": (out / name).relative_to(ROOT).as_posix(), "description": "Station-level A/B/C counts"})
    return entries


def yc_existing_results() -> tuple[list[dict[str, Any]], dict[tuple[int, int], list[list[Any]]]]:
    base = ROOT / "results/yc_random_weather_ppo/004_15_75mm_all_seed_five_scenario_figures"
    metrics_path = base / "all_seeds_five_scenario_metrics.csv"
    events_path = base / "all_seeds_five_scenario_management_events.csv"
    if not metrics_path.is_file() or not events_path.is_file():
        raise FileNotFoundError(f"YC existing 004_15 files missing under {base}")
    metrics = [r for r in read_csv(metrics_path) if r.get("scenario") == "random_weather_ppo"]
    keys = {(int(number(r.get("seed"), -1)), int(number(r.get("year"), -1))) for r in metrics}
    if keys != {(seed, year) for seed in SEEDS for year in YEARS}:
        raise ValueError("YC 004_15 PPO observed-weather metrics are not 8 seeds x 10 years")
    by_year: dict[tuple[int, int], list[list[Any]]] = {(seed, year): [] for seed in SEEDS for year in YEARS}
    for row in read_csv(events_path):
        if row.get("scenario") != "random_weather_ppo":
            continue
        seed, year = int(number(row.get("seed"), -1)), int(number(row.get("year"), -1))
        if seed not in SEEDS or year not in YEARS:
            continue
        i_mm = rounded_action(row.get("irrigation"))
        n_kg = rounded_action(row.get("nitrogen"))
        if i_mm > 0 or n_kg > 0:
            by_year[(seed, year)].append([int(number(row.get("dap"), -1)), i_mm, n_kg])
    for key in by_year:
        by_year[key].sort(key=lambda r: (r[0], r[1], r[2]))
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_metric = [r for r in metrics if int(number(r.get("seed"), -1)) == seed]
        sigs = [json.dumps(by_year[(seed, year)], separators=(",", ":"), ensure_ascii=False) for year in YEARS]
        classification, mode_count, unique_count = classify_years(sigs)
        rows.append({
            "site": "YC", "seed": seed, "validation_years": "2014-2023",
            "year_to_year_classification": classification,
            "years_matching_modal_action_sequence": mode_count,
            "unique_yearly_action_sequences": unique_count,
            "mean_yield_kg_ha": mean_number(number(r.get("yield")) for r in seed_metric),
            "mean_irrigation_mm": mean_number(number(r.get("irrigation")) for r in seed_metric),
            "mean_n_kg_ha": mean_number(number(r.get("nitrogen")) for r in seed_metric),
            "mean_pfp_n": mean_number(number(r.get("pfp_n")) for r in seed_metric),
            "mean_wp_et": mean_number(number(r.get("wp_et")) for r in seed_metric),
            "training_weather_source": "RANDOM_WEATHER_WGEN",
            "classification_evidence": "Existing 004_15 event table; no YC training or replay performed",
        })
    return rows, by_year


def mean_number(values: Any) -> float:
    valid = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(statistics.mean(valid)) if valid else math.nan


def config_audit(snapshots: dict[str, dict[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for site, spec in SITE_SPECS.items():
        weather_root = ROOT / spec["weather_root"]
        config0 = read_json(ROOT / spec["seed0_config"])
        train_years = list(map(int, config0["scope"]["train_years"]))
        val_years = list(map(int, config0["scope"]["validation_years"]))
        training_wth = [weather_root / f"CN{spec['prefix']}{year % 100:02d}01.WTH" for year in train_years]
        validation_wth = [weather_root / f"CN{spec['prefix']}{year % 100:02d}01.WTH" for year in val_years]
        expected_wth = training_wth + validation_wth
        missing_train_wth = [p.relative_to(ROOT).as_posix() for p in training_wth if not p.is_file()]
        missing_validation_wth = [p.relative_to(ROOT).as_posix() for p in validation_wth if not p.is_file()]
        mzx_path = ROOT / spec["mzx"]
        if not mzx_path.is_file():
            raise FileNotFoundError(mzx_path)
        cfgs = []
        for seed in SEEDS:
            if seed in snapshots[site] and "config" in snapshots[site][seed]:
                cfgs.append(snapshots[site][seed]["config"])
            else:
                config_path, _ = config_and_output(site, seed)
                cfgs.append(read_json(config_path))
        for cfg in cfgs:
            if cfg["training"]["total_timesteps"] != 100000 or cfg["scope"]["validation_years"] != val_years:
                raise ValueError(f"Seed config scope mismatch for {site}: {cfg}")
            if cfg["input_profile"] != spec["profile"]:
                raise ValueError(f"Unexpected input profile for {site}: {cfg['input_profile']}")
            if cfg["actions"]["irrigation_levels_mm"] != [0.0, 15.0, 30.0, 45.0]:
                raise ValueError(f"Unexpected I action grid for {site} seed {cfg['seed']}")
            if cfg["actions"]["nitrogen_levels_kg_ha"] != [0.0, 40.0, 80.0, 120.0]:
                raise ValueError(f"Unexpected N action grid for {site} seed {cfg['seed']}")
        baseline_config_content = {k: v for k, v in config0.items() if k not in {"task_id", "task_name", "seed"}}
        seed_configs_only_identity_fields_changed = all(
            {k: v for k, v in cfg.items() if k not in {"task_id", "task_name", "seed"}} == baseline_config_content
            for cfg in cfgs
        )
        if not seed_configs_only_identity_fields_changed:
            raise ValueError(f"Non-seed config differences found for {site}; refusing report")
        rows.append({
            "site": site,
            "station_code": spec["station_code"],
            "algorithm": "MaskablePPO",
            "ppo_seeds": "0,1,2,3,4,5,6,7",
            "total_timesteps": 100000,
            "checkpoint_selection": "fixed 100000 environment-step checkpoint",
            "reward_contract": "040_28 reward_04026 minus 50*max(SWFAC_after-0.05,0)*reward_scale; unchanged across seed configs",
            "effective_runner_chain": "051/054 site runner -> 042_10 -> 040_36 -> 040_28",
            "declared_config_reward_safety": config0["scope"]["reward_and_safety"],
            "action_space": "16 joint actions; I=[0,15,30,45] mm x N=[0,40,80,120] kg/ha",
            "action_mask": "effective 040_36 late-reserve/staged mask with 040_28 SWFAC guardrail threshold 0.05; daily-output flags audited",
            "irrigation_limits": "DAP<=30:75; DAP<=60:150; DAP<=90:195; season:240 mm",
            "nitrogen_limit": "season <=250 kg/ha",
            "minimum_management_interval": "7 days separately for repeat irrigation and repeat nitrogen applications",
            "IC_mode": spec["profile"],
            "soil_id": spec["soil_id"],
            "cultivar_id": spec["cultivar_id"],
            "cultivar_name": spec["cultivar_name"],
            "MZX_soil_cultivar_IC_source": spec["mzx"],
            "MZX_sha256": sha256(mzx_path),
            "training_weather_source": "station-specific observed historical .WTH pool; WGEN/random weather disabled",
            "training_weather_root": spec["weather_root"],
            "training_years": f"{min(train_years)}-{max(train_years)}",
            "training_WTH_files": ";".join(p.name for p in training_wth),
            "validation_years": f"{min(val_years)}-{max(val_years)}",
            "validation_WTH_files": ";".join(p.name for p in validation_wth),
            "training_WTH_missing_count": len(missing_train_wth),
            "validation_WTH_missing_count": len(missing_validation_wth),
            "configuration_seed_changes_only": seed_configs_only_identity_fields_changed,
            "configuration_source": spec["seed0_config"],
        })
    return rows


def build_report(
    config_rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    station_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
    image_rows: list[dict[str, str]],
    out_dir: Path,
    smoke_summary: list[dict[str, str]],
) -> str:
    failed_mask_rows = [row for row in mask_rows if not row["all_observed_mask_checks_pass"]]
    failed_gap_rows = [
        row for row in mask_rows
        if not row["minimum_irrigation_interval_pass"] or not row["minimum_fertilization_interval_pass"]
    ]
    gap_details = "; ".join(
        f"{row['site']} seed{row['seed']} {row['year']} I={row['minimum_irrigation_interval_days']}d/N={row['minimum_fertilization_interval_days']}d"
        for row in failed_gap_rows[:12]
    )
    lines = [
        "# HL/FQ 历史天气 PPO 8-seed baseline 进度与跨站点策略诊断",
        "",
        "## 1. 任务目的与边界",
        "",
        "本轮完成 HL seeds 0–7（8/8）与 FQ seeds 0–6（7/8）的历史天气 MaskablePPO 100K baseline，使用 2014–2023 十个独立验证年份。seed 0 复用已有正式结果；只对缺失 seed 运行正式训练与验证。FQ seed 7 因其 2K smoke action-response gate 未通过，被正式 runner 的 preflight 拒绝，未开始 100K 训练，因此本轮整体为 15/16 个 seed 正式结果。YC 仅读取现有 004_15 结果作描述性对照。未生成 WGEN 天气，未修改 YC、reward、action mask、动作空间、IC 或天气/训练年份配置。",
        "",
        "比较 YC 时需注意：YC PPO 训练来源为 RANDOM_WEATHER_WGEN；本轮 HL/FQ 均为各站历史 .WTH 年份池。因此是跨站点、跨训练天气来源的描述性策略对照，不是严格同天气处理效应比较。",
        "",
        "## 2. 配置审计",
        "",
        "| Site | Seeds | PPO/步数/选点 | Reward | 动作空间 | 安全约束 | IC 与 MZX | 训练天气/年份 | 验证年份 | WGEN |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in config_rows:
        lines.append(
            f"| {row['site']} | {row['ppo_seeds']} | {row['algorithm']} / {row['total_timesteps']} / {row['checkpoint_selection']} | "
            f"{row['reward_contract']} | {row['action_space']} | {row['irrigation_limits']}; N {row['nitrogen_limit']}; 间隔 {row['minimum_management_interval']} | "
            f"{row['IC_mode']}; soil `{row['soil_id']}`; cultivar `{row['cultivar_id']} {row['cultivar_name']}`; `{row['MZX_soil_cultivar_IC_source']}` | "
            f"`{row['training_weather_root']}`，{row['training_years']}（WTH） | {row['validation_years']} | 否 |"
        )
    lines.extend([
        "",
        "HL 的土壤、品种、IC 来源锁定在 `CNHL0701_corrected_IC123.MZX` 与 lowIC profile；FQ 锁定在 `CNFQ0801.MZX` 与 originIC profile。逐年天气文件清单和 MZX SHA-256 在 `results/hl_fq_8seed_cross_site/analysis/configuration_audit.csv`。每个 seed 的 config snapshot、checkpoint 路径/hash、正式日志状态在 `run_metadata.json`。",
        "",
        "配置 JSON 的 `reward_and_safety` 字段文字标记 `inherit_042_15_via_046_02_unchanged`；但本次实际 051/054 runner 的 import/delegation 链是 `042_10 -> 040_36 -> 040_28`。因此当前 reward 为 `reward_04026 - 50 × max(SWFAC_after - 0.05, 0) × reward_scale`；040_36 在继承的 staged/guardrail wrapper 上新增 DAP≤90 累计灌溉≤195 mm late-reserve mask。逐日输出 flag 与参数均用于核验生效状态。报告按实际执行代码与输出记录审计，未更改该声明或实际配置。",
        "",
        "每日输出还核对 action level、mask 输出标记、DAP 30/60/90 与全季灌溉上限、全季 N 上限及分别计算的灌溉/施氮事件间隔。实际代码设定 I 与 N 各自至少 7 天。全量逐年结果在 `mask_and_closure_audit.csv`。",
        f"逐日 action trace 安全审计状态：{'FAIL' if failed_mask_rows else 'PASS'}（{len(failed_mask_rows)} 个 seed-year 行未通过）。其中事件间隔不符的逐年行数为 {len(failed_gap_rows)}；前若干行：{gap_details or '无'}。HL seed 5 的所有十个验证年均在记录 DAP 上呈施氮间隔 6 天（DAP 2→8），尽管现行代码配置要求 7 天且日志记录动作未被裁剪；此项保留为实测配置/输出不一致，未修改 action mask，也不宣称其通过。",
        "逐日日志存在 DAP 从 5 回到 1 的一次重置记录；完整 action hash 按 CSV 原始行序计算，不按 DAP 重排。上述 DAP 重置和 HL seed 5 间隔差异均应在解读 action-sequence 时保留。",
        "",
        "## 3. Seeds、正式训练与验证完整性",
        "",
        "标准 seed 列表：`0,1,2,3,4,5,6,7`。HL seed 0、FQ seed 0 复用既有 100K 正式 checkpoint；HL 1–7 与 FQ 1–6 均各正式运行一次且 exit 0。已完成 seed 均有 100K checkpoint、十个 2014–2023 日过程 CSV、十条 100K 验证汇总行，且水氮总量/事件数与逐日 action trace 闭合。全量核验见 `seed_completion.csv`。",
        "",
        "FQ seed 7 的 2K smoke 输出保留在 `benchmark_results/8fq07_fq_seed7_8seed_smoke2k`，`smoke_gate_summary.csv` 记为 fail。其余 smoke 检查通过，但 `not_all_positive_actions_at_dap1=false`；FQ 正式 runner 的 `verify_completed_smoke()` 因此返回 `next_step_allowed=false`，正式训练在 preflight 阶段明确拒绝（本轮尝试退出码 1），没有 100K checkpoint，也没有正式验证结果。为遵守项目 smoke gate，本轮不绕过该 preflight。seed 7 保留为 `NOT_EVALUATED_SMOKE_GATE_BLOCKED`，FQ 站点层级 8-seed 判断标记为 coverage 不完整。",
        "",
        "## 4. 指标口径与限制",
        "",
        "产量、灌溉、施氮、事件数、PFP_N、episode return 取固定 100K 的正式验证汇总，并与逐日记录闭合。PFP_N 按现有正式表口径报告；N=0 时为 N/A。当前 051/054 验证产物没有精确作物吸氮量，因此 NUE 不估算；没有本轮各 seed 的 Summary.OUT/ETCP 精确回放，因此 WUE/WP_ET 不从灌溉量推算，均记 N/A。YC 的 WP_ET 只沿用既有有精确 ETCP 回放的 seed。",
        "",
        "## 5. 年际 action-sequence 判定",
        "",
        "标准化序列以每个验证年的实际非零管理事件构成，事件字段为 DAP、灌溉量、施氮量；逐日完整 action index/水氮剂量序列也另存于 `normalized_action_sequences.json`。对年际管理措施分类时忽略不改变管理量的收获后/no-op 行，避免生育期长度差异造成虚假策略差异；总量相同但日期不同仍视为不同序列。分类阈值固定为：全十年事件序列完全相同为 C；否则若同一完整序列至少覆盖 7/10 年为 B；其他为 A。",
        "",
        "A=`YEAR_SPECIFIC_POLICY`；B=`MOSTLY_STABLE_WITH_VARIATION`；C=`IDENTICAL_ACROSS_YEARS`。逐 seed 结果如下：",
        "",
        "| Site | Seed | 管理事件分类 | 匹配事件序列年数/10 | 唯一事件序列数 | 完整逐日 action 是否十年全同 | 唯一完整逐日序列数 | 平均 yield | 平均 irrigation | 平均 N | 平均 PFP_N |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in comparison_rows:
        lines.append(
            f"| {row['site']} | {row['seed']} | {row['year_to_year_classification']} | {row['years_matching_modal_action_sequence']}/10 | "
            f"{row['unique_yearly_action_sequences']} | {row.get('full_daily_action_sequence_identical_across_all_years', 'N/A')} | "
            f"{row.get('unique_full_daily_action_sequences', 'N/A')} | {fmt(row.get('mean_yield_kg_ha'))} | {fmt(row.get('mean_irrigation_mm'))} | "
            f"{fmt(row.get('mean_n_kg_ha'))} | {fmt(row.get('mean_pfp_n'))} |"
        )
    lines.extend([
        "",
        "逐年水氮措施与指标完整表见 `metrics_per_seed_year.csv` 和 `management_events_per_seed_year.csv`。以下列出所有已完成正式运行的 seed-year 完整事件序列；FQ seed 7 没有正式评估序列：",
        "",
    ])
    for site in ("HL", "FQ"):
        lines.extend([f"### {site} 每年水氮措施", "", "| Seed | Year | Yield kg/ha | Irrigation mm | N kg/ha | I/N events | PFP_N | DAP 与联合措施序列 |", "|---:|---:|---:|---:|---:|---:|---:|---|"])
        for metric in [r for r in metrics if r["site"] == site]:
            sequence = metric.get("management_action_sequence_text", "")
            lines.append(
                f"| {metric['seed']} | {metric['year']} | {fmt(metric.get('grain_yield_kg_ha'))} | {fmt(metric.get('total_irrigation_mm'))} | "
                f"{fmt(metric.get('total_n_kg_ha'))} | {metric['irrigation_event_count']}/{metric['nitrogen_event_count']} | "
                f"{fmt(metric.get('PFP_N_kg_grain_per_kg_N'))} | {sequence} |"
            )
        lines.append("")

    lines.extend([
        "## 6. 跨 seed 稳定性与站点判断",
        "",
        "站点层级分类使用已完成 seed 之间按年份计算的管理事件集合 Jaccard 相似度中位数：≥0.80 为 `SEED_STABLE`，≤0.35 为 `SEED_SENSITIVE`，其余为 `MIXED_SEED_BEHAVIOR`。FQ 仅完成 7/8，因此其最终 8-seed 站点判断不下结论，标记 `INCOMPLETE_EIGHT_SEED_COVERAGE`，表内相似度与 provisional assessment 只描述已完成的 7 个 seed。这描述的是策略输出相似度，不代表产量最优或因果稳健性。",
        "",
        "| Site | Seeds | A | B | C | seed 间中位相似度 | 站点判断 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in station_rows:
        counts = Counter(r["year_to_year_classification"] for r in seed_rows if r["site"] == row["site"])
        lines.append(
            f"| {row['site']} | {row['seed_count']} | {counts['YEAR_SPECIFIC_POLICY']} | {counts['MOSTLY_STABLE_WITH_VARIATION']} | "
            f"{counts['IDENTICAL_ACROSS_YEARS']} | {fmt(row['median_pairwise_yearly_event_jaccard'])} | {row['site_level_assessment']} |"
        )
    lines.extend([
        "",
        "## 7. 与 YC 现有结果对照",
        "",
        "YC 的每个 seed 同样用 observed-weather 2014–2023 事件表作分类；未训练、未回放、未改动 YC。YC 使用 random-weather 训练来源，HL/FQ 使用各自历史 WTH，因此该对照用于判断‘完全一致’是否在站点间普遍出现，不用于宣称任一固定策略更优。逐 seed 对照表见 `site_seed_action_classification.csv`。",
        "",
        "## 8. 图表索引",
        "",
        "每个有正式结果的 HL/FQ seed 输出 6 张与 YC 004_15 同类图：年度指标、年度水氮总量、reward、年度趋势、DAP action calendar、逐年剂量图。FQ seed 7 没有正式逐年输出，因此不绘制 seed 图；站点图将其灰显为 smoke gate blocked。站点 seed 分布和 A/B/C 汇总图也一并生成。完整清单见 `figure_manifest.csv`。",
        "",
    ])
    for row in image_rows:
        lines.append(f"- `{row['site']} seed {row['seed']}`：[{Path(row['file']).name}](../{Path(row['file']).as_posix()}) — {row['description']}")
    lines.extend([
        "",
        "## 9. 当前能得出的结论",
        "",
        "- HL 8/8 和 FQ 7/8 已完成 seed 均固定选择 100K checkpoint，并在独立验证年逐年记录管理措施；训练天气仍是历史 WTH。FQ seed 7 受 smoke preflight gate 阻断，没有正式结果。",
        "- 是否普遍出现‘十年措施完全一致’由上表 C 计数直接判定；仅在事件日期与剂量完全相同时记为 C。",
        "- YC 若 C 数明显高于 HL/FQ，只能称为当前站点/训练天气情境中更突出的现象；不能据此断言固定策略正确或最优。",
        "- NUE 与本轮 WUE/WP_ET 没有精确来源，均不作推断；yield-only 不能替代资源效率结论。",
        "",
        "## 10. 当前不能得出的结论与下一步",
        "",
        "本结果不能证明跨站点统计显著性、模型最优性或随机天气鲁棒性；YC 与 HL/FQ 的训练天气来源不同。若后续需要 WUE/WP_ET 或 NUE，须先取得每个 seed-year 精确 ETCP/Summary.OUT 或作物吸氮量来源并核对闭合。WGEN 版本留给后续独立任务，本轮未生成。",
        "",
        f"完整 machine-readable 表、图和 provenance 位于 `{out_dir.relative_to(ROOT).as_posix()}`。",
    ])
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    num = number(value)
    return "N/A" if not math.isfinite(num) else f"{num:.2f}"


def main() -> None:
    if FIGURE_ROOT.exists() and any(FIGURE_ROOT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite figure directory: {FIGURE_ROOT}")
    analysis_dirs = sorted(RESULT_ROOT.glob("analysis_v*"))
    version = len(analysis_dirs) + 1
    out_dir = RESULT_ROOT / f"analysis_v{version}"
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite analysis directory: {out_dir}")
    report_path = ROOT / "docs/hl_fq_8seed_cross_site_policy_check.md"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report_path}")

    metrics: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    snapshots: dict[str, dict[int, dict[str, Any]]] = {site: {} for site in SITE_SPECS}
    by_seed_year: dict[tuple[str, int, int], list[list[Any]]] = {}
    completion_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    training_log_rows: list[dict[str, Any]] = []

    for site, spec in SITE_SPECS.items():
        for seed in SEEDS:
            config_path, output_root = config_and_output(site, seed)
            if not config_path.is_file():
                raise FileNotFoundError(config_path)
            config = read_json(config_path)
            if int(config["seed"]) != seed:
                raise ValueError(f"Config seed mismatch: {config_path}")
            if site == "FQ" and seed == 7:
                smoke_root = ROOT / "benchmark_results/8fq07_fq_seed7_8seed_smoke2k"
                smoke_results = sorted(smoke_root.glob("*smoke_result.json"))
                if len(smoke_results) != 1:
                    raise FileNotFoundError(f"Expected one FQ seed 7 smoke result under {smoke_root}")
                smoke_result = read_json(smoke_results[0])
                smoke_gate = smoke_result.get("smoke_gate", {})
                if smoke_gate.get("next_step_allowed") is not False:
                    raise ValueError(f"Unexpected FQ seed 7 smoke-gate status: {smoke_gate}")
                failed_formal_console = ROOT / "results/hl_fq_8seed_cross_site/console_logs/fq_seed7_formal.log"
                snapshots[site][seed] = {
                    "config": config,
                    "config_path": config_path.relative_to(ROOT).as_posix(),
                    "config_sha256": sha256(config_path),
                    "smoke_output_root": smoke_root.relative_to(ROOT).as_posix(),
                    "smoke_result_path": smoke_results[0].relative_to(ROOT).as_posix(),
                    "smoke_gate": smoke_gate,
                    "formal_blocked_by_smoke_gate": True,
                    "formal_preflight_console_log": failed_formal_console.relative_to(ROOT).as_posix(),
                    "formal_preflight_console_log_sha256": sha256(failed_formal_console) if failed_formal_console.is_file() else "",
                }
                completion_rows.append({
                    "site": site, "seed": seed, "training_timesteps": 100000,
                    "training_completed": False, "fixed_100k_checkpoint": False,
                    "training_year_logs_expected": len(config["scope"]["train_years"]),
                    "training_year_logs_found": 0,
                    "validation_years_expected": 10, "validation_years_found": 0,
                    "smoke_checkpoint": smoke_gate.get("final_checkpoint"),
                    "smoke_validation_rows": smoke_gate.get("observed_validation_rows"),
                    "smoke_gate_next_step_allowed": smoke_gate.get("next_step_allowed"),
                    "smoke_gate_failed_check": "not_all_positive_actions_at_dap1",
                    "formal_runner_completed_and_audited": False,
                    "formal_status": "BLOCKED_BY_SMOKE_GATE",
                    "formal_preflight_console_log": failed_formal_console.relative_to(ROOT).as_posix(),
                })
                continue
            run_config_snapshot = output_root / "configs" / config_path.name
            if not run_config_snapshot.is_file():
                raise FileNotFoundError(f"Run config snapshot missing for {site} seed {seed}: {run_config_snapshot}")
            copied_config = read_json(run_config_snapshot)
            if copied_config != config:
                raise ValueError(f"Run config snapshot differs from canonical config for {site} seed {seed}")
            training_log_dir = output_root / "logs" / spec["station_code"]
            seed_training_logs: list[Path] = []
            missing_training_log_years: list[int] = []
            for training_year in map(int, config["scope"]["train_years"]):
                year_logs = sorted(training_log_dir.glob(f"{spec['station_code']}_{training_year}_*_train.log"))
                if not year_logs:
                    missing_training_log_years.append(training_year)
                seed_training_logs.extend(year_logs)
            if missing_training_log_years:
                raise FileNotFoundError(f"Training-year logs missing for {site} seed {seed}: {missing_training_log_years}")
            for log_path in seed_training_logs:
                training_log_rows.append({
                    "site": site, "seed": seed, "training_year": int(log_path.name.split("_")[1]),
                    "training_log_path": log_path.relative_to(ROOT).as_posix(),
                    "training_log_bytes": log_path.stat().st_size,
                    "training_log_sha256": sha256(log_path),
                })
            summaries = selected_summary(output_root, spec["station_code"], seed)
            checkpoint = output_root / "models" / spec["station_code"] / f"{spec['station_code']}_half_split_stress_aware_maskableppo_seed{seed}_ckpt100000.zip"
            if not checkpoint.is_file():
                matches = list((output_root / "models").rglob(f"*seed{seed}_ckpt100000.zip"))
                if len(matches) != 1:
                    raise FileNotFoundError(f"Expected one fixed 100K checkpoint for {site} seed {seed}; got {matches}")
                checkpoint = matches[0]
            checkpoint_digest = sha256(checkpoint)
            snapshots[site][seed] = {
                "config": config,
                "config_path": config_path.relative_to(ROOT).as_posix(),
                "config_sha256": sha256(config_path),
                "run_config_snapshot_path": run_config_snapshot.relative_to(ROOT).as_posix(),
                "run_config_snapshot_sha256": sha256(run_config_snapshot),
                "training_log_paths": [path.relative_to(ROOT).as_posix() for path in seed_training_logs],
                "output_root": output_root.relative_to(ROOT).as_posix(),
                "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(),
                "checkpoint_sha256": checkpoint_digest,
                "formal_console_log": f"results/hl_fq_8seed_cross_site/console_logs/{site.lower()}_seed{seed}_formal.log" if seed else "existing_formal_logs_under_benchmark_root",
            }
            checkpoint_rows.append({
                "site": site, "station_code": spec["station_code"], "seed": seed,
                "config_path": config_path.relative_to(ROOT).as_posix(), "config_sha256": sha256(config_path),
                "run_config_snapshot_path": run_config_snapshot.relative_to(ROOT).as_posix(),
                "run_config_snapshot_sha256": sha256(run_config_snapshot),
                "checkpoint_step": 100000, "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(),
                "checkpoint_sha256": checkpoint_digest,
            })
            console_path = ROOT / "results/hl_fq_8seed_cross_site/console_logs" / f"{site.lower()}_seed{seed}_formal.log"
            formal_log_sha256 = sha256(console_path) if console_path.is_file() else ""
            formal_completed = seed == 0
            if seed > 0:
                audit_path = output_root / "seed_override_audit.json"
                if not audit_path.is_file():
                    raise FileNotFoundError(f"Seed override audit missing for {site} seed {seed}: {audit_path}")
                seed_audit = read_json(audit_path)
                if seed_audit.get("mode") != "formal" or int(seed_audit.get("configured_seed", -1)) != seed:
                    raise ValueError(f"Formal seed audit mismatch for {site} seed {seed}: {seed_audit}")
                if not seed_audit.get("formal_100k_checkpoint_present"):
                    raise RuntimeError(f"Formal run status failed for {site} seed {seed}: audit={seed_audit}")
                if not console_path.is_file():
                    raise FileNotFoundError(f"Formal console log missing for {site} seed {seed}: {console_path}")
                formal_completed = True
            year_results: list[dict[str, Any]] = []
            for year in YEARS:
                daily_path = daily_output_path(output_root, spec["station_code"], year, seed)
                metric, events, artifact = extract_year(site, seed, year, summaries[year], daily_path, output_root)
                metric["checkpoint_sha256"] = checkpoint_digest
                metric["canonical_config"] = snapshots[site][seed]["config_path"]
                event_rows.extend(events)
                metrics.append(metric)
                year_results.append(metric)
                mask_rows.append(artifact["mask_audit"])
                sig = artifact["sequence"]
                by_seed_year[(site, seed, year)] = sig["event_sequence"]
                sequences.append(sig)
            if len(year_results) != 10 or not all(r["water_n_closure"] and r["terminal_yield_closure"] for r in year_results):
                raise ValueError(f"Daily closure failed for {site} seed {seed}")
            completion_rows.append({
                "site": site, "seed": seed, "training_timesteps": 100000,
                "training_completed": True, "fixed_100k_checkpoint": True,
                "training_year_logs_expected": len(config["scope"]["train_years"]),
                "training_year_logs_found": len({int(row["training_year"]) for row in training_log_rows if row["site"] == site and int(row["seed"]) == seed}),
                "training_log_files_found": len(seed_training_logs),
                "validation_years_expected": 10, "validation_years_found": len(year_results),
                "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(),
                "checkpoint_sha256": checkpoint_digest,
                "formal_console_log": console_path.relative_to(ROOT).as_posix() if console_path.is_file() else "existing_formal_logs_under_benchmark_root",
                "formal_console_log_sha256": formal_log_sha256,
                "formal_runner_completed_and_audited": formal_completed,
                "formal_status": "PASS" if formal_completed else "audit evidence missing",
            })

    yc_seed_rows, yc_by_seed_year = yc_existing_results()
    for (seed, year), events in yc_by_seed_year.items():
        by_seed_year[("YC", seed, year)] = events
    seed_rows: list[dict[str, Any]] = []
    full_sequence_hashes = {
        (row["site"], int(row["seed"]), int(row["year"])): row["full_daily_action_sequence_sha256"]
        for row in sequences
    }
    for site in ("YC", "HL", "FQ"):
        for seed in SEEDS:
            if site == "FQ" and seed == 7:
                seed_rows.append({
                    "site": "FQ", "seed": 7, "validation_years": "no formal evaluation (2K smoke only)",
                    "year_to_year_classification": "NOT_EVALUATED_SMOKE_GATE_BLOCKED",
                    "years_matching_modal_action_sequence": 0,
                    "unique_yearly_action_sequences": 0,
                    "full_daily_action_sequence_identical_across_all_years": None,
                    "unique_full_daily_action_sequences": None,
                    "mean_yield_kg_ha": None, "mean_irrigation_mm": None, "mean_n_kg_ha": None,
                    "mean_pfp_n": None, "mean_wp_et": None,
                    "training_weather_source": "historical station-specific WTH; formal training blocked",
                    "classification_evidence": "2K smoke action-response gate failed; formal preflight exit 1",
                })
                continue
            signatures = [json_signature(by_seed_year[(site, seed, year)]) for year in YEARS]
            classification, mode_count, unique_count = classify_years(signatures)
            full_sequence_hash_list = [full_sequence_hashes.get((site, seed, year)) for year in YEARS]
            full_sequence_hash_list = [value for value in full_sequence_hash_list if value is not None]
            full_sequence_unique_count = len(set(full_sequence_hash_list))
            full_sequence_identical = full_sequence_unique_count == 1 if len(full_sequence_hash_list) == len(YEARS) else None
            if site == "YC":
                yc = yc_seed_rows[seed]
                seed_rows.append({**yc, "year_to_year_classification": classification,
                                  "years_matching_modal_action_sequence": mode_count,
                                  "unique_yearly_action_sequences": unique_count,
                                  "full_daily_action_sequence_identical_across_all_years": None,
                                  "unique_full_daily_action_sequences": None})
            else:
                frame = [r for r in metrics if r["site"] == site and int(r["seed"]) == seed]
                seed_rows.append({
                    "site": site, "seed": seed, "validation_years": "2014-2023",
                    "year_to_year_classification": classification,
                    "years_matching_modal_action_sequence": mode_count,
                    "unique_yearly_action_sequences": unique_count,
                    "full_daily_action_sequence_identical_across_all_years": full_sequence_identical,
                    "unique_full_daily_action_sequences": full_sequence_unique_count,
                    "mean_yield_kg_ha": mean_number(r["grain_yield_kg_ha"] for r in frame),
                    "mean_irrigation_mm": mean_number(r["total_irrigation_mm"] for r in frame),
                    "mean_n_kg_ha": mean_number(r["total_n_kg_ha"] for r in frame),
                    "mean_pfp_n": mean_number(r["PFP_N_kg_grain_per_kg_N"] for r in frame),
                    "mean_wp_et": None,
                    "training_weather_source": "historical station-specific WTH",
                    "classification_evidence": "normalized actual management-event sequences from all ten daily outputs",
                })
    station_rows = [station_classification(site, by_seed_year) for site in ("HL", "FQ")]
    yc_station_row = station_classification("YC", by_seed_year)
    yc_counts = Counter(r["year_to_year_classification"] for r in seed_rows if r["site"] == "YC")
    yc_station_row["A_seed_count"] = yc_counts["YEAR_SPECIFIC_POLICY"]
    yc_station_row["B_seed_count"] = yc_counts["MOSTLY_STABLE_WITH_VARIATION"]
    yc_station_row["C_seed_count"] = yc_counts["IDENTICAL_ACROSS_YEARS"]
    station_rows.append(yc_station_row)
    for station_row in station_rows:
        class_counts = Counter(
            r["year_to_year_classification"] for r in seed_rows if r["site"] == station_row["site"]
        )
        station_row["A_seed_count"] = class_counts["YEAR_SPECIFIC_POLICY"]
        station_row["B_seed_count"] = class_counts["MOSTLY_STABLE_WITH_VARIATION"]
        station_row["C_seed_count"] = class_counts["IDENTICAL_ACROSS_YEARS"]
    hl_c = next(row["C_seed_count"] for row in station_rows if row["site"] == "HL")
    fq_c = next(row["C_seed_count"] for row in station_rows if row["site"] == "FQ")
    failed_mask_rows = [row for row in mask_rows if not row["all_observed_mask_checks_pass"]]
    diagnosis = (
        "BLOCKED_BY_CONFIGURATION_OR_RUNTIME_ERROR"
        if len([r for r in completion_rows if r["site"] == "FQ" and r["formal_status"] == "PASS"]) < 8 or failed_mask_rows
        else "YC_APPEARS_MORE_SITE_SPECIFIC"
        if yc_counts["IDENTICAL_ACROSS_YEARS"] >= max(hl_c, fq_c) + 2 and hl_c <= 2 and fq_c <= 2
        else "IDENTICAL_POLICY_ALSO_COMMON_AT_HL_FQ"
        if hl_c >= 4 and fq_c >= 4
        else "MIXED_RESULT"
    )

    # Ensure there are no missing or conflicting historical WTH files/configs.
    config_rows = config_audit(snapshots)
    if any(r["training_WTH_missing_count"] or r["validation_WTH_missing_count"] for r in config_rows):
        raise FileNotFoundError("One or more configured historical WTH files are missing")
    # Preserve every audit row, including failures, alongside the derived results.
    out_dir.mkdir(parents=True)
    figures = FIGURE_ROOT
    figures.mkdir(parents=True, exist_ok=True)
    image_rows: list[dict[str, str]] = []
    for site in ("HL", "FQ"):
        for seed in SEEDS:
            if site == "FQ" and seed == 7:
                continue
            image_rows.extend(plot_seed(site, seed, metrics, event_rows, figures / site.lower() / f"seed_{seed}"))
    image_rows.extend(plot_site_summaries(seed_rows, station_rows, figures))

    for metric in metrics:
        metric["management_action_sequence_text"] = next(
            s["management_action_sequence_text"] for s in sequences
            if s["site"] == metric["site"] and int(s["seed"]) == int(metric["seed"]) and int(s["year"]) == int(metric["year"])
        )
    write_csv(out_dir / "metrics_per_seed_year.csv", metrics)
    write_json(out_dir / "metrics_per_seed_year.json", metrics)
    write_csv(out_dir / "management_events_per_seed_year.csv", event_rows)
    write_json(out_dir / "management_events_per_seed_year.json", event_rows)
    sequence_csv_rows = [{
        "site": row["site"], "station_code": row["station_code"], "seed": row["seed"], "year": row["year"],
        "event_sequence_json": json.dumps(row["event_sequence"], separators=(",", ":"), ensure_ascii=False),
        "event_sequence_sha256": row["event_sequence_sha256"],
        "full_daily_action_sequence_json": json.dumps(row["full_daily_action_sequence"], separators=(",", ":"), ensure_ascii=False),
        "full_daily_action_sequence_sha256": row["full_daily_action_sequence_sha256"],
        "management_action_sequence_text": row["management_action_sequence_text"],
        "event_count": row["event_count"], "daily_rows": row["daily_rows"],
    } for row in sequences]
    write_csv(out_dir / "normalized_action_sequence_comparison.csv", sequence_csv_rows)
    write_json(out_dir / "normalized_action_sequences.json", sequences)
    write_csv(out_dir / "seed_classification_summary.csv", seed_rows)
    write_json(out_dir / "seed_classification_summary.json", seed_rows)
    write_csv(out_dir / "station_level_summary.csv", station_rows)
    write_json(out_dir / "station_level_summary.json", station_rows)
    similarity_rows = []
    for row in station_rows:
        matrix = row["pairwise_seed_similarity_matrix"]
        for seed_a in SEEDS:
            for seed_b in SEEDS:
                if seed_a < seed_b and math.isfinite(matrix[seed_a][seed_b]):
                    similarity_rows.append({
                        "site": row["site"], "seed_a": seed_a, "seed_b": seed_b,
                        "mean_annual_event_jaccard": matrix[seed_a][seed_b],
                    })
    write_csv(out_dir / "cross_seed_pairwise_similarity.csv", similarity_rows)
    comparison_rows = seed_rows
    write_csv(out_dir / "site_seed_action_classification.csv", comparison_rows)
    write_csv(out_dir / "mask_and_closure_audit.csv", mask_rows)
    write_csv(out_dir / "configuration_audit.csv", config_rows)
    write_csv(out_dir / "checkpoint_inventory.csv", checkpoint_rows)
    write_csv(out_dir / "training_log_inventory.csv", training_log_rows)
    write_csv(out_dir / "seed_completion.csv", completion_rows)
    write_csv(out_dir / "figure_manifest.csv", image_rows)
    smoke_path = RESULT_ROOT / "smoke_gate_summary.csv"
    smoke_rows = read_csv(smoke_path) if smoke_path.is_file() else []
    metadata = {
        "task": "HL/FQ historical-weather 8-seed PPO baseline and cross-site policy diagnosis",
        "seed_list": SEEDS,
        "validation_years": YEARS,
        "checkpoint_selection": "fixed 100000 environment-step checkpoint",
        "current_weather_only": "historical station-specific WTH; no WGEN/random weather generated",
        "source_mzx_and_weather_hashes_in": "configuration_audit.csv",
        "checkpoint_and_config_hashes_in": "checkpoint_inventory.csv",
        "smoke_gate_summary": smoke_rows,
        "observed_mask_audit_status": "FAIL" if failed_mask_rows else "PASS",
        "observed_mask_audit_failed_year_count": len(failed_mask_rows),
        "yc_inputs": [
            "results/yc_random_weather_ppo/004_15_75mm_all_seed_five_scenario_figures/all_seeds_five_scenario_metrics.csv",
            "results/yc_random_weather_ppo/004_15_75mm_all_seed_five_scenario_figures/all_seeds_five_scenario_management_events.csv",
        ],
        "yc_modified_or_retrained": False,
        "reward_modified": False,
        "action_mask_modified": False,
        "action_space_modified": False,
        "weather_configuration_modified": False,
        "no_nue_or_wue_inference": True,
        "event_classification": {
            "C": "all ten complete nonzero event sequences identical",
            "B": "not C and the modal complete sequence appears in at least 7/10 years",
            "A": "otherwise",
            "site_seed_stable_thresholds": {"SEED_STABLE": ">=0.80 median pairwise per-year Jaccard", "SEED_SENSITIVE": "<=0.35", "MIXED_SEED_BEHAVIOR": "otherwise"},
        },
        "outputs": str(out_dir.relative_to(ROOT).as_posix()),
        "figures": str(FIGURE_ROOT.relative_to(ROOT).as_posix()),
        "configuration_snapshots": {
            site: {
                str(seed): {k: v for k, v in data.items() if k != "config"} | {"config_snapshot": data["config"]}
                for seed, data in seed_map.items()
            }
            for site, seed_map in snapshots.items()
        },
    }
    write_json(out_dir / "run_metadata.json", metadata)
    report = build_report(config_rows, metrics, seed_rows, station_rows, comparison_rows, mask_rows, image_rows, out_dir, smoke_rows)
    report += f"\n> `CROSS_SITE_MULTI_SEED_DIAGNOSIS = {diagnosis}`\n"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({
        "HL_training_seeds_completed": "8/8",
        "FQ_training_seeds_completed": "7/8",
        "HL_station_level_assessment": next(r["site_level_assessment"] for r in station_rows if r["site"] == "HL"),
        "FQ_station_level_assessment": next(r["site_level_assessment"] for r in station_rows if r["site"] == "FQ"),
        "HL_identical_across_years_seeds": sum(r["site"] == "HL" and r["year_to_year_classification"] == "IDENTICAL_ACROSS_YEARS" for r in seed_rows),
        "FQ_identical_across_years_seeds": sum(r["site"] == "FQ" and r["year_to_year_classification"] == "IDENTICAL_ACROSS_YEARS" for r in seed_rows),
        "YC_C_seed_count": yc_counts["IDENTICAL_ACROSS_YEARS"],
        "CROSS_SITE_MULTI_SEED_DIAGNOSIS": diagnosis,
        "figures": len(image_rows),
        "report": report_path.relative_to(ROOT).as_posix(),
        "analysis": out_dir.relative_to(ROOT).as_posix(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
