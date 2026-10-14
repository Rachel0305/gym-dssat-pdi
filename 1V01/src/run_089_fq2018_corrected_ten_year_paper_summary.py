"""Build an isolated, paper-native FQ ten-year summary with corrected 2018.

This task is intentionally read-only with respect to all source result/input
files. It does not train or run DSSAT; it combines already validated CSVs and
recomputes transparent summaries in a new output root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


METRICS = {
    "yield": "grain_yield_kg_ha",
    "WP_ET": "WP_ET_kg_m3",
    "PFP_N": "PFP_N_kg_kg",
}
BASELINE_SCENARIOS = [
    "null",
    "recorded_farmer_template",
    "dssat_auto",
    "official_extension_expert",
]
PAPER_SCENARIOS = BASELINE_SCENARIOS + ["ppo"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(repo_root: Path, rel: str) -> Path:
    path = (repo_root / rel).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing input: {rel} -> {path}")
    return path


def numeric(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy scalars and NaN values to strict JSON values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def normalize_scenario(value: Any) -> str:
    if pd.isna(value) or str(value).strip() in {"", "nan", "None"}:
        return "null"
    return str(value).strip()


def choose_row(df: pd.DataFrame, year: int, scenario: str) -> pd.Series:
    rows = df[(df["year"].astype(int) == year) & (df["scenario"] == scenario)]
    if len(rows) != 1:
        raise AssertionError(f"expected one row for year={year}, scenario={scenario}; got {len(rows)}")
    return rows.iloc[0]


def standard_row(row: pd.Series, scenario: str, source_record: str, weather: str, source_path: str, checkpoint_step: Any = math.nan) -> dict[str, Any]:
    return {
        "station_code": str(row.get("station_code", "FQA")),
        "site": str(row.get("site", "FQ")),
        "year": int(row["year"]),
        "scenario": scenario,
        "checkpoint_step": checkpoint_step,
        "grain_yield_kg_ha": numeric(row.get("grain_yield_kg_ha")),
        "biomass_kg_ha": numeric(row.get("biomass_kg_ha")),
        "actual_irrigation_mm": numeric(row.get("actual_irrigation_mm")),
        "actual_nitrogen_kg_ha": numeric(row.get("actual_nitrogen_kg_ha")),
        "etcp_mm": numeric(row.get("etcp_mm")),
        "WP_ET_kg_m3": numeric(row.get("WP_ET_kg_m3")),
        "PFP_N_kg_kg": numeric(row.get("PFP_N_kg_kg")),
        "season_last_dap": numeric(row.get("season_last_dap")),
        "source_record": source_record,
        "weather_provenance": weather,
        "source_path": source_path,
    }


def load_config(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def build(repo_root: Path, config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    years = [int(y) for y in config["years"]]
    if years != list(range(2014, 2024)):
        raise AssertionError(f"unexpected years: {years}")
    inputs = {name: resolve(repo_root, rel) for name, rel in config["inputs"].items()}
    out_root = (repo_root / config["output_root"]).resolve()
    if dry_run:
        return {
            "task_id": config["task_id"],
            "dry_run": True,
            "inputs_exist": True,
            "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
            "output_root": str(out_root),
            "years": years,
            "paper_scenarios": PAPER_SCENARIOS,
            "will_overwrite_old_outputs": False,
            "will_train": False,
            "will_use_external_n": False,
        }

    baseline = pd.read_csv(inputs["baseline_051_02"])
    baseline["scenario"] = baseline["scenario"].map(normalize_scenario)
    season = pd.read_csv(inputs["season_051_03"])
    season["scenario"] = season["scenario"].map(normalize_scenario)
    corrected_baseline = pd.read_csv(inputs["corrected_baseline_088"])
    corrected_baseline["scenario"] = corrected_baseline["scenario"].map(normalize_scenario)
    corrected_ppo = pd.read_csv(inputs["corrected_ppo_088"])
    corrected_ppo["scenario"] = corrected_ppo["scenario"].map(normalize_scenario)

    baseline_rows: list[dict[str, Any]] = []
    for year in years:
        source_df = corrected_baseline if year == 2018 else baseline
        source_name = "088_corrected2018" if year == 2018 else "051_02_original"
        weather = "corrected_TMIN_abs_isolated_copy" if year == 2018 else "original_FQ_WTH"
        source_rel = config["inputs"]["corrected_baseline_088"] if year == 2018 else config["inputs"]["baseline_051_02"]
        for scenario in BASELINE_SCENARIOS:
            row = choose_row(source_df, year, scenario)
            baseline_rows.append(standard_row(row, scenario, source_name, weather, source_rel))

    ppo_rows: list[dict[str, Any]] = []
    existing_ppo = season[season["scenario"] == "rl_candidate"].copy()
    if len(existing_ppo) != len(years):
        raise AssertionError(f"expected 10 existing PPO rows, got {len(existing_ppo)}")
    for year in years:
        if year == 2018:
            rows = corrected_ppo[
                (corrected_ppo["year"].astype(int) == year)
                & (corrected_ppo["checkpoint_step"].astype(int) == int(config["checkpoint_step"]))
            ]
            if len(rows) != 1:
                raise AssertionError(f"expected one corrected PPO 100K row for 2018, got {len(rows)}")
            row = rows.iloc[0]
            ppo_rows.append(
                standard_row(
                    row,
                    "ppo",
                    "088_corrected2018_ppo_replay",
                    "corrected_TMIN_abs_isolated_copy",
                    config["inputs"]["corrected_ppo_088"],
                    int(config["checkpoint_step"]),
                )
            )
        else:
            row = choose_row(existing_ppo, year, "rl_candidate")
            ppo_rows.append(
                standard_row(
                    row,
                    "ppo",
                    "051_03_existing_ckpt100000",
                    "original_FQ_WTH",
                    config["inputs"]["season_051_03"],
                    int(config["checkpoint_step"]),
                )
            )

    combined = pd.DataFrame(baseline_rows + ppo_rows)
    combined = combined.sort_values(["year", "scenario"]).reset_index(drop=True)
    expected_keys = {(year, scenario) for year in years for scenario in PAPER_SCENARIOS}
    actual_keys = {(int(r.year), str(r.scenario)) for r in combined.itertuples()}
    if actual_keys != expected_keys or len(combined) != 50:
        raise AssertionError(f"combined key/row mismatch: rows={len(combined)}")
    if "dssat_auto_external_n" in set(combined["scenario"]):
        raise AssertionError("external-N scenario leaked into paper table")

    replacement_rows: list[dict[str, Any]] = []
    old_2018_baselines = baseline[baseline["year"].astype(int) == 2018]
    old_2018_baselines = old_2018_baselines.set_index("scenario", drop=False)
    corrected_2018_baselines = corrected_baseline[corrected_baseline["year"].astype(int) == 2018].set_index("scenario", drop=False)
    old_ppo_2018 = choose_row(existing_ppo, 2018, "rl_candidate")
    new_ppo_2018 = choose_row(corrected_ppo[(corrected_ppo["checkpoint_step"].astype(int) == int(config["checkpoint_step"]))], 2018, "rl_candidate")
    old_rows = {scenario: old_2018_baselines.loc[scenario] for scenario in BASELINE_SCENARIOS}
    new_rows = {scenario: corrected_2018_baselines.loc[scenario] for scenario in BASELINE_SCENARIOS}
    old_rows["ppo"] = old_ppo_2018
    new_rows["ppo"] = new_ppo_2018
    replacement_metrics = [
        "grain_yield_kg_ha",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "actual_irrigation_mm",
        "actual_nitrogen_kg_ha",
        "etcp_mm",
        "season_last_dap",
    ]
    for scenario in PAPER_SCENARIOS:
        for metric in replacement_metrics:
            old_value = numeric(old_rows[scenario].get(metric))
            new_value = numeric(new_rows[scenario].get(metric))
            replacement_rows.append(
                {
                    "year": 2018,
                    "scenario": scenario,
                    "metric": metric,
                    "old_value": old_value,
                    "corrected_value": new_value,
                    "delta_corrected_minus_old": (new_value - old_value) if math.isfinite(old_value) and math.isfinite(new_value) else math.nan,
                    "old_source": "051_02/051_03_corrupted_weather_record",
                    "corrected_source": "088_isolated_corrected_weather",
                }
            )
    replacement = pd.DataFrame(replacement_rows)

    means = (
        combined.groupby("scenario", as_index=False)[["grain_yield_kg_ha", "biomass_kg_ha", "actual_irrigation_mm", "actual_nitrogen_kg_ha", "etcp_mm", "WP_ET_kg_m3", "PFP_N_kg_kg"]]
        .mean(numeric_only=True)
        .rename(
            columns={
                "grain_yield_kg_ha": "mean_grain_yield_kg_ha",
                "biomass_kg_ha": "mean_biomass_kg_ha",
                "actual_irrigation_mm": "mean_irrigation_mm",
                "actual_nitrogen_kg_ha": "mean_nitrogen_kg_ha",
                "etcp_mm": "mean_etcp_mm",
                "WP_ET_kg_m3": "mean_WP_ET_kg_m3",
                "PFP_N_kg_kg": "mean_PFP_N_kg_kg",
            }
        )
    )
    means["n_years"] = means["scenario"].map(lambda s: int(combined.loc[combined["scenario"] == s, "year"].nunique()))
    means["n_valid_PFP_N"] = means["scenario"].map(lambda s: int(combined.loc[combined["scenario"] == s, "PFP_N_kg_kg"].notna().sum()))
    means = means[["scenario", "n_years", "n_valid_PFP_N", "mean_grain_yield_kg_ha", "mean_biomass_kg_ha", "mean_irrigation_mm", "mean_nitrogen_kg_ha", "mean_etcp_mm", "mean_WP_ET_kg_m3", "mean_PFP_N_kg_kg"]]

    comparison_rows: list[dict[str, Any]] = []
    for year in years:
        ppo = combined[(combined["year"] == year) & (combined["scenario"] == "ppo")].iloc[0]
        bases = combined[(combined["year"] == year) & (combined["scenario"].isin(BASELINE_SCENARIOS))]
        for metric_name, col in METRICS.items():
            ppo_value = numeric(ppo[col])
            valid = [(str(r.scenario), numeric(getattr(r, col))) for r in bases.itertuples() if math.isfinite(numeric(getattr(r, col)))]
            if math.isfinite(ppo_value) and valid:
                best_scenario, best_value = max(valid, key=lambda x: x[1])
                gap = ppo_value - best_value
                strict_win = bool(gap > float(config["strict_tolerance"]))
                tie = bool(abs(gap) <= float(config["strict_tolerance"]))
            else:
                best_scenario, best_value, gap, strict_win, tie = "", math.nan, math.nan, False, False
            comparison_rows.append(
                {
                    "year": year,
                    "metric": metric_name,
                    "ppo_value": ppo_value,
                    "best_baseline_scenario": best_scenario,
                    "best_baseline_value": best_value,
                    "gap_ppo_minus_best_baseline": gap,
                    "strict_win": strict_win,
                    "tie": tie,
                    "valid_comparison": bool(math.isfinite(ppo_value) and bool(valid)),
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    metric_comparison = (
        comparison.groupby("metric", as_index=False)
        .agg(
            years_compared=("valid_comparison", "sum"),
            strict_wins=("strict_win", "sum"),
            ties=("tie", "sum"),
            mean_gap_ppo_minus_best_baseline=("gap_ppo_minus_best_baseline", "mean"),
            min_gap_ppo_minus_best_baseline=("gap_ppo_minus_best_baseline", "min"),
            max_gap_ppo_minus_best_baseline=("gap_ppo_minus_best_baseline", "max"),
        )
    )
    metric_comparison["note"] = metric_comparison["metric"].map(
        lambda metric: "PFP_N compares only years with nonzero-N baseline PFP_N values; zero-N scenarios are N/A." if metric == "PFP_N" else "Strict PPO win means PPO > best of four native-DSSAT baselines in that year."
    )

    output_files: dict[str, Path] = {}
    if out_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {out_root}")
    (out_root / "tables").mkdir(parents=True, exist_ok=False)
    combined_path = out_root / "tables/089_00_fqa_native_auto_corrected_season_summary_10y.csv"
    replacement_path = out_root / "tables/089_00_fqa_native_auto_corrected_2018_replacement.csv"
    means_path = out_root / "tables/089_00_fqa_native_auto_corrected_site_means.csv"
    yearly_path = out_root / "tables/089_00_fqa_native_auto_corrected_ppo_yearly_comparison.csv"
    metric_path = out_root / "tables/089_00_fqa_native_auto_corrected_ppo_metric_comparison.csv"
    combined.to_csv(combined_path, index=False, float_format="%.10g")
    replacement.to_csv(replacement_path, index=False, float_format="%.10g")
    means.to_csv(means_path, index=False, float_format="%.10g")
    comparison.to_csv(yearly_path, index=False, float_format="%.10g")
    metric_comparison.to_csv(metric_path, index=False, float_format="%.10g")
    for path in [combined_path, replacement_path, means_path, yearly_path, metric_path]:
        output_files[path.name] = path

    result = {
        "task_id": config["task_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "rows": int(len(combined)),
        "years": years,
        "paper_scenarios": PAPER_SCENARIOS,
        "external_n_included": False,
        "training_run": False,
        "replay_run": False,
        "replacement_year": 2018,
        "replacement_rows": int(len(replacement)),
        "ppo_checkpoint_step": int(config["checkpoint_step"]),
        "metric_summary": json_safe(metric_comparison.to_dict(orient="records")),
        "means": json_safe(means.to_dict(orient="records")),
        "source_files": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in inputs.items()},
        "outputs": {name: str(path) for name, path in output_files.items()},
        "old_outputs_preserved": True,
    }
    result_path = out_root / "089_00_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    manifest = {
        **result,
        "output_sha256": {name: sha256_file(path) for name, path in output_files.items()},
        "result_sha256": sha256_file(result_path),
        "correction_manifest": json.loads(inputs["correction_manifest_088"].read_text(encoding="utf-8")),
    }
    manifest_path = out_root / "089_00_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else config_path.parent.parent.resolve()
    result = build(repo_root, load_config(config_path), dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
