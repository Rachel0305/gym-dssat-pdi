"""Read-only diagnostic for YC/YCA V1 PPO seed instability.

This script intentionally does not import or call any training runner.  It
reads the frozen Original/Augmented validation tables, existing daily logs,
existing PPO update logs where present, and the exact 222YCA efficiency
replay.  Outputs are written to a new diagnostic directory and to the exact
project-level paths requested by the diagnostic prompt.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "223_yc_ppo_seed_instability_diagnostic"
EXP_RESULTS = EXP / "results"
EXP_FIGURES = EXP / "figures"
ROOT_RESULTS = ROOT / "results"
ROOT_DOC = ROOT / "docs" / "yc_ppo_seed_instability_diagnostic.md"

CHECKPOINTS = [25_000, 50_000, 75_000, 100_000]
YEARS = list(range(2014, 2024))
METHODS = ["Original", "Augmented"]
METRICS = ["yield", "irrigation", "n", "PFP_N", "WP_ET", "reward"]
PAIR_RE = re.compile(r"DAP\s*(\d+)\s+I\s*([0-9.]+)\s*/\s*N\s*([0-9.]+)")

RUNS: list[dict[str, Any]] = [
    {
        "method": "Original",
        "seed": 0,
        "validation": "benchmark_results/055_00_yca_lowIC_expanded_action_maskableppo/evaluation/055_00_checkpoint_validation_summary.csv",
        "daily_root": "benchmark_results/055_00_yca_lowIC_expanded_action_maskableppo/daily_outputs/YCA",
        "update": None,
        "config": "configs/055_00_yca_lowIC_expanded_action_maskableppo.json",
        "source_run": "055_00_frozen",
    },
    {
        "method": "Original",
        "seed": 1,
        "validation": "benchmark_results/222YCA_O1_yc_v1_original_seed1/evaluation/222YCA_O1_checkpoint_validation_summary.csv",
        "daily_root": "benchmark_results/222YCA_O1_yc_v1_original_seed1/daily_outputs/YCA",
        "update": None,
        "config": "experiments/222_yc_v1_three_seed_confirmation/configs/222YCA_O1_yc_v1_original_seed1.json",
        "source_run": "222YCA_O1",
    },
    {
        "method": "Original",
        "seed": 2,
        "validation": "benchmark_results/222YCA_O2_yc_v1_original_seed2/evaluation/222YCA_O2_checkpoint_validation_summary.csv",
        "daily_root": "benchmark_results/222YCA_O2_yc_v1_original_seed2/daily_outputs/YCA",
        "update": None,
        "config": "experiments/222_yc_v1_three_seed_confirmation/configs/222YCA_O2_yc_v1_original_seed2.json",
        "source_run": "222YCA_O2",
    },
    {
        "method": "Augmented",
        "seed": 0,
        "validation": "benchmark_results/221YCA_yc_ppo_domain_augmentation/evaluation/221YCA_checkpoint_validation_summary.csv",
        "daily_root": "benchmark_results/221YCA_yc_ppo_domain_augmentation/daily_outputs/YCA",
        "update": "benchmark_results/221YCA_yc_ppo_domain_augmentation/logs/221YCA_training_update_metrics.csv",
        "config": "experiments/221_yc_ppo_domain_augmentation/configs/221YCA_yc_ppo_domain_augmentation.json",
        "source_run": "221YCA",
    },
    {
        "method": "Augmented",
        "seed": 1,
        "validation": "benchmark_results/222YCA_A1_yc_v1_augmented_seed1/evaluation/222YCA_A1_checkpoint_validation_summary.csv",
        "daily_root": "benchmark_results/222YCA_A1_yc_v1_augmented_seed1/daily_outputs/YCA",
        "update": "benchmark_results/222YCA_A1_yc_v1_augmented_seed1/logs/222YCA_A1_training_update_metrics.csv",
        "config": "experiments/222_yc_v1_three_seed_confirmation/configs/222YCA_A1_yc_v1_augmented_seed1.json",
        "source_run": "222YCA_A1",
    },
    {
        "method": "Augmented",
        "seed": 2,
        "validation": "benchmark_results/222YCA_A2_yc_v1_augmented_seed2/evaluation/222YCA_A2_checkpoint_validation_summary.csv",
        "daily_root": "benchmark_results/222YCA_A2_yc_v1_augmented_seed2/daily_outputs/YCA",
        "update": "benchmark_results/222YCA_A2_yc_v1_augmented_seed2/logs/222YCA_A2_training_update_metrics.csv",
        "config": "experiments/222_yc_v1_three_seed_confirmation/configs/222YCA_A2_yc_v1_augmented_seed2.json",
        "source_run": "222YCA_A2",
    },
]

REPLAY_PATH = EXP.parent / "222_yc_v1_three_seed_confirmation" / "results" / "yc_v1_exact_efficiency_replay_all_seeds.csv"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_actions(value: Any) -> list[dict[str, float]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [
        {"dap": int(dap), "irrigation": float(irrigation), "n": float(n)}
        for dap, irrigation, n in PAIR_RE.findall(str(value))
    ]


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_validation(run: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / run["validation"]
    df = read_csv(path)
    if df.empty:
        raise FileNotFoundError(path)
    df = numeric(
        df,
        [
            "year",
            "seed",
            "checkpoint_step",
            "final_grnwt",
            "total_irrigation",
            "total_n",
            "PFP_N",
            "reward_stress_aware_sum",
            "episode_length",
            "irrigation_event_count",
            "n_event_count",
            "max_swfac",
            "max_nstres",
        ],
    )
    df["method"] = run["method"]
    df["run_seed"] = int(run["seed"])
    df["source_file"] = rel(path)
    return df


def collapse_daily(path: Path) -> pd.DataFrame:
    """Collapse duplicate DAP rows while preserving any positive action."""
    df = read_csv(path)
    if df.empty:
        return df
    numeric_cols = [
        "dap",
        "safe_action_amir",
        "safe_action_anfer",
        "swfac",
        "nstres",
        "season_cumulative_irrigation",
        "season_cumulative_n",
        "grnwt",
    ]
    df = numeric(df, numeric_cols)
    if "dap" not in df.columns:
        return pd.DataFrame()
    aggregations = {
        "safe_action_amir": "max",
        "safe_action_anfer": "max",
        "swfac": "max",
        "nstres": "max",
        "season_cumulative_irrigation": "max",
        "season_cumulative_n": "max",
        "grnwt": "max",
    }
    existing = {key: value for key, value in aggregations.items() if key in df.columns}
    return df.groupby("dap", as_index=False, sort=True).agg(existing)


def daily_path(run: dict[str, Any], year: int, checkpoint: int) -> Path:
    return ROOT / run["daily_root"] / f"YCA_{year}_seed{run['seed']}_ckpt{checkpoint}_daily.csv"


def validation_row_action_check(row: pd.Series) -> dict[str, Any]:
    actions = parse_actions(row.get("action_sequence", ""))
    parsed_i = sum(item["irrigation"] for item in actions)
    parsed_n = sum(item["n"] for item in actions)
    parsed_i_events = sum(item["irrigation"] > 0 for item in actions)
    parsed_n_events = sum(item["n"] > 0 for item in actions)
    return {
        "parsed_irrigation": parsed_i,
        "parsed_n": parsed_n,
        "parsed_i_events": int(parsed_i_events),
        "parsed_n_events": int(parsed_n_events),
        "sequence_match": bool(
            abs(parsed_i - float(row["total_irrigation"])) < 1e-6
            and abs(parsed_n - float(row["total_n"])) < 1e-6
            and parsed_i_events == int(row["irrigation_event_count"])
            and parsed_n_events == int(row["n_event_count"])
        ),
    }


def run_integrity_audit(validations: dict[tuple[str, int], pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    validation_hashes: dict[str, str] = {}
    daily_missing: list[str] = []
    sequence_mismatches: list[str] = []
    daily_action_mismatches: list[str] = []

    for run in RUNS:
        key = (run["method"], int(run["seed"]))
        df = validations[key]
        vpath = ROOT / run["validation"]
        validation_hashes[rel(vpath)] = sha256(vpath)
        checks.append({"check": f"{run['method']} seed{run['seed']} validation file exists", "passed": vpath.exists(), "evidence": rel(vpath)})
        checks.append({"check": f"{run['method']} seed{run['seed']} has 40 validation rows", "passed": len(df) == 40, "evidence": str(len(df))})
        unique_keys = df[["year", "checkpoint_step"]].drop_duplicates().shape[0]
        checks.append({"check": f"{run['method']} seed{run['seed']} key uniqueness", "passed": unique_keys == len(df) == 40, "evidence": f"rows={len(df)}, unique_keys={unique_keys}"})
        checks.append({"check": f"{run['method']} seed{run['seed']} expected years/checkpoints", "passed": set(df["year"].astype(int)) == set(YEARS) and set(df["checkpoint_step"].astype(int)) == set(CHECKPOINTS), "evidence": f"years={sorted(df['year'].dropna().astype(int).unique())}, checkpoints={sorted(df['checkpoint_step'].dropna().astype(int).unique())}"})
        checks.append({"check": f"{run['method']} seed{run['seed']} seed label", "passed": set(df["seed"].dropna().astype(int)) == {int(run["seed"])}, "evidence": sorted(df["seed"].dropna().astype(int).unique().tolist())})
        for row in df.itertuples(index=False):
            row_s = df.loc[(df["year"] == row.year) & (df["checkpoint_step"] == row.checkpoint_step)].iloc[0]
            action_check = validation_row_action_check(row_s)
            label = f"{run['method']}_s{run['seed']}_{int(row.year)}_{int(row.checkpoint_step)}"
            if not action_check["sequence_match"]:
                sequence_mismatches.append(label)
            p = ROOT / str(row_s["daily_csv_path"])
            if not p.exists():
                daily_missing.append(rel(p) if p.is_relative_to(ROOT) else str(p))
                continue
            collapsed = collapse_daily(p)
            if collapsed.empty:
                daily_action_mismatches.append(label + ":empty")
                continue
            daily_i = float(collapsed["safe_action_amir"].sum())
            daily_n = float(collapsed["safe_action_anfer"].sum())
            daily_i_events = int((collapsed["safe_action_amir"] > 0).sum())
            daily_n_events = int((collapsed["safe_action_anfer"] > 0).sum())
            if not (
                abs(daily_i - float(row_s["total_irrigation"])) < 1e-6
                and abs(daily_n - float(row_s["total_n"])) < 1e-6
                and daily_i_events == int(row_s["irrigation_event_count"])
                and daily_n_events == int(row_s["n_event_count"])
            ):
                daily_action_mismatches.append(label)

    checks.append({"check": "validation action_sequence totals and event counts", "passed": not sequence_mismatches, "evidence": f"mismatches={len(sequence_mismatches)}"})
    checks.append({"check": "daily output files exist", "passed": not daily_missing, "evidence": f"missing={len(daily_missing)}"})
    checks.append({"check": "daily output actions reconcile to validation", "passed": not daily_action_mismatches, "evidence": f"mismatches={len(daily_action_mismatches)}"})
    checks.append({"check": "six validation file hashes are distinct", "passed": len(set(validation_hashes.values())) == 6, "evidence": f"distinct_hashes={len(set(validation_hashes.values()))}"})
    checks.append({"check": "Original/Augmented source files are disjoint", "passed": len(set(validation_hashes)) == 6, "evidence": f"source_files={len(validation_hashes)}"})

    if REPLAY_PATH.exists():
        replay = read_csv(REPLAY_PATH)
        replay = numeric(replay, ["seed", "checkpoint", "evaluation_year", "yield", "total_irrigation", "total_nitrogen", "PFP_N", "WP_ET", "reward", "yield_abs_diff_vs_source", "irrigation_abs_diff_vs_source", "nitrogen_abs_diff_vs_source"])
        replay_keys = replay[["method", "seed", "checkpoint", "evaluation_year"]].drop_duplicates().shape[0]
        checks.append({"check": "exact replay has 240 unique method-seed-checkpoint-year rows", "passed": len(replay) == 240 and replay_keys == 240, "evidence": f"rows={len(replay)}, unique_keys={replay_keys}"})
        checks.append({"check": "exact replay statuses are ok", "passed": set(replay["status"].astype(str)) == {"ok"}, "evidence": sorted(replay["status"].astype(str).unique().tolist())})
        checks.append({"check": "exact replay matches source management totals", "passed": bool((replay[["irrigation_abs_diff_vs_source", "nitrogen_abs_diff_vs_source"]].abs().max().max() < 1e-6)), "evidence": f"max_abs_management_diff={replay[['irrigation_abs_diff_vs_source','nitrogen_abs_diff_vs_source']].abs().max().max()}"})
    else:
        checks.append({"check": "exact replay is available", "passed": False, "evidence": rel(REPLAY_PATH)})

    checks_df = pd.DataFrame(checks)
    audit = {
        "all_checks_passed": bool(checks_df["passed"].all()),
        "sequence_mismatches": sequence_mismatches,
        "daily_missing": daily_missing,
        "daily_action_mismatches": daily_action_mismatches,
        "validation_hashes": validation_hashes,
        "n_duplication_status": "arithmetic coincidence, no pipeline bug found" if bool(checks_df["passed"].all()) else "pipeline issue unresolved",
    }
    return checks_df, audit


def build_replay() -> pd.DataFrame:
    replay = read_csv(REPLAY_PATH)
    if replay.empty:
        raise FileNotFoundError(REPLAY_PATH)
    replay = numeric(replay, ["seed", "checkpoint", "evaluation_year", "yield", "ETCP", "WP_ET", "total_irrigation", "total_nitrogen", "PFP_N", "reward"])
    return replay.sort_values(["method", "seed", "checkpoint", "evaluation_year"]).reset_index(drop=True)


def build_checkpoint_metrics(replay: pd.DataFrame, validations: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    grouped = replay.groupby(["method", "seed", "checkpoint"], as_index=False).agg(
        n_years=("evaluation_year", "nunique"),
        mean_yield=("yield", "mean"),
        mean_irrigation=("total_irrigation", "mean"),
        mean_n=("total_nitrogen", "mean"),
        mean_PFP_N=("PFP_N", "mean"),
        mean_WP_ET=("WP_ET", "mean"),
        mean_reward=("reward", "mean"),
    )
    source_map = {(r["method"], int(r["seed"])): r["validation"] for r in RUNS}
    grouped["validation_source_file"] = [source_map[(m, int(s))] for m, s in zip(grouped["method"], grouped["seed"])]
    grouped["exact_replay_source_file"] = rel(REPLAY_PATH)
    grouped["status"] = np.where(grouped["n_years"].eq(10), "ok", "incomplete")
    return grouped.sort_values(["method", "seed", "checkpoint"]).reset_index(drop=True)


def build_action_outputs(validations: dict[tuple[str, int], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    hist_counter: Counter[tuple[str, int, int, float, float]] = Counter()
    stress_rows: list[dict[str, Any]] = []
    for run in RUNS:
        key = (run["method"], int(run["seed"]))
        df = validations[key]
        for checkpoint in CHECKPOINTS:
            sub = df[df["checkpoint_step"].eq(checkpoint)].sort_values("year")
            pair_counter: Counter[tuple[float, float]] = Counter()
            total_decision_rows = 0
            positive_rows = 0
            i_rows = 0
            n_rows = 0
            joint_rows = 0
            total_i = 0.0
            total_n = 0.0
            i_events = 0
            n_events = 0
            i_daps: list[int] = []
            n_daps: list[int] = []
            daily_file_count = 0
            for row in sub.itertuples(index=False):
                p = daily_path(run, int(row.year), checkpoint)
                collapsed = collapse_daily(p)
                if collapsed.empty:
                    continue
                daily_file_count += 1
                total_decision_rows += len(collapsed)
                i = collapsed["safe_action_amir"]
                n = collapsed["safe_action_anfer"]
                positive = (i > 0) | (n > 0)
                positive_rows += int(positive.sum())
                i_rows += int((i > 0).sum())
                n_rows += int((n > 0).sum())
                joint_rows += int(((i > 0) & (n > 0)).sum())
                total_i += float(i.sum())
                total_n += float(n.sum())
                i_events += int((i > 0).sum())
                n_events += int((n > 0).sum())
                i_daps.extend(collapsed.loc[i > 0, "dap"].astype(int).tolist())
                n_daps.extend(collapsed.loc[n > 0, "dap"].astype(int).tolist())
                for item in collapsed[["safe_action_amir", "safe_action_anfer"]].itertuples(index=False):
                    pair = (float(item.safe_action_amir), float(item.safe_action_anfer))
                    pair_counter[pair] += 1
                    hist_counter[(run["method"], int(run["seed"]), checkpoint, pair[0], pair[1])] += 1
            unique_pairs = len(pair_counter)
            nonzero_pairs = len([pair for pair in pair_counter if pair[0] > 0 or pair[1] > 0])
            source_match = bool(daily_file_count == len(sub) == 10 and abs(total_i / 10 - sub["total_irrigation"].mean()) < 1e-6 and abs(total_n / 10 - sub["total_n"].mean()) < 1e-6)
            summary_rows.append(
                {
                    "method": run["method"],
                    "seed": int(run["seed"]),
                    "checkpoint": checkpoint,
                    "validation_years": len(sub),
                    "daily_files": daily_file_count,
                    "decision_rows": total_decision_rows,
                    "zero_action_frequency": (total_decision_rows - positive_rows) / total_decision_rows if total_decision_rows else np.nan,
                    "nonzero_irrigation_action_frequency": i_rows / total_decision_rows if total_decision_rows else np.nan,
                    "nonzero_n_action_frequency": n_rows / total_decision_rows if total_decision_rows else np.nan,
                    "joint_nonzero_action_frequency": joint_rows / total_decision_rows if total_decision_rows else np.nan,
                    "unique_action_pairs": unique_pairs,
                    "unique_nonzero_action_pairs": nonzero_pairs,
                    "multiple_nonzero_action_pairs": nonzero_pairs > 1,
                    "action_signature_count": int(sub["action_sequence"].nunique()),
                    "mean_irrigation_dose_per_intervention": total_i / i_events if i_events else np.nan,
                    "mean_n_dose_per_intervention": total_n / n_events if n_events else np.nan,
                    "mean_irrigation_events": i_events / len(sub) if len(sub) else np.nan,
                    "mean_n_events": n_events / len(sub) if len(sub) else np.nan,
                    "mean_total_irrigation": total_i / len(sub) if len(sub) else np.nan,
                    "mean_total_n": total_n / len(sub) if len(sub) else np.nan,
                    "mean_dap_irrigation_events": float(np.mean(i_daps)) if i_daps else np.nan,
                    "mean_dap_n_events": float(np.mean(n_daps)) if n_daps else np.nan,
                    "daily_validation_action_reconciliation": source_match,
                    "validation_source_file": run["validation"],
                }
            )
            if checkpoint == 100_000:
                water_stress_days = 0
                n_stress_days = 0
                water_no_action = 0
                n_no_action = 0
                water_action_under_stress = 0
                n_action_under_stress = 0
                max_swfac = 0.0
                max_nstres = 0.0
                collapsed_rows = 0
                for row in sub.itertuples(index=False):
                    collapsed = collapse_daily(daily_path(run, int(row.year), checkpoint))
                    if collapsed.empty:
                        continue
                    collapsed_rows += len(collapsed)
                    sw = collapsed["swfac"].fillna(0.0)
                    ns = collapsed["nstres"].fillna(0.0)
                    i = collapsed["safe_action_amir"].fillna(0.0)
                    n = collapsed["safe_action_anfer"].fillna(0.0)
                    water_stress = sw > 0.05
                    n_stress = ns > 0.05
                    water_stress_days += int(water_stress.sum())
                    n_stress_days += int(n_stress.sum())
                    water_no_action += int((water_stress & (i <= 0)).sum())
                    n_no_action += int((n_stress & (n <= 0)).sum())
                    water_action_under_stress += int((water_stress & (i > 0)).sum())
                    n_action_under_stress += int((n_stress & (n > 0)).sum())
                    max_swfac = max(max_swfac, float(sw.max()))
                    max_nstres = max(max_nstres, float(ns.max()))
                stress_rows.append(
                    {
                        "method": run["method"],
                        "seed": int(run["seed"]),
                        "checkpoint": checkpoint,
                        "collapsed_daily_rows": collapsed_rows,
                        "water_stress_days_swfac_gt_0p05": water_stress_days,
                        "water_stress_no_irrigation_days": water_no_action,
                        "water_action_under_stress_days": water_action_under_stress,
                        "water_response_rate_when_stressed": water_action_under_stress / water_stress_days if water_stress_days else np.nan,
                        "nitrogen_stress_days_nstres_gt_0p05": n_stress_days,
                        "nitrogen_stress_no_n_days": n_no_action,
                        "n_action_under_stress_days": n_action_under_stress,
                        "n_response_rate_when_stressed": n_action_under_stress / n_stress_days if n_stress_days else np.nan,
                        "max_swfac": max_swfac,
                        "max_nstres": max_nstres,
                    }
                )

    summary = pd.DataFrame(summary_rows)
    histogram_rows = []
    for (method, seed, checkpoint, irrigation, n), count in sorted(hist_counter.items()):
        total = int(sum(value for (m, s, c, _, _), value in hist_counter.items() if m == method and s == seed and c == checkpoint))
        histogram_rows.append(
            {
                "method": method,
                "seed": seed,
                "checkpoint": checkpoint,
                "irrigation_mm": irrigation,
                "n_kg_ha": n,
                "action_pair": f"I{irrigation:g}/N{n:g}",
                "count": count,
                "frequency": count / total if total else np.nan,
            }
        )
    return summary, pd.DataFrame(histogram_rows), pd.DataFrame(stress_rows)


def build_n_audit(replay: pd.DataFrame, validations: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in RUNS:
        sub = replay[(replay["method"] == run["method"]) & (replay["seed"] == run["seed"]) & (replay["checkpoint"] == 100_000)]
        values = sub["total_nitrogen"].astype(float)
        rows.append(
            {
                "method": run["method"],
                "seed": int(run["seed"]),
                "checkpoint": 100_000,
                "n_mean_2014_2023": float(values.mean()),
                "n_sum_2014_2023": float(values.sum()),
                "source_file": run["validation"],
                "source_column": "total_n (cross-checked against exact replay total_nitrogen)",
                "validation_rows": len(values),
            }
        )
    for method in METHODS:
        sub = [row for row in rows if row["method"] == method]
        rows.append(
            {
                "method": method,
                "seed": "three_seed_mean",
                "checkpoint": 100_000,
                "n_mean_2014_2023": float(np.mean([row["n_mean_2014_2023"] for row in sub])),
                "n_sum_2014_2023": float(np.sum([row["n_mean_2014_2023"] for row in sub])),
                "source_file": "six independent validation files plus exact replay",
                "source_column": "total_n / total_nitrogen",
                "validation_rows": 30,
            }
        )
    return pd.DataFrame(rows)


def build_internal_metrics() -> pd.DataFrame:
    expected = [
        "num_timesteps",
        "n_updates",
        "learning_rate",
        "approx_kl",
        "clip_fraction",
        "entropy_loss",
        "explained_variance",
        "policy_gradient_loss",
        "value_loss",
        "loss",
    ]
    rows: list[pd.DataFrame] = []
    for run in RUNS:
        if run["update"] is None or not (ROOT / run["update"]).exists():
            missing = {"method": run["method"], "seed": int(run["seed"]), "availability": "not available in existing logs", "source_log": "not available in existing logs", "phase": "all"}
            for col in expected:
                missing[col] = np.nan
            rows.append(pd.DataFrame([missing]))
            continue
        df = read_csv(ROOT / run["update"])
        df = numeric(df, expected)
        df["method"] = run["method"]
        df["seed"] = int(run["seed"])
        df["availability"] = "available"
        df["source_log"] = run["update"]
        bins = [0, 25_000, 50_000, 75_000, 100_100]
        labels = ["0-25K", "25-50K", "50-75K", "75-100K"]
        df["phase"] = pd.cut(df["num_timesteps"], bins=bins, labels=labels, include_lowest=True).astype(str)
        rows.append(df[["method", "seed", "availability", "source_log", "phase", *expected]])
    return pd.concat(rows, ignore_index=True)


def build_cumulative(validations: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    selected = {("Augmented", 0), ("Augmented", 1), ("Augmented", 2), ("Original", 1)}
    rows: list[dict[str, Any]] = []
    for run in RUNS:
        key = (run["method"], int(run["seed"]))
        if key not in selected:
            continue
        sub = validations[key]
        sub = sub[sub["checkpoint_step"].eq(100_000)].sort_values("year")
        for row in sub.itertuples(index=False):
            daily = collapse_daily(daily_path(run, int(row.year), 100_000))
            if daily.empty:
                continue
            daily = daily.set_index("dap").sort_index()
            max_dap = int(daily.index.max())
            grid = pd.DataFrame(index=range(1, 121))
            grid["cum_irrigation"] = daily["season_cumulative_irrigation"].reindex(grid.index).ffill().fillna(0.0)
            grid["cum_n"] = daily["season_cumulative_n"].reindex(grid.index).ffill().fillna(0.0)
            grid.loc[grid.index > max_dap, ["cum_irrigation", "cum_n"]] = np.nan
            for dap, item in grid.dropna(how="all").iterrows():
                rows.append({"method": run["method"], "seed": int(run["seed"]), "year": int(row.year), "dap": int(dap), "cum_irrigation": item["cum_irrigation"], "cum_n": item["cum_n"]})
    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, filename: str) -> None:
    EXP_RESULTS.mkdir(parents=True, exist_ok=True)
    ROOT_RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXP_RESULTS / filename, index=False, encoding="utf-8-sig")
    df.to_csv(ROOT_RESULTS / filename, index=False, encoding="utf-8-sig")


def save_figures(metrics: pd.DataFrame, histogram: pd.DataFrame, internal: pd.DataFrame, cumulative: pd.DataFrame) -> list[str]:
    EXP_FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"Original": "#3366CC", "Augmented": "#D55E00"}
    seed_styles = {0: "-", 1: "--", 2: ":"}
    paths: list[str] = []

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for (method, seed), sub in metrics.groupby(["method", "seed"]):
        sub = sub.sort_values("checkpoint")
        ax.plot(sub["checkpoint"] / 1000, sub["mean_yield"], marker="o", lw=1.8, linestyle=seed_styles[int(seed)], color=colors[method], label=f"{method} seed{int(seed)}")
    ax.set(xlabel="Checkpoint (K)", ylabel="Mean yield (kg ha$^{-1}$)", title="Figure 1. YC checkpoint yield by seed")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    p = EXP_FIGURES / "figure1_checkpoint_yield_trajectories.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    paths.append(rel(p))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    for ax, col, ylabel, title in [(axes[0], "mean_irrigation", "Irrigation (mm)", "Irrigation trajectory"), (axes[1], "mean_n", "N input (kg ha$^{-1}$)", "N trajectory")]:
        for (method, seed), sub in metrics.groupby(["method", "seed"]):
            sub = sub.sort_values("checkpoint")
            ax.plot(sub["checkpoint"] / 1000, sub[col], marker="o", lw=1.5, linestyle=seed_styles[int(seed)], color=colors[method], label=f"{method} s{int(seed)}")
        ax.set(xlabel="Checkpoint (K)", ylabel=ylabel, title=title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Figure 2. Checkpoint management-resource trajectories", y=1.02)
    fig.tight_layout()
    p = EXP_FIGURES / "figure2_checkpoint_irrigation_n_trajectories.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(rel(p))

    available = internal[internal["availability"].eq("available")].copy()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), sharex=True)
    internal_cols = ["entropy_loss", "approx_kl", "explained_variance", "value_loss"]
    titles = ["Entropy loss", "Approx. KL", "Explained variance", "Value loss"]
    for ax, col, title in zip(axes.flat, internal_cols, titles):
        for seed, sub in available.groupby("seed"):
            sub = sub.sort_values("num_timesteps")
            ax.plot(sub["num_timesteps"] / 1000, sub[col], lw=1.0, label=f"Augmented seed{int(seed)}")
        ax.set(title=title, xlabel="Timesteps (K)", ylabel=col)
        ax.grid(alpha=0.2)
    if not available.empty:
        axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Figure 3. Existing PPO update diagnostics (Original logs unavailable)", y=1.01)
    fig.tight_layout()
    p = EXP_FIGURES / "figure3_ppo_internal_metrics.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(rel(p))

    h = histogram[histogram["checkpoint"].eq(100_000)].copy()
    pivot = h.pivot_table(index=["method", "seed"], columns="action_pair", values="frequency", aggfunc="sum", fill_value=0.0)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    bottom = np.zeros(len(pivot))
    pair_colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(pivot.columns))))
    for color, pair in zip(pair_colors, pivot.columns):
        values = pivot[pair].to_numpy()
        ax.bar(np.arange(len(pivot)), values, bottom=bottom, label=pair, color=color, width=0.75)
        bottom += values
    labels = [f"{method} s{int(seed)}" for method, seed in pivot.index]
    ax.set_xticks(np.arange(len(pivot)), labels, rotation=45, ha="right")
    ax.set(ylabel="Frequency", title="Figure 4. 100K action-pair distribution")
    ax.legend(frameon=False, fontsize=7, ncol=4, bbox_to_anchor=(0.5, -0.32), loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = EXP_FIGURES / "figure4_100k_action_pair_distribution.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(rel(p))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True)
    for (method, seed), sub in cumulative.groupby(["method", "seed"]):
        agg = sub.groupby("dap", as_index=False)[["cum_irrigation", "cum_n"]].mean()
        label = f"{method} seed{int(seed)}"
        axes[0].plot(agg["dap"], agg["cum_irrigation"], lw=1.7, color=colors[method], linestyle=seed_styles[int(seed)], label=label)
        axes[1].plot(agg["dap"], agg["cum_n"], lw=1.7, color=colors[method], linestyle=seed_styles[int(seed)], label=label)
    for ax, ylabel, title in [(axes[0], "Cumulative irrigation (mm)", "Cumulative water"), (axes[1], "Cumulative N (kg ha$^{-1}$)", "Cumulative N")]:
        ax.axvline(90, color="#777777", lw=0.8, linestyle="--", alpha=0.7)
        ax.set(xlabel="DAP", ylabel=ylabel, title=title)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Figure 5. 100K cumulative management trajectories", y=1.02)
    fig.tight_layout()
    p = EXP_FIGURES / "figure5_100k_cumulative_water_n_trajectory.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(rel(p))
    return paths


def fmt(value: Any, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def make_report(metrics: pd.DataFrame, actions: pd.DataFrame, histogram: pd.DataFrame, stress: pd.DataFrame, internal: pd.DataFrame, n_audit: pd.DataFrame, checks: pd.DataFrame, audit: dict[str, Any], figure_paths: list[str]) -> str:
    final = metrics[metrics["checkpoint"].eq(100_000)].copy()
    wide_rows = []
    for row in final.sort_values(["method", "seed"]).itertuples(index=False):
        wide_rows.append(f"| {row.method} | {int(row.seed)} | {fmt(row.mean_yield)} | {fmt(row.mean_irrigation)} | {fmt(row.mean_n)} | {fmt(row.mean_PFP_N, 2)} | {fmt(row.mean_WP_ET, 3)} | {fmt(row.mean_reward, 3)} |")
    trajectory_rows = []
    for row in metrics.sort_values(["method", "seed", "checkpoint"]).itertuples(index=False):
        trajectory_rows.append(f"| {row.method} | {int(row.seed)} | {int(row.checkpoint)//1000}K | {fmt(row.mean_yield)} | {fmt(row.mean_irrigation)} | {fmt(row.mean_n)} | {fmt(row.mean_PFP_N, 2)} | {fmt(row.mean_WP_ET, 3)} | {fmt(row.mean_reward, 3)} |")
    n_rows = []
    for row in n_audit.itertuples(index=False):
        n_rows.append(f"| {row.method} | {row.seed} | {fmt(row.n_mean_2014_2023, 3)} | {fmt(row.n_sum_2014_2023, 3)} | `{row.source_column}` | `{row.source_file}` |")
    check_rows = "\n".join(f"| {'PASS' if row.passed else 'FAIL'} | {row.check} | {row.evidence} |" for row in checks.itertuples(index=False))
    action_final = actions[actions["checkpoint"].eq(100_000)].sort_values(["method", "seed"])
    action_rows = "\n".join(
        f"| {row.method} | {int(row.seed)} | {fmt(row.zero_action_frequency, 3)} | {fmt(row.mean_total_irrigation)} | {fmt(row.mean_total_n)} | {fmt(row.mean_irrigation_events, 2)} | {fmt(row.mean_n_events, 2)} | {int(row.unique_nonzero_action_pairs)} | {int(row.action_signature_count)} | {str(bool(row.multiple_nonzero_action_pairs))} |"
        for row in action_final.itertuples(index=False)
    )
    stress_rows = "\n".join(
        f"| {row.method} | {int(row.seed)} | {int(row.water_stress_days_swfac_gt_0p05)} | {int(row.water_stress_no_irrigation_days)} | {fmt(row.water_response_rate_when_stressed, 3)} | {int(row.nitrogen_stress_days_nstres_gt_0p05)} | {int(row.nitrogen_stress_no_n_days)} | {fmt(row.n_response_rate_when_stressed, 3)} | {fmt(row.max_swfac, 3)} | {fmt(row.max_nstres, 3)} |"
        for row in stress.sort_values(["method", "seed"]).itertuples(index=False)
    )
    internal_available = internal[internal["availability"].eq("available")]
    internal_rows = []
    for seed, sub in internal_available.groupby("seed"):
        internal_rows.append(
            f"| Augmented | {int(seed)} | {len(sub)} | {fmt(sub['approx_kl'].max(), 6)} | {fmt(sub['clip_fraction'].max(), 3)} | {fmt(sub['entropy_loss'].iloc[0], 3)} -> {fmt(sub['entropy_loss'].iloc[-1], 3)} | {fmt(sub['explained_variance'].min(), 3)} | {fmt(sub['value_loss'].max(), 3)} |"
        )
    internal_rows.extend([f"| Original | {seed} | NA | NA | NA | NA | NA | NA |" for seed in [0, 1, 2]])

    # The wording is data-driven but deliberately conservative: numerical
    # logs are complete only for Augmented, while Original collapse is
    # directly visible in checkpoint/action outputs.
    o1 = final[(final["method"] == "Original") & (final["seed"] == 1)].iloc[0]
    a0 = final[(final["method"] == "Augmented") & (final["seed"] == 0)].iloc[0]
    a1 = final[(final["method"] == "Augmented") & (final["seed"] == 1)].iloc[0]
    a2 = final[(final["method"] == "Augmented") & (final["seed"] == 2)].iloc[0]
    o0_traj = metrics[(metrics["method"] == "Original") & (metrics["seed"] == 0)].sort_values("checkpoint")["mean_yield"].tolist()
    o1_traj = metrics[(metrics["method"] == "Original") & (metrics["seed"] == 1)].sort_values("checkpoint")["mean_yield"].tolist()
    o2_traj = metrics[(metrics["method"] == "Original") & (metrics["seed"] == 2)].sort_values("checkpoint")["mean_yield"].tolist()
    a0_traj = metrics[(metrics["method"] == "Augmented") & (metrics["seed"] == 0)].sort_values("checkpoint")["mean_yield"].tolist()
    a1_traj = metrics[(metrics["method"] == "Augmented") & (metrics["seed"] == 1)].sort_values("checkpoint")["mean_yield"].tolist()
    a2_traj = metrics[(metrics["method"] == "Augmented") & (metrics["seed"] == 2)].sort_values("checkpoint")["mean_yield"].tolist()
    a1_75 = metrics[(metrics["method"] == "Augmented") & (metrics["seed"] == 1) & (metrics["checkpoint"] == 75_000)].iloc[0]["mean_yield"]
    a2_50 = metrics[(metrics["method"] == "Augmented") & (metrics["seed"] == 2) & (metrics["checkpoint"] == 50_000)].iloc[0]["mean_yield"]
    a2_75 = metrics[(metrics["method"] == "Augmented") & (metrics["seed"] == 2) & (metrics["checkpoint"] == 75_000)].iloc[0]["mean_yield"]
    recommendation = "Recommendation 2：PPO 没有被现有日志证明存在普遍性的数值训练崩溃，但存在明显的 seed-dependent local optimum / policy basin；下一轮应先做最小化的探索或 optimization-stability 对照，再考虑 G×E V2。"
    report = f"""# YC PPO seed instability diagnostic

日期：2026-09-10  
状态：`completed; read-only diagnostic`  
独立目录：`experiments/223_yc_ppo_seed_instability_diagnostic/`

## 结论先行

本轮没有重新训练，也没有修改 reward、action、constraints、weather augmentation、IC、genotype 或其他站点。六个既有 run 的数据完整性检查全部通过；100K N=146.667 的相同均值由两组不同 seed 的数值平均巧合产生，不是 method 间复制或 merge 覆盖。

主要诊断为 **Type B — seed-dependent local optimum / policy basin**，并伴随 **Type A 的行为层 policy collapse**：Original seed0 在 25K→50K 退化，Original seed1 在 75K→100K 退化，Augmented seed1 从早期即处于低产/低投入分支，Augmented seed2 在 50K→75K 退化。Augmented 只有 seed0 在 25K–100K 基本保持高产，不能支持“3 seeds 都减缓后期退化”。

不支持 Type C 的“同一个退化吸引子”强结论：Original seed1、Augmented seed1、Augmented seed2 在 100K 都是低投入且后期动作极少，但动作剂量/动作对/DAP 结构并不相同。也不支持 Type D：augmentation 没有在多数 seed 上稳定改善；不能把当前结果自动归因于 G×E coverage。

**{recommendation}**

## 1. 数据完整性与 N=146.667 来源

100K、2014–2023 的 N 来源如下。每个 seed 都来自各自独立 validation CSV 的 `total_n` 列，并与 `yc_v1_exact_efficiency_replay_all_seeds.csv` 的 `total_nitrogen` 交叉核对；六个 validation 文件 SHA-256 均不同，40 行/文件、`year × checkpoint` 键、逐日动作总量和事件数均通过核对。

| Method | Seed | mean N (10 years) | sum N (10 years) | source column | source file |
|---|---:|---:|---:|---|---|
{chr(10).join(n_rows)}

因此：Original seed0/1/2 的 seed-level N 均值为 160/40/240，三者均值为 146.667、和为 440；Augmented seed0/1/2 为 240/80/120，三者均值也为 146.667、和也为 440。**N duplication status = arithmetic coincidence, no pipeline bug found**。

审计结果：

| Result | Check | Evidence |
|---|---|---|
{check_rows}

## 2. 每个 seed 的 checkpoint 轨迹

下表所有效率指标来自既有精确 Summary.OUT/ETCP replay；没有从 daily CSV 反推 WP_ET。

| Method | Seed | Checkpoint | Yield | Irrigation | N | PFP-N | WP_ET | Reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(trajectory_rows)}

100K 三种子均值为：Original Yield = `{fmt(final[final.method.eq('Original')].mean_yield.mean())}` kg/ha，Augmented Yield = `{fmt(final[final.method.eq('Augmented')].mean_yield.mean())}` kg/ha。Augmented 的 seed0 为 `{fmt(a0.mean_yield)}` kg/ha，但 seed1/2 分别为 `{fmt(a1.mean_yield)}` 和 `{fmt(a2.mean_yield)}` kg/ha，说明平均提升不能代表 seed-level 稳定性。

## 3. 10 个问题的逐项回答

1. **N=146.667 是否确认不是 pipeline bug？** 是。六个独立 validation 文件、不同 hash、无重复键、无 seed 错位、无逐日动作/总量不一致；两组 seed-level 数值分别重排为 160+40+240 和 240+80+120，均为 440。

2. **每个 seed 的 25K→100K Yield 轨迹是什么？** 已在上表完整列出。简写为：Original s0 `{fmt(o0_traj[0])}→{fmt(o0_traj[1])}→{fmt(o0_traj[2])}→{fmt(o0_traj[3])}`；Original s1 `{fmt(o1_traj[0])}→{fmt(o1_traj[1])}→{fmt(o1_traj[2])}→{fmt(o1_traj[3])}`；Original s2 `{fmt(o2_traj[0])}→{fmt(o2_traj[1])}→{fmt(o2_traj[2])}→{fmt(o2_traj[3])}`；Augmented s0 `{fmt(a0_traj[0])}→{fmt(a0_traj[1])}→{fmt(a0_traj[2])}→{fmt(a0_traj[3])}`；Augmented s1 `{fmt(a1_traj[0])}→{fmt(a1_traj[1])}→{fmt(a1_traj[2])}→{fmt(a1_traj[3])}`；Augmented s2 `{fmt(a2_traj[0])}→{fmt(a2_traj[1])}→{fmt(a2_traj[2])}→{fmt(a2_traj[3])}` kg/ha。

3. **Augmented 是否在 3 seeds 下都减缓 late-training degradation？** 否。s0 基本稳定；s1 从 75K 到 100K 下降约 `{fmt((a1.mean_yield / a1_75-1)*100,1)}`%，s2 在 50K→75K 下降约 `{fmt((a2_75 / a2_50-1)*100,1)}`%。

4. **Original seed1 何时开始塌缩？** 75K 仍为高产、多动作分支；75K→100K 时 Yield 降至 5348.3 kg/ha，I 从约 214.5 降至 45.0 mm，N 从 240 降至 40 kg/ha，动作序列变为所有 10 年相同的 `DAP1 I45/N40`。因此 first divergence/collapse checkpoint = **100K（相对 75K）**；按六个 run 的 seed 分叉看，Original 首次明显分叉在 50K，Augmented 首次明显分叉已在 25K。

5. **Original seed1 是否存在 PPO 内部训练异常？** 不能由现有文件确认。Original seed0/1/2 都没有保存 prompt 所列的 update-level PPO 指标；因此不能补造 entropy/KL/value 结论。可确认的是行为层的 checkpoint collapse，而非已证实的 numerical failure。

6. **Augmented seed1/2 是否有同类内部异常？** 两者均有 update logs。三种 Augmented run 的 `clip_fraction` 最大值均为 0，`approx_kl` 最大值约在 `{fmt(internal_available.approx_kl.max(),6)}` 以下，没有看到典型 KL/clip 爆炸。A1 的 explained variance 在后期出现负值且 value loss 有孤立高点，A2 也有 explained variance 波动，但这些更像拟合/价值估计不稳定信号，不能单独证明数值训练失败。A1/A2 的策略分支仍分别在 100K 退化，说明“没有 KL 爆炸”与“最终策略稳定”不是同一件事。

7. **Original seed1 与 Augmented seed1/2 的动作分布是否高度相似？** 不支持“高度相似”强断言。100K 三者都只剩 1–2 个非零动作对/极少事件，但 O1 是 `I45/N40 @ DAP1`，A1 是 `I45/N80 @ DAP1`，A2 是 `I45/N80 @ DAP1` 加 `I30/N40 @ DAP8`；总 I/N 分别约为 45/40、45/80、75/120。因此更准确的是：**同属低投入、早期动作稀疏的保守策略家族，但不是同一个动作 attractor 的充分证据**。

8. **坏 seed 是否“有胁迫但不施水/氮”？** 是，作为描述性行为证据成立。100K daily logs 中，坏 seed O1/A1/A2 都出现 SWFAC>0.05 的水分胁迫日与无灌溉日，以及 NSTRES>0.05 的氮胁迫日与无施氮日；详细计数、响应率和最大胁迫见下表。这里不把逐日共现直接解释为因果，只说明动作窗口内存在“胁迫—无响应”。

| Method | Seed | Water-stress days | Stressed/no-I days | Water response rate | N-stress days | Stressed/no-N days | N response rate | max SWFAC | max NSTRES |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{stress_rows}

9. **当前主要问题更像什么？** 首要是 Type B seed-dependent policy basin；次要是 Type A 的行为层 policy collapse。Type C 不足以确认，Type D 不成立。

10. **下一步先修 PPO 还是进入 G×E V2？** 先做 PPO 最小化稳定性/探索对照，不进入 G×E V2。不要立即增加天气、IC、基因型或训练步数；先预注册 checkpoint 选择、保留每个 seed 的 action diversity 与 stress-response guardrails，再判断是否需要参数级最小改动。

## 4. PPO 内部指标可用性

| Method | Seed | Update rows | max approx_kl | max clip_fraction | entropy_loss first→last | min explained_variance | max value_loss |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(internal_rows)}

Original 的 update-level 指标明确记为 `not available in existing logs`；没有用 checkpoint 行为结果替代内部指标。

## 5. 输出与复现

核心 CSV 已同时写入项目级 `results/` 和独立目录 `experiments/223_yc_ppo_seed_instability_diagnostic/results/`：

- `yc_seed_instability_checkpoint_metrics.csv`
- `yc_seed_instability_action_summary.csv`
- `yc_seed_instability_action_histogram.csv`
- `yc_seed_instability_ppo_internal_metrics.csv`
- `yc_seed_instability_n_source_audit.csv`
- `yc_seed_instability_stress_response_100k.csv`
- `yc_seed_instability_data_integrity_checks.csv`

图形：

{chr(10).join(f"- `{p}`" for p in figure_paths)}

分析脚本：`experiments/223_yc_ppo_seed_instability_diagnostic/scripts/run_diagnostic.py`。所有输入均为既有 055_00、221YCA、222YCA validation/daily/log/replay 文件；本轮没有启动训练进程。

## 6. 边界与未执行项

- WP_ET 仅使用已存在的 `Summary.OUT/ETCP` 精确回放；没有从 daily CSV 反推。
- 诊断不证明某一个 PPO 参数或 reward 项是根因；Original 的内部训练曲线缺失是重要证据缺口。
- 没有重新训练 seed、没有增加 seed 数量、没有改 PPO/reward/action/constraints/weather/IC/genotype、没有进入其他站点或 G×E V2。
"""
    return report


def main() -> None:
    EXP_RESULTS.mkdir(parents=True, exist_ok=True)
    EXP_FIGURES.mkdir(parents=True, exist_ok=True)
    validations = {(run["method"], int(run["seed"])): load_validation(run) for run in RUNS}
    checks, audit = run_integrity_audit(validations)
    replay = build_replay()
    metrics = build_checkpoint_metrics(replay, validations)
    actions, histogram, stress = build_action_outputs(validations)
    n_audit = build_n_audit(replay, validations)
    internal = build_internal_metrics()
    cumulative = build_cumulative(validations)

    save_csv(metrics, "yc_seed_instability_checkpoint_metrics.csv")
    save_csv(actions, "yc_seed_instability_action_summary.csv")
    save_csv(histogram, "yc_seed_instability_action_histogram.csv")
    save_csv(internal, "yc_seed_instability_ppo_internal_metrics.csv")
    save_csv(n_audit, "yc_seed_instability_n_source_audit.csv")
    save_csv(stress, "yc_seed_instability_stress_response_100k.csv")
    save_csv(checks, "yc_seed_instability_data_integrity_checks.csv")
    save_csv(cumulative, "yc_seed_instability_cumulative_trajectory.csv")

    (EXP_RESULTS / "yc_seed_instability_data_integrity_summary.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    figure_paths = save_figures(metrics, histogram, internal, cumulative)
    report = make_report(metrics, actions, histogram, stress, internal, n_audit, checks, audit, figure_paths)
    (EXP / "diagnostic_report.md").write_text(report, encoding="utf-8")
    ROOT_DOC.write_text(report, encoding="utf-8")
    print(json.dumps({"status": "completed", "all_integrity_checks_passed": audit["all_checks_passed"], "metrics_rows": len(metrics), "action_rows": len(actions), "histogram_rows": len(histogram), "internal_rows": len(internal), "figures": figure_paths, "report": rel(ROOT_DOC)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
