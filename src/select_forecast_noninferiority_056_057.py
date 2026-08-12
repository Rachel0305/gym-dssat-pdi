"""Select forecast checkpoints only if they are non-inferior to accepted PPO.

This script is intentionally file-path driven.  It does not know which figure
set is "nice"; it only compares checkpoint summaries against a user-supplied
accepted PPO row and writes a transparent gate table.

Example:

python src/select_forecast_noninferiority_056_057.py `
  --site SYA `
  --forecast-summary benchmark_results/056_00_sya_originIC_forecast_engineered_maskableppo/evaluation/056_00_checkpoint_validation_summary.csv `
  --baseline-yield 10072.7 `
  --baseline-wp-et 2.110 `
  --baseline-pfp-n 42.66 `
  --out benchmark_results/056_00_sya_originIC_forecast_engineered_maskableppo/audits/056_00_noninferiority_gate.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lower = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise KeyError(f"None of these columns were found: {candidates}. Available: {list(df.columns)}")


def summarize_checkpoint_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ckpt_col = _pick_column(df, ["checkpoint_step", "checkpoint", "step"])
    yield_col = _pick_column(df, ["yield_kg_ha", "yield", "HWAM", "hwan_kg_ha"])
    wp_col = _pick_column(df, ["WP_ET", "wp_et", "water_productivity_et"])
    pfp_col = _pick_column(df, ["PFP_N", "pfp_n", "partial_factor_productivity_n"])
    out = (
        df.assign(
            checkpoint_step=pd.to_numeric(df[ckpt_col], errors="coerce"),
            yield_kg_ha=pd.to_numeric(df[yield_col], errors="coerce"),
            WP_ET=pd.to_numeric(df[wp_col], errors="coerce"),
            PFP_N=pd.to_numeric(df[pfp_col], errors="coerce"),
        )
        .dropna(subset=["checkpoint_step"])
        .groupby("checkpoint_step", as_index=False)
        .agg(
            mean_yield_kg_ha=("yield_kg_ha", "mean"),
            mean_WP_ET=("WP_ET", "mean"),
            mean_PFP_N=("PFP_N", "mean"),
            year_count=("checkpoint_step", "size"),
        )
        .sort_values("checkpoint_step")
    )
    out["checkpoint_step"] = out["checkpoint_step"].astype(int)
    return out


def run(args: argparse.Namespace) -> dict:
    summary = summarize_checkpoint_table(args.forecast_summary)
    summary["baseline_yield_kg_ha"] = float(args.baseline_yield)
    summary["baseline_WP_ET"] = float(args.baseline_wp_et)
    summary["baseline_PFP_N"] = float(args.baseline_pfp_n)
    summary["yield_gap_kg_ha"] = summary["mean_yield_kg_ha"] - float(args.baseline_yield)
    summary["WP_ET_gap"] = summary["mean_WP_ET"] - float(args.baseline_wp_et)
    summary["PFP_N_gap"] = summary["mean_PFP_N"] - float(args.baseline_pfp_n)
    summary["yield_noninferior"] = summary["yield_gap_kg_ha"] >= -float(args.yield_tolerance)
    summary["WP_ET_noninferior"] = summary["WP_ET_gap"] >= -float(args.wp_et_tolerance)
    summary["PFP_N_noninferior"] = summary["PFP_N_gap"] >= -float(args.pfp_n_tolerance)
    summary["all_three_noninferior"] = summary["yield_noninferior"] & summary["WP_ET_noninferior"] & summary["PFP_N_noninferior"]
    summary["improves_at_least_one_metric"] = (
        (summary["yield_gap_kg_ha"] > float(args.yield_tolerance))
        | (summary["WP_ET_gap"] > float(args.wp_et_tolerance))
        | (summary["PFP_N_gap"] > float(args.pfp_n_tolerance))
    )
    summary["paper_status"] = np.where(
        summary["all_three_noninferior"] & summary["improves_at_least_one_metric"],
        "main_compare_forecast_vs_no_forecast",
        np.where(summary["all_three_noninferior"], "forecast_acceptable_tie", "reject_forecast_checkpoint"),
    )
    accepted = summary[summary["all_three_noninferior"]].copy()
    if not accepted.empty:
        accepted = accepted.sort_values(
            ["improves_at_least_one_metric", "mean_yield_kg_ha", "mean_WP_ET", "mean_PFP_N"],
            ascending=[False, False, False, False],
        )
        selected = accepted.iloc[0].to_dict()
    else:
        selected = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False, encoding="utf-8-sig")
    payload = {
        "site": args.site,
        "forecast_summary": str(args.forecast_summary),
        "gate_csv": str(args.out),
        "accepted_checkpoint_count": int(summary["all_three_noninferior"].sum()),
        "selected_checkpoint": selected,
    }
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--forecast-summary", type=Path, required=True)
    parser.add_argument("--baseline-yield", type=float, required=True)
    parser.add_argument("--baseline-wp-et", type=float, required=True)
    parser.add_argument("--baseline-pfp-n", type=float, required=True)
    parser.add_argument("--yield-tolerance", type=float, default=25.0)
    parser.add_argument("--wp-et-tolerance", type=float, default=0.005)
    parser.add_argument("--pfp-n-tolerance", type=float, default=0.1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
