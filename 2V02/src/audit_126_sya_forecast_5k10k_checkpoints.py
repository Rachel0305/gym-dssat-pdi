"""Read-only checkpoint summary for experiment 126."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results/126_00_sya_originIC_rolling7d_perfect_forecast_5k10k"
SUMMARY = OUT / "evaluation/126_00_checkpoint_validation_summary.csv"


def main() -> int:
    df = pd.read_csv(SUMMARY, keep_default_na=False)
    for col in ["checkpoint_step", "year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)
    rows = []
    for ckpt, frame in df.groupby("checkpoint_step", sort=True):
        rows.append({
            "checkpoint_step": int(ckpt),
            "n_validation_years": int(len(frame)),
            "yield_mean_kg_ha": float(pd.to_numeric(frame["final_grnwt"], errors="coerce").mean()),
            "irrigation_mean_mm": float(pd.to_numeric(frame["total_irrigation"], errors="coerce").mean()),
            "total_n_mean_kg_ha": float(pd.to_numeric(frame["total_n"], errors="coerce").mean()),
            "pfp_n_mean": float(pd.to_numeric(frame["PFP_N"], errors="coerce").mean()),
            "distinct_action_sequences": int(frame["action_sequence"].astype(str).nunique()),
            "post_dap1_irrigation_events": int(pd.to_numeric(frame["irrigation_event_count"], errors="coerce").sum()),
            "wp_et_available": False,
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "126_00_checkpoint_metrics.csv", index=False, encoding="utf-8-sig")
    sy = df[df["year"].eq(2017)].copy().sort_values("checkpoint_step")
    sy_cols = ["checkpoint_step", "year", "final_grnwt", "total_irrigation", "total_n", "PFP_N", "action_sequence", "irrigation_event_count"]
    sy[sy_cols].to_csv(OUT / "126_00_sy2017_checkpoint_metrics.csv", index=False, encoding="utf-8-sig")
    result = {"status": "completed_read_only", "checkpoint_metrics": "benchmark_results/126_00_sya_originIC_rolling7d_perfect_forecast_5k10k/126_00_checkpoint_metrics.csv", "sy2017_metrics": "benchmark_results/126_00_sya_originIC_rolling7d_perfect_forecast_5k10k/126_00_sy2017_checkpoint_metrics.csv", "wp_et_available": False, "collapse_gate": {"distinct_sequences_5k": int(metrics.loc[metrics["checkpoint_step"].eq(5000), "distinct_action_sequences"].iloc[0]), "distinct_sequences_10k": int(metrics.loc[metrics["checkpoint_step"].eq(10000), "distinct_action_sequences"].iloc[0])}}
    (OUT / "126_00_checkpoint_audit_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
