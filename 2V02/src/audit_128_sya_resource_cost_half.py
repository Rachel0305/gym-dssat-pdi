"""Read-only comparison of 128 reward-scale smoke against 111 forecast baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/128_00_sya_originIC_rolling7d_forecast_resource_cost_half.json"
OUT = ROOT / "benchmark_results/128_00_sya_originIC_rolling7d_forecast_resource_cost_half_smoke2k"
BASE = ROOT / "benchmark_results/111_00_sya_originIC_rolling7d_perfect_forecast_smoke2k_smoke2k"


def entropy(values: pd.Series) -> float:
    p = values.astype(str).value_counts(normalize=True)
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


def summary(path: Path, task: str) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    df["checkpoint_step"] = pd.to_numeric(df["checkpoint_step"], errors="coerce").astype(int)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    return df[df["checkpoint_step"].eq(2000)].copy()


def daily_stats(root: Path) -> dict:
    files = sorted((root / "daily_outputs" / "SYA").glob("*ckpt2000_daily.csv"))
    rows = 0
    positive = 0
    pairs: list[str] = []
    mismatch = forced_noop = safety = 0
    for path in files:
        d = pd.read_csv(path, keep_default_na=False)
        rows += len(d)
        i = pd.to_numeric(d["safe_action_amir"], errors="coerce").fillna(0.0)
        n = pd.to_numeric(d["safe_action_anfer"], errors="coerce").fillna(0.0)
        pairs.extend((i.round(6).astype(str) + "/" + n.round(6).astype(str)).tolist())
        positive += int(((i > 1e-9) | (n > 1e-9)).sum())
        mismatch += int((pd.to_numeric(d["raw_action_amir"], errors="coerce") - i).abs().gt(1e-9).sum())
        forced_noop += int(d["mask_forced_noop"].astype(str).str.lower().eq("true").sum())
        safety += int(d["action_safety_triggered"].astype(str).str.lower().eq("true").sum())
    s = pd.Series(pairs).value_counts(normalize=True) if pairs else pd.Series(dtype=float)
    return {"daily_files": len(files), "daily_rows": rows, "pair_entropy_bits": float(-(s * np.log2(s)).sum()) if len(s) else 0.0, "distinct_action_pairs": int(s.size), "positive_action_rate": float(positive / rows) if rows else 0.0, "raw_safe_irrigation_mismatch": mismatch, "mask_forced_noop": forced_noop, "safety_trigger": safety}


def main() -> int:
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    out_summary = summary(OUT / "evaluation/042_10_checkpoint_validation_summary.csv", "128_00")
    base_summary = summary(BASE / "evaluation/111_00_checkpoint_validation_summary.csv", "111_00")
    merged = out_summary[["year", "final_grnwt", "total_irrigation", "total_n", "PFP_N"]].merge(base_summary[["year", "final_grnwt", "total_irrigation", "total_n", "PFP_N"]], on="year", suffixes=("_cost_half", "_baseline"))
    merged["yield_delta"] = merged["final_grnwt_cost_half"] - merged["final_grnwt_baseline"]
    merged["irrigation_delta"] = merged["total_irrigation_cost_half"] - merged["total_irrigation_baseline"]
    merged["pfp_n_delta"] = merged["PFP_N_cost_half"] - merged["PFP_N_baseline"]
    merged.to_csv(OUT / "128_00_vs_111_year_comparison.csv", index=False, encoding="utf-8-sig")
    stats = {"baseline_forecast_111": daily_stats(BASE), "resource_cost_half_128": daily_stats(OUT)}
    means = {"cost_half_yield_mean": float(out_summary["final_grnwt"].mean()), "baseline_yield_mean": float(base_summary["final_grnwt"].mean()), "yield_delta_mean": float(merged["yield_delta"].mean()), "cost_half_irrigation_mean": float(out_summary["total_irrigation"].mean()), "baseline_irrigation_mean": float(base_summary["total_irrigation"].mean()), "cost_half_pfp_n_mean": float(out_summary["PFP_N"].mean()), "baseline_pfp_n_mean": float(base_summary["PFP_N"].mean()), "pfp_n_delta_mean": float(merged["pfp_n_delta"].mean())}
    sy = merged[merged["year"].eq(2017)].iloc[0].to_dict()
    result = {"status": "completed_read_only", "reward_intervention": cfg["reward_intervention"], "means": means, "sy2017": sy, "daily_stats": stats, "wp_et_available": False, "next_step_allowed": False}
    (OUT / "128_00_audit_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
