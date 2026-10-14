"""Paired E2 forecast versus no-forecast 5K comparison for seeds 0/1/2."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

OUT = ROOT / "benchmark_results/150E2b_sya_paired_forecast_noforecast_5k_seeds"
DOC = ROOT / "docs/2026-08-16_sya_E2_paired_forecast_noforecast_5k_seeds.md"
SEED_RUNS = {
    1: {
        "forecast": ROOT / "benchmark_results/145E2S1_sya_originIC_dual_branch_weather_reader_10k_seed1/evaluation/145E2S1_checkpoint_validation_summary.csv",
        "no_forecast": ROOT / "benchmark_results/148N1_sya_originIC_paired_noforecast_5k_seed1/evaluation/148N1_checkpoint_validation_summary.csv",
    },
    2: {
        "forecast": ROOT / "benchmark_results/146E2S2_sya_originIC_dual_branch_weather_reader_10k_seed2/evaluation/146E2S2_checkpoint_validation_summary.csv",
        "no_forecast": ROOT / "benchmark_results/149N2_sya_originIC_paired_noforecast_5k_seed2/evaluation/149N2_checkpoint_validation_summary.csv",
    },
}
SEED0_EXISTING = ROOT / "benchmark_results/143E2_sya_dual_branch_weather_reader_5k_validation/143E2_5k_vs_noforecast_by_year.csv"


def daily(row: pd.Series) -> pd.DataFrame:
    frame = pd.read_csv(ROOT / str(row["daily_csv_path"]), keep_default_na=False)
    for col in ["dap", "safe_action_amir", "safe_action_anfer"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return frame


def paired_for_seed(seed: int) -> pd.DataFrame:
    specs = SEED_RUNS[seed]
    f = pd.read_csv(specs["forecast"], keep_default_na=False)
    n = pd.read_csv(specs["no_forecast"], keep_default_na=False)
    f = f.loc[pd.to_numeric(f["checkpoint_step"], errors="coerce").eq(5000)].copy()
    n = n.loc[pd.to_numeric(n["checkpoint_step"], errors="coerce").eq(5000)].copy()
    if len(f) != 10 or len(n) != 10 or set(pd.to_numeric(f["seed"], errors="coerce").astype(int)) != {seed} or set(pd.to_numeric(n["seed"], errors="coerce").astype(int)) != {seed}:
        raise RuntimeError(f"unexpected paired rows or seeds for seed {seed}")
    rows = []
    for year in range(2014, 2024):
        fr = f.loc[pd.to_numeric(f["year"], errors="coerce").eq(year)].iloc[0]
        nr = n.loc[pd.to_numeric(n["year"], errors="coerce").eq(year)].iloc[0]
        fd, nd = daily(fr), daily(nr)
        aligned = fd[["dap", "safe_action_amir", "safe_action_anfer"]].merge(nd[["dap", "safe_action_amir", "safe_action_anfer"]], on="dap", suffixes=("_forecast", "_no_forecast"))
        changed = (~np.isclose(aligned["safe_action_amir_forecast"], aligned["safe_action_amir_no_forecast"])) | (~np.isclose(aligned["safe_action_anfer_forecast"], aligned["safe_action_anfer_no_forecast"]))
        rows.append({
            "seed": seed, "year": year,
            "forecast_yield": float(fr["final_grnwt"]), "no_forecast_yield": float(nr["final_grnwt"]), "yield_delta_forecast_minus_no_forecast": float(fr["final_grnwt"] - nr["final_grnwt"]),
            "forecast_irrigation": float(fr["total_irrigation"]), "no_forecast_irrigation": float(nr["total_irrigation"]), "irrigation_delta": float(fr["total_irrigation"] - nr["total_irrigation"]),
            "forecast_n": float(fr["total_n"]), "no_forecast_n": float(nr["total_n"]), "n_delta": float(fr["total_n"] - nr["total_n"]),
            "forecast_pfp_n": float(fr["PFP_N"]), "no_forecast_pfp_n": float(nr["PFP_N"]), "pfp_n_delta": float(fr["PFP_N"] - nr["PFP_N"]),
            "action_change_rate": float(changed.mean()), "action_changed_rows": int(changed.sum()), "aligned_rows": int(len(aligned)),
            "forecast_action_signature_count": int((fd["safe_action_amir"].round(6).astype(str) + "/" + fd["safe_action_anfer"].round(6).astype(str)).nunique()),
            "no_forecast_action_signature_count": int((nd["safe_action_amir"].round(6).astype(str) + "/" + nd["safe_action_anfer"].round(6).astype(str)).nunique()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    existing = pd.read_csv(SEED0_EXISTING, keep_default_na=False)
    existing = existing.rename(columns={
        "final_grnwt_e2": "forecast_yield", "final_grnwt_nof": "no_forecast_yield", "total_irrigation_e2": "forecast_irrigation", "total_irrigation_nof": "no_forecast_irrigation", "total_n_e2": "forecast_n", "total_n_nof": "no_forecast_n", "PFP_N_e2": "forecast_pfp_n", "PFP_N_nof": "no_forecast_pfp_n", "pfp_n_delta_e2_minus_nof": "pfp_n_delta", "action_change_rate_e2_vs_nof": "action_change_rate",
    })
    existing["seed"] = 0
    existing["irrigation_delta"] = existing["forecast_irrigation"] - existing["no_forecast_irrigation"]
    existing["n_delta"] = existing["forecast_n"] - existing["no_forecast_n"]
    existing["action_changed_rows"] = np.nan
    existing["aligned_rows"] = np.nan
    existing["forecast_action_signature_count"] = existing.get("action_sequence_e2", pd.Series(dtype=str)).astype(str).map(lambda x: len([p for p in x.split("; ") if p]))
    existing["no_forecast_action_signature_count"] = existing.get("action_sequence_nof", pd.Series(dtype=str)).astype(str).map(lambda x: len([p for p in x.split("; ") if p]))
    needed = ["seed", "year", "forecast_yield", "no_forecast_yield", "yield_delta_e2_minus_nof", "yield_delta_forecast_minus_no_forecast", "forecast_irrigation", "no_forecast_irrigation", "irrigation_delta", "forecast_n", "no_forecast_n", "n_delta", "forecast_pfp_n", "no_forecast_pfp_n", "pfp_n_delta", "action_change_rate", "action_changed_rows", "aligned_rows", "forecast_action_signature_count", "no_forecast_action_signature_count"]
    existing["yield_delta_forecast_minus_no_forecast"] = existing["yield_delta_e2_minus_nof"]
    seed0 = existing[[c for c in needed if c in existing.columns]].copy()
    frames = [seed0]
    frames.extend(paired_for_seed(seed) for seed in (1, 2))
    comparison = pd.concat(frames, ignore_index=True, sort=False)
    comparison = comparison.sort_values(["seed", "year"]).reset_index(drop=True)
    comparison.to_csv(OUT / "150E2_paired_5k_by_seed_year.csv", index=False, encoding="utf-8-sig")
    summary = comparison.groupby("seed", as_index=False).agg(
        years=("year", "count"), mean_forecast_yield=("forecast_yield", "mean"), mean_no_forecast_yield=("no_forecast_yield", "mean"), mean_yield_delta=("yield_delta_forecast_minus_no_forecast", "mean"), yield_win_years=("yield_delta_forecast_minus_no_forecast", lambda s: int((s > 0).sum())),
        mean_forecast_irrigation=("forecast_irrigation", "mean"), mean_no_forecast_irrigation=("no_forecast_irrigation", "mean"), mean_irrigation_delta=("irrigation_delta", "mean"), mean_forecast_n=("forecast_n", "mean"), mean_no_forecast_n=("no_forecast_n", "mean"), mean_n_delta=("n_delta", "mean"), mean_forecast_pfp_n=("forecast_pfp_n", "mean"), mean_no_forecast_pfp_n=("no_forecast_pfp_n", "mean"), mean_pfp_n_delta=("pfp_n_delta", "mean"), pfp_n_win_years=("pfp_n_delta", lambda s: int((s > 0).sum())), mean_action_change_rate=("action_change_rate", "mean"),
    )
    summary.to_csv(OUT / "150E2_paired_5k_summary_by_seed.csv", index=False, encoding="utf-8-sig")
    overall = {"years": int(len(comparison)), "seeds": int(comparison["seed"].nunique()), "mean_forecast_yield": float(comparison["forecast_yield"].mean()), "mean_no_forecast_yield": float(comparison["no_forecast_yield"].mean()), "mean_yield_delta": float(comparison["yield_delta_forecast_minus_no_forecast"].mean()), "yield_win_seed_years": int((comparison["yield_delta_forecast_minus_no_forecast"] > 0).sum()), "mean_irrigation_delta": float(comparison["irrigation_delta"].mean()), "mean_n_delta": float(comparison["n_delta"].mean()), "mean_forecast_pfp_n": float(comparison["forecast_pfp_n"].mean()), "mean_no_forecast_pfp_n": float(comparison["no_forecast_pfp_n"].mean()), "mean_pfp_n_delta": float(comparison["pfp_n_delta"].mean()), "pfp_n_win_seed_years": int((comparison["pfp_n_delta"] > 0).sum()), "positive_yield_delta_seeds": int((summary["mean_yield_delta"] > 0).sum()), "positive_pfp_delta_seeds": int((summary["mean_pfp_n_delta"] > 0).sum()), "wp_et": "unavailable_without_valid_ETCP"}
    result = {"status": "completed_paired_E2_noforecast_5k_seeds", "by_seed": summary.to_dict(orient="records"), "overall": overall, "scope": "SYA originIC validation years 2014-2023; E2 checkpoint 5000; no-forecast controls seed1/2 freshly trained to 5K; seed0 paired table reused from validated 143E2 audit."}
    (OUT / "150E2_paired_5k_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# E2 5K forecast / no-forecast 配对验证", "", "| seed | forecast yield | no-forecast yield | delta | yield wins | forecast PFP_N | no-forecast PFP_N | PFP delta | PFP wins | I delta | N delta |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict(orient="records"):
        lines.append(f"| {row['seed']} | {row['mean_forecast_yield']:.2f} | {row['mean_no_forecast_yield']:.2f} | {row['mean_yield_delta']:.2f} | {row['yield_win_years']}/10 | {row['mean_forecast_pfp_n']:.3f} | {row['mean_no_forecast_pfp_n']:.3f} | {row['mean_pfp_n_delta']:.3f} | {row['pfp_n_win_years']}/10 | {row['mean_irrigation_delta']:.1f} | {row['mean_n_delta']:.1f} |")
    lines += ["", f"- 三个seed×年份总体平均产量差：`{overall['mean_yield_delta']:.2f}` kg/ha。", f"- 三个seed×年份总体PFP_N差：`{overall['mean_pfp_n_delta']:.3f}`。", f"- 产量为正的seed数：`{overall['positive_yield_delta_seeds']}/3`；PFP_N为正的seed数：`{overall['positive_pfp_delta_seeds']}/3`。", "- WP_ET 因缺少有效 ETCP 保持 unavailable。", "- 该配对比较只支持当前5K checkpoint，不外推到10K或更长训练。", ""]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
