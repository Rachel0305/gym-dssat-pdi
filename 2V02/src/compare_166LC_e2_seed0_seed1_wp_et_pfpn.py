"""Compare LC-E2 seed0/seed1 replay metrics against LC baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results/166LC_lca_lowIC_e2_seed0_seed1_wp_et_pfpn_comparison"
YEARS = list(range(2014, 2024))
BASELINE = ROOT / "benchmark_results/053_02_lca_lowIC_four_baselines_static_level1/evaluation/053_02_baseline_summary.csv"
REPLAYS = {
    "e2_s0": ROOT / "benchmark_results/157LC_lca_lowIC_dual_branch_weather_reader_wp_et_5k_replay/157LC_wp_et_replay_by_policy_year.csv",
    "e2_s1": ROOT / "benchmark_results/165LC_lca_lowIC_dual_branch_weather_reader_wp_et_5k_seed1_replay/165LC_wp_et_replay_by_policy_year.csv",
    "nof_s0": ROOT / "benchmark_results/160LC_lca_lowIC_e2_noforecast_wp_et_5k_paired_replay/160LC_replay_by_policy_year.csv",
}


def normalize_replay(path: Path, policy: str, seed: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    # The paired 160LC file contains both e2_s0 and nof_s0; retain only the
    # requested policy before assigning the normalized label.
    if policy == "nof_s0" and "policy" in frame.columns:
        frame = frame[frame["policy"].astype(str).str.strip().eq("nof_s0")].copy()
    frame["policy"] = policy
    frame["seed"] = seed
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype(int)
    frame["grain_yield_kg_ha"] = pd.to_numeric(frame["final_grnwt"], errors="coerce")
    frame["WP_ET_kg_m3"] = pd.to_numeric(frame["WP_ET_kg_m3"], errors="coerce")
    frame["PFP_N_kg_kg"] = pd.to_numeric(frame["PFP_N_kg_kg"], errors="coerce")
    frame["total_irrigation_mm"] = pd.to_numeric(frame["total_irrigation"], errors="coerce")
    frame["total_n_kg_ha"] = pd.to_numeric(frame["total_n"], errors="coerce")
    frame["ETCP_mm"] = pd.to_numeric(frame["etcp_mm"], errors="coerce")
    return frame[["policy", "seed", "year", "grain_yield_kg_ha", "total_irrigation_mm", "total_n_kg_ha", "ETCP_mm", "WP_ET_kg_m3", "PFP_N_kg_kg"]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    rows = []
    for policy, path in REPLAYS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(normalize_replay(path, policy, 0 if policy.endswith("s0") else 1))
    replay = pd.concat(rows, ignore_index=True)
    baseline = pd.read_csv(BASELINE, keep_default_na=False)
    for col in ["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "total_irrigation_mm", "total_n_kg_ha", "ETCP_mm"]:
        if col in baseline.columns:
            baseline[col] = pd.to_numeric(baseline[col], errors="coerce")
    baseline = baseline.rename(columns={"total_irrigation": "total_irrigation_mm", "total_n": "total_n_kg_ha"})
    baseline["policy"] = baseline["scenario"].astype(str)
    baseline["seed"] = np.nan
    base_cols = ["policy", "seed", "year", "grain_yield_kg_ha", "total_irrigation_mm", "total_n_kg_ha", "ETCP_mm", "WP_ET_kg_m3", "PFP_N_kg_kg"]
    baseline = baseline[[c for c in base_cols if c in baseline.columns]]
    replay.to_csv(OUT / "tables/166LC_e2_seed0_seed1_noforecast_by_year.csv", index=False, encoding="utf-8-sig")
    summary = replay.groupby(["policy", "seed"], dropna=False, as_index=False).agg(years=("year", "nunique"), mean_yield=("grain_yield_kg_ha", "mean"), mean_irrigation=("total_irrigation_mm", "mean"), mean_n=("total_n_kg_ha", "mean"), mean_ETCP=("ETCP_mm", "mean"), mean_WP_ET=("WP_ET_kg_m3", "mean"), mean_PFP_N=("PFP_N_kg_kg", "mean"), total_yield=("grain_yield_kg_ha", "sum"), total_n=("total_n_kg_ha", "sum"), total_ETCP=("ETCP_mm", "sum"))
    summary["weighted_WP_ET"] = summary["total_yield"] / (summary["total_ETCP"] * 10.0)
    summary["weighted_PFP_N"] = summary["total_yield"] / summary["total_n"].replace(0, np.nan)
    summary.to_csv(OUT / "tables/166LC_replay_summary.csv", index=False, encoding="utf-8-sig")

    baseline_max = baseline.groupby("year", as_index=False)[["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]].max()
    win_rows = []
    for policy in ["e2_s0", "e2_s1"]:
        candidate = replay[replay.policy.eq(policy)][["year", "grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]].merge(baseline_max, on="year", suffixes=("_e2", "_baseline_max"))
        for metric in ["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]:
            gap = candidate[f"{metric}_e2"] - candidate[f"{metric}_baseline_max"]
            win_rows.append({"policy": policy, "metric": metric, "years": len(candidate), "strict_wins_vs_four_baseline_max": int((gap > 0).sum()), "mean_gap": float(gap.mean())})
    wins = pd.DataFrame(win_rows)
    wins.to_csv(OUT / "tables/166LC_four_baseline_max_win_summary_by_seed.csv", index=False, encoding="utf-8-sig")

    e2 = replay[replay.policy.isin(["e2_s0", "e2_s1"])].pivot(index="year", columns="policy", values=["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"])
    e2.to_csv(OUT / "tables/166LC_seed0_seed1_metric_by_year.csv", encoding="utf-8-sig")
    e2_delta = pd.DataFrame([{"metric": metric, "mean_seed1_minus_seed0": float((e2[(metric, "e2_s1")] - e2[(metric, "e2_s0")]).mean()), "seed1_win_years": int((e2[(metric, "e2_s1")] > e2[(metric, "e2_s0")]).sum()), "seed0_win_years": int((e2[(metric, "e2_s0")] > e2[(metric, "e2_s1")]).sum())} for metric in ["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]])
    e2_delta.to_csv(OUT / "tables/166LC_seed1_minus_seed0_metric_summary.csv", index=False, encoding="utf-8-sig")

    nof = replay[replay.policy.eq("nof_s0")].set_index("year")
    paired = replay[replay.policy.isin(["e2_s0", "nof_s0"])].pivot(index="year", columns="policy", values=["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"])
    paired.to_csv(OUT / "tables/166LC_seed0_forecast_vs_noforecast_by_year.csv", encoding="utf-8-sig")
    report = [
        "# LC-E2 seed0/seed1 5K 产量、WP_ET、PFP-N 比较", "",
        "- 两个 E2 checkpoint 均为只读 10 年回放，不重新训练。",
        "- 四基线为 LC lowIC 053_02；无预报对照目前只有 matched seed0，未把它误标为 seed1。",
        "- WP_ET = grain yield / (10 × ETCP)；PFP-N = grain yield / total N。", "",
        "## E2 与无预报 seed0 汇总", "", summary[summary.policy.isin(["e2_s0", "nof_s0"])].round(4).to_markdown(index=False), "",
        "## 两个 seed 的相对四基线最高值", "", wins.round(4).to_markdown(index=False), "",
        "## seed1 - seed0", "", e2_delta.round(4).to_markdown(index=False), "",
        "## 解释边界", "",
        "相对四基线的胜出不等于天气预报的因果收益；要做 seed1 的严格天气信号因果对照，还需另跑 matched seed1 no-forecast control。",
    ]
    report_path = ROOT / "docs/166LC_lca_lowIC_e2_seed0_seed1_wp_et_pfpn_comparison_record.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    result = {"status": "completed", "policies": list(REPLAYS), "years": YEARS, "output_root": OUT.relative_to(ROOT).as_posix(), "report": report_path.relative_to(ROOT).as_posix()}
    (OUT / "166LC_comparison_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("---summary---")
    print(summary.round(4).to_string(index=False))
    print("---wins---")
    print(wins.round(4).to_string(index=False))
    print("---seed delta---")
    print(e2_delta.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
