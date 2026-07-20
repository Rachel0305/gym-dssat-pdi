#!/usr/bin/env python3
"""Combine 029 anchors and frozen transfers into the complete 17-year comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "benchmark_results" / "029_02_five_site_stage_mask_aware_dqn"
TRANSFER = ROOT / "benchmark_results" / "029_03_frozen_maskaware_dqn_crossyear"
PPO_PATH = ROOT / "benchmark_results" / "028_13_screened_year_maskableppo_advisor_summary" / "028_13_site_year_overview.csv"
OUT = ROOT / "benchmark_results" / "029_04_maskableppo_vs_maskaware_dqn_evidence"


def anchor_seed_rows() -> pd.DataFrame:
    source = pd.read_csv(ANCHOR / "029_02_dqn_anchor_seed_summary.csv")
    return pd.DataFrame({
        "site": source["site"],
        "year": source["year"].astype(int),
        "seed": source["seed"].astype(int),
        "algorithm": "Mask-aware DQN",
        "training_role": "local_anchor_training",
        "checkpoint": source["checkpoint"].astype(int),
        "final_gwad": source["yield_kg_ha"],
        "WP_ET_kg_m3": source["WP_ET_kg_m3"],
        "PFP_N_kg_kg": source["PFP_N_kg_kg"],
        "irrigation_total": source["irrigation_mm"],
        "fertilizer_total": source["nitrogen_kg_ha"],
        "episode_reward": source["reward"],
        "four_baseline_max_yield": source["four_baseline_max_yield"],
        "four_baseline_max_WP_ET": source["four_baseline_max_WP_ET"],
        "four_baseline_max_PFP_N": source["four_baseline_max_PFP_N"],
        "yield_strict_win": source["winning_yield"],
        "wp_et_strict_win": source["winning_WP_ET"],
        "pfp_n_strict_win": source["winning_PFP_N"],
        "advisor_any_metric_win": source["advisor_any_metric_winner"],
    })


def transfer_seed_rows() -> pd.DataFrame:
    frame = pd.read_csv(TRANSFER / "029_03_dqn_frozen_crossyear_seed_summary.csv")
    frame["training_role"] = "frozen_same_site_crossyear_transfer"
    frame["checkpoint"] = frame["source_model"].str.extract(r"checkpoint_(\d+)", expand=False).astype(int)
    # Reconstruct the exact frozen training objective for evaluators that did
    # not persist the accumulated scalar.  All component values are in the
    # already-persisted result and reused four-baseline tables.
    baseline_paths = {
        "SY": ROOT / "benchmark_results/026_07_attempt2/026_07_sy_all_years_four_baselines.csv",
        "HLA": ROOT / "benchmark_results/028_08_hla_frozen_maskableppo_screened_year_transfer/028_08_reused_four_baselines.csv",
        "YC": ROOT / "benchmark_results/028_09_yc2014_frozen_maskableppo_yc2008_transfer/028_09_reused_four_baselines.csv",
        "FQ": ROOT / "benchmark_results/028_11_fq_frozen_maskableppo_screened_year_transfer/028_11_reused_four_baselines.csv",
    }
    lookup = {}
    for site, path in baseline_paths.items():
        base = pd.read_csv(path).assign(scenario=lambda x: x.scenario.fillna("null"))
        for year, group in base.groupby(base.year.astype(int)):
            null_y = float(group.loc[group.scenario == "null", "final_gwad"].iloc[0])
            gate_y = float(group.loc[group.scenario == "official_extension_expert", "final_gwad"].iloc[0])
            lookup[(site, int(year))] = (null_y, gate_y)
    reconstructed = []
    for _, row in frame.iterrows():
        null_y, gate_y = lookup[(str(row.site), int(row.year))]
        value = -(float(row.irrigation_total) + 5.0 * float(row.fertilizer_total))
        value += max(0.0, float(row.final_gwad) - null_y)
        if float(row.final_gwad) >= gate_y:
            value += 1620.0
        reconstructed.append(value / 1000.0)
    frame["episode_reward"] = reconstructed
    return frame


def representative_rows(seed_rows: pd.DataFrame) -> pd.DataFrame:
    # Visualization only: reward max, exact tie lower seed. Main result remains all-seed.
    rows = []
    for (_, _), group in seed_rows.groupby(["site", "year"], sort=True):
        ordered = group.sort_values(["episode_reward", "seed"], ascending=[False, True])
        row = ordered.iloc[0].copy()
        row["representative_selection_rule"] = "max local episode reward; exact tie lower seed; visualization only"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    anchors = anchor_seed_rows()
    transfers = transfer_seed_rows()
    seed_rows = pd.concat([anchors, transfers], ignore_index=True, sort=False)
    seed_rows.to_csv(OUT / "029_04_dqn_all_17_year_seed_results.csv", index=False, encoding="utf-8-sig")
    reps = representative_rows(seed_rows)
    reps.to_csv(OUT / "029_04_dqn_visualization_representatives.csv", index=False, encoding="utf-8-sig")

    dqn_matrix = seed_rows.groupby(["site", "year"], as_index=False).agg(
        DQN_winner_seed_count=("advisor_any_metric_win", "sum"),
        DQN_yield_win_count=("yield_strict_win", "sum"),
        DQN_WP_ET_win_count=("wp_et_strict_win", "sum"),
        DQN_PFP_N_win_count=("pfp_n_strict_win", "sum"),
        DQN_seed_count=("seed", "count"),
    )
    dqn_matrix["DQN_2of3"] = dqn_matrix["DQN_winner_seed_count"] >= 2
    ppo = pd.read_csv(PPO_PATH)
    paired = ppo[["site", "year", "winner_seed_count", "seed_count", "cross_seed_status", "representative_any_metric_win"]].rename(columns={
        "winner_seed_count": "PPO_winner_seed_count",
        "seed_count": "PPO_seed_count",
        "cross_seed_status": "PPO_cross_seed_status",
        "representative_any_metric_win": "PPO_representative_any_metric_win",
    }).merge(dqn_matrix, on=["site", "year"], how="inner", validate="one_to_one")
    paired["PPO_2of3"] = paired["PPO_winner_seed_count"] >= 2
    paired["winner_count_difference_DQN_minus_PPO"] = paired["DQN_winner_seed_count"] - paired["PPO_winner_seed_count"]
    paired["algorithm_comparison"] = np.select(
        [paired["winner_count_difference_DQN_minus_PPO"] > 0, paired["winner_count_difference_DQN_minus_PPO"] < 0],
        ["DQN_higher", "PPO_higher"], default="tie",
    )
    paired.to_csv(OUT / "029_04_ppo_dqn_all_17_year_comparison.csv", index=False, encoding="utf-8-sig")

    site = paired.groupby("site", as_index=False).agg(
        years=("year", "count"),
        PPO_total_winner_seeds=("PPO_winner_seed_count", "sum"),
        DQN_total_winner_seeds=("DQN_winner_seed_count", "sum"),
        PPO_years_2of3=("PPO_2of3", "sum"),
        DQN_years_2of3=("DQN_2of3", "sum"),
        DQN_higher_years=("algorithm_comparison", lambda x: int((x == "DQN_higher").sum())),
        PPO_higher_years=("algorithm_comparison", lambda x: int((x == "PPO_higher").sum())),
        tied_years=("algorithm_comparison", lambda x: int((x == "tie").sum())),
    )
    site.to_csv(OUT / "029_04_ppo_dqn_site_summary.csv", index=False, encoding="utf-8-sig")

    order = paired.sort_values(["site", "year"])
    labels = [f"{s}{int(y)}" for s, y in zip(order.site, order.year)]
    x = np.arange(len(labels)); width = 0.38
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(x - width / 2, order.PPO_winner_seed_count, width, label="MaskablePPO", color="#4C78A8")
    ax.bar(x + width / 2, order.DQN_winner_seed_count, width, label="Mask-aware DQN", color="#F58518")
    ax.axhline(2, color="#777777", linestyle="--", linewidth=1, label="2/3 threshold")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Advisor-rule winning seeds (of 3)")
    ax.set_title("MaskablePPO vs Mask-aware DQN on the same 17 screened site-years")
    ax.set_ylim(0, 3.35); ax.legend(ncol=3); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "029_04_ppo_dqn_all_17_year_winner_counts.png", dpi=220)
    fig.savefig(OUT / "029_04_ppo_dqn_all_17_year_winner_counts.svg")
    plt.close(fig)

    checks = {
        "exact_17_site_years": len(paired) == 17,
        "exact_51_dqn_seed_results": len(seed_rows) == 51,
        "all_dqn_three_seed": bool((dqn_matrix["DQN_seed_count"] == 3).all()),
        "ppo_dqn_same_scope": len(paired) == len(ppo) == 17,
    }
    result = {
        "status": "completed" if all(checks.values()) else "failed",
        "checks": checks,
        "overall": {
            "PPO_total_winner_seeds": int(paired.PPO_winner_seed_count.sum()),
            "DQN_total_winner_seeds": int(paired.DQN_winner_seed_count.sum()),
            "PPO_years_2of3": int(paired.PPO_2of3.sum()),
            "DQN_years_2of3": int(paired.DQN_2of3.sum()),
            "DQN_higher_years": int((paired.algorithm_comparison == "DQN_higher").sum()),
            "PPO_higher_years": int((paired.algorithm_comparison == "PPO_higher").sum()),
            "tied_years": int((paired.algorithm_comparison == "tie").sum()),
        },
    }
    (OUT / "029_04_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "completed":
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
