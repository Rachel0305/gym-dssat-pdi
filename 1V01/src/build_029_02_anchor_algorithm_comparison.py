from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DQN = ROOT / "benchmark_results" / "029_02_five_site_stage_mask_aware_dqn"
PPO_OVERVIEW = ROOT / "benchmark_results" / "028_13_screened_year_maskableppo_advisor_summary" / "028_13_site_year_overview.csv"
ANCHOR_YEARS = {"SY": 2014, "HLA": 2010, "YC": 2014, "FQ": 2016, "LC": 2010}


def baseline_max(value: float, gap_percent: float) -> float:
    return float(value) / (1.0 + float(gap_percent) / 100.0)


def main() -> None:
    ppo = pd.read_csv(PPO_OVERVIEW)
    rows = []
    site_rows = []
    for site, year in ANCHOR_YEARS.items():
        anchor = ppo[(ppo.site == site) & (ppo.year == year)].iloc[0]
        maxima = {
            "yield": baseline_max(anchor.yield_kg_ha, anchor.yield_gap_pct_vs_four_max),
            "wp": baseline_max(anchor.WP_ET_kg_m3, anchor.WP_ET_gap_pct_vs_four_max),
            "pfp": baseline_max(anchor.PFP_N_kg_kg, anchor.PFP_N_gap_pct_vs_four_max),
        }
        winners = 0
        for seed in (0, 1, 2):
            payload = json.loads((DQN / site / f"seed{seed}" / "result.json").read_text(encoding="utf-8"))
            selected = payload["selected"]
            pfp = float(selected["PFP_N_kg_kg"])
            win_yield = float(selected["final_yield"]) > maxima["yield"]
            win_wp = float(selected["WP_ET_kg_m3"]) > maxima["wp"]
            win_pfp = math.isfinite(pfp) and pfp > maxima["pfp"]
            any_win = bool(win_yield or win_wp or win_pfp)
            winners += int(any_win)
            rows.append(
                {
                    "site": site,
                    "year": year,
                    "seed": seed,
                    "algorithm": "mask_aware_DQN",
                    "checkpoint": int(selected["checkpoint"]),
                    "yield_kg_ha": float(selected["final_yield"]),
                    "WP_ET_kg_m3": float(selected["WP_ET_kg_m3"]),
                    "PFP_N_kg_kg": pfp,
                    "irrigation_mm": float(selected["irrigation_total"]),
                    "nitrogen_kg_ha": float(selected["nitrogen_total"]),
                    "reward": float(selected["episode_total_reward"]),
                    "four_baseline_max_yield": maxima["yield"],
                    "four_baseline_max_WP_ET": maxima["wp"],
                    "four_baseline_max_PFP_N": maxima["pfp"],
                    "winning_yield": win_yield,
                    "winning_WP_ET": win_wp,
                    "winning_PFP_N": win_pfp,
                    "advisor_any_metric_winner": any_win,
                }
            )
        ppo_winners = int(anchor.winner_seed_count)
        site_rows.append(
            {
                "site": site,
                "year": year,
                "PPO_winner_seed_count": ppo_winners,
                "DQN_winner_seed_count": winners,
                "PPO_2of3_initially_stable": ppo_winners >= 2,
                "DQN_2of3_initially_stable": winners >= 2,
                "winner_count_difference_DQN_minus_PPO": winners - ppo_winners,
            }
        )
    detail = pd.DataFrame(rows)
    summary = pd.DataFrame(site_rows)
    detail.to_csv(DQN / "029_02_dqn_anchor_seed_summary.csv", index=False)
    summary.to_csv(DQN / "029_02_ppo_dqn_anchor_comparison.csv", index=False)

    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - 0.18, summary.PPO_winner_seed_count, width=0.36, label="MaskablePPO", color="#5B6FB5")
    ax.bar(x + 0.18, summary.DQN_winner_seed_count, width=0.36, label="mask-aware DQN", color="#2A8F62")
    ax.axhline(2, color="#555", linestyle="--", linewidth=1, label="2/3 initial stability")
    ax.set_xticks(x, [f"{s}{y}" for s, y in zip(summary.site, summary.year)])
    ax.set_ylim(0, 3.4)
    ax.set_ylabel("Seeds with at least one metric above all four baselines")
    ax.set_title("Training-anchor comparison under matched 240-step protocol", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#e5e5e5")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(DQN / "029_02_ppo_dqn_anchor_winner_counts.png", dpi=220, bbox_inches="tight")
    fig.savefig(DQN / "029_02_ppo_dqn_anchor_winner_counts.svg", bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
