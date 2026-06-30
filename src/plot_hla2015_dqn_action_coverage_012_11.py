from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COVERAGE_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2015_dqn_action_coverage_012_11"
DQN_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_dqn_economic_reward_probe_012_03" / "2015"
OUT_DIR = COVERAGE_DIR / "figures"


def style_axis(ax) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, axis="y", color="#E6E8F0", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D7DBE7")
    ax.spines["bottom"].set_color("#D7DBE7")
    ax.tick_params(colors="#464C55", labelsize=10)


def load_eval_summary() -> pd.DataFrame:
    rows = []
    for seed in [0, 1, 2]:
        d = json.loads((DQN_DIR / f"medium_N_cost_seed{seed}_5000steps" / "event_summary.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "seed": seed,
                "eval_yield_kg_ha": d["dqn_harvest_yield_kg_ha"],
                "eval_irrigation_mm": d["irrigation_total"],
                "eval_nitrogen_kg_ha": d["nitrogen_total"],
                "eval_reward": d["economic_reward_total_eval"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coverage = pd.read_csv(COVERAGE_DIR / "hla2015_dqn_action_coverage_5000steps_summary.csv")
    eval_summary = load_eval_summary()
    merged = coverage.merge(eval_summary, on="seed", how="left")
    merged.to_csv(OUT_DIR / "hla2015_dqn_action_coverage_with_eval_summary.csv", index=False, encoding="utf-8-sig")

    labels = ["seed0", "seed1", "seed2"]
    colors = ["#464C55", "#386411", "#B23A48"]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8))
    fig.patch.set_facecolor("#FCFCFD")
    fig.suptitle("HLA 2015 DQN action coverage diagnostic", x=0.06, ha="left", fontsize=15, fontweight="bold")

    axes[0, 0].bar(labels, merged["irrigation_window_wants_irrigation_rate"], color=colors)
    axes[0, 0].set_title("Training: irrigation action rate inside irrigation windows")
    axes[0, 0].set_ylabel("rate")
    style_axis(axes[0, 0])

    axes[0, 1].bar(labels, merged["episode_safe_irrigation_rate"], color=colors)
    axes[0, 1].set_title("Training: episodes with at least one safe irrigation")
    axes[0, 1].set_ylabel("rate")
    axes[0, 1].set_ylim(0, 1.05)
    style_axis(axes[0, 1])

    axes[1, 0].bar(labels, merged["eval_irrigation_mm"], color=colors)
    axes[1, 0].set_title("Final deterministic evaluation: irrigation")
    axes[1, 0].set_ylabel("mm")
    style_axis(axes[1, 0])

    axes[1, 1].bar(labels, merged["eval_reward"], color=colors)
    axes[1, 1].set_title("Final deterministic evaluation: economic reward")
    axes[1, 1].set_ylabel("reward")
    style_axis(axes[1, 1])

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = OUT_DIR / "hla2015_dqn_action_coverage_diagnostic.png"
    fig.savefig(out, dpi=280, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(
        {
            "file_type": ["figure", "merged_summary_csv", "raw_coverage_summary_csv"],
            "path": [
                str(out),
                str(OUT_DIR / "hla2015_dqn_action_coverage_with_eval_summary.csv"),
                str(COVERAGE_DIR / "hla2015_dqn_action_coverage_5000steps_summary.csv"),
            ],
        }
    ).to_csv(OUT_DIR / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    print(merged[[
        "seed",
        "irrigation_window_wants_irrigation_rate",
        "episode_safe_irrigation_rate",
        "mean_episode_safe_irrigation",
        "eval_irrigation_mm",
        "eval_nitrogen_kg_ha",
        "eval_reward",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
