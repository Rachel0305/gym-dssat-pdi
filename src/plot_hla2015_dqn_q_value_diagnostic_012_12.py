from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2015_dqn_q_value_diagnostic_012_12"
OUT_DIR = IN_DIR / "figures"


COLORS = {
    "seed0_5k": "#464C55",
    "seed1_5k": "#386411",
    "seed2_5k": "#B23A48",
    "seed0_20k": "#CC6F47",
}


def style_axis(ax) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, axis="y", color="#E6E8F0", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D7DBE7")
    ax.spines["bottom"].set_color("#D7DBE7")
    ax.tick_params(colors="#464C55", labelsize=10)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(IN_DIR / "hla2015_dqn_q_value_summary.csv")
    order = ["seed0_5k", "seed1_5k", "seed2_5k", "seed0_20k"]
    summary = summary.set_index("run").loc[order].reset_index()
    colors = [COLORS[r] for r in summary["run"]]

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.8))
    fig.patch.set_facecolor("#FCFCFD")
    fig.suptitle("HLA 2015 DQN Q-value diagnostic", x=0.06, ha="left", fontsize=15, fontweight="bold")

    axes[0, 0].bar(summary["run"], summary["iw_best_irrigation_or_both_rate"], color=colors)
    axes[0, 0].set_title("Irrigation-window states where irrigation/both has max Q")
    axes[0, 0].set_ylabel("rate")
    axes[0, 0].tick_params(axis="x", rotation=25)
    style_axis(axes[0, 0])

    axes[0, 1].bar(summary["run"], summary["iw_mean_q_irrigation_minus_noop"], color=colors)
    axes[0, 1].axhline(0, color="#222222", linewidth=1.0)
    axes[0, 1].set_title("Mean Q(irrigation) - Q(no-op) in irrigation windows")
    axes[0, 1].set_ylabel("Q difference")
    axes[0, 1].tick_params(axis="x", rotation=25)
    style_axis(axes[0, 1])

    axes[1, 0].bar(summary["run"], summary["safe_irrigation_total"], color=colors)
    axes[1, 0].set_title("Final deterministic evaluation: irrigation")
    axes[1, 0].set_ylabel("mm")
    axes[1, 0].tick_params(axis="x", rotation=25)
    style_axis(axes[1, 0])

    axes[1, 1].bar(summary["run"], summary["total_reward"], color=colors)
    axes[1, 1].set_title("Final deterministic evaluation: economic reward")
    axes[1, 1].set_ylabel("reward")
    axes[1, 1].tick_params(axis="x", rotation=25)
    style_axis(axes[1, 1])

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = OUT_DIR / "hla2015_dqn_q_value_summary.png"
    fig.savefig(out, dpi=280, bbox_inches="tight")
    plt.close(fig)

    key = pd.read_csv(IN_DIR / "hla2015_dqn_q_values_key_daps.csv")
    key.to_csv(OUT_DIR / "hla2015_dqn_q_values_key_daps_for_plot.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "file_type": ["summary_figure", "summary_csv", "key_daps_csv", "all_q_daily_csv"],
            "path": [
                str(out),
                str(IN_DIR / "hla2015_dqn_q_value_summary.csv"),
                str(IN_DIR / "hla2015_dqn_q_values_key_daps.csv"),
                str(IN_DIR / "hla2015_dqn_q_values_all_runs_daily.csv"),
            ],
        }
    ).to_csv(OUT_DIR / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    print(summary[[
        "run",
        "iw_best_irrigation_or_both_rate",
        "iw_mean_q_irrigation_minus_noop",
        "safe_irrigation_total",
        "safe_nitrogen_total",
        "final_grnwt",
        "total_reward",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
