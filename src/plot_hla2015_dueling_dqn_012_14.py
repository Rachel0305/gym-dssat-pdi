from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2015_dueling_dqn_economic_012_14" / "summary"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2015_dueling_dqn_economic_012_14" / "figures"


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
    df = pd.read_csv(IN_DIR / "hla2015_dueling_dqn_seed0_5k_summary.csv")
    colors = ["#464C55", "#386411", "#CC6F47", "#B23A48", "#8B5FBF", "#78A757"]
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.0))
    fig.patch.set_facecolor("#FCFCFD")
    fig.suptitle("HLA 2015 Dueling DQN seed0 5K probe", x=0.06, ha="left", fontsize=15, fontweight="bold")

    axes[0].bar(df["run"], df["yield_kg_ha"], color=colors, edgecolor="#333333", linewidth=0.5)
    axes[0].set_ylabel("Harvest yield (kg/ha)")
    axes[0].set_title("Yield")
    axes[0].tick_params(axis="x", rotation=38)
    style_axis(axes[0])

    axes[1].bar(df["run"], df["irrigation_mm"], color=colors, edgecolor="#333333", linewidth=0.5)
    axes[1].set_ylabel("Irrigation (mm)")
    axes[1].set_title("Irrigation")
    axes[1].tick_params(axis="x", rotation=38)
    style_axis(axes[1])

    axes[2].bar(df["run"], df["economic_reward"], color=colors, edgecolor="#333333", linewidth=0.5)
    axes[2].set_ylabel("Economic reward")
    axes[2].set_title("Economic reward")
    axes[2].tick_params(axis="x", rotation=38)
    style_axis(axes[2])

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = OUT_DIR / "hla2015_dueling_dqn_seed0_5k_summary.png"
    fig.savefig(out, dpi=280, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
