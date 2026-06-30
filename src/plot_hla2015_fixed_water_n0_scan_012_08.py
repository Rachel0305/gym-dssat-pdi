from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2015_fixed_water_n0_scan_012_08"
OUT_DIR = IN_DIR / "figures"


def style_axis(ax) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, axis="both", color="#E6E8F0", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D7DBE7")
    ax.spines["bottom"].set_color("#D7DBE7")
    ax.tick_params(colors="#464C55", labelsize=10)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(IN_DIR / "hla2015_fixed_water_n0_scan_summary.csv")
    summary["label"] = summary["irrigation_total"].astype(int).astype(str) + " mm"
    colors = ["#9AA1AE", "#86A6D9", "#386411", "#5477C4", "#2F5597"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    fig.patch.set_facecolor("#FCFCFD")
    fig.suptitle("HLA 2015 fixed N0 water scan", x=0.06, ha="left", fontsize=15, fontweight="bold")
    axes[0].bar(summary["label"], summary["harvest_yield_kg_ha"], color=colors, edgecolor="#333333", linewidth=0.5)
    axes[0].set_ylabel("Harvest yield (kg/ha)")
    axes[0].set_title("Yield")
    style_axis(axes[0])
    axes[1].bar(summary["label"], summary["economic_reward_total"], color=colors, edgecolor="#333333", linewidth=0.5)
    axes[1].set_ylabel("Economic reward")
    axes[1].set_title("Reward = ΔGRNWT - 1.0*I")
    style_axis(axes[1])
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = OUT_DIR / "hla2015_fixed_water_n0_scan_summary.png"
    fig.savefig(out, dpi=280, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(
        {
            "file_type": ["summary_figure", "daily_csv", "summary_csv"],
            "path": [
                str(out),
                str(IN_DIR / "hla2015_fixed_water_n0_scan_daily.csv"),
                str(IN_DIR / "hla2015_fixed_water_n0_scan_summary.csv"),
            ],
        }
    ).to_csv(OUT_DIR / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    print(out)


if __name__ == "__main__":
    main()
