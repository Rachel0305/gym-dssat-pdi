from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "027_02_attempt2"


def main() -> None:
    checkpoints = pd.read_csv(OUT / "027_02_hla2010_seed0_checkpoint_summary.csv")
    episodes = pd.read_csv(OUT / "027_02_hla2010_seed0_training_episode_summary.csv")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    x = checkpoints["checkpoint"]
    axes[0, 0].plot(x, checkpoints["final_yield"], marker="o", color="#0072B2")
    axes[0, 0].axhline(7853.665161, color="#D55E00", linestyle="--", label="auto/expert gate")
    axes[0, 0].axhline(7679.0, color="#777777", linestyle=":", label="recorded")
    axes[0, 0].set_ylabel("Yield (kg/ha)")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(x, checkpoints["irrigation_total"], marker="o", label="Irrigation (mm)", color="#56B4E9")
    axes[0, 1].plot(x, checkpoints["nitrogen_total"], marker="s", label="Nitrogen (kg/ha)", color="#009E73")
    axes[0, 1].set_ylabel("Seasonal input")
    axes[0, 1].legend(fontsize=8)

    axes[0, 2].plot(x, checkpoints["episode_total_reward"], marker="o", color="#CC79A7")
    axes[0, 2].set_ylabel("Deterministic season reward")

    axes[1, 0].plot(x, checkpoints["WP_ET_kg_m3"], marker="o", color="#E69F00")
    axes[1, 0].axhline(1.64, color="#D55E00", linestyle="--", label="primary gate")
    axes[1, 0].axhline(1.70, color="#777777", linestyle=":", label="recorded")
    axes[1, 0].set_ylabel("WP_ET (kg/m3)")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(x, checkpoints["PFP_N_kg_kg"], marker="o", color="#009E73")
    axes[1, 1].axhline(26.2, color="#D55E00", linestyle="--", label="expert")
    axes[1, 1].axhline(46.5, color="#777777", linestyle=":", label="recorded")
    axes[1, 1].set_ylabel("PFP_N (kg/kg)")
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].plot(episodes["episode_index"], episodes["episode_total_reward"], color="#0072B2", alpha=0.75)
    axes[1, 2].set_ylabel("Training episode reward")
    axes[1, 2].set_xlabel("Training season index")

    for ax in axes.flat:
        ax.grid(alpha=0.25)
        if ax is not axes[1, 2]:
            ax.set_xlabel("Checkpoint stage steps")
    fig.suptitle("HLA2010 stage MaskablePPO seed0: preregistered 240-step run")
    fig.savefig(OUT / "027_02_hla2010_seed0_checkpoint_and_training_curve.png", dpi=220)
    fig.savefig(OUT / "027_02_hla2010_seed0_checkpoint_and_training_curve.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
