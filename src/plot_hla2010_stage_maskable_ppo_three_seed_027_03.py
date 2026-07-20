from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "027_03"


def main() -> None:
    frames = []
    seed0 = pd.read_csv(ROOT / "benchmark_results" / "027_02_attempt2" / "027_02_hla2010_seed0_checkpoint_summary.csv")
    seed0["seed"] = 0
    frames.append(seed0)
    for seed in (1, 2):
        frames.append(pd.read_csv(OUT / f"seed{seed}" / f"027_03_hla2010_seed{seed}_checkpoint_summary.csv"))
    data = pd.concat(frames, ignore_index=True)
    selected = pd.read_csv(OUT / "027_03_hla2010_three_seed_selected_summary.csv")
    colors = {0: "#0072B2", 1: "#D55E00", 2: "#009E73"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    specs = [
        ("final_yield", "Yield (kg/ha)", [(7853.665161, "auto/expert gate", "--"), (7679.0, "recorded", ":")]),
        ("irrigation_total", "Irrigation (mm)", []),
        ("nitrogen_total", "Nitrogen (kg/ha)", []),
        ("WP_ET_kg_m3", "WP_ET (kg/m3)", [(1.64, "primary gate", "--"), (1.70, "recorded", ":")]),
        ("PFP_N_kg_kg", "PFP_N (kg/kg)", [(26.2, "expert", "--"), (46.5, "recorded", ":")]),
        ("episode_total_reward", "Deterministic season reward", []),
    ]
    for ax, (column, ylabel, references) in zip(axes.flat, specs):
        for seed, group in data.groupby("seed"):
            ax.plot(group["checkpoint"], group[column], marker="o", label=f"seed{int(seed)}", color=colors[int(seed)])
            picked = selected[selected.seed.eq(seed)].iloc[0]
            point = group[group.checkpoint.eq(int(picked.selected_checkpoint))].iloc[0]
            ax.scatter([point.checkpoint], [point[column]], s=110, facecolors="none", edgecolors=colors[int(seed)], linewidths=2)
        for value, label, style in references:
            ax.axhline(value, color="#666666", linestyle=style, label=label)
        ax.set_xlabel("Checkpoint stage steps")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("HLA2010 stage MaskablePPO: three-seed preregistered replication\nOpen circles mark selected checkpoints")
    fig.savefig(OUT / "027_03_hla2010_three_seed_checkpoint_comparison.png", dpi=220)
    fig.savefig(OUT / "027_03_hla2010_three_seed_checkpoint_comparison.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
