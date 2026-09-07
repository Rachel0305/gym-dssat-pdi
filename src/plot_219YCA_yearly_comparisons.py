from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "benchmark_results/218YCA_yca_lowIC_10y_six_weather_simple_profit_aug_maskableppo_500k_five_scenario_figures_ckpt5000_attempt2"
NEW = ROOT / "benchmark_results/219YCA_yca_lowIC_10y_six_weather_normobs_simple_profit_pilot"
FIG = NEW / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False})


def main():
    five = pd.read_csv(OLD / "tables/218YCA_five_scenario_season_summary.csv")
    five["scenario_label"] = five["scenario"].fillna("rainfed_null")
    labels = {
        "dssat_auto_external_n": "DSSAT auto irrigation",
        "rainfed_null": "Rainfed/null",
        "official_extension_expert": "Official expert",
        "recorded_farmer_template": "Recorded farmer",
        "rl_candidate": "PPO 218-5K",
    }
    five["scenario_label"] = five["scenario_label"].map(labels)
    metrics = [
        ("grain_yield_kg_ha", "Grain yield (kg/ha)", "yield"),
        ("actual_irrigation_mm", "Irrigation input (mm)", "irrigation"),
        ("actual_nitrogen_kg_ha", "Nitrogen input (kg/ha)", "nitrogen"),
        ("WP_ET_kg_m3", "WP_ET (kg/m³)", "wp_et"),
        ("PFP_N_kg_kg", "PFP_N (kg/kg)", "pfp_n"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=True)
    axes = axes.ravel()
    for ax, (col, ylabel, _) in zip(axes, metrics):
        for name, g in five.groupby("scenario_label", sort=False):
            ax.plot(g.year, g[col], marker="o", linewidth=1.8, label=name)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.set_xticks(sorted(five.year.unique()))
    axes[-1].axis("off")
    axes[0].set_title("YC/YCA 2014–2023 five-scenario yearly comparison (218YCA-5K replay)")
    axes[0].legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "219YCA_available_five_scenario_yearly_comparison.png", dpi=220)
    fig.savefig(FIG / "219YCA_available_five_scenario_yearly_comparison.pdf")
    fig.savefig(FIG / "219YCA_available_five_scenario_yearly_comparison.svg")
    plt.close(fig)

    old = five[five["scenario_label"] == "PPO 218-5K"].copy()
    old = old.rename(columns={"grain_yield_kg_ha": "old_yield", "actual_irrigation_mm": "old_I", "actual_nitrogen_kg_ha": "old_N", "WP_ET_kg_m3": "old_WP_ET", "PFP_N_kg_kg": "old_PFP_N"})
    new = pd.read_csv(NEW / "tables/219YCA_development_validation_episode_metrics.csv")
    new = new[new["checkpoint_step"] == 20160].copy()
    new = new[new["seed"] == 1].rename(columns={"final_grnwt": "new_yield", "total_irrigation": "new_I", "total_n": "new_N", "WP_ET_kg_m3": "new_WP_ET", "PFP_N": "new_PFP_N"})
    cmp = old[["year", "old_yield", "old_I", "old_N", "old_WP_ET", "old_PFP_N"]].merge(new[["year", "new_yield", "new_I", "new_N", "new_WP_ET", "new_PFP_N"]], on="year", how="inner")
    cmp.to_csv(NEW / "tables/219YCA_seed1_ck20160_vs_218YCA5k_yearly.csv", index=False)
    panels = [("old_I", "new_I", "Input: irrigation (mm)"), ("old_N", "new_N", "Input: nitrogen (kg/ha)"), ("old_yield", "new_yield", "Grain yield (kg/ha)"), ("old_WP_ET", "new_WP_ET", "WP_ET (kg/m³)"), ("old_PFP_N", "new_PFP_N", "PFP_N (kg/kg)")]
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=True)
    axes = axes.ravel()
    for ax, (a, b, title) in zip(axes, panels):
        ax.plot(cmp.year, cmp[a], marker="o", linewidth=2, label="218YCA-5K")
        ax.plot(cmp.year, cmp[b], marker="s", linewidth=2, label="219YCA normobs-20K seed1")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.set_xticks(cmp.year)
    axes[-1].axis("off")
    axes[0].legend()
    fig.suptitle("YC/YCA yearly inputs and outcomes: 218YCA-5K vs 219YCA normobs-20K", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG / "219YCA_seed1_ck20160_vs_218YCA5k_yearly_metrics.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG / "219YCA_seed1_ck20160_vs_218YCA5k_yearly_metrics.pdf", bbox_inches="tight")
    fig.savefig(FIG / "219YCA_seed1_ck20160_vs_218YCA5k_yearly_metrics.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures and {len(cmp)} yearly comparison rows to {FIG}")


if __name__ == "__main__":
    main()
