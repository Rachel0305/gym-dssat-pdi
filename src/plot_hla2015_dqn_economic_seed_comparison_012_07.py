from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DQN_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_dqn_economic_reward_probe_012_03"
DQN_2015 = DQN_ROOT / "2015"
FIXED_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2015_fixed_action_counterfactual_012_06"
FOUR_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_2010_2015_four_scenario_with_ppo"
OUT_DIR = DQN_ROOT / "figures_012_07_hla2015_seed_comparison"


COLORS = {
    "dqn_seed0": "#464C55",
    "dqn_seed1": "#386411",
    "dqn_seed2": "#B23A48",
    "fixed_I0_N0": "#9AA1AE",
    "fixed_I120_N0": "#5477C4",
    "fixed_I120_N50": "#78A757",
    "fixed_I120_N100": "#CC6F47",
    "fixed_I120_N150": "#804126",
}
LABELS = {
    "dqn_seed0": "DQN seed0",
    "dqn_seed1": "DQN seed1",
    "dqn_seed2": "DQN seed2",
    "fixed_I0_N0": "fixed I0/N0",
    "fixed_I120_N0": "fixed I120/N0",
    "fixed_I120_N50": "fixed I120/N50",
    "fixed_I120_N100": "fixed I120/N100",
    "fixed_I120_N150": "fixed I120/N150",
}
STYLES = {
    "dqn_seed0": "--",
    "dqn_seed1": "-",
    "dqn_seed2": "-.",
    "fixed_I0_N0": ":",
    "fixed_I120_N0": ":",
    "fixed_I120_N50": ":",
    "fixed_I120_N100": ":",
    "fixed_I120_N150": ":",
}


def style_axis(ax) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, axis="both", color="#E6E8F0", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D7DBE7")
    ax.spines["bottom"].set_color("#D7DBE7")
    ax.tick_params(colors="#464C55", labelsize=10)


def load_rain() -> pd.DataFrame:
    base = pd.read_csv(FOUR_DIR / "hla_2010_2015_four_scenario_daily.csv", keep_default_na=False)
    base["scenario"] = base["scenario"].replace({"": "null_zero", "null": "null_zero"}).fillna("null_zero")
    base = base[base["requested_year"].astype(int).eq(2015) & base["scenario"].eq("null_zero")]
    return base[["dap", "rain"]].drop_duplicates("dap").sort_values("dap")


def load_dqn(seed: int) -> pd.DataFrame:
    p = DQN_2015 / f"medium_N_cost_seed{seed}_5000steps" / "dqn_economic_eval_daily.csv"
    df = pd.read_csv(p)
    return pd.DataFrame(
        {
            "scenario": f"dqn_seed{seed}",
            "dap": df["dap"],
            "water_stress": df["swfac"],
            "nitrogen_stress": df["nstres"],
            "grain_kg_ha": df["grnwt"],
            "biomass_kg_ha": df["topwt"],
            "irrigation_mm": df["safe_amir"],
            "fertilizer_kg_ha": df["safe_anfer"],
            "reward": df["reward"],
        }
    )


def process_plot(data: pd.DataFrame, rain: pd.DataFrame, out_path: Path) -> None:
    scenarios = ["dqn_seed0", "dqn_seed1", "dqn_seed2"]
    max_dap = int(np.nanmax(data["dap"]))
    x_max = max_dap + 5
    x_ticks = np.arange(0, x_max + 1, 25)
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(15.8, 13.2),
        sharex=True,
        gridspec_kw={"height_ratios": [0.85, 1.0, 1.0, 1.0, 1.15], "hspace": 0.26},
    )
    fig.patch.set_facecolor("#FCFCFD")
    fig.subplots_adjust(top=0.91)
    fig.text(0.08, 0.965, "HLA 2015 economic DQN seed comparison", ha="left", va="top", fontsize=16, color="#1F2430", fontweight="bold")
    fig.text(0.08, 0.94, "Both runs use 5K timesteps and reward = delta GRNWT - 1.0*I - 5.0*N.", ha="left", va="top", fontsize=9.5, color="#6F768A")

    axes[0].bar(rain["dap"], rain["rain"], width=1.0, color="#C5CAD3", edgecolor="#7A828F", linewidth=0.45)
    axes[0].set_ylabel("Rain\n(mm)")
    style_axis(axes[0])

    for ax, col, ylabel, title in [
        (axes[1], "water_stress", "Water\nstress", "Water stress index"),
        (axes[2], "nitrogen_stress", "Nitrogen\nstress", "Nitrogen stress index"),
    ]:
        for scenario in scenarios:
            sub = data[data["scenario"].eq(scenario)].sort_values("dap")
            ax.plot(sub["dap"], sub[col], color=COLORS[scenario], linestyle=STYLES[scenario], linewidth=2.2)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, max(1.05, float(data[col].max()) * 1.15))
        ax.set_title(title, loc="left", fontsize=10)
        style_axis(ax)

    for scenario in scenarios:
        sub = data[data["scenario"].eq(scenario)].sort_values("dap")
        color = COLORS[scenario]
        irrig = sub[sub["irrigation_mm"].fillna(0) > 1e-6]
        fert = sub[sub["fertilizer_kg_ha"].fillna(0) > 1e-6]
        if not irrig.empty:
            axes[3].vlines(irrig["dap"], 0, irrig["irrigation_mm"], colors=color, linewidth=2.6, alpha=0.95)
        if not fert.empty:
            axes[3].scatter(fert["dap"], fert["fertilizer_kg_ha"], marker="^", s=55, color=color, edgecolor="#FFFFFF", linewidth=0.6)
    axes[3].set_ylabel("Mgmt\namount")
    axes[3].set_title("Management events: irrigation as vertical lines; fertilization as triangle markers", loc="left", fontsize=10)
    axes[3].set_ylim(-5, 36)
    style_axis(axes[3])

    for scenario in scenarios:
        sub = data[data["scenario"].eq(scenario)].sort_values("dap")
        axes[4].plot(sub["dap"], sub["grain_kg_ha"], color=COLORS[scenario], linestyle=STYLES[scenario], linewidth=2.2)
        axes[4].plot(sub["dap"], sub["biomass_kg_ha"], color=COLORS[scenario], linestyle="--", linewidth=1.6, alpha=0.65)
    axes[4].set_ylabel("kg/ha")
    axes[4].set_xlabel("DAP")
    axes[4].set_title("Crop outcome: solid = grain weight, dashed = aboveground biomass", loc="left", fontsize=10)
    style_axis(axes[4])

    for ax in axes:
        ax.set_xlim(-2, x_max)
        ax.set_xticks(x_ticks)
    handles = [Line2D([0], [0], color=COLORS[s], lw=2.2, linestyle=STYLES[s], label=LABELS[s]) for s in scenarios]
    axes[1].legend(handles=handles, frameon=False, loc="upper left", ncol=2, fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)


def write_summary() -> pd.DataFrame:
    rows = []
    for seed in [0, 1, 2]:
        p = DQN_2015 / f"medium_N_cost_seed{seed}_5000steps" / "event_summary.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append(
            {
                "source": "DQN",
                "scenario": f"dqn_seed{seed}",
                "label": LABELS[f"dqn_seed{seed}"],
                "yield_kg_ha": d["dqn_harvest_yield_kg_ha"],
                "biomass_kg_ha": d["daily_last_topwt"],
                "irrigation_mm": d["irrigation_total"],
                "nitrogen_kg_ha": d["nitrogen_total"],
                "economic_reward_total": d["economic_reward_total_eval"],
                "input_cost_total": d["input_cost_total_eval"],
                "max_water_stress": d["max_swfac_eval"],
                "max_nitrogen_stress": d["max_nstres_eval"],
            }
        )
    fixed = pd.read_csv(FIXED_DIR / "hla2015_fixed_action_counterfactual_summary.csv")
    for _, r in fixed.iterrows():
        rows.append(
            {
                "source": "fixed_counterfactual",
                "scenario": r["scenario"],
                "label": LABELS.get(r["scenario"], r["scenario"]),
                "yield_kg_ha": r["harvest_yield_kg_ha"],
                "biomass_kg_ha": r["daily_last_topwt"],
                "irrigation_mm": r["irrigation_total"],
                "nitrogen_kg_ha": r["nitrogen_total"],
                "economic_reward_total": r["economic_reward_total"],
                "input_cost_total": r["input_cost_total"],
                "max_water_stress": r["max_swfac"],
                "max_nitrogen_stress": r["max_nstres"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "hla2015_dqn_seed0_seed1_vs_fixed_summary.csv", index=False, encoding="utf-8-sig")
    return out


def summary_plot(summary: pd.DataFrame, out_path: Path) -> None:
    order = ["dqn_seed0", "dqn_seed1", "dqn_seed2", "fixed_I0_N0", "fixed_I120_N0", "fixed_I120_N50", "fixed_I120_N100", "fixed_I120_N150"]
    summary = summary.set_index("scenario").loc[order].reset_index()
    colors = [COLORS[s] for s in summary["scenario"]]
    labels = summary["label"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.1))
    fig.patch.set_facecolor("#FCFCFD")
    fig.suptitle("HLA 2015 DQN seeds vs fixed counterfactuals", x=0.06, ha="left", fontsize=15, fontweight="bold")
    axes[0].bar(labels, summary["yield_kg_ha"], color=colors, edgecolor="#333333", linewidth=0.5)
    axes[0].set_title("Yield")
    axes[0].set_ylabel("Harvest yield (kg/ha)")
    axes[0].tick_params(axis="x", rotation=35)
    style_axis(axes[0])
    axes[1].bar(labels, summary["economic_reward_total"], color=colors, edgecolor="#333333", linewidth=0.5)
    axes[1].set_title("Economic reward")
    axes[1].set_ylabel("Reward")
    axes[1].tick_params(axis="x", rotation=35)
    style_axis(axes[1])
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rain = load_rain()
    dqn = pd.concat([load_dqn(0), load_dqn(1), load_dqn(2)], ignore_index=True)
    dqn.to_csv(OUT_DIR / "hla2015_dqn_seed0_seed1_daily.csv", index=False, encoding="utf-8-sig")
    process_plot(dqn, rain, OUT_DIR / "hla2015_dqn_seed0_seed1_process.png")
    summary = write_summary()
    summary_plot(summary, OUT_DIR / "hla2015_dqn_seed0_seed1_vs_fixed_summary.png")
    pd.DataFrame(
        {
            "file_type": ["process_figure", "summary_figure", "daily_csv", "summary_csv"],
            "path": [
                str(OUT_DIR / "hla2015_dqn_seed0_seed1_process.png"),
                str(OUT_DIR / "hla2015_dqn_seed0_seed1_vs_fixed_summary.png"),
                str(OUT_DIR / "hla2015_dqn_seed0_seed1_daily.csv"),
                str(OUT_DIR / "hla2015_dqn_seed0_seed1_vs_fixed_summary.csv"),
            ],
        }
    ).to_csv(OUT_DIR / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    print(summary[["source", "scenario", "yield_kg_ha", "irrigation_mm", "nitrogen_kg_ha", "economic_reward_total"]].to_string(index=False))


if __name__ == "__main__":
    main()
