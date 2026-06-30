from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DQN_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_dqn_economic_reward_probe_012_03" / "2015"
FIXED_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2015_fixed_water_n0_scan_012_08"
FOUR_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_2010_2015_four_scenario_with_ppo"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2015_dqn_seed0_20k_012_10"


RUNS = {
    "seed0_5k": "medium_N_cost_seed0_5000steps",
    "seed0_20k": "medium_N_cost_seed0_20000steps",
    "seed1_5k": "medium_N_cost_seed1_5000steps",
    "seed2_5k": "medium_N_cost_seed2_5000steps",
}
COLORS = {
    "seed0_5k": "#464C55",
    "seed0_20k": "#CC6F47",
    "seed1_5k": "#386411",
    "seed2_5k": "#B23A48",
}
STYLES = {
    "seed0_5k": "--",
    "seed0_20k": "-",
    "seed1_5k": "-",
    "seed2_5k": "-.",
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


def load_run(label: str, run_dir: str) -> pd.DataFrame:
    df = pd.read_csv(DQN_DIR / run_dir / "dqn_economic_eval_daily.csv")
    return pd.DataFrame(
        {
            "scenario": label,
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


def write_summary() -> pd.DataFrame:
    rows = []
    for label, run_dir in RUNS.items():
        d = json.loads((DQN_DIR / run_dir / "event_summary.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "source": "DQN",
                "scenario": label,
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
    fixed = pd.read_csv(FIXED_DIR / "hla2015_fixed_water_n0_scan_summary.csv")
    for scenario in ["fixed_I60_N0", "fixed_I90_N0", "fixed_I120_N0"]:
        r = fixed[fixed["scenario"].eq(scenario)].iloc[0]
        rows.append(
            {
                "source": "fixed_N0_water_scan",
                "scenario": scenario.replace("fixed_", ""),
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "hla2015_dqn_seed0_20k_summary.csv", index=False, encoding="utf-8-sig")
    return out


def process_plot(data: pd.DataFrame, rain: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(15.8, 13.2),
        sharex=True,
        gridspec_kw={"height_ratios": [0.85, 1.0, 1.0, 1.0, 1.15], "hspace": 0.26},
    )
    fig.patch.set_facecolor("#FCFCFD")
    fig.subplots_adjust(top=0.91)
    fig.text(0.08, 0.965, "HLA 2015 DQN seed0 20K diagnostic", ha="left", va="top", fontsize=16, fontweight="bold")
    fig.text(0.08, 0.94, "Compare seed0 5K vs seed0 20K, with seed1 and seed2 5K as references.", ha="left", va="top", fontsize=9.5, color="#6F768A")
    axes[0].bar(rain["dap"], rain["rain"], width=1.0, color="#C5CAD3", edgecolor="#7A828F", linewidth=0.45)
    axes[0].set_ylabel("Rain\n(mm)")
    style_axis(axes[0])
    for ax, col, ylabel, title in [
        (axes[1], "water_stress", "Water\nstress", "Water stress index"),
        (axes[2], "nitrogen_stress", "Nitrogen\nstress", "Nitrogen stress index"),
    ]:
        for label in RUNS:
            sub = data[data["scenario"].eq(label)].sort_values("dap")
            ax.plot(sub["dap"], sub[col], color=COLORS[label], linestyle=STYLES[label], linewidth=2.1)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, 1.08)
        ax.set_title(title, loc="left", fontsize=10)
        style_axis(ax)
    for label in RUNS:
        sub = data[data["scenario"].eq(label)].sort_values("dap")
        irrig = sub[sub["irrigation_mm"].fillna(0) > 1e-6]
        fert = sub[sub["fertilizer_kg_ha"].fillna(0) > 1e-6]
        if not irrig.empty:
            axes[3].vlines(irrig["dap"], 0, irrig["irrigation_mm"], color=COLORS[label], linewidth=2.4)
        if not fert.empty:
            axes[3].scatter(fert["dap"], fert["fertilizer_kg_ha"], marker="^", s=52, color=COLORS[label], edgecolor="#FFFFFF", linewidth=0.6)
    axes[3].set_ylabel("Mgmt\namount")
    axes[3].set_ylim(-5, 58)
    axes[3].set_title("Management events: irrigation as vertical lines; fertilization as triangle markers", loc="left", fontsize=10)
    style_axis(axes[3])
    for label in RUNS:
        sub = data[data["scenario"].eq(label)].sort_values("dap")
        axes[4].plot(sub["dap"], sub["grain_kg_ha"], color=COLORS[label], linestyle=STYLES[label], linewidth=2.1)
        axes[4].plot(sub["dap"], sub["biomass_kg_ha"], color=COLORS[label], linestyle="--", linewidth=1.5, alpha=0.6)
    axes[4].set_ylabel("kg/ha")
    axes[4].set_xlabel("DAP")
    axes[4].set_title("Crop outcome: solid = grain weight, dashed = aboveground biomass", loc="left", fontsize=10)
    style_axis(axes[4])
    for ax in axes:
        ax.set_xlim(-2, 165)
        ax.set_xticks(range(0, 166, 25))
    handles = [Line2D([0], [0], color=COLORS[label], linestyle=STYLES[label], lw=2.1, label=label) for label in RUNS]
    axes[1].legend(handles=handles, frameon=False, loc="upper left", ncol=2, fontsize=10)
    fig.savefig(OUT_DIR / "hla2015_dqn_seed0_20k_process.png", dpi=280, bbox_inches="tight")
    plt.close(fig)


def summary_plot(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.0))
    fig.patch.set_facecolor("#FCFCFD")
    fig.suptitle("HLA 2015 seed0 20K vs references", x=0.06, ha="left", fontsize=15, fontweight="bold")
    labels = summary["scenario"].tolist()
    colors = ["#464C55", "#CC6F47", "#386411", "#B23A48", "#78A757", "#5477C4", "#2F5597"]
    axes[0].bar(labels, summary["yield_kg_ha"], color=colors, edgecolor="#333333", linewidth=0.5)
    axes[0].set_ylabel("Harvest yield (kg/ha)")
    axes[0].set_title("Yield")
    axes[0].tick_params(axis="x", rotation=35)
    style_axis(axes[0])
    axes[1].bar(labels, summary["economic_reward_total"], color=colors, edgecolor="#333333", linewidth=0.5)
    axes[1].set_ylabel("Economic reward")
    axes[1].set_title("Economic reward")
    axes[1].tick_params(axis="x", rotation=35)
    style_axis(axes[1])
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT_DIR / "hla2015_dqn_seed0_20k_summary.png", dpi=280, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.concat([load_run(label, run_dir) for label, run_dir in RUNS.items()], ignore_index=True)
    data.to_csv(OUT_DIR / "hla2015_dqn_seed0_20k_daily.csv", index=False, encoding="utf-8-sig")
    summary = write_summary()
    process_plot(data, load_rain())
    summary_plot(summary)
    pd.DataFrame(
        {
            "file_type": ["process_figure", "summary_figure", "daily_csv", "summary_csv"],
            "path": [
                str(OUT_DIR / "hla2015_dqn_seed0_20k_process.png"),
                str(OUT_DIR / "hla2015_dqn_seed0_20k_summary.png"),
                str(OUT_DIR / "hla2015_dqn_seed0_20k_daily.csv"),
                str(OUT_DIR / "hla2015_dqn_seed0_20k_summary.csv"),
            ],
        }
    ).to_csv(OUT_DIR / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    print(summary[["source", "scenario", "yield_kg_ha", "irrigation_mm", "nitrogen_kg_ha", "economic_reward_total"]].to_string(index=False))


if __name__ == "__main__":
    main()
