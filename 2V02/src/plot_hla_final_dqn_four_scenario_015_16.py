from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004"
SOURCE_DIR = BASE_DIR / "hla_2010_2015_four_scenario_with_ppo"
DQN_DIR = BASE_DIR / "hla_baseline_relative_dqn_checkpoint_015_12"
OUT_DIR = BASE_DIR / "hla_2010_2015_final_dqn_four_scenario_015_16"
FIG_DIR = OUT_DIR / "figures"


SCENARIOS = {
    "null": {
        "label": "Null",
        "color": "#222222",
        "linestyle": "-",
        "marker": "o",
    },
    "expert_2007_shifted": {
        "label": "Recorded expert",
        "color": "#C9252D",
        "linestyle": "--",
        "marker": "^",
    },
    "dssat_auto": {
        "label": "DSSAT auto",
        "color": "#B8860B",
        "linestyle": "-",
        "marker": "s",
    },
    "dqn": {
        "label": "DQN best checkpoint",
        "color": "#1F7A3A",
        "linestyle": "-",
        "marker": "D",
    },
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def best_checkpoint(year: int, seed: int) -> int:
    summary_path = DQN_DIR / str(year) / f"baseline_relative_seed{seed}_50000steps" / "checkpoint_summary.csv"
    summary = pd.read_csv(summary_path)
    ranked = summary.sort_values(
        ["total_reward", "final_grain_kg_ha", "final_biomass_kg_ha", "checkpoint_step"],
        ascending=[False, False, False, True],
    )
    return int(ranked.iloc[0]["checkpoint_step"])


def load_baseline_daily(year: int) -> pd.DataFrame:
    daily = pd.read_csv(SOURCE_DIR / "hla_2010_2015_four_scenario_daily.csv")
    daily["scenario"] = daily["scenario"].fillna("null").replace("", "null")
    daily = daily[daily["requested_year"].eq(year)].copy()
    daily = daily[daily["scenario"].isin(["null", "expert_2007_shifted", "dssat_auto"])].copy()
    daily = daily.rename(
        columns={
            "wspd": "water_stress",
            "nstd": "nitrogen_stress",
            "gwad": "grain_kg_ha",
            "cwad": "biomass_kg_ha",
        }
    )
    daily["source"] = "baseline"
    daily["checkpoint_step"] = np.nan
    daily["seed"] = np.nan
    return daily[
        [
            "requested_year",
            "scenario",
            "source",
            "seed",
            "checkpoint_step",
            "doy",
            "dap",
            "rain",
            "water_stress",
            "nitrogen_stress",
            "irrigation_mm",
            "fertilizer_kg_ha",
            "grain_kg_ha",
            "biomass_kg_ha",
        ]
    ]


def load_dqn_daily(year: int, seed: int) -> tuple[pd.DataFrame, int]:
    checkpoint = best_checkpoint(year, seed)
    path = DQN_DIR / str(year) / f"baseline_relative_seed{seed}_50000steps" / "dqn_eval_daily.csv"
    daily = pd.read_csv(path)
    daily = daily[daily["checkpoint_step"].eq(checkpoint)].copy()
    daily["requested_year"] = year
    daily["scenario"] = "dqn"
    daily["source"] = "baseline_relative_dqn"
    daily["seed"] = seed
    daily = daily.rename(
        columns={
            "swfac": "water_stress",
            "nstres": "nitrogen_stress",
            "grnwt": "grain_kg_ha",
            "topwt": "biomass_kg_ha",
        }
    )
    baseline_rain = (
        load_baseline_daily(year)
        .loc[lambda df: df["scenario"].eq("null"), ["dap", "rain"]]
        .drop_duplicates("dap")
    )
    daily = daily.drop(columns=["rain"], errors="ignore").merge(baseline_rain, on="dap", how="left")
    daily["rain"] = daily["rain"].fillna(0.0)
    return (
        daily[
            [
                "requested_year",
                "scenario",
                "source",
                "seed",
                "checkpoint_step",
                "doy",
                "dap",
                "rain",
                "water_stress",
                "nitrogen_stress",
                "irrigation_mm",
                "fertilizer_kg_ha",
                "grain_kg_ha",
                "biomass_kg_ha",
            ]
        ],
        checkpoint,
    )


def summarize(panel: pd.DataFrame, year: int, seed: int, checkpoint: int) -> pd.DataFrame:
    rain_total = float(
        panel.loc[panel["scenario"].eq("null"), ["dap", "rain"]]
        .drop_duplicates("dap")["rain"]
        .sum()
    )
    rows = []
    for scenario, sub in panel.groupby("scenario"):
        sub = sub.sort_values("dap")
        rows.append(
            {
                "requested_year": year,
                "scenario": scenario,
                "label": SCENARIOS[scenario]["label"],
                "dqn_seed": seed if scenario == "dqn" else np.nan,
                "dqn_checkpoint_step": checkpoint if scenario == "dqn" else np.nan,
                "days": int(sub["dap"].nunique()),
                "final_dap": float(sub["dap"].max()),
                "final_gwad": float(sub["grain_kg_ha"].dropna().iloc[-1]),
                "final_cwad": float(sub["biomass_kg_ha"].dropna().iloc[-1]),
                "rain_total": rain_total,
                "irrigation_total": float(sub["irrigation_mm"].fillna(0).sum()),
                "fertilizer_total": float(sub["fertilizer_kg_ha"].fillna(0).sum()),
                "max_water_stress": float(sub["water_stress"].max()),
                "max_nitrogen_stress": float(sub["nitrogen_stress"].max()),
            }
        )
    return pd.DataFrame(rows)


def plot_panel(panel: pd.DataFrame, year: int, seed: int, checkpoint: int, out_base: Path) -> None:
    max_dap = int(panel["dap"].max())
    rain = panel[panel["scenario"].eq("null")][["dap", "rain"]].drop_duplicates().sort_values("dap")

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(8.0, 8.8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.7, 1.0, 1.0, 0.9, 1.25]},
    )

    fig.suptitle(
        f"HLA {year}: four-scenario process comparison with DQN seed{seed} checkpoint {checkpoint // 1000}K",
        x=0.06,
        y=0.995,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )

    axes[0].bar(rain["dap"], rain["rain"], width=0.9, color="#BFC5CF", edgecolor="#87909C", linewidth=0.25)
    axes[0].set_ylabel("Rain\n(mm)")

    order = ["null", "expert_2007_shifted", "dssat_auto", "dqn"]
    for scenario in order:
        sub = panel[panel["scenario"].eq(scenario)].sort_values("dap")
        style = SCENARIOS[scenario]
        axes[1].plot(
            sub["dap"],
            sub["water_stress"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.7,
            label=style["label"],
        )
        axes[2].plot(
            sub["dap"],
            sub["nitrogen_stress"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.7,
            label=style["label"],
        )
        axes[4].plot(
            sub["dap"],
            sub["grain_kg_ha"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            label=style["label"],
        )
        axes[4].plot(
            sub["dap"],
            sub["biomass_kg_ha"],
            color=style["color"],
            linestyle=":",
            linewidth=1.5,
            alpha=0.95,
        )

        irrig = sub[sub["irrigation_mm"].fillna(0) > 0]
        fert = sub[sub["fertilizer_kg_ha"].fillna(0) > 0]
        if not irrig.empty:
            axes[3].vlines(
                irrig["dap"],
                0,
                irrig["irrigation_mm"],
                colors=style["color"],
                linestyles=style["linestyle"],
                linewidth=2.1,
                alpha=0.95,
            )
        if not fert.empty:
            axes[3].scatter(
                fert["dap"],
                fert["fertilizer_kg_ha"],
                marker=style["marker"],
                s=33,
                color=style["color"],
                edgecolor="white",
                linewidth=0.4,
                zorder=4,
            )

    axes[1].set_ylabel("Water\nstress")
    axes[2].set_ylabel("Nitrogen\nstress")
    axes[3].set_ylabel("Mgmt\namount")
    axes[4].set_ylabel("kg/ha")
    axes[4].set_xlabel("DAP")

    axes[1].legend(loc="upper left", ncol=2, fontsize=7)
    axes[3].set_title("Irrigation: vertical lines; fertilization: markers", loc="left", fontsize=8)
    axes[4].set_title("Solid/dashed lines: grain yield; dotted lines: aboveground biomass", loc="left", fontsize=8)

    for ax in axes:
        ax.set_xlim(-2, max_dap + 2)
        ax.grid(True, axis="x", color="#E2E7EF", linewidth=0.55)
        ax.grid(True, axis="y", color="#EDF1F5", linewidth=0.45, linestyle="--")
        ax.tick_params(length=2.5, width=0.6)

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(f"{out_base}.png", dpi=350, bbox_inches="tight")
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_daily = []
    all_summary = []
    for year in [2010, 2015]:
        base_daily = load_baseline_daily(year)
        for seed in [0, 1]:
            dqn_daily, checkpoint = load_dqn_daily(year, seed)
            panel = pd.concat([base_daily, dqn_daily], ignore_index=True)
            panel["dqn_selected_seed"] = seed
            panel["dqn_selected_checkpoint"] = checkpoint

            out_base = FIG_DIR / f"hla_{year}_four_scenario_final_dqn_seed{seed}_checkpoint_{checkpoint}"
            plot_panel(panel, year, seed, checkpoint, out_base)

            panel.to_csv(
                OUT_DIR / f"hla_{year}_four_scenario_final_dqn_seed{seed}_daily.csv",
                index=False,
                encoding="utf-8-sig",
            )
            summary = summarize(panel, year, seed, checkpoint)
            summary.to_csv(
                OUT_DIR / f"hla_{year}_four_scenario_final_dqn_seed{seed}_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )
            all_daily.append(panel)
            all_summary.append(summary)

    pd.concat(all_daily, ignore_index=True).to_csv(
        OUT_DIR / "hla_2010_2015_four_scenario_final_dqn_all_daily.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(all_summary, ignore_index=True).to_csv(
        OUT_DIR / "hla_2010_2015_four_scenario_final_dqn_all_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
