from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOUR_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_2010_2015_four_scenario_with_ppo"
DQN_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_dqn_economic_reward_probe_012_03" / "2010"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_dqn_economic_reward_probe_012_03" / "figures_012_04"


COLORS = {
    "null_zero": "#464C55",
    "expert_2007_shifted": "#CC6F47",
    "dssat_auto": "#5477C4",
    "dqn_economic_seed0": "#386411",
    "dqn_economic_seed1": "#804126",
}
LABELS = {
    "null_zero": "Null zero",
    "expert_2007_shifted": "Recorded expert shifted",
    "dssat_auto": "DSSAT auto irrigation + auto-N attempt",
    "dqn_economic_seed0": "Economic DQN seed0",
    "dqn_economic_seed1": "Economic DQN seed1",
}
STYLES = {
    "null_zero": "-",
    "expert_2007_shifted": "-",
    "dssat_auto": "-",
    "dqn_economic_seed0": "-",
    "dqn_economic_seed1": "--",
}


def style_axis(ax) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, axis="both", color="#E6E8F0", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D7DBE7")
    ax.spines["bottom"].set_color("#D7DBE7")
    ax.tick_params(colors="#464C55", labelsize=10)


def load_base_2010() -> pd.DataFrame:
    daily = pd.read_csv(FOUR_DIR / "hla_2010_2015_four_scenario_daily.csv", keep_default_na=False)
    daily["scenario"] = daily["scenario"].replace({"": "null_zero", "null": "null_zero"}).fillna("null_zero")
    daily = daily[daily["requested_year"].astype(int).eq(2010)].copy()
    daily = daily.rename(
        columns={
            "requested_year": "target_year",
            "wspd": "water_stress",
            "nstd": "nitrogen_stress",
            "gwad": "grain_kg_ha",
            "cwad": "biomass_kg_ha",
        }
    )
    return daily[
        [
            "target_year",
            "scenario",
            "dap",
            "rain",
            "water_stress",
            "nitrogen_stress",
            "grain_kg_ha",
            "biomass_kg_ha",
            "irrigation_mm",
            "fertilizer_kg_ha",
        ]
    ].copy()


def load_dqn_seed(seed: int, rain: pd.DataFrame) -> pd.DataFrame:
    case = DQN_DIR / f"medium_N_cost_seed{seed}_5000steps"
    df = pd.read_csv(case / "dqn_economic_eval_daily.csv")
    out = pd.DataFrame(
        {
            "target_year": 2010,
            "scenario": f"dqn_economic_seed{seed}",
            "dap": df["dap"].astype(float),
            "water_stress": df["swfac"].astype(float),
            "nitrogen_stress": df["nstres"].astype(float),
            "grain_kg_ha": df["grnwt"].astype(float),
            "biomass_kg_ha": df["topwt"].astype(float),
            "irrigation_mm": df["safe_amir"].astype(float),
            "fertilizer_kg_ha": df["safe_anfer"].astype(float),
        }
    )
    out = out.merge(rain[["dap", "rain"]].drop_duplicates("dap"), on="dap", how="left")
    out["rain"] = out["rain"].fillna(0.0)
    return out


def process_plot(data: pd.DataFrame, scenarios: list[str], title: str, subtitle: str, out_path: Path) -> None:
    max_dap = int(np.nanmax(data["dap"]))
    x_max = max_dap + 5
    x_ticks = np.arange(0, x_max + 1, 25)

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(15.8, 13.4),
        sharex=True,
        gridspec_kw={"height_ratios": [0.9, 1.0, 1.0, 1.0, 1.1], "hspace": 0.25},
    )
    fig.patch.set_facecolor("#FCFCFD")
    fig.subplots_adjust(top=0.925)
    fig.text(0.08, 0.975, title, ha="left", va="top", fontsize=16, color="#1F2430", fontweight="bold")
    fig.text(0.08, 0.955, subtitle, ha="left", va="top", fontsize=9.5, color="#6F768A")

    rain = data[["dap", "rain"]].drop_duplicates("dap").sort_values("dap")
    axes[0].bar(rain["dap"], rain["rain"], width=1.0, color="#C5CAD3", edgecolor="#7A828F", linewidth=0.45)
    axes[0].set_ylabel("Rain\n(mm)")
    style_axis(axes[0])

    for ax, col, ylabel, ttl in [
        (axes[1], "water_stress", "Water\nstress", "Water stress index"),
        (axes[2], "nitrogen_stress", "Nitrogen\nstress", "Nitrogen stress index"),
    ]:
        for scenario in scenarios:
            sub = data[data["scenario"].eq(scenario)].sort_values("dap")
            if sub.empty:
                continue
            ax.plot(sub["dap"], sub[col], color=COLORS[scenario], linestyle=STYLES[scenario], linewidth=2.0)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, max(1.05, float(data[col].max()) * 1.15))
        ax.set_title(ttl, loc="left", fontsize=10)
        style_axis(ax)

    for scenario in scenarios:
        if scenario == "null_zero":
            continue
        sub = data[data["scenario"].eq(scenario)].sort_values("dap")
        color = COLORS[scenario]
        irrig = sub[sub["irrigation_mm"].fillna(0) > 1e-6]
        fert = sub[sub["fertilizer_kg_ha"].fillna(0) > 1e-6]
        if not irrig.empty:
            axes[3].vlines(irrig["dap"], 0, irrig["irrigation_mm"], colors=color, linewidth=2.4, alpha=0.95)
        if not fert.empty:
            axes[3].scatter(fert["dap"], fert["fertilizer_kg_ha"], marker="^", s=48, color=color, edgecolor="#FFFFFF", linewidth=0.6, zorder=4)
    axes[3].set_ylabel("Mgmt\namount")
    max_mgmt = max(float(data["irrigation_mm"].max()), float(data["fertilizer_kg_ha"].max()), 10.0)
    axes[3].set_ylim(-max_mgmt * 0.05, max_mgmt * 1.2)
    axes[3].set_title("Management events: irrigation as vertical lines; fertilization as triangle markers", loc="left", fontsize=10)
    style_axis(axes[3])

    for scenario in scenarios:
        sub = data[data["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty:
            continue
        axes[4].plot(sub["dap"], sub["grain_kg_ha"], color=COLORS[scenario], linestyle=STYLES[scenario], linewidth=2.0)
        axes[4].plot(sub["dap"], sub["biomass_kg_ha"], color=COLORS[scenario], linestyle="--", linewidth=1.6, alpha=0.65)
    axes[4].set_ylabel("kg/ha")
    axes[4].set_xlabel("DAP")
    axes[4].set_title("Crop outcome: solid = grain weight, dashed = aboveground biomass", loc="left", fontsize=10)
    style_axis(axes[4])

    for ax in axes:
        ax.set_xlim(-2, x_max)
        ax.set_xticks(x_ticks)

    handles = [Line2D([0], [0], color=COLORS[s], lw=2.0, linestyle=STYLES[s], label=LABELS[s]) for s in scenarios]
    axes[1].legend(handles=handles, frameon=False, loc="upper left", ncol=2, fontsize=10)
    axes[3].legend(
        handles=[
            Line2D([0], [0], color="#464C55", marker="|", markersize=12, linestyle="None", label="Irrigation"),
            Line2D([0], [0], color="#464C55", marker="^", markersize=6, linestyle="None", label="Fertilizer"),
            Patch(facecolor="#C5CAD3", edgecolor="#7A828F", label="Rainfall"),
        ],
        frameon=False,
        loc="upper right",
        ncol=3,
        fontsize=9.5,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)


def write_summary(seed0: pd.DataFrame, seed1: pd.DataFrame, four: pd.DataFrame) -> None:
    rows = []
    for scenario, sub in four.groupby("scenario"):
        if scenario not in ["null_zero", "expert_2007_shifted", "dssat_auto"]:
            continue
        sub = sub.sort_values("dap")
        rows.append(
            {
                "scenario": scenario,
                "label": LABELS.get(scenario, scenario),
                "yield_kg_ha": float(sub["grain_kg_ha"].iloc[-1]),
                "biomass_kg_ha": float(sub["biomass_kg_ha"].iloc[-1]),
                "irrigation_mm": float(sub["irrigation_mm"].sum()),
                "nitrogen_kg_ha": float(sub["fertilizer_kg_ha"].sum()),
                "max_water_stress": float(sub["water_stress"].max()),
                "max_nitrogen_stress": float(sub["nitrogen_stress"].max()),
            }
        )
    for seed, sub in [(0, seed0), (1, seed1)]:
        sub = sub.sort_values("dap")
        summary_path = DQN_DIR / f"medium_N_cost_seed{seed}_5000steps" / "event_summary.json"
        d = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "scenario": f"dqn_economic_seed{seed}",
                "label": LABELS[f"dqn_economic_seed{seed}"],
                "yield_kg_ha": float(d["dqn_harvest_yield_kg_ha"]),
                "biomass_kg_ha": float(d["daily_last_topwt"]),
                "irrigation_mm": float(d["irrigation_total"]),
                "nitrogen_kg_ha": float(d["nitrogen_total"]),
                "max_water_stress": float(d["max_swfac_eval"]),
                "max_nitrogen_stress": float(d["max_nstres_eval"]),
                "economic_reward_total": float(d["economic_reward_total_eval"]),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "hla2010_economic_dqn_four_scenario_and_seed_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_2010()
    rain = base[base["scenario"].eq("null_zero")][["dap", "rain"]].drop_duplicates("dap")
    seed0 = load_dqn_seed(0, rain)
    seed1 = load_dqn_seed(1, rain)

    four_with_seed0 = pd.concat(
        [
            base[base["scenario"].isin(["null_zero", "expert_2007_shifted", "dssat_auto"])],
            seed0,
        ],
        ignore_index=True,
        sort=False,
    )
    seed_compare = pd.concat([seed0, seed1], ignore_index=True, sort=False)

    four_with_seed0.to_csv(OUT_DIR / "hla2010_four_scenario_with_economic_dqn_seed0_daily.csv", index=False, encoding="utf-8-sig")
    seed_compare.to_csv(OUT_DIR / "hla2010_economic_dqn_seed0_seed1_daily.csv", index=False, encoding="utf-8-sig")
    write_summary(seed0, seed1, base)

    process_plot(
        four_with_seed0,
        ["null_zero", "expert_2007_shifted", "dssat_auto", "dqn_economic_seed0"],
        "HLA 2010 four-scenario process plot with economic DQN",
        "DQN uses economic reward: ΔGRNWT - 1.0×I - 5.0×N; DSSAT auto label means auto-irrigation plus auto-N attempt.",
        OUT_DIR / "hla2010_four_scenario_with_economic_dqn_seed0_process.png",
    )
    process_plot(
        seed_compare,
        ["dqn_economic_seed0", "dqn_economic_seed1"],
        "HLA 2010 economic DQN seed stability plot",
        "Both runs use 5K timesteps and the same economic reward; compare yield, water, nitrogen, and timing.",
        OUT_DIR / "hla2010_economic_dqn_seed0_seed1_process.png",
    )

    manifest = pd.DataFrame(
        {
            "figure": [
                str(OUT_DIR / "hla2010_four_scenario_with_economic_dqn_seed0_process.png"),
                str(OUT_DIR / "hla2010_economic_dqn_seed0_seed1_process.png"),
            ]
        }
    )
    manifest.to_csv(OUT_DIR / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
