from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOUR_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_2010_2015_four_scenario_with_ppo"
PPO_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_2015_official_reward_restart" / "ppo_smoke"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_2015_official_reward_restart" / "figures_011_13_windowed_ppo"


SCENARIO_ORDER = ["null_zero", "expert_2007_shifted", "dssat_auto", "ppo_windowed_seed0"]
SCENARIO_LABELS = {
    "null_zero": "Null zero",
    "expert_2007_shifted": "Recorded expert shifted",
    "dssat_auto": "DSSAT auto irrigation + auto-N attempt",
    "ppo_windowed_seed0": "Windowed PPO seed0",
}
SCENARIO_COLORS = {
    "null_zero": "#464C55",
    "expert_2007_shifted": "#CC6F47",
    "dssat_auto": "#5477C4",
    "ppo_windowed_seed0": "#386411",
}
SCENARIO_STYLES = {
    "null_zero": "-",
    "expert_2007_shifted": "-",
    "dssat_auto": "-",
    "ppo_windowed_seed0": "-",
}

WINDOW_LABELS = {
    "2010_windowed_seed0": "2010 seed0",
    "2010_windowed_seed1": "2010 seed1",
    "2015_windowed_seed0": "2015 seed0",
}
WINDOW_COLORS = {
    "2010_windowed_seed0": "#2E4780",
    "2010_windowed_seed1": "#804126",
    "2015_windowed_seed0": "#386411",
}
WINDOW_STYLES = {
    "2010_windowed_seed0": "-",
    "2010_windowed_seed1": "--",
    "2015_windowed_seed0": "-.",
}


def style_axis(ax) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, axis="both", color="#E6E8F0", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D7DBE7")
    ax.spines["bottom"].set_color("#D7DBE7")
    ax.tick_params(colors="#464C55", labelsize=10)


def load_four_daily() -> pd.DataFrame:
    daily = pd.read_csv(FOUR_DIR / "hla_2010_2015_four_scenario_daily.csv", keep_default_na=False)
    daily = daily.rename(
        columns={
            "requested_year": "target_year",
            "wspd": "water_stress",
            "nstd": "nitrogen_stress",
            "gwad": "grain_kg_ha",
            "cwad": "biomass_kg_ha",
        }
    )
    daily["scenario"] = daily["scenario"].replace({"": "null_zero", "null": "null_zero"}).fillna("null_zero")
    daily["source"] = "four_scenario_existing"
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
            "source",
        ]
    ].copy()


def load_rain_by_year(four_daily: pd.DataFrame) -> dict[int, pd.DataFrame]:
    rain = four_daily[four_daily["scenario"].eq("null_zero")][["target_year", "dap", "rain"]].copy()
    out = {}
    for year, sub in rain.groupby("target_year"):
        out[int(year)] = sub.sort_values("dap").drop_duplicates("dap")
    return out


def load_windowed_case(year: int, seed: int) -> pd.DataFrame:
    run_name = f"windowed_seed{seed}_5000steps"
    path = PPO_ROOT / str(year) / run_name / "ppo_smoke_eval_daily.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    scenario = f"ppo_windowed_seed{seed}"
    return pd.DataFrame(
        {
            "target_year": year,
            "scenario": scenario,
            "dap": df["dap"].astype(float),
            "water_stress": df["swfac"].astype(float),
            "nitrogen_stress": df["nstres"].astype(float),
            "grain_kg_ha": df["grnwt"].astype(float),
            "biomass_kg_ha": df["topwt"].astype(float),
            "irrigation_mm": df["safe_amir"].astype(float),
            "fertilizer_kg_ha": df["safe_anfer"].astype(float),
            "source": f"joint_windowed_5k_seed{seed}",
        }
    )


def add_rain(df: pd.DataFrame, rain_by_year: dict[int, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for year, sub in df.groupby("target_year"):
        rain = rain_by_year.get(int(year), pd.DataFrame(columns=["dap", "rain"]))[["dap", "rain"]]
        merged = sub.drop(columns=["rain"], errors="ignore").merge(rain, on="dap", how="left")
        merged["rain"] = merged["rain"].fillna(0.0)
        parts.append(merged)
    return pd.concat(parts, ignore_index=True) if parts else df


def harvest_yields_from_mgmt(path: Path) -> list[float]:
    if not path.exists():
        return []
    values = []
    for line in path.read_text(encoding="latin1", errors="ignore").splitlines():
        if "Harvest Yield" in line:
            m = re.search(r"Harvest Yield\s+([0-9.]+)", line)
            if m:
                values.append(float(m.group(1)))
    return values


def write_outputs(combined_four: pd.DataFrame, windowed_all: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_four.to_csv(OUT_DIR / "hla_2010_2015_four_scenario_with_windowed_ppo_daily.csv", index=False, encoding="utf-8-sig")
    windowed_all.to_csv(OUT_DIR / "hla_windowed_ppo_seed_year_daily.csv", index=False, encoding="utf-8-sig")

    rows = []
    for year, seed in [(2010, 0), (2010, 1), (2015, 0)]:
        case = PPO_ROOT / str(year) / f"windowed_seed{seed}_5000steps"
        vals = harvest_yields_from_mgmt(case / "pdi_tmp_snapshot_eval" / "MgmtEvent.OUT")
        sub = load_windowed_case(year, seed)
        rows.append(
            {
                "year": year,
                "seed": seed,
                "null_harvest_yield_kg_ha_from_mgmtevent": vals[0] if vals else np.nan,
                "ppo_harvest_yield_kg_ha_from_mgmtevent": vals[-1] if vals else np.nan,
                "daily_last_grain_kg_ha": float(sub["grain_kg_ha"].dropna().iloc[-1]),
                "daily_last_biomass_kg_ha": float(sub["biomass_kg_ha"].dropna().iloc[-1]),
                "irrigation_total_mm": float(sub["irrigation_mm"].sum()),
                "fertilizer_total_kg_ha": float(sub["fertilizer_kg_ha"].sum()),
                "first_irrigation_dap": float(sub.loc[sub["irrigation_mm"] > 1e-6, "dap"].min()),
                "first_fertilizer_dap": float(sub.loc[sub["fertilizer_kg_ha"] > 1e-6, "dap"].min()),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "hla_windowed_ppo_seed_year_summary.csv", index=False, encoding="utf-8-sig")


def line_plot(ax, data: pd.DataFrame, scenarios: list[str], y_col: str, colors: dict, styles: dict, labels: dict) -> None:
    for scenario in scenarios:
        sub = data[data["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty:
            continue
        ax.plot(
            sub["dap"],
            sub[y_col],
            color=colors[scenario],
            linestyle=styles.get(scenario, "-"),
            linewidth=2.0,
            label=labels[scenario],
        )


def management_plot(ax, data: pd.DataFrame, scenarios: list[str], colors: dict) -> None:
    for scenario in scenarios:
        if scenario == "null_zero":
            continue
        sub = data[data["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty:
            continue
        color = colors[scenario]
        irrig = sub[sub["irrigation_mm"].fillna(0) > 1e-6]
        fert = sub[sub["fertilizer_kg_ha"].fillna(0) > 1e-6]
        if not irrig.empty:
            ax.vlines(irrig["dap"], 0, irrig["irrigation_mm"], colors=color, linewidth=2.4, alpha=0.95)
        if not fert.empty:
            ax.scatter(
                fert["dap"],
                fert["fertilizer_kg_ha"],
                marker="^",
                s=48,
                color=color,
                edgecolor="#FFFFFF",
                linewidth=0.6,
                zorder=4,
            )


def process_plot(
    data: pd.DataFrame,
    scenarios: list[str],
    colors: dict,
    styles: dict,
    labels: dict,
    title: str,
    subtitle: str,
    out_path: Path,
) -> None:
    max_dap = int(np.nanmax(data["dap"])) if not data.empty else 170
    x_max = max(10, max_dap + 5)
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

    rain = data[["dap", "rain"]].drop_duplicates("dap").sort_values("dap")
    axes[0].bar(rain["dap"], rain["rain"].fillna(0), width=1.0, color="#C5CAD3", edgecolor="#7A828F", linewidth=0.45)
    axes[0].set_ylabel("Rain\n(mm)")
    fig.text(0.08, 0.975, title, ha="left", va="top", fontsize=16, color="#1F2430", fontweight="bold")
    fig.text(0.08, 0.955, subtitle, ha="left", va="top", fontsize=9.5, color="#6F768A")
    style_axis(axes[0])

    line_plot(axes[1], data, scenarios, "water_stress", colors, styles, labels)
    axes[1].set_ylabel("Water\nstress")
    axes[1].set_ylim(-0.05, max(1.05, float(data["water_stress"].max()) * 1.15))
    axes[1].set_title("Water stress index", loc="left", fontsize=10)
    style_axis(axes[1])

    line_plot(axes[2], data, scenarios, "nitrogen_stress", colors, styles, labels)
    axes[2].set_ylabel("Nitrogen\nstress")
    axes[2].set_ylim(-0.05, max(1.05, float(data["nitrogen_stress"].max()) * 1.15))
    axes[2].set_title("Nitrogen stress index", loc="left", fontsize=10)
    style_axis(axes[2])

    management_plot(axes[3], data, scenarios, colors)
    axes[3].set_ylabel("Mgmt\namount")
    max_mgmt = max(float(data["irrigation_mm"].max()), float(data["fertilizer_kg_ha"].max()), 10.0)
    axes[3].set_ylim(-max_mgmt * 0.05, max_mgmt * 1.2)
    axes[3].set_title("Management events: irrigation as vertical lines; fertilization as triangle markers", loc="left", fontsize=10)
    style_axis(axes[3])

    for scenario in scenarios:
        sub = data[data["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty:
            continue
        axes[4].plot(sub["dap"], sub["grain_kg_ha"], color=colors[scenario], linestyle=styles.get(scenario, "-"), linewidth=2.0)
        axes[4].plot(sub["dap"], sub["biomass_kg_ha"], color=colors[scenario], linestyle="--", linewidth=1.6, alpha=0.65)
    axes[4].set_ylabel("kg/ha")
    axes[4].set_xlabel("DAP")
    axes[4].set_title("Crop outcome: solid = grain weight, dashed = aboveground biomass", loc="left", fontsize=10)
    style_axis(axes[4])

    for ax in axes:
        ax.set_xlim(-2, x_max)
        ax.set_xticks(x_ticks)

    line_handles = [
        Line2D([0], [0], color=colors[s], lw=2.0, linestyle=styles.get(s, "-"), label=labels[s])
        for s in scenarios
        if not data[data["scenario"].eq(s)].empty
    ]
    axes[1].legend(handles=line_handles, frameon=False, loc="upper left", ncol=2, fontsize=10)
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    four = load_four_daily()
    rain_by_year = load_rain_by_year(four)

    # Build current windowed PPO daily data.
    ppo_2010_s0 = add_rain(load_windowed_case(2010, 0), rain_by_year)
    ppo_2010_s1 = add_rain(load_windowed_case(2010, 1), rain_by_year)
    ppo_2015_s0 = add_rain(load_windowed_case(2015, 0), rain_by_year)
    windowed_all = pd.concat(
        [
            ppo_2010_s0.assign(scenario="2010_windowed_seed0"),
            ppo_2010_s1.assign(scenario="2010_windowed_seed1"),
            ppo_2015_s0.assign(scenario="2015_windowed_seed0"),
        ],
        ignore_index=True,
    )

    # Four-scenario figures use windowed PPO seed0 as the PPO scenario.
    ppo_four = pd.concat(
        [
            ppo_2010_s0.assign(scenario="ppo_windowed_seed0"),
            ppo_2015_s0.assign(scenario="ppo_windowed_seed0"),
        ],
        ignore_index=True,
    )
    base_four = four[four["scenario"].isin(["null_zero", "expert_2007_shifted", "dssat_auto"])].copy()
    combined_four = pd.concat([base_four, ppo_four], ignore_index=True, sort=False)
    write_outputs(combined_four, windowed_all)

    outputs = []
    for year in [2010, 2015]:
        data = combined_four[combined_four["target_year"].eq(year)].copy()
        outputs.append(OUT_DIR / f"hla_{year}_four_scenario_with_windowed_ppo_process.png")
        process_plot(
            data=data,
            scenarios=SCENARIO_ORDER,
            colors=SCENARIO_COLORS,
            styles=SCENARIO_STYLES,
            labels=SCENARIO_LABELS,
            title=f"HLA {year} four-scenario process plot with windowed PPO",
            subtitle="Same DAP axis; PPO is current joint_windowed 5K seed0; DSSAT auto label means auto-irrigation plus auto-N attempt.",
            out_path=outputs[-1],
        )

    # Seed/year windowed PPO diagnostic: separate 2010 and 2015 rain patterns make one combined plot misleading.
    data_2010 = pd.concat(
        [
            ppo_2010_s0.assign(scenario="2010_windowed_seed0"),
            ppo_2010_s1.assign(scenario="2010_windowed_seed1"),
        ],
        ignore_index=True,
    )
    outputs.append(OUT_DIR / "hla_2010_windowed_ppo_seed0_seed1_process.png")
    process_plot(
        data=data_2010,
        scenarios=["2010_windowed_seed0", "2010_windowed_seed1"],
        colors=WINDOW_COLORS,
        styles=WINDOW_STYLES,
        labels=WINDOW_LABELS,
        title="HLA 2010 windowed PPO seed stability plot",
        subtitle="Both runs are joint_windowed 5K under I120/N150; expected check is no DAP2 action and seed-consistent timing.",
        out_path=outputs[-1],
    )

    data_2015 = ppo_2015_s0.assign(scenario="2015_windowed_seed0")
    outputs.append(OUT_DIR / "hla_2015_windowed_ppo_seed0_process.png")
    process_plot(
        data=data_2015,
        scenarios=["2015_windowed_seed0"],
        colors=WINDOW_COLORS,
        styles=WINDOW_STYLES,
        labels=WINDOW_LABELS,
        title="HLA 2015 windowed PPO seed0 process plot",
        subtitle="Single-seed diagnostic for transfer to a second year; seed1 is still needed for stability confirmation.",
        out_path=outputs[-1],
    )

    manifest = pd.DataFrame({"figure": [str(p) for p in outputs]})
    manifest.to_csv(OUT_DIR / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    print("\n".join(str(p) for p in outputs))


if __name__ == "__main__":
    main()
