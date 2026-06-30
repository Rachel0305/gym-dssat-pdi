from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_2010_2015_four_scenario_with_ppo"
FIG_DIR = OUT_DIR / "figures_split" / "process_style"

SCENARIOS = ["null", "expert_2007_shifted", "dssat_auto", "ppo_00906_seed0"]
LABELS = {
    "null": "Null",
    "expert_2007_shifted": "Recorded expert shifted",
    "dssat_auto": "DSSAT auto irrigation + auto-N attempt",
    "ppo_00906_seed0": "PPO 009_06 seed0",
}
COLORS = {
    "null": "#4B5563",
    "expert_2007_shifted": "#D97745",
    "dssat_auto": "#5B7FD1",
    "ppo_00906_seed0": "#3F7D20",
}
LINESTYLES = {
    "null": "-",
    "expert_2007_shifted": "-",
    "dssat_auto": "-",
    "ppo_00906_seed0": "-",
}


def _style_axis(ax) -> None:
    ax.grid(True, axis="both", color="#E9EDF5", linewidth=0.8, alpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D5DAE3")
    ax.spines["bottom"].set_color("#D5DAE3")
    ax.tick_params(colors="#4A4A4A")


def _present_scenarios(year_data: pd.DataFrame) -> list[str]:
    present = set(year_data["scenario"].dropna().astype(str).unique())
    return [scenario for scenario in SCENARIOS if scenario in present]


def _plot_lines(ax, year_data: pd.DataFrame, y_col: str, scenarios: list[str]) -> None:
    for scenario in scenarios:
        sub = year_data[year_data["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty or y_col not in sub:
            continue
        ax.plot(
            sub["dap"],
            sub[y_col],
            color=COLORS[scenario],
            linestyle=LINESTYLES[scenario],
            linewidth=2.0,
            label=LABELS[scenario],
        )


def _plot_management(ax, year_data: pd.DataFrame, scenarios: list[str]) -> None:
    for scenario in scenarios:
        if scenario == "null":
            continue
        sub = year_data[year_data["scenario"].eq(scenario)].sort_values("dap").copy()
        if sub.empty:
            continue
        irrig = sub[sub["irrigation_mm"].fillna(0) > 1e-6]
        fert = sub[sub["fertilizer_kg_ha"].fillna(0) > 1e-6]
        color = COLORS[scenario]
        if not irrig.empty:
            ax.vlines(
                irrig["dap"],
                ymin=0,
                ymax=irrig["irrigation_mm"],
                colors=color,
                linewidth=2.5,
                alpha=0.9,
            )
        if not fert.empty:
            ax.scatter(
                fert["dap"],
                fert["fertilizer_kg_ha"],
                marker="^",
                s=38,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )


def plot_year(daily: pd.DataFrame, summary: pd.DataFrame, year: int) -> Path:
    year_data = daily[daily["requested_year"].eq(year)].copy()
    if year_data.empty:
        raise RuntimeError(f"No daily data for {year}")
    scenarios = _present_scenarios(year_data)

    max_dap = int(np.nanmax(year_data["dap"])) if "dap" in year_data else 170
    x_max = max(10, max_dap + 5)
    x_ticks = np.arange(0, x_max + 1, 25)

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(15.5, 13.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.0, 1.05], "hspace": 0.24},
    )

    rain = year_data[year_data["scenario"].eq("null")].sort_values("dap")
    if not rain.empty:
        axes[0].bar(
            rain["dap"],
            rain["rain"].fillna(0),
            width=1.0,
            color="#C9CED8",
            edgecolor="#AEB6C2",
            linewidth=0.6,
            alpha=0.95,
            label="Rainfall",
        )
    axes[0].set_ylabel("Rain\n(mm)")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set_title(f"HLA {year} four-scenario process plot", loc="left", fontsize=15, fontweight="bold", pad=8)
    _style_axis(axes[0])

    _plot_lines(axes[1], year_data, "wspd", scenarios)
    axes[1].set_ylabel("Water\nstress")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Water stress index by scenario", loc="left", fontsize=10)
    _style_axis(axes[1])

    _plot_lines(axes[2], year_data, "nstd", scenarios)
    axes[2].set_ylabel("Nitrogen\nstress")
    axes[2].set_ylim(-0.05, max(1.05, float(year_data["nstd"].max()) * 1.12 if "nstd" in year_data else 1.05))
    axes[2].set_title("Nitrogen stress index by scenario", loc="left", fontsize=10)
    _style_axis(axes[2])

    _plot_management(axes[3], year_data, scenarios)
    axes[3].set_ylabel("Mgmt\namount")
    max_mgmt = max(
        float(year_data["irrigation_mm"].max()) if "irrigation_mm" in year_data else 0,
        float(year_data["fertilizer_kg_ha"].max()) if "fertilizer_kg_ha" in year_data else 0,
        10.0,
    )
    axes[3].set_ylim(-max_mgmt * 0.05, max_mgmt * 1.22)
    axes[3].set_title("Management events: irrigation as vertical lines; fertilization as triangle markers", loc="left", fontsize=10)
    _style_axis(axes[3])

    for scenario in scenarios:
        sub = year_data[year_data["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty:
            continue
        axes[4].plot(sub["dap"], sub["gwad"], color=COLORS[scenario], linewidth=2.0)
        axes[4].plot(sub["dap"], sub["cwad"], color=COLORS[scenario], linewidth=1.7, linestyle="--", alpha=0.65)
    axes[4].set_ylabel("kg/ha")
    axes[4].set_title("Crop outcome: solid = grain weight, dashed = aboveground biomass", loc="left", fontsize=10)
    axes[4].set_xlabel("DAP")
    _style_axis(axes[4])

    for ax in axes:
        ax.set_xlim(-2, x_max)
        ax.set_xticks(x_ticks)

    line_handles = [
        Line2D([0], [0], color=COLORS[s], lw=2.0, linestyle=LINESTYLES[s], label=LABELS[s])
        for s in scenarios
    ]
    mgmt_handles = [
        Line2D([0], [0], color="#333333", marker="|", markersize=12, linestyle="None", label="Irrigation event"),
        Line2D([0], [0], color="#333333", marker="^", markersize=6, linestyle="None", label="Fertilization event"),
        Patch(facecolor="#C9CED8", edgecolor="#AEB6C2", label="Rainfall"),
    ]
    axes[1].legend(handles=line_handles, frameon=False, loc="upper left", ncol=2)
    axes[3].legend(handles=mgmt_handles[:2], frameon=False, loc="upper right", ncol=2)

    missing = [s for s in SCENARIOS if s not in scenarios]
    missing_note = f" Missing in source CSV: {', '.join(missing)}." if missing else ""
    note = (
        "All panels share the same DAP axis. "
        "Existing forward-simulation results only; no new DSSAT/PPO run."
        + missing_note
    )
    fig.text(0.08, 0.985, note, ha="left", va="top", fontsize=9, color="#7A828F")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"hla_{year}_process_style_aligned.png"
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    daily_path = OUT_DIR / "hla_2010_2015_four_scenario_daily.csv"
    summary_path = OUT_DIR / "hla_2010_2015_four_scenario_summary.csv"
    # Important: pandas treats the literal string "null" as NA by default.
    # Here "null" is a valid scenario name, so keep_default_na must be False.
    daily = pd.read_csv(daily_path, keep_default_na=False)
    summary = pd.read_csv(summary_path, keep_default_na=False) if summary_path.exists() else pd.DataFrame()
    outputs = []
    for year in sorted(daily["requested_year"].dropna().astype(int).unique()):
        outputs.append(plot_year(daily, summary, year))
    manifest = pd.DataFrame({"figure": [str(p) for p in outputs]})
    manifest.to_csv(FIG_DIR / "process_style_manifest.csv", index=False, encoding="utf-8-sig")
    print("\n".join(str(p) for p in outputs))


if __name__ == "__main__":
    main()
