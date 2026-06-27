from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UFGA_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "UFGA_Windows480_vs_gym_pdi"
ANALYSIS_DIR = UFGA_DIR / "analysis_ufga_windows480_vs_pdi_with_rain"


def parse_mgmt_events(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^\s*(?P<run>\d+)\s+"
        r"(?P<mon>[A-Z]{3})\s+(?P<day>\d+),\s+(?P<year>\d{4})\s+"
        r"(?P<doy>\d+)\s+(?P<das>\d+)\s+(?P<dap>\d+)\s+"
        r"(?P<cr>\S+)\s+(?P<rest>.*)$"
    )
    with path.open("r", encoding="latin1", errors="ignore") as handle:
        for line in handle:
            match = pattern.match(line.rstrip("\n"))
            if not match:
                continue
            rest = match.group("rest")
            operation = None
            quantity = None
            unit = None
            if "Irrigation" in rest:
                operation = "Irrigation"
                q = re.search(r"Irrigation\s+([\d.]+)\s*mm", rest)
                if q:
                    quantity = float(q.group(1))
                    unit = "mm"
            elif "Fertilizer" in rest:
                operation = "Fertilizer"
                q = re.search(r"Fertilizer\s+([\d.]+)\s*kg\[N\]/ha", rest)
                if q:
                    quantity = float(q.group(1))
                    unit = "kgN/ha"
            elif "Planting" in rest:
                operation = "Planting"
            elif "Harvest Yield" in rest:
                operation = "Harvest"
                q = re.search(r"Harvest Yield\s+([\d.]+)\s*kg/ha", rest)
                if q:
                    quantity = float(q.group(1))
                    unit = "kg/ha"
            if operation is None:
                continue
            rows.append(
                {
                    "run": int(match.group("run")),
                    "year": int(match.group("year")),
                    "doy": int(match.group("doy")),
                    "das": int(match.group("das")),
                    "dap": int(match.group("dap")),
                    "operation": operation,
                    "quantity": quantity,
                    "unit": unit,
                    "raw": rest.strip(),
                }
            )
    return pd.DataFrame(rows)


def plot_management_stress(daily: pd.DataFrame, events: pd.DataFrame, runs: list[int], out_path: Path, title: str) -> None:
    colors = {
        "rain": "#C5CAD3",
        "irrigation": "#2E4780",
        "fertilizer": "#386411",
        "wspd": "#CC6F47",
        "nstd": "#8A3A6F",
    }
    fig, axes = plt.subplots(len(runs), 1, figsize=(13, 2.8 * len(runs)), sharex=True)
    if len(runs) == 1:
        axes = [axes]

    for ax, run in zip(axes, runs):
        sub = daily[daily["run"].eq(run)].sort_values("dap")
        ev = events[events["run"].eq(run)].copy()
        label = str(sub["treatment"].dropna().iloc[0]) if not sub.empty else f"Run {run}"

        ax2 = ax.twinx()
        ax2.bar(
            sub["dap"],
            sub["rain"],
            color=colors["rain"],
            alpha=0.40,
            width=1.0,
            label="Rainfall",
            zorder=1,
        )
        irr = ev[ev["operation"].eq("Irrigation")]
        fert = ev[ev["operation"].eq("Fertilizer")]
        if not irr.empty:
            ax2.bar(
                irr["dap"],
                irr["quantity"],
                color=colors["irrigation"],
                alpha=0.75,
                width=1.8,
                label="Irrigation",
                zorder=2,
            )
        if not fert.empty:
            ax2.scatter(
                fert["dap"],
                fert["quantity"],
                color=colors["fertilizer"],
                edgecolor="#1F2430",
                marker="^",
                s=55,
                label="Fertilizer N",
                zorder=4,
            )
            for _, row in fert.iterrows():
                ax2.vlines(row["dap"], 0, row["quantity"], color=colors["fertilizer"], linewidth=1.0, alpha=0.45, zorder=3)

        ax.plot(sub["dap"], sub["wspd"], color=colors["wspd"], linewidth=2.1, label="WSPD", zorder=5)
        ax.plot(sub["dap"], sub["nstd"], color=colors["nstd"], linewidth=2.1, linestyle=(0, (4, 2)), label="NSTD", zorder=6)
        ax.set_ylim(-0.03, max(1.0, pd.to_numeric(sub[["wspd", "nstd"]].max(), errors="coerce").max() * 1.08))
        ax2.set_ylim(0, max(110, pd.to_numeric(pd.concat([sub["rain"], irr["quantity"], fert["quantity"]]), errors="coerce").max() * 1.15))

        ax.set_title(f"Run {run}: {label}", loc="left", fontsize=11, color="#1F2430")
        ax.set_ylabel("Stress index")
        ax2.set_ylabel("Rain/Irr/Fert")
        ax.grid(True, axis="y", color="#E6E8F0", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)

    axes[-1].set_xlabel("DAP")
    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[0].twinx().get_legend_handles_labels()
    # Build an explicit legend from representative artists because twinx handles
    # from existing axes are awkward after multiple twins.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_items = [
        Patch(facecolor=colors["rain"], alpha=0.40, label="Rainfall (mm/day)"),
        Patch(facecolor=colors["irrigation"], alpha=0.75, label="Irrigation (mm)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=colors["fertilizer"], markeredgecolor="#1F2430", markersize=8, label="Fertilizer N (kg/ha)"),
        Line2D([0], [0], color=colors["wspd"], lw=2.1, label="WSPD"),
        Line2D([0], [0], color=colors["nstd"], lw=2.1, linestyle=(0, (4, 2)), label="NSTD"),
    ]
    fig.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=5, frameon=False, fontsize=9)
    fig.suptitle(title, y=1.005, fontsize=14)
    fig.text(
        0.01,
        0.985,
        "Left axis: DSSAT stress indices. Right axis: rainfall/irrigation/fertilizer event magnitudes. Lower WSPD/NSTD means stronger stress in DSSAT output convention.",
        fontsize=8.5,
        color="#6F768A",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    daily = pd.read_csv(ANALYSIS_DIR / "ufga_windows480_daily_stress_growth.csv")
    events = parse_mgmt_events(UFGA_DIR / "MgmtEvent.OUT")
    events.to_csv(ANALYSIS_DIR / "ufga_management_events_irrigation_fertilizer.csv", index=False, encoding="utf-8-sig")

    summary = (
        events[events["operation"].isin(["Irrigation", "Fertilizer"])]
        .pivot_table(index="run", columns="operation", values="quantity", aggfunc=["count", "sum"], fill_value=0)
    )
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    yields = daily.sort_values(["run", "dap"]).groupby("run", as_index=False).tail(1)[["run", "treatment", "gwad"]]
    summary = yields.merge(summary, on="run", how="left")
    summary.to_csv(ANALYSIS_DIR / "ufga_management_summary_by_treatment.csv", index=False, encoding="utf-8-sig")

    plot_management_stress(
        daily,
        events,
        runs=[1, 2, 3, 4, 5, 6],
        out_path=ANALYSIS_DIR / "ufga_rain_irrigation_fertilizer_stress_all_treatments.png",
        title="UFGA official example: management events aligned with WSPD/NSTD",
    )
    plot_management_stress(
        daily,
        events,
        runs=[3, 4],
        out_path=ANALYSIS_DIR / "ufga_run3_run4_irrigated_management_stress_focus.png",
        title="UFGA irrigated treatments: management events and stress response",
    )
    print(
        {
            "events_csv": str(ANALYSIS_DIR / "ufga_management_events_irrigation_fertilizer.csv"),
            "summary_csv": str(ANALYSIS_DIR / "ufga_management_summary_by_treatment.csv"),
            "all_treatments_png": str(ANALYSIS_DIR / "ufga_rain_irrigation_fertilizer_stress_all_treatments.png"),
            "run3_run4_focus_png": str(ANALYSIS_DIR / "ufga_run3_run4_irrigated_management_stress_focus.png"),
        }
    )


if __name__ == "__main__":
    main()
