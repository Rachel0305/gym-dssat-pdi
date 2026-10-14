from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "lc_fixed_input_year_screening_017_11"
FIG_DIR = OUT_DIR / "figures"
INPUT_RUNS = OUT_DIR / "runs"


COLORS = {
    "null": "#222222",
    "recorded": "#D04A3A",
    "dssat_auto": "#C49A00",
}
STYLES = {
    "null": "-",
    "recorded": "--",
    "dssat_auto": "-",
}
LABELS = {
    "null": "Null",
    "recorded": "Recorded expert",
    "dssat_auto": "DSSAT auto",
}


def setup_style() -> None:
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
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def read_weather(year: int) -> pd.DataFrame:
    wth = INPUT_RUNS / str(year) / "null" / "input" / f"CNLC{year % 100:02d}01.WTH"
    rows = []
    with wth.open("r", encoding="latin-1", errors="ignore") as f:
        in_data = False
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("@") and "DATE" in line and "RAIN" in line:
                in_data = True
                continue
            if not in_data:
                continue
            parts = line.split()
            if len(parts) >= 5 and parts[0].isdigit():
                date = int(parts[0])
                rows.append({"doy": date % 1000, "rain": float(parts[4])})
    return pd.DataFrame(rows)


def infer_planting_doy(daily: pd.DataFrame) -> int:
    valid = daily[pd.to_numeric(daily["doy"], errors="coerce").notna()].copy()
    valid["doy"] = pd.to_numeric(valid["doy"], errors="coerce")
    valid["dap"] = pd.to_numeric(valid["dap"], errors="coerce")
    valid = valid.dropna(subset=["doy", "dap"])
    if valid.empty:
        return 150
    return int((valid["doy"] - valid["dap"]).median())


def plot_year(year: int, daily: pd.DataFrame, events: pd.DataFrame, summary: pd.DataFrame) -> None:
    sub = daily[daily["requested_year"].astype(int).eq(year)].copy()
    sub["scenario"] = sub["scenario"].fillna("").replace("", "null")
    for col in ["dap", "doy", "swfac", "nstres", "grnwt", "topwt"]:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna(subset=["dap"])

    ev = events[events["requested_year"].astype(int).eq(year)].copy()
    ev["scenario"] = ev["scenario"].fillna("").replace("", "null")
    ev["dap"] = pd.to_numeric(ev["dap"], errors="coerce")
    ev["amount"] = pd.to_numeric(ev["amount"], errors="coerce")

    summ = summary[summary["year"].astype(int).eq(year)].copy()

    rain = read_weather(year)
    pdate = infer_planting_doy(sub)
    rain["dap"] = rain["doy"] - pdate
    max_dap = max(120, int(sub["dap"].max() if not sub.empty else 120))
    rain = rain[(rain["dap"] >= 0) & (rain["dap"] <= max_dap + 5)]

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(8.3, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": [0.8, 1.1, 1.1, 1.1, 1.2]},
    )
    fig.suptitle(f"LC {year} fixed-input baseline screening", x=0.08, ha="left", fontsize=11, fontweight="bold")

    axes[0].bar(rain["dap"], rain["rain"], width=1.0, color="#C9CED8", edgecolor="#7B8491", linewidth=0.25)
    axes[0].set_ylabel("Rain\n(mm)")

    for scenario in ["null", "recorded", "dssat_auto"]:
        s = sub[sub["scenario"].eq(scenario)].sort_values("dap")
        if s.empty:
            continue
        axes[1].plot(s["dap"], s["swfac"], color=COLORS[scenario], ls=STYLES[scenario], lw=1.8, label=LABELS[scenario])
        axes[2].plot(s["dap"], s["nstres"], color=COLORS[scenario], ls=STYLES[scenario], lw=1.8, label=LABELS[scenario])
        axes[4].plot(s["dap"], s["grnwt"], color=COLORS[scenario], ls=STYLES[scenario], lw=1.8)
        axes[4].plot(s["dap"], s["topwt"], color=COLORS[scenario], ls=STYLES[scenario], lw=1.2, alpha=0.55)

        sev = ev[ev["scenario"].eq(scenario)]
        irr = sev[sev["unit"].astype(str).str.contains("mm", na=False)]
        fert = sev[sev["unit"].astype(str).str.contains("kg", na=False)]
        if not irr.empty:
            axes[3].vlines(irr["dap"], 0, irr["amount"], color=COLORS[scenario], linestyles=STYLES[scenario], lw=2.0)
        if not fert.empty:
            axes[3].scatter(
                fert["dap"],
                fert["amount"],
                marker="^",
                s=34,
                color=COLORS[scenario],
                edgecolor="white",
                linewidth=0.35,
                zorder=4,
            )

    axes[1].set_ylabel("Water\nstress")
    axes[2].set_ylabel("Nitrogen\nstress")
    axes[3].set_ylabel("Mgmt\namount")
    axes[4].set_ylabel("kg/ha")
    axes[4].set_xlabel("DAP")
    axes[4].set_title("solid/faint same colour: grain weight and aboveground biomass", loc="left", fontsize=8)

    axes[1].legend(ncol=3, loc="upper left", fontsize=8)
    for ax in axes:
        ax.grid(True, color="#E8EBF2", linewidth=0.6)
        ax.set_xlim(0, max_dap + 5)

    if not summ.empty:
        label_parts = []
        for scenario in ["null", "recorded", "dssat_auto"]:
            row = summ[summ["scenario"].fillna("").replace("", "null").eq(scenario)]
            if not row.empty:
                r = row.iloc[0]
                label_parts.append(
                    f"{LABELS[scenario]}: GWAD {float(r['final_gwad']):.0f}, I {float(r['event_irrigation_total']):.1f}, N {float(r['event_fertilizer_total']):.1f}"
                )
        axes[4].text(0.01, -0.38, " | ".join(label_parts), transform=axes[4].transAxes, fontsize=7, va="top")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = FIG_DIR / f"LC_{year}_fixed_input_baseline_process"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    daily = pd.read_csv(OUT_DIR / "017_11_lc_fixed_input_daily.csv", keep_default_na=False)
    events = pd.read_csv(OUT_DIR / "017_11_lc_fixed_input_events.csv", keep_default_na=False)
    summary = pd.read_csv(OUT_DIR / "017_11_lc_fixed_input_summary.csv", keep_default_na=False)
    for year in sorted(summary["year"].dropna().astype(int).unique()):
        plot_year(year, daily, events, summary)
    print(f"[done] figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
