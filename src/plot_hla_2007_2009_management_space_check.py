"""Plot HLA 2007/2009 management-space check outputs.

Reads already completed forward simulations and creates one combined process
figure per year:
- rainfall
- water and nitrogen stress indices
- irrigation and fertilization events
- grain yield and biomass

No DSSAT/PDI simulation is run here.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_2007_2009_management_space_check"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

YEAR_TO_PDATE_DOY = {2007: 125, 2009: 122}
YEAR_TO_WTH = {
    2007: OUT_DIR / "runs" / "2007_null" / "input" / "CNHL0701.WTH",
    2009: OUT_DIR / "runs" / "2009_null" / "input" / "CNHL0901.WTH",
}
SCENARIOS = ["null", "recorded", "dssat_auto"]
LABELS = {
    "null": "Null",
    "recorded": "Recorded management",
    "dssat_auto": "DSSAT auto",
}
COLORS = {
    "null": "#111111",
    "recorded": "#1F77B4",
    "dssat_auto": "#D62728",
}
LINESTYLES = {
    "null": "-",
    "recorded": "--",
    "dssat_auto": "-.",
}


def read_weather_rain(path: Path, year: int) -> pd.DataFrame:
    header = None
    rows = []
    for raw in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        stripped = raw.strip()
        if stripped.startswith("@") and "DATE" in stripped and "RAIN" in stripped:
            header = stripped.replace("@", "", 1).split()
            continue
        if not header or not stripped or not stripped[0].isdigit():
            continue
        parts = stripped.split()
        row = dict(zip(header, parts))
        date = int(row["DATE"])
        if date // 1000 != year:
            continue
        doy = date % 1000
        rows.append({"year": year, "doy": doy, "dap": doy - YEAR_TO_PDATE_DOY[year], "rain": float(row.get("RAIN", 0.0))})
    return pd.DataFrame(rows)


def parse_events_from_raw(year: int, scenario: str) -> pd.DataFrame:
    path = OUT_DIR / "runs" / f"{year}_{scenario}" / "pdi_tmp_snapshot" / "MgmtEvent.OUT"
    rows = []
    if not path.exists():
        return pd.DataFrame(columns=["year", "scenario", "dap", "operation", "amount", "unit"])
    pattern = re.compile(
        r"^\s*\d+\s+\w+\s+\d+,\s+\d{4}\s+\d+\s+\d+\s+(-?\d+)\s+\w+\s+(.+?)\s+([-+]?\d+(?:\.\d*)?)\s+(\S+)"
    )
    for raw in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        if "Irrigation" not in raw and "Fertil" not in raw and "Nitrogen" not in raw:
            continue
        m = pattern.match(raw)
        if not m:
            continue
        rows.append(
            {
                "year": year,
                "scenario": scenario,
                "dap": int(m.group(1)),
                "operation": m.group(2).strip(),
                "amount": float(m.group(3)),
                "unit": m.group(4),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["year", "scenario", "dap", "operation", "amount", "unit"])
    return out.drop_duplicates(["year", "scenario", "dap", "operation", "amount", "unit"], keep="first")


def build_corrected_summary(daily: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in sorted(daily["year"].dropna().unique()):
        for scenario in SCENARIOS:
            sub = daily[(daily["year"].eq(year)) & (daily["scenario"].eq(scenario))].sort_values("dap")
            if sub.empty:
                continue
            ev = events[(events["year"].eq(year)) & (events["scenario"].eq(scenario))]
            irrig = ev[ev["operation"].str.contains("Irrigation", case=False, na=False)] if not ev.empty else pd.DataFrame()
            fert = ev[ev["operation"].str.contains("Fertil|Nitrogen", case=False, na=False)] if not ev.empty else pd.DataFrame()
            final = sub.iloc[-1]
            rows.append(
                {
                    "year": int(year),
                    "scenario": scenario,
                    "final_dap": float(final["dap"]),
                    "gwad": float(final["gwad"]),
                    "cwad": float(final["cwad"]),
                    "irrigation_events": len(irrig),
                    "total_irrigation_mm": float(irrig["amount"].sum()) if not irrig.empty else 0.0,
                    "fertilizer_events": len(fert),
                    "total_n_kg_ha": float(fert["amount"].sum()) if not fert.empty else 0.0,
                    "max_wspd": float(sub["wspd"].max()),
                    "mean_wspd": float(sub["wspd"].mean()),
                    "max_nstd": float(sub["nstd"].max()),
                    "mean_nstd": float(sub["nstd"].mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_year(year: int, daily: pd.DataFrame, events: pd.DataFrame) -> Path:
    rain = read_weather_rain(YEAR_TO_WTH[year], year)
    ymax = max(1.0, daily[daily["year"].eq(year)]["dap"].max())
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(14, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.2, 1.2, 1.15, 1.45]},
    )

    axes[0].bar(rain["dap"], rain["rain"], color="#BDBDBD", edgecolor="#555555", linewidth=0.25, width=1.0)
    axes[0].set_ylabel("Rain\n(mm)")
    axes[0].set_title(f"HLA {year}: rainfall, stress, management, and crop growth", loc="left", fontsize=13)

    for scenario in SCENARIOS:
        sub = daily[(daily["year"].eq(year)) & (daily["scenario"].eq(scenario))].sort_values("dap")
        if sub.empty:
            continue
        color = COLORS[scenario]
        ls = LINESTYLES[scenario]
        label = LABELS[scenario]
        axes[1].plot(sub["dap"], sub["wspd"], color=color, linestyle=ls, linewidth=2.0, label=label)
        axes[2].plot(sub["dap"], sub["nstd"], color=color, linestyle=ls, linewidth=2.0, label=label)
        axes[4].plot(sub["dap"], sub["gwad"], color=color, linestyle=ls, linewidth=2.1, label=f"{label} GWAD")
        axes[4].plot(sub["dap"], sub["cwad"], color=color, linestyle=":", linewidth=2.0, alpha=0.95, label=f"{label} CWAD")

        ev = events[(events["year"].eq(year)) & (events["scenario"].eq(scenario))]
        if not ev.empty:
            irrig = ev[ev["operation"].str.contains("Irrigation", case=False, na=False)]
            fert = ev[ev["operation"].str.contains("Fertil|Nitrogen", case=False, na=False)]
            if not irrig.empty:
                axes[3].vlines(irrig["dap"], 0, irrig["amount"], color=color, linestyle=ls, linewidth=2.6, label=f"{label} irrigation")
            if not fert.empty:
                axes[3].scatter(fert["dap"], fert["amount"], color=color, marker="^", s=70, edgecolor="white", linewidth=0.5, label=f"{label} N")

    axes[1].set_ylabel("WSPD")
    axes[1].set_title("DSSAT water-stress index", loc="left", fontsize=10)
    axes[2].set_ylabel("NSTD")
    axes[2].set_title("DSSAT nitrogen-stress index", loc="left", fontsize=10)
    axes[3].set_ylabel("Mgmt\namount")
    axes[3].set_title("Irrigation as vertical lines; fertilization as triangle markers", loc="left", fontsize=10)
    axes[4].set_ylabel("kg/ha")
    axes[4].set_xlabel("DAP")
    axes[4].set_title("Crop growth: solid/dash-dot/dashed = GWAD, dotted = CWAD", loc="left", fontsize=10)

    for ax in axes:
        ax.grid(True, axis="y", linestyle="--", alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, loc="upper left", ncol=3)
    axes[3].legend(frameon=False, loc="upper left", ncol=3, fontsize=8)
    axes[4].legend(frameon=False, loc="upper left", ncol=2, fontsize=8)
    axes[-1].set_xlim(0, ymax + 2)
    fig.tight_layout()
    path = FIG_DIR / f"hla_{year}_management_space_rain_stress_mgmt_yield.png"
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    daily = pd.read_csv(OUT_DIR / "hla_2007_2009_management_space_daily.csv", keep_default_na=False)
    event_frames = [parse_events_from_raw(year, scenario) for year in YEAR_TO_PDATE_DOY for scenario in SCENARIOS]
    events = pd.concat(event_frames, ignore_index=True)
    events.to_csv(OUT_DIR / "hla_2007_2009_management_space_events_deduplicated.csv", index=False, encoding="utf-8-sig")
    summary = build_corrected_summary(daily, events)
    summary.to_csv(OUT_DIR / "hla_2007_2009_management_space_summary_corrected.csv", index=False, encoding="utf-8-sig")
    paths = [plot_year(year, daily, events) for year in YEAR_TO_PDATE_DOY]
    print(summary.to_string(index=False))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
