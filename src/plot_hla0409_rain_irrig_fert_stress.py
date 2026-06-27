from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0409_DSSAT480_2004"
FIG_DIR = RUN_DIR / "figures"


def parse_table_out(path: Path) -> pd.DataFrame:
    header: list[str] | None = None
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="latin1", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("@"):
                header = stripped.replace("@", "", 1).split()
                continue
            if header and re.match(r"^\d{4}\s+\d+", stripped):
                parts = stripped.split()
                if len(parts) >= len(header):
                    rows.append(dict(zip(header, parts[: len(header)])))
    df = pd.DataFrame(rows)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def parse_mgmt_events(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="latin1", errors="ignore") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 9 or not parts[0].isdigit() or not parts[3].isdigit():
                continue
            try:
                run = int(parts[0])
                day = int(parts[2].rstrip(","))
                year = int(parts[3])
                doy = int(parts[4])
                das = int(parts[5])
                dap = int(parts[6])
            except ValueError:
                continue
            op_tokens = parts[8:]
            if op_tokens and op_tokens[0].isdigit():
                op_tokens = op_tokens[1:]
            operation = " ".join(op_tokens)
            quantity = 0.0
            unit = ""
            qmatch = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(mm|kg/ha|kg|%)", operation)
            if qmatch:
                quantity = float(qmatch.group(1))
                unit = qmatch.group(2)
            rows.append(
                {
                    "run": run,
                    "date_label": f"{parts[1]} {day}, {year}",
                    "year": year,
                    "doy": doy,
                    "das": das,
                    "dap": dap,
                    "operation": operation,
                    "quantity": quantity,
                    "unit": unit,
                }
            )
    return pd.DataFrame(rows)


def build_daily_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    plant = parse_table_out(RUN_DIR / "PlantGro.OUT")
    weather = parse_table_out(RUN_DIR / "Weather.OUT")
    events = parse_mgmt_events(RUN_DIR / "MgmtEvent.OUT")

    plant = plant.rename(
        columns={
            "YEAR": "year",
            "DOY": "doy",
            "DAP": "dap",
            "WSPD": "wspd",
            "NSTD": "nstd",
            "CWAD": "cwad",
            "GWAD": "gwad",
        }
    )
    weather = weather.rename(columns={"YEAR": "year", "DOY": "doy", "PRED": "rain"})

    keep = ["year", "doy", "dap", "wspd", "nstd", "cwad", "gwad"]
    daily = plant[[c for c in keep if c in plant.columns]].copy()
    if {"year", "doy", "rain"}.issubset(weather.columns):
        daily = daily.merge(weather[["year", "doy", "rain"]], on=["year", "doy"], how="left")
    else:
        daily["rain"] = 0.0
    daily["rain"] = pd.to_numeric(daily["rain"], errors="coerce").fillna(0.0)

    daily["irrigation_mm"] = 0.0
    daily["fertilizer_kg_ha"] = 0.0
    if not events.empty:
        for _, ev in events.iterrows():
            op = str(ev["operation"])
            dap = int(ev["dap"])
            qty = float(ev["quantity"])
            if "Irrigation" in op:
                daily.loc[daily["dap"].eq(dap), "irrigation_mm"] += qty
            if "Fertil" in op or "Fertilizer" in op:
                daily.loc[daily["dap"].eq(dap), "fertilizer_kg_ha"] += qty

    return daily, events


def plot(daily: pd.DataFrame, events: pd.DataFrame) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    daily_path = FIG_DIR / "hla0409_windows480_ic1_daily_rain_irrig_fert_stress.csv"
    events_path = FIG_DIR / "hla0409_windows480_ic1_management_events.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    events.to_csv(events_path, index=False, encoding="utf-8-sig")

    fig, ax_stress = plt.subplots(figsize=(13.5, 5.8))
    ax_water = ax_stress.twinx()

    x = pd.to_numeric(daily["dap"], errors="coerce")
    rain = pd.to_numeric(daily["rain"], errors="coerce").fillna(0.0)
    irrigation = pd.to_numeric(daily["irrigation_mm"], errors="coerce").fillna(0.0)
    fertilizer = pd.to_numeric(daily["fertilizer_kg_ha"], errors="coerce").fillna(0.0)

    rain_bar = ax_water.bar(
        x,
        rain,
        width=1.0,
        color="#BFC5D2",
        edgecolor="#6F7785",
        linewidth=0.25,
        alpha=0.55,
        label="Rainfall (mm)",
        zorder=1,
    )
    irrig_bar = ax_water.bar(
        x,
        irrigation,
        width=2.5,
        color="#1F77B4",
        edgecolor="#0B3D70",
        linewidth=0.35,
        alpha=0.85,
        label="Irrigation (mm)",
        zorder=2,
    )
    fert_bar = ax_water.bar(
        x,
        fertilizer,
        width=3.2,
        color="#FF7F0E",
        edgecolor="#9A4B00",
        linewidth=0.35,
        alpha=0.80,
        label="Fertilizer (kg/ha)",
        zorder=3,
    )

    wspd_line = ax_stress.plot(
        x,
        pd.to_numeric(daily["wspd"], errors="coerce"),
        color="#D62728",
        linewidth=2.2,
        label="WSPD water stress index",
        zorder=5,
    )[0]
    nstd_line = ax_stress.plot(
        x,
        pd.to_numeric(daily["nstd"], errors="coerce"),
        color="#2CA02C",
        linewidth=2.2,
        linestyle=(0, (5, 2)),
        label="NSTD nitrogen stress index",
        zorder=6,
    )[0]

    ax_stress.set_title("HLA 2004 IC=1 null Windows DSSAT 4.8.0: rainfall, management events, and stress indices", fontsize=12)
    ax_stress.set_xlabel("DAP")
    ax_stress.set_ylabel("Stress index (larger = stronger stress in this output)")
    ax_water.set_ylabel("Rain / irrigation / fertilizer amount")
    ax_stress.set_ylim(-0.03, 1.03)
    max_amt = max(float(rain.max()), float(irrigation.max()), float(fertilizer.max()), 10.0)
    ax_water.set_ylim(0, max_amt * 1.25)
    ax_stress.grid(True, axis="y", color="#E4E7EF", linewidth=0.8)
    ax_stress.spines["top"].set_visible(False)
    ax_water.spines["top"].set_visible(False)

    handles = [wspd_line, nstd_line, rain_bar, irrig_bar, fert_bar]
    labels = [h.get_label() for h in handles]
    ax_stress.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    out = FIG_DIR / "hla0409_windows480_ic1_rain_irrig_fert_wspd_nstd.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    daily, events = build_daily_data()
    out = plot(daily, events)
    summary = {
        "figure": str(out),
        "daily_csv": str(FIG_DIR / "hla0409_windows480_ic1_daily_rain_irrig_fert_stress.csv"),
        "events_csv": str(FIG_DIR / "hla0409_windows480_ic1_management_events.csv"),
        "rain_total": float(pd.to_numeric(daily["rain"], errors="coerce").sum()),
        "irrigation_total": float(pd.to_numeric(daily["irrigation_mm"], errors="coerce").sum()),
        "fertilizer_total": float(pd.to_numeric(daily["fertilizer_kg_ha"], errors="coerce").sum()),
        "wspd_min": float(pd.to_numeric(daily["wspd"], errors="coerce").min()),
        "wspd_max": float(pd.to_numeric(daily["wspd"], errors="coerce").max()),
        "nstd_min": float(pd.to_numeric(daily["nstd"], errors="coerce").min()),
        "nstd_max": float(pd.to_numeric(daily["nstd"], errors="coerce").max()),
    }
    pd.DataFrame([summary]).to_csv(FIG_DIR / "hla0409_windows480_ic1_plot_summary.csv", index=False, encoding="utf-8-sig")
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
