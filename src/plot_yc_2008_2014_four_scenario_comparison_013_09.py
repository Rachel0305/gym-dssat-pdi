from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


FIG_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc_2008_2014_four_scenario_comparison_013_09" / "figures"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc_2008_2014_four_scenario_comparison_013_09"

BASELINE_DAILY = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_forward_screening_013_01" / "013_01_fq_yc_forward_daily.csv"
BASELINE_EVENTS = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_forward_screening_013_01" / "013_01_fq_yc_forward_events.csv"

BASELINE_RUNS = {
    2008: PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_forward_screening_013_01" / "runs" / "YC" / "2008_null" / "pdi_tmp_snapshot" / "Weather.OUT",
    2014: PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_forward_screening_013_01" / "runs" / "YC" / "2014_null" / "pdi_tmp_snapshot" / "Weather.OUT",
}

DQN_FILES = {
    2008: PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_yc2008_linked_dqn_5k_multiseed_013_08" / "seed1" / "013_08_yc2008_linked_dqn_5k_seed1_daily.csv",
    2014: PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_yc2014_linked_dqn_5k_multiseed_013_07" / "seed0" / "013_07_yc2014_linked_dqn_5k_seed0_daily.csv",
}

DQN_SELECTED_SCENARIO = {
    2008: "dqn_linked_agronomic_window",
    2014: "dqn_linked_free_daily",
}

SCENARIO_ORDER = ["null", "recorded", "dssat_auto", "dqn"]
SCENARIO_LABELS = {
    "null": "Null",
    "recorded": "Recorded",
    "dssat_auto": "DSSAT auto",
    "dqn": "DQN linked",
}
SCENARIO_COLORS = {
    "null": "#464C55",
    "recorded": "#CC6F47",
    "dssat_auto": "#5477C4",
    "dqn": "#2E7D32",
}


def parse_weather_out(path: Path) -> pd.DataFrame:
    header: list[str] | None = None
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="latin1", errors="ignore").splitlines():
        stripped = raw.strip()
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
    if df.empty:
        return df
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = [c for c in ["YEAR", "DOY", "PRED"] if c in df.columns]
    df = df[keep].rename(columns={"YEAR": "year", "DOY": "doy", "PRED": "rain"})
    return df


def load_baseline_year(year: int) -> pd.DataFrame:
    daily = pd.read_csv(BASELINE_DAILY)
    daily = daily[(daily["site"].eq("YC")) & (daily["requested_year"].eq(year))].copy()
    daily["year"] = year
    daily = daily.rename(columns={"wspd": "swfac", "nstd": "nstres", "gwad": "grnwt", "cwad": "topwt"})
    event_rows = pd.read_csv(BASELINE_EVENTS)
    event_rows = event_rows[(event_rows["site"].eq("YC")) & (event_rows["requested_year"].eq(year))].copy()
    if not event_rows.empty:
        event_rows["irrigation_mm"] = event_rows["unit"].eq("mm").astype(float) * event_rows["amount"]
        event_rows["fertilizer_kg_ha"] = event_rows["unit"].str.contains("kg", na=False).astype(float) * event_rows["amount"]
        # Keep management events scenario-specific.  A previous version grouped
        # only by DAP, which incorrectly copied recorded fertilization events
        # into null and DSSAT-auto scenarios.
        event_map = event_rows.groupby(["scenario", "dap"], as_index=False)[["irrigation_mm", "fertilizer_kg_ha"]].sum()
        daily = daily.merge(event_map, on=["scenario", "dap"], how="left")
    else:
        daily["irrigation_mm"] = 0.0
        daily["fertilizer_kg_ha"] = 0.0
    daily["irrigation_mm"] = daily["irrigation_mm"].fillna(0.0)
    daily["fertilizer_kg_ha"] = daily["fertilizer_kg_ha"].fillna(0.0)
    weather = parse_weather_out(BASELINE_RUNS[year])
    daily = daily.merge(weather[["year", "doy", "rain"]], on=["year", "doy"], how="left")
    daily["rain"] = daily["rain"].fillna(0.0)
    daily["scenario"] = daily["scenario"].astype(str)
    return daily


def load_rain_by_dap(year: int) -> pd.DataFrame:
    daily = load_baseline_year(year)
    return daily[["dap", "rain"]].drop_duplicates("dap").sort_values("dap").copy()


def load_dqn_year(year: int) -> pd.DataFrame:
    daily = pd.read_csv(DQN_FILES[year]).copy()
    daily = daily[daily["scenario"].eq(DQN_SELECTED_SCENARIO[year])].copy()
    daily["year"] = year
    if "rain" in daily.columns:
        daily["rain"] = pd.to_numeric(daily["rain"], errors="coerce")
    else:
        daily["rain"] = pd.NA
    rain_map = load_rain_by_dap(year)
    daily = daily.merge(rain_map, on="dap", how="left", suffixes=("", "_dap"))
    daily["rain"] = pd.to_numeric(daily["rain_dap"].fillna(daily["rain"]), errors="coerce").fillna(0.0)
    daily = daily.drop(columns=[c for c in ["rain_dap"] if c in daily.columns])
    daily["scenario"] = "dqn"
    return daily


def prepare_year_frame(year: int) -> pd.DataFrame:
    baseline = load_baseline_year(year)
    dqn = load_dqn_year(year)
    use = baseline[baseline["scenario"].isin(SCENARIO_ORDER[:-1])].copy()
    use = use.rename(columns={"gwad": "grnwt", "cwad": "topwt"})
    use = use[["year", "scenario", "doy", "dap", "swfac", "nstres", "grnwt", "topwt", "irrigation_mm", "fertilizer_kg_ha", "rain"]]
    dqn = dqn[["year", "scenario", "doy", "dap", "swfac", "nstres", "grnwt", "topwt", "irrigation_mm", "fertilizer_kg_ha", "rain"]]
    out = pd.concat([use, dqn], ignore_index=True, sort=False)
    out = out.sort_values(["year", "scenario", "dap"]).reset_index(drop=True)
    return out


def plot_year(data: pd.DataFrame, year: int, out_path: Path) -> None:
    sub = data[data["year"].eq(year)].copy()
    max_dap = int(pd.to_numeric(sub["dap"], errors="coerce").max())
    x_max = max(10, max_dap + 5)
    x_ticks = pd.Series(range(0, x_max + 1, 25)).tolist()
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(15.8, 13.2),
        sharex=True,
        gridspec_kw={"height_ratios": [0.9, 1.0, 1.0, 1.0, 1.1], "hspace": 0.25},
    )
    rain = sub[["dap", "rain"]].drop_duplicates("dap").sort_values("dap")
    axes[0].bar(rain["dap"], rain["rain"], width=1.0, color="#C5CAD3", edgecolor="#7A828F", linewidth=0.45)
    axes[0].set_ylabel("Rain\n(mm)")

    for scenario in SCENARIO_ORDER:
        s = sub[sub["scenario"].eq(scenario)].sort_values("dap")
        if s.empty:
            continue
        color = SCENARIO_COLORS[scenario]
        label = SCENARIO_LABELS[scenario]
        axes[1].plot(s["dap"], s["swfac"], color=color, linewidth=2.0, label=label)
        axes[2].plot(s["dap"], s["nstres"], color=color, linewidth=2.0, label=label)
        axes[4].plot(s["dap"], s["grnwt"], color=color, linewidth=2.0)
        axes[4].plot(s["dap"], s["topwt"], color=color, linewidth=1.6, linestyle="--", alpha=0.65)
        mg_i = s[s["irrigation_mm"].fillna(0) > 1e-6]
        mg_n = s[s["fertilizer_kg_ha"].fillna(0) > 1e-6]
        if not mg_i.empty:
            axes[3].vlines(mg_i["dap"], 0, mg_i["irrigation_mm"], colors=color, linewidth=2.4, alpha=0.95)
        if not mg_n.empty:
            axes[3].scatter(mg_n["dap"], mg_n["fertilizer_kg_ha"], marker="^", s=48, color=color, edgecolor="#FFFFFF", linewidth=0.6, zorder=4)

    axes[1].set_ylabel("Water\nstress")
    axes[2].set_ylabel("Nitrogen\nstress")
    axes[3].set_ylabel("Mgmt\namount")
    axes[4].set_ylabel("kg/ha")
    axes[4].set_xlabel("DAP")
    axes[1].set_title(f"YC {year} four-scenario process plot", loc="left", fontsize=14)
    axes[2].set_title("Water and nitrogen stress", loc="left", fontsize=10)
    axes[3].set_title("Management events", loc="left", fontsize=10)
    axes[4].set_title("Crop outcome: solid=grain, dashed=biomass", loc="left", fontsize=10)

    for ax in axes:
        ax.grid(True, axis="both", color="#E6E8F0", linewidth=0.8, alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(x_ticks)

    legend_lines = [Line2D([0], [0], color=SCENARIO_COLORS[s], lw=2, label=SCENARIO_LABELS[s]) for s in SCENARIO_ORDER]
    axes[1].legend(handles=legend_lines, loc="upper left", ncol=2, frameon=False, fontsize=10)
    mgmt_legend = [
        Line2D([0], [0], color="#333333", lw=2, label="Irrigation"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#333333", markeredgecolor="#FFFFFF", markersize=8, label="Fertilization"),
    ]
    axes[3].legend(handles=mgmt_legend, loc="upper left", frameon=False, fontsize=9)
    outcome_legend = [
        Line2D([0], [0], color="#333333", lw=2, label="Grain"),
        Line2D([0], [0], color="#333333", lw=1.5, linestyle="--", label="Biomass"),
    ]
    axes[4].legend(handles=outcome_legend, loc="upper left", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_csv_with_fallback(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_updated{path.suffix}")
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        return fallback


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    all_frames = []
    for year in (2008, 2014):
        frame = prepare_year_frame(year)
        all_frames.append(frame)
        write_csv_with_fallback(frame, OUT_DIR / f"yc_{year}_four_scenario_daily_for_plot.csv")
        out_path = FIG_DIR / f"yc_{year}_four_scenario_process.png"
        plot_year(frame, year, out_path)
        outputs.append(str(out_path))
    write_csv_with_fallback(pd.concat(all_frames, ignore_index=True), OUT_DIR / "yc_2008_2014_four_scenario_all_daily.csv")
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
