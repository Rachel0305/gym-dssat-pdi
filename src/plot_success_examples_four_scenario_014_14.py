from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "DSSAT_auto_validation" / "success_examples_four_scenario_014_14"
FIG = OUT / "figures"
TABLE = OUT / "daily_tables"


SCENARIO_LABELS = {
    "null_zero": "Null",
    "recorded": "Recorded expert",
    "recorded_shifted": "Recorded expert shifted",
    "expert_2007_shifted": "Expert 2007 shifted",
    "dssat_auto": "DSSAT auto",
    "dqn_action9_seed1": "DQN seed1",
    "dqn_linked_free_daily_seed1": "DQN seed1 free",
    "dqn_linked_agronomic_window_seed1": "DQN seed1 window",
    "dqn_linked_agronomic_window": "DQN seed1 window",
}

COLORS = {
    "null_zero": "#464C55",
    "recorded": "#CC6F47",
    "recorded_shifted": "#CC6F47",
    "expert_2007_shifted": "#CC6F47",
    "dssat_auto": "#5477C4",
    "dqn_action9_seed1": "#386411",
    "dqn_linked_free_daily_seed1": "#386411",
    "dqn_linked_agronomic_window_seed1": "#386411",
    "dqn_linked_agronomic_window": "#386411",
}


def ensure_dirs() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)


def normalize_daily(df: pd.DataFrame, site: str, year: int) -> pd.DataFrame:
    df = df.copy()
    rename = {
        "target_year": "year",
        "requested_year": "year",
        "water_stress": "swfac",
        "nitrogen_stress": "nstres",
        "grain_kg_ha": "grnwt",
        "biomass_kg_ha": "topwt",
        "gwad": "grnwt",
        "cwad": "topwt",
        "wspd": "swfac",
        "nstd": "nstres",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "scenario" not in df.columns:
        df["scenario"] = "unknown"
    df["scenario"] = df["scenario"].fillna("null_zero").replace({"": "null_zero", "null": "null_zero"})
    if "year" not in df.columns:
        df["year"] = year
    df["year"] = df["year"].fillna(year).astype(int)
    df["site"] = site
    for col in ["dap", "rain", "swfac", "nstres", "grnwt", "topwt", "irrigation_mm", "fertilizer_kg_ha"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    keep = ["site", "year", "scenario", "dap", "rain", "swfac", "nstres", "grnwt", "topwt", "irrigation_mm", "fertilizer_kg_ha"]
    return df[keep].sort_values(["scenario", "dap"]).reset_index(drop=True)


def parse_wth_rain_by_dap(wth_path: Path, planting_doy: int, max_dap: int) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    if not wth_path.exists():
        return pd.DataFrame({"dap": np.arange(max_dap + 1), "rain": 0.0})
    for line in wth_path.read_text(errors="ignore").splitlines():
        if not re.match(r"^\s*\d{7}", line):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        yrdoy = int(parts[0])
        doy = yrdoy % 1000
        dap = doy - planting_doy
        if 0 <= dap <= max_dap:
            rows.append({"dap": float(dap), "rain": float(parts[4])})
    if not rows:
        return pd.DataFrame({"dap": np.arange(max_dap + 1), "rain": 0.0})
    return pd.DataFrame(rows)


def attach_rain_from_map(df: pd.DataFrame, rain_map: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=["rain"], errors="ignore").merge(rain_map[["dap", "rain"]], on="dap", how="left")
    out["rain"] = out["rain"].fillna(0.0)
    return out


def build_hla(year: int) -> pd.DataFrame:
    base_path = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_2015_official_reward_restart" / "figures_011_13_windowed_ppo" / "hla_2010_2015_four_scenario_with_windowed_ppo_daily.csv"
    base = normalize_daily(pd.read_csv(base_path), "HLA", year)
    base = base[(base["year"] == year) & (base["scenario"] != "ppo_windowed_seed0")].copy()
    dqn_path = {
        2010: ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_success_strategy_validation_014_09" / "2010" / "action9_seed1_5000steps" / "action9_dqn_baseline_rel_eval_daily.csv",
        2015: ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2015_success_strategy_validation_014_10" / "2015" / "baseline_relative_seed1_5000steps" / "dqn_baseline_rel_eval_daily.csv",
    }[year]
    dqn_raw = pd.read_csv(dqn_path)
    # In these HLA DQN traces, used_irrigation/used_nitrogen are cumulative totals.
    # safe_amir/safe_anfer are the daily event amounts that should be plotted and summed.
    if "safe_amir" in dqn_raw.columns:
        dqn_raw["irrigation_mm"] = dqn_raw["safe_amir"]
    if "safe_anfer" in dqn_raw.columns:
        dqn_raw["fertilizer_kg_ha"] = dqn_raw["safe_anfer"]
    dqn_raw["scenario"] = "dqn_action9_seed1"
    dqn = normalize_daily(dqn_raw, "HLA", year)
    rain_map = base[base["scenario"].isin(["null_zero"])][["dap", "rain"]].drop_duplicates("dap")
    if rain_map.empty:
        rain_map = base[["dap", "rain"]].drop_duplicates("dap")
    dqn = attach_rain_from_map(dqn, rain_map)
    return pd.concat([base, dqn], ignore_index=True)


def build_yc2014() -> pd.DataFrame:
    base_path = ROOT / "DSSAT_auto_validation" / "yc_2008_2014_four_scenario_comparison_013_09" / "yc_2014_four_scenario_daily_for_plot_updated.csv"
    base = normalize_daily(pd.read_csv(base_path), "YC", 2014)
    base = base[base["scenario"] != "dqn"].copy()
    dqn_path = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_yc2014_linked_dqn_5k_multiseed_013_07" / "seed1" / "dqn_linked_free_daily" / "dqn_linked_free_daily_eval_daily.csv"
    dqn_raw = pd.read_csv(dqn_path)
    dqn_raw["scenario"] = "dqn_linked_free_daily_seed1"
    dqn = normalize_daily(dqn_raw, "YC", 2014)
    rain_map = base[["dap", "rain"]].drop_duplicates("dap")
    dqn = attach_rain_from_map(dqn, rain_map)
    return pd.concat([base, dqn], ignore_index=True)


def build_fq2016() -> pd.DataFrame:
    screen_path = ROOT / "DSSAT_auto_validation" / "fq_all_year_screen_and_dqn_transfer_014_01" / "014_01_fq_all_year_screening_daily.csv"
    screen = normalize_daily(pd.read_csv(screen_path), "FQ", 2016)
    base = screen[(screen["year"] == 2016) & (screen["scenario"].isin(["null_zero", "recorded_shifted", "dssat_auto"]))].copy()
    dqn_path = ROOT / "DSSAT_auto_validation" / "fq_all_year_screen_and_dqn_transfer_014_01" / "014_01_fq_dqn_daily.csv"
    dqn_raw = pd.read_csv(dqn_path)
    dqn_raw = dqn_raw[(dqn_raw["year"] == 2016) & (dqn_raw["seed"] == 1) & (dqn_raw["scenario"] == "dqn_linked_agronomic_window")].copy()
    dqn_raw["scenario"] = "dqn_linked_agronomic_window_seed1"
    dqn = normalize_daily(dqn_raw, "FQ", 2016)
    max_dap = int(max(base["dap"].max(), dqn["dap"].max()))
    wth = ROOT / "DSSAT_auto_validation" / "fq_all_year_screen_and_dqn_transfer_014_01" / "runs" / "2016" / "seed1" / "dqn_linked_agronomic_window" / "input" / "CNFQ1601.WTH"
    rain_map = parse_wth_rain_by_dap(wth, planting_doy=162, max_dap=max_dap)
    base = attach_rain_from_map(base, rain_map)
    dqn = attach_rain_from_map(dqn, rain_map)
    return pd.concat([base, dqn], ignore_index=True)


def scenario_order(df: pd.DataFrame) -> list[str]:
    preferred = [
        "null_zero",
        "recorded",
        "recorded_shifted",
        "expert_2007_shifted",
        "dssat_auto",
        "dqn_action9_seed1",
        "dqn_linked_free_daily_seed1",
        "dqn_linked_agronomic_window_seed1",
    ]
    present = list(df["scenario"].dropna().unique())
    return [x for x in preferred if x in present] + [x for x in present if x not in preferred]


def plot_process(df: pd.DataFrame, site: str, year: int, note: str) -> Path:
    order = scenario_order(df)
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(15, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.2, 1.2, 1.2, 1.4], "hspace": 0.28},
    )
    fig.patch.set_facecolor("#FCFCFD")
    for ax in axes:
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, axis="both", color="#E6E8F0", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#D7DBE7")
        ax.tick_params(colors="#464C55", labelsize=10)

    rain = df[["dap", "rain"]].drop_duplicates("dap").sort_values("dap")
    axes[0].bar(rain["dap"], rain["rain"], color="#C5CAD3", edgecolor="#7A828F", width=0.85, label="Rainfall")
    axes[0].set_ylabel("Rain\n(mm)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=10)

    for scen in order:
        sdf = df[df["scenario"] == scen].sort_values("dap")
        label = SCENARIO_LABELS.get(scen, scen)
        color = COLORS.get(scen, "#1F2430")
        axes[1].plot(sdf["dap"], sdf["swfac"], label=label, color=color, linewidth=1.9)
        axes[2].plot(sdf["dap"], sdf["nstres"], label=label, color=color, linewidth=1.9)
    axes[1].set_ylabel("Water\nstress")
    axes[2].set_ylabel("Nitrogen\nstress")
    axes[1].legend(loc="upper left", ncol=2, frameon=False, fontsize=10)

    for i, scen in enumerate(order):
        sdf = df[df["scenario"] == scen].sort_values("dap")
        color = COLORS.get(scen, "#1F2430")
        irr = sdf[sdf["irrigation_mm"] > 0]
        fer = sdf[sdf["fertilizer_kg_ha"] > 0]
        axes[3].vlines(irr["dap"], 0, irr["irrigation_mm"], color=color, linewidth=2.0, alpha=0.9)
        axes[3].scatter(fer["dap"], fer["fertilizer_kg_ha"], color=color, marker="^", s=30, alpha=0.95)
    axes[3].set_ylabel("Mgmt\namount")
    axes[3].text(0.0, 1.03, "Irrigation = vertical lines; fertilization = triangle markers", transform=axes[3].transAxes, fontsize=10, color="#464C55")

    for scen in order:
        sdf = df[df["scenario"] == scen].sort_values("dap")
        label = SCENARIO_LABELS.get(scen, scen)
        color = COLORS.get(scen, "#1F2430")
        axes[4].plot(sdf["dap"], sdf["grnwt"], color=color, linewidth=2.1, label=f"{label} grain")
        axes[4].plot(sdf["dap"], sdf["topwt"], color=color, linewidth=1.4, linestyle="--", alpha=0.75, label=f"{label} biomass")
    axes[4].set_ylabel("kg/ha")
    axes[4].set_xlabel("DAP")
    axes[4].legend(loc="upper left", ncol=2, frameon=False, fontsize=9)

    fig.subplots_adjust(top=0.90)
    fig.text(
        0.08,
        0.975,
        f"{site} {year} management scenario process plot",
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#1F2430",
    )
    fig.text(0.08, 0.952, note, ha="left", va="top", fontsize=10, color="#6F768A")
    out = FIG / f"{site.lower()}_{year}_scenario_process.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def summarize(df: pd.DataFrame, site: str, year: int) -> pd.DataFrame:
    rows = []
    for scen, sdf in df.groupby("scenario", sort=False):
        last = sdf.sort_values("dap").iloc[-1]
        rows.append(
            {
                "site": site,
                "year": year,
                "scenario": scen,
                "scenario_label": SCENARIO_LABELS.get(scen, scen),
                "total_irrigation_mm": float(sdf["irrigation_mm"].sum()),
                "total_fertilizer_kg_ha": float(sdf["fertilizer_kg_ha"].sum()),
                "final_grain_kg_ha": float(last["grnwt"]),
                "final_biomass_kg_ha": float(last["topwt"]),
                "max_water_stress": float(sdf["swfac"].max()),
                "max_nitrogen_stress": float(sdf["nstres"].max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    examples = [
        ("HLA", 2010, build_hla(2010), "Four scenarios; DQN uses action9 baseline-relative seed1 5K."),
        ("HLA", 2015, build_hla(2015), "Four scenarios; DQN uses baseline-relative seed1 5K."),
        ("YC", 2014, build_yc2014(), "Available scenarios only: recorded, DSSAT auto, DQN seed1. Null daily table was not found."),
        ("FQ", 2016, build_fq2016(), "Four scenarios; DQN uses linked agronomic-window seed1 5K."),
    ]
    all_daily = []
    all_summary = []
    manifest = []
    for site, year, df, note in examples:
        df = df.sort_values(["scenario", "dap"]).reset_index(drop=True)
        daily_path = TABLE / f"{site.lower()}_{year}_scenario_daily.csv"
        df.to_csv(daily_path, index=False, encoding="utf-8-sig")
        fig_path = plot_process(df, site, year, note)
        summary = summarize(df, site, year)
        summary_path = TABLE / f"{site.lower()}_{year}_scenario_summary.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        all_daily.append(df)
        all_summary.append(summary)
        manifest.append(
            {
                "site": site,
                "year": year,
                "daily_csv": str(daily_path.relative_to(ROOT)),
                "summary_csv": str(summary_path.relative_to(ROOT)),
                "figure_png": str(fig_path.relative_to(ROOT)),
                "note": note,
            }
        )
    pd.concat(all_daily, ignore_index=True).to_csv(OUT / "014_14_all_examples_daily.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_summary, ignore_index=True).to_csv(OUT / "014_14_all_examples_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(manifest).to_csv(OUT / "014_14_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"Saved outputs to {OUT}")


if __name__ == "__main__":
    main()
