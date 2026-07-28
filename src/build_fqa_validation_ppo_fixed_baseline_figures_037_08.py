from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "benchmark_results" / "037_08_FQA_validation_ppo_fixed_baseline_figures"
FIG_DIR = OUT_ROOT / "figures"
TABLE_DIR = OUT_ROOT / "tables"

BASE_SUMMARY = ROOT / "benchmark_results" / "037_07_static_level1_four_baseline_rebuild" / "evaluation" / "037_07_full_FQA_only_summary.csv"
PPO_YEAR_LEVEL = ROOT / "benchmark_results" / "036_04_select_checkpoint_and_plot_03601_03603_summary" / "tables" / "036_04_selected_year_level_comparison.csv"
PPO_SELECTION = ROOT / "benchmark_results" / "036_04_select_checkpoint_and_plot_03601_03603_summary" / "tables" / "036_04_selected_checkpoints.csv"

STATION = "FQA"
YEARS = list(range(2014, 2024))


SCENARIO_LABELS = {
    "null": "Null",
    "recorded_farmer_template": "Recorded farmer",
    "official_extension_expert": "Official expert",
    "dssat_auto": "DSSAT auto",
    "ppo_candidate": "MaskablePPO candidate",
}

SCENARIO_STYLES = {
    "null": dict(color="#404040", linestyle="-", marker="o"),
    "recorded_farmer_template": dict(color="#c44e52", linestyle="--", marker="o"),
    "official_extension_expert": dict(color="#7b61b6", linestyle=":", marker="o"),
    "dssat_auto": dict(color="#d08b00", linestyle="-.", marker="o"),
    "ppo_candidate": dict(color="#1f8f4e", linestyle="-", marker="D"),
}


def _to_float(v) -> float:
    try:
        if pd.isna(v):
            return np.nan
        if str(v).strip() == "":
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def read_dssat_table(path: Path) -> pd.DataFrame:
    """Read DSSAT OUT file table beginning with @ header."""
    lines = path.read_text(errors="replace").splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("@"):
            header_idx = i
            break
    if header_idx is None:
        return pd.DataFrame()
    cols = lines[header_idx].replace("@", " ").split()
    rows = []
    for line in lines[header_idx + 1 :]:
        if not line.strip() or line.startswith("*") or line.startswith("!") or line.startswith("@"):
            continue
        parts = line.split()
        if len(parts) < len(cols):
            continue
        rows.append(parts[: len(cols)])
    df = pd.DataFrame(rows, columns=cols)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def parse_mgmt_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["dap", "irrigation", "nitrogen"])
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if "Irrigation" in line:
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?", line)
            # RUN, year-ish date fragments may exist; DAP is the fourth DSSAT numeric field after RUN/DOY/DAS.
            parts = line.split()
            try:
                dap = int(parts[6])
            except Exception:
                dap = np.nan
            q = np.nan
            m = re.search(r"([0-9]+(?:\.[0-9]*)?)\s*mm", line, flags=re.I)
            if m:
                q = float(m.group(1))
            rows.append({"dap": dap, "irrigation": q, "nitrogen": 0.0})
        elif "Fertilizer" in line:
            parts = line.split()
            try:
                dap = int(parts[6])
            except Exception:
                dap = np.nan
            # DSSAT text may show several numbers; use first kg/ha-like amount after operation text when possible.
            q = np.nan
            m = re.search(r"([0-9]+(?:\.[0-9]*)?)\s*kg", line, flags=re.I)
            if m:
                q = float(m.group(1))
            else:
                nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", line)]
                if nums:
                    q = nums[-1]
            rows.append({"dap": dap, "irrigation": 0.0, "nitrogen": q})
    out = pd.DataFrame(rows, columns=["dap", "irrigation", "nitrogen"])
    if not out.empty:
        out = out.drop_duplicates(subset=["dap", "irrigation", "nitrogen"]).reset_index(drop=True)
    return out


def load_baseline_daily(summary_row: pd.Series) -> pd.DataFrame:
    scenario = summary_row["scenario"]
    snap = ROOT / str(summary_row["snapshot_path"])
    plant = read_dssat_table(snap / "PlantGro.OUT")
    soil = read_dssat_table(snap / "SoilWat.OUT")
    weather = read_dssat_table(snap / "Weather.OUT")
    events = parse_mgmt_events(snap / "MgmtEvent.OUT")

    keep = pd.DataFrame()
    if not plant.empty:
        keep = plant[[c for c in ["YEAR", "DOY", "DAS", "DAP", "WSPD", "NSTD", "GWAD", "CWAD"] if c in plant.columns]].copy()
    if keep.empty:
        return keep
    keep = keep.drop_duplicates(subset=[c for c in ["YEAR", "DOY", "DAP"] if c in keep.columns]).reset_index(drop=True)
    if not weather.empty:
        weather = weather.drop_duplicates(subset=[c for c in ["YEAR", "DOY"] if c in weather.columns]).reset_index(drop=True)
        wcols = [c for c in ["YEAR", "DOY", "PRED", "TMXD", "TMND"] if c in weather.columns]
        keep = keep.merge(weather[wcols], on=["YEAR", "DOY"], how="left")
    if not soil.empty:
        soil = soil.drop_duplicates(subset=[c for c in ["YEAR", "DOY"] if c in soil.columns]).reset_index(drop=True)
        scols = [c for c in ["YEAR", "DOY", "SWTD"] if c in soil.columns]
        keep = keep.merge(soil[scols], on=["YEAR", "DOY"], how="left")
    keep["scenario"] = scenario
    keep["irrigation_event"] = 0.0
    keep["nitrogen_event"] = 0.0
    if not events.empty:
        for _, ev in events.dropna(subset=["dap"]).iterrows():
            mask = keep["DAP"].astype(float).round().astype(int) == int(ev["dap"])
            if ev.get("irrigation", 0) and not pd.isna(ev["irrigation"]):
                keep.loc[mask, "irrigation_event"] = keep.loc[mask, "irrigation_event"] + float(ev["irrigation"])
            if ev.get("nitrogen", 0) and not pd.isna(ev["nitrogen"]):
                keep.loc[mask, "nitrogen_event"] = keep.loc[mask, "nitrogen_event"] + float(ev["nitrogen"])
    return keep


def load_ppo_daily(row: pd.Series) -> pd.DataFrame:
    p = ROOT / str(row["daily_csv_path"])
    df = pd.read_csv(p)
    out = pd.DataFrame()
    out["YEAR"] = df.get("year", row["year"])
    out["DOY"] = df.get("doy", np.nan)
    out["DAP"] = df["dap"]
    out["PRED"] = df.get("rain", np.nan)
    out["TMXD"] = df.get("tmax", np.nan)
    out["TMND"] = df.get("tmin", np.nan)
    out["WSPD"] = df.get("swfac", np.nan)
    out["NSTD"] = df.get("nstres", np.nan)
    out["GWAD"] = df.get("grnwt", np.nan)
    out["CWAD"] = df.get("topwt", np.nan)
    out["SWTD"] = np.nan
    out["irrigation_event"] = df.get("safe_action_amir", df.get("raw_action_amir", 0.0))
    out["nitrogen_event"] = df.get("safe_action_anfer", df.get("raw_action_anfer", 0.0))
    out["scenario"] = "ppo_candidate"
    return out


def add_common_reward(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for scenario, g in df.groupby("scenario", sort=False):
        g = g.sort_values("DAP").copy()
        grn = pd.to_numeric(g["GWAD"], errors="coerce").fillna(method="ffill").fillna(0.0)
        delta = grn.diff().fillna(grn)
        irr = pd.to_numeric(g.get("irrigation_event", 0.0), errors="coerce").fillna(0.0)
        nit = pd.to_numeric(g.get("nitrogen_event", 0.0), errors="coerce").fillna(0.0)
        g["common_reward_step"] = delta - irr - 5.0 * nit
        g["common_reward_cum"] = g["common_reward_step"].cumsum()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def plot_year(year: int, daily: pd.DataFrame, metrics: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), constrained_layout=True)
    axes = axes.ravel()

    # Weather: use first available scenario weather, identical by year.
    wx = daily[daily["scenario"] == "null"]
    if wx.empty:
        wx = daily.iloc[:0]
    ax = axes[0]
    if not wx.empty:
        ax.bar(wx["DAP"], wx["PRED"], color="#9bbbd3", alpha=0.8, label="Rain")
        ax.set_ylabel("Rain (mm)")
        ax2 = ax.twinx()
        ax2.plot(wx["DAP"], wx["TMXD"], color="#d9473f", lw=1.5, label="Tmax")
        ax2.plot(wx["DAP"], wx["TMND"], color="#777777", lw=1.2, ls="--", label="Tmin")
        ax2.set_ylabel("Temperature (°C)")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)
    ax.set_title("Weather")

    def line_panel(ax, col, title, ylabel, include_ppo=True):
        for sc, g in daily.groupby("scenario", sort=False):
            if (not include_ppo) and sc == "ppo_candidate":
                continue
            if col not in g.columns or g[col].isna().all():
                continue
            sty = SCENARIO_STYLES.get(sc, {})
            ax.plot(g["DAP"], g[col], label=SCENARIO_LABELS.get(sc, sc), color=sty.get("color"), linestyle=sty.get("linestyle", "-"), lw=1.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="best")

    line_panel(axes[1], "SWTD", "Soil water (PPO SWTD not saved)", "SWTD (mm)", include_ppo=False)
    line_panel(axes[2], "WSPD", "Water stress index", "WSPD/SWFAC (0=no stress)")
    line_panel(axes[3], "NSTD", "Nitrogen stress index", "NSTD/NSTRES (0=no stress)")

    def event_panel(ax, col, title, ylabel):
        for sc, g in daily.groupby("scenario", sort=False):
            ev = g[pd.to_numeric(g[col], errors="coerce").fillna(0.0) > 0]
            if ev.empty:
                continue
            sty = SCENARIO_STYLES.get(sc, {})
            ax.vlines(ev["DAP"], 0, ev[col], color=sty.get("color"), linestyles=sty.get("linestyle", "-"), lw=2, alpha=0.85)
            ax.scatter(ev["DAP"], ev[col], color=sty.get("color"), marker=sty.get("marker", "o"), s=30, label=SCENARIO_LABELS.get(sc, sc))
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="best")

    event_panel(axes[4], "irrigation_event", "Irrigation events", "mm/event")
    event_panel(axes[5], "nitrogen_event", "Nitrogen application events", "kg/ha/event")

    ax = axes[6]
    for sc, g in daily.groupby("scenario", sort=False):
        sty = SCENARIO_STYLES.get(sc, {})
        if "GWAD" in g.columns:
            ax.plot(g["DAP"], g["GWAD"], color=sty.get("color"), linestyle=sty.get("linestyle", "-"), lw=2, label=SCENARIO_LABELS.get(sc, sc))
        if "CWAD" in g.columns:
            ax.plot(g["DAP"], g["CWAD"], color=sty.get("color"), linestyle=sty.get("linestyle", "-"), lw=0.8, alpha=0.35)
    ax.set_title("Grain and biomass trajectories")
    ax.set_ylabel("kg/ha")
    ax.set_xlabel("DAP")
    ax.legend(fontsize=7, loc="best")

    line_panel(axes[7], "common_reward_cum", "Cumulative common reward", "kg/ha-equivalent")
    axes[7].set_xlabel("DAP")

    fig.suptitle(f"FQ{year} PPO five-scenario daily process (PPO 036_04 ckpt25000; baselines fixed 037_07)", fontsize=14, fontweight="bold")
    out = FIG_DIR / f"037_08_FQ{year}_ppo25k_five_scenario_daily.png"
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    return out


def build_metrics(base: pd.DataFrame, ppo_rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for year in YEARS:
        b = base[(base["station_code"] == STATION) & (base["year"].astype(int) == year)].copy()
        p = ppo_rows[(ppo_rows["station_code"] == STATION) & (ppo_rows["year"].astype(int) == year)].copy()
        if p.empty:
            continue
        p = p.iloc[0]
        b["grain_yield_kg_ha"] = b["grain_yield_kg_ha"].apply(_to_float)
        b["WP_ET_kg_m3"] = b["WP_ET_kg_m3"].apply(_to_float)
        b["PFP_N_kg_kg"] = b["PFP_N_kg_kg"].apply(_to_float)
        rec = {
            "station_code": STATION,
            "year": year,
            "ppo_checkpoint_step": int(float(p["checkpoint_step"])),
            "ppo_yield": _to_float(p["final_grnwt"]),
            "ppo_irrigation": _to_float(p["total_irrigation"]),
            "ppo_n": _to_float(p["total_n"]),
            "ppo_WP_ET": _to_float(p["WP_ET_kg_m3"]),
            "ppo_PFP_N": _to_float(p["PFP_N_kg_kg"]),
            "four_max_yield": b["grain_yield_kg_ha"].max(skipna=True),
            "four_max_WP_ET": b["WP_ET_kg_m3"].max(skipna=True),
            "four_max_PFP_N": b["PFP_N_kg_kg"].max(skipna=True),
            "expert_yield": b.loc[b["scenario"] == "official_extension_expert", "grain_yield_kg_ha"].max(skipna=True),
            "expert_irrigation": b.loc[b["scenario"] == "official_extension_expert", "actual_irrigation_mm"].apply(_to_float).max(skipna=True),
            "expert_n": b.loc[b["scenario"] == "official_extension_expert", "actual_nitrogen_kg_ha"].apply(_to_float).max(skipna=True),
            "action_sequence": p.get("action_sequence", ""),
        }
        rec["yield_gap_vs_four_max"] = rec["ppo_yield"] - rec["four_max_yield"]
        rec["WP_ET_gap_vs_four_max"] = rec["ppo_WP_ET"] - rec["four_max_WP_ET"]
        rec["PFP_N_gap_vs_four_max"] = rec["ppo_PFP_N"] - rec["four_max_PFP_N"]
        rec["water_saved_vs_expert"] = rec["expert_irrigation"] - rec["ppo_irrigation"]
        rec["n_saved_vs_expert"] = rec["expert_n"] - rec["ppo_n"]
        rec["FQ2018_zero_yield_abnormal"] = bool(year == 2018 and rec["four_max_yield"] == 0 and rec["ppo_yield"] == 0)
        records.append(rec)
    return pd.DataFrame(records)


def plot_metric_summary(metrics: pd.DataFrame, metric: str, ylabel: str, out_name: str) -> Path:
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(13, 5.8), constrained_layout=True)
    ppo_col = f"ppo_{metric}"
    base_col = f"four_max_{metric}"
    if metric == "PFP_N":
        ppo_col = "ppo_PFP_N"
        base_col = "four_max_PFP_N"
    if metric == "WP_ET":
        ppo_col = "ppo_WP_ET"
        base_col = "four_max_WP_ET"
    if metric == "yield":
        ppo_col = "ppo_yield"
        base_col = "four_max_yield"

    width = 0.38
    ax.bar(x - width / 2, metrics[base_col], width=width, color="#b8b8b8", label="Best of four baselines")
    ax.bar(x + width / 2, metrics[ppo_col], width=width, color="#1f8f4e", label="PPO candidate")
    for i, row in metrics.iterrows():
        gap = row[ppo_col] - row[base_col]
        txt = f"{gap:+.1f}"
        ax.text(i, max(row[ppo_col], row[base_col]) * 1.01 if max(row[ppo_col], row[base_col]) > 0 else 0.05, txt, ha="center", va="bottom", fontsize=8, rotation=0)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["year"].astype(str), rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(f"FQ validation years: PPO vs best of four baselines ({ylabel})")
    ax.set_xlabel("Year")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(BASE_SUMMARY, keep_default_na=False)
    base = base[(base["station_code"] == STATION) & (base["year"].astype(int).isin(YEARS))].copy()
    ppo_all = pd.read_csv(PPO_YEAR_LEVEL, keep_default_na=False)
    # Use pre-registered FQA selected checkpoint from 036_04.
    sel = pd.read_csv(PPO_SELECTION, keep_default_na=False)
    fqa_sel = sel[sel["station_code"] == STATION].iloc[0]
    selected_step = int(float(fqa_sel["checkpoint_step"]))
    ppo = ppo_all[
        (ppo_all["station_code"] == STATION)
        & (ppo_all["year"].astype(int).isin(YEARS))
        & (ppo_all["checkpoint_step"].astype(float).astype(int) == selected_step)
    ].copy()

    metrics = build_metrics(base, ppo)
    metrics.to_csv(TABLE_DIR / "037_08_FQA_validation_ppo25k_vs_fixed_four_baseline_metrics.csv", index=False, encoding="utf-8-sig")

    daily_files = []
    for year in YEARS:
        daily_parts = []
        for _, row in base[base["year"].astype(int) == year].iterrows():
            daily_parts.append(load_baseline_daily(row))
        ppo_row = ppo[ppo["year"].astype(int) == year]
        if not ppo_row.empty:
            daily_parts.append(load_ppo_daily(ppo_row.iloc[0]))
        daily = pd.concat(daily_parts, ignore_index=True)
        daily = add_common_reward(daily)
        daily.to_csv(TABLE_DIR / f"037_08_FQ{year}_five_scenario_daily_merged.csv", index=False, encoding="utf-8-sig")
        daily_files.append(plot_year(year, daily, metrics[metrics["year"] == year]))

    summary_paths = [
        plot_metric_summary(metrics, "yield", "Yield (kg/ha)", "037_08_FQA_validation_yield_vs_four_baselines.png"),
        plot_metric_summary(metrics, "WP_ET", "WP_ET (kg/m³)", "037_08_FQA_validation_wp_et_vs_four_baselines.png"),
        plot_metric_summary(metrics, "PFP_N", "PFP_N (kg/kg)", "037_08_FQA_validation_pfp_n_vs_four_baselines.png"),
    ]

    note = {
        "station": STATION,
        "years": f"{YEARS[0]}-{YEARS[-1]}",
        "selected_checkpoint_step": selected_step,
        "daily_figures": [str(p.relative_to(ROOT)) for p in daily_files],
        "summary_figures": [str(p.relative_to(ROOT)) for p in summary_paths],
        "metrics_csv": str((TABLE_DIR / "037_08_FQA_validation_ppo25k_vs_fixed_four_baseline_metrics.csv").relative_to(ROOT)),
        "warning": "PPO daily CSV does not include SWTD; soil-water panel plots four fixed baselines only.",
    }
    pd.Series(note).to_json(TABLE_DIR / "037_08_run_manifest.json", force_ascii=False, indent=2)
    print(note)


if __name__ == "__main__":
    main()
