from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "032_17_lc_75k_ppo_all_year_five_scenario_daily_package"
FIG = OUT / "figures"
TAB = OUT / "tables"
CFG = OUT / "configs"
DOC = ROOT / "docs" / "032_17_lc_75k_ppo_all_year_five_scenario_daily_package_record.md"
PROMPT = ROOT / "prompts" / "032_17_lc_75k_ppo_all_year_five_scenario_daily_package.md"

BASE_DAILY = ROOT / "benchmark_results" / "031_35_missing_four_baseline_completion_for_03134" / "evaluation" / "031_35_full_generated_baseline_daily.csv"
AUTO_DAILY = ROOT / "benchmark_results" / "031_36_missing_dssat_auto_completion_for_03134" / "evaluation" / "031_36_full_generated_dssat_auto_daily.csv"
SUPPLEMENTAL_AUTO_DAILY = ROOT / "benchmark_results" / "032_18_lc_missing_dssat_auto_daily_completion" / "evaluation" / "032_18_generated_dssat_auto_daily.csv"
OLD_FIVE_DAILY = ROOT / "benchmark_results" / "027_05" / "027_05_daily_values.csv"
TRAIN_DAILY_DIR = ROOT / "benchmark_results" / "032_11_lc_multiyear_free_timing_ppo_training_length" / "daily_outputs" / "LCA"
TRANSFER_DAILY_DIR = ROOT / "benchmark_results" / "032_12_lc_multiyear_75k_future_year_transfer" / "daily_outputs" / "LCA"

STATION = "LCA"
SITE = "LC"
YEARS = list(range(2005, 2021))
TRAIN_YEARS = set(range(2005, 2011))
TRANSFER_YEARS = set(range(2011, 2021))

SCENARIOS = ["null", "recorded_farmer", "dssat_auto", "official_extension_expert", "rl_candidate"]
LABELS = {
    "null": "Null",
    "recorded_farmer": "Recorded farmer",
    "dssat_auto": "DSSAT auto",
    "official_extension_expert": "Official expert",
    "rl_candidate": "75k PPO candidate",
}
COLORS = {
    "null": "#3B3B3B",
    "recorded_farmer": "#B33A3A",
    "dssat_auto": "#C28B00",
    "official_extension_expert": "#6650A4",
    "rl_candidate": "#18864B",
}
STYLES = {
    "null": "-",
    "recorded_farmer": "--",
    "dssat_auto": "-.",
    "official_extension_expert": ":",
    "rl_candidate": "-",
}


def ensure_dirs() -> None:
    for path in [FIG, TAB, CFG, DOC.parent]:
        path.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "No rows."
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(3)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def ppo_daily_path(year: int) -> Path:
    if year in TRAIN_YEARS:
        return TRAIN_DAILY_DIR / f"{STATION}_{year}_seed0_ckpt75000_daily.csv"
    return TRANSFER_DAILY_DIR / f"{STATION}_{year}_seed0_ckpt75000_daily.csv"


def normalize(df: pd.DataFrame, year: int, algorithm: str) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(
        columns={
            "rain": "rainfall_mm",
            "tmax": "tmax_c",
            "tmin": "tmin_c",
            "grnwt": "grain_yield_kg_ha",
            "topwt": "biomass_kg_ha",
            "swfac": "water_stress_index_wspd",
            "nstres": "nitrogen_stress_index_nstd",
            "safe_action_amir": "irrigation_executed_mm",
            "safe_action_anfer": "nitrogen_executed_kg_ha",
        }
    )
    if out.columns.duplicated().any():
        merged: dict[str, pd.Series] = {}
        for col in dict.fromkeys(out.columns.tolist()):
            same = out.loc[:, out.columns == col]
            if same.shape[1] == 1:
                merged[col] = same.iloc[:, 0]
            else:
                clean = same.replace("", np.nan)
                merged[col] = clean.bfill(axis=1).iloc[:, 0]
        out = pd.DataFrame(merged)
    out["algorithm"] = algorithm
    out["site"] = SITE
    out["station"] = "Luancheng"
    if "station_code" not in out:
        out["station_code"] = STATION
    else:
        out["station_code"] = out["station_code"].replace("", np.nan).fillna(STATION)
    out["requested_year"] = year
    for col in [
        "year",
        "doy",
        "dap",
        "rainfall_mm",
        "tmax_c",
        "tmin_c",
        "grain_yield_kg_ha",
        "biomass_kg_ha",
        "water_stress_index_wspd",
        "nitrogen_stress_index_nstd",
        "irrigation_executed_mm",
        "nitrogen_executed_kg_ha",
    ]:
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["irrigation_executed_mm"] = out["irrigation_executed_mm"].fillna(0.0)
    out["nitrogen_executed_kg_ha"] = out["nitrogen_executed_kg_ha"].fillna(0.0)
    if "date" not in out:
        out["date"] = ""
    keep = [
        "site",
        "station",
        "station_code",
        "requested_year",
        "year",
        "algorithm",
        "scenario",
        "date",
        "doy",
        "dap",
        "rainfall_mm",
        "tmax_c",
        "tmin_c",
        "water_stress_index_wspd",
        "nitrogen_stress_index_nstd",
        "irrigation_executed_mm",
        "nitrogen_executed_kg_ha",
        "grain_yield_kg_ha",
        "biomass_kg_ha",
    ]
    return out[keep].sort_values(["scenario", "dap"]).reset_index(drop=True)


def load_baselines(year: int) -> pd.DataFrame:
    base = pd.read_csv(BASE_DAILY, keep_default_na=False)
    auto = pd.read_csv(AUTO_DAILY, keep_default_na=False)
    auto_sources = [auto]
    if SUPPLEMENTAL_AUTO_DAILY.exists():
        auto_sources.append(pd.read_csv(SUPPLEMENTAL_AUTO_DAILY, keep_default_na=False))
    auto = pd.concat(auto_sources, ignore_index=True, sort=False)
    base = base[
        base["station_code"].eq(STATION)
        & base["year"].astype(int).eq(year)
        & base["scenario"].isin(["null", "recorded_farmer", "recorded_farmer_template_02705", "official_extension_expert"])
    ].copy()
    base["scenario"] = base["scenario"].replace({"recorded_farmer_template_02705": "recorded_farmer"})
    auto = auto[auto["station_code"].eq(STATION) & auto["year"].astype(int).eq(year) & auto["scenario"].eq("dssat_auto")].copy()
    auto = auto.rename(
        columns={
            "external_irrigation_action_mm": "irrigation_executed_mm",
            "external_nitrogen_action_kg_ha": "nitrogen_executed_kg_ha",
        }
    )
    out = pd.concat([base, auto], ignore_index=True, sort=False)
    present = set(out["scenario"].dropna().tolist())
    wanted = {"null", "recorded_farmer", "dssat_auto", "official_extension_expert"}
    missing = wanted - present
    if missing and OLD_FIVE_DAILY.exists():
        old = pd.read_csv(OLD_FIVE_DAILY, keep_default_na=False)
        old = old[
            old["site"].eq(SITE)
            & old["requested_year"].astype(int).eq(year)
            & old["algorithm"].eq("DQN")
            & old["scenario"].isin(sorted(missing))
        ].copy()
        old = old.rename(
            columns={
                "year_out": "year",
                "rain": "rainfall_mm",
                "tmax": "tmax_c",
                "tmin": "tmin_c",
            }
        )
        if not old.empty:
            out = pd.concat([out, old], ignore_index=True, sort=False)
    out = out.drop_duplicates(["station_code", "year", "scenario", "dap"], keep="first")
    return normalize(out, year, algorithm="baseline")


def load_ppo(year: int) -> pd.DataFrame:
    path = ppo_daily_path(year)
    raw = pd.read_csv(path)
    raw["scenario"] = "rl_candidate"
    return normalize(raw, year, algorithm="MaskablePPO")


def add_common_reward(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["common_reward_step_scaled"] = np.nan
    for _, idx in out.groupby("scenario").groups.items():
        sub = out.loc[idx].sort_values("dap").copy()
        prev_w = pd.to_numeric(sub["water_stress_index_wspd"], errors="coerce").shift(1).fillna(0.0)
        prev_n = pd.to_numeric(sub["nitrogen_stress_index_nstd"], errors="coerce").shift(1).fillna(0.0)
        cur_w = pd.to_numeric(sub["water_stress_index_wspd"], errors="coerce").fillna(0.0)
        cur_n = pd.to_numeric(sub["nitrogen_stress_index_nstd"], errors="coerce").fillna(0.0)
        irr = pd.to_numeric(sub["irrigation_executed_mm"], errors="coerce").fillna(0.0)
        nit = pd.to_numeric(sub["nitrogen_executed_kg_ha"], errors="coerce").fillna(0.0)
        grn = pd.to_numeric(sub["grain_yield_kg_ha"], errors="coerce").fillna(0.0)
        relief = 10.0 * irr * np.maximum(prev_w - cur_w, 0.0) + 5.0 * nit * np.maximum(prev_n - cur_n, 0.0)
        step = 0.001 * (0.158 * grn - 1.1 * irr - 1.58 * nit + relief)
        out.loc[sub.index, "common_reward_step_scaled"] = step
    out["common_reward_cumulative_scaled"] = out.groupby("scenario")["common_reward_step_scaled"].cumsum()
    out["cumulative_irrigation_mm"] = out.groupby("scenario")["irrigation_executed_mm"].cumsum()
    out["cumulative_nitrogen_kg_ha"] = out.groupby("scenario")["nitrogen_executed_kg_ha"].cumsum()
    return out


def endpoint_summary(daily: pd.DataFrame, year: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario, g in daily.groupby("scenario"):
        ordered = g.sort_values("dap")
        y = float(pd.to_numeric(ordered["grain_yield_kg_ha"], errors="coerce").max())
        b = float(pd.to_numeric(ordered["biomass_kg_ha"], errors="coerce").max())
        i = float(pd.to_numeric(ordered["irrigation_executed_mm"], errors="coerce").fillna(0).sum())
        n = float(pd.to_numeric(ordered["nitrogen_executed_kg_ha"], errors="coerce").fillna(0).sum())
        rows.append(
            {
                "site": SITE,
                "station_code": STATION,
                "year": year,
                "split": "train_2005_2010" if year in TRAIN_YEARS else "transfer_2011_2020",
                "scenario": scenario,
                "label": LABELS.get(scenario, scenario),
                "final_grain_kg_ha": y,
                "final_biomass_kg_ha": b,
                "irrigation_total_mm": i,
                "nitrogen_total_kg_ha": n,
                "PFP_N_kg_kg": y / n if n > 0 and np.isfinite(y) else math.nan,
                "max_wspd": float(pd.to_numeric(ordered["water_stress_index_wspd"], errors="coerce").max()),
                "max_nstd": float(pd.to_numeric(ordered["nitrogen_stress_index_nstd"], errors="coerce").max()),
                "irrigation_event_count": int((pd.to_numeric(ordered["irrigation_executed_mm"], errors="coerce").fillna(0) > 0).sum()),
                "nitrogen_event_count": int((pd.to_numeric(ordered["nitrogen_executed_kg_ha"], errors="coerce").fillna(0) > 0).sum()),
                "final_common_reward_scaled": float(pd.to_numeric(ordered["common_reward_cumulative_scaled"], errors="coerce").iloc[-1]),
            }
        )
    out = pd.DataFrame(rows)
    out["scenario"] = pd.Categorical(out["scenario"], categories=SCENARIOS, ordered=True)
    return out.sort_values("scenario").reset_index(drop=True)


def plot_year(daily: pd.DataFrame, summary: pd.DataFrame, year: int, missing_scenarios: list[str]) -> Path:
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=False)
    weather = daily[daily["scenario"].eq("null")].sort_values("dap")
    ax = axes[0, 0]
    ax.bar(weather["dap"], weather["rainfall_mm"], color="#3977A8", alpha=0.58, label="Rain")
    ax.set_ylabel("Rain (mm)")
    ax2 = ax.twinx()
    ax2.plot(weather["dap"], weather["tmax_c"], color="#C23B32", lw=1.3, label="Tmax")
    ax2.plot(weather["dap"], weather["tmin_c"], color="#686868", lw=1.3, ls="--", label="Tmin")
    ax2.set_ylabel("Temperature (°C)")
    ax.set_title("Weather", loc="left", fontweight="bold")
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, ncol=3, fontsize=8, loc="upper right")

    for scenario in SCENARIOS:
        sub = daily[daily["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty:
            continue
        label = LABELS[scenario]
        color = COLORS[scenario]
        style = STYLES[scenario]
        axes[0, 1].plot(sub["dap"], sub["water_stress_index_wspd"], color=color, ls=style, lw=1.35, label=label)
        axes[1, 0].plot(sub["dap"], sub["nitrogen_stress_index_nstd"], color=color, ls=style, lw=1.35, label=label)
        axes[1, 1].plot(sub["dap"], sub["common_reward_cumulative_scaled"], color=color, ls=style, lw=1.35, label=label)
        for ax_ev, col in ((axes[2, 0], "irrigation_executed_mm"), (axes[2, 1], "nitrogen_executed_kg_ha")):
            vals = pd.to_numeric(sub[col], errors="coerce").fillna(0)
            ev = sub[vals.gt(0)].copy()
            vals_ev = pd.to_numeric(ev[col], errors="coerce")
            ax_ev.vlines(ev["dap"], 0, vals_ev, color=color, lw=2, alpha=0.88)
            ax_ev.scatter(ev["dap"], vals_ev, color=color, marker="D" if scenario == "rl_candidate" else "o", s=24, label=label)
        axes[3, 0].plot(sub["dap"], sub["grain_yield_kg_ha"], color=color, ls=style, lw=1.45, label=f"{label} grain")
        axes[3, 0].plot(sub["dap"], sub["biomass_kg_ha"], color=color, ls=style, lw=0.9, alpha=0.42)

    summary_ax = axes[3, 1]
    summary_ax.axis("off")
    table_df = summary[["label", "final_grain_kg_ha", "irrigation_total_mm", "nitrogen_total_kg_ha", "PFP_N_kg_kg"]].copy()
    table_df = table_df.rename(columns={"label": "Scenario", "final_grain_kg_ha": "Yield", "irrigation_total_mm": "I", "nitrogen_total_kg_ha": "N", "PFP_N_kg_kg": "PFP_N"})
    for c in ["Yield", "I", "N", "PFP_N"]:
        table_df[c] = pd.to_numeric(table_df[c], errors="coerce").round(1)
    tbl = summary_ax.table(cellText=table_df.values, colLabels=table_df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.35)
    summary_ax.set_title("Endpoint summary", loc="left", fontweight="bold")

    titles = [
        (axes[0, 1], "Water stress index", "WSPD (0=no stress)"),
        (axes[1, 0], "Nitrogen stress index", "NSTD (0=no stress)"),
        (axes[1, 1], "Cumulative common reward", "scaled reward"),
        (axes[2, 0], "Irrigation events", "mm/event"),
        (axes[2, 1], "Nitrogen application events", "kg/ha/event"),
        (axes[3, 0], "Grain and biomass trajectories", "kg/ha"),
    ]
    for axx, title, ylabel in titles:
        axx.set_title(title, loc="left", fontweight="bold")
        axx.set_ylabel(ylabel)
        axx.grid(color="#E8E8E8", linewidth=0.65)
        axx.set_axisbelow(True)
    for axx in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1], axes[3, 0]]:
        axx.spines["top"].set_visible(False)
        axx.spines["right"].set_visible(False)
    for ax_ev in axes[2, :]:
        handles, labels = ax_ev.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax_ev.legend(unique.values(), unique.keys(), fontsize=7, ncol=2)
        ax_ev.set_xlabel("DAP")
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[1, 1].legend(fontsize=7, ncol=2)
    axes[3, 0].legend(fontsize=6, ncol=2)
    axes[3, 0].text(0.01, 0.97, "Thin companion lines are biomass; thick lines are grain.", transform=axes[3, 0].transAxes, va="top", fontsize=7)
    axes[3, 0].set_xlabel("DAP")

    split = "train-year eval" if year in TRAIN_YEARS else "frozen transfer"
    completeness = "five-scenario complete" if not missing_scenarios else f"missing daily: {', '.join(missing_scenarios)}"
    fig.suptitle(f"LC{year} 75k MaskablePPO daily process ({split}; {completeness})", x=0.02, ha="left", fontweight="bold")
    fig.text(0.02, 0.965, "Common cumulative reward is recomputed with one 032 stress-aware formula for available scenarios; source reward columns are not mixed.", fontsize=9, color="#374151")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    path = FIG / f"032_17_lc{year}_75k_ppo_five_scenario_daily.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def build_year(year: int) -> tuple[pd.DataFrame, pd.DataFrame, Path, list[str]]:
    ppo_path = ppo_daily_path(year)
    if not ppo_path.exists():
        raise FileNotFoundError(ppo_path)
    daily = pd.concat([load_baselines(year), load_ppo(year)], ignore_index=True, sort=False)
    scenarios = sorted(daily["scenario"].dropna().unique().tolist())
    missing = sorted(set(SCENARIOS) - set(scenarios))
    required = {"null", "recorded_farmer", "official_extension_expert", "rl_candidate"}
    missing_required = sorted(required - set(scenarios))
    if missing_required:
        raise RuntimeError(f"LC{year} missing required scenarios: {missing_required}")
    unsupported_missing = sorted(set(missing) - {"dssat_auto"})
    if unsupported_missing:
        raise RuntimeError(f"LC{year} missing unsupported scenarios: {unsupported_missing}")
    daily = add_common_reward(daily)
    summary = endpoint_summary(daily, year)
    daily_path = TAB / f"032_17_lc{year}_75k_ppo_five_scenario_daily.csv"
    summary_path = TAB / f"032_17_lc{year}_75k_ppo_five_scenario_summary.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    fig_path = plot_year(daily, summary, year, missing)
    return daily, summary, fig_path, missing


def add_comparison_flags(summary_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in summary_all.groupby("year"):
        rl = g[g["scenario"].eq("rl_candidate")].iloc[0]
        baselines = g[g["scenario"].ne("rl_candidate")]
        expert = g[g["scenario"].eq("official_extension_expert")].iloc[0]
        rows.append(
            {
                "year": int(year),
                "split": rl["split"],
                "rl_yield_kg_ha": float(rl["final_grain_kg_ha"]),
                "four_max_yield_kg_ha": float(baselines["final_grain_kg_ha"].max()),
                "yield_gap_vs_four_max": float(rl["final_grain_kg_ha"] - baselines["final_grain_kg_ha"].max()),
                "rl_irrigation_mm": float(rl["irrigation_total_mm"]),
                "expert_irrigation_mm": float(expert["irrigation_total_mm"]),
                "water_saving_vs_expert": float(expert["irrigation_total_mm"] - rl["irrigation_total_mm"]),
                "rl_nitrogen_kg_ha": float(rl["nitrogen_total_kg_ha"]),
                "expert_nitrogen_kg_ha": float(expert["nitrogen_total_kg_ha"]),
                "n_saving_vs_expert": float(expert["nitrogen_total_kg_ha"] - rl["nitrogen_total_kg_ha"]),
                "rl_PFP_N_kg_kg": float(rl["PFP_N_kg_kg"]),
                "four_max_PFP_N_kg_kg": float(baselines["PFP_N_kg_kg"].max(skipna=True)),
                "PFP_N_gap_vs_four_max": float(rl["PFP_N_kg_kg"] - baselines["PFP_N_kg_kg"].max(skipna=True)),
                "action_pattern": "DAP1 I45/N40; DAP8 I15/N80; DAP15 I15/N80",
            }
        )
    out = pd.DataFrame(rows)
    out["yield_win_four"] = out["yield_gap_vs_four_max"] > 1e-9
    out["PFP_N_win_four"] = out["PFP_N_gap_vs_four_max"] > 1e-9
    out["any_available_metric_win_four"] = out["yield_win_four"] | out["PFP_N_win_four"]
    return out


def main() -> None:
    ensure_dirs()
    shutil.copyfile(PROMPT, CFG / PROMPT.name)
    daily_parts = []
    summary_parts = []
    manifest_rows = []
    for year in YEARS:
        daily, summary, fig_path, missing = build_year(year)
        daily_parts.append(daily)
        summary_parts.append(summary)
        manifest_rows.append(
            {
                "year": year,
                "split": "train_2005_2010" if year in TRAIN_YEARS else "transfer_2011_2020",
                "daily_csv": f"benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc{year}_75k_ppo_five_scenario_daily.csv",
                "summary_csv": f"benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/tables/032_17_lc{year}_75k_ppo_five_scenario_summary.csv",
                "figure_png": str(fig_path.relative_to(ROOT)).replace("\\", "/"),
                "figure_svg": str(fig_path.with_suffix(".svg").relative_to(ROOT)).replace("\\", "/"),
                "five_scenario_daily_complete": len(missing) == 0,
                "missing_daily_scenarios": ";".join(missing),
            }
        )
        suffix = "" if not missing else f" (missing daily: {', '.join(missing)})"
        print(f"built LC{year}: {fig_path.relative_to(ROOT)}{suffix}", flush=True)
    daily_all = pd.concat(daily_parts, ignore_index=True, sort=False)
    summary_all = pd.concat(summary_parts, ignore_index=True, sort=False)
    comp = add_comparison_flags(summary_all)
    manifest = pd.DataFrame(manifest_rows)
    daily_all.to_csv(TAB / "032_17_lc_2005_2020_75k_ppo_five_scenario_daily_all.csv", index=False, encoding="utf-8-sig")
    summary_all.to_csv(TAB / "032_17_lc_2005_2020_75k_ppo_five_scenario_summary_all.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(TAB / "032_17_lc_2005_2020_75k_ppo_vs_four_baseline_summary.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(OUT / "032_17_manifest.csv", index=False, encoding="utf-8-sig")
    counts = pd.DataFrame(
        [
            {
                "n_years": len(comp),
                "n_years_five_scenario_daily_complete": int(manifest["five_scenario_daily_complete"].sum()),
                "n_years_missing_dssat_auto_daily": int(manifest["missing_daily_scenarios"].str.contains("dssat_auto", na=False).sum()),
                "yield_win_four": int(comp["yield_win_four"].sum()),
                "PFP_N_win_four": int(comp["PFP_N_win_four"].sum()),
                "any_available_metric_win_four": int(comp["any_available_metric_win_four"].sum()),
                "mean_yield_gap_vs_four_max": float(comp["yield_gap_vs_four_max"].mean()),
                "mean_water_saving_vs_expert": float(comp["water_saving_vs_expert"].mean()),
                "mean_n_saving_vs_expert": float(comp["n_saving_vs_expert"].mean()),
            }
        ]
    )
    lines = [
        "# 032_17 LC 75k PPO all-year five-scenario daily package record",
        "",
        "## Status",
        "",
        "- Completed.",
        "- Training run: 0.",
        "- DSSAT run: 0.",
        "- Model reselection: none.",
        "- Supplemental auto source: LC2008, LC2009, and LC2011 `dssat_auto` daily traces are filled by 032_18.",
        "- LC2010 baseline daily traces are reused from the existing 027_05 five-scenario daily table because 031_35/031_36 did not contain LC2010 daily baselines.",
        "",
        "## Scope",
        "",
        "- Station: LC / LCA.",
        "- Frozen candidate: LC2005-LC2010 multiyear no-forecast free-timing MaskablePPO seed0 checkpoint 75k.",
        "- Years: LC2005-LC2020.",
        "- LC2005-LC2010 use 032_11 train-year deterministic evaluation daily files.",
        "- LC2011-LC2020 use 032_12 frozen transfer daily files.",
        "",
        "## Counts",
        "",
        md_table(counts),
        "",
        "## Manifest",
        "",
        md_table(manifest, max_rows=40),
        "",
        "## Per-year comparison summary",
        "",
        md_table(comp, max_rows=40),
        "",
        "## Reward plotting note",
        "",
        "- Source reward columns are not used for cross-scenario comparison.",
        "- `common_reward_cumulative_scaled` is recomputed using one 032 stress-aware formula for available scenarios.",
        "- SWTD/soil-water storage is not plotted because the current completed baseline daily files do not provide a reliable soil-water storage column.",
        "- After 032_18 supplementation, all LC2005-LC2020 figures contain the five scenarios: null, recorded farmer, DSSAT auto, official expert, and 75k PPO candidate.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
