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
OUT = ROOT / "benchmark_results" / "032_15_lc2018_75k_ppo_five_scenario_daily_figure"
FIG = OUT / "figures"
TAB = OUT / "tables"
CFG = OUT / "configs"
DOC = ROOT / "docs" / "032_15_lc2018_75k_ppo_five_scenario_daily_figure_record.md"
PROMPT = ROOT / "prompts" / "032_15_lc2018_75k_ppo_five_scenario_daily_figure.md"

BASE_DAILY = ROOT / "benchmark_results" / "031_35_missing_four_baseline_completion_for_03134" / "evaluation" / "031_35_full_generated_baseline_daily.csv"
AUTO_DAILY = ROOT / "benchmark_results" / "031_36_missing_dssat_auto_completion_for_03134" / "evaluation" / "031_36_full_generated_dssat_auto_daily.csv"
PPO_DAILY = ROOT / "benchmark_results" / "032_12_lc_multiyear_75k_future_year_transfer" / "daily_outputs" / "LCA" / "LCA_2018_seed0_ckpt75000_daily.csv"

STATION = "LCA"
SITE = "LC"
YEAR = 2018

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


def load_baselines() -> pd.DataFrame:
    base = pd.read_csv(BASE_DAILY, keep_default_na=False)
    auto = pd.read_csv(AUTO_DAILY, keep_default_na=False)
    base = base[
        base["station_code"].eq(STATION)
        & base["year"].astype(str).eq(str(YEAR))
        & base["scenario"].isin(["null", "recorded_farmer", "recorded_farmer_template_02705", "official_extension_expert"])
    ].copy()
    base["scenario"] = base["scenario"].replace({"recorded_farmer_template_02705": "recorded_farmer"})
    auto = auto[auto["station_code"].eq(STATION) & auto["year"].astype(str).eq(str(YEAR)) & auto["scenario"].eq("dssat_auto")].copy()
    auto = auto.rename(
        columns={
            "external_irrigation_action_mm": "irrigation_executed_mm",
            "external_nitrogen_action_kg_ha": "nitrogen_executed_kg_ha",
        }
    )
    out = pd.concat([base, auto], ignore_index=True, sort=False)
    out = out.drop_duplicates(["station_code", "year", "scenario", "dap"], keep="first")
    return normalize(out, algorithm="baseline")


def load_ppo() -> pd.DataFrame:
    raw = pd.read_csv(PPO_DAILY)
    raw["scenario"] = "rl_candidate"
    return normalize(raw, algorithm="MaskablePPO")


def normalize(df: pd.DataFrame, algorithm: str) -> pd.DataFrame:
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
    out["algorithm"] = algorithm
    out["site"] = SITE
    out["station"] = "Luancheng"
    out["requested_year"] = YEAR
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
    out["date"] = out.get("date", "")
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


def add_common_reward(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["common_reward_step_scaled"] = np.nan
    for scenario, idx in out.groupby("scenario").groups.items():
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


def endpoint_summary(daily: pd.DataFrame) -> pd.DataFrame:
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
                "year": YEAR,
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


def plot(daily: pd.DataFrame, summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=False)
    weather = daily[daily["scenario"].eq("null")].sort_values("dap")
    ax = axes[0, 0]
    ax.bar(weather["dap"], pd.to_numeric(weather["rainfall_mm"], errors="coerce"), color="#3977A8", alpha=0.58, label="Rain")
    ax.set_ylabel("Rain (mm)")
    ax2 = ax.twinx()
    ax2.plot(weather["dap"], pd.to_numeric(weather["tmax_c"], errors="coerce"), color="#C23B32", lw=1.3, label="Tmax")
    ax2.plot(weather["dap"], pd.to_numeric(weather["tmin_c"], errors="coerce"), color="#686868", lw=1.3, ls="--", label="Tmin")
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
    table_cols = ["label", "final_grain_kg_ha", "irrigation_total_mm", "nitrogen_total_kg_ha", "PFP_N_kg_kg"]
    table_df = summary[table_cols].copy()
    table_df = table_df.rename(
        columns={
            "label": "Scenario",
            "final_grain_kg_ha": "Yield",
            "irrigation_total_mm": "I",
            "nitrogen_total_kg_ha": "N",
            "PFP_N_kg_kg": "PFP_N",
        }
    )
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

    fig.suptitle("LC2018 75k MaskablePPO five-scenario daily process", x=0.02, ha="left", fontweight="bold")
    fig.text(0.02, 0.965, "Common cumulative reward is recomputed with the 032 stress-aware formula for all scenarios; source reward columns are not mixed.", fontsize=9, color="#374151")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    path = FIG / "032_15_lc2018_75k_ppo_five_scenario_daily.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ensure_dirs()
    shutil.copyfile(PROMPT, CFG / PROMPT.name)
    daily = pd.concat([load_baselines(), load_ppo()], ignore_index=True, sort=False)
    daily = add_common_reward(daily)
    scenarios = sorted(daily["scenario"].dropna().unique().tolist())
    missing = sorted(set(SCENARIOS) - set(scenarios))
    if missing:
        raise RuntimeError(f"Missing scenarios: {missing}")
    summary = endpoint_summary(daily)
    daily_path = TAB / "032_15_lc2018_75k_ppo_five_scenario_daily.csv"
    summary_path = TAB / "032_15_lc2018_75k_ppo_five_scenario_summary.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    fig_path = plot(daily, summary)
    lines = [
        "# 032_15 LC2018 75k PPO five-scenario daily figure record",
        "",
        "## Status",
        "",
        "- Completed.",
        "- Training run: 0.",
        "- DSSAT run: 0.",
        "- Model reselection: none.",
        "",
        "## Inputs",
        "",
        f"- PPO daily: `{PPO_DAILY.relative_to(ROOT)}`",
        f"- Baseline daily: `{BASE_DAILY.relative_to(ROOT)}`",
        f"- DSSAT-auto daily: `{AUTO_DAILY.relative_to(ROOT)}`",
        "",
        "## Outputs",
        "",
        f"- Daily table: `{daily_path.relative_to(ROOT)}`",
        f"- Summary table: `{summary_path.relative_to(ROOT)}`",
        f"- Figure: `{fig_path.relative_to(ROOT)}`",
        "",
        "## Scenario summary",
        "",
        md_table(summary),
        "",
        "## Reward plotting note",
        "",
        "- Source reward columns are not used for cross-scenario comparison.",
        "- `common_reward_cumulative_scaled` is recomputed using one 032 stress-aware formula for all five scenarios.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")
    print(fig_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
