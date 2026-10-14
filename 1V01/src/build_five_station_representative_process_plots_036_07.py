from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK = "036_07_five_station_representative_five_scenario_process_plots"
OUT = ROOT / "benchmark_results" / TASK
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
DOC = ROOT / "docs" / f"{TASK}_record.md"

SELECTED_COMPARISON = (
    ROOT
    / "benchmark_results"
    / "036_04_select_checkpoint_and_plot_03601_03603_summary"
    / "tables"
    / "036_04_selected_year_level_comparison.csv"
)
BASELINE_DAILY = (
    ROOT
    / "benchmark_results"
    / "034_00_multisite_input_ic1_four_baseline_rebuild"
    / "evaluation"
    / "034_00_full_baseline_daily.csv"
)

STATION_ORDER = ["FQA", "HLA", "LCA", "SYA", "YCA"]
SCENARIO_ORDER = [
    "null",
    "recorded_farmer",
    "dssat_auto",
    "official_extension_expert",
    "ppo_candidate",
]
LABELS = {
    "null": "Null",
    "recorded_farmer": "Recorded farmer",
    "dssat_auto": "DSSAT auto",
    "official_extension_expert": "Official expert",
    "ppo_candidate": "MaskablePPO candidate",
}
COLORS = {
    "null": "#444444",
    "recorded_farmer": "#c94c4c",
    "dssat_auto": "#d99a00",
    "official_extension_expert": "#7b61b8",
    "ppo_candidate": "#2e8b57",
}
LINESTYLES = {
    "null": "-",
    "recorded_farmer": "--",
    "dssat_auto": "-.",
    "official_extension_expert": ":",
    "ppo_candidate": "-",
}


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def scenario_as_text(value: object) -> str:
    if pd.isna(value):
        return "null"
    text = str(value).strip()
    if text.lower() in {"", "nan", "<na>"}:
        return "null"
    if text.startswith("recorded_farmer"):
        return "recorded_farmer"
    return text


def numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def choose_representative_years() -> pd.DataFrame:
    df = pd.read_csv(SELECTED_COMPARISON)
    numeric(
        df,
        [
            "year",
            "final_grnwt",
            "checkpoint_step",
            "yield_gap_vs_four_max",
            "wp_et_gap_vs_four_max",
            "pfp_n_gap_vs_four_max",
        ],
    )
    rows = []
    for station in STATION_ORDER:
        sub = df[df["station_code"].eq(station)].copy()
        sub = sub[sub["final_grnwt"].fillna(0) > 0].copy()
        if sub.empty:
            raise RuntimeError(f"{station} has no nonzero PPO candidate in {SELECTED_COMPARISON}")
        for col in ["yield_gap_vs_four_max", "wp_et_gap_vs_four_max", "pfp_n_gap_vs_four_max"]:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub["positive_metric_count"] = (
            (sub["yield_gap_vs_four_max"] > 0).astype(int)
            + (sub["wp_et_gap_vs_four_max"] > 0).astype(int)
            + (sub["pfp_n_gap_vs_four_max"] > 0).astype(int)
        )
        sub["has_any_four_max_win"] = sub["positive_metric_count"] > 0
        ranked = sub.sort_values(
            ["has_any_four_max_win", "positive_metric_count", "final_grnwt"],
            ascending=[False, False, False],
        )
        chosen = ranked.iloc[0].copy()
        rows.append(chosen)
    return pd.DataFrame(rows)


def finalize_daily(df: pd.DataFrame, source_file: Path, station: str, year: int) -> pd.DataFrame:
    keep = [
        "station_code",
        "site",
        "year",
        "scenario",
        "date",
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
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = np.nan
    out = df[keep].copy()
    out["station_code"] = out["station_code"].fillna(station)
    out["year"] = out["year"].fillna(year)
    numeric(
        out,
        [
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
        ],
    )
    out["irrigation_executed_mm"] = out["irrigation_executed_mm"].fillna(0.0)
    out["nitrogen_executed_kg_ha"] = out["nitrogen_executed_kg_ha"].fillna(0.0)
    out["scenario"] = out["scenario"].map(scenario_as_text)
    out["source_file"] = str(source_file.relative_to(ROOT))
    return out


def load_daily(station: str, year: int, ppo_daily_path: Path) -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_DAILY)
    baseline = baseline[
        baseline["station_code"].eq(station)
        & pd.to_numeric(baseline["year"], errors="coerce").eq(year)
    ].copy()
    baseline["scenario"] = baseline["scenario"].map(scenario_as_text)
    baseline = baseline[baseline["scenario"].isin(SCENARIO_ORDER[:-1])].copy()
    baseline = baseline.rename(
        columns={
            "rain": "rainfall_mm",
            "tmax": "tmax_c",
            "tmin": "tmin_c",
            "grnwt": "grain_yield_kg_ha",
            "topwt": "biomass_kg_ha",
            "swfac": "water_stress_index_wspd",
            "nstres": "nitrogen_stress_index_nstd",
            "irrigation_requested_mm": "irrigation_executed_mm",
            "nitrogen_requested_kg_ha": "nitrogen_executed_kg_ha",
        }
    )
    baseline = finalize_daily(baseline, BASELINE_DAILY, station, year)

    ppo = pd.read_csv(ppo_daily_path)
    ppo["scenario"] = "ppo_candidate"
    ppo = ppo.rename(
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
    ppo = finalize_daily(ppo, ppo_daily_path, station, year)

    combined = pd.concat([baseline, ppo], ignore_index=True, sort=False)
    combined = combined[combined["scenario"].isin(SCENARIO_ORDER)].copy()
    combined = combined.sort_values(["scenario", "dap"]).reset_index(drop=True)
    if combined["scenario"].nunique() < 5:
        present = sorted(combined["scenario"].dropna().unique())
        raise RuntimeError(f"{station}{year} missing scenarios: present={present}")

    combined["grain_delta_kg_ha"] = combined.groupby("scenario")["grain_yield_kg_ha"].diff()
    combined["grain_delta_kg_ha"] = combined["grain_delta_kg_ha"].fillna(combined["grain_yield_kg_ha"])
    combined["common_step_reward"] = (
        combined["grain_delta_kg_ha"].fillna(0.0)
        - 1.1 * combined["irrigation_executed_mm"].fillna(0.0)
        - 1.58 * combined["nitrogen_executed_kg_ha"].fillna(0.0)
    )
    combined["cumulative_common_reward"] = combined.groupby("scenario")["common_step_reward"].cumsum()
    combined["cumulative_irrigation_mm"] = combined.groupby("scenario")["irrigation_executed_mm"].cumsum()
    combined["cumulative_n_kg_ha"] = combined.groupby("scenario")["nitrogen_executed_kg_ha"].cumsum()
    return combined


def summarize(df: pd.DataFrame, station: str, year: int, checkpoint_step: int) -> pd.DataFrame:
    rows = []
    for scenario, g in df.groupby("scenario"):
        gs = g.sort_values("dap")
        final = gs.iloc[-1]
        total_i = float(gs["irrigation_executed_mm"].sum())
        total_n = float(gs["nitrogen_executed_kg_ha"].sum())
        yield_kg = float(final["grain_yield_kg_ha"])
        rows.append(
            {
                "station_code": station,
                "year": year,
                "checkpoint_step": checkpoint_step if scenario == "ppo_candidate" else np.nan,
                "scenario": scenario,
                "label": LABELS.get(scenario, scenario),
                "final_grain_kg_ha": yield_kg,
                "final_biomass_kg_ha": float(final["biomass_kg_ha"]),
                "total_irrigation_mm": total_i,
                "total_nitrogen_kg_ha": total_n,
                "PFP_N_kg_kg": yield_kg / total_n if total_n > 0 else np.nan,
                "irrigation_event_count": int((gs["irrigation_executed_mm"] > 0).sum()),
                "nitrogen_event_count": int((gs["nitrogen_executed_kg_ha"] > 0).sum()),
                "first_irrigation_dap": int(gs.loc[gs["irrigation_executed_mm"] > 0, "dap"].iloc[0])
                if (gs["irrigation_executed_mm"] > 0).any()
                else np.nan,
                "first_n_dap": int(gs.loc[gs["nitrogen_executed_kg_ha"] > 0, "dap"].iloc[0])
                if (gs["nitrogen_executed_kg_ha"] > 0).any()
                else np.nan,
                "max_water_stress_index_wspd": float(gs["water_stress_index_wspd"].max()),
                "max_nitrogen_stress_index_nstd": float(gs["nitrogen_stress_index_nstd"].max()),
                "final_cumulative_common_reward": float(final["cumulative_common_reward"]),
                "source_file": final["source_file"],
            }
        )
    order_map = {scenario: i for i, scenario in enumerate(SCENARIO_ORDER)}
    out = pd.DataFrame(rows)
    out["scenario_order"] = out["scenario"].map(order_map)
    return out.sort_values("scenario_order").drop(columns=["scenario_order"])


def scenario_subset(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    return df[df["scenario"].eq(scenario)].sort_values("dap")


def draw_one(df: pd.DataFrame, station: str, year: int, checkpoint_step: int) -> list[Path]:
    fig, axes = plt.subplots(4, 2, figsize=(15, 12), constrained_layout=True)
    ax = axes.ravel()

    weather = scenario_subset(df, "null")
    if weather.empty:
        weather = df.sort_values("dap").drop_duplicates("dap")

    ax0 = ax[0]
    ax0.bar(weather["dap"], weather["rainfall_mm"], color="#8bb8d8", alpha=0.8, label="Rain")
    ax0.set_title("Weather")
    ax0.set_ylabel("Rain (mm)")
    ax0b = ax0.twinx()
    ax0b.plot(weather["dap"], weather["tmax_c"], color="#d94841", label="Tmax")
    ax0b.plot(weather["dap"], weather["tmin_c"], color="#777777", linestyle="--", label="Tmin")
    ax0b.set_ylabel("Temperature (deg C)")
    handles, labels = ax0.get_legend_handles_labels()
    handles2, labels2 = ax0b.get_legend_handles_labels()
    ax0.legend(handles + handles2, labels + labels2, fontsize=8, loc="upper right")

    ax1 = ax[1]
    for sc in SCENARIO_ORDER:
        g = scenario_subset(df, sc)
        ax1.plot(g["dap"], g["cumulative_irrigation_mm"], color=COLORS[sc], linestyle=LINESTYLES[sc], label=LABELS[sc])
    ax1.set_title("Cumulative irrigation")
    ax1.set_ylabel("mm")
    ax1.legend(fontsize=8, loc="best")

    ax2 = ax[2]
    for sc in SCENARIO_ORDER:
        g = scenario_subset(df, sc)
        ax2.plot(g["dap"], g["water_stress_index_wspd"], color=COLORS[sc], linestyle=LINESTYLES[sc], label=LABELS[sc])
    ax2.set_title("Water stress index")
    ax2.set_ylabel("WSPD (0=no stress)")

    ax3 = ax[3]
    for sc in SCENARIO_ORDER:
        g = scenario_subset(df, sc)
        ax3.plot(g["dap"], g["nitrogen_stress_index_nstd"], color=COLORS[sc], linestyle=LINESTYLES[sc], label=LABELS[sc])
    ax3.set_title("Nitrogen stress index")
    ax3.set_ylabel("NSTD (0=no stress)")

    ax4 = ax[4]
    for sc in SCENARIO_ORDER:
        g = scenario_subset(df, sc)
        ev = g[g["irrigation_executed_mm"] > 0]
        if not ev.empty:
            ax4.vlines(ev["dap"], 0, ev["irrigation_executed_mm"], colors=COLORS[sc], linestyles=LINESTYLES[sc], alpha=0.9, label=LABELS[sc])
            ax4.scatter(ev["dap"], ev["irrigation_executed_mm"], color=COLORS[sc], s=24)
    ax4.set_title("Irrigation events")
    ax4.set_ylabel("mm/event")
    ax4.legend(fontsize=8, loc="upper right", ncol=2)

    ax5 = ax[5]
    for sc in SCENARIO_ORDER:
        g = scenario_subset(df, sc)
        ev = g[g["nitrogen_executed_kg_ha"] > 0]
        if not ev.empty:
            ax5.vlines(ev["dap"], 0, ev["nitrogen_executed_kg_ha"], colors=COLORS[sc], linestyles=LINESTYLES[sc], alpha=0.9, label=LABELS[sc])
            ax5.scatter(ev["dap"], ev["nitrogen_executed_kg_ha"], color=COLORS[sc], s=24)
    ax5.set_title("Nitrogen application events")
    ax5.set_ylabel("kg/ha/event")
    ax5.legend(fontsize=8, loc="upper right", ncol=2)

    ax6 = ax[6]
    ax6.text(
        0.01,
        0.97,
        "Thin companion lines are biomass; thick lines are grain.",
        transform=ax6.transAxes,
        va="top",
        fontsize=8,
    )
    for sc in SCENARIO_ORDER:
        g = scenario_subset(df, sc)
        ax6.plot(g["dap"], g["biomass_kg_ha"], color=COLORS[sc], linestyle=LINESTYLES[sc], alpha=0.35)
        ax6.plot(g["dap"], g["grain_yield_kg_ha"], color=COLORS[sc], linestyle=LINESTYLES[sc], linewidth=2.0, label=LABELS[sc])
    ax6.set_title("Grain and biomass trajectories")
    ax6.set_ylabel("kg/ha")
    ax6.set_xlabel("DAP")

    ax7 = ax[7]
    for sc in SCENARIO_ORDER:
        g = scenario_subset(df, sc)
        ax7.plot(g["dap"], g["cumulative_common_reward"], color=COLORS[sc], linestyle=LINESTYLES[sc], label=LABELS[sc])
    ax7.set_title("Cumulative common reward")
    ax7.set_ylabel("dGRNWT - 1.1I - 1.58N")
    ax7.set_xlabel("DAP")
    ax7.legend(fontsize=8, loc="best")

    for axis in ax:
        axis.grid(alpha=0.25)

    title = f"{station}{year} five-scenario daily process (036_01 selected ckpt{checkpoint_step})"
    fig.suptitle(title, x=0.02, ha="left", fontsize=14, fontweight="bold")
    base = f"036_07_{station.lower()}{year}_five_scenario_process"
    paths = []
    for suffix in ["png", "svg"]:
        path = FIG_DIR / f"{base}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def write_record(selection: pd.DataFrame, combined_summary: pd.DataFrame, fig_paths: list[Path]) -> None:
    lines = [
        "# 036_07 五站点代表年份五情景过程图记录",
        "",
        "## 任务边界",
        "",
        "- 不训练。",
        "- 不重跑 DSSAT。",
        "- 不修改已有结果。",
        "- PPO daily 来自 036_01 当前修正后 rerun。",
        "- 四情景 daily 来自 034_00 统一 IC=1 四基线重建结果。",
        "- 每站点选择一个代表年份，仅用于备份和措施合理性人工审查。",
        "- 累计奖励统一按 `ΔGRNWT - 1.1*irrigation - 1.58*nitrogen` 重算。",
        "",
        "## 代表年份选择结果",
        "",
        selection[
            [
                "station_code",
                "year",
                "checkpoint_step",
                "final_grnwt",
                "positive_metric_count",
                "yield_gap_vs_four_max",
                "wp_et_gap_vs_four_max",
                "pfp_n_gap_vs_four_max",
                "daily_csv_path",
            ]
        ].round(4).to_markdown(index=False),
        "",
        "## 五情景终值汇总",
        "",
        combined_summary.round(4).to_markdown(index=False),
        "",
        "## 输出图件",
        "",
    ]
    lines.extend([f"- `{p.relative_to(ROOT)}`" for p in fig_paths])
    lines.extend(
        [
            "",
            "## 输出表格",
            "",
            f"- `{(TABLE_DIR / '036_07_representative_selection.csv').relative_to(ROOT)}`",
            f"- `{(TABLE_DIR / '036_07_five_station_five_scenario_summary.csv').relative_to(ROOT)}`",
            f"- `{(TABLE_DIR / '036_07_five_station_five_scenario_daily.csv').relative_to(ROOT)}`",
        ]
    )
    DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    selection = choose_representative_years()
    all_daily = []
    all_summary = []
    all_figs = []
    for _, row in selection.iterrows():
        station = str(row["station_code"])
        year = int(row["year"])
        checkpoint_step = int(row["checkpoint_step"])
        ppo_daily_path = ROOT / str(row["daily_csv_path"])
        df = load_daily(station, year, ppo_daily_path)
        summary = summarize(df, station, year, checkpoint_step)
        all_daily.append(df)
        all_summary.append(summary)
        all_figs.extend(draw_one(df, station, year, checkpoint_step))

    daily = pd.concat(all_daily, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True)
    selection.to_csv(TABLE_DIR / "036_07_representative_selection.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(TABLE_DIR / "036_07_five_station_five_scenario_daily.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLE_DIR / "036_07_five_station_five_scenario_summary.csv", index=False, encoding="utf-8-sig")
    write_record(selection, summary, all_figs)
    print(
        json.dumps(
            {
                "task": TASK,
                "station_count": int(selection["station_code"].nunique()),
                "selected": selection[["station_code", "year", "checkpoint_step", "positive_metric_count"]].to_dict(orient="records"),
                "figure_count": len(all_figs),
                "figures": [str(p.relative_to(ROOT)) for p in all_figs],
                "record_md": str(DOC.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
