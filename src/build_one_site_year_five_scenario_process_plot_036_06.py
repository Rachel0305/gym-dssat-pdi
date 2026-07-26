from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK = "036_06_one_site_year_five_scenario_process_plot"
OUT = ROOT / "benchmark_results" / TASK
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
DOC = ROOT / "docs" / f"{TASK}_record.md"

STATION = "LCA"
YEAR = 2021
PPO_DAILY = ROOT / "benchmark_results" / "036_01_original_free_timing_maskableppo_ic1_linked_five_site_half_split_rerun" / "daily_outputs" / "LCA" / "LCA_2021_seed0_ckpt100000_daily.csv"
BASELINE_DAILY = ROOT / "benchmark_results" / "031_35_missing_four_baseline_completion_for_03134" / "evaluation" / "031_35_full_generated_baseline_daily.csv"
AUTO_DAILY = ROOT / "benchmark_results" / "031_36_missing_dssat_auto_completion_for_03134" / "evaluation" / "031_36_full_generated_dssat_auto_daily.csv"

SCENARIO_ORDER = ["null", "recorded_farmer", "dssat_auto", "official_extension_expert", "ppo_candidate"]
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


def finalize_daily(df: pd.DataFrame, source_file: Path) -> pd.DataFrame:
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


def load_daily() -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_DAILY)
    baseline = baseline[(baseline["station_code"].eq(STATION)) & (pd.to_numeric(baseline["year"], errors="coerce").eq(YEAR))].copy()
    baseline["scenario"] = baseline["scenario"].map(scenario_as_text)
    baseline = baseline[baseline["scenario"].isin(["null", "recorded_farmer", "official_extension_expert"])].copy()
    baseline = baseline.rename(
        columns={
            "rain": "rainfall_mm",
            "tmax": "tmax_c",
            "tmin": "tmin_c",
            "grnwt": "grain_yield_kg_ha",
            "topwt": "biomass_kg_ha",
            "swfac": "water_stress_index_wspd",
            "nstres": "nitrogen_stress_index_nstd",
        }
    )
    baseline = finalize_daily(baseline, BASELINE_DAILY)

    auto = pd.read_csv(AUTO_DAILY)
    auto = auto[(auto["station_code"].eq(STATION)) & (pd.to_numeric(auto["year"], errors="coerce").eq(YEAR))].copy()
    auto["scenario"] = "dssat_auto"
    auto = auto.rename(
        columns={
            "rain": "rainfall_mm",
            "tmax": "tmax_c",
            "tmin": "tmin_c",
            "grnwt": "grain_yield_kg_ha",
            "topwt": "biomass_kg_ha",
            "swfac": "water_stress_index_wspd",
            "nstres": "nitrogen_stress_index_nstd",
            "external_irrigation_action_mm": "irrigation_executed_mm",
            "external_nitrogen_action_kg_ha": "nitrogen_executed_kg_ha",
        }
    )
    auto = finalize_daily(auto, AUTO_DAILY)

    ppo = pd.read_csv(PPO_DAILY)
    ppo["scenario"] = "ppo_candidate"
    ppo["site"] = "LC"
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
    ppo = finalize_daily(ppo, PPO_DAILY)

    combined = pd.concat([baseline, auto, ppo], ignore_index=True, sort=False)
    combined = combined[combined["scenario"].isin(SCENARIO_ORDER)].copy()
    combined = combined.sort_values(["scenario", "dap"]).reset_index(drop=True)

    # 统一累计奖励：ΔGRNWT - 1.1*I - 1.58*N，避免混用各源 reward。
    combined["grain_delta_kg_ha"] = combined.groupby("scenario")["grain_yield_kg_ha"].diff().fillna(combined["grain_yield_kg_ha"])
    combined["common_step_reward"] = (
        combined["grain_delta_kg_ha"].fillna(0.0)
        - 1.1 * combined["irrigation_executed_mm"].fillna(0.0)
        - 1.58 * combined["nitrogen_executed_kg_ha"].fillna(0.0)
    )
    combined["cumulative_common_reward"] = combined.groupby("scenario")["common_step_reward"].cumsum()
    combined["cumulative_irrigation_mm"] = combined.groupby("scenario")["irrigation_executed_mm"].cumsum()
    combined["cumulative_n_kg_ha"] = combined.groupby("scenario")["nitrogen_executed_kg_ha"].cumsum()
    return combined


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, g in df.groupby("scenario"):
        gs = g.sort_values("dap")
        final = gs.iloc[-1]
        total_i = float(gs["irrigation_executed_mm"].sum())
        total_n = float(gs["nitrogen_executed_kg_ha"].sum())
        yield_kg = float(final["grain_yield_kg_ha"])
        rows.append(
            {
                "station_code": STATION,
                "year": YEAR,
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


def plot(df: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(4, 2, figsize=(15, 12), constrained_layout=True)
    ax = axes.ravel()

    weather = scenario_subset(df, "null")
    if weather.empty:
        weather = df.sort_values("dap").drop_duplicates("dap")
    ax0 = ax[0]
    ax0.bar(weather["dap"], weather["rainfall_mm"], color="#8bb8d8", alpha=0.8, label="Rain")
    ax0.set_ylabel("Rain (mm)")
    ax0b = ax0.twinx()
    ax0b.plot(weather["dap"], weather["tmax_c"], color="#d94841", label="Tmax")
    ax0b.plot(weather["dap"], weather["tmin_c"], color="#777777", linestyle="--", label="Tmin")
    ax0b.set_ylabel("Temperature (°C)")
    handles, labels = ax0.get_legend_handles_labels()
    handles2, labels2 = ax0b.get_legend_handles_labels()
    ax0.legend(handles + handles2, labels + labels2, fontsize=8, loc="upper right")
    ax0.set_title("Weather")

    ax1 = ax[1]
    for sc in SCENARIO_ORDER:
        g = scenario_subset(df, sc)
        if not g.empty:
            ax1.plot(g["dap"], g["cumulative_irrigation_mm"], color=COLORS[sc], linestyle=LINESTYLES[sc], label=LABELS[sc])
    ax1.set_title("Cumulative irrigation")
    ax1.set_ylabel("mm")
    ax1.legend(fontsize=8, loc="best")

    ax2 = ax[2]
    for sc in SCENARIO_ORDER:
        g = scenario_subset(df, sc)
        if not g.empty:
            ax2.plot(g["dap"], g["water_stress_index_wspd"], color=COLORS[sc], linestyle=LINESTYLES[sc], label=LABELS[sc])
    ax2.set_title("Water stress index")
    ax2.set_ylabel("WSPD (0=no stress)")

    ax3 = ax[3]
    for sc in SCENARIO_ORDER:
        g = scenario_subset(df, sc)
        if not g.empty:
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
    ax6.text(0.01, 0.97, "Thin companion lines are biomass; thick lines are grain.", transform=ax6.transAxes, va="top", fontsize=8)
    for sc in SCENARIO_ORDER:
        g = scenario_subset(df, sc)
        if not g.empty:
            ax6.plot(g["dap"], g["biomass_kg_ha"], color=COLORS[sc], linestyle=LINESTYLES[sc], alpha=0.35)
            ax6.plot(g["dap"], g["grain_yield_kg_ha"], color=COLORS[sc], linestyle=LINESTYLES[sc], linewidth=2.0, label=LABELS[sc])
    ax6.set_title("Grain and biomass trajectories")
    ax6.set_ylabel("kg/ha")
    ax6.set_xlabel("DAP")

    ax7 = ax[7]
    for sc in SCENARIO_ORDER:
        g = scenario_subset(df, sc)
        if not g.empty:
            ax7.plot(g["dap"], g["cumulative_common_reward"], color=COLORS[sc], linestyle=LINESTYLES[sc], label=LABELS[sc])
    ax7.set_title("Cumulative common reward")
    ax7.set_ylabel("ΔGRNWT - 1.1I - 1.58N")
    ax7.set_xlabel("DAP")
    ax7.legend(fontsize=8, loc="best")

    for axis in ax:
        axis.grid(alpha=0.25)

    fig.suptitle("LCA2021 five-scenario daily process (036_01 PPO selected ckpt100000)", x=0.02, ha="left", fontsize=14, fontweight="bold")
    paths = []
    for suffix in ["png", "svg"]:
        path = FIG_DIR / f"036_06_lca2021_five_scenario_process.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def write_record(summary: pd.DataFrame, fig_paths: list[Path]) -> None:
    lines = [
        "# 036_06 单站点五情景过程图记录",
        "",
        "## 任务边界",
        "",
        "- 不训练。",
        "- 不重跑DSSAT。",
        "- PPO daily来自036_01修复后正式重跑。",
        "- 四情景daily来自031_35/031_36基线daily源，与036_03比较口径一致。",
        "- 累计奖励面板使用统一公式 `ΔGRNWT - 1.1*irrigation - 1.58*nitrogen` 重算，不使用各源原始reward。",
        "",
        "## 五情景终值摘要",
        "",
        summary.round(4).to_markdown(index=False),
        "",
        "## 输出",
        "",
    ]
    lines.extend([f"- `{p.relative_to(ROOT)}`" for p in fig_paths])
    lines.extend(
        [
            f"- `{(TABLE_DIR / '036_06_lca2021_five_scenario_daily.csv').relative_to(ROOT)}`",
            f"- `{(TABLE_DIR / '036_06_lca2021_five_scenario_summary.csv').relative_to(ROOT)}`",
        ]
    )
    DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = load_daily()
    summary = summarize(df)
    df.to_csv(TABLE_DIR / "036_06_lca2021_five_scenario_daily.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLE_DIR / "036_06_lca2021_five_scenario_summary.csv", index=False, encoding="utf-8-sig")
    fig_paths = plot(df)
    write_record(summary, fig_paths)
    print(
        json.dumps(
            {
                "task": TASK,
                "station": STATION,
                "year": YEAR,
                "daily_rows": int(len(df)),
                "summary_rows": int(len(summary)),
                "figures": [str(p.relative_to(ROOT)) for p in fig_paths],
                "record_md": str(DOC.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
