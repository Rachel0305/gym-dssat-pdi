from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_weather


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_station_level3_true_model_transfer_016_04"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_cross_year_transfer_success_plots_016_11"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-05_016_11_yc2014_cross_year_transfer_success_plots_record.md"

DAILY_CSV = SRC_DIR / "yc2014_true_model_transfer_daily.csv"
SUMMARY_CSV = SRC_DIR / "yc2014_true_model_transfer_summary.csv"
SUCCESS_CSV = SRC_DIR / "yc2014_transfer_success_by_year.csv"

YEARS = [2006, 2009, 2015, 2018]

SCENARIO_ORDER = ["null", "recorded_shifted", "dssat_auto", "dqn_transfer"]
SCENARIO_LABELS = {
    "null": "Null",
    "recorded_shifted": "Recorded expert",
    "dssat_auto": "DSSAT auto",
    "dqn_transfer": "DQN transfer best",
}
SCENARIO_COLORS = {
    "null": "#3F3F3F",
    "recorded_shifted": "#C73E3A",
    "dssat_auto": "#B8860B",
    "dqn_transfer": "#2E8B57",
}
SCENARIO_LINESTYLES = {
    "null": "-",
    "recorded_shifted": "--",
    "dssat_auto": "-",
    "dqn_transfer": "-",
}

WATER_COST = 1.0
N_COST = 5.0


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 9,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(DAILY_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    success = pd.read_csv(SUCCESS_CSV)
    daily["scenario"] = daily["scenario"].fillna("null")
    summary["scenario"] = summary["scenario"].fillna("null")
    success["scenario"] = success["scenario"].fillna("null")
    return daily, summary, success


def pick_best_transfer(success: pd.DataFrame, year: int) -> pd.Series:
    sub = success[(success["year"] == year) & (success["scenario"].astype(str).str.startswith("transfer_"))].copy()
    if sub.empty:
        raise ValueError(f"No transfer rows for year {year}")
    sub = sub.sort_values(
        ["success_flag", "final_grain_kg_ha", "irrigation_total", "fertilizer_total"],
        ascending=[False, False, True, True],
    )
    return sub.iloc[0]


def add_rain(daily_year: pd.DataFrame, year: int) -> pd.DataFrame:
    daily_year = daily_year.copy()
    weather = parse_weather("YC", year)[["doy", "rain"]].rename(columns={"rain": "rain_weather"})
    # For these YC shifted templates, sowing day is fixed at DOY 153.
    # The cached transfer daily table often lacks explicit DOY, so rebuild
    # a stable DAP->DOY map directly from the known calendar alignment.
    doy_map = daily_year[["dap"]].drop_duplicates().sort_values("dap").copy()
    doy_map["doy"] = 153 + doy_map["dap"].round().astype(int)
    rain_map = doy_map.merge(weather, on="doy", how="left")
    rain_map["rain_weather"] = rain_map["rain_weather"].fillna(0.0)
    daily_year = daily_year.drop(columns=["rain"], errors="ignore")
    daily_year = daily_year.merge(rain_map[["dap", "rain_weather"]], on="dap", how="left")
    daily_year["rain"] = daily_year["rain_weather"].fillna(0.0)
    daily_year = daily_year.drop(columns=["rain_weather"])
    return daily_year


def add_reward_proxy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["reward_proxy"] = 0.0
    for scenario in df["scenario"].dropna().unique():
        sub = df[df["scenario"] == scenario].sort_values("dap").copy()
        grain_delta = sub["grnwt"].fillna(0).diff().fillna(sub["grnwt"].fillna(0)).clip(lower=0)
        cost = WATER_COST * sub["irrigation_mm"].fillna(0) + N_COST * sub["fertilizer_kg_ha"].fillna(0)
        df.loc[sub.index, "reward_proxy"] = (grain_delta - cost).cumsum()
    return df


def event_table_from_daily(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df[df["irrigation_mm"].fillna(0) > 1e-8].iterrows():
        rows.append({"scenario": row["scenario"], "dap": row["dap"], "operation": "Irrigation", "amount": row["irrigation_mm"]})
    for _, row in df[df["fertilizer_kg_ha"].fillna(0) > 1e-8].iterrows():
        rows.append({"scenario": row["scenario"], "dap": row["dap"], "operation": "Fertilizer", "amount": row["fertilizer_kg_ha"]})
    return pd.DataFrame(rows)


def line_handles():
    return [
        Line2D([0], [0], color=SCENARIO_COLORS[s], lw=2.2, ls=SCENARIO_LINESTYLES[s], label=SCENARIO_LABELS[s])
        for s in SCENARIO_ORDER
    ]


def plot_year(year: int, daily_year: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(
        6,
        1,
        figsize=(14.8, 12.8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.7, 1.0, 1.0, 1.0, 1.15, 1.0], "hspace": 0.22},
    )

    rain = daily_year[["dap", "rain"]].drop_duplicates("dap").sort_values("dap")
    axes[0].bar(rain["dap"], rain["rain"], color="#C9CED8", edgecolor="#AEB6C2", linewidth=0.4, width=0.9)
    axes[0].set_ylabel("Rain\n(mm)")
    axes[0].set_title(f"YC {year} cross-year transfer four-scenario process plot", loc="left", fontsize=15, fontweight="bold", pad=8)
    axes[0].legend(handles=[Patch(facecolor="#C9CED8", edgecolor="#AEB6C2", label="Rainfall")], loc="upper left")

    for scenario in SCENARIO_ORDER:
        sub = daily_year[daily_year["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty:
            continue
        color = SCENARIO_COLORS[scenario]
        ls = SCENARIO_LINESTYLES[scenario]
        axes[1].plot(sub["dap"], sub["swfac"], color=color, ls=ls, lw=2.2)
        axes[2].plot(sub["dap"], sub["nstres"], color=color, ls=ls, lw=2.2)
        axes[4].plot(sub["dap"], sub["grnwt"], color=color, ls=ls, lw=2.2)
        axes[4].plot(sub["dap"], sub["topwt"], color=color, ls=":", lw=1.8, alpha=0.95)
        axes[5].plot(sub["dap"], sub["reward_proxy"], color=color, ls=ls, lw=2.2)

        mg_i = sub[sub["irrigation_mm"].fillna(0) > 1e-8]
        mg_n = sub[sub["fertilizer_kg_ha"].fillna(0) > 1e-8]
        if not mg_i.empty:
            axes[3].vlines(mg_i["dap"], 0, mg_i["irrigation_mm"], colors=color, linestyles=ls, linewidth=2.4, alpha=0.95)
        if not mg_n.empty:
            axes[3].scatter(mg_n["dap"], mg_n["fertilizer_kg_ha"], marker="^", s=65, color=color, edgecolor="white", linewidth=0.7, zorder=5)

    handles = line_handles()
    axes[1].set_ylabel("Water\nstress")
    axes[1].set_title("Water stress index by scenario", loc="left", fontsize=10)
    axes[1].legend(handles=handles, loc="upper left", ncol=2, fontsize=9)
    axes[2].set_ylabel("Nitrogen\nstress")
    axes[2].set_title("Nitrogen stress index by scenario", loc="left", fontsize=10)
    axes[3].set_ylabel("Mgmt\namount")
    axes[3].set_title("Management events: irrigation as vertical lines; fertilization as triangle markers", loc="left", fontsize=10)
    axes[3].legend(handles=handles, loc="upper left", ncol=2, fontsize=9)
    axes[4].set_ylabel("kg/ha")
    axes[4].set_title("Crop outcome: solid = grain yield, dotted = aboveground biomass", loc="left", fontsize=10)
    axes[5].set_ylabel("Cum.\nreward")
    axes[5].set_title("Cumulative reward proxy: grain increment - 1×irrigation - 5×fertilizer", loc="left", fontsize=10)
    axes[5].legend(handles=handles, loc="upper left", ncol=2, fontsize=9)
    axes[5].set_xlabel("DAP")

    max_dap = float(daily_year["dap"].max()) if not daily_year.empty else 120
    for ax in axes:
        ax.grid(True, color="#E6E8F0", linewidth=0.8, alpha=0.9)
        ax.set_xlim(0, max_dap + 2)
    axes[1].set_ylim(bottom=-0.02)
    axes[2].set_ylim(bottom=-0.02)
    axes[3].set_ylim(bottom=-10)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.97, bottom=0.06, left=0.07, right=0.985, hspace=0.22)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_record(selected_summary: pd.DataFrame) -> None:
    lines = [
        "# 016_11 YC2014 跨年份迁移成功案例四情景图记录",
        "",
        "## 目的",
        "",
        "基于 016_04 已经完成的 YC2014 -> 其他年份真实模型迁移结果，不新增训练，只挑选最有代表性的成功年份，输出逐年四情景过程图与对应日值表。",
        "",
        "## 选中的年份",
        "",
        selected_summary.to_markdown(index=False),
        "",
        "## 说明",
        "",
        "- 迁移情景使用每年综合表现最好的 transfer 记录；",
        "- 图表包含：降雨、水分胁迫、氮胁迫、管理措施、两种产量、累积奖励；",
        "- 日值表与图一一对应，便于后续汇报与追溯。",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "daily_tables").mkdir(parents=True, exist_ok=True)

    daily, summary, success = load_tables()
    chosen_rows = []

    for year in YEARS:
        best = pick_best_transfer(success, year)
        chosen_rows.append(best)

        transfer_scenario = best["scenario"]
        base = daily[(daily["year"] == year) & (daily["scenario"].isin(["null", "recorded_shifted", "dssat_auto"]))].copy()
        dqn = daily[
            (daily["year"] == year)
            & (daily["scenario"] == transfer_scenario)
            & (daily["train_seed"].fillna(-1) == float(best["train_seed"]))
            & (daily["checkpoint_step"].fillna(-1) == float(best["checkpoint_step"]))
            & (daily["selection"].fillna("") == str(best["selection"]))
        ].copy()
        dqn["scenario"] = "dqn_transfer"

        year_daily = pd.concat([base, dqn], ignore_index=True, sort=False)
        year_daily["scenario"] = year_daily["scenario"].fillna("null")
        year_daily = add_rain(year_daily, year)
        year_daily = add_reward_proxy(year_daily)

        year_daily_path = OUT_DIR / "daily_tables" / f"yc_{year}_four_scenario_daily.csv"
        year_daily.to_csv(year_daily_path, index=False, encoding="utf-8-sig")

        fig_path = OUT_DIR / "figures" / f"yc_{year}_four_scenario_transfer.png"
        plot_year(year, year_daily, fig_path)

    selected_summary = pd.DataFrame(chosen_rows)
    selected_summary.to_csv(OUT_DIR / "summary_selected_years.csv", index=False, encoding="utf-8-sig")
    write_record(selected_summary)
    print(selected_summary[["year", "train_seed", "checkpoint_step", "selection", "final_grain_kg_ha", "yield_diff_vs_auto"]].to_string(index=False))


if __name__ == "__main__":
    main()
