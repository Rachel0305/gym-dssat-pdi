from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_fq_all_year_screen_and_dqn_transfer_014_01 import parse_weather


YEAR = 2019
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "fq2019_process_plot_and_seed1_stability_017_06"
FIG_DIR = OUT_DIR / "figures"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-06_017_06_fq2019_process_plot_seed1_record.md"

BASE_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "fq2016_seed1_checkpoint30000_all_year_transfer_017_03"
DQN_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "fq2019_baseline_relative_dqn_train_transfer_017_05" / "seed0_50000steps"
DQN_CKPT = 10000

SCENARIO_ORDER = ["null_zero", "recorded_shifted", "dssat_auto", "fq2019_dqn_seed0_ckpt10000"]
SCENARIO_LABELS = {
    "null_zero": "Null",
    "recorded_shifted": "Recorded shifted",
    "dssat_auto": "DSSAT auto",
    "fq2019_dqn_seed0_ckpt10000": "FQ2019 DQN seed0 ckpt10000",
}
SCENARIO_COLORS = {
    "null_zero": "#333333",
    "recorded_shifted": "#C73E3A",
    "dssat_auto": "#B8860B",
    "fq2019_dqn_seed0_ckpt10000": "#2E8B57",
}
SCENARIO_LINESTYLES = {
    "null_zero": "-",
    "recorded_shifted": "--",
    "dssat_auto": "-",
    "fq2019_dqn_seed0_ckpt10000": "-",
}
WATER_COST = 1.0
NITROGEN_COST = 5.0


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def add_rain(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["doy"] = pd.to_numeric(daily["doy"], errors="coerce")
    daily["dap"] = pd.to_numeric(daily["dap"], errors="coerce")
    missing = daily["doy"].isna() & daily["dap"].notna()
    daily.loc[missing, "doy"] = 162 + daily.loc[missing, "dap"].round().astype(int)
    weather = parse_weather(YEAR).rename(columns={"rain": "rain_weather"})
    daily = daily.drop(columns=["rain"], errors="ignore").merge(weather[["doy", "rain_weather"]], on="doy", how="left")
    daily["rain"] = daily["rain_weather"].fillna(0.0)
    return daily.drop(columns=["rain_weather"])


def add_reward_proxy(daily: pd.DataFrame, null_yield: float) -> pd.DataFrame:
    daily = daily.copy()
    daily["reward_proxy"] = 0.0
    daily["cumulative_reward_proxy"] = 0.0
    for scenario, sub in daily.groupby("scenario"):
        sub = sub.sort_values("dap")
        cost = WATER_COST * sub["irrigation_mm"].fillna(0) + NITROGEN_COST * sub["fertilizer_kg_ha"].fillna(0)
        terminal = pd.Series(0.0, index=sub.index)
        if not sub.empty:
            terminal.loc[sub.index[-1]] = max(0.0, float(sub.iloc[-1]["grnwt"]) - null_yield)
        proxy = terminal - cost
        daily.loc[sub.index, "reward_proxy"] = proxy
        daily.loc[sub.index, "cumulative_reward_proxy"] = proxy.cumsum()
    return daily


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_daily = pd.read_csv(BASE_ROOT / "fq2016_seed1_ckpt30000_transfer_daily.csv")
    base_events = pd.read_csv(BASE_ROOT / "fq2016_seed1_ckpt30000_transfer_events.csv")
    base_summary = pd.read_csv(BASE_ROOT / "fq2016_seed1_ckpt30000_transfer_summary.csv")
    base_daily = base_daily[base_daily["year"].eq(YEAR) & base_daily["scenario"].isin(["null_zero", "recorded_shifted", "dssat_auto"])].copy()
    base_events = base_events[base_events["year"].eq(YEAR) & base_events["scenario"].isin(["recorded_shifted", "dssat_auto"])].copy()
    base_summary = base_summary[base_summary["year"].eq(YEAR) & base_summary["scenario"].isin(["null_zero", "recorded_shifted", "dssat_auto"])].copy()

    dqn_daily = pd.read_csv(DQN_ROOT / "fq2019_checkpoint_eval_daily.csv")
    dqn_daily = dqn_daily[dqn_daily["checkpoint_step"].eq(DQN_CKPT)].copy()
    dqn_daily["scenario"] = "fq2019_dqn_seed0_ckpt10000"

    dqn_events = pd.read_csv(DQN_ROOT / "fq2019_checkpoint_eval_events.csv")
    dqn_events = dqn_events[dqn_events["checkpoint_step"].eq(DQN_CKPT)].copy()
    dqn_events["scenario"] = "fq2019_dqn_seed0_ckpt10000"

    dqn_summary = pd.read_csv(DQN_ROOT / "fq2019_checkpoint_eval_summary.csv")
    dqn_summary = dqn_summary[dqn_summary["checkpoint_step"].eq(DQN_CKPT)].copy()
    dqn_summary["site"] = "FQ"
    dqn_summary["station"] = "Fengqiu"
    dqn_summary["scenario"] = "fq2019_dqn_seed0_ckpt10000"
    dqn_summary = dqn_summary.rename(
        columns={
            "action_irrigation_total": "irrigation_total",
            "action_fertilizer_total": "fertilizer_total",
            "final_grain_kg_ha": "final_grain_kg_ha",
            "final_biomass_kg_ha": "final_biomass_kg_ha",
        }
    )

    daily = pd.concat([base_daily, dqn_daily], ignore_index=True, sort=False)
    events = pd.concat([base_events, dqn_events], ignore_index=True, sort=False)
    summary = pd.concat([base_summary, dqn_summary], ignore_index=True, sort=False)
    daily = add_rain(daily)
    null_yield = float(summary.loc[summary["scenario"].eq("null_zero"), "final_grain_kg_ha"].iloc[0])
    daily = add_reward_proxy(daily, null_yield)
    return daily, events, summary


def make_summary_clean(summary: pd.DataFrame, daily: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIO_ORDER:
        sub = daily[daily["scenario"].eq(scenario)]
        ev = events[events["scenario"].eq(scenario)] if not events.empty else pd.DataFrame()
        rows.append(
            {
                "year": YEAR,
                "scenario": scenario,
                "final_grain_kg_ha": float(sub["grnwt"].dropna().iloc[-1]) if not sub.empty else np.nan,
                "final_biomass_kg_ha": float(sub["topwt"].dropna().iloc[-1]) if not sub.empty else np.nan,
                "irrigation_total": float(ev.loc[ev["unit"].eq("mm"), "amount"].sum()) if not ev.empty else 0.0,
                "fertilizer_total": float(ev.loc[ev["unit"].astype(str).str.contains("kg", na=False), "amount"].sum()) if not ev.empty else 0.0,
                "max_water_stress": float(sub["swfac"].max()) if not sub.empty else np.nan,
                "max_nitrogen_stress": float(sub["nstres"].max()) if not sub.empty else np.nan,
                "total_reward_proxy": float(sub["reward_proxy"].sum()) if not sub.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=SCENARIO_COLORS[s], lw=2.2, ls=SCENARIO_LINESTYLES[s], label=SCENARIO_LABELS[s])
        for s in SCENARIO_ORDER
    ]


def plot_process(daily: pd.DataFrame, events: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        6,
        1,
        figsize=(14.8, 12.8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.75, 1.0, 1.0, 1.0, 1.15, 1.0], "hspace": 0.24},
    )
    rain = daily[["dap", "rain"]].drop_duplicates("dap").sort_values("dap")
    axes[0].bar(rain["dap"], rain["rain"], color="#C9CED8", edgecolor="#AEB6C2", linewidth=0.4, width=0.9)
    axes[0].set_ylabel("Rain\n(mm)")
    axes[0].set_title("FQ2019 four-scenario process plot with DQN seed0 checkpoint10000", loc="left", fontsize=14, fontweight="bold")
    axes[0].legend(handles=[Patch(facecolor="#C9CED8", edgecolor="#AEB6C2", label="Rainfall")], loc="upper left")

    for scenario in SCENARIO_ORDER:
        sub = daily[daily["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty:
            continue
        color = SCENARIO_COLORS[scenario]
        ls = SCENARIO_LINESTYLES[scenario]
        axes[1].plot(sub["dap"], sub["swfac"], color=color, ls=ls, lw=2.25)
        axes[2].plot(sub["dap"], sub["nstres"], color=color, ls=ls, lw=2.25)
        axes[4].plot(sub["dap"], sub["grnwt"], color=color, ls=ls, lw=2.25)
        axes[4].plot(sub["dap"], sub["topwt"], color=color, ls=":", lw=1.9, alpha=0.95)
        axes[5].plot(sub["dap"], sub["cumulative_reward_proxy"], color=color, ls=ls, lw=2.25)
        ev_i = events[(events["scenario"].eq(scenario)) & (events["unit"].eq("mm"))]
        ev_n = events[(events["scenario"].eq(scenario)) & (events["unit"].astype(str).str.contains("kg", na=False))]
        if not ev_i.empty:
            axes[3].vlines(ev_i["dap"], 0, ev_i["amount"], colors=color, linestyles=ls, linewidth=2.6, alpha=0.95)
        if not ev_n.empty:
            axes[3].scatter(ev_n["dap"], ev_n["amount"], marker="^", s=70, color=color, edgecolor="white", linewidth=0.7, zorder=5)

    handles = legend_handles()
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
    axes[5].set_title("Cumulative reward proxy: terminal max(0, GWAD-null) - 1*irrigation - 5*fertilizer", loc="left", fontsize=10)
    axes[5].legend(handles=handles, loc="upper left", ncol=2, fontsize=9)
    axes[5].set_xlabel("DAP")
    max_dap = float(daily["dap"].max()) if not daily.empty else 120
    for ax in axes:
        ax.grid(True, color="#E6E8F0", linewidth=0.8, alpha=0.9)
        ax.set_xlim(0, max_dap + 2)
    axes[1].set_ylim(bottom=-0.02)
    axes[2].set_ylim(bottom=-0.02)
    axes[3].set_ylim(bottom=-10)
    stem = FIG_DIR / "fq2019_seed0_ckpt10000_four_scenario_process"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_record(summary_clean: pd.DataFrame) -> None:
    def md(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, (float, np.floating)):
                    vals.append("" if pd.isna(val) else f"{float(val):.3f}")
                else:
                    vals.append("" if pd.isna(val) else str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# 017_06 FQ2019 过程图与 seed1 稳定性验证记录",
        "",
        "## 已完成：FQ2019 seed0 checkpoint10000 四情景过程图",
        "",
        md(summary_clean),
        "",
        "## 输出",
        "",
        f"- 日值表：`{(OUT_DIR / 'fq2019_seed0_ckpt10000_four_scenario_daily.csv').relative_to(PROJECT_ROOT)}`",
        f"- 事件表：`{(OUT_DIR / 'fq2019_seed0_ckpt10000_four_scenario_events.csv').relative_to(PROJECT_ROOT)}`",
        f"- summary：`{(OUT_DIR / 'fq2019_seed0_ckpt10000_four_scenario_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- 图：`{(FIG_DIR / 'fq2019_seed0_ckpt10000_four_scenario_process.png').relative_to(PROJECT_ROOT)}`",
        "",
        "## 待补充",
        "",
        "- FQ2019 seed1 50K checkpoint 稳定性结果。",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    daily, events, summary = load_tables()
    summary_clean = make_summary_clean(summary, daily, events)
    daily.to_csv(OUT_DIR / "fq2019_seed0_ckpt10000_four_scenario_daily.csv", index=False, encoding="utf-8-sig")
    events.to_csv(OUT_DIR / "fq2019_seed0_ckpt10000_four_scenario_events.csv", index=False, encoding="utf-8-sig")
    summary_clean.to_csv(OUT_DIR / "fq2019_seed0_ckpt10000_four_scenario_summary.csv", index=False, encoding="utf-8-sig")
    plot_process(daily, events)
    write_record(summary_clean)
    print(summary_clean.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

