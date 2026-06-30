from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_formal_four_scenario_015_06"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-06-30_015_06_yc2014_formal_four_scenario_record.md"

BASE_DAILY = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_yc2014_four_scenario_smoke_013_03" / "013_03_yc2014_four_scenario_daily.csv"
BASE_SUMMARY = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_yc2014_four_scenario_smoke_013_03" / "013_03_yc2014_four_scenario_summary.csv"
SEED0_SUMMARY = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_unified_dqn_checkpoint_diagnostic_015_04" / "seed0" / "015_04_yc2014_unified_dqn_checkpoint_summary.csv"
SEED0_DAILY_10K = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_unified_dqn_checkpoint_diagnostic_015_04" / "seed0" / "015_04_yc2014_checkpoint_10k_daily.csv"
SEED1_SUMMARY = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_unified_dqn_checkpoint_seed1_015_05" / "seed1" / "015_05_yc2014_unified_dqn_checkpoint_seed1_summary.csv"
SEED1_DAILY_10K = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_unified_dqn_checkpoint_seed1_015_05" / "seed1" / "015_05_yc2014_seed1_checkpoint_10k_daily.csv"

SCENARIO_LABELS = {
    "null": "Null",
    "recorded": "Recorded expert",
    "dssat_auto": "DSSAT auto",
    "dqn_best": "DQN best checkpoint",
}
SCENARIO_COLORS = {
    "null": "#404040",
    "recorded": "#D95F0E",
    "dssat_auto": "#1F78B4",
    "dqn_best": "#238B45",
}
SCENARIO_ORDER = ["null", "recorded", "dssat_auto", "dqn_best"]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def load_base() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(BASE_DAILY)
    summary = pd.read_csv(BASE_SUMMARY)
    return daily, summary


def load_best_checkpoint() -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame, dict[str, float]]:
    s0 = pd.read_csv(SEED0_SUMMARY)
    s1 = pd.read_csv(SEED1_SUMMARY)
    b0 = s0.loc[s0["final_grain_kg_ha"].idxmax()].to_dict()
    b1 = s1.loc[s1["final_grain_kg_ha"].idxmax()].to_dict()
    d0 = pd.read_csv(SEED0_DAILY_10K).copy()
    d1 = pd.read_csv(SEED1_DAILY_10K).copy()
    d0["scenario"] = "dqn_best"
    d1["scenario"] = "dqn_best"
    d0["seed"] = 0
    d1["seed"] = 1
    return d0, b0, d1, b1


def fix_base_daily(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    if "scenario" in daily.columns:
        daily["scenario"] = daily["scenario"].fillna("null")
        daily.loc[daily["scenario"].eq("recorded_shifted"), "scenario"] = "recorded"
        daily.loc[daily["scenario"].eq("dssat_auto"), "scenario"] = "dssat_auto"
    return daily


def build_reward_proxy(df: pd.DataFrame) -> pd.Series:
    reward = pd.Series(0.0, index=df.index)
    for scenario in SCENARIO_ORDER:
        mask = df["scenario"].eq(scenario)
        sub = df.loc[mask].sort_values("step")
        cum_i = sub["irrigation_mm"].fillna(0).cumsum()
        cum_n = sub["fertilizer_kg_ha"].fillna(0).cumsum()
        cum_grain = sub["grnwt"].fillna(0)
        # Simple cumulative proxy to show the tradeoff trajectory.
        reward.loc[sub.index] = cum_grain - 1.0 * cum_i - 5.0 * cum_n
    return reward


def assemble_daily() -> pd.DataFrame:
    base_daily, _ = load_base()
    base_daily = fix_base_daily(base_daily)
    base_daily = base_daily[base_daily["scenario"].isin(["null", "recorded", "dssat_auto"])].copy()
    base_daily["seed"] = np.nan
    base_daily["source"] = "base"
    d0, b0, d1, b1 = load_best_checkpoint()
    d0 = d0[d0["step"].notna()].copy()
    d1 = d1[d1["step"].notna()].copy()
    d1["source"] = "seed1_best"
    combined = pd.concat([base_daily, d1], ignore_index=True, sort=False)
    combined["reward_proxy"] = 0.0
    for scenario in SCENARIO_ORDER:
        mask = combined["scenario"].eq(scenario)
        sub = combined.loc[mask].sort_values(["source", "step"], na_position="last")
        if sub.empty:
            continue
        ss = combined.loc[mask].sort_values("step")
        combined.loc[ss.index, "reward_proxy"] = ss["grnwt"].fillna(0).cumsum() - ss["irrigation_mm"].fillna(0).cumsum() - 5.0 * ss["fertilizer_kg_ha"].fillna(0).cumsum()
    return combined


def assemble_summary() -> pd.DataFrame:
    _, base_summary = load_base()
    base_summary = base_summary.copy()
    base_summary.loc[base_summary["scenario"].isna(), "scenario"] = "null"
    base_summary.loc[base_summary["scenario"].eq("recorded_shifted"), "scenario"] = "recorded"
    base_summary.loc[base_summary["scenario"].eq("dssat_auto"), "scenario"] = "dssat_auto"
    base_summary = base_summary[base_summary["scenario"].isin(["null", "recorded", "dssat_auto"])].copy()
    base_summary["source"] = "base"
    d0, b0, d1, b1 = load_best_checkpoint()
    seed_rows = []
    for seed, best in [(0, b0), (1, b1)]:
        seed_rows.append(
            {
                "scenario": "dqn_best",
                "seed": seed,
                "checkpoint_step": int(best["checkpoint_step"]),
                "action_irrigation_total": float(best["action_irrigation_total"]),
                "action_fertilizer_total": float(best["action_fertilizer_total"]),
                "mgmt_event_irrigation_total": float(best["mgmt_event_irrigation_total"]),
                "mgmt_event_fertilizer_total": float(best["mgmt_event_fertilizer_total"]),
                "final_grain_kg_ha": float(best["final_grain_kg_ha"]),
                "final_biomass_kg_ha": float(best["final_biomass_kg_ha"]),
                "max_water_stress": float(best["max_water_stress"]),
                "max_nitrogen_stress": float(best["max_nitrogen_stress"]),
                "total_reward": float(best["total_reward"]),
                "source": f"seed{seed}",
            }
        )
    dqn_summary = pd.DataFrame(seed_rows)
    for col in ["checkpoint_step", "action_irrigation_total", "action_fertilizer_total", "mgmt_event_irrigation_total", "mgmt_event_fertilizer_total", "final_grain_kg_ha", "final_biomass_kg_ha", "max_water_stress", "max_nitrogen_stress", "total_reward"]:
        if col not in base_summary.columns:
            base_summary[col] = np.nan
    base_summary["seed"] = np.nan
    keep = ["scenario", "seed", "checkpoint_step", "action_irrigation_total", "action_fertilizer_total", "mgmt_event_irrigation_total", "mgmt_event_fertilizer_total", "final_grain_kg_ha", "final_biomass_kg_ha", "max_water_stress", "max_nitrogen_stress", "total_reward", "source"]
    return pd.concat([base_summary[keep], dqn_summary[keep]], ignore_index=True, sort=False)


def plot_four_scenario(daily: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(
        6,
        1,
        figsize=(16.2, 14.2),
        sharex=True,
        gridspec_kw={"height_ratios": [0.75, 1.0, 1.0, 1.0, 1.0, 1.1], "hspace": 0.22},
    )

    rain = daily[["dap", "rain"]].drop_duplicates("dap").sort_values("dap")
    axes[0].bar(rain["dap"], rain["rain"], width=1.0, color="#C5CAD3", edgecolor="#7A828F", linewidth=0.45)
    axes[0].set_ylabel("Rain\n(mm)")
    axes[0].set_title("YC2014 formal four-scenario comparison", loc="left", fontsize=14)

    for scenario in SCENARIO_ORDER:
        sub = daily[daily["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty:
            continue
        color = SCENARIO_COLORS[scenario]
        label = SCENARIO_LABELS[scenario]
        axes[1].plot(sub["dap"], sub["swfac"], color=color, lw=2.0, label=label)
        axes[2].plot(sub["dap"], sub["nstres"], color=color, lw=2.0, label=label)
        mg_i = sub[sub["irrigation_mm"].fillna(0) > 1e-8]
        mg_n = sub[sub["fertilizer_kg_ha"].fillna(0) > 1e-8]
        if not mg_i.empty:
            axes[3].vlines(mg_i["dap"], 0, mg_i["irrigation_mm"], colors=color, linewidth=2.1, alpha=0.95)
        if not mg_n.empty:
            axes[3].scatter(mg_n["dap"], mg_n["fertilizer_kg_ha"], marker="^", s=46, color=color, edgecolor="#FFFFFF", linewidth=0.6, zorder=4)
        axes[4].plot(sub["dap"], sub["grnwt"], color=color, lw=2.0)
        axes[4].plot(sub["dap"], sub["topwt"], color=color, lw=1.6, ls="--", alpha=0.68)
        axes[5].plot(sub["dap"], sub["reward_proxy"], color=color, lw=2.0, label=label)

    axes[1].set_ylabel("Water\nstress")
    axes[2].set_ylabel("Nitrogen\nstress")
    axes[3].set_ylabel("Mgmt\namount")
    axes[4].set_ylabel("kg/ha")
    axes[5].set_ylabel("Cum.\nproxy")
    axes[5].set_xlabel("DAP")
    axes[4].set_title("Crop outcome: solid = grain, dashed = biomass", loc="left", fontsize=10)
    axes[3].set_title("Management events: irrigation as bars; fertilization as triangles", loc="left", fontsize=10)
    axes[5].set_title("Cumulative reward proxy: grain - irrigation - 5×fertilizer", loc="left", fontsize=10)

    for ax in axes:
        ax.grid(True, axis="both", color="#E6E8F0", linewidth=0.8, alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(left=0)
    axes[1].legend(loc="upper left", ncol=2, frameon=False, fontsize=9)
    mgmt_handles = [plt.Line2D([0], [0], color=SCENARIO_COLORS[s], lw=2, label=SCENARIO_LABELS[s]) for s in SCENARIO_ORDER]
    axes[3].legend(handles=mgmt_handles, loc="upper left", ncol=2, frameon=False, fontsize=9)
    axes[5].legend(handles=mgmt_handles, loc="upper left", ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(top=0.95, bottom=0.06, left=0.07, right=0.98, hspace=0.22)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_record(summary: pd.DataFrame, out_daily: Path, out_summary: Path, fig_path: Path) -> None:
    seed_view = summary[summary["scenario"].eq("dqn_best")].copy()
    seed0 = seed_view[seed_view["seed"].eq(0)].iloc[0]
    seed1 = seed_view[seed_view["seed"].eq(1)].iloc[0]
    lines = [
        "# 015_06 YC2014 正式四情景对照与 best checkpoint 汇总记录",
        "",
        "## 结论",
        "",
        "YC2014 已形成可以汇报的正式案例：null、recorded expert、DSSAT auto 和 DQN best checkpoint 的比较结果都已整理；DQN 必须采用 checkpoint selection，而不是 final model。",
        "",
        "## 输出文件",
        "",
        f"- 主图：`{fig_path.relative_to(PROJECT_ROOT)}`",
        f"- 日值总表：`{out_daily.relative_to(PROJECT_ROOT)}`",
        f"- 汇总表：`{out_summary.relative_to(PROJECT_ROOT)}`",
        "",
        "## seed best checkpoint",
        "",
        f"- seed0 best checkpoint={int(seed0['checkpoint_step'])}, I={seed0['action_irrigation_total']:.1f}, N={seed0['action_fertilizer_total']:.1f}, grain={seed0['final_grain_kg_ha']:.1f}",
        f"- seed1 best checkpoint={int(seed1['checkpoint_step'])}, I={seed1['action_irrigation_total']:.1f}, N={seed1['action_fertilizer_total']:.1f}, grain={seed1['final_grain_kg_ha']:.1f}",
        "",
        "## 下一步",
        "",
        "如果导师认可这张正式图，就把同一协议迁移到 FQ2016 / HLA2010 / 其他候选站点年份。",
    ]
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = assemble_daily()
    summary = assemble_summary()
    run_dir = OUT_DIR / "seed0_seed1_best"
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    daily_path = run_dir / "015_06_yc2014_formal_four_scenario_daily.csv"
    summary_path = run_dir / "015_06_yc2014_formal_four_scenario_summary.csv"
    fig_path = fig_dir / "yc2014_formal_four_scenario.png"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    plot_four_scenario(daily, fig_path)
    write_record(summary, daily_path, summary_path, fig_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
