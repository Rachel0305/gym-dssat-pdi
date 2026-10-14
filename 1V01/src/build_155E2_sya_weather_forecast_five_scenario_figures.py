"""Build SY E2 5K weather-forecast five-scenario strategy figures.

The E2 policy is taken from the existing seed-0 5K ETCP replay.  Baselines
come from the frozen SY originIC four-baseline package.  This keeps the plots
diagnostic and avoids retraining or mixing a forecast PPO daily trace with a
different checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from build_sya_configured_five_scenario_daily_plots_046_06 import (
    add_unified_reward,
    daily_from_snapshot,
)


BASELINE = ROOT / "benchmark_results/046_03_sya_originIC_four_baselines/evaluation/046_03_baseline_summary.csv"
E2_REPLAY = ROOT / "benchmark_results/151E2N_wp_et_5k_replay/151_wp_et_replay_by_policy_year.csv"
E2_REPLAY_ROOT = ROOT / "benchmark_results/151E2N_wp_et_5k_replay/replays/e2_s0/snapshots/SYA"
OUT = ROOT / "benchmark_results/155E2_sya_weather_forecast_five_scenario_figures"
DOC = ROOT / "docs/155E2_sya_weather_forecast_five_scenario_figures_record.md"
YEARS = list(range(2014, 2024))
SCENARIOS = ["null", "recorded_farmer_template", "dssat_auto", "official_extension_expert", "rl_candidate"]
LABELS = {
    "null": "Null",
    "recorded_farmer_template": "Recorded template",
    "dssat_auto": "DSSAT auto irrigation",
    "official_extension_expert": "Official expert",
    "rl_candidate": "E2 weather PPO",
}
COLORS = {
    "null": "#444444",
    "recorded_farmer_template": "#C44E52",
    "dssat_auto": "#D8A305",
    "official_extension_expert": "#7861B2",
    "rl_candidate": "#238B45",
}
STYLES = {
    "null": "-",
    "recorded_farmer_template": "--",
    "dssat_auto": "-.",
    "official_extension_expert": ":",
    "rl_candidate": "-",
}
METRICS = [
    ("grain_yield_kg_ha", "Grain yield (kg/ha)"),
    ("WP_ET_kg_m3", "WP_ET (kg/m³)"),
    ("PFP_N_kg_kg", "PFP-N (kg/kg)"),
]
RESOURCES = [
    ("actual_irrigation_mm", "Total irrigation (mm)"),
    ("actual_nitrogen_kg_ha", "Total nitrogen (kg/ha)"),
]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_summary() -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE, keep_default_na=False)
    replay = pd.read_csv(E2_REPLAY, keep_default_na=False)
    replay = replay[replay["policy"].eq("e2_s0")].copy()
    replay["year"] = _numeric(replay["year"]).astype(int)
    replay = replay[replay["year"].isin(YEARS)]
    if len(replay) != len(YEARS):
        raise RuntimeError(f"Expected one E2 seed0 replay row per validation year, got {len(replay)}")

    rows: list[dict[str, object]] = []
    baseline = baseline[baseline["scenario"].isin(SCENARIOS[:-1])].copy()
    for _, row in baseline.iterrows():
        rows.append(
            {
                "year": int(row["year"]),
                "scenario": str(row["scenario"]),
                "grain_yield_kg_ha": float(row["grain_yield_kg_ha"]),
                "actual_irrigation_mm": float(row["actual_irrigation_mm"]),
                "actual_nitrogen_kg_ha": float(row["actual_nitrogen_kg_ha"]),
                "WP_ET_kg_m3": float(row["WP_ET_kg_m3"]) if str(row["WP_ET_kg_m3"]).strip() else np.nan,
                "PFP_N_kg_kg": float(row["PFP_N_kg_kg"]) if str(row["PFP_N_kg_kg"]).strip() else np.nan,
                "source": "046_03_sya_originIC_four_baselines",
            }
        )
    for _, row in replay.iterrows():
        rows.append(
            {
                "year": int(row["year"]),
                "scenario": "rl_candidate",
                "grain_yield_kg_ha": float(row["final_grnwt"]),
                "actual_irrigation_mm": float(row["total_irrigation"]),
                "actual_nitrogen_kg_ha": float(row["total_n"]),
                "WP_ET_kg_m3": float(row["WP_ET_kg_m3"]),
                "PFP_N_kg_kg": float(row["PFP_N_kg_kg"]),
                "etcp_mm": float(row["etcp_mm"]),
                "action_sequence": str(row["action_sequence"]),
                "source": "151E2N_wp_et_5k_replay_e2_s0",
            }
        )
    result = pd.DataFrame(rows)
    result["year"] = _numeric(result["year"]).astype(int)
    if result.duplicated(["year", "scenario"]).any() or len(result) != len(YEARS) * len(SCENARIOS):
        raise RuntimeError("Five-scenario summary is not exactly 10 years x 5 scenarios")
    return result.sort_values(["year", "scenario"]).reset_index(drop=True)


def plot_year(daily: pd.DataFrame, year: int, path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    weather = daily[daily["scenario"].eq("null")].sort_values("dap")
    axes[0, 0].bar(weather["dap"], weather["rainfall_mm"], color="#80A9C7", alpha=0.85, label="Rain")
    temp = axes[0, 0].twinx()
    temp.plot(weather["dap"], weather["tmax_c"], color="#C23B32", lw=1.25, label="Tmax")
    temp.plot(weather["dap"], weather["tmin_c"], color="#666666", lw=1.1, ls="--", label="Tmin")
    axes[0, 0].set_title("Weather"); axes[0, 0].set_ylabel("Rain (mm)"); temp.set_ylabel("Temperature (°C)")
    for scenario in SCENARIOS:
        sub = daily[daily["scenario"].eq(scenario)].sort_values("dap")
        color, style, label = COLORS[scenario], STYLES[scenario], LABELS[scenario]
        axes[0, 1].plot(sub["dap"], sub["soil_water_mm"], color=color, ls=style, lw=1.35, label=label)
        axes[1, 0].plot(sub["dap"], sub["water_stress_index_wspd"], color=color, ls=style, lw=1.35, label=label)
        axes[1, 1].plot(sub["dap"], sub["nitrogen_stress_index_nstd"], color=color, ls=style, lw=1.35, label=label)
        positive_i = sub[sub["irrigation_executed_mm"].gt(0)]
        positive_n = sub[sub["nitrogen_executed_kg_ha"].gt(0)]
        if not positive_i.empty:
            axes[2, 0].vlines(positive_i["dap"], 0, positive_i["irrigation_executed_mm"], color=color, lw=1.35, label=label)
            axes[2, 0].scatter(positive_i["dap"], positive_i["irrigation_executed_mm"], color=color, s=18)
        if not positive_n.empty:
            axes[2, 1].vlines(positive_n["dap"], 0, positive_n["nitrogen_executed_kg_ha"], color=color, lw=1.35, label=label)
            axes[2, 1].scatter(positive_n["dap"], positive_n["nitrogen_executed_kg_ha"], color=color, s=18)
        axes[3, 0].plot(sub["dap"], sub["grain_yield_kg_ha"], color=color, ls=style, lw=1.45, label=f"{label} grain")
        axes[3, 0].plot(sub["dap"], sub["biomass_kg_ha"], color=color, ls=style, lw=0.85, alpha=0.4)
        axes[3, 1].plot(sub["dap"], sub["unified_cumulative_reward"], color=color, ls=style, lw=1.35, label=label)
    titles = [
        (axes[0, 1], "Soil water", "SWTD (mm)"),
        (axes[1, 0], "Water stress index", "WSPD"),
        (axes[1, 1], "Nitrogen stress index", "NSTD"),
        (axes[2, 0], "Irrigation events", "mm/event"),
        (axes[2, 1], "Nitrogen application events", "kg/ha/event"),
        (axes[3, 0], "Grain and biomass trajectories", "kg/ha"),
        (axes[3, 1], "Unified cumulative reward", "common units"),
    ]
    for axis, title, ylabel in titles:
        axis.set_title(title); axis.set_ylabel(ylabel); axis.grid(alpha=0.2)
    axes[0, 1].legend(fontsize=7, ncol=2); axes[2, 0].legend(fontsize=7, ncol=2); axes[2, 1].legend(fontsize=7, ncol=2); axes[3, 1].legend(fontsize=7, ncol=2)
    axes[3, 0].set_xlabel("DAP"); axes[3, 1].set_xlabel("DAP")
    fig.suptitle(f"SY{year} E2 weather-forecast five-scenario daily process", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def draw_grouped(summary: pd.DataFrame, fields: list[tuple[str, str]], path: Path, title: str) -> None:
    x = np.arange(len(YEARS))
    width = 0.15
    fig, axes = plt.subplots(len(fields), 1, figsize=(16, 4.2 * len(fields)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, (field, ylabel) in zip(axes, fields):
        for idx, scenario in enumerate(SCENARIOS):
            subset = summary[summary["scenario"].eq(scenario)].set_index("year")
            values = [float(subset[field].get(year, np.nan)) for year in YEARS]
            axis.bar(x + (idx - 2) * width, values, width, label=LABELS[scenario], color=COLORS[scenario])
        axis.set_ylabel(ylabel); axis.grid(axis="y", alpha=0.25)
        axis.legend(ncol=3, fontsize=8)
    axes[-1].set_xticks(x); axes[-1].set_xticklabels(YEARS); axes[-1].set_xlabel("Validation year")
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(path, dpi=220); plt.close(fig)


def draw_aggregate(summary: pd.DataFrame, fields: list[tuple[str, str]], path: Path, title: str, reducer: str) -> None:
    aggregate = summary.groupby("scenario", as_index=False)[[field for field, _ in fields]].agg(reducer)
    x = np.arange(len(SCENARIOS)); width = 0.72 / len(fields)
    fig, axes = plt.subplots(len(fields), 1, figsize=(12, 4.0 * len(fields)))
    axes = np.atleast_1d(axes)
    for axis, (field, ylabel) in zip(axes, fields):
        values = [float(aggregate.loc[aggregate["scenario"].eq(scenario), field].iloc[0]) for scenario in SCENARIOS]
        axis.bar(x, values, color=[COLORS[s] for s in SCENARIOS], width=0.62)
        axis.set_ylabel(ylabel); axis.set_xticks(x); axis.set_xticklabels([LABELS[s] for s in SCENARIOS], rotation=18, ha="right"); axis.grid(axis="y", alpha=0.25)
        for pos, value in zip(x, values):
            if np.isfinite(value): axis.text(pos, value, f"{value:.2f}" if "WP" in ylabel or "PFP" in ylabel else f"{value:.0f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold"); fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(path, dpi=220); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    summary = load_summary()
    summary.to_csv(OUT / "tables/155E2_five_scenario_summary.csv", index=False, encoding="utf-8-sig")

    daily_frames: list[pd.DataFrame] = []
    for year in YEARS:
        snapshot_map = {
            "null": ROOT / f"benchmark_results/046_03_sya_originIC_four_baselines/snapshots/SYA/{year}/null",
            "recorded_farmer_template": ROOT / f"benchmark_results/046_03_sya_originIC_four_baselines/snapshots/SYA/{year}/recorded_farmer_template",
            "dssat_auto": ROOT / f"benchmark_results/046_03_sya_originIC_four_baselines/snapshots/SYA/{year}/dssat_auto",
            "official_extension_expert": ROOT / f"benchmark_results/046_03_sya_originIC_four_baselines/snapshots/SYA/{year}/official_extension_expert",
            "rl_candidate": E2_REPLAY_ROOT / str(year) / "ckpt5000",
        }
        missing = [str(path) for path in snapshot_map.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"SY{year} missing snapshot(s): {missing}")
        daily = add_unified_reward(pd.concat([daily_from_snapshot(path, year, scenario) for scenario, path in snapshot_map.items()], ignore_index=True))
        daily_frames.append(daily)
        daily.to_csv(OUT / f"tables/155E2_sy{year}_five_scenario_daily.csv", index=False, encoding="utf-8-sig")
        plot_year(daily, year, OUT / f"figures/155E2_sy{year}_five_scenario_daily.png")
    all_daily = pd.concat(daily_frames, ignore_index=True)
    all_daily.to_csv(OUT / "tables/155E2_all_years_five_scenario_daily.csv", index=False, encoding="utf-8-sig")

    draw_grouped(summary, RESOURCES, OUT / "figures/155E2_five_scenario_management.png", "SY E2 forecast: yearly total irrigation and nitrogen")
    draw_grouped(summary, METRICS, OUT / "figures/155E2_five_scenario_metrics.png", "SY E2 forecast: yearly yield and resource-use efficiency")
    draw_aggregate(summary, RESOURCES, OUT / "figures/155E2_total_irrigation_nitrogen_bar.png", "SY E2 forecast: cumulative resource use, 2014–2023", "sum")
    draw_aggregate(summary, METRICS, OUT / "figures/155E2_mean_yield_efficiency_bar.png", "SY E2 forecast: mean yield and resource-use efficiency, 2014–2023", "mean")

    gaps: list[dict[str, object]] = []
    for year, group in summary.groupby("year"):
        candidate = group[group["scenario"].eq("rl_candidate")].iloc[0]
        baselines = group[~group["scenario"].eq("rl_candidate")]
        for field, label in METRICS:
            maximum = _numeric(baselines[field]).max(skipna=True)
            value = float(candidate[field])
            gaps.append({"year": int(year), "metric": label, "e2_value": value, "baseline_max": float(maximum), "gap": value - maximum, "win": bool(value > maximum)})
    pd.DataFrame(gaps).to_csv(OUT / "tables/155E2_metric_gaps_vs_four_baselines.csv", index=False, encoding="utf-8-sig")

    e2_actions = summary[summary["scenario"].eq("rl_candidate")][["year", "action_sequence"]]
    e2_actions.to_csv(OUT / "tables/155E2_e2_action_sequences.csv", index=False, encoding="utf-8-sig")
    unique_sequences = int(e2_actions["action_sequence"].nunique())
    lines = [
        "# SY E2 天气预报版五情景策略查验图",
        "",
        "- PPO policy: SY E2 dual-branch weather reader, seed0, 5K checkpoint.",
        "- PPO/WP_ET/PFP-N 来源：`151E2N_wp_et_5k_replay/replays/e2_s0`，已完成 ETCP replay。",
        "- 四个基线：Null、Recorded template、DSSAT native automatic irrigation、Official expert。未纳入 external-N exploratory row。",
        "- 验证年份：2014–2023；每年一个五情景日过程图。",
        f"- E2 seed0 跨验证年份的不同 action sequence 数：`{unique_sequences}`；固定 checkpoint 下策略是确定性的，但输入天气/状态不同会产生不同的逐年动作序列。",
        "- WP_ET 定义：`grain yield / (10 × ETCP)`；PFP-N 在实际施氮为 0 的情景中保持 N/A。",
        "",
        "## 图件",
        "",
        "- `figures/155E2_syYYYY_five_scenario_daily.png`: 逐年五情景天气、土壤水、胁迫、动作、产量轨迹与累计奖励。",
        "- `figures/155E2_five_scenario_management.png`: 逐年总灌溉/施氮柱状图，复用旧 046_05 样式。",
        "- `figures/155E2_five_scenario_metrics.png`: 逐年产量、WP_ET、PFP-N 柱状图，复用旧 046_05 样式。",
        "- `figures/155E2_total_irrigation_nitrogen_bar.png`: 2014–2023 累计资源量。",
        "- `figures/155E2_mean_yield_efficiency_bar.png`: 2014–2023 平均产量与效率。",
        "",
        "## 解释边界",
        "",
        "这些图用于查验策略是否随年度状态、天气和胁迫变化而调整，不把某一年或某一指标的领先写成全面优于四基线。历史 WTH perfect hindcast 只说明框架可读到未来天气摘要，不等同于真实预报误差评估。",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "completed", "output_root": OUT.relative_to(ROOT).as_posix(), "years": YEARS, "figures": len(list((OUT / 'figures').glob('*.png'))), "unique_e2_action_sequences": unique_sequences}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
