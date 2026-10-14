"""Build LC-E2 5K five-scenario figures from the completed WP_ET replay."""

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

import build_155E2_sya_weather_forecast_five_scenario_figures as sf


BASELINE = ROOT / "benchmark_results/053_02_lca_lowIC_four_baselines_static_level1/evaluation/053_02_baseline_summary.csv"
E2_REPLAY = ROOT / "benchmark_results/157LC_lca_lowIC_dual_branch_weather_reader_wp_et_5k_replay/157LC_wp_et_replay_by_policy_year.csv"
E2_REPLAY_ROOT = ROOT / "benchmark_results/157LC_lca_lowIC_dual_branch_weather_reader_wp_et_5k_replay/replays/e2_s0/snapshots/LCA"
BASELINE_ROOT = ROOT / "benchmark_results/053_02_lca_lowIC_four_baselines_static_level1/snapshots/LCA"
OUT = ROOT / "benchmark_results/159LC_lca_lowIC_e2_five_scenario_figures"
DOC = ROOT / "docs/159LC_lca_lowIC_e2_five_scenario_figures_record.md"
YEARS = list(range(2014, 2024))


def configure_shared_loader() -> None:
    sf.BASELINE = BASELINE
    sf.E2_REPLAY = E2_REPLAY
    sf.E2_REPLAY_ROOT = E2_REPLAY_ROOT
    sf.YEARS = YEARS


def plot_year(daily: pd.DataFrame, year: int, path: Path) -> None:
    scenarios = sf.SCENARIOS
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    weather = daily[daily["scenario"].eq("null")].sort_values("dap")
    axes[0, 0].bar(weather["dap"], weather["rainfall_mm"], color="#80A9C7", alpha=0.85, label="Rain")
    temp = axes[0, 0].twinx()
    temp.plot(weather["dap"], weather["tmax_c"], color="#C23B32", lw=1.25, label="Tmax")
    temp.plot(weather["dap"], weather["tmin_c"], color="#666666", lw=1.1, ls="--", label="Tmin")
    axes[0, 0].set_title("Weather"); axes[0, 0].set_ylabel("Rain (mm)"); temp.set_ylabel("Temperature (°C)")
    for scenario in scenarios:
        sub = daily[daily["scenario"].eq(scenario)].sort_values("dap")
        color, style, label = sf.COLORS[scenario], sf.STYLES[scenario], sf.LABELS[scenario]
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
    for axis, title, ylabel in [
        (axes[0, 1], "Soil water", "SWTD (mm)"), (axes[1, 0], "Water stress index", "WSPD"),
        (axes[1, 1], "Nitrogen stress index", "NSTD"), (axes[2, 0], "Irrigation events", "mm/event"),
        (axes[2, 1], "Nitrogen application events", "kg/ha/event"), (axes[3, 0], "Grain and biomass trajectories", "kg/ha"),
        (axes[3, 1], "Unified cumulative reward", "common units"),
    ]:
        axis.set_title(title); axis.set_ylabel(ylabel); axis.grid(alpha=0.2)
    axes[0, 1].legend(fontsize=7, ncol=2); axes[2, 0].legend(fontsize=7, ncol=2); axes[2, 1].legend(fontsize=7, ncol=2); axes[3, 1].legend(fontsize=7, ncol=2)
    axes[3, 0].set_xlabel("DAP"); axes[3, 1].set_xlabel("DAP")
    fig.suptitle(f"LC{year} E2 weather-forecast five-scenario daily process", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(path, dpi=220); plt.close(fig)


def main() -> None:
    configure_shared_loader()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    summary = sf.load_summary()
    # The shared SY plotting loader carries source labels from its original
    # experiment; replace them here so the LC artifact is self-contained.
    summary["source"] = summary["source"].replace({
        "046_03_sya_originIC_four_baselines": "053_02_lca_lowIC_four_baselines_static_level1",
        "151E2N_wp_et_5k_replay_e2_s0": "157LC_lca_lowIC_dual_branch_weather_reader_wp_et_5k_replay_e2_s0",
    })
    summary.to_csv(OUT / "tables/159LC_five_scenario_summary.csv", index=False, encoding="utf-8-sig")
    daily_frames: list[pd.DataFrame] = []
    for year in YEARS:
        snapshot_map = {
            "null": BASELINE_ROOT / str(year) / "null",
            "recorded_farmer_template": BASELINE_ROOT / str(year) / "recorded_farmer_template",
            "dssat_auto": BASELINE_ROOT / str(year) / "dssat_auto",
            "official_extension_expert": BASELINE_ROOT / str(year) / "official_extension_expert",
            "rl_candidate": E2_REPLAY_ROOT / str(year) / "ckpt5000",
        }
        missing = [str(p) for p in snapshot_map.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"LC{year} missing snapshot(s): {missing}")
        daily = sf.add_unified_reward(pd.concat([sf.daily_from_snapshot(p, year, s) for s, p in snapshot_map.items()], ignore_index=True))
        daily_frames.append(daily)
        daily.to_csv(OUT / f"tables/159LC_lc{year}_five_scenario_daily.csv", index=False, encoding="utf-8-sig")
        plot_year(daily, year, OUT / f"figures/159LC_lc{year}_five_scenario_daily.png")
    pd.concat(daily_frames, ignore_index=True).to_csv(OUT / "tables/159LC_all_years_five_scenario_daily.csv", index=False, encoding="utf-8-sig")
    sf.YEARS = YEARS
    sf.draw_grouped(summary, sf.RESOURCES, OUT / "figures/159LC_five_scenario_management.png", "LC E2 forecast: yearly total irrigation and nitrogen")
    sf.draw_grouped(summary, sf.METRICS, OUT / "figures/159LC_five_scenario_metrics.png", "LC E2 forecast: yearly yield and resource-use efficiency")
    sf.draw_aggregate(summary, sf.RESOURCES, OUT / "figures/159LC_total_irrigation_nitrogen_bar.png", "LC E2 forecast: cumulative resource use, 2014–2023", "sum")
    sf.draw_aggregate(summary, sf.METRICS, OUT / "figures/159LC_mean_yield_efficiency_bar.png", "LC E2 forecast: mean yield and resource-use efficiency, 2014–2023", "mean")
    gaps = []
    for year, group in summary.groupby("year"):
        candidate = group[group["scenario"].eq("rl_candidate")].iloc[0]
        baselines = group[~group["scenario"].eq("rl_candidate")]
        for field, label in sf.METRICS:
            maximum = pd.to_numeric(baselines[field], errors="coerce").max(skipna=True)
            value = float(candidate[field])
            gaps.append({"year": int(year), "metric": label, "e2_value": value, "baseline_max": float(maximum), "gap": value - maximum, "win": bool(value > maximum)})
    pd.DataFrame(gaps).to_csv(OUT / "tables/159LC_metric_gaps_vs_four_baselines.csv", index=False, encoding="utf-8-sig")
    actions = summary[summary["scenario"].eq("rl_candidate")][["year", "action_sequence"]]
    actions.to_csv(OUT / "tables/159LC_e2_action_sequences.csv", index=False, encoding="utf-8-sig")
    unique = int(actions["action_sequence"].nunique())
    report = [
        "# LC E2 天气预报版五情景策略查验图", "", "- 策略：156LC LC-E2 dual-branch weather reader，seed0，5K checkpoint。",
        "- WP_ET/PFP-N：157LC 只读 ETCP replay；四基线来自 053_02 LC lowIC four baselines。",
        "- 验证年份：2014–2023；每年一个五情景日过程图。", f"- 跨验证年份不同 action sequence 数：`{unique}`。",
        "- WP_ET = grain yield / (10 × ETCP)；零施氮情景 PFP-N 为 N/A。", "", "## 解释边界", "",
        "这些图用于检查 LC-E2 是否随年度天气、土壤水和胁迫调整动作，不把某一年或单一指标领先写成全面优于四基线。历史 WTH 是 perfect hindcast，不等于真实预报误差评估。", "",
    ]
    DOC.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": "completed", "output_root": OUT.relative_to(ROOT).as_posix(), "years": YEARS, "figures": len(list((OUT / 'figures').glob('*.png'))), "unique_e2_action_sequences": unique}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
