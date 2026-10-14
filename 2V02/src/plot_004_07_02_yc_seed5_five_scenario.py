"""Recreate the seed-3 five-scenario figure suite for fixed PPO seed 5."""

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
sys.path.insert(0, str(ROOT / "src"))
from build_relaxed_success_five_scenario_daily_evidence_027_05 import (  # noqa: E402
    parse_management_events,
    parse_table,
)

REPLAY_ROOT = ROOT / "results/yc_random_weather_ppo/004_07_02_seed345_etcp_replay"
BASE_ROOT = ROOT / "benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000"
BASE_DAILY = BASE_ROOT / "tables/055_03_yca_five_scenario_daily.csv"
BASE_SUMMARY = BASE_ROOT / "tables/055_03_yca_five_scenario_season_summary.csv"
REPLAY_EPISODES = REPLAY_ROOT / "replay_episode_summary.csv"
OUT = ROOT / "results/yc_random_weather_ppo/004_05_review_figures/055_03_five_scenario/best_seed_seed5"
FIGURES = OUT / "figures"
YEARS = list(range(2014, 2024))
SEED = 5
PPO = "random_weather_ppo_seed5"
SCENARIOS = ["null", "recorded_farmer_template", "dssat_auto_external_n", "official_extension_expert", PPO]
LABELS = ["Null", "Recorded template", "DSSAT auto + external N", "Official expert", "Random-weather PPO (seed 5)"]
COLORS = ["#555555", "#C44E52", "#D8A305", "#7E63B6", "#2A9D55"]
STYLES = ["-", "--", "-.", ":", "-"]


def numeric(frame: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")


def ppo_daily(snapshot: Path, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    weather, plant, soil = (parse_table(snapshot / name) for name in ("Weather.OUT", "PlantGro.OUT", "SoilWat.OUT"))
    if weather.empty or plant.empty or soil.empty:
        raise RuntimeError(f"Missing daily DSSAT output for seed 5, {year}: {snapshot}")
    weather = weather[[c for c in ["YEAR", "DOY", "DAS", "PRED", "TMXD", "TMND"] if c in weather]].rename(
        columns={"YEAR": "year_out", "DOY": "doy", "DAS": "das", "PRED": "rainfall_mm", "TMXD": "tmax_c", "TMND": "tmin_c"})
    plant = plant[[c for c in ["DOY", "DAS", "DAP", "GWAD", "CWAD", "WSPD", "NSTD"] if c in plant]].rename(
        columns={"DOY": "doy", "DAS": "das", "DAP": "dap", "GWAD": "grain_yield_kg_ha", "CWAD": "biomass_kg_ha", "WSPD": "water_stress_index_wspd", "NSTD": "nitrogen_stress_index_nstd"})
    soil = soil[[c for c in ["DOY", "DAS", "SWTD"] if c in soil]].rename(
        columns={"DOY": "doy", "DAS": "das", "SWTD": "soil_water_mm"})
    daily = weather.merge(plant, on=["doy", "das"], how="left").merge(soil, on=["doy", "das"], how="left")
    fields = ["dap", "grain_yield_kg_ha", "biomass_kg_ha", "water_stress_index_wspd", "nitrogen_stress_index_nstd", "soil_water_mm"]
    for field in fields:
        if field not in daily:
            daily[field] = np.nan
    daily[fields] = daily[fields].ffill().fillna(0.0)
    daily["irrigation_executed_mm"] = 0.0
    daily["nitrogen_executed_kg_ha"] = 0.0
    parsed_events = parse_management_events(snapshot / "MgmtEvent.OUT")
    event_rows = []
    for event in parsed_events.itertuples(index=False):
        irrigation = float(event.amount) if str(event.operation).lower().startswith("irrig") else 0.0
        nitrogen = float(event.amount) if str(event.operation).lower().startswith("fert") else 0.0
        daily.loc[daily.doy.eq(int(event.doy)), "irrigation_executed_mm"] += irrigation
        daily.loc[daily.doy.eq(int(event.doy)), "nitrogen_executed_kg_ha"] += nitrogen
        event_rows.append({"year": year, "doy": int(event.doy), "dap": int(event.dap),
                           "irrigation_action_mm": irrigation, "fertilizer_action_kgN_ha": nitrogen})
    daily.insert(0, "year", year)
    daily.insert(1, "scenario", PPO)
    event_cols = ["year", "doy", "dap", "irrigation_action_mm", "fertilizer_action_kgN_ha"]
    events = pd.DataFrame(event_rows, columns=event_cols)
    if not events.empty:
        events = events.groupby(["year", "doy", "dap"], as_index=False).sum(numeric_only=True)
    return daily, events


def add_common_reward(daily: pd.DataFrame, ppo: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["unified_reward_step"] = -1.1 * out.irrigation_executed_mm - 1.58 * out.nitrogen_executed_kg_ha
    yields = ppo.set_index("year")["yield"]
    for year, indexes in out.groupby("year").groups.items():
        last = out.loc[indexes].sort_values(["doy", "das"]).index[-1]
        out.loc[last, "unified_reward_step"] += 0.158 * float(yields.loc[int(year)])
    out["unified_cumulative_reward"] = out.sort_values(["year", "doy", "das"]).groupby(
        ["year", "scenario"]).unified_reward_step.cumsum()
    return out


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_daily = pd.read_csv(BASE_DAILY, keep_default_na=False)
    base_daily = base_daily.loc[base_daily.scenario.isin(SCENARIOS[:-1])].copy()
    if set(base_daily.scenario.unique()) != set(SCENARIOS[:-1]):
        raise ValueError("Four complete 055_03 baseline daily traces are required")
    base = pd.read_csv(BASE_SUMMARY, keep_default_na=False)
    base = base.loc[base.scenario.isin(SCENARIOS[:-1])].copy().rename(columns={
        "grain_yield_kg_ha": "yield", "actual_irrigation_mm": "total_irrigation",
        "actual_nitrogen_kg_ha": "total_fertilizer", "unified_reward": "common_reward",
        "WP_ET_kg_m3": "wp_et", "PFP_N_kg_kg": "pfp_n"})
    base["year"] = pd.to_numeric(base.year).astype(int)
    numeric(base, ["yield", "total_irrigation", "total_fertilizer", "common_reward", "wp_et", "pfp_n"])

    replay = pd.read_csv(REPLAY_EPISODES, keep_default_na=False)
    replay = replay.loc[pd.to_numeric(replay.seed, errors="coerce").eq(SEED)].copy()
    replay["year"] = pd.to_numeric(replay.year).astype(int)
    if len(replay) != 10 or set(replay.year) != set(YEARS):
        raise ValueError("Expected exactly one completed seed-5 replay per validation year")
    ppo = replay.rename(columns={"replay_yield_kg_ha": "yield", "irrigation_mm": "total_irrigation",
        "nitrogen_kg_ha": "total_fertilizer", "wp_et_kg_m3": "wp_et", "pfp_n_kg_kg": "pfp_n"})
    numeric(ppo, ["yield", "total_irrigation", "total_fertilizer", "common_reward", "wp_et", "pfp_n"])
    formula = 0.158 * ppo["yield"] - 1.1 * ppo.total_irrigation - 1.58 * ppo.total_fertilizer
    if not np.allclose(formula, ppo.common_reward, atol=1e-5, rtol=0):
        raise ValueError("Seed-5 reward fails the 055_03 common-formula check")

    cols = ["year", "scenario", "yield", "total_irrigation", "total_fertilizer", "common_reward", "wp_et", "pfp_n"]
    base_metrics = base[cols].copy()
    ppo_metrics = ppo[["year", "yield", "total_irrigation", "total_fertilizer", "common_reward", "wp_et", "pfp_n"]].copy()
    ppo_metrics["scenario"] = PPO
    metrics = pd.concat([base_metrics, ppo_metrics[cols]], ignore_index=True)
    if len(metrics) != 50 or metrics.groupby(["year", "scenario"]).size().ne(1).any():
        raise ValueError("Expected 50 unique year-scenario metric rows")

    ppo_days, event_frames = [], []
    for _, row in ppo.sort_values("year").iterrows():
        day, events = ppo_daily(ROOT / str(row["snapshot_dir"]), int(row["year"]))
        if not np.isclose(day.grain_yield_kg_ha.iloc[-1], row["yield"], atol=1.0, rtol=0):
            raise ValueError(f"Seed-5 daily yield does not close for {int(row['year'])}")
        if not np.isclose(day.irrigation_executed_mm.sum(), row.total_irrigation, atol=0.2, rtol=0):
            raise ValueError(f"Seed-5 irrigation events do not close for {int(row['year'])}")
        if not np.isclose(day.nitrogen_executed_kg_ha.sum(), row.total_fertilizer, atol=0.2, rtol=0):
            raise ValueError(f"Seed-5 nitrogen events do not close for {int(row['year'])}")
        event_days = len(events)
        event_count = int(events.irrigation_action_mm.gt(0).sum() + events.fertilizer_action_kgN_ha.gt(0).sum())
        if event_days != int(row.management_action_days_nonzero) or event_count != int(row.management_event_count):
            raise ValueError(f"Seed-5 management event counts do not close for {int(row['year'])}")
        ppo_days.append(day)
        event_frames.append(events)
    events = pd.concat(event_frames, ignore_index=True)
    seed5_daily = add_common_reward(pd.concat(ppo_days, ignore_index=True), ppo)
    daily = pd.concat([base_daily, seed5_daily], ignore_index=True, sort=False)
    return ppo, metrics, events, daily


def plot_daily_year(daily: pd.DataFrame, year: int) -> None:
    frame = daily.loc[pd.to_numeric(daily.year).eq(year)]
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    weather = frame.loc[frame.scenario.eq("null")].sort_values("dap")
    axes[0, 0].bar(weather.dap, weather.rainfall_mm.astype(float), color="#80A9C7", alpha=0.85, label="Rain")
    temp = axes[0, 0].twinx()
    temp.plot(weather.dap, weather.tmax_c.astype(float), color="#C23B32", lw=1.25)
    temp.plot(weather.dap, weather.tmin_c.astype(float), color="#666666", lw=1.1, ls="--")
    axes[0, 0].set_title("Weather")
    axes[0, 0].set_ylabel("Rain (mm)")
    temp.set_ylabel("Temperature (C)")
    for scenario, label, color, style in zip(SCENARIOS[:-1], LABELS[:-1], COLORS[:-1], STYLES[:-1]):
        sub = frame.loc[frame.scenario.eq(scenario)].sort_values("dap")
        axes[0, 1].plot(sub.dap, sub.soil_water_mm.astype(float), color=color, ls=style, lw=1.35, label=label)
        axes[1, 0].plot(sub.dap, sub.water_stress_index_wspd.astype(float), color=color, ls=style, lw=1.35, label=label)
        axes[1, 1].plot(sub.dap, sub.nitrogen_stress_index_nstd.astype(float), color=color, ls=style, lw=1.35, label=label)
        axes[3, 0].plot(sub.dap, sub.grain_yield_kg_ha.astype(float), color=color, ls=style, lw=1.45, label=f"{label} grain")
        axes[3, 0].plot(sub.dap, sub.biomass_kg_ha.astype(float), color=color, ls=style, lw=0.85, alpha=0.4)
        axes[3, 1].plot(sub.dap, sub.unified_cumulative_reward.astype(float), color=color, ls=style, lw=1.35, label=label)
        for ax, col in ((axes[2, 0], "irrigation_executed_mm"), (axes[2, 1], "nitrogen_executed_kg_ha")):
            event = sub.loc[pd.to_numeric(sub[col], errors="coerce").gt(0)]
            if not event.empty:
                ax.stem(event.dap.astype(float), event[col].astype(float), linefmt=color, markerfmt="o", basefmt=" ", label=label)
    sub = frame.loc[frame.scenario.eq(PPO)].sort_values("dap")
    color = COLORS[-1]
    axes[0, 1].plot(sub.dap, sub.soil_water_mm.astype(float), color=color, lw=1.5, label=LABELS[-1])
    axes[1, 0].plot(sub.dap, sub.water_stress_index_wspd.astype(float), color=color, lw=1.5, label=LABELS[-1])
    axes[1, 1].plot(sub.dap, sub.nitrogen_stress_index_nstd.astype(float), color=color, lw=1.5, label=LABELS[-1])
    axes[3, 0].plot(sub.dap, sub.grain_yield_kg_ha.astype(float), color=color, lw=1.5, label=f"{LABELS[-1]} grain")
    axes[3, 0].plot(sub.dap, sub.biomass_kg_ha.astype(float), color=color, lw=0.85, alpha=0.4)
    axes[3, 1].plot(sub.dap, sub.unified_cumulative_reward.astype(float), color=color, lw=1.5, label=LABELS[-1])
    for ax, col in ((axes[2, 0], "irrigation_executed_mm"), (axes[2, 1], "nitrogen_executed_kg_ha")):
        event = sub.loc[pd.to_numeric(sub[col], errors="coerce").gt(0)]
        if not event.empty:
            ax.stem(event.dap.astype(float), event[col].astype(float), linefmt=color, markerfmt="o", basefmt=" ", label=LABELS[-1])
    panels = [(axes[0, 1], "Soil water", "SWTD (mm)"), (axes[1, 0], "Water stress index", "WSPD"),
              (axes[1, 1], "Nitrogen stress index", "NSTD"), (axes[2, 0], "Irrigation events", "mm/event"),
              (axes[2, 1], "Nitrogen application events", "kg/ha/event"),
              (axes[3, 0], "Grain and biomass trajectories", "kg/ha"), (axes[3, 1], "Common cumulative reward", "common units")]
    for ax, title, ylabel in panels:
        ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(alpha=0.2)
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1], axes[3, 1]):
        ax.legend(fontsize=7, ncol=2)
    for ax in axes[3]:
        ax.set_xlabel("DAP")
    fig.suptitle(f"YCA{year} five-scenario daily process | Random-weather PPO seed 5", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGURES / f"004_05_seed5_yca{year}_five_scenario_daily.png", dpi=220)
    plt.close(fig)


def grouped_bars(metrics: pd.DataFrame, fields: list[tuple[str, str]], name: str) -> None:
    x, width = np.arange(len(YEARS)), 0.16
    fig, axes = plt.subplots(len(fields), 1, figsize=(16, 4 * len(fields)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (field, ylabel) in zip(axes, fields):
        for i, (scenario, label, color) in enumerate(zip(SCENARIOS, LABELS, COLORS)):
            rows = metrics.loc[metrics.scenario.eq(scenario)].set_index("year").reindex(YEARS)
            values = pd.to_numeric(rows[field], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(values).any():
                ax.bar(x + (i - 2) * width, values, width, label=label, color=color)
        ax.legend(ncol=3, fontsize=8); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x, YEARS); axes[-1].set_xlabel("Validation year")
    fig.tight_layout(); fig.savefig(FIGURES / name, dpi=200); plt.close(fig)


def plot_metrics(metrics: pd.DataFrame) -> None:
    fields = [("yield", "Grain yield (kg/ha)"), ("common_reward", "Common reward (055_03 formula)"),
              ("total_irrigation", "Total irrigation (mm)"), ("total_fertilizer", "Total nitrogen (kg N/ha)"),
              ("pfp_n", "PFP_N (kg grain/kg N)")]
    fig, axes = plt.subplots(5, 1, figsize=(16, 17), sharex=True); x = np.arange(len(YEARS))
    for ax, (field, ylabel) in zip(axes, fields):
        for scenario, label, color, style in zip(SCENARIOS, LABELS, COLORS, STYLES):
            rows = metrics.loc[metrics.scenario.eq(scenario)].set_index("year").reindex(YEARS)
            values = pd.to_numeric(rows[field], errors="coerce").to_numpy(dtype=float)
            selected = scenario == PPO
            ax.plot(x, values, color=color, ls=style, lw=2.2 if selected else 1.4,
                    marker="o", ms=4.2 if selected else 3.2, label=label, zorder=4 if selected else 2)
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.22); ax.legend(ncol=3, fontsize=8, frameon=False, loc="best")
    axes[-1].set_xticks(x, YEARS); axes[-1].set_xlabel("Validation year")
    fig.suptitle("YCA lowIC | selected Random-weather PPO seed 5 vs four baselines", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.975]); fig.savefig(OUT / "seed5_metrics_vs_four_baselines.png", dpi=200); plt.close(fig)


def plot_actions(events: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 7.5)); y_pos = {year: i for i, year in enumerate(YEARS)}
    for row in events.itertuples(index=False):
        y = y_pos[int(row.year)]
        if row.irrigation_action_mm > 0:
            ax.scatter(row.dap, y - 0.11, s=50, marker="o", color="#2878B5", edgecolor="white", linewidth=0.5, zorder=3)
            ax.annotate(f"I {row.irrigation_action_mm:g}", (row.dap, y - 0.11), xytext=(-24, -10), textcoords="offset points", fontsize=7, color="#22577A", ha="right")
        if row.fertilizer_action_kgN_ha > 0:
            ax.scatter(row.dap, y + 0.11, s=56, marker="s", color="#D9822B", edgecolor="white", linewidth=0.5, zorder=3)
            ax.annotate(f"N {row.fertilizer_action_kgN_ha:g}", (row.dap, y + 0.11), xytext=(6, 4), textcoords="offset points", fontsize=7, color="#9C5700", ha="left")
    ax.set_yticks(range(len(YEARS)), YEARS); ax.invert_yaxis()
    ax.set_xlabel("Days after planting (DAP)"); ax.set_ylabel("Observed weather year")
    ax.set_title("Actual seed 5 management actions by year", loc="left", fontweight="bold"); ax.grid(axis="x", alpha=0.22)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color="#2878B5", label="Irrigation (mm)"),
                       plt.Line2D([], [], marker="s", ls="", color="#D9822B", label="Nitrogen (kg/ha)")], frameon=False, ncol=2, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "seed5_management_actions_by_year.png", dpi=200); plt.close(fig)


def plot_management(ppo: pd.DataFrame, events: pd.DataFrame) -> None:
    days = events.groupby("year").size(); fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True); x = np.arange(len(YEARS))
    for ax, field, label in zip(axes[:2], ["total_irrigation", "total_fertilizer"], ["Irrigation (mm/season)", "Nitrogen (kg/ha/season)"]):
        ax.bar(x, ppo.set_index("year").reindex(YEARS)[field], width=0.62, color="#2A9D55", edgecolor="#1E6E3B", linewidth=0.6)
        ax.set_ylabel(label); ax.grid(axis="y", alpha=0.22)
    axes[2].bar(x, days.reindex(YEARS).fillna(0), width=0.62, color="#2A9D55", edgecolor="#1E6E3B", linewidth=0.6)
    axes[2].set_ylabel("Positive action days per season"); axes[2].grid(axis="y", alpha=0.22)
    axes[-1].set_xticks(x, YEARS); axes[-1].set_xlabel("Validation year")
    fig.suptitle("Selected seed 5 | yearly management profile", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(OUT / "seed5_yearly_management_profile.png", dpi=200); plt.close(fig)


def report(ppo: pd.DataFrame, metrics: pd.DataFrame, events: pd.DataFrame) -> str:
    p = ppo.set_index("year")
    winners = {}
    for field, high in [("yield", True), ("wp_et", True), ("pfp_n", True), ("common_reward", True),
                        ("total_irrigation", False), ("total_fertilizer", False)]:
        wins = 0
        for year in YEARS:
            group = metrics.loc[metrics.year.eq(year), field]
            optimum = group.max() if high else group.min()
            wins += int(np.isclose(p.loc[year, field], optimum, atol=1e-8, rtol=0))
        winners[field] = wins
    yield_wins = [year for year in YEARS if p.loc[year, "yield"] > metrics.loc[(metrics.year.eq(year)) & (metrics.scenario.ne(PPO)), "yield"].max()]
    event_days = events.groupby("year").size()
    seed3 = ROOT / "results/yc_random_weather_ppo/004_05_review_figures/055_03_five_scenario/best_seed_seed3"
    return "\n".join([
        "# YC seed 5 五情景逐年结果图",
        "",
        "## 样式复用",
        "沿用 `best_seed_seed3` 的五情景顺序、灰/红/金/紫/绿配色、线型、分组柱状图、年度指标折线、管理事件图、管理画像及 4×2 逐年过程布局。seed3 原文件没有被覆盖。seed5 有精确 ETCP 回放，因此 WP_ET 面板展示真实数值，不沿用 seed3 的缺失标注。",
        "",
        f"- 样式参考图：`{(seed3 / 'seed3_metrics_vs_four_baselines.png').relative_to(ROOT).as_posix()}`",
        f"- 逐年过程参考图：`{(seed3 / 'figures/004_05_seed3_yca2014_five_scenario_daily.png').relative_to(ROOT).as_posix()}`",
        f"- 参考脚本：`{(seed3.parent / 'plot_004_05_seed3_diagnostic.py').relative_to(ROOT).as_posix()}`",
        "",
        "## 数据口径",
        f"- seed5 十年 replay 和精确 ETCP/WP_ET/PFP_N：`{REPLAY_EPISODES.relative_to(ROOT).as_posix()}`及其 seed5 年度 DSSAT snapshots。",
        f"- 四基准年度表：`{BASE_SUMMARY.relative_to(ROOT).as_posix()}`；逐日表：`{BASE_DAILY.relative_to(ROOT).as_posix()}`。",
        "- 所有情景的对比 reward 使用 `0.158×yield − 1.1×irrigation − 1.58×nitrogen`，不与 canonical reward 混算。无 PPO 训练或新增 DSSAT 运行。",
        "",
        "## 图上结果摘要",
        f"- seed5 十年均值：产量 {p['yield'].mean():.1f} kg/ha；灌溉 {p.total_irrigation.mean():.1f} mm；施氮 {p.total_fertilizer.mean():.1f} kg N/ha；WP_ET {p.wp_et.mean():.3f} kg/m³；PFP_N {p.pfp_n.mean():.2f} kg grain/kg N；common reward {p.common_reward.mean():.1f}。",
        f"- 产量超过当年四基准最高值的年份：{', '.join(map(str, yield_wins)) if yield_wins else '无'}（{len(yield_wins)}/10）。五情景年度冠军数：产量 {winners['yield']}/10、WP_ET {winners['wp_et']}/10、PFP_N {winners['pfp_n']}/10、common reward {winners['common_reward']}/10。",
        f"- 五情景最低投入年份数：灌溉 {winners['total_irrigation']}/10，氮 {winners['total_fertilizer']}/10。管理画像均值：灌溉 {p.total_irrigation.mean():.1f} mm/年、施氮 {p.total_fertilizer.mean():.1f} kg N/ha/年、施用日 {event_days.mean():.1f} 天/年、灌溉/施氮事件 {pd.to_numeric(ppo.management_event_count).mean():.1f} 次/年。",
        "- 这些是选定单一 checkpoint/seed 对 2014–2023 已观察验证年的结果，不构成独立年份上的确认性泛化结论。",
        "",
        "## 文件",
        "- `figures/004_05_seed5_five_scenario_metrics.png`：逐年产量、WP_ET、PFP_N 五情景柱状图。",
        "- `figures/004_05_seed5_five_scenario_management.png`：逐年灌溉、施氮五情景柱状图。",
        "- `figures/004_05_seed5_five_scenario_reward.png`：逐年 common reward 五情景柱状图。",
        "- `figures/004_05_seed5_yca{year}_five_scenario_daily.png`：十个验证年的五情景逐日过程图。",
        "- 根目录另有 `seed5_metrics_vs_four_baselines.png`、`seed5_management_actions_by_year.png`、`seed5_yearly_management_profile.png`。",
        "",
    ])


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing output: {OUT}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    ppo, metrics, events, daily = load_data()
    metrics.to_csv(OUT / "seed5_yearly_metrics_vs_four_baselines.csv", index=False)
    events.to_csv(OUT / "seed5_observed_management_events.csv", index=False)
    daily.to_csv(OUT / "seed5_five_scenario_daily.csv", index=False)
    for year in YEARS:
        plot_daily_year(daily, year)
    grouped_bars(metrics, [("yield", "grain_yield_kg_ha"), ("wp_et", "WP_ET_kg_m3"), ("pfp_n", "PFP_N_kg_kg")], "004_05_seed5_five_scenario_metrics.png")
    grouped_bars(metrics, [("total_irrigation", "actual_irrigation_mm"), ("total_fertilizer", "actual_nitrogen_kg_ha")], "004_05_seed5_five_scenario_management.png")
    grouped_bars(metrics, [("common_reward", "unified_reward (055_03 common formula)")], "004_05_seed5_five_scenario_reward.png")
    plot_metrics(metrics); plot_actions(events); plot_management(ppo, events)
    manifest = {"selected_seed": SEED, "years": YEARS, "scenario_order": SCENARIOS,
        "style_reference": "results/yc_random_weather_ppo/004_05_review_figures/055_03_five_scenario/best_seed_seed3",
        "style_script_reference": "results/yc_random_weather_ppo/004_05_review_figures/055_03_five_scenario/plot_004_05_seed3_diagnostic.py",
        "baseline_summary_source": str(BASE_SUMMARY.relative_to(ROOT)), "baseline_daily_source": str(BASE_DAILY.relative_to(ROOT)),
        "seed5_replay_source": str(REPLAY_EPISODES.relative_to(ROOT)),
        "seed5_snapshots": "results/yc_random_weather_ppo/004_07_02_seed345_etcp_replay/snapshots/seed_5/YCA/{year}",
        "wp_et_source": "Exact seed-5 Summary.OUT replay ETCP; WP_ET = yield / ETCP",
        "common_reward_formula": "0.158*yield - 1.1*irrigation - 1.58*nitrogen", "training_or_new_dssat_runs": False,
        "png_count": len(list(OUT.rglob("*.png")))}
    (OUT / "seed5_five_scenario_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "seed5_diagnostic_report.md").write_text(report(ppo, metrics, events), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "png_count": len(list(OUT.rglob("*.png"))), "daily_rows": len(daily),
        "event_days": len(events), "mean_yield": float(ppo["yield"].mean()),
        "mean_irrigation": float(ppo.total_irrigation.mean()), "mean_nitrogen": float(ppo.total_fertilizer.mean()),
        "mean_wp_et": float(ppo.wp_et.mean()), "mean_pfp_n": float(ppo.pfp_n.mean()),
        "mean_common_reward": float(ppo.common_reward.mean())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
