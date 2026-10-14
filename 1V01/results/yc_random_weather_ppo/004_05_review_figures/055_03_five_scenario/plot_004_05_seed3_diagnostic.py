"""Plot a transparent, descriptive diagnostic for selected 004_05 PPO seed 3."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[4]
PPO_ROOT = ROOT / "results/yc_random_weather_ppo/004_05"
OLD_ROOT = ROOT / "benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000"
EPISODES = PPO_ROOT / "all_evaluation_episode_level_0_7.csv"
STEPS = PPO_ROOT / "evaluation_step_level_all_models.csv"
BASE_SUMMARY = OLD_ROOT / "tables/055_03_yca_five_scenario_season_summary.csv"
OUT = Path(__file__).resolve().parent / "best_seed_seed3"
FIGURES = OUT / "figures"
YEARS = list(range(2014, 2024))
SEED = 3
SCENARIOS = ["null", "recorded_farmer_template", "dssat_auto_external_n",
             "official_extension_expert", "random_weather_ppo_seed3"]
LABELS = ["Null", "Recorded template", "DSSAT auto + external N",
          "Official expert", "Random-weather PPO (seed 3)"]
COLORS = ["#555555", "#C44E52", "#D8A305", "#7E63B6", "#2A9D55"]
STYLES = ["-", "--", "-.", ":", "-"]
BASE_SCENARIOS = SCENARIOS[:-1]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ep = pd.read_csv(EPISODES, keep_default_na=False)
    ep = ep.loc[
        ep.training_regime.eq("RANDOM_WEATHER_WGEN")
        & ep.ppo_seed.eq(SEED)
        & ep.evaluation_weather_type.eq("observed_weather")
    ].copy()
    if len(ep) != len(YEARS) or set(pd.to_numeric(ep.evaluation_weather_year).astype(int)) != set(YEARS):
        raise ValueError("Expected seed 3 observed-weather episodes for 2014-2023")
    ep["year"] = pd.to_numeric(ep.evaluation_weather_year).astype(int)
    for col in ["yield", "total_irrigation", "total_fertilizer", "reward",
                "irrigation_event_count_events", "fertilizer_event_count_events",
                "action_events"]:
        ep[col] = pd.to_numeric(ep[col], errors="coerce")
    ep["common_reward_05503"] = (
        0.158 * ep["yield"] - 1.1 * ep.total_irrigation - 1.58 * ep.total_fertilizer
    )

    base = pd.read_csv(BASE_SUMMARY, keep_default_na=False)
    base = base.loc[base.scenario.isin(SCENARIOS[:-1])].copy()
    base = base.rename(columns={
        "grain_yield_kg_ha": "yield", "actual_irrigation_mm": "total_irrigation",
        "actual_nitrogen_kg_ha": "total_fertilizer", "unified_reward": "common_reward_05503",
    })
    base["year"] = pd.to_numeric(base.year).astype(int)
    for col in ["yield", "total_irrigation", "total_fertilizer", "common_reward_05503"]:
        base[col] = pd.to_numeric(base[col], errors="coerce")
    base["pfp_n"] = pd.to_numeric(base["PFP_N_kg_kg"], errors="coerce")
    base["scenario"] = base.scenario.astype(str)

    selected = ep[["year", "yield", "total_irrigation", "total_fertilizer", "common_reward_05503"]].copy()
    selected["scenario"] = "random_weather_ppo_seed3"
    selected["pfp_n"] = selected["yield"] / selected.total_fertilizer.replace(0, np.nan)
    metrics = pd.concat([
        base[["year", "scenario", "yield", "total_irrigation", "total_fertilizer",
              "common_reward_05503", "pfp_n"]],
        selected[["year", "scenario", "yield", "total_irrigation", "total_fertilizer",
                  "common_reward_05503", "pfp_n"]],
    ], ignore_index=True)
    if len(metrics) != 50:
        raise ValueError("Expected five scenarios for each of ten years")

    usecols = ["training_regime", "ppo_seed", "evaluation_weather_type",
               "evaluation_weather_year", "evaluation_weather_seed", "episode_key",
               "timestep", "doy", "dap", "swfac", "nstres", "topwt", "grnwt",
               "post_step_state_json", "RAIN", "TMAX", "TMIN", "instant_reward",
               "irrigation_action_mm", "fertilizer_action_kgN_ha"]
    chunks = []
    for chunk in pd.read_csv(STEPS, usecols=usecols, chunksize=10000):
        keep = chunk.loc[
            chunk.training_regime.eq("RANDOM_WEATHER_WGEN")
            & chunk.ppo_seed.eq(SEED)
            & chunk.evaluation_weather_type.eq("observed_weather")
        ]
        if not keep.empty:
            chunks.append(keep)
    if not chunks:
        raise ValueError("No seed 3 observed-weather step records")
    step = pd.concat(chunks, ignore_index=True)
    step["year"] = pd.to_numeric(step.evaluation_weather_year).astype(int)
    for col in ["doy", "dap", "irrigation_action_mm", "fertilizer_action_kgN_ha"]:
        step[col] = pd.to_numeric(step[col], errors="coerce")
    events = step.loc[
        step.irrigation_action_mm.gt(0) | step.fertilizer_action_kgN_ha.gt(0),
        ["year", "evaluation_weather_seed", "episode_key", "timestep", "doy", "dap",
         "irrigation_action_mm", "fertilizer_action_kgN_ha"],
    ].copy()
    if events.empty:
        raise ValueError("Seed 3 action trace contains no positive management events")
    baseline_daily = pd.read_csv(OLD_ROOT / "tables/055_03_yca_five_scenario_daily.csv",
                                 keep_default_na=False)
    baseline_daily = baseline_daily.loc[baseline_daily.scenario.isin(BASE_SCENARIOS)].copy()
    if baseline_daily.empty:
        raise ValueError("Missing the four 055_03 baseline daily traces")
    return ep, metrics, events, baseline_daily, step


def plot_daily_year(baseline: pd.DataFrame, episode: pd.Series,
                    step: pd.DataFrame, year: int) -> None:
    year_base = baseline.loc[pd.to_numeric(baseline.year).eq(year)]
    ppo = step.loc[pd.to_numeric(step.evaluation_weather_year).eq(year)].copy()
    ppo = ppo.sort_values(["timestep"])
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    weather = year_base.loc[year_base.scenario.eq("null")].sort_values("dap")
    axes[0, 0].bar(weather.dap, weather.rainfall_mm.astype(float), color="#80A9C7", alpha=0.85, label="Rain")
    temp = axes[0, 0].twinx()
    temp.plot(weather.dap, weather.tmax_c.astype(float), color="#C23B32", lw=1.25, label="Tmax")
    temp.plot(weather.dap, weather.tmin_c.astype(float), color="#666666", lw=1.1, ls="--", label="Tmin")
    axes[0, 0].set_title("Weather")
    axes[0, 0].set_ylabel("Rain (mm)")
    temp.set_ylabel("Temperature (C)")

    for scenario, label, color, style in zip(BASE_SCENARIOS, LABELS, COLORS, STYLES):
        sub = year_base.loc[year_base.scenario.eq(scenario)].sort_values("dap")
        axes[0, 1].plot(sub.dap, sub.soil_water_mm.astype(float), color=color, ls=style, lw=1.35, label=label)
        axes[1, 0].plot(sub.dap, sub.water_stress_index_wspd.astype(float), color=color, ls=style, lw=1.35, label=label)
        axes[1, 1].plot(sub.dap, sub.nitrogen_stress_index_nstd.astype(float), color=color, ls=style, lw=1.35, label=label)
        axes[3, 0].plot(sub.dap, sub.grain_yield_kg_ha.astype(float), color=color, ls=style, lw=1.45, label=f"{label} grain")
        axes[3, 0].plot(sub.dap, sub.biomass_kg_ha.astype(float), color=color, ls=style, lw=0.85, alpha=0.4)
        axes[3, 1].plot(sub.dap, sub.unified_cumulative_reward.astype(float), color=color, ls=style, lw=1.35, label=label)
        for ax, field in ((axes[2, 0], "irrigation_executed_mm"),
                          (axes[2, 1], "nitrogen_executed_kg_ha")):
            event = sub.loc[pd.to_numeric(sub[field], errors="coerce").gt(0)]
            if not event.empty:
                ax.stem(event.dap.astype(float), event[field].astype(float), linefmt=color,
                        markerfmt="o", basefmt=" ", label=label)

    color = COLORS[-1]
    axes[1, 0].plot(ppo.dap, ppo.swfac, color=color, lw=1.5, label="Random-weather PPO seed 3 SWFAC")
    axes[1, 1].plot(ppo.dap, ppo.nstres, color=color, lw=1.5, label="Random-weather PPO seed 3 NSTRES")
    axes[0, 1].text(0.5, 0.5, "Seed 3 daily soil-water state\nnot recorded in 004_05",
                    transform=axes[0, 1].transAxes, ha="center", va="center", fontsize=9, color="#555555")
    for ax, field in ((axes[2, 0], "irrigation_action_mm"),
                      (axes[2, 1], "fertilizer_action_kgN_ha")):
        event = ppo.loc[pd.to_numeric(ppo[field]).gt(0)]
        if not event.empty:
            ax.stem(event.dap, event[field], linefmt=color, markerfmt="o", basefmt=" ",
                    label="Random-weather PPO seed 3")

    last = ppo.groupby("ppo_seed").timestep.idxmax()
    reward_step = -1.1 * ppo.irrigation_action_mm.astype(float) - 1.58 * ppo.fertilizer_action_kgN_ha.astype(float)
    reward_step.loc[last] += 0.158 * float(episode["yield"])
    axes[3, 1].plot(ppo.dap, reward_step.cumsum(), color=color, lw=1.5,
                    label="Random-weather PPO seed 3")
    axes[3, 0].text(0.5, 0.5, "Seed 3 daily grain/biomass state\nnot recorded; terminal yield is in summary",
                    transform=axes[3, 0].transAxes, ha="center", va="center", fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})

    panels = ((axes[0, 1], "Soil water (baseline only; PPO state unavailable)"),
              (axes[1, 0], "Water status (baseline WSPD; PPO SWFAC)"),
              (axes[1, 1], "Nitrogen stress index"), (axes[2, 0], "Irrigation events"),
              (axes[2, 1], "Nitrogen application events"),
              (axes[3, 0], "Grain and biomass trajectories (baseline only)"),
              (axes[3, 1], "Common cumulative reward"))
    for ax, title in panels:
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[1, 1].set_ylabel("NSTRES / NSTD")
    axes[2, 0].set_ylabel("mm/event")
    axes[2, 1].set_ylabel("kg/ha/event")
    axes[3, 0].set_ylabel("kg/ha")
    axes[3, 1].set_ylabel("common units")
    for ax, twin in ((axes[0, 1], None), (axes[1, 0], None)):
        ax.legend(fontsize=7, ncol=2)
    for ax in (axes[2, 0], axes[2, 1], axes[3, 1], axes[1, 1]):
        ax.legend(fontsize=7, ncol=2)
    for ax in axes[3]:
        ax.set_xlabel("DAP")
    fig.suptitle(f"YCA{year} five-scenario daily process | Random-weather PPO seed 3",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGURES / f"004_05_seed3_yca{year}_five_scenario_daily.png", dpi=220)
    plt.close(fig)


def draw_grouped_bars(metrics: pd.DataFrame, fields: list[tuple[str, str]], filename: str,
                      wp_note: bool = False) -> None:
    x = np.arange(len(YEARS))
    width = 0.16
    fig, axes = plt.subplots(len(fields), 1, figsize=(16, 4 * len(fields)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (field, ylabel) in zip(axes, fields):
        for idx, (scenario, label, color) in enumerate(zip(SCENARIOS, LABELS, COLORS)):
            rows = metrics.loc[metrics.scenario.eq(scenario)].set_index("year").reindex(YEARS)
            values = pd.to_numeric(rows[field], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(values).any():
                ax.bar(x + (idx - 2) * width, values, width, label=label, color=color)
        if field == "wp_et" and wp_note:
            ax.legend(handles=[Patch(color=COLORS[i], label=LABELS[i]) for i in range(4)] +
                      [Patch(facecolor="none", edgecolor=COLORS[-1], label="Seed 3: ETCP unavailable")],
                      ncol=3, fontsize=8)
            ax.text(0.99, 0.94, "PPO seed 3 WP_ET unavailable: no matching ETCP replay",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9)
        else:
            ax.legend(ncol=3, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x, YEARS)
    axes[-1].set_xlabel("Validation year")
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=200)
    plt.close(fig)


def plot_full_05503_suite(ep: pd.DataFrame, metrics: pd.DataFrame,
                          baseline_daily: pd.DataFrame, ppo_steps: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        row = ep.loc[ep.year.eq(year)].iloc[0]
        plot_daily_year(baseline_daily, row, ppo_steps, year)
    # Add explicit unavailable metric columns for the same three-panel structure as 055_03.
    metrics = metrics.copy()
    metrics["wp_et"] = np.nan
    draw_grouped_bars(metrics, [("yield", "grain_yield_kg_ha"), ("wp_et", "WP_ET_kg_m3"),
                                ("pfp_n", "PFP_N_kg_kg")],
                      "004_05_seed3_five_scenario_metrics.png", wp_note=True)
    draw_grouped_bars(metrics, [("total_irrigation", "actual_irrigation_mm"),
                                ("total_fertilizer", "actual_nitrogen_kg_ha")],
                      "004_05_seed3_five_scenario_management.png")
    draw_grouped_bars(metrics, [("common_reward_05503", "unified_reward (055_03 common formula)")],
                      "004_05_seed3_five_scenario_reward.png")


def plot_metrics(metrics: pd.DataFrame) -> None:
    fields = [
        ("yield", "Grain yield (kg/ha)"),
        ("common_reward_05503", "Common reward (055_03 formula)"),
        ("total_irrigation", "Total irrigation (mm)"),
        ("total_fertilizer", "Total nitrogen (kg N/ha)"),
        ("pfp_n", "PFP_N (kg grain/kg N)"),
    ]
    fig, axes = plt.subplots(5, 1, figsize=(16, 17), sharex=True)
    x = np.arange(len(YEARS))
    for ax, (field, ylabel) in zip(axes, fields):
        for scenario, label, color, style in zip(SCENARIOS, LABELS, COLORS, STYLES):
            rows = metrics.loc[metrics.scenario.eq(scenario)].set_index("year").reindex(YEARS)
            y = pd.to_numeric(rows[field], errors="coerce").to_numpy(dtype=float)
            ax.plot(x, y, color=color, ls=style, lw=2.2 if scenario.endswith("seed3") else 1.4,
                    marker="o", ms=4.2 if scenario.endswith("seed3") else 3.2,
                    label=label, zorder=4 if scenario.endswith("seed3") else 2)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(ncol=3, fontsize=8, frameon=False, loc="best")
    axes[-1].set_xticks(x, YEARS)
    axes[-1].set_xlabel("Validation year")
    fig.suptitle("YCA lowIC | 004_05 selected Random-weather PPO seed 3 vs four baselines",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(OUT / "seed3_metrics_vs_four_baselines.png", dpi=200)
    plt.close(fig)


def plot_actions(events: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 7.5))
    y_positions = {year: idx for idx, year in enumerate(YEARS)}
    for _, row in events.iterrows():
        y = y_positions[int(row.year)]
        if row.irrigation_action_mm > 0:
            ax.scatter(row.dap, y - 0.11, s=50, marker="o", color="#2878B5",
                       edgecolor="white", linewidth=0.5, zorder=3)
            ax.annotate(f"I {row.irrigation_action_mm:g}", (row.dap, y - 0.11),
                        xytext=(-28 if row.dap < 1 else -24, -10), textcoords="offset points",
                        fontsize=7, color="#22577A", ha="right")
        if row.fertilizer_action_kgN_ha > 0:
            ax.scatter(row.dap, y + 0.11, s=56, marker="s", color="#D9822B",
                       edgecolor="white", linewidth=0.5, zorder=3)
            ax.annotate(f"N {row.fertilizer_action_kgN_ha:g}", (row.dap, y + 0.11),
                        xytext=(6, 4), textcoords="offset points", fontsize=7,
                        color="#9C5700", ha="left")
    ax.set_yticks(range(len(YEARS)), YEARS)
    ax.invert_yaxis()
    ax.set_xlabel("Days after planting (DAP)")
    ax.set_ylabel("Observed weather year")
    ax.set_title("Actual seed 3 management actions by year", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", color="#2878B5", label="Irrigation (mm)"),
        plt.Line2D([], [], marker="s", ls="", color="#D9822B", label="Nitrogen (kg/ha)"),
    ], frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "seed3_management_actions_by_year.png", dpi=200)
    plt.close(fig)


def plot_management(ep: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    x = np.arange(len(YEARS))
    fields = [
        ("total_irrigation", "Irrigation (mm/season)"),
        ("total_fertilizer", "Nitrogen (kg/ha/season)"),
        ("action_events", "Positive action days per season"),
    ]
    for ax, (field, label) in zip(axes, fields):
        ax.bar(x, ep.set_index("year").reindex(YEARS)[field], width=0.62,
               color="#2A9D55", edgecolor="#1E6E3B", linewidth=0.6)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.22)
    axes[-1].set_xticks(x, YEARS)
    axes[-1].set_xlabel("Validation year")
    fig.suptitle("Selected seed 3 | yearly management profile", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "seed3_yearly_management_profile.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ep, metrics, events, baseline_daily, ppo_steps = load_data()
    metrics.to_csv(OUT / "seed3_yearly_metrics_vs_four_baselines.csv", index=False)
    events.to_csv(OUT / "seed3_observed_management_events.csv", index=False)
    plot_full_05503_suite(ep, metrics, baseline_daily, ppo_steps)
    plot_metrics(metrics)
    plot_actions(events)
    plot_management(ep)

    selection = pd.read_csv(PPO_ROOT / "all_evaluation_episode_level_0_7.csv")
    selection = selection.loc[
        selection.training_regime.eq("RANDOM_WEATHER_WGEN")
        & selection.evaluation_weather_type.isin(["observed_weather", "heldout_wgen"])
    ]
    selection["reward"] = pd.to_numeric(selection.reward, errors="coerce")
    rank = selection.groupby(["ppo_seed", "evaluation_weather_type"]).reward.mean().unstack()
    rank["equal_domain_canonical_reward"] = rank[["observed_weather", "heldout_wgen"]].mean(axis=1)
    rank = rank.sort_values("equal_domain_canonical_reward", ascending=False).reset_index()
    rank.to_csv(OUT / "seed_selection_equal_domain_reward_ranking.csv", index=False)
    manifest = {
        "selected_seed": SEED,
        "selection_rule": "Highest per-seed canonical reward after giving observed-weather and heldout-WGEN domain means equal weight; descriptive post-hoc selection, not confirmatory evidence.",
        "selection_basis": "results/yc_random_weather_ppo/004_05/all_evaluation_episode_level_0_7.csv",
        "style_reference": "benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/figures and src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py",
        "comparison_baseline_source": str(BASE_SUMMARY.relative_to(ROOT)),
        "episode_source": str(EPISODES.relative_to(ROOT)),
        "management_step_source": str(STEPS.relative_to(ROOT)),
        "years": YEARS,
        "WP_ET": "unavailable for this PPO seed: no matching 004_05 ETCP replay",
        "no_retraining_or_DSSAT_replay": True,
        "common_reward_formula": "0.158*yield - 1.1*irrigation - 1.58*nitrogen (055_03 comparison formula)",
        "canonical_reward_kept_separate": True,
    }
    (OUT / "seed3_diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(OUT), "selected_seed": SEED,
        "equal_domain_canonical_reward": float(rank.loc[rank.ppo_seed.eq(SEED), "equal_domain_canonical_reward"].iloc[0]),
        "observed_mean_yield": float(ep["yield"].mean()),
        "observed_mean_irrigation": float(ep.total_irrigation.mean()),
        "observed_mean_nitrogen": float(ep.total_fertilizer.mean()),
        "observed_mean_action_days": float(ep.action_events.mean()),
        "action_records": len(events), "figures": 3,
    }))


if __name__ == "__main__":
    main()
