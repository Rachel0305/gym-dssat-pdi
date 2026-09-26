"""Five-scenario 004_05 figures using the established 055_03 YCA layout."""

from __future__ import annotations

import argparse
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
DAILY_CSV = OLD_ROOT / "tables/055_03_yca_five_scenario_daily.csv"
SUMMARY_CSV = OLD_ROOT / "tables/055_03_yca_five_scenario_season_summary.csv"
EPISODES_CSV = PPO_ROOT / "all_evaluation_episode_level_0_7.csv"
STEPS_CSV = PPO_ROOT / "evaluation_step_level_all_models.csv"
YEARS = tuple(range(2014, 2024))
SCENARIOS = ("null", "recorded_farmer_template", "dssat_auto_external_n",
             "official_extension_expert", "random_weather_ppo")
LABELS = {
    "null": "Null",
    "recorded_farmer_template": "Recorded template",
    "dssat_auto_external_n": "DSSAT auto + external N",
    "official_extension_expert": "Official expert",
    "random_weather_ppo": "Random-weather PPO",
}
COLORS = {
    "null": "#555555",
    "recorded_farmer_template": "#C44E52",
    "dssat_auto_external_n": "#D8A305",
    "official_extension_expert": "#7E63B6",
    "random_weather_ppo": "#2A9D55",
}
STYLES = {
    "null": "-", "recorded_farmer_template": "--",
    "dssat_auto_external_n": "-.", "official_extension_expert": ":",
    "random_weather_ppo": "-",
}
STEP_COLUMNS = [
    "training_regime", "ppo_seed", "evaluation_weather_type", "evaluation_weather_year",
    "timestep", "doy", "dap", "swfac", "nstres", "topwt", "grnwt", "RAIN",
    "TMAX", "TMIN", "irrigation_action_mm", "fertilizer_action_kgN_ha",
    "instant_reward", "post_step_state_json",
]


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Keep the literal "null" scenario; pandas otherwise reads it as missing.
    baseline_daily = pd.read_csv(DAILY_CSV, keep_default_na=False)
    baseline_summary = pd.read_csv(SUMMARY_CSV, keep_default_na=False)
    baseline_daily = baseline_daily.loc[baseline_daily.scenario.isin(SCENARIOS[:-1])].copy()
    baseline_summary = baseline_summary.loc[baseline_summary.scenario.isin(SCENARIOS[:-1])].copy()
    if set(baseline_daily.scenario.unique()) != set(SCENARIOS[:-1]):
        raise ValueError("The 055_03 daily table does not contain all four baselines")
    if len(baseline_summary) != 40 or baseline_summary.groupby("scenario").year.nunique().ne(10).any():
        raise ValueError("Expected four complete baseline summaries for 2014-2023")

    episodes = pd.read_csv(EPISODES_CSV)
    episodes = episodes.loc[
        episodes.evaluation_weather_type.eq("observed_weather")
        & episodes.training_regime.eq("RANDOM_WEATHER_WGEN")
    ].copy()
    key = ["ppo_seed", "evaluation_weather_year"]
    if len(episodes) != 80 or episodes.duplicated(key).any():
        raise ValueError("Expected 80 unique random-weather PPO observed episodes")
    expected = pd.MultiIndex.from_product([range(8), YEARS], names=key)
    if not expected.difference(pd.MultiIndex.from_frame(episodes[key])).empty:
        raise ValueError("Random-weather PPO observed episode coverage is incomplete")

    pieces = []
    for chunk in pd.read_csv(STEPS_CSV, usecols=STEP_COLUMNS, chunksize=10_000):
        keep = chunk.loc[
            chunk.evaluation_weather_type.eq("observed_weather")
            & chunk.training_regime.eq("RANDOM_WEATHER_WGEN")
        ].copy()
        if not keep.empty:
            pieces.append(keep)
    ppo_daily = pd.concat(pieces, ignore_index=True)
    if ppo_daily.duplicated(key + ["timestep"]).any():
        raise ValueError("Duplicate PPO time steps found")
    state_fields = ["grnwt", "topwt", "post_step_state_json"]
    complete_state = ppo_daily.groupby("ppo_seed")[state_fields].apply(
        lambda frame: frame.notna().all().all()
    )
    if [int(seed) for seed in range(8) if bool(complete_state.loc[seed])] != [0, 1, 2]:
        raise ValueError("Expected complete soil/grain/biomass traces only for PPO seeds 0-2")
    if ppo_daily[["swfac", "nstres", "irrigation_action_mm",
                  "fertilizer_action_kgN_ha", "instant_reward"]].isna().any().any():
        raise ValueError("A daily variable required for all-seed means contains missing values")
    traces = ppo_daily.groupby(key, as_index=False).agg(
        reward_sum=("instant_reward", "sum"),
        irrigation_sum=("irrigation_action_mm", "sum"),
        nitrogen_sum=("fertilizer_action_kgN_ha", "sum"),
    )
    check = traces.merge(episodes, on=key, validate="one_to_one")
    for found, saved in (("reward_sum", "reward"), ("irrigation_sum", "total_irrigation"),
                         ("nitrogen_sum", "total_fertilizer")):
        if not np.allclose(check[found], check[saved], atol=1e-6):
            raise ValueError(f"PPO daily and episode values do not reconcile: {saved}")

    for col in ["yield", "total_irrigation", "total_fertilizer"]:
        episodes[col] = pd.to_numeric(episodes[col], errors="coerce")
    episodes["unified_reward"] = (
        0.158 * episodes["yield"] - 1.1 * episodes.total_irrigation
        - 1.58 * episodes.total_fertilizer
    )
    episodes["PFP_N_kg_kg"] = episodes["yield"] / episodes.total_fertilizer.replace(0, np.nan)

    # Check that the baseline and new-policy observed evaluations use the same
    # historical weather by day of year, with only small WTH rounding drift.
    weather_rows = ppo_daily.loc[ppo_daily.ppo_seed.eq(0)]
    null_weather = baseline_daily.loc[baseline_daily.scenario.eq("null")]
    weather_max_delta = 0.0
    for year in YEARS:
        old = null_weather.loc[null_weather.year.astype(int).eq(year)].rename(
            columns={"rainfall_mm": "rain_old", "tmax_c": "tmax_old", "tmin_c": "tmin_old"}
        )
        new = weather_rows.loc[weather_rows.evaluation_weather_year.eq(year)].rename(
            columns={"RAIN": "rain_new", "TMAX": "tmax_new", "TMIN": "tmin_new"}
        )
        joined = old.merge(new, on="doy", validate="one_to_one")
        if len(joined) < 80:
            raise ValueError(f"Insufficient weather-date overlap in {year}")
        for a, b in (("rain_old", "rain_new"), ("tmax_old", "tmax_new"), ("tmin_old", "tmin_new")):
            delta = (pd.to_numeric(joined[a]) - pd.to_numeric(joined[b])).abs().max()
            weather_max_delta = max(weather_max_delta, float(delta))
            if a == "rain_old" and delta > 1e-3:
                raise ValueError(f"Rainfall differs by DOY in {year}")
            if a != "rain_old" and delta > 0.1:
                raise ValueError(f"Temperature differs by more than WTH rounding in {year}")
    return baseline_daily, baseline_summary, episodes, ppo_daily.assign(weather_max_delta=weather_max_delta)


def ppo_daily_summary(episodes: pd.DataFrame, ppo_daily: pd.DataFrame, year: int) -> pd.DataFrame:
    subset = ppo_daily.loc[ppo_daily.evaluation_weather_year.eq(year)].copy()
    subset["soil_water_top_m3_m3"] = np.nan
    state = subset.loc[subset.ppo_seed.lt(3), "post_step_state_json"]
    subset.loc[state.index, "soil_water_top_m3_m3"] = state.map(
        lambda value: float(json.loads(value)["sw"]) if pd.notna(value) and value else np.nan
    )
    # Mean by DAP for repeated pre-plant days. Management doses are summed by
    # seed/day before taking the across-seed mean, so annual closure is retained.
    action = subset.groupby(["ppo_seed", "dap"], as_index=False).agg(
        irrigation=("irrigation_action_mm", "sum"),
        nitrogen=("fertilizer_action_kgN_ha", "sum"),
    )
    rewards = episodes.loc[episodes.evaluation_weather_year.eq(year)].set_index("ppo_seed")
    last_step = subset.groupby("ppo_seed").timestep.idxmax()
    subset["unified_reward_step"] = -1.1 * subset.irrigation_action_mm - 1.58 * subset.fertilizer_action_kgN_ha
    for idx in last_step:
        seed = int(subset.loc[idx, "ppo_seed"])
        subset.loc[idx, "unified_reward_step"] += 0.158 * float(rewards.loc[seed, "yield"])
    subset = subset.sort_values(["ppo_seed", "timestep"])
    subset["unified_cumulative_reward"] = subset.groupby("ppo_seed").unified_reward_step.cumsum()

    state_mean = subset.groupby("dap", as_index=False).agg(
        swfac=("swfac", "mean"), nstres=("nstres", "mean"),
        topwt=("topwt", "mean"), grnwt=("grnwt", "mean"),
        soil_water_top_m3_m3=("soil_water_top_m3_m3", "mean"),
        unified_cumulative_reward=("unified_cumulative_reward", "mean"),
        state_seed_count=("grnwt", "count"),
    )
    action_mean = action.groupby("dap", as_index=False).agg(
        irrigation=("irrigation", "mean"), nitrogen=("nitrogen", "mean"),
    )
    return state_mean.merge(action_mean, on="dap", how="outer").sort_values("dap")


def plot_year(baseline_daily: pd.DataFrame, episodes: pd.DataFrame,
              ppo_daily: pd.DataFrame, year: int, out: Path) -> None:
    year_base = baseline_daily.loc[baseline_daily.year.astype(int).eq(year)]
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    weather = year_base.loc[year_base.scenario.eq("null")].sort_values("dap")
    axes[0, 0].bar(weather.dap, weather.rainfall_mm.astype(float), color="#80A9C7", alpha=0.85, label="Rain")
    temp = axes[0, 0].twinx()
    temp.plot(weather.dap, weather.tmax_c.astype(float), color="#C23B32", lw=1.25, label="Tmax")
    temp.plot(weather.dap, weather.tmin_c.astype(float), color="#666666", lw=1.1, ls="--", label="Tmin")
    axes[0, 0].set_title("Weather")
    axes[0, 0].set_ylabel("Rain (mm)")
    temp.set_ylabel("Temperature (C)")

    ppo = ppo_daily_summary(episodes, ppo_daily, year)
    for scenario in SCENARIOS[:-1]:
        sub = year_base.loc[year_base.scenario.eq(scenario)].sort_values("dap")
        color, style, label = COLORS[scenario], STYLES[scenario], LABELS[scenario]
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

    ppo_color = COLORS["random_weather_ppo"]
    ppo_style = STYLES["random_weather_ppo"]
    soil_right = axes[0, 1].twinx()
    soil_right.plot(ppo.dap, ppo.soil_water_top_m3_m3, color=ppo_color, ls=ppo_style,
                    lw=1.5, label="Random-weather PPO top-layer SW (seeds 0-2)")
    soil_right.set_ylabel("PPO top-layer SW (m3/m3)")
    axes[0, 1].set_ylabel("Baseline SWTD (mm)")
    water_right = axes[1, 0].twinx()
    water_right.plot(ppo.dap, ppo.swfac, color=ppo_color, ls=ppo_style, lw=1.5,
                     label="Random-weather PPO SWFAC (8-seed mean)")
    water_right.set_ylabel("PPO SWFAC")
    axes[1, 0].set_ylabel("Baseline WSPD")
    axes[1, 1].plot(ppo.dap, ppo.nstres, color=ppo_color, ls=ppo_style, lw=1.5,
                    label="Random-weather PPO NSTRES (8-seed mean)")
    for ax, values, field in ((axes[2, 0], ppo.irrigation, "irrigation"),
                              (axes[2, 1], ppo.nitrogen, "nitrogen")):
        events = ppo.loc[ppo[field].gt(0)]
        if not events.empty:
            ax.stem(events.dap, events[field], linefmt=ppo_color,
                    markerfmt="o", basefmt=" ", label="Random-weather PPO (8-seed mean)")
    axes[3, 0].plot(ppo.dap, ppo.grnwt, color=ppo_color, ls=ppo_style, lw=1.45,
                    label="Random-weather PPO grain (seeds 0-2)")
    axes[3, 0].plot(ppo.dap, ppo.topwt, color=ppo_color, ls=ppo_style, lw=0.85, alpha=0.4)
    axes[3, 1].plot(ppo.dap, ppo.unified_cumulative_reward, color=ppo_color, ls=ppo_style,
                    lw=1.5, label="Random-weather PPO (8-seed mean)")

    panels = (
        (axes[0, 1], "Soil water (PPO top layer uses right axis)"),
        (axes[1, 0], "Water status (baseline WSPD; PPO SWFAC)"),
        (axes[1, 1], "Nitrogen stress index"),
        (axes[2, 0], "Irrigation events"),
        (axes[2, 1], "Nitrogen application events"),
        (axes[3, 0], "Grain and biomass trajectories"),
        (axes[3, 1], "Unified cumulative reward"),
    )
    for ax, title in panels:
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[1, 1].set_ylabel("NSTRES / NSTD")
    axes[2, 0].set_ylabel("mm/event")
    axes[2, 1].set_ylabel("kg/ha/event")
    axes[3, 0].set_ylabel("kg/ha")
    axes[3, 1].set_ylabel("common units")
    for ax in (axes[0, 1], axes[1, 0]):
        twin = soil_right if ax is axes[0, 1] else water_right
        handles, labels = ax.get_legend_handles_labels()
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        ax.legend(handles + twin_handles, labels + twin_labels, fontsize=7, ncol=2)
    for ax in (axes[2, 0], axes[2, 1], axes[3, 1]):
        ax.legend(fontsize=7, ncol=2)
    for ax in axes[3]:
        ax.set_xlabel("DAP")
    fig.suptitle(f"YCA{year} five-scenario daily process", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out / f"004_05_yca{year}_five_scenario_daily.png", dpi=220)
    plt.close(fig)


def make_summary(base: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    base = base.rename(columns={
        "grain_yield_kg_ha": "yield", "WP_ET_kg_m3": "wp_et",
        "PFP_N_kg_kg": "pfp_n", "actual_irrigation_mm": "irrigation",
        "actual_nitrogen_kg_ha": "nitrogen", "unified_reward": "reward",
    }).copy()
    result = []
    for year, frame in episodes.groupby("evaluation_weather_year"):
        result.append({
            "year": int(year), "scenario": "random_weather_ppo",
            "yield": frame["yield"].mean(), "yield_sd": frame["yield"].std(),
            "wp_et": np.nan,
            "pfp_n": frame.PFP_N_kg_kg.mean(), "pfp_n_sd": frame.PFP_N_kg_kg.std(),
            "irrigation": frame.total_irrigation.mean(), "irrigation_sd": frame.total_irrigation.std(),
            "nitrogen": frame.total_fertilizer.mean(), "nitrogen_sd": frame.total_fertilizer.std(),
            "reward": frame.unified_reward.mean(), "reward_sd": frame.unified_reward.std(),
            "seed_count": frame.ppo_seed.nunique(),
            "pfp_seed_count": frame.PFP_N_kg_kg.notna().sum(),
        })
    for _, row in base.iterrows():
        result.append({
            "year": int(row["year"]), "scenario": row["scenario"],
            "yield": float(row["yield"]), "yield_sd": np.nan,
            "wp_et": pd.to_numeric(row["wp_et"], errors="coerce"),
            "pfp_n": pd.to_numeric(row["pfp_n"], errors="coerce"), "pfp_n_sd": np.nan,
            "irrigation": float(row["irrigation"]), "irrigation_sd": np.nan,
            "nitrogen": float(row["nitrogen"]), "nitrogen_sd": np.nan,
            "reward": float(row["reward"]), "reward_sd": np.nan,
            "seed_count": 1, "pfp_seed_count": int(pd.notna(row["pfp_n"])),
        })
    summary = pd.DataFrame(result)
    if len(summary) != 50 or summary.groupby(["year", "scenario"]).size().ne(1).any():
        raise ValueError("Five-scenario summary must have one row per year/scenario")
    return summary


def draw_bars(summary: pd.DataFrame, metrics: list[tuple[str, str]], path: Path,
              wp_et_missing_note: bool = False) -> None:
    x = np.arange(len(YEARS))
    width = 0.16
    fig, axes = plt.subplots(len(metrics), 1, figsize=(16, 4 * len(metrics)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (field, ylabel) in zip(axes, metrics):
        for idx, scenario in enumerate(SCENARIOS):
            rows = summary.loc[summary.scenario.eq(scenario)].set_index("year").reindex(YEARS)
            vals = pd.to_numeric(rows[field], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(vals).any():
                continue
            err = (rows[f"{field}_sd"].to_numpy(dtype=float)
                   if scenario == "random_weather_ppo" and f"{field}_sd" in rows else None)
            ax.bar(x + (idx - 2) * width, vals, width,
                   yerr=err, error_kw={"lw": 0.8, "capsize": 2},
                   label=LABELS[scenario], color=COLORS[scenario])
        if field == "wp_et" and wp_et_missing_note:
            ax.legend(handles=[
                Patch(color=COLORS[s], label=LABELS[s]) for s in SCENARIOS[:-1]
            ] + [Patch(facecolor="none", edgecolor=COLORS["random_weather_ppo"],
                       label="Random-weather PPO: ETCP unavailable")], ncol=3, fontsize=8)
            ax.text(0.99, 0.94, "PPO WP_ET unavailable: no matching ETCP replay",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
        else:
            ax.legend(ncol=3, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(YEARS)
    axes[-1].set_xlabel("Validation year")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    daily, old_summary, episodes, ppo_steps = load_sources()
    summary = make_summary(old_summary, episodes)
    for year in YEARS:
        plot_year(daily, episodes, ppo_steps, year, out)
    draw_bars(summary, [("yield", "grain_yield_kg_ha"), ("wp_et", "WP_ET_kg_m3"),
                        ("pfp_n", "PFP_N_kg_kg")],
              out / "004_05_five_scenario_metrics.png", wp_et_missing_note=True)
    draw_bars(summary, [("irrigation", "actual_irrigation_mm"),
                        ("nitrogen", "actual_nitrogen_kg_ha")],
              out / "004_05_five_scenario_management.png")
    draw_bars(summary, [("reward", "unified_reward (055_03 common formula)")],
              out / "004_05_five_scenario_reward.png")
    summary.to_csv(out / "004_05_five_scenario_season_summary.csv", index=False)
    weather_delta = float(ppo_steps.weather_max_delta.iloc[0])
    manifest = {
        "style_source_figure_daily": str((OLD_ROOT / "figures/055_03_yca2014_five_scenario_daily.png").relative_to(ROOT)),
        "style_source_figure_metrics": str((OLD_ROOT / "figures/055_03_yca_five_scenario_metrics.png").relative_to(ROOT)),
        "style_source_figure_management": str((OLD_ROOT / "figures/055_03_yca_five_scenario_management.png").relative_to(ROOT)),
        "style_source_script": "src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py",
        "baseline_daily_source": str(DAILY_CSV.relative_to(ROOT)),
        "baseline_summary_source": str(SUMMARY_CSV.relative_to(ROOT)),
        "ppo_episode_source": str(EPISODES_CSV.relative_to(ROOT)),
        "ppo_step_source": str(STEPS_CSV.relative_to(ROOT)),
        "scenario_count": 5,
        "years": list(YEARS),
        "new_ppo_seed_count_per_year": 8,
        "complete_new_ppo_soil_water_grain_biomass_daily_seed_count": 3,
        "new_ppo_wp_et": "unavailable: no matching ETCP replay in 004_05 formal outputs",
        "new_ppo_daily_unified_reward": "0.158*yield - 1.1*irrigation - 1.58*nitrogen; matched to 055_03 formula",
        "max_weather_field_absolute_difference": weather_delta,
        "push_status": "not_pushed",
    }
    (out / "004_05_five_scenario_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(out), "png_count": len(list(out.glob("*.png"))),
                      "episodes": len(episodes), "weather_max_delta": weather_delta}))


if __name__ == "__main__":
    main()
