"""Render 004_05 observed-weather results in the 055_03 YCA figure layout.

This is a read-only renderer: it uses the saved episode and step tables, never
replays DSSAT or loads PPO checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "results/yc_random_weather_ppo/004_05"
DEFAULT_OUT = Path(__file__).resolve().parent
EPISODES = SOURCE / "all_evaluation_episode_level_0_7.csv"
STEPS = SOURCE / "evaluation_step_level_all_models.csv"
REGIMES = ("HISTORICAL_WEATHER", "RANDOM_WEATHER_WGEN")
LABELS = {REGIMES[0]: "Historical PPO", REGIMES[1]: "Random-weather PPO"}
COLORS = {REGIMES[0]: "#2A9D55", REGIMES[1]: "#7E63B6"}
STYLES = {REGIMES[0]: "-", REGIMES[1]: ":"}
YEARS = tuple(range(2014, 2024))
STEP_COLUMNS = [
    "training_regime", "ppo_seed", "evaluation_weather_type", "evaluation_weather_year",
    "timestep", "dap", "swfac", "nstres", "topwt", "grnwt", "RAIN",
    "TMAX", "TMIN", "irrigation_action_mm", "fertilizer_action_kgN_ha",
    "instant_reward", "post_step_state_json",
]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    episodes = pd.read_csv(EPISODES)
    episodes = episodes.loc[episodes.evaluation_weather_type.eq("observed_weather")].copy()
    key = ["training_regime", "ppo_seed", "evaluation_weather_year"]
    if len(episodes) != 160 or episodes.duplicated(key).any():
        raise ValueError("Expected exactly 160 unique observed-weather episodes")
    expected = pd.MultiIndex.from_product([REGIMES, range(8), YEARS], names=key)
    if not expected.difference(pd.MultiIndex.from_frame(episodes[key])).empty:
        raise ValueError("Missing a regime/seed/year episode")

    chunks = []
    for chunk in pd.read_csv(STEPS, usecols=STEP_COLUMNS, chunksize=10_000):
        observed = chunk.loc[chunk.evaluation_weather_type.eq("observed_weather")].copy()
        if not observed.empty:
            chunks.append(observed)
    steps = pd.concat(chunks, ignore_index=True)
    if len(steps.groupby(key)) != 160 or steps.duplicated(key + ["timestep"]).any():
        raise ValueError("Daily trace is incomplete or has duplicate time steps")
    steps["day"] = steps.timestep.astype(int) - 5
    if not np.allclose(steps.loc[steps.dap.gt(0), "day"], steps.loc[steps.dap.gt(0), "dap"]):
        raise ValueError("DAP and time-step alignment changed")

    totals = steps.groupby(key, as_index=False).agg(
        step_reward=("instant_reward", "sum"),
        step_irrigation=("irrigation_action_mm", "sum"),
        step_nitrogen=("fertilizer_action_kgN_ha", "sum"),
    )
    joined = totals.merge(episodes, on=key, validate="one_to_one")
    for trace_col, episode_col in (
        ("step_reward", "reward"),
        ("step_irrigation", "total_irrigation"),
        ("step_nitrogen", "total_fertilizer"),
    ):
        if not np.allclose(joined[trace_col], joined[episode_col], atol=1e-6):
            raise ValueError(f"Step/episode mismatch: {trace_col}")

    complete = steps.groupby(["training_regime", "ppo_seed"])[
        ["grnwt", "topwt", "RAIN", "TMAX", "TMIN", "post_step_state_json"]
    ].apply(lambda frame: frame.notna().all().all())
    available = [int(seed) for seed in range(8) if all(bool(complete.loc[(r, seed)]) for r in REGIMES)]
    if available != [0, 1, 2]:
        raise ValueError(f"Unexpected state-trajectory availability: {available}")
    steps["soil_water_top_m3_m3"] = np.nan
    mask = steps.ppo_seed.isin(available)
    steps.loc[mask, "soil_water_top_m3_m3"] = steps.loc[mask, "post_step_state_json"].map(
        lambda value: float(json.loads(value)["sw"])
    )
    endpoint = steps.loc[mask].sort_values("timestep").groupby(key, as_index=False).tail(1)
    endpoint = endpoint.merge(episodes[key + ["yield"]], on=key, validate="one_to_one")
    if not np.allclose(endpoint.grnwt, endpoint["yield"], atol=1e-6):
        raise ValueError("Available grain trajectories do not close to episode yield")

    weather = steps.loc[mask].groupby(["evaluation_weather_year", "timestep"])[["RAIN", "TMAX", "TMIN"]]
    if ((weather.max() - weather.min()).abs() > 1e-3).any().any():
        raise ValueError("Observed weather differs between complete traces")
    return episodes, steps


def group_daily(sub: pd.DataFrame, field: str) -> pd.DataFrame:
    return sub.groupby("day", as_index=False)[field].mean().sort_values("day")


def draw_mean_and_seeds(ax: plt.Axes, sub: pd.DataFrame, field: str, color: str,
                        style: str, label: str, seed_lines: bool = True) -> None:
    if seed_lines:
        for _, one in sub.groupby("ppo_seed"):
            ax.plot(one.day, one[field], color=color, ls=style, lw=0.55, alpha=0.18)
    mean = group_daily(sub, field)
    ax.plot(mean.day, mean[field], color=color, ls=style, lw=1.55, label=label)


def plot_year(steps: pd.DataFrame, year: int, out: Path) -> None:
    daily = steps.loc[steps.evaluation_weather_year.eq(year)].copy()
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    weather = daily.loc[daily.training_regime.eq(REGIMES[0]) & daily.ppo_seed.eq(0)].sort_values("day")
    axes[0, 0].bar(weather.day, weather.RAIN, color="#80A9C7", alpha=0.85)
    temp = axes[0, 0].twinx()
    temp.plot(weather.day, weather.TMAX, color="#C23B32", lw=1.25)
    temp.plot(weather.day, weather.TMIN, color="#666666", lw=1.1, ls="--")
    axes[0, 0].set_title("Weather")
    axes[0, 0].set_ylabel("Rain (mm)")
    temp.set_ylabel("Temperature (C)")

    for regime in REGIMES:
        sub = daily.loc[daily.training_regime.eq(regime)].sort_values(["ppo_seed", "timestep"])
        three = sub.loc[sub.ppo_seed.lt(3)]
        color, style, label = COLORS[regime], STYLES[regime], LABELS[regime]
        draw_mean_and_seeds(axes[0, 1], three, "soil_water_top_m3_m3", color, style, label)
        draw_mean_and_seeds(axes[1, 0], sub, "swfac", color, style, label)
        draw_mean_and_seeds(axes[1, 1], sub, "nstres", color, style, label)

        for ax, field in ((axes[2, 0], "irrigation_action_mm"),
                          (axes[2, 1], "fertilizer_action_kgN_ha")):
            event = group_daily(sub, field)
            event = event.loc[event[field].gt(0)]
            if not event.empty:
                shift = -0.22 if regime == REGIMES[0] else 0.22
                ax.stem(event.day + shift, event[field], linefmt=color,
                        markerfmt="o", basefmt=" ", label=label)

        draw_mean_and_seeds(axes[3, 0], three, "grnwt", color, style, f"{label} grain")
        biomass = group_daily(three, "topwt")
        axes[3, 0].plot(biomass.day, biomass.topwt, color=color, ls=style, lw=0.85, alpha=0.4)
        sub = sub.copy()
        sub["cumulative_reward"] = sub.groupby("ppo_seed").instant_reward.cumsum()
        draw_mean_and_seeds(axes[3, 1], sub, "cumulative_reward", color, style, label)

    panels = (
        (axes[0, 1], "Top-layer soil water (seeds 0-2)", "SW (m3/m3)"),
        (axes[1, 0], "Water stress factor (all 8 seeds)", "SWFAC"),
        (axes[1, 1], "Nitrogen stress state (all 8 seeds)", "NSTRES"),
        (axes[2, 0], "Irrigation events (8-seed mean)", "mm/seed/day"),
        (axes[2, 1], "Nitrogen application events (8-seed mean)", "kg/ha/seed/day"),
        (axes[3, 0], "Grain and biomass (seeds 0-2)", "kg/ha"),
        (axes[3, 1], "Cumulative reward (all 8 seeds)", "reward units"),
    )
    for ax, title, ylabel in panels:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    for ax in (axes[0, 1], axes[2, 0], axes[2, 1], axes[3, 1]):
        ax.legend(fontsize=7, ncol=2)
    for ax in axes[3]:
        ax.set_xlabel("DAP (negative = pre-plant days)")
    fig.suptitle(f"YCA{year} historical vs random-weather PPO daily process",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out / f"004_05_yca{year}_two_ppo_daily.png", dpi=220)
    plt.close(fig)


def add_summary_columns(episodes: pd.DataFrame) -> pd.DataFrame:
    summary = episodes.groupby(["evaluation_weather_year", "training_regime"], as_index=False).agg(
        yield_mean=("yield", "mean"), yield_sd=("yield", "std"),
        reward_mean=("reward", "mean"), reward_sd=("reward", "std"),
        irrigation_mean=("total_irrigation", "mean"), irrigation_sd=("total_irrigation", "std"),
        nitrogen_mean=("total_fertilizer", "mean"), nitrogen_sd=("total_fertilizer", "std"),
        irrigation_events_mean=("irrigation_event_count", "mean"),
        nitrogen_events_mean=("fertilizer_event_count", "mean"),
        seed_count=("ppo_seed", "nunique"),
    )
    summary["pooled_pfp_n"] = summary.yield_mean / summary.nitrogen_mean.replace(0, np.nan)
    if not summary.seed_count.eq(8).all():
        raise ValueError("All annual bars require eight seeds")
    return summary


def draw_grouped(summary: pd.DataFrame, metrics: list[tuple[str, str, str | None]], path: Path) -> None:
    x = np.arange(len(YEARS))
    width = 0.32
    fig, axes = plt.subplots(len(metrics), 1, figsize=(16, 4 * len(metrics)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (field, label, sd_field) in zip(axes, metrics):
        for idx, regime in enumerate(REGIMES):
            rows = summary.loc[summary.training_regime.eq(regime)].set_index("evaluation_weather_year").loc[list(YEARS)]
            ax.bar(x + (idx - 0.5) * width, rows[field].to_numpy(), width,
                   yerr=rows[sd_field].to_numpy() if sd_field else None,
                   error_kw={"lw": 0.8, "capsize": 2},
                   color=COLORS[regime], label=LABELS[regime])
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(YEARS)
    axes[-1].set_xlabel("Validation year")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_paired(episodes: pd.DataFrame, out: Path) -> pd.DataFrame:
    fields = (
        ("reward", "Reward difference"),
        ("yield", "Yield difference (kg/ha)"),
        ("total_irrigation", "Irrigation difference (mm)"),
        ("total_fertilizer", "Nitrogen difference (kg/ha)"),
    )
    key = ["ppo_seed", "evaluation_weather_year"]
    h = episodes.loc[episodes.training_regime.eq(REGIMES[0])].set_index(key)
    w = episodes.loc[episodes.training_regime.eq(REGIMES[1])].set_index(key)
    if not h.index.equals(w.index):
        h, w = h.sort_index(), w.sort_index()
    paired = pd.DataFrame(index=h.index)
    for field, _ in fields:
        paired[field + "_difference"] = w[field] - h[field]
    paired = paired.reset_index().sort_values(key)
    summary = paired.groupby("evaluation_weather_year").agg(
        **{f"{field}_difference_{stat}": (f"{field}_difference", func)
           for field, _ in fields
           for stat, func in (("mean", "mean"), ("std", "std"),
                              ("positive_pairs", lambda s: int(s.gt(0).sum())))}
    ).reset_index()
    x = np.arange(len(YEARS))
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
    for ax, (field, label) in zip(axes, fields):
        means = summary[field + "_difference_mean"].to_numpy()
        colors = [COLORS[REGIMES[0]] if value > 0 else "#C44E52" for value in means]
        ax.bar(x, means, width=0.52, color=colors)
        for seed in range(8):
            values = paired.loc[paired.ppo_seed.eq(seed)].set_index("evaluation_weather_year").loc[
                list(YEARS), field + "_difference"
            ].to_numpy()
            ax.scatter(x + (seed - 3.5) * 0.065, values, s=11, color="#555555", alpha=0.4, zorder=3)
        ax.axhline(0, color="#555555", lw=1.1)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(YEARS)
    axes[-1].set_xlabel("Validation year")
    fig.tight_layout()
    fig.savefig(out / "004_05_two_ppo_paired_by_year.png", dpi=200)
    plt.close(fig)
    paired.to_csv(out / "004_05_paired_seed_year.csv", index=False)
    summary.to_csv(out / "004_05_paired_by_year.csv", index=False)
    return summary


def plot_seed_robustness(episodes: pd.DataFrame, out: Path) -> None:
    per_seed = episodes.groupby(["ppo_seed", "training_regime"], as_index=False)[
        ["yield", "reward", "total_irrigation", "total_fertilizer"]
    ].mean()
    metrics = (
        ("reward", "Mean reward"), ("yield", "Mean yield (kg/ha)"),
        ("total_irrigation", "Mean irrigation (mm)"),
        ("total_fertilizer", "Mean nitrogen (kg/ha)"),
    )
    x = np.arange(8)
    width = 0.32
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
    for ax, (field, label) in zip(axes, metrics):
        for idx, regime in enumerate(REGIMES):
            values = per_seed.loc[per_seed.training_regime.eq(regime)].set_index("ppo_seed").loc[range(8), field]
            ax.bar(x + (idx - 0.5) * width, values, width,
                   color=COLORS[regime], label=LABELS[regime])
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(range(8))
    axes[-1].set_xlabel("Paired PPO seed")
    fig.tight_layout()
    fig.savefig(out / "004_05_two_ppo_seed_robustness.png", dpi=200)
    plt.close(fig)
    per_seed.to_csv(out / "004_05_observed_seed_means.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    episodes, steps = load_inputs()
    summary = add_summary_columns(episodes)
    for year in YEARS:
        plot_year(steps, year, out)
    draw_grouped(summary, [
        ("yield_mean", "grain_yield_kg_ha", "yield_sd"),
        ("reward_mean", "episode_reward", "reward_sd"),
        ("pooled_pfp_n", "PFP_N_kg_kg (pooled)", None),
    ], out / "004_05_two_ppo_metrics.png")
    draw_grouped(summary, [
        ("irrigation_mean", "actual_irrigation_mm", "irrigation_sd"),
        ("nitrogen_mean", "actual_nitrogen_kg_ha", "nitrogen_sd"),
    ], out / "004_05_two_ppo_management.png")
    draw_grouped(summary, [
        ("irrigation_events_mean", "irrigation_events_per_season", None),
        ("nitrogen_events_mean", "nitrogen_events_per_season", None),
    ], out / "004_05_two_ppo_management_events.png")
    summary.to_csv(out / "004_05_observed_year_summary.csv", index=False)
    paired = plot_paired(episodes, out)
    plot_seed_robustness(episodes, out)
    manifest = {
        "style_script": "src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py",
        "input_episode": str(EPISODES.relative_to(ROOT)),
        "input_step": str(STEPS.relative_to(ROOT)),
        "observed_episodes": len(episodes),
        "observed_step_rows": len(steps),
        "complete_daily_state_seeds": [0, 1, 2],
        "all_seed_daily_fields": ["swfac", "nstres", "irrigation_action_mm", "fertilizer_action_kgN_ha", "instant_reward"],
        "years": list(YEARS),
        "reward_mean_difference_by_year": paired.set_index("evaluation_weather_year")["reward_difference_mean"].to_dict(),
    }
    (out / "004_05_055_03_style_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(out), "png_count": len(list(out.glob("*.png"))),
                      "observed_episodes": len(episodes), "step_rows": len(steps)}))


if __name__ == "__main__":
    main()
