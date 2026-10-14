"""Plot one consistent YC five-scenario figure set for each 75 mm PPO seed."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EPISODE_CSV = ROOT / "results/yc_random_weather_ppo/004_05/all_evaluation_episode_level_0_7.csv"
STEP_CSV = ROOT / "results/yc_random_weather_ppo/004_05/evaluation_step_level_all_models.csv"
REPLAY_CSV = ROOT / "results/yc_random_weather_ppo/004_07_02_seed345_etcp_replay/replay_episode_summary.csv"
BASE_TABLES = ROOT / "benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables"
BASE_SUMMARY = BASE_TABLES / "055_03_yca_five_scenario_season_summary.csv"
BASE_EVENTS = BASE_TABLES / "055_03_yca_five_scenario_management_events.csv"
OUT = ROOT / "results/yc_random_weather_ppo/004_15_75mm_all_seed_five_scenario_figures"

YEARS = list(range(2014, 2024))
SEEDS = list(range(8))
BASE_SCENARIOS = ["null", "recorded_farmer_template", "dssat_auto_external_n", "official_extension_expert"]
SCENARIOS = BASE_SCENARIOS + ["random_weather_ppo"]
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
    "null": "-",
    "recorded_farmer_template": "--",
    "dssat_auto_external_n": "-.",
    "official_extension_expert": ":",
    "random_weather_ppo": "-",
}
COMMON_REWARD_FORMULA = "0.158 * yield - 1.1 * irrigation_mm - 1.58 * nitrogen_kg_ha"


def number(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def read_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_year(value: object) -> int:
    result = number(value)
    if not math.isfinite(result):
        raise ValueError(f"Invalid year value: {value!r}")
    return int(result)


def load_baselines() -> tuple[dict[tuple[str, int], dict[str, float]], dict[str, list[dict[str, float]]]]:
    summary: dict[tuple[str, int], dict[str, float]] = {}
    for row in read_dicts(BASE_SUMMARY):
        scenario = row["scenario"]
        year = normalize_year(row["year"])
        if scenario not in BASE_SCENARIOS or year not in YEARS:
            continue
        key = (scenario, year)
        if key in summary:
            raise ValueError(f"Duplicate baseline summary row: {key}")
        summary[key] = {
            "yield": number(row["grain_yield_kg_ha"]),
            "wp_et": number(row["WP_ET_kg_m3"]),
            "pfp_n": number(row["PFP_N_kg_kg"]),
            "irrigation": number(row["actual_irrigation_mm"]),
            "nitrogen": number(row["actual_nitrogen_kg_ha"]),
            "common_reward": number(row["unified_reward"]),
            "canonical_reward": math.nan,
        }
    expected = {(scenario, year) for scenario in BASE_SCENARIOS for year in YEARS}
    if set(summary) != expected:
        raise ValueError(f"Baseline summary coverage mismatch: missing={sorted(expected - set(summary))}")

    events: dict[str, list[dict[str, float]]] = {scenario: [] for scenario in BASE_SCENARIOS}
    for row in read_dicts(BASE_EVENTS):
        scenario = row["scenario"]
        year = normalize_year(row["year"])
        if scenario not in BASE_SCENARIOS or year not in YEARS:
            continue
        irrigation = number(row["irrigation_executed_mm"], 0.0)
        nitrogen = number(row["nitrogen_executed_kg_ha"], 0.0)
        if irrigation <= 0 and nitrogen <= 0:
            continue
        events[scenario].append({
            "year": year, "dap": number(row["dap"]),
            "irrigation": max(irrigation, 0.0), "nitrogen": max(nitrogen, 0.0),
        })
    return summary, events


def load_ppo_episodes() -> dict[tuple[int, int], dict[str, float]]:
    selected: dict[tuple[int, int], dict[str, float]] = {}
    for row in read_dicts(EPISODE_CSV):
        if row["training_regime"] != "RANDOM_WEATHER_WGEN" or row["evaluation_weather_type"] != "observed_weather":
            continue
        seed = int(number(row["ppo_seed"], -1))
        year = normalize_year(row["evaluation_weather_year"])
        if seed not in SEEDS or year not in YEARS:
            continue
        key = (seed, year)
        if key in selected:
            raise ValueError(f"Duplicate PPO episode row: {key}")
        selected[key] = {
            "yield": number(row["yield"]),
            "irrigation": number(row["total_irrigation"]),
            "nitrogen": number(row["total_fertilizer"]),
            "canonical_reward": number(row["reward"]),
            "episode_return": number(row["episode_return"]),
            "episode_days": number(row["episode_days"]),
            "irrigation_event_count": number(row["irrigation_event_count"]),
            "nitrogen_event_count": number(row["fertilizer_event_count"]),
            "management_event_count": number(row["action_events"]),
        }
    expected = {(seed, year) for seed in SEEDS for year in YEARS}
    if set(selected) != expected:
        raise ValueError(f"PPO episode coverage mismatch: missing={sorted(expected - set(selected))}")
    for metrics in selected.values():
        n = metrics["nitrogen"]
        metrics["pfp_n"] = metrics["yield"] / n if n > 0 else math.nan
        metrics["common_reward"] = (
            0.158 * metrics["yield"] - 1.1 * metrics["irrigation"] - 1.58 * metrics["nitrogen"]
        )
        metrics["wp_et"] = math.nan
    return selected


def add_exact_wp_et(episodes: dict[tuple[int, int], dict[str, float]]) -> list[tuple[int, int]]:
    available: list[tuple[int, int]] = []
    for row in read_dicts(REPLAY_CSV):
        seed = int(number(row["seed"], -1))
        year = normalize_year(row["year"])
        key = (seed, year)
        if key not in episodes:
            continue
        metrics = episodes[key]
        for field, replay_field, tolerance in (
            ("yield", "replay_yield_kg_ha", 1.0),
            ("irrigation", "irrigation_mm", 0.1),
            ("nitrogen", "nitrogen_kg_ha", 0.1),
        ):
            delta = metrics[field] - number(row[replay_field])
            if abs(delta) > tolerance:
                raise ValueError(f"ETCP replay does not match formal {field}, seed/year={key}, delta={delta}")
        et = number(row["etcp_mm"])
        yield_value = number(row["replay_yield_kg_ha"])
        reported_wp = number(row["wp_et_kg_m3"])
        if et <= 0 or not math.isclose(yield_value / (et * 10.0), reported_wp, rel_tol=0, abs_tol=0.01):
            raise ValueError(f"Invalid exact WP_ET replay row: {key}")
        metrics["wp_et"] = reported_wp
        available.append(key)
    return sorted(available)


def load_ppo_events(episodes: dict[tuple[int, int], dict[str, float]]) -> tuple[dict[int, list[dict[str, float]]], list[dict[str, float]]]:
    totals = {
        key: {"irrigation": 0.0, "nitrogen": 0.0, "irrigation_events": 0, "nitrogen_events": 0, "active_days": 0, "steps": 0}
        for key in episodes
    }
    by_dap: dict[tuple[int, int, float], dict[str, float]] = defaultdict(lambda: {"irrigation": 0.0, "nitrogen": 0.0})
    with STEP_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["training_regime"] != "RANDOM_WEATHER_WGEN" or row["evaluation_weather_type"] != "observed_weather":
                continue
            seed = int(number(row["ppo_seed"], -1))
            year = normalize_year(row["evaluation_weather_year"])
            key = (seed, year)
            if key not in totals:
                continue
            irrigation = max(number(row["irrigation_action_mm"], 0.0), 0.0)
            nitrogen = max(number(row["fertilizer_action_kgN_ha"], 0.0), 0.0)
            dap = number(row["dap"])
            totals[key]["steps"] += 1
            totals[key]["irrigation"] += irrigation
            totals[key]["nitrogen"] += nitrogen
            totals[key]["irrigation_events"] += int(irrigation > 0)
            totals[key]["nitrogen_events"] += int(nitrogen > 0)
            totals[key]["active_days"] += int(irrigation > 0 or nitrogen > 0)
            if irrigation > 0 or nitrogen > 0:
                slot = by_dap[(seed, year, dap)]
                slot["irrigation"] += irrigation
                slot["nitrogen"] += nitrogen

    ppo_events: dict[int, list[dict[str, float]]] = {seed: [] for seed in SEEDS}
    closure: list[dict[str, float]] = []
    for key, expected in episodes.items():
        observed = totals[key]
        d_i = observed["irrigation"] - expected["irrigation"]
        d_n = observed["nitrogen"] - expected["nitrogen"]
        d_i_count = observed["irrigation_events"] - expected["irrigation_event_count"]
        d_n_count = observed["nitrogen_events"] - expected["nitrogen_event_count"]
        if abs(d_i) > 0.1 or abs(d_n) > 0.1 or d_i_count != 0 or d_n_count != 0:
            raise ValueError(f"Step-to-episode action closure failed for {key}: dI={d_i}, dN={d_n}, dI_events={d_i_count}, dN_events={d_n_count}")
        closure.append({
            "seed": key[0], "year": key[1], "episode_steps": observed["steps"],
            "irrigation_step_sum_mm": observed["irrigation"], "irrigation_episode_mm": expected["irrigation"],
            "nitrogen_step_sum_kg_ha": observed["nitrogen"], "nitrogen_episode_kg_ha": expected["nitrogen"],
            "irrigation_event_count": observed["irrigation_events"], "nitrogen_event_count": observed["nitrogen_events"],
            "management_action_days": observed["active_days"],
        })
    for (seed, year, dap), action in sorted(by_dap.items()):
        ppo_events[seed].append({"year": year, "dap": dap, **action})
    return ppo_events, closure


def metric_rows(
    baseline: dict[tuple[str, int], dict[str, float]],
    episodes: dict[tuple[int, int], dict[str, float]],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for seed in SEEDS:
        for year in YEARS:
            for scenario in BASE_SCENARIOS:
                rows.append({"seed": seed, "year": year, "scenario": scenario, **baseline[(scenario, year)]})
            rows.append({"seed": seed, "year": year, "scenario": "random_weather_ppo", **episodes[(seed, year)]})
    return rows


def save_metrics_figures(seed: int, frame: pd.DataFrame, out: Path, wp_available: bool) -> None:
    x = np.arange(len(YEARS))
    width = 0.15
    metrics = [("yield", "Grain yield (kg/ha)"), ("wp_et", "WP_ET (kg/m3)"), ("pfp_n", "PFP_N (kg grain/kg N)")]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(16, 12), sharex=True)
    for ax, (metric, label) in zip(axes, metrics):
        for idx, scenario in enumerate(SCENARIOS):
            data = frame[frame.scenario.eq(scenario)].set_index("year").reindex(YEARS)
            values = pd.to_numeric(data[metric], errors="coerce").to_numpy(dtype=float)
            ax.bar(x + (idx - 2) * width, values, width, label=LABELS[scenario], color=COLORS[scenario])
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=3, fontsize=8, frameon=False)
    if not wp_available:
        axes[1].text(0.99, 0.92, "PPO WP_ET unavailable: no exact ETCP replay", transform=axes[1].transAxes,
                     ha="right", va="top", fontsize=9, color="#555555")
    axes[-1].set_xticks(x, YEARS)
    axes[-1].set_xlabel("Validation year")
    fig.suptitle(f"YCA lowIC | 75 mm Random-weather PPO seed {seed} vs four baselines", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out / f"seed{seed}_five_scenario_metrics.png", dpi=200)
    plt.close(fig)


def save_management_figure(seed: int, frame: pd.DataFrame, out: Path) -> None:
    x = np.arange(len(YEARS))
    width = 0.15
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    for ax, (metric, ylabel) in zip(axes, [("irrigation", "Total irrigation (mm)"), ("nitrogen", "Total nitrogen (kg N/ha)")]):
        for idx, scenario in enumerate(SCENARIOS):
            data = frame[frame.scenario.eq(scenario)].set_index("year").reindex(YEARS)
            values = pd.to_numeric(data[metric], errors="coerce").to_numpy(dtype=float)
            ax.bar(x + (idx - 2) * width, values, width, label=LABELS[scenario], color=COLORS[scenario])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=3, fontsize=8, frameon=False)
    axes[-1].set_xticks(x, YEARS)
    axes[-1].set_xlabel("Validation year")
    fig.suptitle(f"YCA lowIC | yearly management totals | PPO seed {seed}", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out / f"seed{seed}_five_scenario_management.png", dpi=200)
    plt.close(fig)


def save_reward_figure(seed: int, frame: pd.DataFrame, out: Path) -> None:
    x = np.arange(len(YEARS))
    width = 0.15
    fig, axes = plt.subplots(2, 1, figsize=(16, 7.5), sharex=True, gridspec_kw={"height_ratios": [1.25, 0.8]})
    for idx, scenario in enumerate(SCENARIOS):
        data = frame[frame.scenario.eq(scenario)].set_index("year").reindex(YEARS)
        values = pd.to_numeric(data["common_reward"], errors="coerce").to_numpy(dtype=float)
        axes[0].bar(x + (idx - 2) * width, values, width, label=LABELS[scenario], color=COLORS[scenario])
    axes[0].set_ylabel("Common reward")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8, frameon=False)
    ppo = frame[frame.scenario.eq("random_weather_ppo")].set_index("year").reindex(YEARS)
    axes[1].plot(x, pd.to_numeric(ppo["canonical_reward"], errors="coerce"), color=COLORS["random_weather_ppo"],
                 lw=2.0, marker="o", ms=4, label="PPO canonical reward (PPO-only scale)")
    axes[1].set_ylabel("Canonical reward")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    axes[-1].set_xticks(x, YEARS)
    axes[-1].set_xlabel("Validation year")
    fig.suptitle(f"YCA lowIC | common reward and PPO canonical reward | seed {seed}", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / f"seed{seed}_five_scenario_reward.png", dpi=200)
    plt.close(fig)


def save_trend_figure(seed: int, frame: pd.DataFrame, out: Path) -> None:
    panels = [
        ("yield", "Grain yield (kg/ha)"),
        ("common_reward", "Common reward (055_03 formula)"),
        ("irrigation", "Total irrigation (mm)"),
        ("nitrogen", "Total nitrogen (kg N/ha)"),
        ("pfp_n", "PFP_N (kg grain/kg N)"),
    ]
    x = np.arange(len(YEARS))
    fig, axes = plt.subplots(len(panels), 1, figsize=(16, 17), sharex=True)
    for ax, (metric, ylabel) in zip(axes, panels):
        for scenario in SCENARIOS:
            data = frame[frame.scenario.eq(scenario)].set_index("year").reindex(YEARS)
            values = pd.to_numeric(data[metric], errors="coerce").to_numpy(dtype=float)
            is_ppo = scenario == "random_weather_ppo"
            label = f"{LABELS[scenario]} (seed {seed})" if is_ppo else LABELS[scenario]
            ax.plot(x, values, color=COLORS[scenario], ls=STYLES[scenario], lw=2.2 if is_ppo else 1.4,
                    marker="o", ms=4.2 if is_ppo else 3.2, label=label, zorder=4 if is_ppo else 2)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(ncol=3, fontsize=8, frameon=False, loc="best")
    axes[-1].set_xticks(x, YEARS)
    axes[-1].set_xlabel("Validation year")
    fig.suptitle(f"YCA lowIC | 75 mm PPO seed {seed} vs four baselines", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out / f"seed{seed}_five_scenario_yearly_trends.png", dpi=200)
    plt.close(fig)


def save_scenario_action_calendar(seed: int, event_map: dict[str, list[dict[str, float]]], out: Path) -> None:
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(17, 7), sharex=True, sharey=True)
    year_pos = {year: i for i, year in enumerate(YEARS)}
    for ax, scenario in zip(axes, SCENARIOS):
        for event in event_map[scenario]:
            y = year_pos[int(event["year"])]
            if event["irrigation"] > 0:
                ax.scatter(event["dap"], y - 0.13, marker="o", s=40, color=COLORS[scenario], edgecolor="white", linewidth=0.45)
            if event["nitrogen"] > 0:
                ax.scatter(event["dap"], y + 0.13, marker="s", s=42, color=COLORS[scenario], edgecolor="white", linewidth=0.45)
        ax.set_title(LABELS[scenario] + (f"\n(seed {seed})" if scenario == "random_weather_ppo" else ""), fontsize=9)
        ax.set_xlim(-5, 125)
        ax.set_xticks([0, 30, 60, 90, 120])
        ax.grid(axis="x", alpha=0.22)
    axes[0].set_yticks(range(len(YEARS)), YEARS)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Observed weather year")
    for ax in axes:
        ax.set_xlabel("DAP")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color="#444444", label="Irrigation event"),
        plt.Line2D([], [], marker="s", ls="", color="#444444", label="Nitrogen event"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(0.99, 1.01))
    fig.suptitle(f"YCA lowIC | five-scenario management timing | PPO seed {seed}", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / f"seed{seed}_five_scenario_action_calendar.png", dpi=200)
    plt.close(fig)


def save_ppo_action_doses(seed: int, events: list[dict[str, float]], out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    year_pos = {year: i for i, year in enumerate(YEARS)}
    for ax, key, title, color, marker in (
        (axes[0], "irrigation", "Irrigation actions (mm/event)", "#2878B5", "o"),
        (axes[1], "nitrogen", "Nitrogen actions (kg N/ha/event)", "#D9822B", "s"),
    ):
        for event in events:
            amount = event[key]
            if amount <= 0:
                continue
            y = year_pos[int(event["year"])]
            ax.scatter(event["dap"], y, marker=marker, s=48, color=color, edgecolor="white", linewidth=0.5, zorder=3)
            ax.annotate(f"{amount:g}", (event["dap"], y), xytext=(4, -8 if key == "irrigation" else 4),
                        textcoords="offset points", fontsize=7, color=color)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
        ax.set_yticks(range(len(YEARS)), YEARS)
        ax.invert_yaxis()
        ax.set_xlim(-5, 125)
        ax.set_xticks([0, 15, 30, 45, 60, 75, 90, 105, 120])
        ax.grid(axis="x", alpha=0.22)
    axes[-1].set_xlabel("Days after planting (DAP)")
    axes[0].set_ylabel("Observed weather year")
    axes[1].set_ylabel("Observed weather year")
    fig.suptitle(f"YCA lowIC | actual PPO management schedule | 75 mm seed {seed}", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / f"seed{seed}_management_action_doses_by_year.png", dpi=200)
    plt.close(fig)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUT}")
    for path in [EPISODE_CSV, STEP_CSV, REPLAY_CSV, BASE_SUMMARY, BASE_EVENTS]:
        if not path.is_file():
            raise FileNotFoundError(path)

    baseline, baseline_events = load_baselines()
    episodes = load_ppo_episodes()
    exact_wp_keys = add_exact_wp_et(episodes)
    ppo_events, closure = load_ppo_events(episodes)
    all_metrics = pd.DataFrame(metric_rows(baseline, episodes))
    OUT.mkdir(parents=True)

    for seed in SEEDS:
        seed_out = OUT / f"seed_{seed}"
        seed_out.mkdir()
        frame = all_metrics[all_metrics.seed.eq(seed)].copy()
        wp_available = all(math.isfinite(episodes[(seed, year)]["wp_et"]) for year in YEARS)
        save_metrics_figures(seed, frame, seed_out, wp_available)
        save_management_figure(seed, frame, seed_out)
        save_reward_figure(seed, frame, seed_out)
        save_trend_figure(seed, frame, seed_out)
        event_map = {scenario: baseline_events[scenario] for scenario in BASE_SCENARIOS}
        event_map["random_weather_ppo"] = ppo_events[seed]
        save_scenario_action_calendar(seed, event_map, seed_out)
        save_ppo_action_doses(seed, ppo_events[seed], seed_out)
        frame.to_csv(seed_out / f"seed{seed}_five_scenario_metrics.csv", index=False)
        pd.DataFrame(ppo_events[seed]).to_csv(seed_out / f"seed{seed}_management_events.csv", index=False)

    all_metrics.to_csv(OUT / "all_seeds_five_scenario_metrics.csv", index=False)
    pd.DataFrame(closure).sort_values(["seed", "year"]).to_csv(OUT / "ppo_action_trace_closure.csv", index=False)
    all_events = []
    for seed in SEEDS:
        for scenario in BASE_SCENARIOS:
            all_events.extend({"seed": seed, "scenario": scenario, **event} for event in baseline_events[scenario])
        all_events.extend({"seed": seed, "scenario": "random_weather_ppo", **event} for event in ppo_events[seed])
    pd.DataFrame(all_events).to_csv(OUT / "all_seeds_five_scenario_management_events.csv", index=False)

    summary_rows = []
    for seed in SEEDS:
        frame = all_metrics[(all_metrics.seed.eq(seed)) & all_metrics.scenario.eq("random_weather_ppo")]
        summary_rows.append({
            "seed": seed,
            "mean_yield_kg_ha": frame["yield"].mean(),
            "mean_irrigation_mm": frame["irrigation"].mean(),
            "mean_nitrogen_kg_ha": frame["nitrogen"].mean(),
            "mean_wp_et_kg_m3": frame["wp_et"].mean(),
            "mean_pfp_n_kg_kg": frame["pfp_n"].mean(),
            "mean_common_reward": frame["common_reward"].mean(),
            "mean_canonical_reward": frame["canonical_reward"].mean(),
            "yield_years_above_best_baseline": sum(
                float(frame.loc[frame.year.eq(year), "yield"].iloc[0])
                > max(baseline[(scenario, year)]["yield"] for scenario in BASE_SCENARIOS)
                for year in YEARS
            ),
            "positive_action_days_mean": float(np.mean([r["management_action_days"] for r in closure if r["seed"] == seed])),
            "exact_wp_et_years": sum((seed, year) in exact_wp_keys for year in YEARS),
        })
    pd.DataFrame(summary_rows).to_csv(OUT / "seed_screening_summary.csv", index=False)

    manifest = {
        "task": "YC 75 mm all-seed five-scenario figure sets",
        "seeds": SEEDS,
        "years": YEARS,
        "scenario_order": SCENARIOS,
        "baseline_scenarios": BASE_SCENARIOS,
        "style_source_figure": "benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/figures/055_03_yca_five_scenario_metrics.png",
        "style_source_script": "src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py",
        "episode_source": str(EPISODE_CSV.relative_to(ROOT)),
        "step_action_source": str(STEP_CSV.relative_to(ROOT)),
        "baseline_summary_source": str(BASE_SUMMARY.relative_to(ROOT)),
        "baseline_management_event_source": str(BASE_EVENTS.relative_to(ROOT)),
        "exact_wp_et_source": str(REPLAY_CSV.relative_to(ROOT)),
        "exact_wp_et_seeds": sorted({seed for seed, _ in exact_wp_keys}),
        "common_reward_formula": COMMON_REWARD_FORMULA,
        "canonical_reward_comparison": "PPO-only panel; not directly compared against the common baseline formula",
        "action_trace_episode_rows_closed": len(closure),
        "new_training_or_dssat_replay": False,
        "daily_process_figures_omitted": "No complete exact DSSAT daily snapshots are available for all eight seeds; identical annual metrics and action figures are produced for every seed.",
        "figure_files_per_seed": [
            "seedN_five_scenario_metrics.png",
            "seedN_five_scenario_management.png",
            "seedN_five_scenario_reward.png",
            "seedN_five_scenario_yearly_trends.png",
            "seedN_five_scenario_action_calendar.png",
            "seedN_management_action_doses_by_year.png",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(OUT), "seed_count": len(SEEDS), "figure_count": len(list(OUT.rglob("*.png"))),
        "episode_rows": len(episodes), "action_closure_rows": len(closure),
        "exact_wp_et_seeds": manifest["exact_wp_et_seeds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
