"""218YCA: YC five-scenario daily process figures at checkpoint 5,000.

The figure follows the established HL/054_03 layout:

* four frozen non-PPO YC baselines;
* one PPO candidate from the weather-augmented 218YCA run;
* original 2014--2023 validation weather;
* yearly 4x2 daily process panels plus endpoint and management bars.

This script does not train PPO.  It replays the saved 218YCA checkpoint to
DSSAT snapshots so that the PPO line has the same Weather/PlantGro/SoilWat
fields and WP_ET calculation as the four baseline scenarios.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def load_055_figure_module():
    source = ROOT / "src" / "055_yca_lowIC_site_transfer" / "run_055_03_yca_lowIC_five_scenario_figures.py"
    spec = importlib.util.spec_from_file_location("yca_055_figure_helpers", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load YCA figure helpers: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIG055 = load_055_figure_module()
import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_218YCA_yca_lowIC_10y_six_weather_simple_profit_aug_maskableppo_500k as runner
from run_fqa2014_project_profit_reward_ppo_checkpoint_guardrail_035_04 import ProjectProfitDiscreteWrapper
from run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 import snapshot_from_env


TASK_ID = "218YCA"
TASK_NAME = "yca_lowIC_10y_six_weather_simple_profit_aug_maskableppo_500k"
CHECKPOINT = 5_000
STATION = "YCA"
SITE = "YC"
YEARS = list(range(2014, 2024))

PPO_ROOT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
REPLAY_ROOT = PPO_ROOT / "replay_2014_2023"
REPLAY_DETAIL = REPLAY_ROOT / "evaluation" / "218YCA_replay_validation_summary.csv"
MODEL_PATH = PPO_ROOT / "models" / STATION / f"YCA_half_split_stress_aware_maskableppo_seed0_ckpt{CHECKPOINT}.zip"
CONFIG_PATH = PPO_ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
MANIFEST_PATH = PPO_ROOT / "configs" / "218YCA_selected_manifest.csv"

BASELINE_ROOT = ROOT / "benchmark_results" / "055_02_yca_lowIC_four_baselines_static_level1"
AUTO_ROOT = ROOT / "benchmark_results" / "055_01_yca_lowIC_external_auto_n_rule_nstd050_minimal"
FROZEN_PPO_ROOT = ROOT / "benchmark_results" / "055_00_yca_lowIC_expanded_action_maskableppo"
FROZEN_PPO_FIGURE_ROOT = ROOT / "benchmark_results" / "055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000"
FROZEN_PPO_CHECKPOINT = 100_000

OUT_ROOT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}_five_scenario_figures_ckpt{CHECKPOINT}_attempt2"
RECORD = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_five_scenario_figures_ckpt{CHECKPOINT}_record.md"
PROMPT = ROOT / "prompts" / "2026-09-04_yca_simple_profit_weather_augmentation_500k.md"

SCENARIOS = [
    "null",
    "recorded_farmer_template",
    "dssat_auto_external_n",
    "official_extension_expert",
    "rl_candidate",
]
LABELS = {
    "null": "Null",
    "recorded_farmer_template": "Recorded template",
    "dssat_auto_external_n": "DSSAT auto + external N",
    "official_extension_expert": "Official expert",
    "rl_candidate": "PPO (218YCA-5K)",
}
COLORS = {
    "null": "#555555",
    "recorded_farmer_template": "#C44E52",
    "dssat_auto_external_n": "#D8A305",
    "official_extension_expert": "#7E63B6",
    "rl_candidate": "#2A9D55",
}
STYLES = {
    "null": "-",
    "recorded_farmer_template": "--",
    "dssat_auto_external_n": "-.",
    "official_extension_expert": ":",
    "rl_candidate": "-",
}
METRIC_COLUMNS = ["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]
RESOURCE_COLUMNS = ["actual_irrigation_mm", "actual_nitrogen_kg_ha"]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check_inputs() -> list[str]:
    required = [
        CONFIG_PATH,
        MANIFEST_PATH,
        MODEL_PATH,
        REPLAY_DETAIL,
        BASELINE_ROOT / "evaluation" / "055_02_baseline_summary.csv",
        AUTO_ROOT / "evaluation" / "055_01_external_auto_n_summary.csv",
        FROZEN_PPO_ROOT / "evaluation" / "055_00_checkpoint_validation_summary.csv",
        FROZEN_PPO_FIGURE_ROOT / "tables" / "055_03_yca_five_scenario_season_summary.csv",
    ]
    for year in YEARS:
        required.extend(
            [
                BASELINE_ROOT / "snapshots" / STATION / str(year) / "null" / "Summary.OUT",
                BASELINE_ROOT / "snapshots" / STATION / str(year) / "recorded_farmer_template" / "Summary.OUT",
                BASELINE_ROOT / "snapshots" / STATION / str(year) / "official_extension_expert" / "Summary.OUT",
                AUTO_ROOT / "snapshots" / STATION / str(year) / "dssat_auto_irrigation_external_n_rule" / "Summary.OUT",
            ]
        )
    return [rel(path) for path in required if not path.exists()]


def checkpoint_model_path() -> Path:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    return MODEL_PATH


def replay_ppo_snapshot(year: int, out_root: Path) -> Path:
    """Replay the saved 218YCA model once and save a DSSAT snapshot."""
    expected = pd.read_csv(REPLAY_DETAIL, keep_default_na=False)
    expected = expected[
        pd.to_numeric(expected["checkpoint_step"], errors="coerce").eq(CHECKPOINT)
        & pd.to_numeric(expected["year"], errors="coerce").eq(year)
    ]
    if len(expected) != 1:
        raise RuntimeError(f"Expected one 218YCA replay row for YCA{year}, got {len(expected)}")

    target = out_root / "snapshots" / STATION / str(year) / "rl_candidate"
    if target.exists():
        raise FileExistsError(f"Refusing to reuse an existing replay target: {rel(target)}")

    run_contract = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = pd.read_csv(MANIFEST_PATH, keep_default_na=False)
    split = runner.base.make_split(manifest)
    config = runner.base03222.load_config()
    config["reward"] = dict(runner.prior.simple_profit_engine_config()["reward"])
    config["seed"] = int(run_contract["seed"])
    env_config = direct_ppo.build_env_config(config, split)
    env_config["paths"]["output_root"] = rel(out_root)

    old_wrapper = runner.base03222.base.StressAwareDiscreteWrapper
    old_renderer_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    env = None
    try:
        # This is the exact simple-profit wrapper used by the 218YCA run;
        # replay itself is deterministic and does not update model weights.
        runner.base03222.base.StressAwareDiscreteWrapper = ProjectProfitDiscreteWrapper
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = ROOT / run_contract["input_root"]

        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import get_action_masks

        model = MaskablePPO.load(str(checkpoint_model_path()), device="cpu")
        env = runner.base03222.base.make_env(
            config,
            env_config,
            STATION,
            year,
            int(run_contract["seed"]),
            f"YCA_{year}_218YCA_figure_replay",
            evaluation=True,
        )
        obs, _info = env.reset()
        done = False
        steps = 0
        while not done and steps < 260:
            action, _ = model.predict(obs, action_masks=get_action_masks(env), deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
        if not done:
            raise RuntimeError(f"218YCA replay did not finish within 260 steps for YCA{year}")

        snapshot_tmp = snapshot_from_env(env)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(snapshot_tmp, target)

        daily = FIG055.daily_from_snapshot(target, year, "rl_candidate")
        replay_yield = float(pd.to_numeric(daily["grain_yield_kg_ha"], errors="coerce").iloc[-1])
        saved_yield = float(pd.to_numeric(expected.iloc[0]["final_grnwt"], errors="coerce"))
        if not np.isclose(replay_yield, saved_yield, rtol=0.0, atol=1.0):
            raise RuntimeError(
                f"218YCA replay endpoint mismatch for YCA{year}: "
                f"saved={saved_yield:.4f}, replay={replay_yield:.4f}"
            )
        return target
    finally:
        if env is not None:
            env.close()
        runner.base03222.base.StressAwareDiscreteWrapper = old_wrapper
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_renderer_root


def save_figure_all_formats(fig: plt.Figure, stem: Path) -> list[Path]:
    outputs = [
        stem.with_suffix(".png"),
        stem.with_suffix(".svg"),
        stem.with_suffix(".pdf"),
        stem.with_suffix(".tiff"),
    ]
    for path in outputs:
        fig.savefig(path, dpi=600, bbox_inches="tight")
    return outputs


def plot_year(daily: pd.DataFrame, year: int, fig_dir: Path) -> list[Path]:
    """The established HL/054_03 4x2 daily process layout."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    weather = daily[daily["scenario"].eq("null")].sort_values("dap")
    axes[0, 0].bar(weather["dap"], weather["rainfall_mm"], color="#80A9C7", alpha=0.85, label="Rain")
    temp = axes[0, 0].twinx()
    temp.plot(weather["dap"], weather["tmax_c"], color="#C23B32", lw=1.25, label="Tmax")
    temp.plot(weather["dap"], weather["tmin_c"], color="#666666", lw=1.1, ls="--", label="Tmin")
    axes[0, 0].set_title("Weather")
    axes[0, 0].set_ylabel("Rain (mm)")
    temp.set_ylabel("Temperature (C)")

    for scenario in SCENARIOS:
        sub = daily[daily["scenario"].eq(scenario)].sort_values("dap")
        color, style, label = COLORS[scenario], STYLES[scenario], LABELS[scenario]
        axes[0, 1].plot(sub["dap"], sub["soil_water_mm"], color=color, ls=style, lw=1.35, label=label)
        axes[1, 0].plot(sub["dap"], sub["water_stress_index_wspd"], color=color, ls=style, lw=1.35, label=label)
        axes[1, 1].plot(sub["dap"], sub["nitrogen_stress_index_nstd"], color=color, ls=style, lw=1.35, label=label)

        positive_i = sub[sub["irrigation_executed_mm"].gt(0)]
        positive_n = sub[sub["nitrogen_executed_kg_ha"].gt(0)]
        if not positive_i.empty:
            axes[2, 0].stem(
                positive_i["dap"], positive_i["irrigation_executed_mm"],
                linefmt=color, markerfmt="o", basefmt=" ", label=label,
            )
        if not positive_n.empty:
            axes[2, 1].stem(
                positive_n["dap"], positive_n["nitrogen_executed_kg_ha"],
                linefmt=color, markerfmt="o", basefmt=" ", label=label,
            )
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
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    for axis in [axes[0, 1], axes[2, 0], axes[2, 1], axes[3, 1]]:
        axis.legend(fontsize=7, ncol=2)
    axes[3, 0].set_xlabel("DAP")
    axes[3, 1].set_xlabel("DAP")
    fig.suptitle(
        f"YC{year} five-scenario daily process | PPO: 218YCA checkpoint {CHECKPOINT:,} "
        "(weather augmentation; simple profit)",
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    paths = save_figure_all_formats(fig, fig_dir / f"218YCA_yc{year}_five_scenario_daily_ckpt{CHECKPOINT}")
    plt.close(fig)
    return paths


def draw_grouped(combined: pd.DataFrame, years: list[int], metrics: list[str], stem: Path) -> list[Path]:
    x = np.arange(len(years))
    width = 0.16
    fig, axes = plt.subplots(len(metrics), 1, figsize=(16, 4 * len(metrics)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        for idx, scenario in enumerate(SCENARIOS):
            values = []
            for year in years:
                row = combined[(combined["year"].eq(year)) & (combined["scenario"].eq(scenario))]
                values.append(float(row[metric].iloc[0]) if len(row) and pd.notna(row[metric].iloc[0]) else np.nan)
            ax.bar(x + (idx - 2) * width, values, width, label=LABELS[scenario], color=COLORS[scenario])
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(years)
    axes[-1].set_xlabel("Validation year")
    fig.suptitle("YC five-scenario endpoint comparison", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    paths = save_figure_all_formats(fig, stem)
    plt.close(fig)
    return paths


def build_snapshot_map(ppo_snapshot: Path, year: int) -> dict[str, Path]:
    return {
        "null": BASELINE_ROOT / "snapshots" / STATION / str(year) / "null",
        "recorded_farmer_template": BASELINE_ROOT / "snapshots" / STATION / str(year) / "recorded_farmer_template",
        "dssat_auto_external_n": AUTO_ROOT / "snapshots" / STATION / str(year) / "dssat_auto_irrigation_external_n_rule",
        "official_extension_expert": BASELINE_ROOT / "snapshots" / STATION / str(year) / "official_extension_expert",
        "rl_candidate": ppo_snapshot,
    }


def make_summary(snapshot_map: dict[str, Path], daily: pd.DataFrame, year: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario, sub in daily.groupby("scenario"):
        sub = sub.sort_values("dap")
        last = sub.iloc[-1]
        metrics = FIG055.baseline.metrics_from_snapshot(snapshot_map[str(scenario)], float(last["grain_yield_kg_ha"]))
        n_total = float(metrics["actual_nitrogen_kg_ha"])
        pfp_n = float(metrics["PFP_N_kg_kg"]) if n_total > 0 and pd.notna(metrics["PFP_N_kg_kg"]) else np.nan
        yield_value = float(last["grain_yield_kg_ha"])
        irrigation = float(metrics["actual_irrigation_mm"])
        simple_profit = yield_value - 1.1 * irrigation - 1.58 * n_total
        rows.append(
            {
                "station_code": STATION,
                "site": SITE,
                "year": int(year),
                "scenario": scenario,
                "grain_yield_kg_ha": yield_value,
                "biomass_kg_ha": float(last["biomass_kg_ha"]),
                "actual_irrigation_mm": irrigation,
                "actual_nitrogen_kg_ha": n_total,
                "etcp_mm": float(metrics["etcp_mm"]),
                "WP_ET_kg_m3": float(metrics["WP_ET_kg_m3"]),
                "PFP_N_kg_kg": pfp_n,
                "simple_profit": simple_profit,
                "max_wspd": float(pd.to_numeric(sub["water_stress_index_wspd"], errors="coerce").max()),
                "max_nstd": float(pd.to_numeric(sub["nitrogen_stress_index_nstd"], errors="coerce").max()),
                "water_stress_days_gt_0p05": int((pd.to_numeric(sub["water_stress_index_wspd"], errors="coerce") > 0.05).sum()),
                "nitrogen_stress_days_gt_0p05": int((pd.to_numeric(sub["nitrogen_stress_index_nstd"], errors="coerce") > 0.05).sum()),
                "irrigation_event_count": int((sub["irrigation_executed_mm"] > 0).sum()),
                "n_event_count": int((sub["nitrogen_executed_kg_ha"] > 0).sum()),
                "unified_cumulative_reward": float(last["unified_cumulative_reward"]),
                "snapshot_path": rel(snapshot_map[str(scenario)]),
            }
        )
    return rows


def compare_with_frozen(summary: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(
        FROZEN_PPO_ROOT / "evaluation" / "055_00_checkpoint_validation_summary.csv",
        keep_default_na=False,
    )
    old = old[pd.to_numeric(old["checkpoint_step"], errors="coerce").eq(FROZEN_PPO_CHECKPOINT)].copy()
    old["year"] = pd.to_numeric(old["year"], errors="coerce").astype(int)
    old_figure = pd.read_csv(
        FROZEN_PPO_FIGURE_ROOT / "tables" / "055_03_yca_five_scenario_season_summary.csv",
        keep_default_na=False,
    )
    old_figure = old_figure[
        old_figure["scenario"].eq("rl_candidate")
        & pd.to_numeric(old_figure["year"], errors="coerce").between(min(YEARS), max(YEARS))
    ].copy()
    old_figure["year"] = pd.to_numeric(old_figure["year"], errors="coerce").astype(int)
    current = summary[summary["scenario"].eq("rl_candidate")].copy()
    columns = {
        "final_grnwt": "grain_yield_kg_ha",
        "total_irrigation": "actual_irrigation_mm",
        "total_n": "actual_nitrogen_kg_ha",
        "profit_simple": "simple_profit",
        "PFP_N": "PFP_N_kg_kg",
        "WP_ET_kg_m3": "WP_ET_kg_m3",
        "max_wspd": "max_wspd",
        "max_nstd": "max_nstd",
        "water_stress_days_gt_0p05": "swfac_days_gt_0p05",
        "nitrogen_stress_days_gt_0p05": "nstres_days_gt_0p05",
    }
    rows: list[dict[str, Any]] = []
    for year in YEARS:
        new_row = current[current["year"].eq(year)]
        old_row = old[old["year"].eq(year)]
        old_figure_row = old_figure[old_figure["year"].eq(year)]
        if len(new_row) != 1 or len(old_row) != 1 or len(old_figure_row) != 1:
            continue
        new_item = new_row.iloc[0]
        old_item = old_row.iloc[0]
        old_figure_item = old_figure_row.iloc[0]
        row: dict[str, Any] = {"year": year, "new_checkpoint": CHECKPOINT, "frozen_checkpoint": FROZEN_PPO_CHECKPOINT}
        for old_col, new_col in columns.items():
            new_value = pd.to_numeric(pd.Series([new_item.get(new_col)]), errors="coerce").iloc[0]
            old_source = old_figure_item if old_col in {"WP_ET_kg_m3", "max_wspd", "max_nstd"} else old_item
            old_value = pd.to_numeric(pd.Series([old_source.get(old_col)]), errors="coerce").iloc[0]
            row[f"new_{new_col}"] = float(new_value) if pd.notna(new_value) else np.nan
            row[f"frozen_{old_col}"] = float(old_value) if pd.notna(old_value) else np.nan
            row[f"new_minus_frozen_{new_col}"] = float(new_value - old_value) if pd.notna(new_value) and pd.notna(old_value) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run(years: list[int], dry_run: bool = False) -> dict[str, Any]:
    missing = check_inputs()
    result: dict[str, Any] = {
        "task": f"{TASK_ID}_{TASK_NAME}_five_scenario_figures",
        "ppo_checkpoint": CHECKPOINT,
        "years": years,
        "ppo_run_dir": rel(PPO_ROOT),
        "frozen_baseline_root": rel(BASELINE_ROOT),
        "frozen_auto_root": rel(AUTO_ROOT),
        "frozen_ppo_reference": rel(FROZEN_PPO_ROOT),
        "frozen_ppo_checkpoint": FROZEN_PPO_CHECKPOINT,
        "output_root": rel(OUT_ROOT),
        "missing_inputs": missing,
        "five_scenarios": SCENARIOS,
        "figure_contract": {
            "core_conclusion": "Compare the current YC weather-augmented simple-profit PPO candidate with four frozen YC baselines on original 2014-2023 weather; separately quantify the difference from the older frozen PPO checkpoint.",
            "archetype": "quantitative grid and grouped endpoint bars",
            "panel_logic": "weather; soil water; water stress; nitrogen stress; irrigation; nitrogen; grain/biomass; unified cumulative score",
            "backend": "Python/matplotlib only",
            "validation_n": len(years),
            "export": "600 dpi PNG, SVG, PDF, TIFF",
            "source_data": "DSSAT snapshots for all five scenarios; endpoint and comparison CSV tables",
            "integrity_notes": [
                "The current 218YCA formal run stopped before 500K; checkpoint 5K is selected because it was the best saved validation checkpoint by mean simple profit.",
                "The five-scenario figure contains four frozen non-PPO baselines plus the current PPO; the older frozen PPO is reported in a separate comparison table.",
                "The current PPO and frozen PPO use different training reward contracts, so their difference is descriptive rather than a pure weather-augmentation effect.",
            ],
        },
    }
    if dry_run:
        return result
    if missing:
        raise FileNotFoundError("Missing required YC five-scenario figure inputs: " + "; ".join(missing))
    if OUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {rel(OUT_ROOT)}")
    if RECORD.exists():
        raise FileExistsError(f"Refusing to overwrite existing record: {rel(RECORD)}")

    fig_dir = OUT_ROOT / "figures"
    tab_dir = OUT_ROOT / "tables"
    cfg_dir = OUT_ROOT / "configs"
    for directory in (fig_dir, tab_dir, cfg_dir):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG_PATH, cfg_dir / CONFIG_PATH.name)
    if PROMPT.exists():
        shutil.copy2(PROMPT, cfg_dir / PROMPT.name)

    daily_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    figure_paths: list[str] = []
    for year in years:
        print(f"[218YCA figure] replay checkpoint={CHECKPOINT} year={year}", flush=True)
        ppo_snapshot = replay_ppo_snapshot(year, OUT_ROOT)
        snapshot_map = build_snapshot_map(ppo_snapshot, year)
        year_frames = [
            FIG055.daily_from_snapshot(snapshot, year, scenario)
            for scenario, snapshot in snapshot_map.items()
        ]
        merged = FIG055.add_unified_reward(pd.concat(year_frames, ignore_index=True))
        figure_paths.extend(rel(path) for path in plot_year(merged, year, fig_dir))
        daily_frames.append(merged)
        summaries.extend(make_summary(snapshot_map, merged, year))

    all_daily = pd.concat(daily_frames, ignore_index=True)
    actions = all_daily[
        all_daily["irrigation_executed_mm"].gt(0)
        | all_daily["nitrogen_executed_kg_ha"].gt(0)
    ].copy()
    summary = pd.DataFrame(summaries).sort_values(["year", "scenario"])
    comparison = compare_with_frozen(summary)
    gaps = []
    for year, subset in summary.groupby("year"):
        candidate = subset[subset["scenario"].eq("rl_candidate")].iloc[0]
        four = subset[~subset["scenario"].eq("rl_candidate")]
        for metric in METRIC_COLUMNS:
            maximum = pd.to_numeric(four[metric], errors="coerce").max(skipna=True)
            value = float(candidate[metric]) if pd.notna(candidate[metric]) else np.nan
            gaps.append(
                {
                    "year": int(year),
                    "metric": metric,
                    "ppo_value": value,
                    "four_baseline_max": maximum,
                    "ppo_minus_four_max": value - maximum if pd.notna(maximum) else np.nan,
                }
            )
    gaps_df = pd.DataFrame(gaps)

    daily_path = tab_dir / "218YCA_five_scenario_daily.csv"
    action_path = tab_dir / "218YCA_five_scenario_management_events.csv"
    summary_path = tab_dir / "218YCA_five_scenario_season_summary.csv"
    comparison_path = tab_dir / "218YCA_ppo5k_vs_frozen055_ppo100k_by_year.csv"
    gaps_path = tab_dir / "218YCA_metric_gaps_vs_four_baseline_max.csv"
    all_daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    actions.to_csv(action_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    gaps_df.to_csv(gaps_path, index=False, encoding="utf-8-sig")

    endpoint_paths = draw_grouped(summary, years, METRIC_COLUMNS, fig_dir / "218YCA_five_scenario_metrics")
    management_paths = draw_grouped(summary, years, RESOURCE_COLUMNS, fig_dir / "218YCA_five_scenario_management")
    figure_paths.extend(rel(path) for path in endpoint_paths + management_paths)

    result.update(
        {
            "figures": figure_paths,
            "daily_csv": rel(daily_path),
            "management_events_csv": rel(action_path),
            "season_summary_csv": rel(summary_path),
            "comparison_vs_frozen_csv": rel(comparison_path),
            "metric_gaps_csv": rel(gaps_path),
            "replay_validation_rows": len(years),
            "note": "The 218YCA 5K checkpoint is an exploratory current candidate; the formal 500K run stopped early and this package does not freeze the model.",
        }
    )
    manifest_path = OUT_ROOT / "218YCA_five_scenario_figure_manifest.json"
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    RECORD.write_text(
        "\n".join(
            [
                "# 218YCA YC 逐年五情景日过程图记录",
                "",
                "## 口径",
                "",
                f"- 五情景：四个已冻结的 YC 非 PPO 基线 + 218YCA checkpoint {CHECKPOINT:,} PPO。",
                "- PPO 使用天气增强训练线的保存模型，在原始 2014--2023 YC 天气上逐年确定性回放；没有重新训练。",
                "- 图的 4×2 面板、DAP 横轴、情景顺序和配色沿用 HL/054_03 五情景日过程图。",
                "- PPO 重新回放并保存 DSSAT snapshot，因此五条线使用相同的 Weather/PlantGro/SoilWat 字段和 WP_ET 计算口径。",
                "- 218YCA formal 500K 训练在约 223,776 步处因内存保护提前停止；本包选用目前保存 checkpoint 中验证期平均 simple profit 最好的 5K。",
                "- 旧冻结 PPO（055 checkpoint 100K）不放入五情景图，以保持“五情景”结构；其逐年对照单独写入比较表。",
                "- 新旧 PPO 的训练奖励定义不同，因此新旧差值是描述性比较，不是纯粹的天气增强因果效应。",
                "",
                "## 文件",
                "",
                f"- 逐年图和汇总图：`{rel(fig_dir)}`。",
                f"- 日过程源数据：`{rel(daily_path)}`。",
                f"- 管理动作源数据：`{rel(action_path)}`。",
                f"- 五情景季节汇总：`{rel(summary_path)}`。",
                f"- 与旧冻结 PPO 的逐年比较：`{rel(comparison_path)}`。",
                f"- 与四个基线最大值的差值：`{rel(gaps_path)}`。",
                f"- 机器可读清单：`{rel(manifest_path)}`。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    years = [int(value.strip()) for value in args.years.split(",") if value.strip()] or YEARS
    result = run(years, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
