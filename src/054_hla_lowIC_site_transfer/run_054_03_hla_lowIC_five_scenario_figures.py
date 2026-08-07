"""054_03: HLA lowIC five-scenario daily and bar figures.

This is the HLA-local reporting entry point for the 054 site-transfer series.
It creates:

- yearly five-scenario daily process figures;
- season-level summary and management-event tables;
- grouped bar charts for yield, WP_ET, PFP_N, irrigation, and nitrogen;
- PPO-vs-four-baseline metric gap table.
"""

from __future__ import annotations

import argparse
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


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
from build_relaxed_success_five_scenario_daily_evidence_027_05 import parse_management_events, parse_table
from run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 import snapshot_from_env
from sya_experiment_artifacts import resolve_training_inventory, resolve_validation_summary


DEFAULT_CONFIG = ROOT / "configs" / "054_00_hla_lowIC_expanded_action_maskableppo.json"
DEFAULT_AUTO_CONFIG = ROOT / "configs" / "054_01_hla_lowIC_external_auto_n_rule_nstd050_minimal.json"
PROMPT = ROOT / "prompts" / "054_hla_lowIC_site_transfer_expanded_action_ppo_and_auto.md"
PPO_RUN_DIR = ROOT / "benchmark_results" / "054_00_hla_lowIC_expanded_action_maskableppo"
BASELINE_ROOT = ROOT / "benchmark_results" / "054_02_hla_lowIC_four_baselines_static_level1"
STATION = "HLA"
SITE = "HLA"
SCENARIOS = ["null", "recorded_farmer_template", "dssat_auto_external_n", "official_extension_expert", "rl_candidate"]
LABELS = {
    "null": "Null",
    "recorded_farmer_template": "Recorded template",
    "dssat_auto_external_n": "DSSAT auto + external N",
    "official_extension_expert": "Official expert",
    "rl_candidate": "PPO",
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
PPO_REPLAY_YIELD_ATOL_KG_HA = 1.0
INPUT_PROFILES = {
    "originIC": ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013",
    "lowIC": ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_suffix(run_id: str = "") -> str:
    run_id = str(run_id).strip()
    return f"_run_{run_id}" if run_id else ""


def auto_suffix(config: dict[str, Any]) -> str:
    return str(config.get("external_auto_n_rule", {}).get("output_suffix", "")).strip()


def auto_root(auto_config: dict[str, Any], run_id: str = "") -> Path:
    suffix = auto_suffix(auto_config)
    name = f"{auto_config['task_id']}_{auto_config['task_name']}"
    if suffix and suffix not in name:
        name = f"{name}_{suffix}"
    return ROOT / "benchmark_results" / f"{name}{run_suffix(run_id)}"


def output_root(cfg: dict[str, Any], auto_config: dict[str, Any], checkpoint: int, label: str, run_id: str) -> Path:
    suffix = auto_suffix(auto_config)
    auto_part = f"_auto_{suffix}" if suffix else ""
    label_part = f"_{label}" if label else ""
    return ROOT / "benchmark_results" / (
        f"054_03_hla_{cfg['input_profile']}_{cfg['task_id']}_{cfg['task_name']}"
        f"{auto_part}_five_scenario_figures_ckpt{checkpoint}{label_part}{run_suffix(run_id)}"
    )


def daily_from_snapshot(snapshot: Path, year: int, scenario: str) -> pd.DataFrame:
    weather = parse_table(snapshot / "Weather.OUT")
    plant = parse_table(snapshot / "PlantGro.OUT")
    soil = parse_table(snapshot / "SoilWat.OUT")
    if weather.empty or plant.empty or soil.empty:
        raise RuntimeError(f"{scenario} {year}: missing Weather.OUT, PlantGro.OUT, or SoilWat.OUT in {snapshot}")
    w = weather[[c for c in ["YEAR", "DOY", "DAS", "PRED", "TMXD", "TMND"] if c in weather]].rename(
        columns={"YEAR": "year_out", "DOY": "doy", "DAS": "das", "PRED": "rainfall_mm", "TMXD": "tmax_c", "TMND": "tmin_c"}
    )
    p = plant[[c for c in ["DOY", "DAS", "DAP", "GWAD", "CWAD", "WSPD", "NSTD"] if c in plant]].rename(
        columns={"DOY": "doy", "DAS": "das", "DAP": "dap", "GWAD": "grain_yield_kg_ha", "CWAD": "biomass_kg_ha", "WSPD": "water_stress_index_wspd", "NSTD": "nitrogen_stress_index_nstd"}
    )
    s = soil[[c for c in ["DOY", "DAS", "SWTD"] if c in soil]].rename(columns={"DOY": "doy", "DAS": "das", "SWTD": "soil_water_mm"})
    daily = w.merge(p, on=["doy", "das"], how="left").merge(s, on=["doy", "das"], how="left")
    for column in ["dap", "grain_yield_kg_ha", "biomass_kg_ha", "water_stress_index_wspd", "nitrogen_stress_index_nstd", "soil_water_mm"]:
        if column not in daily:
            daily[column] = np.nan
    daily[["dap", "grain_yield_kg_ha", "biomass_kg_ha", "water_stress_index_wspd", "nitrogen_stress_index_nstd", "soil_water_mm"]] = (
        daily[["dap", "grain_yield_kg_ha", "biomass_kg_ha", "water_stress_index_wspd", "nitrogen_stress_index_nstd", "soil_water_mm"]]
        .ffill()
        .fillna(0.0)
    )
    daily["irrigation_executed_mm"] = 0.0
    daily["nitrogen_executed_kg_ha"] = 0.0
    for _, event in parse_management_events(snapshot / "MgmtEvent.OUT").iterrows():
        target = daily["doy"].eq(int(event["doy"]))
        operation = str(event["operation"]).lower()
        if operation.startswith("irrig"):
            daily.loc[target, "irrigation_executed_mm"] += float(event["amount"])
        elif operation.startswith("fert"):
            daily.loc[target, "nitrogen_executed_kg_ha"] += float(event["amount"])
    daily["date"] = pd.to_datetime(
        daily["year_out"].astype(int).astype(str) + daily["doy"].astype(int).astype(str).str.zfill(3),
        format="%Y%j",
        errors="coerce",
    )
    daily.insert(0, "station_code", STATION)
    daily.insert(1, "site", SITE)
    daily.insert(2, "year", int(year))
    daily.insert(3, "scenario", scenario)
    return daily


def replay_ppo_snapshot(cfg: dict[str, Any], checkpoint: int, year: int, out: Path, ppo_root: Path) -> Path:
    profile = str(cfg["input_profile"])
    inventory_path = resolve_training_inventory(ppo_root, checkpoint, preferred_prefix=str(cfg["task_id"]))
    eval_path = resolve_validation_summary(ppo_root, checkpoint, list(map(int, cfg["scope"]["validation_years"])), preferred_prefix=str(cfg["task_id"]))
    inventory = pd.read_csv(inventory_path, keep_default_na=False)
    source_eval = pd.read_csv(eval_path, keep_default_na=False)
    model_rows = inventory[pd.to_numeric(inventory["checkpoint_step"], errors="coerce").eq(checkpoint)]
    if model_rows.empty:
        raise RuntimeError(f"No model inventory row for checkpoint {checkpoint}")
    model_path = ROOT / str(model_rows.iloc[0]["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    expected = source_eval[
        pd.to_numeric(source_eval["checkpoint_step"], errors="coerce").eq(checkpoint)
        & pd.to_numeric(source_eval["year"], errors="coerce").eq(year)
    ]
    if len(expected) != 1:
        raise RuntimeError(f"Expected one PPO validation row for HLA{year} checkpoint {checkpoint}")
    target = out / "snapshots" / STATION / str(year) / "rl_candidate"
    if target.exists() and (target / "Summary.OUT").exists():
        existing = daily_from_snapshot(target, year, "rl_candidate")
        replay_yield = float(pd.to_numeric(existing["grain_yield_kg_ha"], errors="coerce").iloc[-1])
        saved_yield = float(pd.to_numeric(expected.iloc[0]["final_grnwt"], errors="coerce"))
        if np.isclose(replay_yield, saved_yield, rtol=0.0, atol=0.5):
            return target
        stale = out / "stale_snapshots" / STATION / str(year) / "rl_candidate_before_regeneration"
        suffix = 1
        while stale.exists():
            stale = stale.with_name(f"rl_candidate_before_regeneration_{suffix}")
            suffix += 1
        stale.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(stale))

    old_values = {
        key: getattr(engine, key)
        for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]
    }
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_site_names = dict(base03222.SITE_NAMES)
    env = None
    try:
        engine.TASK_ID = str(cfg["task_id"])
        engine.TASK_NAME = str(cfg["task_name"])
        engine.BASE_OUT = ppo_root
        engine.BASE_DOC = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
        engine.PROMPT = PROMPT
        engine.LOWIC_INPUT_ROOT = INPUT_PROFILES[profile]
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = INPUT_PROFILES[profile]
        engine.STATION = STATION
        engine.SITES = [STATION]
        engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
        engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
        base03222.SITE_NAMES[STATION] = SITE
        engine.patch_base_module(ppo_root, engine.BASE_DOC, int(cfg["training"]["total_timesteps"]), list(map(int, cfg["training"]["checkpoint_steps"])))
        config = base03222.load_config()
        selection = base03222.build_selection(base03222.load_split())
        env_config = direct_ppo.build_env_config(config, selection)
        env_config["paths"]["output_root"] = rel(ppo_root)
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import get_action_masks

        model = MaskablePPO.load(str(model_path), device="cpu")
        env = base03222.base.make_env(config, env_config, STATION, int(year), int(cfg.get("seed", 0)), f"HLA_{year}_054_03_replay", evaluation=True)
        obs, _info = env.reset()
        done, steps = False, 0
        while not done and steps < 260:
            action, _ = model.predict(obs, action_masks=get_action_masks(env), deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
        if not done:
            raise RuntimeError(f"PPO replay did not finish within 260 steps for HLA{year}")
        snapshot_tmp = snapshot_from_env(env)
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(snapshot_tmp, target)
        daily = daily_from_snapshot(target, year, "rl_candidate")
        replay_yield = float(pd.to_numeric(daily["grain_yield_kg_ha"], errors="coerce").iloc[-1])
        saved_yield = float(pd.to_numeric(expected.iloc[0]["final_grnwt"], errors="coerce"))
        if not np.isclose(replay_yield, saved_yield, rtol=0.0, atol=PPO_REPLAY_YIELD_ATOL_KG_HA):
            raise RuntimeError(
                f"PPO replay endpoint mismatch for HLA{year}: "
                f"saved={saved_yield:.4f}, replay={replay_yield:.4f}, "
                f"atol={PPO_REPLAY_YIELD_ATOL_KG_HA:.2f}"
            )
        return target
    finally:
        if env is not None:
            env.close()
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
        base03222.SITE_NAMES.clear()
        base03222.SITE_NAMES.update(old_site_names)
        for key, value in old_values.items():
            setattr(engine, key, value)


def add_unified_reward(daily: pd.DataFrame) -> pd.DataFrame:
    result = daily.copy()
    result["unified_reward_step"] = -1.1 * result["irrigation_executed_mm"] - 1.58 * result["nitrogen_executed_kg_ha"]
    for _, idx in result.groupby(["year", "scenario"]).groups.items():
        positions = list(idx)
        final_yield = float(pd.to_numeric(result.loc[positions, "grain_yield_kg_ha"], errors="coerce").iloc[-1])
        result.loc[positions[-1], "unified_reward_step"] += 0.158 * final_yield
    result["unified_cumulative_reward"] = result.groupby(["year", "scenario"])["unified_reward_step"].cumsum()
    return result


def plot_year(daily: pd.DataFrame, year: int, fig_dir: Path) -> Path:
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
            axes[2, 0].stem(positive_i["dap"], positive_i["irrigation_executed_mm"], linefmt=color, markerfmt="o", basefmt=" ", label=label)
        if not positive_n.empty:
            axes[2, 1].stem(positive_n["dap"], positive_n["nitrogen_executed_kg_ha"], linefmt=color, markerfmt="o", basefmt=" ", label=label)
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
    fig.suptitle(f"HLA{year} five-scenario daily process", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = fig_dir / f"054_03_hla{year}_five_scenario_daily.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def draw_grouped(combined: pd.DataFrame, years: list[int], metrics: list[str], path: Path) -> None:
    x = np.arange(len(years))
    width = 0.16
    fig, axes = plt.subplots(len(metrics), 1, figsize=(16, 4 * len(metrics)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        for idx, scenario in enumerate(SCENARIOS):
            values = []
            for year in years:
                row = combined[(combined["year"].eq(year)) & (combined["scenario"].eq(scenario))]
                values.append(float(row[metric].iloc[0]) if len(row) else np.nan)
            ax.bar(x + (idx - 2) * width, values, width, label=LABELS[scenario], color=COLORS[scenario])
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(years)
    axes[-1].set_xlabel("Validation year")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_snapshot_map(baseline_root: Path, auto_output_root: Path, ppo_snapshot: Path, year: int) -> dict[str, Path]:
    return {
        "null": baseline_root / "snapshots" / STATION / str(year) / "null",
        "recorded_farmer_template": baseline_root / "snapshots" / STATION / str(year) / "recorded_farmer_template",
        "dssat_auto_external_n": auto_output_root / "snapshots" / STATION / str(year) / "dssat_auto_irrigation_external_n_rule",
        "official_extension_expert": baseline_root / "snapshots" / STATION / str(year) / "official_extension_expert",
        "rl_candidate": ppo_snapshot,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--auto-config", type=Path, default=DEFAULT_AUTO_CONFIG)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--years", type=str, default="")
    parser.add_argument("--ppo-run-dir", type=Path, default=PPO_RUN_DIR)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--auto-run-id", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--skip-daily", action="store_true")
    parser.add_argument("--skip-bars", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else (ROOT / args.config).resolve()
    auto_config_path = args.auto_config if args.auto_config.is_absolute() else (ROOT / args.auto_config).resolve()
    cfg = read_json(cfg_path)
    auto_config = read_json(auto_config_path)
    if cfg.get("station_code") != STATION or auto_config.get("station_code") != STATION:
        raise ValueError("054_03 only accepts HLA PPO and HLA auto configs.")
    checkpoint = int(args.checkpoint or cfg.get("report_checkpoint", 100_000))
    years = [int(x.strip()) for x in args.years.split(",") if x.strip()] or list(map(int, cfg["scope"]["validation_years"]))
    ppo_root = args.ppo_run_dir if args.ppo_run_dir.is_absolute() else (ROOT / args.ppo_run_dir).resolve()
    baseline_root = args.baseline_root if args.baseline_root.is_absolute() else (ROOT / args.baseline_root).resolve()
    auto_output_root = auto_root(auto_config, args.auto_run_id)
    out = output_root(cfg, auto_config, checkpoint, args.label.strip(), args.auto_run_id.strip())
    fig_dir, tab_dir, cfg_dir = out / "figures", out / "tables", out / "configs"
    required = [
        cfg_path,
        auto_config_path,
        ppo_root / f"{cfg['task_id']}_formal_result.json",
        resolve_validation_summary(ppo_root, checkpoint, years, preferred_prefix=str(cfg["task_id"])),
        resolve_training_inventory(ppo_root, checkpoint, preferred_prefix=str(cfg["task_id"])),
        auto_output_root / "evaluation" / "054_01_external_auto_n_summary.csv",
        baseline_root / "evaluation" / "054_02_baseline_summary.csv",
    ]
    for year in years:
        required.extend(
            [
                baseline_root / "snapshots" / STATION / str(year) / "null" / "Summary.OUT",
                baseline_root / "snapshots" / STATION / str(year) / "recorded_farmer_template" / "Summary.OUT",
                baseline_root / "snapshots" / STATION / str(year) / "official_extension_expert" / "Summary.OUT",
                auto_output_root / "snapshots" / STATION / str(year) / "dssat_auto_irrigation_external_n_rule" / "Summary.OUT",
            ]
        )
    missing = [rel(path) for path in required if not path.exists()]
    payload = {
        "task": "054_03_hla_five_scenario_figures",
        "checkpoint": checkpoint,
        "years": years,
        "output_root": rel(out),
        "ppo_run_dir": rel(ppo_root),
        "baseline_root": rel(baseline_root),
        "auto_output_root": rel(auto_output_root),
        "missing_inputs": missing,
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if missing:
        raise FileNotFoundError("Missing required 054 figure inputs: " + "; ".join(missing))
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing figure output root: {rel(out)}")
    for directory in [fig_dir, tab_dir, cfg_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, cfg_dir / cfg_path.name)
    shutil.copy2(auto_config_path, cfg_dir / auto_config_path.name)
    if PROMPT.exists():
        shutil.copy2(PROMPT, cfg_dir / PROMPT.name)

    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    figures: list[Path] = []
    for year in years:
        ppo_snapshot = replay_ppo_snapshot(cfg, checkpoint, year, out, ppo_root)
        snapshot_map = build_snapshot_map(baseline_root, auto_output_root, ppo_snapshot, year)
        year_frames = [daily_from_snapshot(snapshot, year, scenario) for scenario, snapshot in snapshot_map.items()]
        merged = add_unified_reward(pd.concat(year_frames, ignore_index=True))
        if not args.skip_daily:
            figures.append(plot_year(merged, year, fig_dir))
        frames.append(merged)
        for scenario, sub in merged.groupby("scenario"):
            last = sub.iloc[-1]
            metrics = baseline.metrics_from_snapshot(snapshot_map[str(scenario)], float(last["grain_yield_kg_ha"]))
            n_total = float(metrics["actual_nitrogen_kg_ha"])
            pfp_n = float(metrics["PFP_N_kg_kg"]) if pd.notna(metrics["PFP_N_kg_kg"]) and n_total > 0 else np.nan
            summaries.append(
                {
                    "station_code": STATION,
                    "site": SITE,
                    "year": int(year),
                    "scenario": scenario,
                    "grain_yield_kg_ha": float(last["grain_yield_kg_ha"]),
                    "biomass_kg_ha": float(last["biomass_kg_ha"]),
                    "actual_irrigation_mm": float(metrics["actual_irrigation_mm"]),
                    "actual_nitrogen_kg_ha": n_total,
                    "etcp_mm": float(metrics["etcp_mm"]),
                    "WP_ET_kg_m3": float(metrics["WP_ET_kg_m3"]),
                    "PFP_N_kg_kg": pfp_n,
                    "max_wspd": float(pd.to_numeric(sub["water_stress_index_wspd"], errors="coerce").max()),
                    "max_nstd": float(pd.to_numeric(sub["nitrogen_stress_index_nstd"], errors="coerce").max()),
                    "unified_reward": float(last["unified_cumulative_reward"]),
                    "snapshot_path": rel(snapshot_map[str(scenario)]),
                }
            )
    all_daily = pd.concat(frames, ignore_index=True)
    actions = all_daily[(all_daily["irrigation_executed_mm"].gt(0)) | (all_daily["nitrogen_executed_kg_ha"].gt(0))].copy()
    summary = pd.DataFrame(summaries)
    all_daily.to_csv(tab_dir / "054_03_hla_five_scenario_daily.csv", index=False, encoding="utf-8-sig")
    actions.to_csv(tab_dir / "054_03_hla_five_scenario_management_events.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(tab_dir / "054_03_hla_five_scenario_season_summary.csv", index=False, encoding="utf-8-sig")

    gaps = []
    for year, subset in summary.groupby("year"):
        candidate = subset[subset["scenario"].eq("rl_candidate")].iloc[0]
        four = subset[~subset["scenario"].eq("rl_candidate")]
        for metric in METRIC_COLUMNS:
            maximum = pd.to_numeric(four[metric], errors="coerce").max(skipna=True)
            ppo_value = float(candidate[metric]) if pd.notna(candidate[metric]) else np.nan
            gaps.append(
                {
                    "year": int(year),
                    "metric": metric,
                    "ppo_value": ppo_value,
                    "four_baseline_max": maximum,
                    "ppo_minus_four_max": ppo_value - maximum if pd.notna(maximum) else np.nan,
                }
            )
    gaps_df = pd.DataFrame(gaps)
    gaps_df.to_csv(tab_dir / "054_03_hla_metric_gaps_vs_four_baseline_max.csv", index=False, encoding="utf-8-sig")
    if not args.skip_bars:
        draw_grouped(summary, years, METRIC_COLUMNS, fig_dir / "054_03_hla_five_scenario_metrics.png")
        draw_grouped(summary, years, RESOURCE_COLUMNS, fig_dir / "054_03_hla_five_scenario_management.png")

    wins = gaps_df.groupby("metric")["ppo_minus_four_max"].apply(lambda values: int((pd.to_numeric(values, errors="coerce") > 0).sum())).to_dict()
    record = ROOT / "docs" / f"054_03_hla_{cfg['input_profile']}_{cfg['task_id']}_{cfg['task_name']}_five_scenario_figures_ckpt{checkpoint}_record.md"
    record.write_text(
        "\n".join(
            [
                f"# 054_03 HLA {cfg['input_profile']} five-scenario figures record",
                "",
                f"- PPO checkpoint: `{checkpoint}`.",
                f"- Years: `{years}`.",
                f"- PPO run dir: `{rel(ppo_root)}`.",
                f"- Baseline root: `{rel(baseline_root)}`.",
                f"- Auto root: `{rel(auto_output_root)}`.",
                "- Four non-PPO baselines are expected to come from 054_02 static level-1 corrected baseline outputs, not old 034_00 outputs.",
                "- PFP_N is kept as N/A when actual nitrogen is zero.",
                f"- PPO win counts versus the best of the other four scenarios: `{wins}`.",
                f"- Daily table: `{rel(tab_dir / '054_03_hla_five_scenario_daily.csv')}`.",
                f"- Season summary: `{rel(tab_dir / '054_03_hla_five_scenario_season_summary.csv')}`.",
                f"- Gap table: `{rel(tab_dir / '054_03_hla_metric_gaps_vs_four_baseline_max.csv')}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    payload.update(
        {
            "figures": [rel(path) for path in figures],
            "metric_bars": rel(fig_dir / "054_03_hla_five_scenario_metrics.png"),
            "management_bars": rel(fig_dir / "054_03_hla_five_scenario_management.png"),
            "daily_csv": rel(tab_dir / "054_03_hla_five_scenario_daily.csv"),
            "season_summary": rel(tab_dir / "054_03_hla_five_scenario_season_summary.csv"),
            "management_events": rel(tab_dir / "054_03_hla_five_scenario_management_events.csv"),
            "gaps": rel(tab_dir / "054_03_hla_metric_gaps_vs_four_baseline_max.csv"),
            "record_md": rel(record),
            "ppo_win_counts": wins,
        }
    )
    (out / "054_03_result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


