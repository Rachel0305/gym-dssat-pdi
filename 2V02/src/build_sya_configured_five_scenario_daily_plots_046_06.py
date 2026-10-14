"""046_06: replay one configured PPO checkpoint and make auditable daily plots.

Unlike older reports, this script refuses to mix PPO daily data with a soil-water
curve from a different DSSAT replay.  It creates a deterministic replay snapshot
and requires its final grain value to reproduce the saved 046_02 evaluation.
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

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
from build_relaxed_success_five_scenario_daily_evidence_027_05 import parse_management_events, parse_table
from run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 import snapshot_from_env
from sya_experiment_artifacts import resolve_training_inventory, resolve_validation_summary


DEFAULT_CONFIG = ROOT / "configs" / "046_02_sya_originIC_binary_timing_ppo.json"
PROMPT = ROOT / "prompts" / "046_06_sya_configured_five_scenario_daily_plots.md"
INPUT_PROFILES = {
    "originIC": ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013",
    "lowIC": ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual",
}
SCENARIOS = ["null", "recorded_farmer_template", "dssat_auto_external_n", "official_extension_expert", "rl_candidate"]
LABELS = {"null": "Null", "recorded_farmer_template": "Recorded template", "dssat_auto_external_n": "DSSAT auto + external N", "official_extension_expert": "Official expert", "rl_candidate": "PPO"}
COLORS = {"null": "#444444", "recorded_farmer_template": "#C44E52", "dssat_auto_external_n": "#D8A305", "official_extension_expert": "#7861B2", "rl_candidate": "#238B45"}
STYLES = {"null": "-", "recorded_farmer_template": "--", "dssat_auto_external_n": "-.", "official_extension_expert": ":", "rl_candidate": "-"}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_suffix(run_id: str = "") -> str:
    run_id = str(run_id).strip()
    return f"_run_{run_id}" if run_id else ""


def external_auto_root(cfg: dict[str, Any], run_id: str = "") -> Path:
    override = str(cfg.get("figure_package", {}).get("auto_output_root", "")).strip()
    if override:
        path = ROOT / override
        return path if path.is_absolute() else (ROOT / override)
    suffix = str(cfg.get("external_auto_n_rule", {}).get("output_suffix", "")).strip()
    name = f"046_04_sya_{cfg['input_profile']}_external_auto_n_rule"
    if suffix:
        name = f"{name}_{suffix}"
    name = f"{name}{run_suffix(run_id)}"
    return ROOT / "benchmark_results" / name


def output_root(cfg: dict[str, Any], checkpoint: int, run_id: str = "") -> Path:
    # Replay snapshots are model-specific.  The PPO task identity is therefore
    # part of the folder name; never reuse a 046_02 snapshot for a 046_07 run.
    auto_suffix = str(cfg.get("external_auto_n_rule", {}).get("output_suffix", "")).strip()
    auto_part = f"_auto_{auto_suffix}" if auto_suffix else ""
    return ROOT / "benchmark_results" / f"046_06_sya_{cfg['input_profile']}_{cfg['task_id']}_{cfg['task_name']}{auto_part}_five_scenario_daily_ckpt{checkpoint}{run_suffix(run_id)}"


def daily_from_snapshot(snapshot: Path, year: int, scenario: str) -> pd.DataFrame:
    weather, plant, soil = parse_table(snapshot / "Weather.OUT"), parse_table(snapshot / "PlantGro.OUT"), parse_table(snapshot / "SoilWat.OUT")
    if weather.empty or plant.empty or soil.empty:
        raise RuntimeError(f"{scenario} {year}: missing Weather.OUT, PlantGro.OUT, or SoilWat.OUT in {snapshot}")
    w = weather[[c for c in ["YEAR", "DOY", "DAS", "PRED", "TMXD", "TMND"] if c in weather]].rename(columns={"YEAR": "year_out", "DOY": "doy", "DAS": "das", "PRED": "rainfall_mm", "TMXD": "tmax_c", "TMND": "tmin_c"})
    p = plant[[c for c in ["DOY", "DAS", "DAP", "GWAD", "CWAD", "WSPD", "NSTD"] if c in plant]].rename(columns={"DOY": "doy", "DAS": "das", "DAP": "dap", "GWAD": "grain_yield_kg_ha", "CWAD": "biomass_kg_ha", "WSPD": "water_stress_index_wspd", "NSTD": "nitrogen_stress_index_nstd"})
    s = soil[[c for c in ["DOY", "DAS", "SWTD"] if c in soil]].rename(columns={"DOY": "doy", "DAS": "das", "SWTD": "soil_water_mm"})
    daily = w.merge(p, on=["doy", "das"], how="left").merge(s, on=["doy", "das"], how="left")
    for column in ["dap", "grain_yield_kg_ha", "biomass_kg_ha", "water_stress_index_wspd", "nitrogen_stress_index_nstd", "soil_water_mm"]:
        if column not in daily:
            daily[column] = np.nan
    daily[["dap", "grain_yield_kg_ha", "biomass_kg_ha", "water_stress_index_wspd", "nitrogen_stress_index_nstd", "soil_water_mm"]] = daily[["dap", "grain_yield_kg_ha", "biomass_kg_ha", "water_stress_index_wspd", "nitrogen_stress_index_nstd", "soil_water_mm"]].ffill().fillna(0.0)
    daily["irrigation_executed_mm"] = 0.0
    daily["nitrogen_executed_kg_ha"] = 0.0
    for _, event in parse_management_events(snapshot / "MgmtEvent.OUT").iterrows():
        target = daily["doy"].eq(int(event["doy"]))
        if str(event["operation"]).lower().startswith("irrig"):
            daily.loc[target, "irrigation_executed_mm"] += float(event["amount"])
        elif str(event["operation"]).lower().startswith("fert"):
            daily.loc[target, "nitrogen_executed_kg_ha"] += float(event["amount"])
    daily["date"] = pd.to_datetime(daily["year_out"].astype(int).astype(str) + daily["doy"].astype(int).astype(str).str.zfill(3), format="%Y%j", errors="coerce")
    daily.insert(0, "station_code", "SYA")
    daily.insert(1, "site", "SY")
    daily.insert(2, "year", int(year))
    daily.insert(3, "scenario", scenario)
    return daily


def replay_ppo_snapshot(cfg: dict[str, Any], checkpoint: int, year: int, out: Path, ppo_root: Path) -> Path:
    profile = str(cfg["input_profile"])
    inventory = pd.read_csv(resolve_training_inventory(ppo_root, checkpoint, preferred_prefix=str(cfg["task_id"])), keep_default_na=False)
    source_eval = pd.read_csv(resolve_validation_summary(ppo_root, checkpoint, list(map(int, cfg["scope"]["validation_years"])), preferred_prefix=str(cfg["task_id"])), keep_default_na=False)
    model_rows = inventory[pd.to_numeric(inventory["checkpoint_step"], errors="coerce").eq(checkpoint)]
    if model_rows.empty:
        raise RuntimeError(f"No 046_02 model inventory row for checkpoint {checkpoint}")
    model_path = ROOT / str(model_rows.iloc[0]["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    expected_rows = source_eval[(pd.to_numeric(source_eval["checkpoint_step"], errors="coerce").eq(checkpoint)) & (pd.to_numeric(source_eval["year"], errors="coerce").eq(year))]
    if len(expected_rows) != 1:
        raise RuntimeError(f"Expected exactly one saved 046_02 evaluation row for {year} checkpoint {checkpoint}")
    target = out / "snapshots" / "SYA" / str(year) / "rl_candidate"

    # Reuse a prior replay snapshot only after validating it against the saved
    # 046_02 endpoint.  This matters when an earlier plotting attempt stopped
    # after creating the snapshot but before writing its final figure.
    if target.exists() and (target / "Summary.OUT").exists():
        existing = daily_from_snapshot(target, year, "rl_candidate")
        replay_yield = float(pd.to_numeric(existing["grain_yield_kg_ha"], errors="coerce").iloc[-1])
        saved_yield = float(pd.to_numeric(expected_rows.iloc[0]["final_grnwt"], errors="coerce"))
        if np.isclose(replay_yield, saved_yield, rtol=0.0, atol=0.5):
            return target
        # Preserve a partially generated snapshot from an earlier failed plot
        # attempt, then regenerate it with the current model-input pipeline.
        stale_base = out / "stale_snapshots" / "SYA" / str(year) / "rl_candidate_before_input_pipeline_fix"
        stale = stale_base
        suffix = 1
        while stale.exists():
            stale = stale_base.with_name(f"{stale_base.name}_{suffix}")
            suffix += 1
        stale.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(stale))

    old_values = {name: getattr(engine, name) for name in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_make_env = base03222.base.make_env
    env = None
    try:
        engine.TASK_ID = str(cfg["task_id"])
        engine.TASK_NAME = str(cfg["task_name"])
        engine.BASE_OUT = ppo_root
        engine.BASE_DOC = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
        engine.LOWIC_INPUT_ROOT = INPUT_PROFILES[profile]
        engine.STATION, engine.SITES = "SYA", ["SYA"]
        engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
        engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
        engine.patch_base_module(ppo_root, engine.BASE_DOC, int(cfg["training"]["total_timesteps"]), list(map(int, cfg["training"]["checkpoint_steps"])))
        forecast_raw = dict(cfg.get("weather_forecast_observation_04609", {}))
        if bool(forecast_raw.get("enabled", False)):
            from run_sya_originIC_forecast_raw_observation_maskableppo_046_09 import make_env_04609

            if bool(forecast_raw.get("normalization_enabled", True)):
                raise RuntimeError("046_06 raw-forecast replay expected normalization_enabled=false")
            base03222.base.make_env = make_env_04609
        config = base03222.load_config()
        selection = base03222.build_selection(base03222.load_split())
        env_config = direct_ppo.build_env_config(config, selection)
        env_config["paths"]["output_root"] = rel(ppo_root)
        normalization = dict(cfg.get("observation_normalization", {}))
        if bool(normalization.get("enabled", False)):
            if normalization.get("type") != "train_year_noop_mean_std":
                raise RuntimeError(f"Unsupported observation normalization for PPO replay: {normalization.get('type')}")
            stats_path = ppo_root / str(normalization.get("statistics_relative_path", ""))
            if not stats_path.exists():
                raise FileNotFoundError(f"PPO replay needs saved normalization statistics: {stats_path}")
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)
            clip = float(normalization.get("clip", 5.0))
            from run_sya_originIC_train_stats_normalized_maskableppo_046_07 import FixedTrainStatisticsObservationWrapper

            def normalized_make_env(config: dict[str, Any], env_config: dict[str, Any], station: str, run_year: int, seed: int, run_tag: str, evaluation: bool = False):
                wrapper = FixedTrainStatisticsObservationWrapper(old_make_env(config, env_config, station, run_year, seed, run_tag, evaluation=evaluation), mean, std)
                if not np.isclose(clip, 5.0):
                    raise RuntimeError("046_06 currently requires the trained fixed normalization clip=5.0")
                return wrapper

            base03222.base.make_env = normalized_make_env
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import get_action_masks
        model = MaskablePPO.load(str(model_path), device="cpu")
        env = base03222.base.make_env(config, env_config, "SYA", int(year), int(cfg.get("seed", 0)), f"SYA_{year}_046_06_replay", evaluation=True)
        obs, _info = env.reset()
        done, steps = False, 0
        while not done and steps < 260:
            action, _ = model.predict(obs, action_masks=get_action_masks(env), deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
        if not done:
            raise RuntimeError(f"PPO replay did not finish within 260 steps for SYA{year}")
        snapshot_tmp = snapshot_from_env(env)
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(snapshot_tmp, target)
        daily = daily_from_snapshot(target, year, "rl_candidate")
        replay_yield = float(pd.to_numeric(daily["grain_yield_kg_ha"], errors="coerce").iloc[-1])
        saved_yield = float(pd.to_numeric(expected_rows.iloc[0]["final_grnwt"], errors="coerce"))
        # PlantGro.OUT prints GWAD at coarser precision than the in-memory PPO
        # evaluation CSV.  A difference below 0.5 kg/ha is therefore a display
        # rounding difference, not evidence that the replay policy differs.
        if not np.isclose(replay_yield, saved_yield, rtol=0.0, atol=0.5):
            raise RuntimeError(f"PPO replay endpoint mismatch for {year}: saved={saved_yield:.4f}, replay={replay_yield:.4f}; no mixed-source plot was written")
        return target
    finally:
        if env is not None:
            env.close()
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
        base03222.base.make_env = old_make_env
        for name, value in old_values.items():
            setattr(engine, name, value)


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
            axes[2, 0].stem(positive_i["dap"], positive_i["irrigation_executed_mm"], linefmt=color, markerfmt="o", basefmt=" ", label=label)
        if not positive_n.empty:
            axes[2, 1].stem(positive_n["dap"], positive_n["nitrogen_executed_kg_ha"], linefmt=color, markerfmt="o", basefmt=" ", label=label)
        axes[3, 0].plot(sub["dap"], sub["grain_yield_kg_ha"], color=color, ls=style, lw=1.45, label=f"{label} grain")
        axes[3, 0].plot(sub["dap"], sub["biomass_kg_ha"], color=color, ls=style, lw=0.85, alpha=0.4)
        axes[3, 1].plot(sub["dap"], sub["unified_cumulative_reward"], color=color, ls=style, lw=1.35, label=label)
    titles = [(axes[0, 1], "Soil water", "SWTD (mm)"), (axes[1, 0], "Water stress index", "WSPD"), (axes[1, 1], "Nitrogen stress index", "NSTD"), (axes[2, 0], "Irrigation events", "mm/event"), (axes[2, 1], "Nitrogen application events", "kg/ha/event"), (axes[3, 0], "Grain and biomass trajectories", "kg/ha"), (axes[3, 1], "Unified cumulative reward", "common units")]
    for axis, title, ylabel in titles:
        axis.set_title(title); axis.set_ylabel(ylabel); axis.grid(alpha=0.2)
    axes[0, 1].legend(fontsize=7, ncol=2); axes[2, 0].legend(fontsize=7, ncol=2); axes[2, 1].legend(fontsize=7, ncol=2); axes[3, 1].legend(fontsize=7, ncol=2)
    axes[3, 0].set_xlabel("DAP"); axes[3, 1].set_xlabel("DAP")
    fig.suptitle(f"SY{year} configured five-scenario daily process", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = fig_dir / f"046_06_sy{year}_five_scenario_daily.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--years", type=str, default="")
    parser.add_argument("--ppo-run-dir", type=Path, default=None, help="Optional PPO output directory; default is derived from task_id/task_name in config")
    parser.add_argument("--run-id", type=str, default="", help="Optional run batch id; reads matching 046_03/046_04 outputs and writes separate daily plots")
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    cfg = read_config(cfg_path)
    checkpoint = int(args.checkpoint or cfg.get("report_checkpoint", 25_000))
    years = [int(x.strip()) for x in args.years.split(",") if x.strip()] or list(map(int, cfg["scope"]["validation_years"]))
    profile = str(cfg["input_profile"])
    ppo_root = args.ppo_run_dir if args.ppo_run_dir else ROOT / "benchmark_results" / f"{cfg['task_id']}_{cfg['task_name']}"
    if not ppo_root.is_absolute():
        ppo_root = (Path.cwd() / ppo_root).resolve()
    out = output_root(cfg, checkpoint, args.run_id)
    fig_dir, tab_dir, cfg_dir = out / "figures", out / "tables", out / "configs"
    for directory in [fig_dir, tab_dir, cfg_dir]: directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, cfg_dir / cfg_path.name)
    if PROMPT.exists(): shutil.copy2(PROMPT, cfg_dir / PROMPT.name)
    roots = {
        "null": ROOT / "benchmark_results" / f"046_03_sya_{profile}_four_baselines{run_suffix(args.run_id)}" / "snapshots",
        "recorded_farmer_template": ROOT / "benchmark_results" / f"046_03_sya_{profile}_four_baselines{run_suffix(args.run_id)}" / "snapshots",
        "official_extension_expert": ROOT / "benchmark_results" / f"046_03_sya_{profile}_four_baselines{run_suffix(args.run_id)}" / "snapshots",
        "dssat_auto_external_n": external_auto_root(cfg, args.run_id) / "snapshots",
    }
    frames, summaries, figures = [], [], []
    for year in years:
        ppo_snapshot = replay_ppo_snapshot(cfg, checkpoint, year, out, ppo_root)
        snapshot_map = {
            "null": roots["null"] / "SYA" / str(year) / "null",
            "recorded_farmer_template": roots["recorded_farmer_template"] / "SYA" / str(year) / "recorded_farmer_template",
            "dssat_auto_external_n": roots["dssat_auto_external_n"] / "SYA" / str(year) / "dssat_auto_irrigation_external_n_rule",
            "official_extension_expert": roots["official_extension_expert"] / "SYA" / str(year) / "official_extension_expert",
            "rl_candidate": ppo_snapshot,
        }
        year_frames = []
        for scenario, snapshot in snapshot_map.items():
            year_frames.append(daily_from_snapshot(snapshot, year, scenario))
        merged = add_unified_reward(pd.concat(year_frames, ignore_index=True))
        figures.append(plot_year(merged, year, fig_dir))
        frames.append(merged)
        for scenario, sub in merged.groupby("scenario"):
            last = sub.iloc[-1]
            metrics = baseline.metrics_from_snapshot(snapshot_map[str(scenario)], float(last["grain_yield_kg_ha"]))
            pfp_n = metrics["PFP_N_kg_kg"]
            summaries.append({
                "year": year,
                "scenario": scenario,
                "final_grain_kg_ha": float(last["grain_yield_kg_ha"]),
                "final_biomass_kg_ha": float(last["biomass_kg_ha"]),
                "irrigation_mm": float(metrics["actual_irrigation_mm"]),
                "nitrogen_kg_ha": float(metrics["actual_nitrogen_kg_ha"]),
                "etcp_mm": float(metrics["etcp_mm"]),
                "WP_ET_kg_m3": float(metrics["WP_ET_kg_m3"]),
                "PFP_N_kg_kg": float(pfp_n) if pd.notna(pfp_n) else np.nan,
                "max_wspd": float(sub["water_stress_index_wspd"].max()),
                "max_nstd": float(sub["nitrogen_stress_index_nstd"].max()),
                "unified_reward": float(last["unified_cumulative_reward"]),
            })
    all_daily = pd.concat(frames, ignore_index=True)
    actions = all_daily[(all_daily["irrigation_executed_mm"].gt(0)) | (all_daily["nitrogen_executed_kg_ha"].gt(0))].copy()
    all_daily.to_csv(tab_dir / "046_06_sya_five_scenario_daily.csv", index=False, encoding="utf-8-sig")
    actions.to_csv(tab_dir / "046_06_sya_five_scenario_management_events.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(tab_dir / "046_06_sya_five_scenario_season_summary.csv", index=False, encoding="utf-8-sig")
    doc = ROOT / "docs" / f"046_06_sya_{profile}_{cfg['task_id']}_{cfg['task_name']}_five_scenario_daily_ckpt{checkpoint}_record.md"
    doc.write_text("\n".join([f"# 046_06 SYA {profile} 五情景逐日过程图", "", f"- PPO checkpoint: `{checkpoint}`。", f"- 年份：{years}。", "- PPO 重放终值逐年与 046_02 保存终值核对；PlantGro.OUT 的 GWAD 为整数打印，因此允许不超过 0.5 kg/ha 的显示精度差，超出则中止。", "- 统一累计奖励：0.158×籽粒产量 − 1.1×灌溉 − 1.58×施氮；仅用于报告比较。", f"- 合并日值：`{rel(tab_dir / '046_06_sya_five_scenario_daily.csv')}`。", ""]), encoding="utf-8")
    print(json.dumps({"task": "046_06_sya_configured_five_scenario_daily_plots", "years": years, "checkpoint": checkpoint, "figures": [rel(path) for path in figures], "daily_csv": rel(tab_dir / "046_06_sya_five_scenario_daily.csv"), "summary_csv": rel(tab_dir / "046_06_sya_five_scenario_season_summary.csv"), "record_md": rel(doc)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
