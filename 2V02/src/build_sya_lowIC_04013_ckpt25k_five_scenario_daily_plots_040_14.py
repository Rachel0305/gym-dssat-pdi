from __future__ import annotations

import json
import math
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base032
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline034
import run_static_level1_four_baseline_rebuild_037_07 as baseline037
import run_sya_lowIC_ppo_early_irrigation_hard_reserve_mask_040_13 as run04013
from build_relaxed_success_five_scenario_daily_evidence_027_05 import (
    COLORS,
    STYLES,
    parse_management_events,
    parse_table,
    summary_metrics,
)
from run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 import snapshot_from_env


ROOT = Path(__file__).resolve().parents[1]
TASK = "040_14_sya_lowIC_04013_ckpt25k_five_scenario_daily_plots"
OUT = ROOT / "benchmark_results" / TASK
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
SNAP_DIR = OUT / "snapshots"
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

LOWIC_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
BASE039_SUMMARY = ROOT / "benchmark_results" / "039_02_original_vs_lowIC_three_baseline_audit" / "tables" / "039_02_full_summary.csv"
PPO04013_EVAL = ROOT / "benchmark_results" / "040_13_sya_lowIC_ppo_early_irrigation_hard_reserve_mask" / "evaluation" / "040_13_checkpoint_validation_summary.csv"
STATION_CODE = "SYA"
SITE = "SY"
STATION_NAME = "Shenyang"
SEED = 0
CHECKPOINT = 25_000
YEARS = list(range(2014, 2024))
MAX_STEPS = 260

SCENARIOS = ["null", "recorded_farmer", "dssat_auto", "official_extension_expert", "rl_candidate"]
LABELS = {
    "null": "Null",
    "recorded_farmer": "Recorded farmer",
    "dssat_auto": "DSSAT auto",
    "official_extension_expert": "Official expert",
    "rl_candidate": "MaskablePPO candidate",
}


@dataclass
class PlotCase:
    site: str
    station: str
    year: int
    seed: int
    checkpoint: int


def ensure_dirs() -> None:
    for path in [FIG_DIR, TAB_DIR, SNAP_DIR, OUT / "configs", OUT / "logs"]:
        path.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def load_039_lowic_snapshot(year: int, scenario: str) -> Path:
    if not BASE039_SUMMARY.exists():
        raise FileNotFoundError(BASE039_SUMMARY)
    df = pd.read_csv(BASE039_SUMMARY, keep_default_na=False)
    sub = df[
        df["condition"].astype(str).eq("lowIC")
        & df["station_code"].astype(str).eq(STATION_CODE)
        & pd.to_numeric(df["year"], errors="coerce").eq(int(year))
        & df["scenario"].astype(str).eq(scenario)
        & df["run_status"].astype(str).eq("ok")
    ]
    if len(sub) != 1:
        raise RuntimeError(f"Expected one 039_02 lowIC snapshot for {STATION_CODE}{year} {scenario}, found {len(sub)}")
    path = ROOT / str(sub.iloc[0]["snapshot_path"])
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def configure_recorded_runner() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[int, dict[str, float]]]]:
    baseline034.TASK_ID = "040_14"
    baseline034.OUT = OUT
    baseline034.DOC = DOC
    baseline034.PROMPT = PROMPT
    baseline034.MAX_STEPS = MAX_STEPS
    baseline034.replace_static_application_rows = baseline037.replace_static_application_rows_level1  # type: ignore[assignment]
    run_config = direct_ppo.load_yaml(baseline034.BASE_CONFIG)
    run_config = json.loads(json.dumps(run_config))
    run_config["seed"] = SEED
    run_config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    run_config["runtime"]["max_steps"] = MAX_STEPS
    split = pd.read_csv(baseline034.SPLIT_CSV, keep_default_na=False)
    split = split[split["station_code"].astype(str).eq(STATION_CODE) & split["year"].astype(int).isin(YEARS)].copy()
    split["selected_for_train"] = False
    split["selected_for_eval"] = True
    split["selection_reason"] = TASK
    env_config = baseline034.build_env_config(run_config, split)
    env_config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    env_config["runtime"]["max_steps"] = MAX_STEPS
    env_config["seed"] = SEED
    split.to_csv(OUT / "configs" / "040_14_selected_years.csv", index=False, encoding="utf-8-sig")
    direct_ppo.write_yaml(env_config, OUT / "configs" / "040_14_recorded_resolved_env_config.yaml")
    return run_config, env_config, baseline034.recorded_template_schedules()


def run_recorded_farmer_template(year: int, run_config: dict[str, Any], env_config: dict[str, Any], recorded_templates: dict[str, dict[int, dict[str, float]]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    scenario = "recorded_farmer_template"
    target = SNAP_DIR / "lowIC" / STATION_CODE / str(year) / "recorded_farmer"
    if target.exists() and (target / "Summary.OUT").exists():
        daily, summary = scenario_from_snapshot(year, "recorded_farmer", target)
        summary["source_status"] = "existing_040_14_recorded_snapshot"
        return daily, summary
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_INPUT_ROOT
    try:
        daily, summary, _checks = baseline034.evaluate_scenario(
            run_config,
            env_config,
            STATION_CODE,
            int(year),
            scenario,
            recorded_templates,
        )
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
    old_snapshot = OUT / "snapshots" / STATION_CODE / str(year) / scenario
    if not old_snapshot.exists():
        raise FileNotFoundError(old_snapshot)
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_snapshot), str(target))
    daily = daily.copy()
    daily["scenario"] = "recorded_farmer"
    summary = dict(summary)
    summary["scenario"] = "recorded_farmer"
    summary["snapshot_path"] = str(target.relative_to(ROOT)).replace("\\", "/")
    summary["source_status"] = "generated_040_14_recorded_farmer_template"
    summary["recorded_farmer_status"] = "template_reused_not_year_specific"
    return daily, summary


def ppo_model_row() -> pd.Series:
    df = pd.read_csv(PPO04013_EVAL, keep_default_na=False)
    sub = df[
        df["station_code"].astype(str).eq(STATION_CODE)
        & pd.to_numeric(df["checkpoint_step"], errors="coerce").eq(CHECKPOINT)
        & pd.to_numeric(df["year"], errors="coerce").eq(YEARS[0])
    ]
    if sub.empty:
        # Any year at the selected checkpoint has the same model path.
        sub = df[
            df["station_code"].astype(str).eq(STATION_CODE)
            & pd.to_numeric(df["checkpoint_step"], errors="coerce").eq(CHECKPOINT)
        ]
    if sub.empty:
        raise RuntimeError(f"No 040_13 evaluation row for checkpoint {CHECKPOINT}")
    return sub.iloc[0]


def replay_ppo_snapshot(year: int, config: dict[str, Any], env_config: dict[str, Any], model_path: Path) -> Path:
    target = SNAP_DIR / "lowIC" / STATION_CODE / str(year) / "rl_candidate"
    if target.exists() and (target / "Summary.OUT").exists():
        return target
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    old_wrapper = base032.StressAwareDiscreteWrapper
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    base032.StressAwareDiscreteWrapper = run04013.EarlyIrrigationHardReserveMaskWrapper
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = LOWIC_INPUT_ROOT
    env = None
    try:
        model = MaskablePPO.load(str(model_path), device="cpu")
        env = base032.make_env(config, env_config, STATION_CODE, int(year), SEED, f"{STATION_CODE}_{year}_040_14_ppo_replay", evaluation=True)
        obs, info = env.reset()
        done = False
        steps = 0
        while not done and steps < MAX_STEPS:
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
        if not done:
            raise RuntimeError(f"PPO replay did not finish within {MAX_STEPS} steps for {year}")
        tmp = snapshot_from_env(env)
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp, target)
        return target
    finally:
        if env is not None:
            env.close()
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
        base032.StressAwareDiscreteWrapper = old_wrapper


def configure_ppo_replay() -> tuple[dict[str, Any], dict[str, Any], Path]:
    run04013.patch_base_module(OUT, DOC, 100_000, [CHECKPOINT])
    config = base03222.load_config()
    split = base03222.load_split()
    selection = base03222.build_selection(split)
    env_config = direct_ppo.build_env_config(config, selection)
    env_config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    env_config["runtime"]["max_steps"] = MAX_STEPS
    row = ppo_model_row()
    model_path = ROOT / str(row["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    return config, env_config, model_path


def ppo_daily_from_04013(year: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Use 040_13 validation daily CSV as the authoritative PPO evidence.

    A 040_14 snapshot replay was intentionally not used for the PPO line because
    the replayed DSSAT OUT files did not reproduce the 040_13 validation CSV
    endpoint exactly.  The daily CSV is the source used to compute the 040_13
    checkpoint summary, so it is the safest source for management/yield/stress
    plotting here.  It does not contain SoilWat SWTD; that panel leaves the PPO
    soil-water line missing rather than filling it with inconsistent data.
    """

    eval_df = pd.read_csv(PPO04013_EVAL, keep_default_na=False)
    sub = eval_df[
        eval_df["station_code"].astype(str).eq(STATION_CODE)
        & pd.to_numeric(eval_df["year"], errors="coerce").eq(int(year))
        & pd.to_numeric(eval_df["checkpoint_step"], errors="coerce").eq(CHECKPOINT)
    ]
    if len(sub) != 1:
        raise RuntimeError(f"Expected one 040_13 PPO eval row for {STATION_CODE}{year} ckpt{CHECKPOINT}, found {len(sub)}")
    row = sub.iloc[0]
    daily_path = ROOT / str(row["daily_csv_path"])
    if not daily_path.exists():
        raise FileNotFoundError(daily_path)
    source = pd.read_csv(daily_path, keep_default_na=False)
    daily = pd.DataFrame(
        {
            "site": SITE,
            "station": STATION_NAME,
            "requested_year": int(year),
            "algorithm": "MaskablePPO",
            "seed": SEED,
            "checkpoint": CHECKPOINT,
            "scenario": "rl_candidate",
            "year_out": int(year),
            "doy": pd.to_numeric(source["doy"], errors="coerce"),
            "das": pd.to_numeric(source["dap"], errors="coerce"),
            "dap": pd.to_numeric(source["dap"], errors="coerce"),
            "rainfall_mm": pd.to_numeric(source["rain"], errors="coerce").fillna(0.0),
            "tmax_c": pd.to_numeric(source["tmax"], errors="coerce"),
            "tmin_c": pd.to_numeric(source["tmin"], errors="coerce"),
            "grain_yield_kg_ha": pd.to_numeric(source["grnwt"], errors="coerce").ffill().fillna(0.0),
            "biomass_kg_ha": pd.to_numeric(source["topwt"], errors="coerce").ffill().fillna(0.0),
            "water_stress_index_wspd": pd.to_numeric(source["swfac"], errors="coerce").ffill().fillna(0.0),
            "nitrogen_stress_index_nstd": pd.to_numeric(source["nstres"], errors="coerce").ffill().fillna(0.0),
            "soil_water_mm": np.nan,
            "irrigation_executed_mm": pd.to_numeric(source["safe_action_amir"], errors="coerce").fillna(0.0),
            "nitrogen_executed_kg_ha": pd.to_numeric(source["safe_action_anfer"], errors="coerce").fillna(0.0),
            "date": pd.to_datetime(source["date"], errors="coerce"),
        }
    )
    temp_ok = daily["tmax_c"].between(-60.0, 60.0, inclusive="both") & daily["tmin_c"].between(-70.0, 50.0, inclusive="both") & daily["tmax_c"].ge(daily["tmin_c"])
    daily["temperature_source_qc"] = np.where(temp_ok, "pass", "source_anomaly_preserved_not_imputed")
    daily["common_reward_step"] = -(daily["irrigation_executed_mm"] + 5.0 * daily["nitrogen_executed_kg_ha"])
    daily["cumulative_common_reward"] = daily["common_reward_step"].cumsum()
    final_y = float(pd.to_numeric(row["final_grnwt"], errors="coerce"))
    total_i = float(pd.to_numeric(row["total_irrigation"], errors="coerce"))
    total_n = float(pd.to_numeric(row["total_n"], errors="coerce"))
    summary = {
        "site": SITE,
        "station": STATION_NAME,
        "station_code": STATION_CODE,
        "year": int(year),
        "algorithm": "MaskablePPO",
        "seed": SEED,
        "checkpoint": CHECKPOINT,
        "scenario": "rl_candidate",
        "final_grain_kg_ha": final_y,
        "final_biomass_kg_ha": float(pd.to_numeric(daily["biomass_kg_ha"], errors="coerce").dropna().iloc[-1]),
        "rain_total_mm": float(daily["rainfall_mm"].sum()),
        "irrigation_event_total_mm": total_i,
        "nitrogen_event_total_kg_ha": total_n,
        "max_water_stress_wspd": float(daily["water_stress_index_wspd"].max()),
        "max_nitrogen_stress_nstd": float(daily["nitrogen_stress_index_nstd"].max()),
        "common_reward_total": float(daily["cumulative_common_reward"].iloc[-1]),
        "snapshot_path": "",
        "etcp_mm": math.nan,
        "wp_et_kg_m3": math.nan,
        "pfp_n_kg_kg": final_y / total_n if total_n > 0 else math.nan,
        "n_uptake_kg_ha": math.nan,
        "n_leaching_kg_ha": math.nan,
        "source_status": "authoritative_040_13_validation_daily_csv_no_swtd",
        "source_daily_csv": daily_path.relative_to(ROOT).as_posix(),
    }
    return daily, summary


def scenario_from_snapshot(year: int, scenario: str, snapshot: Path, null_yield: float | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    case = PlotCase(SITE, STATION_NAME, int(year), SEED, CHECKPOINT)
    weather = parse_table(snapshot / "Weather.OUT")
    plant = parse_table(snapshot / "PlantGro.OUT")
    water = parse_table(snapshot / "SoilWat.OUT")
    missing = [name for name, frame in (("Weather.OUT", weather), ("PlantGro.OUT", plant), ("SoilWat.OUT", water)) if frame.empty]
    if missing:
        raise RuntimeError(f"{SITE}{year} {scenario}: empty tables {missing}")
    wcols = {"YEAR": "year_out", "DOY": "doy", "DAS": "das", "PRED": "rainfall_mm", "TMXD": "tmax_c", "TMND": "tmin_c"}
    pcols = {"DOY": "doy", "DAS": "das", "DAP": "dap", "GWAD": "grain_yield_kg_ha", "CWAD": "biomass_kg_ha", "WSPD": "water_stress_index_wspd", "NSTD": "nitrogen_stress_index_nstd"}
    swcols = {"DOY": "doy", "DAS": "das", "SWTD": "soil_water_mm"}
    daily = weather[[c for c in wcols if c in weather]].rename(columns=wcols)
    p = plant[[c for c in pcols if c in plant]].rename(columns=pcols)
    sw = water[[c for c in swcols if c in water]].rename(columns=swcols)
    daily = daily.merge(p, on=["doy", "das"], how="left").merge(sw, on=["doy", "das"], how="left")
    for col in ["dap", "grain_yield_kg_ha", "biomass_kg_ha", "water_stress_index_wspd", "nitrogen_stress_index_nstd", "soil_water_mm"]:
        if col not in daily:
            daily[col] = np.nan
    daily[["dap", "grain_yield_kg_ha", "biomass_kg_ha", "water_stress_index_wspd", "nitrogen_stress_index_nstd", "soil_water_mm"]] = daily[["dap", "grain_yield_kg_ha", "biomass_kg_ha", "water_stress_index_wspd", "nitrogen_stress_index_nstd", "soil_water_mm"]].ffill().fillna(0.0)
    events = parse_management_events(snapshot / "MgmtEvent.OUT")
    daily["irrigation_executed_mm"] = 0.0
    daily["nitrogen_executed_kg_ha"] = 0.0
    if not events.empty:
        for _, event in events.iterrows():
            target = daily["doy"].eq(int(event["doy"]))
            col = "irrigation_executed_mm" if event["operation"] == "irrigation" else "nitrogen_executed_kg_ha"
            daily.loc[target, col] += float(event["amount"])
    final_yield = float(pd.to_numeric(plant["GWAD"], errors="coerce").dropna().iloc[-1])
    final_biomass = float(pd.to_numeric(plant["CWAD"], errors="coerce").dropna().iloc[-1])
    irrigation = float(daily["irrigation_executed_mm"].sum())
    nitrogen = float(daily["nitrogen_executed_kg_ha"].sum())
    daily["common_reward_step"] = -(daily["irrigation_executed_mm"] + 5.0 * daily["nitrogen_executed_kg_ha"])
    if null_yield is not None:
        daily.loc[daily.index[-1], "common_reward_step"] += max(0.0, final_yield - float(null_yield))
    daily["cumulative_common_reward"] = daily["common_reward_step"].cumsum()
    daily["date"] = pd.to_datetime(daily["year_out"].astype(int).astype(str) + daily["doy"].astype(int).astype(str).str.zfill(3), format="%Y%j", errors="coerce")
    temp_ok = daily["tmax_c"].between(-60.0, 60.0, inclusive="both") & daily["tmin_c"].between(-70.0, 50.0, inclusive="both") & daily["tmax_c"].ge(daily["tmin_c"])
    daily["temperature_source_qc"] = np.where(temp_ok, "pass", "source_anomaly_preserved_not_imputed")
    daily.insert(0, "site", case.site)
    daily.insert(1, "station", case.station)
    daily.insert(2, "requested_year", case.year)
    daily.insert(3, "algorithm", "MaskablePPO")
    daily.insert(4, "seed", case.seed if scenario == "rl_candidate" else np.nan)
    daily.insert(5, "checkpoint", case.checkpoint if scenario == "rl_candidate" else np.nan)
    daily.insert(6, "scenario", scenario)
    metrics = summary_metrics(snapshot, final_yield)
    summary = {
        "site": SITE,
        "station": STATION_NAME,
        "station_code": STATION_CODE,
        "year": int(year),
        "algorithm": "MaskablePPO",
        "seed": SEED if scenario == "rl_candidate" else np.nan,
        "checkpoint": CHECKPOINT if scenario == "rl_candidate" else np.nan,
        "scenario": scenario,
        "final_grain_kg_ha": final_yield,
        "final_biomass_kg_ha": final_biomass,
        "rain_total_mm": float(daily["rainfall_mm"].sum()),
        "irrigation_event_total_mm": irrigation,
        "nitrogen_event_total_kg_ha": nitrogen,
        "max_water_stress_wspd": float(daily["water_stress_index_wspd"].max()),
        "max_nitrogen_stress_nstd": float(daily["nitrogen_stress_index_nstd"].max()),
        "common_reward_total": float(daily["cumulative_common_reward"].iloc[-1]),
        "snapshot_path": snapshot.relative_to(ROOT).as_posix(),
        **metrics,
    }
    return daily, summary


def recompute_common_reward(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    frames = []
    for year, group in out.groupby("requested_year", sort=True):
        null_final = float(group[group["scenario"].eq("null")]["grain_yield_kg_ha"].dropna().iloc[-1])
        for scenario, sub in group.groupby("scenario", sort=False):
            sub = sub.sort_values("dap").copy()
            final_y = float(sub["grain_yield_kg_ha"].dropna().iloc[-1])
            sub["common_reward_step"] = -(sub["irrigation_executed_mm"] + 5.0 * sub["nitrogen_executed_kg_ha"])
            sub.loc[sub.index[-1], "common_reward_step"] += max(0.0, final_y - null_final)
            sub["cumulative_common_reward"] = sub["common_reward_step"].cumsum()
            frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def plot_daily(daily: pd.DataFrame, year: int) -> list[Path]:
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    weather = daily[daily["scenario"].eq("null")].sort_values("dap")
    ax = axes[0, 0]
    ax.bar(weather["dap"], weather["rainfall_mm"], color="#3977A8", alpha=0.58, label="Rain")
    ax.set_ylabel("Rain (mm)")
    ax2 = ax.twinx()
    ax2.plot(weather["dap"], weather["tmax_c"].where(weather["temperature_source_qc"].eq("pass")), color="#C23B32", lw=1.3, label="Tmax")
    ax2.plot(weather["dap"], weather["tmin_c"].where(weather["temperature_source_qc"].eq("pass")), color="#686868", lw=1.3, ls="--", label="Tmin")
    ax2.set_ylabel("Temperature (°C)")
    ax.set_title("Weather", loc="left", fontweight="bold")
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, ncol=3, fontsize=8, loc="upper right")
    for scenario in SCENARIOS:
        sub = daily[daily["scenario"].eq(scenario)].sort_values("dap")
        label = LABELS[scenario]
        color = COLORS.get(scenario, "#18864B")
        style = STYLES.get(scenario, "-")
        axes[0, 1].plot(sub["dap"], sub["soil_water_mm"], color=color, ls=style, lw=1.35, label=label)
        axes[1, 0].plot(sub["dap"], sub["water_stress_index_wspd"], color=color, ls=style, lw=1.35, label=label)
        axes[1, 1].plot(sub["dap"], sub["nitrogen_stress_index_nstd"], color=color, ls=style, lw=1.35, label=label)
        for ev_ax, col in ((axes[2, 0], "irrigation_executed_mm"), (axes[2, 1], "nitrogen_executed_kg_ha")):
            ev = sub[pd.to_numeric(sub[col], errors="coerce").gt(0)]
            marker = "D" if scenario == "rl_candidate" else "o"
            ev_ax.vlines(ev["dap"], 0, ev[col], color=color, lw=2, alpha=0.88)
            ev_ax.scatter(ev["dap"], ev[col], color=color, marker=marker, s=24, label=label)
        axes[3, 0].plot(sub["dap"], sub["grain_yield_kg_ha"], color=color, ls=style, lw=1.4, label=f"{label} grain")
        axes[3, 0].plot(sub["dap"], sub["biomass_kg_ha"], color=color, ls=style, lw=0.9, alpha=0.42)
        axes[3, 1].plot(sub["dap"], sub["cumulative_common_reward"], color=color, ls=style, lw=1.4, label=label)
    titles = [
        (axes[0, 1], "Soil water", "SWTD (mm)"),
        (axes[1, 0], "Water stress index", "WSPD (0=no stress)"),
        (axes[1, 1], "Nitrogen stress index", "NSTD (0=no stress)"),
        (axes[2, 0], "Irrigation events", "mm/event"),
        (axes[2, 1], "Nitrogen application events", "kg/ha/event"),
        (axes[3, 0], "Grain and biomass trajectories", "kg/ha"),
        (axes[3, 1], "Cumulative common reward", "kg/ha-equivalent"),
    ]
    for ax, title, ylabel in titles:
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(color="#E8E8E8", linewidth=0.65)
    for ax in axes[2, :]:
        handles, names = ax.get_legend_handles_labels()
        unique = dict(zip(names, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=7, ncol=2)
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[3, 1].legend(fontsize=7, ncol=2)
    axes[3, 0].text(0.01, 0.97, "Thin companion lines are biomass; thick lines are grain.", transform=axes[3, 0].transAxes, va="top", fontsize=7)
    axes[3, 0].set_xlabel("DAP")
    axes[3, 1].set_xlabel("DAP")
    fig.suptitle(f"SY{year} lowIC MaskablePPO 040_13 ckpt25k five-scenario daily process", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    base = FIG_DIR / f"040_14_sy{year}_lowIC_04013_ckpt25k_five_scenario_daily"
    png = base.with_suffix(".png")
    svg = base.with_suffix(".svg")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [png, svg]


def main() -> int:
    ensure_dirs()
    failures: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    figures: list[Path] = []

    rec_config, rec_env_config, recorded_templates = configure_recorded_runner()
    # Keep the model path check for provenance, but use the authoritative
    # 040_13 validation daily CSV for PPO plotting.  See ppo_daily_from_04013.
    _ppo_config, _ppo_env_config, model_path = configure_ppo_replay()

    for year in YEARS:
        try:
            snapshots = {
                "null": load_039_lowic_snapshot(year, "null"),
                "dssat_auto": load_039_lowic_snapshot(year, "dssat_auto"),
                "official_extension_expert": load_039_lowic_snapshot(year, "official_extension_expert"),
            }
            rec_daily, rec_summary = run_recorded_farmer_template(year, rec_config, rec_env_config, recorded_templates)
            year_daily: list[pd.DataFrame] = []
            year_summary: list[dict[str, Any]] = []
            null_daily, null_summary = scenario_from_snapshot(year, "null", snapshots["null"], null_yield=None)
            null_y = float(null_summary["final_grain_kg_ha"])
            year_daily.append(null_daily)
            year_summary.append(null_summary)
            for scenario in ["recorded_farmer", "dssat_auto", "official_extension_expert"]:
                if scenario == "recorded_farmer":
                    daily, summary = scenario_from_snapshot(year, scenario, SNAP_DIR / "lowIC" / STATION_CODE / str(year) / "recorded_farmer", null_yield=null_y)
                    summary["recorded_farmer_status"] = "template_reused_not_year_specific"
                else:
                    daily, summary = scenario_from_snapshot(year, scenario, snapshots[scenario], null_yield=null_y)
                year_daily.append(daily)
                year_summary.append(summary)
            ppo_daily, ppo_summary = ppo_daily_from_04013(year)
            year_daily.append(ppo_daily)
            year_summary.append(ppo_summary)
            merged_daily = recompute_common_reward(pd.concat(year_daily, ignore_index=True))
            figures.extend(plot_daily(merged_daily, year))
            daily_frames.append(merged_daily)
            summary_rows.extend(year_summary)
        except Exception:
            failures.append({"station_code": STATION_CODE, "year": year, "traceback": traceback.format_exc()[-4000:]})
            print(f"[040_14] failed {STATION_CODE}{year}", file=sys.stderr, flush=True)
            print(traceback.format_exc()[-1200:], file=sys.stderr, flush=True)

    daily_all = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    failures_df = pd.DataFrame(failures)
    daily_path = TAB_DIR / "040_14_sya_2014_2023_lowIC_04013_ckpt25k_five_scenario_daily.csv"
    summary_path = TAB_DIR / "040_14_sya_2014_2023_lowIC_04013_ckpt25k_five_scenario_summary.csv"
    failures_path = TAB_DIR / "040_14_failures.csv"
    daily_all.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    failures_df.to_csv(failures_path, index=False, encoding="utf-8-sig")

    ppo_actions = pd.DataFrame()
    if not daily_all.empty:
        ppo_actions = daily_all[
            daily_all["scenario"].eq("rl_candidate")
            & (
                pd.to_numeric(daily_all["irrigation_executed_mm"], errors="coerce").fillna(0).gt(0)
                | pd.to_numeric(daily_all["nitrogen_executed_kg_ha"], errors="coerce").fillna(0).gt(0)
            )
        ][["requested_year", "dap", "irrigation_executed_mm", "nitrogen_executed_kg_ha"]].copy()
        ppo_actions.to_csv(TAB_DIR / "040_14_ppo_ckpt25k_management_events.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 040_14 SYA lowIC 040_13 checkpoint25k 五情景日过程图记录",
        "",
        "## 结论先说",
        "",
        f"- 成功绘图年份数：{len(set(daily_all['requested_year'])) if not daily_all.empty else 0} / {len(YEARS)}。",
        f"- 失败年份数：{len(failures_df)}。",
        "- 本任务没有训练模型；PPO 为 040_13 checkpoint 25000 冻结重放。",
        "- PPO 曲线使用 040_13 validation daily CSV 作为权威来源；该 CSV 未保存 SoilWat SWTD，因此 Soil water 面板不绘制 PPO 线。",
        "- recorded farmer 使用项目既有模板复用基线，记录为 `template_reused_not_year_specific`。",
        "",
        "## PPO 措施事件",
        "",
        md_table(ppo_actions, max_rows=120),
        "",
        "## 五情景终值汇总",
        "",
        md_table(summary[["year", "scenario", "final_grain_kg_ha", "final_biomass_kg_ha", "irrigation_event_total_mm", "nitrogen_event_total_kg_ha", "wp_et_kg_m3", "pfp_n_kg_kg", "max_water_stress_wspd", "max_nitrogen_stress_nstd", "common_reward_total"]] if not summary.empty else summary, max_rows=120),
        "",
        "## 图件",
        "",
        *[f"- `{p.relative_to(ROOT).as_posix()}`" for p in figures if p.suffix.lower() == ".png"],
        "",
        "## 输出表",
        "",
        f"- `{daily_path.relative_to(ROOT).as_posix()}`",
        f"- `{summary_path.relative_to(ROOT).as_posix()}`",
        f"- `{(TAB_DIR / '040_14_ppo_ckpt25k_management_events.csv').relative_to(ROOT).as_posix()}`",
        f"- `{failures_path.relative_to(ROOT).as_posix()}`",
        "",
        "## 解释边界",
        "",
        "- 图中累计奖励统一按 `max(0, yield-null) - irrigation - 5*nitrogen` 重算，只用于五情景可比展示。",
        "- 本任务只检查 040_13 seed0 checkpoint25k 的措施合理性，不代表跨 seed 稳定性。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {
        "task": TASK,
        "years": YEARS,
        "successful_years": int(len(set(daily_all["requested_year"])) if not daily_all.empty else 0),
        "failed_years": int(len(failures_df)),
        "figures_png": [p.relative_to(ROOT).as_posix() for p in figures if p.suffix.lower() == ".png"],
        "summary_csv": summary_path.relative_to(ROOT).as_posix(),
        "daily_csv": daily_path.relative_to(ROOT).as_posix(),
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / "040_14_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0 if len(failures_df) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
