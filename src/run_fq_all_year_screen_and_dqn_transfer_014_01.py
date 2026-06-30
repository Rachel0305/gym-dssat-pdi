from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import gymnasium as gymnasium_base
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_yc_new_cultivar_forward_screening_013_01 import (
    parse_dssat_table,
    prepare_text_for_scenario,
    set_management_for_treatment,
)
import run_yc2014_linked_dqn_5k_multiseed_013_07 as yc_dqn


INPUT_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "FQ"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "fq_all_year_screen_and_dqn_transfer_014_01"
FIG_DIR = OUT_DIR / "figures"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-06-30_014_01_fq_all_year_screen_and_dqn_transfer_record.md"

SITE = "FQ"
STATION = "Fengqiu"
MZX_NAME = "CNFQ0801.MZX"
TEMPLATE_TRNO = 2
TEMPLATE_YEAR = 2008
YEARS = list(range(2000, 2024))
SCENARIOS_SCREEN = ["null", "recorded_shifted", "dssat_auto"]
SCENARIOS_DQN = ["dqn_linked_free_daily", "dqn_linked_agronomic_window"]

TIMESTEPS = 5000
SEEDS = [0]

IRRIGATION_BUDGET = 120.0
NITROGEN_BUDGET = 300.0
DAILY_IRRIGATION_CAP = 30.0
DAILY_NITROGEN_CAP = 100.0
MIN_INTERVAL_DAYS = 7

FREE_DAILY_WINDOWS = {
    "irrigation": [(1, 120)],
    "nitrogen": [(1, 120)],
}
AGRONOMIC_WINDOWS = {
    "irrigation": [(35, 65)],
    "nitrogen": [(1, 10), (35, 55)],
}

SCENARIO_LABELS = {
    "null": "Null",
    "recorded_shifted": "Recorded shifted",
    "dssat_auto": "DSSAT auto",
    "dqn_linked_free_daily": "DQN free daily",
    "dqn_linked_agronomic_window": "DQN agronomic window",
}
SCENARIO_COLORS = {
    "null": "#464C55",
    "recorded_shifted": "#CC6F47",
    "dssat_auto": "#5477C4",
    "dqn_linked_free_daily": "#386411",
    "dqn_linked_agronomic_window": "#7A2E8E",
}


def shift_fq2008_template_to_year(source: str, year: int) -> str:
    """Use FQ2008 treatment 2 as a shifted management template for a target weather year."""
    yy = f"{year % 100:02d}"
    text = source
    text = text.replace("CNFQ0801", f"CNFQ{yy}01")
    text = text.replace("CNFQ2008", f"CNFQ{year}")
    text = text.replace("Sim2008", f"Sim{year}")
    text = text.replace(" 2008", f" {year}")
    text = re.sub(r"\b08(\d{3})\b", rf"{yy}\1", text)
    return text


def prepare_text_for_shifted_scenario(year: int, scenario: str) -> str:
    source = (INPUT_ROOT / MZX_NAME).read_text(encoding="latin-1", errors="ignore")
    shifted = shift_fq2008_template_to_year(source, year)
    if scenario == "recorded_shifted":
        return shifted
    if scenario == "null":
        return prepare_text_for_scenario(shifted, TEMPLATE_TRNO, "null")
    if scenario == "dssat_auto":
        return prepare_text_for_scenario(shifted, TEMPLATE_TRNO, "dssat_auto")
    if scenario.startswith("dqn_"):
        return set_management_for_treatment(shifted, TEMPLATE_TRNO, "L", "L")
    raise ValueError(scenario)


def prepare_run_dir(year: int, scenario: str, seed: int = 0) -> Path:
    run_dir = OUT_DIR / "runs" / str(year) / f"seed{seed}" / scenario
    input_dir = run_dir / "input"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    text = prepare_text_for_shifted_scenario(year, scenario)
    filex = input_dir / f"CNFQ{year}_{scenario}.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")

    for src in INPUT_ROOT.iterdir():
        if src.is_file() and src.name != MZX_NAME:
            shutil.copyfile(src, input_dir / src.name)

    aux = [str(p) for p in input_dir.iterdir() if p.suffix.upper() in {".CUL", ".SOL", ".WTH", ".MZA", ".MZT"}]
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": seed,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": TEMPLATE_TRNO,
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir


def make_raw_env(env_args: dict[str, Any]):
    import gym
    from sb3_wrapper import GymDssatWrapper

    raw = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
    return yc_dqn.LazyScalarGymDssatWrapper(GymDssatWrapper(raw))


def parse_events(run_dir: Path, scenario: str, snapshot_name: str = "pdi_tmp_snapshot_eval") -> pd.DataFrame:
    path = run_dir / snapshot_name / "MgmtEvent.OUT"
    rows = []
    if not path.exists():
        return pd.DataFrame(columns=["scenario", "dap", "amount", "unit", "operation"])
    for raw in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        if "Irrigation" not in raw and "Fertil" not in raw and "Nitrogen" not in raw:
            continue
        parts = raw.split()
        dap = np.nan
        if len(parts) >= 7:
            try:
                dap = int(parts[6])
            except ValueError:
                pass
        amount = 0.0
        unit = ""
        m = re.search(r"([-+]?\d+(?:\.\d*)?)\s*(mm|kg(?:\[[A-Za-z]+\])?/ha|kg)", raw)
        if m:
            amount = float(m.group(1))
            unit = m.group(2)
        rows.append({"scenario": scenario, "dap": dap, "amount": amount, "unit": unit, "operation": raw.strip()})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(subset=["scenario", "dap", "amount", "unit", "operation"])
    return out


def parse_weather(year: int) -> pd.DataFrame:
    wth = INPUT_ROOT / f"CNFQ{year % 100:02d}01.WTH"
    rows = []
    if not wth.exists():
        return pd.DataFrame(columns=["doy", "rain"])
    header: list[str] = []
    in_data = False
    for line in wth.read_text(encoding="latin-1", errors="ignore").splitlines():
        if line.startswith("@"):
            header = line.replace("@", "", 1).split()
            in_data = "DATE" in header
            continue
        if in_data and line.strip() and not line.startswith("*") and not line.startswith("!"):
            parts = line.split()
            if len(parts) < len(header):
                continue
            rec = dict(zip(header, parts[: len(header)]))
            try:
                rows.append({"doy": int(str(rec.get("DATE"))[-3:]), "rain": float(rec.get("RAIN", 0))})
            except ValueError:
                pass
    return pd.DataFrame(rows)


def attach_events_to_daily(daily: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["irrigation_mm"] = 0.0
    daily["fertilizer_kg_ha"] = 0.0
    if not events.empty:
        event_map = events.copy()
        event_map["irrigation_mm"] = np.where(event_map["unit"].eq("mm"), event_map["amount"], 0.0)
        event_map["fertilizer_kg_ha"] = np.where(event_map["unit"].str.contains("kg", na=False), event_map["amount"], 0.0)
        event_map = event_map.groupby("dap", as_index=False)[["irrigation_mm", "fertilizer_kg_ha"]].sum()
        daily = daily.merge(event_map, on="dap", how="left", suffixes=("", "_event"))
        daily["irrigation_mm"] = daily["irrigation_mm_event"].fillna(daily["irrigation_mm"])
        daily["fertilizer_kg_ha"] = daily["fertilizer_kg_ha_event"].fillna(daily["fertilizer_kg_ha"])
        daily = daily.drop(columns=["irrigation_mm_event", "fertilizer_kg_ha_event"])
    return daily


def run_zero_action_scenario(year: int, scenario: str) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    run_dir = prepare_run_dir(year, scenario, seed=0)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    env = make_raw_env(env_args)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        for step in range(380):
            action = {"amir": 0.0, "anfer": 0.0}
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, action)
            obs, reward, terminated, truncated, info = env.step(norm)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "site": SITE,
                    "station": STATION,
                    "year": year,
                    "scenario": scenario,
                    "step": step,
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "dap": scalar(latest.get("dap")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "reward": scalar(reward),
                }
            )
            if terminated or truncated:
                break
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, run_dir / "pdi_tmp_snapshot_eval", dirs_exist_ok=True)
        env.close()

    daily = pd.DataFrame(rows)
    events = parse_events(run_dir, scenario)
    daily = attach_events_to_daily(daily, events)
    plantgro = parse_dssat_table(run_dir / "pdi_tmp_snapshot_eval" / "PlantGro.OUT")
    summary = {
        "site": SITE,
        "station": STATION,
        "year": year,
        "scenario": scenario,
        "final_grain_kg_ha": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan,
        "final_biomass_kg_ha": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan,
        "final_dap": float(daily["dap"].dropna().iloc[-1]) if not daily.empty else np.nan,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "event_irrigation_total": float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0,
        "event_fertilizer_total": float(events.loc[events["unit"].str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0,
        "run_dir": str(run_dir),
    }
    return daily, summary, events.assign(site=SITE, station=STATION, year=year)


def child_zero_case(year: int, scenario: str) -> None:
    run_dir = prepare_run_dir(year, scenario, seed=0)
    daily, summary, events = run_zero_action_scenario(year, scenario)
    daily.to_csv(run_dir / "014_01_case_daily.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(run_dir / "014_01_case_summary.csv", index=False, encoding="utf-8-sig")
    events.to_csv(run_dir / "014_01_case_events.csv", index=False, encoding="utf-8-sig")


def run_screening(case_timeout: int = 120) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_daily = []
    summaries = []
    all_events = []
    for year in YEARS:
        for scenario in SCENARIOS_SCREEN:
            print(f"[screen] {year} {scenario}", flush=True)
            run_dir = OUT_DIR / "runs" / str(year) / "seed0" / scenario
            try:
                proc = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--child-zero", str(year), scenario],
                    cwd=str(PROJECT_ROOT),
                    timeout=case_timeout,
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"child returncode={proc.returncode}; stderr={proc.stderr[-800:]}")
                daily_path = run_dir / "014_01_case_daily.csv"
                summary_path = run_dir / "014_01_case_summary.csv"
                events_path = run_dir / "014_01_case_events.csv"
                if daily_path.exists():
                    all_daily.append(pd.read_csv(daily_path))
                if summary_path.exists():
                    summaries.extend(pd.read_csv(summary_path).to_dict("records"))
                if events_path.exists():
                    ev = pd.read_csv(events_path)
                    if not ev.empty:
                        all_events.append(ev)
            except Exception as exc:
                summaries.append(
                    {
                        "site": SITE,
                        "station": STATION,
                        "year": year,
                        "scenario": scenario,
                        "error": repr(exc),
                        "final_grain_kg_ha": np.nan,
                        "run_dir": str(run_dir),
                    }
                )
                print(f"[screen:error] {year} {scenario}: {exc}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_df = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    event_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    daily_df.to_csv(OUT_DIR / "014_01_fq_all_year_screening_daily.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_DIR / "014_01_fq_all_year_screening_summary.csv", index=False, encoding="utf-8-sig")
    event_df.to_csv(OUT_DIR / "014_01_fq_all_year_screening_events.csv", index=False, encoding="utf-8-sig")
    selected = select_years(summary_df)
    selected.to_csv(OUT_DIR / "014_01_fq_selected_years.csv", index=False, encoding="utf-8-sig")
    return daily_df, summary_df, event_df


def select_years(summary_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = summary_df.copy()
    summary_df["scenario"] = summary_df["scenario"].fillna("null").replace("", "null")
    for col in ["year", "final_grain_kg_ha", "max_water_stress", "max_nitrogen_stress", "final_dap"]:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")
    piv = summary_df.pivot_table(index="year", columns="scenario", values="final_grain_kg_ha", aggfunc="first")
    for col in ["null", "recorded_shifted", "dssat_auto"]:
        if col not in piv.columns:
            piv[col] = np.nan
    stress = summary_df[summary_df["scenario"].eq("null")].set_index("year")[["max_water_stress", "max_nitrogen_stress", "final_dap"]]
    out = piv.join(stress, how="left").reset_index()
    out["best_reference"] = out[["recorded_shifted", "dssat_auto"]].max(axis=1)
    out["management_gain"] = out["best_reference"] - out["null"]
    out["stress_space"] = out[["max_water_stress", "max_nitrogen_stress"]].max(axis=1)
    out["usable_for_training_display"] = (
        out["null"].ge(5000)
        & out["final_dap"].ge(90)
        & out["management_gain"].ge(200)
        & out["stress_space"].ge(0.05)
    )
    out = out.sort_values(["usable_for_training_display", "management_gain", "stress_space"], ascending=False)
    return out


class FQDiscreteBudgetedWrapper(yc_dqn.YCDiscreteBudgetedWrapper):
    pass


def set_yc_dqn_globals(seed: int) -> None:
    yc_dqn.SEED = seed
    yc_dqn.TIMESTEPS = TIMESTEPS
    yc_dqn.IRRIGATION_BUDGET = IRRIGATION_BUDGET
    yc_dqn.NITROGEN_BUDGET = NITROGEN_BUDGET
    yc_dqn.DAILY_IRRIGATION_CAP = DAILY_IRRIGATION_CAP
    yc_dqn.DAILY_NITROGEN_CAP = DAILY_NITROGEN_CAP
    yc_dqn.MIN_INTERVAL_DAYS = MIN_INTERVAL_DAYS
    yc_dqn.WATER_COST = 1.0
    yc_dqn.NITROGEN_COST = 5.0


def run_dqn(year: int, scenario: str, seed: int, windows: dict[str, list[tuple[int, int]]]) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    set_yc_dqn_globals(seed)
    print(f"[dqn] {year} {scenario} seed{seed}", flush=True)
    run_dir = prepare_run_dir(year, scenario, seed=seed)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))

    from stable_baselines3 import DQN

    train_env = yc_dqn.EconomicRewardWrapper(
        FQDiscreteBudgetedWrapper(make_raw_env(env_args), windows["irrigation"], windows["nitrogen"])
    )
    model = DQN(
        "MlpPolicy",
        train_env,
        verbose=0,
        seed=seed,
        learning_rate=1e-4,
        buffer_size=10000,
        learning_starts=50,
        batch_size=32,
        train_freq=1,
        gradient_steps=1,
        gamma=0.99,
        exploration_fraction=0.35,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
    )
    try:
        model.learn(total_timesteps=int(TIMESTEPS), progress_bar=False)
        model.save(str(run_dir / "dqn_model"))
    finally:
        train_env.close()

    eval_env = yc_dqn.EconomicRewardWrapper(
        FQDiscreteBudgetedWrapper(make_raw_env(env_args), windows["irrigation"], windows["nitrogen"])
    )
    rows: list[dict[str, Any]] = []
    try:
        obs, info = eval_env.reset()
        for step in range(260):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            latest = latest_observation_dict(eval_env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            safe_action = dict(getattr(eval_env.env, "last_safe_real_action", {}) or {})
            rows.append(
                {
                    "site": SITE,
                    "station": STATION,
                    "year": year,
                    "scenario": scenario,
                    "seed": seed,
                    "step": step,
                    "yrdoy": yrdoy,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "dap": scalar(latest.get("dap")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "irrigation_mm": float(safe_action.get("amir", 0.0)),
                    "fertilizer_kg_ha": float(safe_action.get("anfer", 0.0)),
                    "action_index": int(np.asarray(action).item()),
                    "reward": float(reward),
                    "used_irrigation": float(info.get("used_irrigation", np.nan)) if isinstance(info, dict) else np.nan,
                    "used_nitrogen": float(info.get("used_nitrogen", np.nan)) if isinstance(info, dict) else np.nan,
                }
            )
            if terminated or truncated:
                break
    finally:
        tmp = getattr(eval_env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, run_dir / "pdi_tmp_snapshot_eval", dirs_exist_ok=True)
        eval_env.close()

    daily = pd.DataFrame(rows)
    events = parse_events(run_dir, scenario)
    plantgro = parse_dssat_table(run_dir / "pdi_tmp_snapshot_eval" / "PlantGro.OUT")
    summary = {
        "site": SITE,
        "station": STATION,
        "year": year,
        "scenario": scenario,
        "seed": seed,
        "action_irrigation_total": float(daily["irrigation_mm"].sum()) if not daily.empty else 0.0,
        "action_fertilizer_total": float(daily["fertilizer_kg_ha"].sum()) if not daily.empty else 0.0,
        "event_irrigation_total": float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0,
        "event_fertilizer_total": float(events.loc[events["unit"].str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0,
        "final_grain_kg_ha": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty else np.nan,
        "final_biomass_kg_ha": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty else np.nan,
        "max_water_stress": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nitrogen_stress": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "run_dir": str(run_dir),
    }
    return daily, summary, events.assign(site=SITE, station=STATION, year=year, seed=seed)


def choose_dqn_for_plot(dqn_summary: pd.DataFrame, year: int) -> str:
    sub = dqn_summary[dqn_summary["year"].eq(year)].copy()
    if sub.empty:
        return "dqn_linked_free_daily"
    sub["economic_score"] = sub["final_grain_kg_ha"] - sub["action_irrigation_total"] - 5.0 * sub["action_fertilizer_total"]
    return str(sub.sort_values(["economic_score", "final_grain_kg_ha"], ascending=False).iloc[0]["scenario"])


def run_dqn_for_selected(selected_years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_daily = []
    summaries = []
    all_events = []
    for year in selected_years:
        for seed in SEEDS:
            for scenario, windows in [
                ("dqn_linked_free_daily", FREE_DAILY_WINDOWS),
                ("dqn_linked_agronomic_window", AGRONOMIC_WINDOWS),
            ]:
                try:
                    daily, summary, events = run_dqn(year, scenario, seed, windows)
                    all_daily.append(daily)
                    summaries.append(summary)
                    all_events.append(events)
                except Exception as exc:
                    summaries.append({"site": SITE, "station": STATION, "year": year, "scenario": scenario, "seed": seed, "error": repr(exc)})
                    print(f"[dqn:error] {year} {scenario} seed{seed}: {exc}", flush=True)
    dqn_daily = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    dqn_summary = pd.DataFrame(summaries)
    dqn_events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    dqn_daily.to_csv(OUT_DIR / "014_01_fq_dqn_daily.csv", index=False, encoding="utf-8-sig")
    dqn_summary.to_csv(OUT_DIR / "014_01_fq_dqn_summary.csv", index=False, encoding="utf-8-sig")
    dqn_events.to_csv(OUT_DIR / "014_01_fq_dqn_events.csv", index=False, encoding="utf-8-sig")
    return dqn_daily, dqn_summary, dqn_events


def build_four_scenario_outputs(selected_years: list[int]) -> None:
    screen_daily = pd.read_csv(OUT_DIR / "014_01_fq_all_year_screening_daily.csv", keep_default_na=False)
    screen_daily["scenario"] = screen_daily["scenario"].replace("", "null")
    for col in ["year", "doy", "dap", "swfac", "nstres", "grnwt", "topwt", "irrigation_mm", "fertilizer_kg_ha"]:
        if col in screen_daily.columns:
            screen_daily[col] = pd.to_numeric(screen_daily[col], errors="coerce")
    screen_events = pd.read_csv(OUT_DIR / "014_01_fq_all_year_screening_events.csv", keep_default_na=False) if (OUT_DIR / "014_01_fq_all_year_screening_events.csv").exists() else pd.DataFrame()
    if not screen_events.empty and "scenario" in screen_events.columns:
        screen_events["scenario"] = screen_events["scenario"].replace("", "null")
    dqn_daily = pd.read_csv(OUT_DIR / "014_01_fq_dqn_daily.csv", keep_default_na=False) if (OUT_DIR / "014_01_fq_dqn_daily.csv").exists() else pd.DataFrame()
    for col in ["year", "doy", "dap", "swfac", "nstres", "grnwt", "topwt", "irrigation_mm", "fertilizer_kg_ha"]:
        if col in dqn_daily.columns:
            dqn_daily[col] = pd.to_numeric(dqn_daily[col], errors="coerce")
    dqn_summary = pd.read_csv(OUT_DIR / "014_01_fq_dqn_summary.csv") if (OUT_DIR / "014_01_fq_dqn_summary.csv").exists() else pd.DataFrame()
    dqn_events = pd.read_csv(OUT_DIR / "014_01_fq_dqn_events.csv") if (OUT_DIR / "014_01_fq_dqn_events.csv").exists() else pd.DataFrame()

    frames = []
    events_frames = []
    for year in selected_years:
        base = screen_daily[(screen_daily["year"].eq(year)) & (screen_daily["scenario"].isin(SCENARIOS_SCREEN))].copy()
        chosen = choose_dqn_for_plot(dqn_summary, year)
        dqn = dqn_daily[(dqn_daily["year"].eq(year)) & (dqn_daily["scenario"].eq(chosen))].copy()
        frame = pd.concat([base, dqn], ignore_index=True, sort=False)
        frames.append(frame)
        ev_base = screen_events[(screen_events.get("year", pd.Series(dtype=float)).eq(year)) & (screen_events.get("scenario", pd.Series(dtype=str)).isin(SCENARIOS_SCREEN))].copy() if not screen_events.empty else pd.DataFrame()
        ev_dqn = dqn_events[(dqn_events.get("year", pd.Series(dtype=float)).eq(year)) & (dqn_events.get("scenario", pd.Series(dtype=str)).eq(chosen))].copy() if not dqn_events.empty else pd.DataFrame()
        events_frames.append(pd.concat([ev_base, ev_dqn], ignore_index=True, sort=False))
        plot_year(frame, year, chosen, FIG_DIR / f"fq_{year}_four_scenario_process.png")

    daily_out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    events_out = pd.concat(events_frames, ignore_index=True) if events_frames else pd.DataFrame()
    daily_out.to_csv(OUT_DIR / "014_01_fq_selected_four_scenario_daily.csv", index=False, encoding="utf-8-sig")
    events_out.to_csv(OUT_DIR / "014_01_fq_selected_four_scenario_events.csv", index=False, encoding="utf-8-sig")


def plot_year(frame: pd.DataFrame, year: int, chosen_dqn: str, out_path: Path) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sub = frame.sort_values(["scenario", "dap"]).copy()
    if sub.empty:
        return
    max_dap = int(np.nanmax(sub["dap"])) if not sub["dap"].dropna().empty else 130
    rain = parse_weather(year)
    if not rain.empty:
        pdate_doy = int(sub["doy"].dropna().min() - sub["dap"].dropna().min()) if not sub["doy"].dropna().empty else int(rain["doy"].min())
        rain["dap"] = rain["doy"] - pdate_doy
        rain = rain[(rain["dap"] >= 0) & (rain["dap"] <= max_dap + 5)]

    order = ["null", "recorded_shifted", "dssat_auto", chosen_dqn]
    fig, axes = plt.subplots(5, 1, figsize=(15.5, 13.0), sharex=True, gridspec_kw={"height_ratios": [0.85, 1, 1, 1, 1.15]})
    fig.suptitle(f"FQ {year} four-scenario process plot", x=0.08, ha="left", fontsize=15, fontweight="bold")
    if not rain.empty:
        axes[0].bar(rain["dap"], rain["rain"], width=1.0, color="#C5CAD3", edgecolor="#7A828F", linewidth=0.35, label="Rainfall")
    axes[0].set_ylabel("Rain\n(mm)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)

    for scenario in order:
        s = sub[sub["scenario"].eq(scenario)].copy()
        if s.empty:
            continue
        color = SCENARIO_COLORS[scenario]
        label = SCENARIO_LABELS[scenario]
        axes[1].plot(s["dap"], s["swfac"], color=color, lw=2, label=label)
        axes[2].plot(s["dap"], s["nstres"], color=color, lw=2, label=label)
        axes[4].plot(s["dap"], s["grnwt"], color=color, lw=2)
        axes[4].plot(s["dap"], s["topwt"], color=color, lw=1.6, ls="--", alpha=0.7)
        mg_i = s[s["irrigation_mm"].fillna(0) > 1e-6]
        mg_n = s[s["fertilizer_kg_ha"].fillna(0) > 1e-6]
        if not mg_i.empty:
            axes[3].vlines(mg_i["dap"], 0, mg_i["irrigation_mm"], colors=color, linewidth=2.4, alpha=0.95)
        if not mg_n.empty:
            axes[3].scatter(mg_n["dap"], mg_n["fertilizer_kg_ha"], marker="^", s=52, color=color, edgecolor="#FFFFFF", linewidth=0.6, zorder=4)

    axes[1].set_ylabel("Water\nstress")
    axes[2].set_ylabel("Nitrogen\nstress")
    axes[3].set_ylabel("Mgmt\namount")
    axes[4].set_ylabel("kg/ha")
    axes[4].set_xlabel("DAP")
    axes[1].legend(loc="upper left", ncol=2, frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(True, axis="x", color="#E5EAF1", linewidth=0.8)
        ax.grid(True, axis="y", color="#EEF2F6", linewidth=0.6, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[3].set_title("Management events: irrigation as vertical lines; fertilization as triangle markers", loc="left", fontsize=10)
    axes[4].set_title("Crop outcome: solid = grain weight, dashed = aboveground biomass", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_record(selected_years: list[int] | None = None) -> None:
    summary_path = OUT_DIR / "014_01_fq_all_year_screening_summary.csv"
    selected_path = OUT_DIR / "014_01_fq_selected_years.csv"
    dqn_path = OUT_DIR / "014_01_fq_dqn_summary.csv"
    lines = [
        "# 014_01 封丘站全年份筛选与 DQN 方法迁移记录",
        "",
        "## 执行目的",
        "",
        "筛选封丘站 2000–2023 年中具有水氮优化空间的年份，并将禹城站 linked DQN 离散动作方法迁移到封丘站。",
        "",
        "## 输入与口径",
        "",
        f"- 输入目录：`{INPUT_ROOT}`",
        f"- 基础 MZX：`{MZX_NAME}`",
        f"- 专家迁移模板：FQ{TEMPLATE_YEAR} treatment {TEMPLATE_TRNO}",
        "- 专家迁移方式：将 FQ2008 管理日期和天气站代码平移到目标年份。",
        "- DQN：I120/N300，单次 I30/N100，最小操作间隔 7 天，5K timesteps，seed0。",
        "",
        "## 输出文件",
        "",
        f"- 筛选汇总：`{summary_path}`",
        f"- 候选年份排序：`{selected_path}`",
        f"- DQN 汇总：`{dqn_path}`",
        f"- 四情景日值：`{OUT_DIR / '014_01_fq_selected_four_scenario_daily.csv'}`",
        f"- 四情景事件：`{OUT_DIR / '014_01_fq_selected_four_scenario_events.csv'}`",
        f"- 图目录：`{FIG_DIR}`",
        "",
    ]
    if selected_path.exists():
        selected = pd.read_csv(selected_path).head(8)
        lines += ["## 筛选结果预览", "", "```text", selected.to_string(index=False), "```", ""]
    if dqn_path.exists():
        dqn = pd.read_csv(dqn_path)
        cols = [c for c in ["year", "scenario", "seed", "action_irrigation_total", "action_fertilizer_total", "final_grain_kg_ha", "final_biomass_kg_ha", "max_water_stress", "max_nitrogen_stress", "error"] if c in dqn.columns]
        lines += ["## DQN 结果", "", "```text", dqn[cols].to_string(index=False), "```", ""]
    if selected_years:
        lines += ["## 已生成四情景图年份", "", ", ".join(map(str, selected_years)), ""]
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--run-dqn", action="store_true")
    parser.add_argument("--years", nargs="*", type=int)
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--child-zero", nargs=2, metavar=("YEAR", "SCENARIO"))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.child_zero:
        child_zero_case(int(args.child_zero[0]), args.child_zero[1])
        return

    if args.screen_only or not (OUT_DIR / "014_01_fq_all_year_screening_summary.csv").exists():
        run_screening()
        write_record()
        if args.screen_only:
            return

    if args.years:
        selected_years = args.years[:2]
    else:
        selected = pd.read_csv(OUT_DIR / "014_01_fq_selected_years.csv", keep_default_na=False)
        selected_years = selected["year"].dropna().astype(int).head(2).tolist()

    global SEEDS
    if args.seeds:
        SEEDS = [int(seed) for seed in args.seeds]

    if args.run_dqn:
        run_dqn_for_selected(selected_years)

    if args.plot_only or args.run_dqn:
        build_four_scenario_outputs(selected_years)
        write_record(selected_years)


if __name__ == "__main__":
    main()
