from __future__ import annotations

import argparse
import json
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Inches, Pt
except ModuleNotFoundError:
    Presentation = None
    RGBColor = None
    PP_ALIGN = None
    Inches = None
    Pt = None

from offline_schedule_policy import DeterministicSchedulePolicy, ScheduleEvent
from ppo_action_safety import ActionSafetyState, apply_action_safety, normalize_action, update_action_safety_state
from ppo_evaluate import latest_observation_dict, make_env, scalar
from ppo_plot_results import plot_episode
from ppo_safe_rendering import PROJECT_ROOT, SITE_INFO


CONFIG_PATH = PROJECT_ROOT / "experiments" / "ppo_observed_years" / "config_all_year_weather_calibration_validation.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation"
DOC_MD = PROJECT_ROOT / "docs" / "2026-06-06_all_year_weather_calibration_validation_report.md"
DOC_PPT = PROJECT_ROOT / "docs" / "2026-06-06_all_year_weather_calibration_validation_report.pptx"

STATION_NAMES = {
    "HLA": "Hailun",
    "SYA": "Shenyang",
    "LCA": "Luancheng",
    "YCA": "Yucheng",
    "FQA": "Fengqiu",
}


@dataclass(frozen=True)
class Treatment:
    treatment_id: str
    treatment_name: str
    nitrogen_by_dap: dict[int, float]
    irrigation_by_dap: dict[int, float]


def ensure_dirs() -> None:
    for sub in [
        "configs",
        "weather_inventory",
        "cultivar_calibration_plan",
        "stress_diagnostics",
        "scenario_pool",
        "daily_outputs",
        "evaluation",
        "figures",
        "reports",
        "rendered_inputs",
        "logs",
    ]:
        (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def df_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return ""
    work = df.copy()
    work.columns = [str(col) for col in work.columns]
    if max_rows is not None:
        work = work.head(max_rows)
    for col in work.select_dtypes(include=["float", "int"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    for col in work.columns:
        work[col] = work[col].map(lambda v: "" if pd.isna(v) else str(v))
    header = "| " + " | ".join(work.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in work.astype(str).values.tolist()]
    return "\n".join([header, sep, *rows])


def key_to_int_dict(data: dict | None) -> dict[int, float]:
    if not data:
        return {}
    return {int(k): float(v) for k, v in data.items()}


def load_treatments(config: dict) -> list[Treatment]:
    return [
        Treatment(
            treatment_id=str(item["treatment_id"]),
            treatment_name=str(item["treatment_name"]),
            nitrogen_by_dap=key_to_int_dict(item.get("nitrogen_by_dap")),
            irrigation_by_dap=key_to_int_dict(item.get("irrigation_by_dap")),
        )
        for item in config["fixed_management_treatments"]
    ]


def treatment_policy(treatment: Treatment) -> DeterministicSchedulePolicy:
    events: list[ScheduleEvent] = []
    for dap, amount in treatment.nitrogen_by_dap.items():
        events.append(ScheduleEvent(name=f"N{dap}", dap=int(dap), nitrogen=float(amount)))
    for dap, amount in treatment.irrigation_by_dap.items():
        events.append(ScheduleEvent(name=f"I{dap}", dap=int(dap), irrigation=float(amount)))
    return DeterministicSchedulePolicy(events)


def clip_to_action_space(env, real_action: dict[str, float]) -> tuple[dict[str, float], str]:
    spaces = getattr(env.formator.action_space_dict, "spaces", env.formator.action_space_dict)
    clipped: dict[str, float] = {}
    notes: list[str] = []
    for name in env.formator.action_names:
        space = spaces[name]
        low = float(np.asarray(space.low).flatten()[0])
        high = float(np.asarray(space.high).flatten()[0])
        raw = float(real_action.get(name, 0.0))
        value = min(max(raw, low), high)
        clipped[name] = value
        if abs(raw - value) > 1e-9:
            notes.append(f"{name}_env_bound_clip_{raw:g}_to_{value:g}")
    return clipped, ";".join(notes)


def profit(final_grnwt: float, irrigation: float, nitrogen: float, config: dict, low_water: bool = False) -> float:
    econ = config["economics"]
    water_cost = float(econ["water_cost_low"] if low_water else econ["water_cost_default"])
    return float(econ["grain_value_coef"]) * float(final_grnwt) - water_cost * float(irrigation) - float(econ["n_cost"]) * float(nitrogen)


def representative_phenology() -> tuple[dict[str, dict[str, int]], pd.DataFrame]:
    path = PROJECT_ROOT / "data" / "observed_phenology_dates_standardized.csv"
    if not path.exists():
        path = PROJECT_ROOT / "data" / "observed_phenology_dates.csv"
    observed = pd.read_csv(path)
    observed["planting_date"] = pd.to_datetime(observed["planting_date"])
    observed["harvest_date"] = pd.to_datetime(observed["harvest_date"])
    observed["planting_doy"] = observed["planting_date"].dt.dayofyear
    observed["harvest_days"] = (observed["harvest_date"] - observed["planting_date"]).dt.days + 1
    reps: dict[str, dict[str, int]] = {}
    for station, group in observed.groupby("station_code"):
        reps[station] = {
            "planting_doy": int(round(group["planting_doy"].median())),
            "harvest_days": int(round(group["harvest_days"].median())),
        }
    return reps, observed


def date_from_doy(year: int, doy: int) -> pd.Timestamp:
    return pd.Timestamp(f"{int(year)}-01-01") + pd.Timedelta(days=int(doy) - 1)


def build_observed_years_from_inventory(config: dict, inventory: pd.DataFrame, reps: dict[str, dict[str, int]]) -> dict:
    observed_years: dict[str, list[dict]] = {}
    for station, group in inventory[inventory["weather_qc_status"] == "ok"].groupby("station_code"):
        items = []
        for row in group.sort_values("year").itertuples(index=False):
            planting = date_from_doy(int(row.year), reps[station]["planting_doy"])
            items.append(
                {
                    "year": int(row.year),
                    "label": str(row.scenario_label_for_weather_inventory),
                    "planting_date": planting.strftime("%Y-%m-%d"),
                    "harvest_window_rain_mm": float(row.growing_season_rain),
                }
            )
        observed_years[station] = items
    patched = dict(config)
    patched["observed_years"] = observed_years
    return patched


def read_weather(config: dict) -> pd.DataFrame:
    path = PROJECT_ROOT / config["paths"]["weather_dir"] / "all_sites_weather_cleaned_qc.csv"
    if not path.exists():
        path = PROJECT_ROOT / "weather_clean_qc" / "all_sites_weather_cleaned_qc.csv"
    weather = pd.read_csv(path)
    weather["date"] = pd.to_datetime(weather["date"])
    return weather


def build_weather_inventory(config: dict) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    weather = read_weather(config)
    reps, observed = representative_phenology()
    observed_keys = set(zip(observed["station_code"].astype(str), observed["year"].astype(int)))
    wth_summary_path = PROJECT_ROOT / config["paths"]["wth_dir"] / "wth_generation_summary_qc.csv"
    wth_summary = pd.read_csv(wth_summary_path) if wth_summary_path.exists() else pd.DataFrame()
    wth_lookup = {}
    if not wth_summary.empty:
        for row in wth_summary.itertuples(index=False):
            wth_lookup[(str(row.station), int(row.year))] = row

    rows: list[dict[str, Any]] = []
    for (station, year), group in weather.groupby(["station", "year"]):
        station = str(station)
        year = int(year)
        if station not in reps:
            continue
        full_range = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        dates = pd.to_datetime(group["date"])
        missing_days = sorted(set(full_range.date) - set(dates.dt.date))
        planting = date_from_doy(year, reps[station]["planting_doy"])
        harvest = planting + pd.Timedelta(days=reps[station]["harvest_days"] - 1)
        season = group[(group["date"] >= planting) & (group["date"] <= harvest)].copy()
        expected_season_days = reps[station]["harvest_days"]
        wth = wth_lookup.get((station, year))
        weather_file = getattr(wth, "wth_file", "") if wth is not None else str(PROJECT_ROOT / config["paths"]["wth_dir"] / station / f"{station}{year}.WTH")
        has_full_year = len(group) == len(full_range) and len(missing_days) == 0
        has_growing_season = len(season) == expected_season_days
        range_issue = bool(
            ((group["SRAD"] < 0) | (group["SRAD"] > 40)).any()
            or ((group["TMAX"] < -60) | (group["TMAX"] > 60)).any()
            or ((group["TMIN"] < -70) | (group["TMIN"] > 50)).any()
            or (group["TMIN"] > group["TMAX"]).any()
            or (group["RAIN"] < 0).any()
        )
        annual_rain = float(pd.to_numeric(group["RAIN"], errors="coerce").sum())
        growing_rain = float(pd.to_numeric(season["RAIN"], errors="coerce").sum()) if len(season) else np.nan
        zero_annual_rain = annual_rain <= 0
        zero_growing_rain = pd.notna(growing_rain) and growing_rain <= 0
        wth_ok = bool(wth is not None and str(getattr(wth, "status", "")) == "ok")
        qc_notes: list[str] = []
        if not has_full_year or not has_growing_season or range_issue or not wth_ok:
            qc_notes.append("weather_or_wth_qc_check_needed")
        if zero_annual_rain:
            qc_notes.append("zero_annual_rain_suspect")
        elif zero_growing_rain:
            qc_notes.append("zero_growing_season_rain_suspect")
        qc_status = "ok" if not qc_notes else "check"
        rows.append(
            {
                "station_code": station,
                "station_name": STATION_NAMES.get(station, station),
                "weather_file": weather_file,
                "weather_source_dir": config["paths"]["wth_dir"],
                "year": year,
                "date_start": group["date"].min().strftime("%Y-%m-%d"),
                "date_end": group["date"].max().strftime("%Y-%m-%d"),
                "n_days": int(len(group)),
                "has_full_year": has_full_year,
                "has_growing_season": has_growing_season,
                "missing_days_count": int(len(missing_days)),
                "missing_days_ratio": float(len(missing_days) / len(full_range)),
                "growing_season_start": planting.strftime("%Y-%m-%d"),
                "growing_season_end": harvest.strftime("%Y-%m-%d"),
                "growing_season_days_expected": int(expected_season_days),
                "growing_season_days_available": int(len(season)),
                "growing_season_rain": growing_rain,
                "annual_rain": annual_rain,
                "tmean_mean": float(((group["TMAX"] + group["TMIN"]) / 2).mean()),
                "tmax_max": float(group["TMAX"].max()),
                "tmin_min": float(group["TMIN"].min()),
                "srad_mean": float(group["SRAD"].mean()),
                "weather_qc_status": qc_status,
                "is_observed_experiment_year": (station, year) in observed_keys,
                "is_2020_2023": year in {2020, 2021, 2022, 2023},
                "scenario_label_for_weather_inventory": "",
                "notes": ";".join(qc_notes),
            }
        )
    inventory = pd.DataFrame(rows).sort_values(["station_code", "year"]).reset_index(drop=True)
    inventory["rain_quantile_in_station"] = inventory.groupby("station_code")["growing_season_rain"].rank(pct=True)
    inventory["scenario_label_for_weather_inventory"] = np.select(
        [
            inventory["rain_quantile_in_station"] <= 0.25,
            inventory["rain_quantile_in_station"] >= 0.75,
        ],
        ["dry_weather_candidate", "wet_weather_candidate"],
        default="normal_weather_candidate",
    )
    out = OUTPUT_ROOT / "weather_inventory" / "weather_year_inventory.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(out, index=False, encoding="utf-8-sig")
    write_inventory_summary(inventory)
    return inventory, reps


def write_inventory_summary(inventory: pd.DataFrame) -> None:
    summary = (
        inventory.groupby("station_code")
        .agg(
            available_years=("year", lambda s: ",".join(map(str, sorted(s.astype(int))))),
            n_years=("year", "count"),
            n_ok=("weather_qc_status", lambda s: int((s == "ok").sum())),
            years_2020_2023=("year", lambda s: ",".join(str(y) for y in sorted(set(s.astype(int)) & {2020, 2021, 2022, 2023}))),
            min_rain=("growing_season_rain", "min"),
            max_rain=("growing_season_rain", "max"),
        )
        .reset_index()
    )
    lines = [
        "# Weather Year Inventory Summary",
        "",
        "This inventory uses QC weather data and generated QC WTH files. It does not modify my_data.",
        "",
        df_to_markdown(summary),
    ]
    (OUTPUT_ROOT / "weather_inventory" / "weather_year_inventory_summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_2020_2023_plan(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station in sorted(STATION_NAMES):
        for year in [2020, 2021, 2022, 2023]:
            hit = inventory[(inventory["station_code"] == station) & (inventory["year"] == year)]
            if hit.empty:
                rows.append(
                    {
                        "station_code": station,
                        "station_name": STATION_NAMES[station],
                        "year": year,
                        "weather_file": "",
                        "available": False,
                        "has_full_growing_season": False,
                        "weather_qc_status": "missing",
                        "growing_season_rain": np.nan,
                        "annual_rain": np.nan,
                        "recommended_role": "not_available",
                        "notes": "weather year missing",
                    }
                )
            else:
                row = hit.iloc[0]
                available = bool(row["weather_qc_status"] == "ok")
                role = "calibration" if year in [2020, 2021] and available else ("validation" if year in [2022, 2023] and available else "not_available")
                rows.append(
                    {
                        "station_code": station,
                        "station_name": STATION_NAMES[station],
                        "year": year,
                        "weather_file": row["weather_file"],
                        "available": available,
                        "has_full_growing_season": bool(row["has_growing_season"]),
                        "weather_qc_status": row["weather_qc_status"],
                        "growing_season_rain": row["growing_season_rain"],
                        "annual_rain": row["annual_rain"],
                        "recommended_role": role,
                        "notes": "needs observed yield/phenology for actual cultivar calibration" if available else "not suitable without weather repair",
                    }
                )
    plan = pd.DataFrame(rows)
    out = OUTPUT_ROOT / "cultivar_calibration_plan" / "weather_2020_2023_availability.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(out, index=False, encoding="utf-8-sig")
    lines = [
        "# Cultivar Calibration and Validation Plan for 2020-2023",
        "",
        "Initial split: 2020-2021 for calibration, 2022-2023 for validation when weather is available and QC is ok.",
        "",
        "Important caveat: weather alone is not enough for cultivar calibration. Observed yield, phenology, and fixed management records are still needed.",
        "",
        "Do not mix PPO actions into cultivar calibration; use fixed management only.",
        "",
        df_to_markdown(plan),
    ]
    (OUTPUT_ROOT / "cultivar_calibration_plan" / "cultivar_calibration_validation_plan.md").write_text("\n".join(lines), encoding="utf-8")
    return plan


def daily_path(station: str, year: int, treatment_name: str) -> Path:
    return OUTPUT_ROOT / "daily_outputs" / station / f"{station}_{year}_{treatment_name}_daily.csv"


def summarize_daily(daily: pd.DataFrame, station: str, year: int, treatment: Treatment, inventory_row: pd.Series | None, status: str, error: str, config: dict) -> dict:
    total_i = float(sum(treatment.irrigation_by_dap.values()))
    total_n = float(sum(treatment.nitrogen_by_dap.values()))
    if len(daily):
        total_i = float(pd.to_numeric(daily["real_action_amir"], errors="coerce").fillna(0).sum())
        total_n = float(pd.to_numeric(daily["real_action_anfer"], errors="coerce").fillna(0).sum())
    final_grnwt = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1]) if len(daily) and "grnwt" in daily else np.nan
    final_topwt = float(pd.to_numeric(daily["topwt"], errors="coerce").iloc[-1]) if len(daily) and "topwt" in daily else np.nan
    final_xlai = float(pd.to_numeric(daily["xlai"], errors="coerce").iloc[-1]) if len(daily) and "xlai" in daily else np.nan
    swfac = pd.to_numeric(daily["swfac"], errors="coerce") if len(daily) and "swfac" in daily else pd.Series(dtype=float)
    nstres = pd.to_numeric(daily["nstres"], errors="coerce") if len(daily) and "nstres" in daily else pd.Series(dtype=float)
    inv = inventory_row if inventory_row is not None else pd.Series(dtype=object)
    return {
        "station_code": station,
        "station_name": STATION_NAMES.get(station, station),
        "year": int(year),
        "treatment_id": treatment.treatment_id,
        "treatment_name": treatment.treatment_name,
        "weather_file": inv.get("weather_file", ""),
        "run_status": status,
        "error_message": error,
        "episode_completed": bool(len(daily) and "done" in daily and bool(daily["done"].iloc[-1])),
        "episode_length": int(len(daily)),
        "final_grnwt": final_grnwt,
        "final_topwt": final_topwt,
        "final_xlai": final_xlai,
        "total_irrigation": total_i,
        "total_n": total_n,
        "profit_default": profit(final_grnwt, total_i, total_n, config, low_water=False) if np.isfinite(final_grnwt) else np.nan,
        "profit_low_water_cost": profit(final_grnwt, total_i, total_n, config, low_water=True) if np.isfinite(final_grnwt) else np.nan,
        "growing_season_rain": inv.get("growing_season_rain", np.nan),
        "annual_rain": inv.get("annual_rain", np.nan),
        "mean_swfac": float(swfac.mean()) if len(swfac) else np.nan,
        "max_swfac": float(swfac.max()) if len(swfac) else np.nan,
        "swfac_stress_days_gt_0p05": int((swfac > 0.05).sum()) if len(swfac) else 0,
        "swfac_stress_days_gt_0p10": int((swfac > 0.10).sum()) if len(swfac) else 0,
        "mean_nstres": float(nstres.mean()) if len(nstres) else np.nan,
        "max_nstres": float(nstres.max()) if len(nstres) else np.nan,
        "nstres_days_gt_0p05": int((nstres > 0.05).sum()) if len(nstres) else 0,
        "yield_gain_from_irrigation_at_same_N": np.nan,
        "profit_gain_from_irrigation_default": np.nan,
        "profit_gain_from_irrigation_low_water_cost": np.nan,
        "daily_csv_path": str(daily_path(station, year, treatment.treatment_name).relative_to(PROJECT_ROOT)),
        "notes": "",
    }


def run_fixed_treatment(config: dict, station: str, year: int, treatment: Treatment, inventory_row: pd.Series | None) -> dict:
    path = daily_path(station, year, treatment.treatment_name)
    if path.exists():
        daily = pd.read_csv(path)
        return summarize_daily(daily, station, year, treatment, inventory_row, "ok_cached", "", config)
    path.parent.mkdir(parents=True, exist_ok=True)
    env = None
    records: list[dict[str, Any]] = []
    run_tag = f"allyear_{treatment.treatment_name}"
    try:
        env = make_env(config, station, int(year), int(config.get("seed", 0)), run_tag=run_tag, evaluation=True, action_safety_enabled=False)
        policy = treatment_policy(treatment)
        safety_state = ActionSafetyState()
        safety_config = {**config.get("action_safety", {}), "enabled": bool(config.get("action_safety", {}).get("enabled", False))}
        year_info = [x for x in config["observed_years"][station] if int(x["year"]) == int(year)][0]
        obs, info = env.reset()
        done = False
        step_count = 0
        cumulative_i = 0.0
        cumulative_n = 0.0
        while not done and step_count < int(config.get("runtime", {}).get("max_steps", 260)):
            latest = latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap", np.nan))
            dap = int(round(dap_raw)) if not np.isnan(dap_raw) and dap_raw > 0 else step_count + 1
            planned = policy.action_for_dap(dap)
            bounded, clip_note = clip_to_action_space(env, planned)
            safety_result = apply_action_safety(bounded, dap, safety_state, safety_config)
            safe = safety_result.safe_real_action
            normalized = normalize_action(env.formator.action_names, env.formator.action_space_dict, safe)
            obs, reward, terminated, truncated, info = env.step(normalized)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            update_action_safety_state(safety_state, safe, dap)
            real_i = float(safe.get("amir", 0.0))
            real_n = float(safe.get("anfer", 0.0))
            cumulative_i += real_i
            cumulative_n += real_n
            date = pd.Timestamp(year_info["planting_date"]) + pd.Timedelta(days=max(dap - 1, 0))
            records.append(
                {
                    "station": station,
                    "year": int(year),
                    "treatment_id": treatment.treatment_id,
                    "treatment_name": treatment.treatment_name,
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": scalar(latest.get("dap", dap)),
                    "planned_event_name": policy.event_name_for_dap(dap),
                    "planned_irrigation": float(planned.get("amir", 0.0)),
                    "planned_n": float(planned.get("anfer", 0.0)),
                    "real_action_amir": real_i,
                    "real_action_anfer": real_n,
                    "topwt": scalar(latest.get("topwt")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "xlai": scalar(latest.get("xlai")),
                    "totir": cumulative_i,
                    "tofer": cumulative_n,
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "reward": float(reward),
                    "done": done,
                    "env_clip_note": clip_note,
                    "safety_rule_triggered": safety_result.safety_rule_triggered,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1
        daily = pd.DataFrame(records)
        daily.to_csv(path, index=False, encoding="utf-8-sig")
        if bool(config.get("runtime", {}).get("plot_daily_episodes", False)):
            plot_episode(daily, OUTPUT_ROOT / "figures" / "daily" / station / f"{year}_{treatment.treatment_name}")
        status = "ok" if len(daily) and bool(daily["done"].iloc[-1]) else "failed"
        return summarize_daily(daily, station, year, treatment, inventory_row, status, "" if status == "ok" else "episode_not_completed", config)
    except Exception as exc:
        if records:
            pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8-sig")
        return summarize_daily(pd.DataFrame(records), station, year, treatment, inventory_row, "failed", f"{type(exc).__name__}: {exc}", config)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def add_diagnostic_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (station, year), group in summary.groupby(["station_code", "year"]):
        by_id = {row.treatment_id: row for row in group.itertuples(index=False)}
        t1 = by_id.get("T1")
        for row in group.to_dict("records"):
            if row["treatment_id"] in ["T2", "T3"] and t1 is not None:
                row["yield_gain_from_irrigation_at_same_N"] = float(row["final_grnwt"]) - float(t1.final_grnwt)
                row["profit_gain_from_irrigation_default"] = float(row["profit_default"]) - float(t1.profit_default)
                row["profit_gain_from_irrigation_low_water_cost"] = float(row["profit_low_water_cost"]) - float(t1.profit_low_water_cost)
            rows.append(row)
    return pd.DataFrame(rows)


def run_stress_diagnostics(config: dict, inventory: pd.DataFrame) -> pd.DataFrame:
    treatments = load_treatments(config)
    rows = []
    partial = OUTPUT_ROOT / "stress_diagnostics" / "all_year_fixed_management_stress_summary_partial.csv"
    ok_inventory = inventory[inventory["weather_qc_status"] == "ok"].copy()
    for item in ok_inventory.itertuples(index=False):
        inv_row = pd.Series(item._asdict())
        for treatment in treatments:
            print(f"[all-year-fixed] {item.station_code} {int(item.year)} {treatment.treatment_name}", flush=True)
            rows.append(run_fixed_treatment(config, str(item.station_code), int(item.year), treatment, inv_row))
            pd.DataFrame(rows).to_csv(partial, index=False, encoding="utf-8-sig")
    summary = add_diagnostic_metrics(pd.DataFrame(rows))
    out = OUTPUT_ROOT / "stress_diagnostics" / "all_year_fixed_management_stress_summary.csv"
    summary.to_csv(out, index=False, encoding="utf-8-sig")
    return summary


def build_scenario_pool(inventory: pd.DataFrame, stress: pd.DataFrame, plan_2020: pd.DataFrame, config: dict) -> pd.DataFrame:
    cfg = config["screening"]
    rows = []
    for (station, year), group in stress.groupby(["station_code", "year"]):
        inv = inventory[(inventory["station_code"] == station) & (inventory["year"] == year)].iloc[0]
        by_id = {row.treatment_id: row for row in group.itertuples(index=False)}
        t0, t1, t2, t3, t4 = by_id.get("T0"), by_id.get("T1"), by_id.get("T2"), by_id.get("T3"), by_id.get("T4")
        base = t1 or t0
        has_water_stress = bool(base is not None and (float(base.swfac_stress_days_gt_0p05) > 0 or float(base.max_swfac) > float(cfg["water_stress_threshold"])))
        has_n_stress = bool(t0 is not None and (float(t0.nstres_days_gt_0p05) > 0 or float(t0.max_nstres) > float(cfg["n_stress_threshold"])))
        n_gain = float(t1.final_grnwt) - float(t0.final_grnwt) if t0 is not None and t1 is not None else np.nan
        irrigation_candidates = [x for x in [t2, t3] if x is not None and t1 is not None]
        yield_gains = [float(x.final_grnwt) - float(t1.final_grnwt) for x in irrigation_candidates]
        profit_low_gains = [float(x.profit_low_water_cost) - float(t1.profit_low_water_cost) for x in irrigation_candidates]
        max_yield_gain = max([x for x in yield_gains if np.isfinite(x)], default=np.nan)
        max_profit_low = max([x for x in profit_low_gains if np.isfinite(x)], default=np.nan)
        irrigation_responsive = bool(
            np.isfinite(max_yield_gain)
            and (
                max_yield_gain >= float(cfg["irrigation_yield_gain_abs_threshold"])
                or (float(t1.final_grnwt) > 0 and (float(t1.final_grnwt) + max_yield_gain) >= float(t1.final_grnwt) * float(cfg["irrigation_yield_gain_ratio_threshold"]))
            )
            and np.isfinite(max_profit_low)
            and max_profit_low >= float(cfg["low_water_profit_tolerance"])
        )
        nitrogen_responsive = bool(np.isfinite(n_gain) and n_gain >= 200)
        rain_q = float(inv.rain_quantile_in_station)
        labels = []
        if rain_q <= 0.25:
            labels.append("dry_year")
        elif rain_q >= 0.75:
            labels.append("wet_year")
        else:
            labels.append("normal_year")
        if has_water_stress:
            labels.append("water_stress_year")
        if has_n_stress:
            labels.append("nitrogen_stress_year")
        if irrigation_responsive:
            labels.append("irrigation_responsive_year")
        if not has_water_stress and not irrigation_responsive:
            labels.append("low_response_year")
        role_rows = plan_2020[(plan_2020["station_code"] == station) & (plan_2020["year"] == year)]
        recommended_for_calibration = bool(len(role_rows) and str(role_rows.iloc[0]["recommended_role"]) == "calibration")
        recommended_for_validation = bool(len(role_rows) and str(role_rows.iloc[0]["recommended_role"]) == "validation")
        rows.append(
            {
                "station_code": station,
                "station_name": STATION_NAMES.get(station, station),
                "year": int(year),
                "weather_file": inv.weather_file,
                "scenario_type": ";".join(labels),
                "has_water_stress": has_water_stress,
                "has_nitrogen_stress": has_n_stress,
                "irrigation_responsive": irrigation_responsive,
                "nitrogen_responsive": nitrogen_responsive,
                "growing_season_rain": float(inv.growing_season_rain),
                "annual_rain": float(inv.annual_rain),
                "swfac_stress_days_gt_0p05": float(base.swfac_stress_days_gt_0p05) if base is not None else np.nan,
                "max_swfac": float(base.max_swfac) if base is not None else np.nan,
                "mean_swfac": float(base.mean_swfac) if base is not None else np.nan,
                "nstres_days_gt_0p05": float(t0.nstres_days_gt_0p05) if t0 is not None else np.nan,
                "max_nstres": float(t0.max_nstres) if t0 is not None else np.nan,
                "mean_nstres": float(t0.mean_nstres) if t0 is not None else np.nan,
                "yield_gain_from_irrigation_at_same_N": max_yield_gain,
                "profit_gain_from_irrigation_low_water_cost": max_profit_low,
                "yield_gain_from_n_medium_vs_zero": n_gain,
                "recommended_for_calibration": recommended_for_calibration,
                "recommended_for_validation": recommended_for_validation,
                "recommended_for_ppo_train": bool(has_water_stress or has_n_stress or irrigation_responsive or (0.25 < rain_q < 0.75)),
                "recommended_for_ppo_eval": bool((rain_q <= 0.25 or rain_q >= 0.75) and not recommended_for_calibration),
                "notes": "",
            }
        )
    pool = pd.DataFrame(rows).sort_values(["station_code", "year"]).reset_index(drop=True)
    out = OUTPUT_ROOT / "scenario_pool" / "all_year_weather_scenario_pool.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(out, index=False, encoding="utf-8-sig")
    write_scenario_summary(pool)
    write_route(pool)
    return pool


def write_scenario_summary(pool: pd.DataFrame) -> None:
    summary = (
        pool.groupby("station_code")
        .agg(
            n_years=("year", "count"),
            water_stress_years=("has_water_stress", "sum"),
            nitrogen_stress_years=("has_nitrogen_stress", "sum"),
            irrigation_responsive_years=("irrigation_responsive", "sum"),
            calibration_years=("year", lambda s: ",".join(map(str, sorted(pool.loc[s.index][pool.loc[s.index, "recommended_for_calibration"]]["year"].astype(int))))),
            validation_years=("year", lambda s: ",".join(map(str, sorted(pool.loc[s.index][pool.loc[s.index, "recommended_for_validation"]]["year"].astype(int))))),
        )
        .reset_index()
    )
    lines = [
        "# Scenario Pool Summary",
        "",
        df_to_markdown(summary),
        "",
        "A year is irrigation responsive only when same-N irrigation increases yield and does not reduce low-water-cost profit.",
    ]
    (OUTPUT_ROOT / "scenario_pool" / "scenario_pool_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_route(pool: pd.DataFrame) -> None:
    water_count = int(pool["has_water_stress"].sum()) if not pool.empty else 0
    irr_count = int(pool["irrigation_responsive"].sum()) if not pool.empty else 0
    next_stage = "006_16_all_year_offline_schedule_search_and_imitation_prior" if water_count >= 3 or irr_count >= 3 else "006_16_cultivar_calibration_and_nitrogen_management_focus"
    lines = [
        "# New Experiment Route After Group Meeting",
        "",
        "## Stage A: cultivar calibration / validation",
        "",
        "Use 2020-2023 weather years with fixed management. Do not introduce PPO action effects into cultivar calibration.",
        "",
        "## Stage B: all-year weather scenario stress pool",
        "",
        "Use all available QC weather years and classify them by rainfall, swfac, nstres, and irrigation response.",
        "",
        "## Stage C: expert prior / offline schedule search",
        "",
        "Run this only if the scenario pool includes enough water-stress or irrigation-responsive years.",
        "",
        "## Stage D: PPO",
        "",
        "Run only after scenario pool and calibration route are established. Continue action safety and do not use unrestricted PPO.",
        "",
        "## Recommended next prompt",
        "",
        f"`{next_stage}`",
        "",
        f"Evidence: water_stress_years={water_count}, irrigation_responsive_years={irr_count}.",
    ]
    out = OUTPUT_ROOT / "evaluation" / "new_experiment_route_after_group_meeting.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def plot_outputs(inventory: pd.DataFrame, stress: pd.DataFrame, pool: pd.DataFrame, plan_2020: pd.DataFrame) -> None:
    fig_dir = OUTPUT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    inv_count = inventory.groupby("station_code")["year"].count().sort_index()
    ax = inv_count.plot(kind="bar", figsize=(8, 4), color="#4F81BD")
    ax.set_ylabel("Available weather years")
    ax.grid(axis="y", alpha=0.25)
    ax.get_figure().tight_layout()
    ax.get_figure().savefig(fig_dir / "weather_year_inventory_by_station.png", dpi=180)
    plt.close(ax.get_figure())

    pivot = inventory.pivot_table(index="year", columns="station_code", values="growing_season_rain", aggfunc="mean")
    ax = pivot.plot(figsize=(11, 4.5), marker="o")
    ax.set_ylabel("Growing-season rainfall (mm)")
    ax.grid(axis="y", alpha=0.25)
    ax.get_figure().tight_layout()
    ax.get_figure().savefig(fig_dir / "growing_season_rain_by_year_station.png", dpi=180)
    plt.close(ax.get_figure())

    if not pool.empty:
        for metric, filename, ylabel in [
            ("swfac_stress_days_gt_0p05", "swfac_stress_days_by_year_station.png", "swfac>0.05 days"),
            ("nstres_days_gt_0p05", "nstres_days_by_year_station.png", "nstres>0.05 days"),
            ("yield_gain_from_irrigation_at_same_N", "yield_gain_from_irrigation_by_year_station.png", "Yield gain from irrigation at same N"),
        ]:
            pivot = pool.pivot_table(index="year", columns="station_code", values=metric, aggfunc="mean")
            ax = pivot.plot(figsize=(11, 4.5), marker="o")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)
            ax.get_figure().tight_layout()
            ax.get_figure().savefig(fig_dir / filename, dpi=180)
            plt.close(ax.get_figure())

        cols = ["has_water_stress", "has_nitrogen_stress", "irrigation_responsive", "recommended_for_ppo_train", "recommended_for_ppo_eval"]
        heat = pool.copy()
        heat["site_year"] = heat["station_code"] + "_" + heat["year"].astype(str)
        values = heat.set_index("site_year")[cols].astype(int)
        fig, ax = plt.subplots(figsize=(9, max(5, len(values) * 0.06)))
        im = ax.imshow(values.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_yticks(range(len(values.index)))
        ax.set_yticklabels(values.index, fontsize=4)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=35, ha="right")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.tight_layout()
        fig.savefig(fig_dir / "scenario_type_heatmap.png", dpi=180)
        plt.close(fig)

        train_eval = pool.groupby("station_code").agg(train=("recommended_for_ppo_train", "sum"), eval=("recommended_for_ppo_eval", "sum")).reset_index()
        ax = train_eval.set_index("station_code")[["train", "eval"]].plot(kind="bar", figsize=(8, 4), color=["#4F81BD", "#9BBB59"])
        ax.set_ylabel("Recommended years")
        ax.grid(axis="y", alpha=0.25)
        ax.get_figure().tight_layout()
        ax.get_figure().savefig(fig_dir / "ppo_train_eval_scenario_pool.png", dpi=180)
        plt.close(ax.get_figure())

    cal = plan_2020.copy()
    cal["site_year"] = cal["station_code"] + "_" + cal["year"].astype(str)
    role_map = {"calibration": 1, "validation": 2, "not_available": 0}
    cal["role_code"] = cal["recommended_role"].map(role_map).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    mat = cal.pivot_table(index="station_code", columns="year", values="role_code", aggfunc="max").reindex(index=sorted(STATION_NAMES), columns=[2020, 2021, 2022, 2023])
    im = ax.imshow(mat.values, aspect="auto", cmap="Blues", vmin=0, vmax=2)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            code = int(mat.values[i, j]) if not pd.isna(mat.values[i, j]) else 0
            label = {0: "NA", 1: "C", 2: "V"}[code]
            ax.text(j, i, label, ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration_validation_years_2020_2023.png", dpi=180)
    plt.close(fig)


def build_report(inventory: pd.DataFrame, plan_2020: pd.DataFrame, stress: pd.DataFrame, pool: pd.DataFrame) -> None:
    inv_summary = inventory.groupby("station_code").agg(n_years=("year", "count"), years=("year", lambda s: ",".join(map(str, sorted(s.astype(int)))))).reset_index()
    plan_summary = plan_2020.pivot_table(index="station_code", columns="year", values="weather_qc_status", aggfunc="first").reset_index()
    pool_summary = pool.groupby("station_code").agg(
        water_stress_years=("has_water_stress", "sum"),
        nitrogen_stress_years=("has_nitrogen_stress", "sum"),
        irrigation_responsive_years=("irrigation_responsive", "sum"),
        ppo_train_years=("recommended_for_ppo_train", "sum"),
        ppo_eval_years=("recommended_for_ppo_eval", "sum"),
    ).reset_index()
    observed_only = PROJECT_ROOT / "Leave_One_experiments" / "water_nitrogen_factorial_diagnosis" / "evaluation" / "irrigation_response_screening.csv"
    observed_water = np.nan
    observed_irr = np.nan
    if observed_only.exists():
        obs = pd.read_csv(observed_only)
        observed_water = int(obs["has_water_stress"].sum())
        observed_irr = int(obs["irrigation_candidate"].sum())
    all_water = int(pool["has_water_stress"].sum()) if not pool.empty else 0
    all_irr = int(pool["irrigation_responsive"].sum()) if not pool.empty else 0
    next_route = "006_16_all_year_offline_schedule_search_and_imitation_prior" if all_water >= 3 or all_irr >= 3 else "006_16_cultivar_calibration_and_nitrogen_management_focus"
    lines = [
        "# All-Year Weather Calibration, Validation, and PPO Scenario Pool Report",
        "",
        "## Executive Summary",
        "",
        "- This stage did not train PPO, did not run rainfall-scaling, and did not modify original my_data files.",
        f"- Weather inventory found {len(inventory)} station-years with QC weather/WTH data.",
        f"- 2020-2023 weather is available for all five stations, subject to the caveat that cultivar calibration still needs observed yield/phenology/fixed management records.",
        f"- All-year fixed management diagnostics found {all_water} water-stress years and {all_irr} irrigation-responsive years, compared with observed-year-only water-stress={observed_water} and irrigation-candidate={observed_irr}.",
        f"- Recommended next stage: `{next_route}`.",
        "",
        "## Available Weather Years",
        "",
        df_to_markdown(inv_summary),
        "",
        "## 2020-2023 Availability",
        "",
        df_to_markdown(plan_summary),
        "",
        "## Scenario Pool Summary",
        "",
        df_to_markdown(pool_summary),
        "",
        "## Water Stress / Irrigation Responsive Years",
        "",
        df_to_markdown(pool[(pool["has_water_stress"] | pool["irrigation_responsive"] | pool["has_nitrogen_stress"])][[
            "station_code",
            "year",
            "scenario_type",
            "growing_season_rain",
            "swfac_stress_days_gt_0p05",
            "max_swfac",
            "nstres_days_gt_0p05",
            "yield_gain_from_irrigation_at_same_N",
            "profit_gain_from_irrigation_low_water_cost",
            "recommended_for_ppo_train",
            "recommended_for_ppo_eval",
        ]], 80),
        "",
        "## Calibration / Validation Route",
        "",
        "Use 2020-2021 as calibration candidates and 2022-2023 as validation candidates only where weather QC is ok. Actual cultivar calibration still requires observed yield and phenology records; weather alone is insufficient.",
        "",
        "## PPO Route",
        "",
        "Do not train unrestricted PPO. If the scenario pool has enough water-stress or irrigation-responsive years, first run all-year offline schedule search. Otherwise, prioritize cultivar calibration and nitrogen-management focus.",
    ]
    DOC_MD.write_text("\n".join(lines), encoding="utf-8")
    safe_copy(DOC_MD, OUTPUT_ROOT / "reports" / DOC_MD.name)
    build_ppt(inv_summary, plan_summary, pool_summary, pool, next_route)
    if DOC_PPT.exists():
        safe_copy(DOC_PPT, OUTPUT_ROOT / "reports" / DOC_PPT.name)


def add_slide(prs, title: str, bullets: list[str]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(0.45), Inches(0.3), Inches(12.4), Inches(0.6))
    r = box.text_frame.paragraphs[0].add_run()
    r.text = title
    r.font.name = "Microsoft YaHei"
    r.font.size = Pt(22)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0, 0, 0)
    body = slide.shapes.add_textbox(Inches(0.7), Inches(1.15), Inches(12.0), Inches(5.8))
    tf = body.text_frame
    tf.clear()
    for idx, item in enumerate(bullets):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = item
        p.space_after = Pt(7)
        for run in p.runs:
            run.font.name = "Microsoft YaHei"
            run.font.size = Pt(15)
            run.font.color.rgb = RGBColor(0, 0, 0)


def add_table_slide(prs, title: str, df: pd.DataFrame, max_rows: int = 14) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(0.45), Inches(0.3), Inches(12.4), Inches(0.6))
    r = box.text_frame.paragraphs[0].add_run()
    r.text = title
    r.font.name = "Microsoft YaHei"
    r.font.size = Pt(22)
    r.font.bold = True
    rows = min(max_rows, len(df))
    cols = len(df.columns)
    table_shape = slide.shapes.add_table(rows + 1, cols, Inches(0.35), Inches(1.15), Inches(12.65), Inches(5.8))
    table = table_shape.table
    work = df.head(rows).copy()
    for col in work.select_dtypes(include=["float", "int"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(3)
    for j, col in enumerate(work.columns):
        cell = table.cell(0, j)
        cell.text = str(col)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(68, 114, 196)
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for run in p.runs:
                run.font.name = "Microsoft YaHei"
                run.font.size = Pt(7)
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
    for i, row in enumerate(work.itertuples(index=False), start=1):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.text = "" if pd.isna(value) else str(value)
            if i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(226, 235, 247)
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.name = "Microsoft YaHei"
                    run.font.size = Pt(6)
                    run.font.color.rgb = RGBColor(0, 0, 0)


def add_picture_slide(prs, title: str, path: Path) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(0.45), Inches(0.3), Inches(12.4), Inches(0.6))
    r = box.text_frame.paragraphs[0].add_run()
    r.text = title
    r.font.name = "Microsoft YaHei"
    r.font.size = Pt(22)
    r.font.bold = True
    slide.shapes.add_picture(str(path), Inches(0.65), Inches(1.1), width=Inches(11.8))


def build_ppt(inv_summary: pd.DataFrame, plan_summary: pd.DataFrame, pool_summary: pd.DataFrame, pool: pd.DataFrame, next_route: str) -> None:
    if Presentation is None:
        print("python-pptx is not available; skipped PPT generation.", flush=True)
        return
    prs = Presentation()
    add_slide(
        prs,
        "006_15 All-Year Weather Scenario Pool",
        [
            "Purpose: expand from observed-year-only to all available QC weather years before any further PPO.",
            "No PPO training, no rainfall-scaling, no unrestricted PPO, and no my_data overwrite were performed.",
            "2020-2023 are checked for cultivar calibration / validation, while all years are screened for swfac/nstres and irrigation response.",
        ],
    )
    add_table_slide(prs, "Available Weather Years by Station", inv_summary, 8)
    add_table_slide(prs, "2020-2023 Weather Availability", plan_summary, 8)
    add_table_slide(prs, "Scenario Pool Summary", pool_summary, 8)
    stress = pool[(pool["has_water_stress"] | pool["irrigation_responsive"])][[
        "station_code",
        "year",
        "scenario_type",
        "growing_season_rain",
        "swfac_stress_days_gt_0p05",
        "max_swfac",
        "yield_gain_from_irrigation_at_same_N",
        "profit_gain_from_irrigation_low_water_cost",
    ]]
    add_table_slide(prs, "Water Stress / Irrigation Response Years", stress, 16)
    fig_dir = OUTPUT_ROOT / "figures"
    for title, filename in [
        ("Growing Season Rain by Year", "growing_season_rain_by_year_station.png"),
        ("SWFAC Stress Days", "swfac_stress_days_by_year_station.png"),
        ("Irrigation Yield Gain at Same N", "yield_gain_from_irrigation_by_year_station.png"),
        ("PPO Train/Eval Scenario Pool", "ppo_train_eval_scenario_pool.png"),
    ]:
        path = fig_dir / filename
        if path.exists():
            add_picture_slide(prs, title, path)
    add_slide(
        prs,
        "Recommended Next Step",
        [
            f"Recommended next prompt: {next_route}",
            "If enough water-stress / irrigation-responsive years exist, proceed to all-year offline schedule search before PPO.",
            "If not, prioritize cultivar calibration and nitrogen-management focus.",
        ],
    )
    prs.save(DOC_PPT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    config = load_yaml(Path(args.config))
    ensure_dirs()
    safe_copy(Path(args.config), OUTPUT_ROOT / "configs" / Path(args.config).name)
    inventory, reps = build_weather_inventory(config)
    plan_2020 = build_2020_2023_plan(inventory)
    patched_config = build_observed_years_from_inventory(config, inventory, reps)
    if args.inventory_only:
        print(OUTPUT_ROOT / "weather_inventory" / "weather_year_inventory.csv")
        print(OUTPUT_ROOT / "cultivar_calibration_plan" / "weather_2020_2023_availability.csv")
        return
    stress_path = OUTPUT_ROOT / "stress_diagnostics" / "all_year_fixed_management_stress_summary.csv"
    if args.report_only and stress_path.exists():
        stress = pd.read_csv(stress_path)
    else:
        stress = run_stress_diagnostics(patched_config, inventory)
    pool = build_scenario_pool(inventory, stress, plan_2020, patched_config)
    plot_outputs(inventory, stress, pool, plan_2020)
    build_report(inventory, plan_2020, stress, pool)
    print(OUTPUT_ROOT / "weather_inventory" / "weather_year_inventory.csv")
    print(stress_path)
    print(OUTPUT_ROOT / "scenario_pool" / "all_year_weather_scenario_pool.csv")
    print(DOC_MD)
    print(DOC_PPT)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
