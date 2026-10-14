from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import gym
import numpy as np
import pandas as pd

from baseline_policies import ExpertAgent, NullAgent
from dssat_site_config import SITE_CONFIGS, build_env_args
from sb3_wrapper import GymDssatWrapper


ETCP_FROM_DSSAT_485 = {
    # Values supplied from the DSSAT 4.8.5 comparison table.
    "HL": 496.0,
    "SY": 428.0,
    "LC": 330.0,
    "FQ": 344.0,
    "YC": 350.0,
}

SITE_NAMES = {
    "HL": "Hailun",
    "SY": "Shenyang",
    "LC": "Luancheng",
    "FQ": "Fengqiu",
    "YC": "YC",
}


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def parse_weather(path: str | Path) -> pd.DataFrame:
    rows = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0].isdigit():
            date = parts[0]
            rows.append(
                {
                    "date": date,
                    "year": int(date[:-3]),
                    "doy": int(date[-3:]),
                    "srad": float(parts[1]),
                    "tmax": float(parts[2]),
                    "tmin": float(parts[3]),
                    "rain": float(parts[4]),
                }
            )
    return pd.DataFrame(rows)


def parse_pdate(template_path: str | Path) -> str:
    lines = Path(template_path).read_text(errors="ignore").splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("@P ") and "PDATE" in line:
            for next_line in lines[i + 1 :]:
                parts = next_line.split()
                if parts and parts[0].isdigit() and len(parts) > 1:
                    return parts[1]
    raise ValueError(f"Cannot find PDATE in {template_path}")


def parse_template_irrigation(template_path: str | Path) -> float:
    lines = Path(template_path).read_text(errors="ignore").splitlines()
    in_irrigation_events = False
    total = 0.0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("@I IDATE"):
            in_irrigation_events = True
            continue
        if in_irrigation_events:
            if not stripped or stripped.startswith("*") or stripped.startswith("@"):
                in_irrigation_events = False
                continue
            parts = stripped.split()
            if len(parts) >= 4 and parts[0].isdigit():
                total += safe_float(parts[3])
    return total


def crop_season_prcp(weather_path: str | Path, pdate: str, n_days: int) -> float:
    weather = parse_weather(weather_path)
    year = int(pdate[:2])
    year += 2000 if year < 50 else 1900
    start_doy = int(pdate[2:])
    end_doy = start_doy + max(n_days, 1) - 1
    if end_doy <= 366:
        mask = (weather["year"] == year) & (weather["doy"] >= start_doy) & (weather["doy"] <= end_doy)
    else:
        mask = (
            ((weather["year"] == year) & (weather["doy"] >= start_doy))
            | ((weather["year"] == year + 1) & (weather["doy"] <= end_doy - 366))
        )
    return float(weather.loc[mask, "rain"].sum())


def latest_observation(env: GymDssatWrapper) -> dict:
    history = getattr(env.unwrapped, "history", {})
    if isinstance(history, dict):
        observations = history.get("observation", [])
        if observations and isinstance(observations[-1], dict):
            return observations[-1]
    return {}


def evaluate_site_agent(
    site: str,
    agent_name: str,
    env_args: dict,
    output_dir: Path,
    max_steps: int,
) -> tuple[pd.DataFrame, dict]:
    source_env = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args)
    env = GymDssatWrapper(source_env.unwrapped)
    try:
        observation, _ = env.reset()
        agent = NullAgent(env) if agent_name == "null" else ExpertAgent(env)
        records = []
        done = False
        step_count = 0
        while not done and step_count < max_steps:
            step_count += 1
            normalized_action = agent.predict(observation)[0]
            action_values = env.formator.denormalize_actions(normalized_action)
            action_dict = env.formator.format_actions(action_values)
            observation, reward, terminated, truncated, _info = env.step(normalized_action)
            done = terminated or truncated
            raw_obs = latest_observation(env)
            if safe_float(raw_obs.get("dap")) == 0:
                continue
            history = getattr(env.unwrapped, "history", {})
            dssat_action = {}
            if isinstance(history, dict) and history.get("action"):
                dssat_action = history["action"][-1]
            records.append(
                {
                    "site": site,
                    "agent": agent_name,
                    "dap": safe_float(raw_obs.get("dap")),
                    "swfac": safe_float(raw_obs.get("swfac")),
                    "turfac": safe_float(raw_obs.get("turfac")),
                    "nstres": safe_float(raw_obs.get("nstres")),
                    "topwt": safe_float(raw_obs.get("topwt")),
                    "grnwt": safe_float(raw_obs.get("grnwt")),
                    "xlai": safe_float(raw_obs.get("xlai")),
                    "ep": safe_float(raw_obs.get("ep")),
                    "totir": safe_float(raw_obs.get("totir")),
                    "rain": safe_float(raw_obs.get("rain")),
                    "runoff": safe_float(raw_obs.get("runoff")),
                    "reward": safe_float(reward),
                    "raw_action_amir": safe_float(np.asarray(normalized_action).flatten()[0]),
                    "real_action_amir": safe_float(action_dict.get("amir")),
                    "history_action_amir": safe_float(dssat_action.get("amir") if isinstance(dssat_action, dict) else np.nan),
                }
            )
        df = pd.DataFrame(records)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / f"{site}_{agent_name}_water_stress_trace.csv", index=False)
        summary = summarize_trace(site, agent_name, df, env_args)
        summary["terminated_normally"] = bool(done)
        summary["max_steps"] = max_steps
        return df, summary
    finally:
        env.close()
        gc.collect()


def stress_days(series: pd.Series, threshold: float) -> int:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0
    return int((values > threshold).sum())


def summarize_trace(site: str, agent_name: str, df: pd.DataFrame, env_args: dict) -> dict:
    pdate = parse_pdate(env_args["fileX_template_path"])
    prcp = crop_season_prcp(env_args["auxiliary_file_paths"][1], pdate, len(df))
    etcp = ETCP_FROM_DSSAT_485.get(site, np.nan)
    swfac = pd.to_numeric(df.get("swfac"), errors="coerce")
    turfac = pd.to_numeric(df.get("turfac"), errors="coerce")
    nstres = pd.to_numeric(df.get("nstres"), errors="coerce")
    real_amir = pd.to_numeric(df.get("history_action_amir"), errors="coerce")
    if real_amir.dropna().empty:
        real_amir = pd.to_numeric(df.get("real_action_amir"), errors="coerce")
    final_totir = pd.to_numeric(df.get("totir"), errors="coerce").dropna()
    return {
        "site": site,
        "site_name": SITE_NAMES.get(site, site),
        "agent": agent_name,
        "template": env_args["fileX_template_path"],
        "weather": env_args["auxiliary_file_paths"][1],
        "soil": env_args["auxiliary_file_paths"][2],
        "pdate": pdate,
        "n_days": len(df),
        "PRCP_crop_period": prcp,
        "ETCP_from_DSSAT485": etcp,
        "PRCP_minus_ETCP": prcp - etcp if not np.isnan(etcp) else np.nan,
        "swfac_stress_days_gt_0.05": stress_days(swfac, 0.05),
        "turfac_stress_days_gt_0.05": stress_days(turfac, 0.05),
        "min_swfac": swfac.min(skipna=True),
        "max_swfac": swfac.max(skipna=True),
        "mean_swfac": swfac.mean(skipna=True),
        "mean_nstres": nstres.mean(skipna=True),
        "max_grnwt": pd.to_numeric(df.get("grnwt"), errors="coerce").max(skipna=True),
        "total_irrigation_action": real_amir.sum(skipna=True),
        "final_totir": final_totir.iloc[-1] if not final_totir.empty else np.nan,
        "template_irrigation_mm": parse_template_irrigation(env_args["fileX_template_path"]),
        "total_reward": pd.to_numeric(df.get("reward"), errors="coerce").sum(skipna=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", default="HL,SY,LC,FQ,YC")
    parser.add_argument("--output-dir", default="output_hl/water_stress_diagnostics")
    parser.add_argument("--data-dir", default="./my_data")
    parser.add_argument("--run-dssat-location", default="/opt/dssat_pdi/run_dssat")
    parser.add_argument("--max-steps", type=int, default=260)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summaries = []
    metadata = {}
    for site in [value.strip().upper() for value in args.sites.split(",") if value.strip()]:
        if site not in SITE_CONFIGS:
            raise ValueError(f"Unknown site: {site}")
        env_args = build_env_args(
            site=site,
            mode="irrigation",
            seed=123,
            data_dir=args.data_dir,
            prefer_suffix=None,
            run_dssat_location=args.run_dssat_location,
        )
        metadata[site] = env_args
        print(f"Running {site}: {env_args['fileX_template_path']}")
        for agent_name in ["null", "expert"]:
            print(f"  agent={agent_name}", flush=True)
            _df, summary = evaluate_site_agent(site, agent_name, env_args, output_dir, args.max_steps)
            summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "water_stress_summary.csv", index=False)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(summary_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
