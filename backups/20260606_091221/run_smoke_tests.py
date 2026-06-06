from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from smoke_test_agents import POLICIES, clip_real_action, normalize_real_action


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SMOKE_ROOT = PROJECT_ROOT / "Leave_One_experiments" / "smoke_tests"
CONFIG_PATH = PROJECT_ROOT / "experiments" / "smoke_tests" / "config_observed_year_smoke_tests.yaml"

SITE_INFO = {
    "HLA": {"short": "HL", "template": "UFGA8201-HL.jinja2", "soil": "HL.SOL", "weather_prefix": "HLA", "expected_weather": "CNHL0701.WTH"},
    "SYA": {"short": "SY", "template": "UFGA8201-SY.jinja2", "soil": "SY.SOL", "weather_prefix": "SYA", "expected_weather": "CNSY1201.WTH"},
    "LCA": {"short": "LC", "template": "UFGA8201-LC.jinja2", "soil": "LC.SOL", "weather_prefix": "LCA", "expected_weather": "CNLC0801.WTH"},
    "YCA": {"short": "YC", "template": "UFGA8201-YC.jinja2", "soil": "YC.SOL", "weather_prefix": "YCA", "expected_weather": "CNYC0801.WTH"},
    "FQA": {"short": "FQ", "template": "UFGA8201-FQ.jinja2", "soil": "FQ.SOL", "weather_prefix": "FQA", "expected_weather": "CNFQ0701.WTH"},
}


def ensure_dirs() -> None:
    for sub in ["configs", "rendered_inputs", "logs", "daily_outputs", "evaluation", "figures", "reports"]:
        (SMOKE_ROOT / sub).mkdir(parents=True, exist_ok=True)


def yyddd(date_text: str) -> str:
    ts = pd.Timestamp(date_text)
    return f"{ts.year % 100:02d}{ts.dayofyear:03d}"


def render_template(station: str, year: int, planting_date: str) -> Path:
    info = SITE_INFO[station]
    src = PROJECT_ROOT / "my_data" / info["template"]
    out_dir = SMOKE_ROOT / "rendered_inputs" / station / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{station}_{year}_smoke.jinja2"
    text = src.read_text(encoding="utf-8", errors="replace")
    base_year_match = re.search(r"\b(20\d{2})\b", text)
    base_year = int(base_year_match.group(1)) if base_year_match else year
    old_yy = f"{base_year % 100:02d}"
    new_yy = f"{year % 100:02d}"
    planting = pd.Timestamp(planting_date)
    start = planting - pd.Timedelta(days=4)
    emergence = planting + pd.Timedelta(days=7)

    replacements = {
        r"\b\d{2}\d{3}\b": None,
    }
    text = re.sub(rf"\b{old_yy}(\d{{3}})\b", rf"{new_yy}\1", text)
    text = re.sub(r"\b20\d{2}\b", str(year), text)
    text = re.sub(r"\bSim20\d{2}\b", f"Sim{year}", text)
    text = re.sub(r"CN([A-Z]{2})20\d{2}", lambda m: f"CN{m.group(1)}{year}", text)
    text = re.sub(r"(@P PDATE EDATE[^\n]*\n\s*1\s+)(\d{5})(\s+)(\d{5})", rf"\g<1>{yyddd(planting_date)}\g<3>{yyddd(emergence.strftime('%Y-%m-%d'))}", text)
    text = re.sub(r"(\sS\s+)(\d{5})(\s+2150)", rf"\g<1>{yyddd(start.strftime('%Y-%m-%d'))}\g<3>", text)
    text = re.sub(r"(\sMZ\s+)(\d{5})(\s+100)", rf"\g<1>{yyddd(start.strftime('%Y-%m-%d'))}\g<3>", text)
    out.write_text(text, encoding="utf-8")
    return out


def build_env_args(station: str, year: int, planting_date: str, seed: int, mode: str = "all") -> dict:
    info = SITE_INFO[station]
    template = render_template(station, year, planting_date)
    source_weather = PROJECT_ROOT / "Leave_One_experiments" / "wth_generated_qc" / station / f"{info['weather_prefix']}{year}.WTH"
    rendered_weather = template.parent / info["expected_weather"]
    if source_weather.exists():
        shutil.copyfile(source_weather, rendered_weather)
    weather = rendered_weather
    cultivar = PROJECT_ROOT / "my_data" / "MZCER048.CUL"
    soil = PROJECT_ROOT / "my_data" / info["soil"]
    missing = [p for p in [template, weather, cultivar, soil] if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(str(p) for p in missing))
    return {
        "log_saving_path": str(SMOKE_ROOT / "logs" / f"{station}_{year}.log"),
        "mode": mode,
        "seed": seed,
        "random_weather": False,
        "evaluation": False,
        "fileX_template_path": str(template),
        "experiment_number": 1,
        "auxiliary_file_paths": [str(cultivar), str(weather), str(soil)],
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }


def scalar(value, default=np.nan):
    try:
        arr = np.asarray(value).flatten()
        if len(arr) == 0:
            return default
        return float(arr[0])
    except Exception:
        return default


def latest_observation_dict(env, obs_array, info) -> dict:
    obs_vars = list(getattr(env.unwrapped, "observation_variables", []))
    obs_dict = {name: scalar(value) for name, value in zip(obs_vars, np.asarray(obs_array).flatten())}
    history = getattr(env.unwrapped, "history", {})
    if isinstance(history, dict):
        observations = history.get("observation", [])
        if observations and isinstance(observations[-1], dict):
            obs_dict.update(observations[-1])
    if isinstance(info, dict):
        obs_dict.update(info)
    return obs_dict


def plot_episode(daily: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    x = daily["dap"]
    specs = [
        ("dap_swfac_irrigation_reward.png", ["swfac", "real_action_amir", "reward"], "Water stress, irrigation, reward"),
        ("dap_nstres_fertilization_reward.png", ["nstres", "real_action_anfer", "reward"], "Nitrogen stress, fertilization, reward"),
        ("crop_growth_timeseries.png", ["topwt", "grnwt", "xlai"], "Crop growth"),
        ("cumulative_water_nitrogen.png", ["totir", "tofer"], "Cumulative water and nitrogen"),
        ("daily_actions.png", ["real_action_amir", "real_action_anfer"], "Daily actions"),
    ]
    for filename, cols, title in specs:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        for col in cols:
            if col in daily.columns:
                ax.plot(x, pd.to_numeric(daily[col], errors="coerce"), label=col)
        ax.set_title(title)
        ax.set_xlabel("DAP")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figure_dir / filename, dpi=150)
        plt.close(fig)


def run_single_episode(config: dict, station: str, year: int, policy_name: str) -> dict:
    import gym
    from sb3_wrapper import GymDssatWrapper

    row = next(r for r in config["observed_years"] if r["station"] == station and int(r["year"]) == int(year))
    policy = POLICIES[policy_name]
    env_args = build_env_args(station, year, row["planting_date"], int(config.get("seed", 123)))
    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    records = []
    notes: list[str] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        cumulative_irrig = 0.0
        cumulative_n = 0.0
        while not done and step_count < int(config.get("max_steps", 260)):
            latest = latest_observation_dict(env, obs, info)
            dap = int(round(scalar(latest.get("dap", step_count))))
            real_action = policy.action_for_dap(dap, env.formator.action_names)
            clipped_action, clip_notes = clip_real_action(real_action, env.formator.action_space_dict)
            notes.extend(clip_notes)
            normalized = normalize_real_action(clipped_action, env.formator.action_names, env.formator.action_space_dict)
            obs, reward, terminated, truncated, info = env.step(normalized)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            real_amir = float(clipped_action.get("amir", 0.0))
            real_anfer = float(clipped_action.get("anfer", 0.0))
            cumulative_irrig += real_amir
            cumulative_n += real_anfer
            date = pd.Timestamp(row["planting_date"]) + pd.Timedelta(days=max(dap - 1, 0))
            normalized_by_name = {n: scalar(v) for n, v in zip(env.formator.action_names, normalized)}
            records.append(
                {
                    "station": station,
                    "year": year,
                    "observed_rain_label": row.get("observed_rain_label", ""),
                    "policy_name": policy_name,
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": scalar(latest.get("dap", dap)),
                    "topwt": scalar(latest.get("topwt")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "xlai": scalar(latest.get("xlai")),
                    "totir_raw": scalar(latest.get("totir")),
                    "totir": scalar(latest.get("totir"), cumulative_irrig) if not np.isnan(scalar(latest.get("totir"))) else cumulative_irrig,
                    "tofer": scalar(latest.get("tofer"), cumulative_n) if not np.isnan(scalar(latest.get("tofer"))) else cumulative_n,
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "reward": float(reward),
                    "real_action_amir": real_amir,
                    "real_action_anfer": real_anfer,
                    "normalized_action_amir": normalized_by_name.get("amir", np.nan),
                    "normalized_action_anfer": normalized_by_name.get("anfer", np.nan),
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1
        daily = pd.DataFrame(records)
        daily_dir = SMOKE_ROOT / "daily_outputs" / station
        daily_dir.mkdir(parents=True, exist_ok=True)
        daily_csv = daily_dir / f"{station}_{year}_{policy_name}_daily.csv"
        daily.to_csv(daily_csv, index=False, encoding="utf-8-sig")
        fig_dir = SMOKE_ROOT / "figures" / station / str(year) / policy_name
        plot_episode(daily, fig_dir)
        episode_completed = bool(records and records[-1]["done"])
        summary = {
            "station": station,
            "year": year,
            "observed_rain_label": row.get("observed_rain_label", ""),
            "harvest_window_rain_mm": row.get("harvest_window_rain_mm", np.nan),
            "policy_name": policy_name,
            "run_status": "ok" if episode_completed else "failed",
            "error_message": "" if episode_completed else "episode_not_completed_before_max_steps",
            "episode_completed": episode_completed,
            "episode_length": len(daily),
            "final_dap": daily["dap"].iloc[-1] if len(daily) else np.nan,
            "final_topwt": daily["topwt"].iloc[-1] if len(daily) else np.nan,
            "final_grnwt": daily["grnwt"].iloc[-1] if len(daily) else np.nan,
            "final_xlai": daily["xlai"].iloc[-1] if len(daily) else np.nan,
            "total_irrigation": float(daily["real_action_amir"].sum()) if len(daily) else 0.0,
            "total_n_fertilizer": float(daily["real_action_anfer"].sum()) if len(daily) else 0.0,
            "mean_swfac": float(pd.to_numeric(daily["swfac"], errors="coerce").mean()) if len(daily) else np.nan,
            "mean_nstres": float(pd.to_numeric(daily["nstres"], errors="coerce").mean()) if len(daily) else np.nan,
            "mean_reward": float(daily["reward"].mean()) if len(daily) else np.nan,
            "sum_reward": float(daily["reward"].sum()) if len(daily) else 0.0,
            "daily_csv_path": str(daily_csv.relative_to(PROJECT_ROOT)),
            "figure_dir": str(fig_dir.relative_to(PROJECT_ROOT)),
            "notes": ";".join(dict.fromkeys(notes)),
        }
        return summary
    finally:
        try:
            env.close()
        except Exception:
            pass


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_single_summary(summary: dict, station: str, year: int, policy: str) -> None:
    out_dir = SMOKE_ROOT / "evaluation" / "single"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{station}_{year}_{policy}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run_parent(args: argparse.Namespace) -> None:
    ensure_dirs()
    config = load_config()
    jobs = [(r["station"], int(r["year"]), p) for r in config["observed_years"] for p in config["policies"]]
    if args.minimal:
        jobs = [("HLA", 2007, "null_zero")]
    summaries = []
    for station, year, policy in jobs:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--single", "--station", station, "--year", str(year), "--policy", policy]
        try:
            completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=int(config.get("episode_timeout_seconds", 240)), text=True, capture_output=True)
            single_path = SMOKE_ROOT / "evaluation" / "single" / f"{station}_{year}_{policy}.json"
            if completed.returncode == 0 and single_path.exists():
                summaries.append(json.loads(single_path.read_text(encoding="utf-8")))
            else:
                summaries.append({
                    "station": station, "year": year, "policy_name": policy, "run_status": "failed",
                    "error_message": (completed.stderr or completed.stdout)[-1000:],
                    "episode_completed": False, "daily_csv_path": "", "figure_dir": "", "notes": "subprocess_failed",
                })
        except subprocess.TimeoutExpired:
            summaries.append({
                "station": station, "year": year, "policy_name": policy, "run_status": "timeout",
                "error_message": f"timeout_after_{config.get('episode_timeout_seconds', 240)}s",
                "episode_completed": False, "daily_csv_path": "", "figure_dir": "", "notes": "subprocess_timeout",
            })
    summary_df = pd.DataFrame(summaries)
    out = SMOKE_ROOT / "evaluation" / ("smoke_test_minimal_summary.csv" if args.minimal else "smoke_test_summary.csv")
    summary_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(out.relative_to(PROJECT_ROOT))
    print(summary_df["run_status"].value_counts(dropna=False).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--minimal", action="store_true")
    parser.add_argument("--station")
    parser.add_argument("--year", type=int)
    parser.add_argument("--policy")
    args = parser.parse_args()
    ensure_dirs()
    if args.single:
        try:
            summary = run_single_episode(load_config(), args.station, args.year, args.policy)
        except Exception as exc:
            summary = {
                "station": args.station,
                "year": args.year,
                "policy_name": args.policy,
                "run_status": "failed",
                "error_message": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}",
                "episode_completed": False,
                "daily_csv_path": "",
                "figure_dir": "",
                "notes": "single_exception",
            }
        write_single_summary(summary, args.station, args.year, args.policy)
        if summary["run_status"] != "ok":
            print(summary["error_message"])
            raise SystemExit(1)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
