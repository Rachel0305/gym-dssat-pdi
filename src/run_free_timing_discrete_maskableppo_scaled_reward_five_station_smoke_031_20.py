from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_free_timing_discrete_maskableppo_ncost2x_scaled_reward_031_17 as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_20_free_timing_discrete_maskableppo_scaled_reward_five_station_smoke.yaml"
OUT = ROOT / "benchmark_results" / "031_20_free_timing_discrete_maskableppo_scaled_reward_five_station_smoke"
DOC = ROOT / "docs" / "031_20_free_timing_discrete_maskableppo_scaled_reward_five_station_smoke_record.md"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"


def ensure_dirs() -> None:
    for rel in ["configs", "evaluation", "logs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def selected_site_years(config: dict) -> list[tuple[str, int]]:
    return [(str(x["station_code"]), int(x["year"])) for x in config["smoke_site_years"]]


def make_selection(config: dict) -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    wanted = set(selected_site_years(config))
    frames = []
    for station, year in wanted:
        row = pool[(pool["station_code"].eq(station)) & (pool["year"].astype(int).eq(year))].copy()
        if len(row) != 1:
            raise RuntimeError(f"Expected exactly one row for {station}{year}, found {len(row)}")
        row["selected_for_train"] = True
        row["selected_for_eval"] = True
        row["selection_reason"] = "031_20_free_timing_discrete_maskableppo_scaled_reward_five_station_smoke"
        frames.append(row)
    return pd.concat(frames, ignore_index=True)


def train_one(config: dict, env_config: dict, station: str, year: int, seed: int) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO

    model_dir = OUT / "models" / station
    model_dir.mkdir(parents=True, exist_ok=True)
    (OUT / "tensorboard" / station).mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{station}_{year}_free_timing_discrete_maskableppo_scaled_reward_seed{seed}.zip"
    env = None
    status = "ok"
    notes = ""
    try:
        env = base.make_env(config, env_config, station, year, seed, f"{station}_{year}_031_20_train", evaluation=False)
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            tensorboard_log=str(OUT / "tensorboard" / station),
            **base.ppo_kwargs(config),
        )
        model.learn(total_timesteps=int(config["total_timesteps"]), reset_num_timesteps=True, progress_bar=False)
        model.save(str(model_path.with_suffix("")))
    except Exception:
        status = "failed"
        notes = traceback.format_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return {
        "algorithm": "free_timing_discrete_MaskablePPO_scaled_reward",
        "station_code": station,
        "train_years": str(year),
        "year": int(year),
        "seed": seed,
        "total_timesteps": int(config["total_timesteps"]),
        "run_status": status,
        "model_path": str(model_path.relative_to(ROOT)) if model_path.exists() else "",
        "notes": notes[-3000:] if notes else "",
    }


def summarize_daily(daily: pd.DataFrame, daily_path: Path, split: str, model_path: Path, station: str, year: int, seed: int) -> dict[str, Any]:
    row = base.summarize_daily(daily, daily_path, split, model_path)
    row["algorithm"] = "free_timing_discrete_MaskablePPO_scaled_reward"
    row["policy_name"] = f"{station}_{year}_free_discrete_maskableppo_scaled_reward_20k_seed{seed}"
    row["station_code"] = station
    row["year"] = int(year)
    row["seed"] = int(seed)
    return row


def evaluate_one(config: dict, env_config: dict, train_row: dict[str, Any], split: str) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    station = str(train_row["station_code"])
    year = int(train_row["year"])
    seed = int(train_row["seed"])
    if str(train_row["run_status"]) != "ok":
        return {
            "algorithm": "free_timing_discrete_MaskablePPO_scaled_reward",
            "station_code": station,
            "year": year,
            "seed": seed,
            "split": split,
            "run_status": "failed",
            "notes": train_row.get("notes", ""),
        }
    model_path = ROOT / str(train_row["model_path"])
    model = MaskablePPO.load(str(model_path), device="cpu")
    weather = direct_ppo.weather_for_daily(config)
    env = base.make_env(config, env_config, station, year, seed, f"{station}_{year}_031_20_{split}", evaluation=True)
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, station, year)["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = base.latest_observation_dict(env, obs, info)
            dap_raw = base.scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = base.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(station)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "station_code": station,
                    "year": int(year),
                    "seed": seed,
                    "split": split,
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": base.scalar(wrow.get("rain"), np.nan),
                    "srad": base.scalar(wrow.get("srad"), np.nan),
                    "tmax": base.scalar(wrow.get("tmax"), np.nan),
                    "tmin": base.scalar(wrow.get("tmin"), np.nan),
                    "swfac": base.scalar(latest.get("swfac")),
                    "nstres": base.scalar(latest.get("nstres")),
                    "topwt": base.scalar(latest.get("topwt")),
                    "grnwt": base.scalar(latest.get("grnwt")),
                    "xlai": base.scalar(latest.get("xlai")),
                    "reward": float(reward),
                    **dict(env.last_action_info),
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1
    finally:
        env.close()

    daily = pd.DataFrame(records)
    daily_dir = OUT / "daily_outputs" / station
    daily_dir.mkdir(parents=True, exist_ok=True)
    daily_path = daily_dir / f"{year}_{split}_free_discrete_maskableppo_scaled_reward_seed{seed}_daily.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    return summarize_daily(daily, daily_path, split, model_path, station, year, seed)


def write_record(train_summary: pd.DataFrame, eval_summary: pd.DataFrame) -> None:
    eval_only = eval_summary[eval_summary["split"].eq("eval")].copy() if "split" in eval_summary else eval_summary.copy()
    lines = [
        "# 031_20 Free-timing discrete MaskablePPO scaled-reward five-station smoke record",
        "",
        "## Scope",
        "",
        "- Five stations, one site-year per station.",
        "- Seed0 only.",
        "- Daily free timing; no expert DAP windows.",
        "- Same fixed scaled-reward discrete MaskablePPO configuration as 031_17/031_18.",
        "- This is a cross-station smoke, not a final all-year claim.",
        "",
        "## Training summary",
        "",
        train_summary.to_string(index=False) if not train_summary.empty else "No training rows.",
        "",
        "## Evaluation summary",
        "",
        eval_summary.to_string(index=False) if not eval_summary.empty else "No evaluation rows.",
        "",
        "## Eval-only compact summary",
        "",
        eval_only[[
            "station_code",
            "year",
            "seed",
            "run_status",
            "final_grnwt",
            "total_irrigation",
            "total_n",
            "profit_simple",
            "PFP_N",
            "early_dap1_10_irrigation",
            "early_dap1_10_n",
            "swfac_stress_days_gt_0p05",
            "nstres_days_gt_0p05",
            "action_sequence",
        ]].to_string(index=False) if not eval_only.empty else "No eval rows.",
        "",
        "## Interpretation boundary",
        "",
        "This smoke only checks cross-station execution and gross behavior for the currently strongest free-timing PPO branch. Do not use it as a final all-years conclusion.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    config = direct_ppo.load_yaml(CONFIG)
    seed = int(config["seed"])
    selection = make_selection(config)
    selection.to_csv(OUT / "configs" / "031_20_five_station_selection.csv", index=False, encoding="utf-8-sig")
    direct_ppo.OUTPUT_ROOT = OUT
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "031_20_resolved_env_config.yaml")

    train_rows = []
    eval_rows = []
    for station, year in selected_site_years(config):
        train_row = train_one(config, env_config, station, year, seed)
        train_rows.append(train_row)
        eval_rows.append(evaluate_one(config, env_config, train_row, "train"))
        eval_rows.append(evaluate_one(config, env_config, train_row, "eval"))

    train_summary = pd.DataFrame(train_rows)
    eval_summary = pd.DataFrame(eval_rows)
    train_path = OUT / "evaluation" / "training_run_summary.csv"
    eval_path = OUT / "evaluation" / "eval_summary.csv"
    train_summary.to_csv(train_path, index=False, encoding="utf-8-sig")
    eval_summary.to_csv(eval_path, index=False, encoding="utf-8-sig")
    write_record(train_summary, eval_summary)

    result = {
        "task": "031_20_free_timing_discrete_maskableppo_scaled_reward_five_station_smoke",
        "training_run": True,
        "site_years": [f"{s}{y}" for s, y in selected_site_years(config)],
        "seed": seed,
        "record_md": str(DOC.relative_to(ROOT)),
        "train_summary": str(train_path.relative_to(ROOT)),
        "eval_summary": str(eval_path.relative_to(ROOT)),
    }
    (OUT / "031_20_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(eval_summary[eval_summary["split"].eq("eval")].to_string(index=False))


if __name__ == "__main__":
    main()

