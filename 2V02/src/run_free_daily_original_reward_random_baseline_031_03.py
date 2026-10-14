from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_free_daily_original_reward_dqn_smoke_031_02 as dqn031


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PPO = ROOT / "experiments" / "ppo_observed_years" / "config_031_01_free_daily_original_reward_smoke.yaml"
CONFIG_DQN = ROOT / "experiments" / "ppo_observed_years" / "config_031_02_free_daily_original_reward_dqn_smoke.yaml"
OUT = ROOT / "benchmark_results" / "031_03_free_daily_original_reward_random_baseline"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"


def ensure_dirs() -> None:
    for rel in ["configs", "daily_outputs/SYA", "evaluation", "logs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)


def make_sy2014_selection() -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    row = pool[(pool["station_code"].eq("SYA")) & (pool["year"].astype(int).eq(2014))].copy()
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one SYA 2014 row, found {len(row)}")
    row["selected_for_train"] = True
    row["selected_for_eval"] = True
    row["selection_reason"] = "031_03_single_smoke_SY2014_free_daily_original_reward_random_baseline"
    return row


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def run_policy(
    *,
    policy_name: str,
    wrapper_type: str,
    action_mode: str,
    config: dict,
    env_config: dict,
    weather: pd.DataFrame,
    seed: int = 0,
) -> dict[str, Any]:
    station = "SYA"
    year = 2014
    rng = np.random.default_rng(seed)
    if wrapper_type == "continuous":
        env = direct_ppo.make_training_env(config, env_config, station, year, seed, f"{station}_{year}_{policy_name}")
    elif wrapper_type == "dqn_discrete":
        env = dqn031.make_training_env(config, env_config, station, year, seed, f"{station}_{year}_{policy_name}", evaluation=True)
    else:
        raise ValueError(wrapper_type)

    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        year_info = direct_ppo.find_year(env_config, station, year)
        planting = pd.Timestamp(year_info["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = direct_ppo.latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            if action_mode == "noop":
                if wrapper_type == "continuous":
                    # Continuous PPO actions are normalized. Zero is the midpoint of the
                    # physical action range, not physical no-op. The lower bound maps to
                    # 0 water and 0 nitrogen.
                    action = np.asarray(env.action_space.low, dtype=np.float32)
                else:
                    action = 0
            elif action_mode == "random":
                if wrapper_type == "continuous":
                    action = env.action_space.sample()
                else:
                    action = int(rng.integers(0, int(env.action_space.n)))
            else:
                raise ValueError(action_mode)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = direct_ppo.latest_observation_dict(env, obs, info)
            action_info = dict(env.last_action_info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(station)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "policy_name": policy_name,
                    "station_code": station,
                    "year": year,
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": scalar(wrow.get("rain"), np.nan),
                    "srad": scalar(wrow.get("srad"), np.nan),
                    "tmax": scalar(wrow.get("tmax"), np.nan),
                    "tmin": scalar(wrow.get("tmin"), np.nan),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "topwt": scalar(latest.get("topwt")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "xlai": scalar(latest.get("xlai")),
                    "reward": float(reward),
                    **action_info,
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1
    finally:
        env.close()

    daily = pd.DataFrame(records)
    daily_path = OUT / "daily_outputs" / station / f"2014_{policy_name}_daily.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    swfac = pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce")
    nstres = pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce")
    irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0)
    n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0)
    final_grnwt = float(pd.to_numeric(daily.get("grnwt", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if len(daily) else np.nan
    total_i = float(irr.sum())
    total_n = float(n.sum())
    profit_simple = final_grnwt - total_i - 5.0 * total_n if np.isfinite(final_grnwt) else np.nan
    return {
        "policy_name": policy_name,
        "wrapper_type": wrapper_type,
        "action_mode": action_mode,
        "station_code": station,
        "year": year,
        "run_status": "ok" if len(daily) and bool(daily["done"].iloc[-1]) else "failed",
        "episode_completed": bool(len(daily) and bool(daily["done"].iloc[-1])),
        "episode_length": int(len(daily)),
        "final_grnwt": final_grnwt,
        "total_irrigation": total_i,
        "total_n": total_n,
        "profit_simple": profit_simple,
        "swfac_stress_days_gt_0p05": int((swfac > 0.05).sum()) if len(swfac) else 0,
        "nstres_days_gt_0p05": int((nstres > 0.05).sum()) if len(nstres) else 0,
        "irrigation_event_count": int((irr > 0).sum()),
        "n_event_count": int((n > 0).sum()),
        "first_irrigation_dap": int(daily.loc[irr > 0, "dap"].iloc[0]) if (irr > 0).any() else np.nan,
        "first_n_dap": int(daily.loc[n > 0, "dap"].iloc[0]) if (n > 0).any() else np.nan,
        "daily_csv_path": str(daily_path.relative_to(ROOT)),
    }


def main() -> None:
    ensure_dirs()
    config_ppo = direct_ppo.load_yaml(CONFIG_PPO)
    config_dqn = direct_ppo.load_yaml(CONFIG_DQN)
    selection = make_sy2014_selection()
    selection.to_csv(OUT / "configs" / "031_03_sy2014_selection.csv", index=False, encoding="utf-8-sig")
    direct_ppo.OUTPUT_ROOT = OUT
    env_config_ppo = direct_ppo.build_env_config(config_ppo, selection)
    direct_ppo.write_yaml(env_config_ppo, OUT / "configs" / "031_03_resolved_env_config_ppo.yaml")
    env_config_dqn = direct_ppo.build_env_config(config_dqn, selection)
    direct_ppo.write_yaml(env_config_dqn, OUT / "configs" / "031_03_resolved_env_config_dqn.yaml")
    weather = direct_ppo.weather_for_daily(config_ppo)

    rows = [
        run_policy(policy_name="continuous_noop", wrapper_type="continuous", action_mode="noop", config=config_ppo, env_config=env_config_ppo, weather=weather, seed=0),
        run_policy(policy_name="continuous_random", wrapper_type="continuous", action_mode="random", config=config_ppo, env_config=env_config_ppo, weather=weather, seed=0),
        run_policy(policy_name="dqn_noop", wrapper_type="dqn_discrete", action_mode="noop", config=config_dqn, env_config=env_config_dqn, weather=weather, seed=0),
        run_policy(policy_name="dqn_random", wrapper_type="dqn_discrete", action_mode="random", config=config_dqn, env_config=env_config_dqn, weather=weather, seed=0),
    ]
    summary = pd.DataFrame(rows)
    summary_path = OUT / "evaluation" / "031_03_random_noop_baseline_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    result = {
        "task": "031_03_free_daily_original_reward_random_baseline",
        "training_or_dssat_run": True,
        "training_run": False,
        "non_scientific_failed_attempt": "First local draft used normalized zero for continuous_noop; normalized zero is midpoint action, not physical no-op. Corrected to action_space.low before final record.",
        "summary": str(summary_path.relative_to(ROOT)),
        "rows": summary.to_dict(orient="records"),
        "interpretation_scope": "No-training random/no-op baseline for 031_01/031_02 interpretation.",
    }
    (OUT / "031_03_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    lines = [
        "# 031_03 Free-daily original-reward random/no-op baseline record",
        "",
        "## Purpose",
        "",
        "Check whether cap saturation appears even without learning under the free-daily original-reward setup.",
        "",
        "## Summary",
        "",
        summary.to_string(index=False),
        "",
        "## Non-scientific correction",
        "",
        "The first draft of this script used normalized zero for `continuous_noop`. In this environment normalized zero maps to the midpoint of the physical action range, not physical no-op. The final run uses `action_space.low` for continuous no-op.",
        "",
        "## Interpretation",
        "",
        "- If random policies reach caps, 031_01/031_02 cap saturation should not be described as a learned stable strategy.",
        "- No-op policies verify that caps are not imposed automatically by the wrapper.",
        "- This task contains no training and no reward tuning.",
    ]
    (OUT / "031_03_free_daily_original_reward_random_baseline_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
