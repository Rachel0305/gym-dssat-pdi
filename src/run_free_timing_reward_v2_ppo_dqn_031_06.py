from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

import run_all_year_direct_action_safe_ppo as direct_ppo
from ppo_action_safety import ActionSafetyState, apply_action_safety, normalize_action, update_action_safety_state


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "031_06_free_timing_reward_v2_ppo_dqn"
DOC = ROOT / "docs" / "031_06_free_timing_reward_v2_ppo_dqn_record.md"
CONFIG_PPO = ROOT / "experiments" / "ppo_observed_years" / "config_031_06_free_timing_reward_v2_ppo.yaml"
CONFIG_DQN = ROOT / "experiments" / "ppo_observed_years" / "config_031_06_free_timing_reward_v2_dqn.yaml"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"


def ensure_dirs(base: Path) -> None:
    for rel in ["configs", "models/SYA", "tensorboard/SYA", "evaluation", "daily_outputs/SYA", "logs", "rendered_inputs", "reports"]:
        (base / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def make_sy2014_selection() -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    row = pool[(pool["station_code"].eq("SYA")) & (pool["year"].astype(int).eq(2014))].copy()
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one SYA 2014 row, found {len(row)}")
    row["selected_for_train"] = True
    row["selected_for_eval"] = True
    row["selection_reason"] = "031_06_free_timing_reward_v2_SY2014_seed0"
    return row


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def latest_observation_dict(env, obs=None, info=None) -> dict[str, Any]:
    return direct_ppo.latest_observation_dict(env, obs, info)


def timing_reward(config: dict, prev: dict[str, Any], latest: dict[str, Any], safety_state_before: ActionSafetyState, safe_i: float, safe_n: float, dap: int) -> tuple[float, dict[str, float]]:
    cfg = config["reward"]
    topwt_delta = max(0.0, scalar(latest.get("topwt"), 0.0) - scalar(prev.get("topwt"), 0.0))
    grnwt_delta = max(0.0, scalar(latest.get("grnwt"), 0.0) - scalar(prev.get("grnwt"), 0.0))
    growth_reward = float(cfg["topwt_delta_coef"]) * topwt_delta + float(cfg["grnwt_delta_coef"]) * grnwt_delta
    cost = float(cfg["water_cost"]) * safe_i + float(cfg["nitrogen_cost"]) * safe_n

    repeat_window = float(cfg.get("repeat_window_days", 7))
    repeat_i = 0.0
    repeat_n = 0.0
    if safe_i > 0 and safety_state_before.last_irrigation_dap is not None:
        gap = max(0.0, float(dap - int(safety_state_before.last_irrigation_dap)))
        repeat_i = float(cfg["repeat_irrigation_penalty_coef"]) * safe_i * max(0.0, repeat_window - gap) / repeat_window
    if safe_n > 0 and safety_state_before.last_fertilization_dap is not None:
        gap = max(0.0, float(dap - int(safety_state_before.last_fertilization_dap)))
        repeat_n = float(cfg["repeat_nitrogen_penalty_coef"]) * safe_n * max(0.0, repeat_window - gap) / repeat_window

    early_i = 0.0
    early_n = 0.0
    if dap <= int(cfg["early_dap_threshold"]):
        if safety_state_before.cumulative_irrigation >= float(cfg["early_irrigation_cumulative_threshold"]):
            early_i = float(cfg["early_irrigation_penalty_coef"]) * safe_i
        if safety_state_before.cumulative_n >= float(cfg["early_n_cumulative_threshold"]):
            early_n = float(cfg["early_nitrogen_penalty_coef"]) * safe_n

    late_i = float(cfg["late_irrigation_penalty_coef"]) * safe_i if dap >= int(cfg["late_irrigation_dap_threshold"]) else 0.0
    late_n = float(cfg["late_nitrogen_penalty_coef"]) * safe_n if dap >= int(cfg["late_nitrogen_dap_threshold"]) else 0.0
    shaping_penalty = repeat_i + repeat_n + early_i + early_n + late_i + late_n
    reward = growth_reward - cost - shaping_penalty
    parts = {
        "growth_reward": growth_reward,
        "topwt_delta": topwt_delta,
        "grnwt_delta": grnwt_delta,
        "resource_cost": cost,
        "repeat_penalty": repeat_i + repeat_n,
        "early_excess_penalty": early_i + early_n,
        "late_penalty": late_i + late_n,
        "timing_shaping_penalty": shaping_penalty,
        "reward_v2": reward,
    }
    return float(reward), parts


class TimingRewardPPOWrapper(gym.Env):
    def __init__(self, env, config: dict):
        super().__init__()
        self.env = env
        self.config = config
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = getattr(env, "metadata", {})
        self.safety_state = ActionSafetyState()
        self.last_action_info: dict[str, Any] = {}
        self.last_obs_dict: dict[str, Any] = {}

    def reset(self, *args, **kwargs):
        self.safety_state = ActionSafetyState()
        self.last_action_info = {}
        result = self.env.reset(*args, **kwargs)
        obs, info = result if isinstance(result, tuple) else (result, {})
        self.last_obs_dict = latest_observation_dict(self.env, obs, info)
        return obs, info

    def step(self, action):
        prev = dict(self.last_obs_dict)
        dap = int(round(scalar(prev.get("dap"), 1)))
        action_names = list(self.env.formator.action_names)
        scaled_real = direct_ppo.physical_from_policy_action(self.config, action, action_names)
        before = ActionSafetyState(**self.safety_state.__dict__)
        safety = {**self.config["action_safety"], "enabled": True}
        safety_result = apply_action_safety(scaled_real, dap, self.safety_state, safety)
        safe_norm = normalize_action(action_names, self.env.formator.action_space_dict, safety_result.safe_real_action)
        obs, _env_reward, terminated, truncated, info = self.env.step(safe_norm)
        update_action_safety_state(self.safety_state, safety_result.safe_real_action, dap)
        latest = latest_observation_dict(self.env, obs, info)
        self.last_obs_dict = latest
        safe_i = float(safety_result.safe_real_action.get("amir", 0.0))
        safe_n = float(safety_result.safe_real_action.get("anfer", 0.0))
        reward, parts = timing_reward(self.config, prev, latest, before, safe_i, safe_n, dap)
        raw_arr = np.asarray(action, dtype=float).flatten()
        raw_norm = {name: float(value) for name, value in zip(action_names, raw_arr)}
        self.last_action_info = {
            "raw_action_amir": raw_norm.get("amir", np.nan),
            "raw_action_anfer": raw_norm.get("anfer", np.nan),
            "scaled_action_amir": float(scaled_real.get("amir", 0.0)),
            "scaled_action_anfer": float(scaled_real.get("anfer", 0.0)),
            "safe_action_amir": safe_i,
            "safe_action_anfer": safe_n,
            "season_cumulative_irrigation": float(self.safety_state.cumulative_irrigation),
            "season_cumulative_n": float(self.safety_state.cumulative_n),
            "action_safety_triggered": safety_result.safety_rule_triggered,
            **parts,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)


def action_grid(config: dict) -> list[dict[str, float]]:
    return [{"amir": float(i), "anfer": float(n)} for i in config["discrete_actions"]["irrigation_levels"] for n in config["discrete_actions"]["nitrogen_levels"]]


class TimingRewardDQNWrapper(TimingRewardPPOWrapper):
    def __init__(self, env, config: dict):
        super().__init__(env, config)
        self.grid = action_grid(config)
        self.action_space = spaces.Discrete(len(self.grid))

    def step(self, action):
        prev = dict(self.last_obs_dict)
        dap = int(round(scalar(prev.get("dap"), 1)))
        idx = int(np.asarray(action).item())
        raw_real = dict(self.grid[idx])
        before = ActionSafetyState(**self.safety_state.__dict__)
        safety = {**self.config["action_safety"], "enabled": True}
        safety_result = apply_action_safety(raw_real, dap, self.safety_state, safety)
        action_names = list(self.env.formator.action_names)
        safe_norm = normalize_action(action_names, self.env.formator.action_space_dict, safety_result.safe_real_action)
        obs, _env_reward, terminated, truncated, info = self.env.step(safe_norm)
        update_action_safety_state(self.safety_state, safety_result.safe_real_action, dap)
        latest = latest_observation_dict(self.env, obs, info)
        self.last_obs_dict = latest
        safe_i = float(safety_result.safe_real_action.get("amir", 0.0))
        safe_n = float(safety_result.safe_real_action.get("anfer", 0.0))
        reward, parts = timing_reward(self.config, prev, latest, before, safe_i, safe_n, dap)
        self.last_action_info = {
            "discrete_action_index": idx,
            "raw_action_amir": raw_real["amir"],
            "raw_action_anfer": raw_real["anfer"],
            "scaled_action_amir": raw_real["amir"],
            "scaled_action_anfer": raw_real["anfer"],
            "safe_action_amir": safe_i,
            "safe_action_anfer": safe_n,
            "season_cumulative_irrigation": float(self.safety_state.cumulative_irrigation),
            "season_cumulative_n": float(self.safety_state.cumulative_n),
            "action_safety_triggered": safety_result.safety_rule_triggered,
            **parts,
        }
        return obs, reward, terminated, truncated, info


def make_env(config: dict, env_config: dict, algorithm: str, station: str, year: int, seed: int, tag: str, evaluation: bool):
    base = direct_ppo.make_base_env(env_config, station, year, seed, tag, evaluation=evaluation)
    return TimingRewardDQNWrapper(base, config) if algorithm == "DQN" else TimingRewardPPOWrapper(base, config)


def algo_kwargs(config: dict, algorithm: str) -> dict[str, Any]:
    if algorithm == "PPO":
        return {k: config["ppo"][k] for k in ["learning_rate", "gamma", "n_steps", "batch_size", "ent_coef", "clip_range"]}
    return {k: config["dqn"][k] for k in ["learning_rate", "buffer_size", "learning_starts", "batch_size", "gamma", "train_freq", "gradient_steps", "target_update_interval", "exploration_fraction", "exploration_initial_eps", "exploration_final_eps"]}


def train(config_path: Path, algorithm: str, selection: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from stable_baselines3 import DQN, PPO

    out = OUT / algorithm.lower()
    ensure_dirs(out)
    shutil.copyfile(config_path, out / "configs" / config_path.name)
    config = direct_ppo.load_yaml(config_path)
    direct_ppo.OUTPUT_ROOT = out
    selection.to_csv(out / "configs" / "031_06_sy2014_selection.csv", index=False, encoding="utf-8-sig")
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, out / "configs" / "031_06_resolved_env_config.yaml")
    station = "SYA"
    year = 2014
    seed = int(config["seed"])
    model_path = out / "models" / station / f"{algorithm.lower()}_free_timing_reward_v2_seed0.zip"
    env = None
    status = "ok"
    notes = ""
    try:
        env = make_env(config, env_config, algorithm, station, year, seed, f"{station}_{year}_{algorithm}_031_06_train", evaluation=False)
        cls = PPO if algorithm == "PPO" else DQN
        model = cls("MlpPolicy", env, verbose=1, seed=seed, tensorboard_log=str(out / "tensorboard" / station), **algo_kwargs(config, algorithm))
        model.learn(total_timesteps=int(config["total_timesteps"]), reset_num_timesteps=True, progress_bar=False)
        model.save(str(model_path.with_suffix("")))
    except Exception:
        status = "failed"
        notes = traceback.format_exc()
    finally:
        if env is not None:
            env.close()
    summary = pd.DataFrame([{"algorithm": algorithm, "station_code": station, "train_years": "2014", "seed": seed, "total_timesteps": int(config["total_timesteps"]), "run_status": status, "model_path": str(model_path.relative_to(ROOT)) if model_path.exists() else "", "notes": notes[-1500:] if notes else ""}])
    summary.to_csv(out / "evaluation" / "training_run_summary.csv", index=False, encoding="utf-8-sig")
    return summary, {"config": config, "env_config": env_config, "out": out, "model_path": model_path}


def evaluate(algorithm: str, train_summary: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    from stable_baselines3 import DQN, PPO

    rows: list[dict[str, Any]] = []
    config = ctx["config"]
    env_config = ctx["env_config"]
    out = ctx["out"]
    weather = direct_ppo.weather_for_daily(config)
    row = train_summary.iloc[0]
    if str(row["run_status"]) != "ok":
        return pd.DataFrame()
    model = (PPO if algorithm == "PPO" else DQN).load(str(ROOT / str(row["model_path"])))
    station = "SYA"
    year = 2014
    for split in ["train", "eval"]:
        env = make_env(config, env_config, algorithm, station, year, int(config["seed"]), f"{station}_{year}_{algorithm}_031_06_eval", evaluation=True)
        records: list[dict[str, Any]] = []
        try:
            obs, info = env.reset()
            done = False
            step_count = 0
            planting = pd.Timestamp(direct_ppo.find_year(env_config, station, year)["planting_date"])
            while not done and step_count < int(config["runtime"]["max_steps"]):
                latest = latest_observation_dict(env, obs, info)
                dap = int(round(scalar(latest.get("dap"), step_count + 1)))
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                latest = latest_observation_dict(env, obs, info)
                date = planting + pd.Timedelta(days=max(dap - 1, 0))
                w = weather[(weather["station_code"].eq(station)) & (weather["date"].eq(date))]
                wrow = w.iloc[0].to_dict() if len(w) else {}
                records.append({"algorithm": algorithm, "split": split, "station_code": station, "year": year, "date": date.strftime("%Y-%m-%d"), "doy": int(date.dayofyear), "dap": dap, "rain": scalar(wrow.get("rain"), np.nan), "srad": scalar(wrow.get("srad"), np.nan), "tmax": scalar(wrow.get("tmax"), np.nan), "tmin": scalar(wrow.get("tmin"), np.nan), "swfac": scalar(latest.get("swfac")), "nstres": scalar(latest.get("nstres")), "topwt": scalar(latest.get("topwt")), "grnwt": scalar(latest.get("grnwt")), "xlai": scalar(latest.get("xlai")), "reward": float(reward), **dict(env.last_action_info), "done": done, "info": json.dumps(info, ensure_ascii=False, default=str)})
                step_count += 1
        finally:
            env.close()
        daily = pd.DataFrame(records)
        daily_path = out / "daily_outputs" / station / f"2014_{split}_{algorithm.lower()}_reward_v2_daily.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0)
        n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0)
        swfac = pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce")
        nstres = pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce")
        final_grnwt = float(pd.to_numeric(daily.get("grnwt", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if len(daily) else np.nan
        rows.append({"algorithm": algorithm, "policy_name": f"{algorithm.lower()}_reward_v2_5k", "split": split, "run_status": "ok" if len(daily) and bool(daily["done"].iloc[-1]) else "failed", "episode_length": int(len(daily)), "final_grnwt": final_grnwt, "total_irrigation": float(irr.sum()), "total_n": float(n.sum()), "profit_simple": final_grnwt - float(irr.sum()) - 5.0 * float(n.sum()) if np.isfinite(final_grnwt) else np.nan, "reward_v2_sum": float(pd.to_numeric(daily.get("reward", pd.Series(dtype=float)), errors="coerce").sum()) if len(daily) else np.nan, "timing_penalty_sum": float(pd.to_numeric(daily.get("timing_shaping_penalty", pd.Series(dtype=float)), errors="coerce").sum()) if len(daily) else np.nan, "irrigation_event_count": int((irr > 0).sum()), "n_event_count": int((n > 0).sum()), "first_irrigation_dap": int(daily.loc[irr > 0, "dap"].iloc[0]) if (irr > 0).any() else np.nan, "first_n_dap": int(daily.loc[n > 0, "dap"].iloc[0]) if (n > 0).any() else np.nan, "swfac_stress_days_gt_0p05": int((swfac > 0.05).sum()), "nstres_days_gt_0p05": int((nstres > 0.05).sum()), "daily_csv_path": str(daily_path.relative_to(ROOT))})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "evaluation" / f"{algorithm.lower()}_reward_v2_eval_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def load_reference_rows() -> pd.DataFrame:
    paths = [
        ROOT / "benchmark_results" / "031_04_free_daily_original_reward_5k_train" / "031_04_ppo_dqn_5k_vs_baselines.csv",
        ROOT / "benchmark_results" / "031_05_sy2014_free_timing_rule_sensitivity" / "evaluation" / "031_05_rule_timing_sensitivity_summary.csv",
    ]
    rows = []
    if paths[0].exists():
        df = pd.read_csv(paths[0])
        rows.append(df[df.get("source", "").eq("031_04")].copy())
    if paths[1].exists():
        rules = pd.read_csv(paths[1])
        rules["source"] = "031_05"
        rules["policy_name"] = rules["rule"]
        rows.append(rules)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selection = make_sy2014_selection()
    ppo_train, ppo_ctx = train(CONFIG_PPO, "PPO", selection)
    ppo_eval = evaluate("PPO", ppo_train, ppo_ctx)
    dqn_train, dqn_ctx = train(CONFIG_DQN, "DQN", selection)
    dqn_eval = evaluate("DQN", dqn_train, dqn_ctx)

    eval_rows = pd.concat([ppo_eval[ppo_eval["split"].eq("eval")], dqn_eval[dqn_eval["split"].eq("eval")]], ignore_index=True, sort=False)
    eval_rows["source"] = "031_06"
    refs = load_reference_rows()
    cols = ["source", "policy_name", "run_status", "final_grnwt", "total_irrigation", "total_n", "profit_simple", "reward_v2_sum", "timing_penalty_sum", "irrigation_event_count", "n_event_count", "first_irrigation_dap", "first_n_dap", "swfac_stress_days_gt_0p05", "nstres_days_gt_0p05", "daily_csv_path"]
    comparison = pd.concat([eval_rows, refs], ignore_index=True, sort=False)
    comparison = comparison[[c for c in cols if c in comparison.columns]]
    comparison_path = OUT / "031_06_reward_v2_vs_references.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    result = {"task": "031_06_free_timing_reward_v2_ppo_dqn", "training_or_dssat_run": True, "site_year": "SYA2014", "seed": 0, "timesteps": 5000, "ppo_train": ppo_train.to_dict(orient="records"), "ppo_eval": ppo_eval.to_dict(orient="records"), "dqn_train": dqn_train.to_dict(orient="records"), "dqn_eval": dqn_eval.to_dict(orient="records"), "comparison": str(comparison_path.relative_to(ROOT)), "scope": "single site-year reward-v2 smoke only"}
    (OUT / "031_06_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    lines = ["# 031_06 Free-timing reward v2 PPO/DQN record", "", "## Scope", "", "- SYA2014 seed0 only.", "- Daily free timing; no expert DAP windows.", "- PPO and DQN, 5,000 timesteps each.", "- Reward v2 includes TOPWT delta plus soft timing penalties for repeated, early-excess, and late operations.", "", "## PPO eval", "", ppo_eval.to_string(index=False), "", "## DQN eval", "", dqn_eval.to_string(index=False), "", "## Comparison", "", comparison.to_string(index=False), "", "## Interpretation boundary", "", "This is a first reward-v2 smoke. It can show whether the two algorithms move away from 031_04 bad modes, but it is not a cross-seed or cross-year claim."]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "031_06_free_timing_reward_v2_ppo_dqn_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
