from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
from ppo_action_safety import ActionSafetyState, apply_action_safety, normalize_action, update_action_safety_state


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_08_free_timing_reward_v3_ppo.yaml"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"
OUT = ROOT / "benchmark_results" / "031_08_sy2014_free_timing_reward_v3_ppo_smoke"
DOC = ROOT / "docs" / "031_08_sy2014_free_timing_reward_v3_ppo_smoke_record.md"


def ensure_dirs(base: Path) -> None:
    for rel in ["configs", "models/SYA", "tensorboard/SYA", "evaluation", "daily_outputs/SYA", "logs"]:
        (base / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def make_sy2014_selection() -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    row = pool[(pool["station_code"].eq("SYA")) & (pool["year"].astype(int).eq(2014))].copy()
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one SYA 2014 row, found {len(row)}")
    row["selected_for_train"] = True
    row["selected_for_eval"] = True
    row["selection_reason"] = "031_08_free_timing_reward_v3_SY2014_seed0"
    return row


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def latest_observation_dict(env, obs=None, info=None) -> dict[str, Any]:
    return direct_ppo.latest_observation_dict(env, obs, info)


def reward_v3(
    config: dict,
    prev: dict[str, Any],
    latest: dict[str, Any],
    safety_state_before: ActionSafetyState,
    safe_i: float,
    safe_n: float,
    dap: int,
) -> tuple[float, dict[str, float]]:
    cfg = config["reward"]
    grnwt_delta = max(0.0, scalar(latest.get("grnwt"), 0.0) - scalar(prev.get("grnwt"), 0.0))
    topwt_delta = max(0.0, scalar(latest.get("topwt"), 0.0) - scalar(prev.get("topwt"), 0.0))
    growth_reward = float(cfg["grnwt_delta_coef"]) * grnwt_delta
    cost = float(cfg["water_cost"]) * safe_i + float(cfg["nitrogen_cost"]) * safe_n

    repeat_window = float(cfg["repeat_window_days"])
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

    swfac_before = scalar(prev.get("swfac"), 0.0)
    nstres_before = scalar(prev.get("nstres"), 0.0)
    no_stress_i = 0.0
    no_stress_n = 0.0
    if dap <= int(cfg["no_stress_early_dap_threshold"]):
        if safe_i > 0 and swfac_before <= float(cfg["no_stress_swfac_threshold"]):
            no_stress_i = float(cfg["no_stress_irrigation_penalty_coef"]) * safe_i
        if safe_n > 0 and nstres_before <= float(cfg["no_stress_nstres_threshold"]):
            no_stress_n = float(cfg["no_stress_nitrogen_penalty_coef"]) * safe_n

    late_i = float(cfg["late_irrigation_penalty_coef"]) * safe_i if dap >= int(cfg["late_irrigation_dap_threshold"]) else 0.0
    late_n = float(cfg["late_nitrogen_penalty_coef"]) * safe_n if dap >= int(cfg["late_nitrogen_dap_threshold"]) else 0.0

    total_penalty = repeat_i + repeat_n + early_i + early_n + no_stress_i + no_stress_n + late_i + late_n
    reward = growth_reward - cost - total_penalty
    parts = {
        "growth_reward": growth_reward,
        "topwt_delta_unrewarded": topwt_delta,
        "grnwt_delta": grnwt_delta,
        "resource_cost": cost,
        "repeat_penalty": repeat_i + repeat_n,
        "early_excess_penalty": early_i + early_n,
        "no_stress_early_penalty": no_stress_i + no_stress_n,
        "late_penalty": late_i + late_n,
        "timing_shaping_penalty": total_penalty,
        "reward_v3": reward,
        "swfac_before": swfac_before,
        "nstres_before": nstres_before,
    }
    return float(reward), parts


class RewardV3PPOWrapper(gym.Env):
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
        safety_result = apply_action_safety(scaled_real, dap, self.safety_state, {**self.config["action_safety"], "enabled": True})
        safe_norm = normalize_action(action_names, self.env.formator.action_space_dict, safety_result.safe_real_action)
        obs, _env_reward, terminated, truncated, info = self.env.step(safe_norm)
        update_action_safety_state(self.safety_state, safety_result.safe_real_action, dap)
        latest = latest_observation_dict(self.env, obs, info)
        self.last_obs_dict = latest
        safe_i = float(safety_result.safe_real_action.get("amir", 0.0))
        safe_n = float(safety_result.safe_real_action.get("anfer", 0.0))
        reward, parts = reward_v3(self.config, prev, latest, before, safe_i, safe_n, dap)
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


def make_env(config: dict, env_config: dict, station: str, year: int, seed: int, tag: str, evaluation: bool):
    base = direct_ppo.make_base_env(env_config, station, year, seed, tag, evaluation=evaluation)
    return RewardV3PPOWrapper(base, config)


def train(config: dict, env_config: dict) -> pd.DataFrame:
    from stable_baselines3 import PPO

    station = "SYA"
    year = 2014
    seed = int(config["seed"])
    model_path = OUT / "ppo" / "models" / station / "ppo_free_timing_reward_v3_seed0.zip"
    env = None
    status = "ok"
    notes = ""
    try:
        env = make_env(config, env_config, station, year, seed, f"{station}_{year}_PPO_031_08_train", evaluation=False)
        kwargs = {k: config["ppo"][k] for k in ["learning_rate", "gamma", "n_steps", "batch_size", "ent_coef", "clip_range"]}
        model = PPO("MlpPolicy", env, verbose=1, seed=seed, tensorboard_log=str(OUT / "ppo" / "tensorboard" / station), **kwargs)
        model.learn(total_timesteps=int(config["total_timesteps"]), reset_num_timesteps=True, progress_bar=False)
        model.save(str(model_path.with_suffix("")))
    except Exception:
        status = "failed"
        notes = traceback.format_exc()
    finally:
        if env is not None:
            env.close()
    summary = pd.DataFrame(
        [
            {
                "algorithm": "PPO",
                "station_code": station,
                "train_years": "2014",
                "seed": seed,
                "total_timesteps": int(config["total_timesteps"]),
                "run_status": status,
                "model_path": str(model_path.relative_to(ROOT)) if model_path.exists() else "",
                "notes": notes[-1500:] if notes else "",
            }
        ]
    )
    summary.to_csv(OUT / "ppo" / "evaluation" / "training_run_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def evaluate(config: dict, env_config: dict, train_summary: pd.DataFrame) -> pd.DataFrame:
    from stable_baselines3 import PPO

    rows: list[dict[str, Any]] = []
    row = train_summary.iloc[0]
    if str(row["run_status"]) != "ok":
        return pd.DataFrame()
    model = PPO.load(str(ROOT / str(row["model_path"])))
    weather = direct_ppo.weather_for_daily(config)
    station = "SYA"
    year = 2014
    for split in ["train", "eval"]:
        env = make_env(config, env_config, station, year, int(config["seed"]), f"{station}_{year}_PPO_031_08_eval", evaluation=True)
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
                records.append(
                    {
                        "algorithm": "PPO",
                        "split": split,
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
                        **dict(env.last_action_info),
                        "done": done,
                        "info": json.dumps(info, ensure_ascii=False, default=str),
                    }
                )
                step_count += 1
        finally:
            env.close()
        daily = pd.DataFrame(records)
        daily_path = OUT / "ppo" / "daily_outputs" / station / f"2014_{split}_ppo_reward_v3_daily.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        rows.append(summarize_daily(daily, daily_path, split))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "ppo" / "evaluation" / "ppo_reward_v3_eval_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def summarize_daily(daily: pd.DataFrame, daily_path: Path, split: str) -> dict[str, Any]:
    irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    nfer = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    swfac = pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce")
    nstres = pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce")
    grnwt = pd.to_numeric(daily.get("grnwt", pd.Series(dtype=float)), errors="coerce")
    final_grnwt = float(grnwt.iloc[-1]) if len(grnwt) else np.nan
    total_i = float(irr.sum())
    total_n = float(nfer.sum())
    early = daily[pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce") <= 10].copy()
    early_i = float(pd.to_numeric(early.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if len(early) else 0.0
    early_n = float(pd.to_numeric(early.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if len(early) else 0.0
    return {
        "algorithm": "PPO",
        "policy_name": "ppo_reward_v3_5k",
        "split": split,
        "run_status": "ok" if len(daily) and bool(daily["done"].iloc[-1]) else "failed",
        "episode_length": int(len(daily)),
        "final_grnwt": final_grnwt,
        "total_irrigation": total_i,
        "total_n": total_n,
        "profit_simple": final_grnwt - total_i - 5.0 * total_n if np.isfinite(final_grnwt) else np.nan,
        "reward_v3_sum": float(pd.to_numeric(daily.get("reward", pd.Series(dtype=float)), errors="coerce").sum()) if len(daily) else np.nan,
        "timing_penalty_sum": float(pd.to_numeric(daily.get("timing_shaping_penalty", pd.Series(dtype=float)), errors="coerce").sum()) if len(daily) else np.nan,
        "early_dap1_10_irrigation": early_i,
        "early_dap1_10_n": early_n,
        "reached_both_caps_by_dap10": bool(early_i >= 160.0 and early_n >= 250.0),
        "irrigation_event_count": int((irr > 0).sum()),
        "n_event_count": int((nfer > 0).sum()),
        "first_irrigation_dap": int(daily.loc[irr > 0, "dap"].iloc[0]) if (irr > 0).any() else np.nan,
        "first_n_dap": int(daily.loc[nfer > 0, "dap"].iloc[0]) if (nfer > 0).any() else np.nan,
        "swfac_stress_days_gt_0p05": int((swfac > 0.05).sum()) if len(swfac) else 0,
        "nstres_days_gt_0p05": int((nstres > 0.05).sum()) if len(nstres) else 0,
        "daily_csv_path": str(daily_path.relative_to(ROOT)),
    }


def load_references() -> pd.DataFrame:
    paths = [
        ROOT / "benchmark_results" / "031_03_free_daily_original_reward_random_baseline" / "031_03_random_no_train_baselines.csv",
        ROOT / "benchmark_results" / "031_04_free_daily_original_reward_5k_train" / "031_04_ppo_dqn_5k_vs_baselines.csv",
        ROOT / "benchmark_results" / "031_05_sy2014_free_timing_rule_sensitivity" / "evaluation" / "031_05_rule_timing_sensitivity_summary.csv",
        ROOT / "benchmark_results" / "031_06_free_timing_reward_v2_ppo_dqn" / "031_06_reward_v2_vs_references.csv",
    ]
    frames = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "policy_name" not in df.columns and "rule" in df.columns:
            df["policy_name"] = df["rule"]
        if "source" not in df.columns:
            df["source"] = path.parent.name.split("_")[0]
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]) or pd.api.types.is_integer_dtype(work[col]):
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        else:
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else str(x))
    return "\n".join(
        [
            "| " + " | ".join(work.columns) + " |",
            "| " + " | ".join(["---"] * len(work.columns)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy()],
        ]
    )


def write_record(train_summary: pd.DataFrame, eval_summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    eval_row = eval_summary[eval_summary["split"].eq("eval")].iloc[0] if not eval_summary.empty else pd.Series(dtype=object)
    pass_no_early_caps = bool(not eval_row.get("reached_both_caps_by_dap10", True)) if len(eval_row) else False
    pass_min_yield = bool(float(eval_row.get("final_grnwt", 0.0)) >= 9761.0) if len(eval_row) else False
    lines = [
        "# 031_08 SY2014 free-timing reward v3 PPO smoke record",
        "",
        "## Scope",
        "",
        "- SYA2014 seed0 only.",
        "- PPO only, 5,000 timesteps.",
        "- Daily free timing; no expert DAP windows.",
        "- Reward v3 removes TOPWT reward and adds stronger early/no-stress/repeat penalties.",
        "",
        "## Training summary",
        "",
        markdown_table(train_summary),
        "",
        "## Evaluation summary",
        "",
        markdown_table(eval_summary),
        "",
        "## Smoke pass signals",
        "",
        f"- Does not reach both I160/N250 by DAP10: {pass_no_early_caps}.",
        f"- Final grain >= 031_05 stress_triggered 9761 kg/ha: {pass_min_yield}.",
        "",
        "## Comparison subset",
        "",
        markdown_table(
            comparison[
                [
                    "source",
                    "policy_name",
                    "final_grnwt",
                    "total_irrigation",
                    "total_n",
                    "profit_simple",
                    "first_irrigation_dap",
                    "first_n_dap",
                    "irrigation_event_count",
                    "n_event_count",
                    "swfac_stress_days_gt_0p05",
                    "nstres_days_gt_0p05",
                ]
            ].head(20)
        ),
        "",
        "## Interpretation boundary",
        "",
        "This is a single-seed reward smoke, not a cross-seed or cross-year claim. Do not tune coefficients in-place based on this result.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / "031_08_sy2014_free_timing_reward_v3_ppo_smoke_record.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs(OUT / "ppo")
    shutil.copyfile(CONFIG, OUT / "ppo" / "configs" / CONFIG.name)
    config = direct_ppo.load_yaml(CONFIG)
    direct_ppo.OUTPUT_ROOT = OUT / "ppo"
    selection = make_sy2014_selection()
    selection.to_csv(OUT / "ppo" / "configs" / "031_08_sy2014_selection.csv", index=False, encoding="utf-8-sig")
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "ppo" / "configs" / "031_08_resolved_env_config.yaml")
    train_summary = train(config, env_config)
    eval_summary = evaluate(config, env_config, train_summary)
    eval_rows = eval_summary[eval_summary["split"].eq("eval")].copy()
    eval_rows["source"] = "031_08"
    refs = load_references()
    comparison_cols = [
        "source",
        "policy_name",
        "run_status",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "profit_simple",
        "first_irrigation_dap",
        "first_n_dap",
        "irrigation_event_count",
        "n_event_count",
        "swfac_stress_days_gt_0p05",
        "nstres_days_gt_0p05",
        "daily_csv_path",
    ]
    comparison = pd.concat([eval_rows, refs], ignore_index=True, sort=False)
    comparison = comparison[[c for c in comparison_cols if c in comparison.columns]]
    comparison_path = OUT / "031_08_reward_v3_vs_references.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    write_record(train_summary, eval_summary, comparison)
    result = {
        "task": "031_08_sy2014_free_timing_reward_v3_ppo_smoke",
        "training_run": True,
        "site_year": "SYA2014",
        "algorithm": "PPO",
        "seed": int(config["seed"]),
        "timesteps": int(config["total_timesteps"]),
        "eval": eval_summary.to_dict(orient="records"),
        "comparison": str(comparison_path.relative_to(ROOT)),
        "record_md": str(DOC.relative_to(ROOT)),
        "scope": "single site-year reward-v3 PPO smoke only",
    }
    (OUT / "031_08_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(comparison.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
