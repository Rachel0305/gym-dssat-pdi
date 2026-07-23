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
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_02_free_daily_original_reward_dqn_smoke.yaml"
OUT = ROOT / "benchmark_results" / "031_02_free_daily_original_reward_dqn_smoke"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"


def ensure_dirs() -> None:
    for rel in [
        "configs",
        "models/SYA",
        "evaluation",
        "daily_outputs/SYA",
        "logs",
        "tensorboard/SYA",
        "rendered_inputs",
        "reports",
    ]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)


def make_sy2014_selection() -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    row = pool[(pool["station_code"].eq("SYA")) & (pool["year"].astype(int).eq(2014))].copy()
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one SYA 2014 row, found {len(row)}")
    row["selected_for_train"] = True
    row["selected_for_eval"] = True
    row["selection_reason"] = "031_02_single_smoke_SY2014_free_daily_original_reward_dqn"
    return row


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def latest_observation_dict(env, obs=None, info=None) -> dict[str, Any]:
    return direct_ppo.latest_observation_dict(env, obs, info)


def action_grid(config: dict) -> list[dict[str, float]]:
    i_levels = [float(x) for x in config["discrete_actions"]["irrigation_levels"]]
    n_levels = [float(x) for x in config["discrete_actions"]["nitrogen_levels"]]
    return [{"amir": i, "anfer": n} for i in i_levels for n in n_levels]


class DiscreteActionSafeGrowthRewardWrapper(gym.Env):
    """DQN-compatible discrete action wrapper for free daily DSSAT control."""

    def __init__(self, env, config: dict):
        super().__init__()
        self.env = env
        self.config = config
        self.grid = action_grid(config)
        self.action_space = spaces.Discrete(len(self.grid))
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
        dap_raw = scalar(prev.get("dap", 0))
        dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else 1
        action_index = int(np.asarray(action).item())
        raw_real = dict(self.grid[action_index])
        safety = {**self.config["action_safety"], "enabled": True}
        safety_result = apply_action_safety(raw_real, dap, self.safety_state, safety)
        action_names = list(self.env.formator.action_names)
        safe_norm = normalize_action(action_names, self.env.formator.action_space_dict, safety_result.safe_real_action)
        obs, _env_reward, terminated, truncated, info = self.env.step(safe_norm)
        update_action_safety_state(self.safety_state, safety_result.safe_real_action, dap)
        latest = latest_observation_dict(self.env, obs, info)
        self.last_obs_dict = latest

        topwt_delta = max(0.0, scalar(latest.get("topwt"), 0.0) - scalar(prev.get("topwt"), 0.0))
        grnwt_delta = max(0.0, scalar(latest.get("grnwt"), 0.0) - scalar(prev.get("grnwt"), 0.0))
        reward_cfg = self.config["reward"]
        safe_i = float(safety_result.safe_real_action.get("amir", 0.0))
        safe_n = float(safety_result.safe_real_action.get("anfer", 0.0))
        growth_reward = float(reward_cfg["topwt_delta_coef"]) * topwt_delta + float(reward_cfg["grnwt_delta_coef"]) * grnwt_delta
        reward = growth_reward - float(reward_cfg["water_cost"]) * safe_i - float(reward_cfg["nitrogen_cost"]) * safe_n

        self.last_action_info = {
            "discrete_action_index": action_index,
            "raw_action_amir": raw_real.get("amir", np.nan),
            "raw_action_anfer": raw_real.get("anfer", np.nan),
            "scaled_action_amir": raw_real.get("amir", np.nan),
            "scaled_action_anfer": raw_real.get("anfer", np.nan),
            "safe_action_amir": safe_i,
            "safe_action_anfer": safe_n,
            "season_cumulative_irrigation": float(self.safety_state.cumulative_irrigation),
            "season_cumulative_n": float(self.safety_state.cumulative_n),
            "action_safety_triggered": safety_result.safety_rule_triggered,
            "growth_reward": growth_reward,
            "topwt_delta": topwt_delta,
            "grnwt_delta": grnwt_delta,
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_training_env(config: dict, env_config: dict, station: str, year: int, seed: int, run_tag: str, evaluation: bool = False):
    base = direct_ppo.make_base_env(env_config, station, year, seed, run_tag, evaluation=evaluation)
    return DiscreteActionSafeGrowthRewardWrapper(base, config)


def dqn_kwargs(config: dict) -> dict[str, Any]:
    keys = [
        "learning_rate",
        "buffer_size",
        "learning_starts",
        "batch_size",
        "gamma",
        "train_freq",
        "gradient_steps",
        "target_update_interval",
        "exploration_fraction",
        "exploration_initial_eps",
        "exploration_final_eps",
    ]
    return {k: config["dqn"][k] for k in keys}


def train_model(config: dict, env_config: dict, selection: pd.DataFrame) -> pd.DataFrame:
    from stable_baselines3 import DQN

    seed = int(config["seed"])
    station = "SYA"
    train_years = selection[selection["selected_for_train"]]["year"].astype(int).tolist()
    model_path = OUT / "models" / station / "dqn_free_daily_original_reward_seed0.zip"
    tensorboard_dir = OUT / "tensorboard" / station
    env = None
    status = "ok"
    notes = ""
    try:
        env = make_training_env(config, env_config, station, int(train_years[0]), seed, f"{station}_{train_years[0]}_dqn_train", evaluation=False)
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            tensorboard_log=str(tensorboard_dir),
            **dqn_kwargs(config),
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
    summary = pd.DataFrame(
        [
            {
                "station_code": station,
                "train_years": ",".join(map(str, train_years)),
                "seed": seed,
                "total_timesteps": int(config["total_timesteps"]),
                "run_status": status,
                "model_path": str(model_path.relative_to(ROOT)) if model_path.exists() else "",
                "notes": notes[-1500:] if notes else "",
            }
        ]
    )
    summary.to_csv(OUT / "evaluation" / "training_run_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def evaluate_one(config: dict, env_config: dict, model, station: str, year: int, split: str, model_path: Path, weather: pd.DataFrame) -> dict:
    seed = int(config["seed"])
    env = make_training_env(config, env_config, station, year, seed, f"{station}_{year}_dqn_eval", evaluation=True)
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        year_info = direct_ppo.find_year(env_config, station, year)
        planting = pd.Timestamp(year_info["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            action_info = dict(env.last_action_info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(station)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "station_code": station,
                    "year": int(year),
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
    daily_dir = OUT / "daily_outputs" / station
    daily_dir.mkdir(parents=True, exist_ok=True)
    daily_path = daily_dir / f"{year}_{split}_dqn_daily.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    swfac = pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce")
    nstres = pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce")
    irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0)
    n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0)
    final_grnwt = float(pd.to_numeric(daily.get("grnwt", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if len(daily) else np.nan
    final_topwt = float(pd.to_numeric(daily.get("topwt", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if len(daily) else np.nan
    total_i = float(irr.sum())
    total_n = float(n.sum())
    econ = config["economics"]
    profit_simple = float(econ["grain_value_coef"]) * final_grnwt - float(econ["water_cost"]) * total_i - float(econ["nitrogen_cost"]) * total_n if np.isfinite(final_grnwt) else np.nan
    return {
        "station_code": station,
        "station_name": direct_ppo.STATION_NAMES.get(station, station),
        "year": int(year),
        "split": split,
        "scenario_type": str(direct_ppo.find_year(env_config, station, year).get("label", "")),
        "run_status": "ok" if len(daily) and bool(daily["done"].iloc[-1]) else "failed",
        "episode_completed": bool(len(daily) and bool(daily["done"].iloc[-1])),
        "episode_length": int(len(daily)),
        "final_grnwt": final_grnwt,
        "final_topwt": final_topwt,
        "max_xlai": float(pd.to_numeric(daily.get("xlai", pd.Series(dtype=float)), errors="coerce").max()) if len(daily) else np.nan,
        "total_irrigation": total_i,
        "total_n": total_n,
        "profit_simple": profit_simple,
        "mean_swfac": float(swfac.mean()) if len(swfac) else np.nan,
        "max_swfac": float(swfac.max()) if len(swfac) else np.nan,
        "swfac_stress_days_gt_0p05": int((swfac > 0.05).sum()) if len(swfac) else 0,
        "mean_nstres": float(nstres.mean()) if len(nstres) else np.nan,
        "max_nstres": float(nstres.max()) if len(nstres) else np.nan,
        "nstres_days_gt_0p05": int((nstres > 0.05).sum()) if len(nstres) else 0,
        "irrigation_event_count": int((irr > 0).sum()),
        "n_event_count": int((n > 0).sum()),
        "first_irrigation_dap": int(daily.loc[irr > 0, "dap"].iloc[0]) if (irr > 0).any() else np.nan,
        "first_n_dap": int(daily.loc[n > 0, "dap"].iloc[0]) if (n > 0).any() else np.nan,
        "model_path": str(model_path.relative_to(ROOT)),
        "daily_csv_path": str(daily_path.relative_to(ROOT)),
        "notes": "",
    }


def evaluate_model(config: dict, env_config: dict, selection: pd.DataFrame, train_summary: pd.DataFrame) -> pd.DataFrame:
    from stable_baselines3 import DQN

    weather = direct_ppo.weather_for_daily(config)
    rows = []
    row = train_summary.iloc[0]
    if str(row["run_status"]) != "ok":
        summary = pd.DataFrame(rows)
        summary.to_csv(OUT / "evaluation" / "dqn_free_daily_eval_summary.csv", index=False, encoding="utf-8-sig")
        return summary
    station = str(row["station_code"])
    model_path = ROOT / str(row["model_path"])
    model = DQN.load(str(model_path))
    for split_name in ["train", "eval"]:
        col = "selected_for_train" if split_name == "train" else "selected_for_eval"
        for year in selection[selection[col]]["year"].astype(int).tolist():
            try:
                rows.append(evaluate_one(config, env_config, model, station, int(year), split_name, model_path, weather))
            except Exception:
                rows.append({"station_code": station, "year": int(year), "split": split_name, "run_status": "failed", "notes": traceback.format_exc()[-1500:]})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "evaluation" / "dqn_free_daily_eval_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def build_diagnosis(config: dict, eval_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    i_cap = float(config["action_safety"]["season_irrigation_soft_limit"])
    n_cap = float(config["action_safety"]["season_n_soft_limit"])
    for row in eval_summary.itertuples(index=False):
        if str(row.run_status) != "ok":
            continue
        daily_path = ROOT / str(row.daily_csv_path)
        daily = pd.read_csv(daily_path) if daily_path.exists() else pd.DataFrame()
        irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0)
        n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0)
        cap_sat = float(row.total_irrigation) >= 0.95 * i_cap or float(row.total_n) >= 0.95 * n_cap
        early_ops = int(((pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce") <= 10) & ((irr > 0) | (n > 0))).sum()) if len(daily) else 0
        label = "cap_saturated" if cap_sat else ("early_dense_operations" if early_ops >= 5 else "not_cap_saturated")
        rows.append(
            {
                "station_code": row.station_code,
                "year": int(row.year),
                "split": row.split,
                "total_irrigation": row.total_irrigation,
                "total_n": row.total_n,
                "irrigation_event_count": row.irrigation_event_count,
                "n_event_count": row.n_event_count,
                "early_operation_days_le_dap10": early_ops,
                "first_irrigation_dap": row.first_irrigation_dap,
                "first_n_dap": row.first_n_dap,
                "final_grnwt": row.final_grnwt,
                "profit_simple": row.profit_simple,
                "decision_reasonableness_label": label,
            }
        )
    diagnosis = pd.DataFrame(rows)
    diagnosis.to_csv(OUT / "evaluation" / "dqn_decision_reasonableness_diagnosis.csv", index=False, encoding="utf-8-sig")
    return diagnosis


def main() -> None:
    ensure_dirs()
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    config = direct_ppo.load_yaml(CONFIG)
    selection = make_sy2014_selection()
    selection_path = OUT / "configs" / "031_02_sy2014_smoke_selection.csv"
    selection.to_csv(selection_path, index=False, encoding="utf-8-sig")

    # Reuse 031 direct-PPO environment resolver, but route all helper side-effects to this task folder.
    direct_ppo.OUTPUT_ROOT = OUT
    direct_ppo.ensure_dirs()
    env_config = direct_ppo.build_env_config(config, selection)
    env_config_path = OUT / "configs" / "031_02_resolved_env_config.yaml"
    direct_ppo.write_yaml(env_config, env_config_path)

    train_summary = train_model(config, env_config, selection)
    eval_summary = evaluate_model(config, env_config, selection, train_summary)
    diagnosis = build_diagnosis(config, eval_summary) if not eval_summary.empty else pd.DataFrame()

    result = {
        "task": "031_02_free_daily_original_reward_dqn_smoke",
        "training_or_dssat_run": True,
        "config": str(CONFIG.relative_to(ROOT)),
        "selection": str(selection_path.relative_to(ROOT)),
        "env_config": str(env_config_path.relative_to(ROOT)),
        "train_summary": str((OUT / "evaluation" / "training_run_summary.csv").relative_to(ROOT)),
        "eval_summary": str((OUT / "evaluation" / "dqn_free_daily_eval_summary.csv").relative_to(ROOT)),
        "diagnosis": str((OUT / "evaluation" / "dqn_decision_reasonableness_diagnosis.csv").relative_to(ROOT)),
        "train_status": train_summary.to_dict(orient="records"),
        "eval_status": eval_summary.to_dict(orient="records"),
        "diagnosis_status": diagnosis.to_dict(orient="records"),
        "free_daily": True,
        "min_interval_days": 1,
        "algorithm": "DQN",
        "discrete_action_grid": action_grid(config),
        "reward_formula": "delta_GRNWT - 1.0*irrigation - 5.0*nitrogen",
    }
    (OUT / "031_02_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    lines = [
        "# 031_02 Free-daily original-reward DQN smoke record",
        "",
        "## Run",
        "",
        "- Smoke case: SYA2014 seed0.",
        "- Algorithm: DQN.",
        "- Free daily decision: yes.",
        "- Expert DAP windows: no.",
        "- Minimum operation interval: 1 day.",
        "- Reward: `delta_GRNWT - 1.0 * irrigation - 5.0 * nitrogen`.",
        "- Training timesteps: 512.",
        "- DQN action grid: irrigation `{0,20,40}` mm × nitrogen `{0,40,80}` kg/ha.",
        "",
        "## Training summary",
        "",
        train_summary.to_string(index=False),
        "",
        "## Evaluation summary",
        "",
        eval_summary.to_string(index=False) if not eval_summary.empty else "No evaluation rows.",
        "",
        "## Decision diagnosis",
        "",
        diagnosis.to_string(index=False) if not diagnosis.empty else "No diagnosis rows.",
        "",
        "## Outputs",
        "",
        f"- Train summary: `{result['train_summary']}`",
        f"- Eval summary: `{result['eval_summary']}`",
        f"- Daily outputs: `benchmark_results/031_02_free_daily_original_reward_dqn_smoke/daily_outputs/`",
        "",
        "## Boundary note",
        "",
        "DQN requires a discrete action space, so this is a DQN feasibility smoke under the same caps/reward, not a perfectly continuous-action algorithm-only swap.",
    ]
    (OUT / "031_02_free_daily_original_reward_dqn_smoke_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
