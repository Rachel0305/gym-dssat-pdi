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
import run_free_daily_original_reward_dqn_smoke_031_02 as dqn_base
from ppo_action_safety import ActionSafetyState, apply_action_safety, normalize_action, update_action_safety_state


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_09_literature_aligned_dqn_free_timing_smoke.yaml"
OUT = ROOT / "benchmark_results" / "031_09_literature_aligned_dqn_free_timing_smoke"
DOC = ROOT / "docs" / "031_09_literature_aligned_dqn_free_timing_smoke_record.md"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"


def ensure_dirs() -> None:
    for rel in ["configs", "models/SYA", "evaluation", "daily_outputs/SYA", "logs", "tensorboard/SYA"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def make_sy2014_selection() -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    row = pool[(pool["station_code"].eq("SYA")) & (pool["year"].astype(int).eq(2014))].copy()
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one SYA 2014 row, found {len(row)}")
    row["selected_for_train"] = True
    row["selected_for_eval"] = True
    row["selection_reason"] = "031_09_literature_aligned_dqn_free_timing_smoke"
    return row


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def latest_observation_dict(env, obs=None, info=None) -> dict[str, Any]:
    return direct_ppo.latest_observation_dict(env, obs, info)


def action_grid(config: dict) -> list[dict[str, float]]:
    i_levels = [float(x) for x in config["discrete_actions"]["irrigation_levels"]]
    n_levels = [float(x) for x in config["discrete_actions"]["nitrogen_levels"]]
    return [{"amir": i, "anfer": n} for i in i_levels for n in n_levels]


class LiteratureAlignedDQNWrapper(gym.Env):
    """DQN-compatible discrete action wrapper with paper-style harvest reward."""

    def __init__(self, env, config: dict):
        super().__init__()
        from gymnasium import spaces

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
        before = ActionSafetyState(**self.safety_state.__dict__)
        safety_result = apply_action_safety(raw_real, dap, self.safety_state, safety)
        action_names = list(self.env.formator.action_names)
        safe_norm = normalize_action(action_names, self.env.formator.action_space_dict, safety_result.safe_real_action)
        obs, _env_reward, terminated, truncated, info = self.env.step(safe_norm)
        update_action_safety_state(self.safety_state, safety_result.safe_real_action, dap)
        latest = latest_observation_dict(self.env, obs, info)
        self.last_obs_dict = latest

        safe_i = float(safety_result.safe_real_action.get("amir", 0.0))
        safe_n = float(safety_result.safe_real_action.get("anfer", 0.0))
        reward_cfg = self.config["reward"]
        final_yield_component = 0.0
        if bool(terminated or truncated):
            final_yield_component = float(reward_cfg["yield_coef"]) * scalar(latest.get("grnwt"), scalar(prev.get("grnwt"), 0.0))
        resource_cost = float(reward_cfg["nitrogen_cost"]) * safe_n + float(reward_cfg["water_cost"]) * safe_i
        reward = final_yield_component - resource_cost

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
            "literature_yield_component": final_yield_component,
            "literature_resource_cost": resource_cost,
            "literature_reward": reward,
            "previous_cumulative_irrigation": float(before.cumulative_irrigation),
            "previous_cumulative_n": float(before.cumulative_n),
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_env(config: dict, env_config: dict, station: str, year: int, seed: int, run_tag: str, evaluation: bool = False):
    base = direct_ppo.make_base_env(env_config, station, year, seed, run_tag, evaluation=evaluation)
    return LiteratureAlignedDQNWrapper(base, config)


def dqn_kwargs(config: dict) -> dict[str, Any]:
    cfg = config["dqn"]
    return {
        "learning_rate": float(cfg["learning_rate"]),
        "buffer_size": int(cfg["buffer_size"]),
        "learning_starts": int(cfg["learning_starts"]),
        "batch_size": int(cfg["batch_size"]),
        "gamma": float(cfg["gamma"]),
        "train_freq": int(cfg["train_freq"]),
        "gradient_steps": int(cfg["gradient_steps"]),
        "target_update_interval": int(cfg["target_update_interval"]),
        "exploration_fraction": float(cfg["exploration_fraction"]),
        "exploration_initial_eps": float(cfg["exploration_initial_eps"]),
        "exploration_final_eps": float(cfg["exploration_final_eps"]),
        "max_grad_norm": float(cfg["max_grad_norm"]),
        "policy_kwargs": {
            "net_arch": [int(x) for x in cfg["net_arch"]],
            "optimizer_kwargs": {"weight_decay": float(cfg["weight_decay"])},
        },
    }


def train_model(config: dict, env_config: dict, selection: pd.DataFrame) -> pd.DataFrame:
    from stable_baselines3 import DQN

    seed = int(config["seed"])
    station = "SYA"
    year = int(selection[selection["selected_for_train"]]["year"].astype(int).iloc[0])
    model_path = OUT / "models" / station / "literature_aligned_dqn_seed0.zip"
    status = "ok"
    notes = ""
    env = None
    try:
        env = make_env(config, env_config, station, year, seed, f"{station}_{year}_031_09_lit_dqn_train", evaluation=False)
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            tensorboard_log=str(OUT / "tensorboard" / station),
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
                "train_years": str(year),
                "seed": seed,
                "total_timesteps": int(config["total_timesteps"]),
                "run_status": status,
                "model_path": str(model_path.relative_to(ROOT)) if model_path.exists() else "",
                "notes": notes[-2000:] if notes else "",
            }
        ]
    )
    summary.to_csv(OUT / "evaluation" / "training_run_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def evaluate_one(config: dict, env_config: dict, model, station: str, year: int, split: str, model_path: Path, weather: pd.DataFrame) -> dict:
    seed = int(config["seed"])
    env = make_env(config, env_config, station, year, seed, f"{station}_{year}_031_09_lit_dqn_eval", evaluation=True)
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
    daily_path = daily_dir / f"{year}_{split}_literature_aligned_dqn_daily.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    swfac = pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce")
    nstres = pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce")
    irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0)
    n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0)
    final_grnwt = float(pd.to_numeric(daily.get("grnwt", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if len(daily) else np.nan
    total_i = float(irr.sum())
    total_n = float(n.sum())
    profit_simple = final_grnwt - total_i - 5.0 * total_n if np.isfinite(final_grnwt) else np.nan
    early_i = float(irr[pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce") <= 10].sum()) if len(daily) else np.nan
    early_n = float(n[pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce") <= 10].sum()) if len(daily) else np.nan
    return {
        "algorithm": "literature_aligned_DQN",
        "policy_name": "lit_dqn_5k",
        "station_code": station,
        "year": int(year),
        "split": split,
        "run_status": "ok" if len(daily) and bool(daily["done"].iloc[-1]) else "failed",
        "episode_length": int(len(daily)),
        "final_grnwt": final_grnwt,
        "total_irrigation": total_i,
        "total_n": total_n,
        "profit_simple": profit_simple,
        "literature_reward_sum": float(pd.to_numeric(daily.get("literature_reward", pd.Series(dtype=float)), errors="coerce").sum()) if len(daily) else np.nan,
        "early_dap1_10_irrigation": early_i,
        "early_dap1_10_n": early_n,
        "reached_both_caps_by_dap10": bool(early_i >= 159.999 and early_n >= 249.999),
        "irrigation_event_count": int((irr > 0).sum()),
        "n_event_count": int((n > 0).sum()),
        "first_irrigation_dap": int(daily.loc[irr > 0, "dap"].iloc[0]) if (irr > 0).any() else np.nan,
        "first_n_dap": int(daily.loc[n > 0, "dap"].iloc[0]) if (n > 0).any() else np.nan,
        "swfac_stress_days_gt_0p05": int((swfac > 0.05).sum()) if len(swfac) else 0,
        "nstres_days_gt_0p05": int((nstres > 0.05).sum()) if len(nstres) else 0,
        "model_path": str(model_path.relative_to(ROOT)),
        "daily_csv_path": str(daily_path.relative_to(ROOT)),
    }


def evaluate_model(config: dict, env_config: dict, selection: pd.DataFrame, train_summary: pd.DataFrame) -> pd.DataFrame:
    from stable_baselines3 import DQN

    rows = []
    row = train_summary.iloc[0]
    if str(row["run_status"]) != "ok":
        summary = pd.DataFrame(rows)
        summary.to_csv(OUT / "evaluation" / "literature_aligned_dqn_eval_summary.csv", index=False, encoding="utf-8-sig")
        return summary
    model_path = ROOT / str(row["model_path"])
    model = DQN.load(str(model_path))
    weather = direct_ppo.weather_for_daily(config)
    station = str(row["station_code"])
    for split_name in ["train", "eval"]:
        col = "selected_for_train" if split_name == "train" else "selected_for_eval"
        for year in selection[selection[col]]["year"].astype(int).tolist():
            rows.append(evaluate_one(config, env_config, model, station, int(year), split_name, model_path, weather))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "evaluation" / "literature_aligned_dqn_eval_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def load_reference_rows() -> pd.DataFrame:
    paths = [
        ROOT / "benchmark_results" / "031_08_sy2014_free_timing_reward_v3_ppo_smoke" / "031_08_reward_v3_vs_references.csv",
        ROOT / "benchmark_results" / "031_06_free_timing_reward_v2_ppo_dqn" / "031_06_reward_v2_vs_references.csv",
        ROOT / "benchmark_results" / "031_04_free_daily_original_reward_5k_train" / "031_04_ppo_dqn_5k_vs_baselines.csv",
    ]
    frames = []
    for path in paths:
        if path.exists():
            df = pd.read_csv(path)
            df["reference_file"] = str(path.relative_to(ROOT))
            frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def write_record(train_summary: pd.DataFrame, eval_summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    lines = [
        "# 031_09 Literature-aligned DQN free-timing smoke record",
        "",
        "## Scope",
        "",
        "- SYA2014 seed0 only.",
        "- Algorithm: SB3 DQN with literature-aligned architecture/hyperparameters.",
        "- This is not an exact reproduction of the paper author's private run/script.",
        "- Free daily timing; no expert DAP windows.",
        "- Minimum operation interval: 1 day.",
        "- Training timesteps: 5,000.",
        "- Action grid: irrigation `{0,6,12,18,24}` mm x nitrogen `{0,40,80,120,160}` kg/ha.",
        "- Reward: terminal `0.158*Y - 0.79*N - 1.1*W`; non-terminal `-0.79*N - 1.1*W`.",
        "",
        "## Training summary",
        "",
        train_summary.to_string(index=False),
        "",
        "## Evaluation summary",
        "",
        eval_summary.to_string(index=False) if not eval_summary.empty else "No evaluation rows.",
        "",
        "## Comparison rows",
        "",
        comparison.to_string(index=False) if not comparison.empty else "No comparison rows.",
        "",
        "## Interpretation boundary",
        "",
        "This smoke can only decide whether the literature-aligned DQN setting is promising enough for longer/free-timing work. It does not establish cross-seed, cross-year, or exact-paper reproducibility.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    config = direct_ppo.load_yaml(CONFIG)
    selection = make_sy2014_selection()
    selection_path = OUT / "configs" / "031_09_sy2014_selection.csv"
    selection.to_csv(selection_path, index=False, encoding="utf-8-sig")

    direct_ppo.OUTPUT_ROOT = OUT
    direct_ppo.ensure_dirs()
    env_config = direct_ppo.build_env_config(config, selection)
    env_config_path = OUT / "configs" / "031_09_resolved_env_config.yaml"
    direct_ppo.write_yaml(env_config, env_config_path)

    train_summary = train_model(config, env_config, selection)
    eval_summary = evaluate_model(config, env_config, selection, train_summary)

    refs = load_reference_rows()
    lit_eval = eval_summary[eval_summary["split"].eq("eval")].copy() if not eval_summary.empty else pd.DataFrame()
    lit_eval["source"] = "031_09"
    comparison = pd.concat([lit_eval, refs], ignore_index=True, sort=False) if not refs.empty else lit_eval
    comparison_path = OUT / "031_09_literature_aligned_dqn_vs_references.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    result = {
        "task": "031_09_literature_aligned_dqn_free_timing_smoke",
        "training_run": True,
        "site_year": "SYA2014",
        "seed": int(config["seed"]),
        "timesteps": int(config["total_timesteps"]),
        "config": str(CONFIG.relative_to(ROOT)),
        "selection": str(selection_path.relative_to(ROOT)),
        "env_config": str(env_config_path.relative_to(ROOT)),
        "train_summary": str((OUT / "evaluation" / "training_run_summary.csv").relative_to(ROOT)),
        "eval_summary": str((OUT / "evaluation" / "literature_aligned_dqn_eval_summary.csv").relative_to(ROOT)),
        "comparison": str(comparison_path.relative_to(ROOT)),
        "record_md": str(DOC.relative_to(ROOT)),
        "scope": "literature-aligned reconstruction smoke; not exact paper reproduction",
    }
    (OUT / "031_09_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_record(train_summary, eval_summary, comparison)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not lit_eval.empty:
        print(lit_eval.to_string(index=False))


if __name__ == "__main__":
    main()

