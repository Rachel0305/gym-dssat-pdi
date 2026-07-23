from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_literature_aligned_dqn_free_timing_smoke_031_09 as lit
from ppo_action_safety import ActionSafetyState, apply_action_safety, normalize_action, update_action_safety_state


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_19_free_timing_discrete_masked_dqn_ncost2x_scaled_reward_seed0.yaml"
OUT = ROOT / "benchmark_results" / "031_19_free_timing_discrete_masked_dqn_ncost2x_scaled_reward_seed0"
DOC = ROOT / "docs" / "031_19_free_timing_discrete_masked_dqn_ncost2x_scaled_reward_seed0_record.md"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"


def ensure_dirs() -> None:
    for rel in ["configs", "models/SYA", "evaluation", "daily_outputs/SYA", "logs", "tensorboard/SYA"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def make_sy2014_selection() -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    row = pool[(pool["station_code"].eq("SYA")) & (pool["year"].astype(int).eq(2014))].copy()
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one SYA 2014 row, found {len(row)}")
    row["selected_for_train"] = True
    row["selected_for_eval"] = True
    row["selection_reason"] = "031_19_free_timing_discrete_masked_dqn_ncost2x_scaled_reward_seed0"
    return row


def latest_observation_dict(env, obs=None, info=None) -> dict[str, Any]:
    return direct_ppo.latest_observation_dict(env, obs, info)


def action_grid(config: dict) -> list[dict[str, float]]:
    i_levels = [float(x) for x in config["discrete_actions"]["irrigation_levels"]]
    n_levels = [float(x) for x in config["discrete_actions"]["nitrogen_levels"]]
    return [{"amir": i, "anfer": n} for i in i_levels for n in n_levels]


class FreeTimingDiscreteDQNWrapper(gym.Env):
    """Discrete free-timing water/N wrapper with strict action masks."""

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

    def _dap(self) -> int:
        dap_raw = scalar(self.last_obs_dict.get("dap", 1))
        return int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else 1

    def _action_is_legal_without_clipping(self, raw: dict[str, float], dap: int) -> bool:
        i = float(raw.get("amir", 0.0))
        n = float(raw.get("anfer", 0.0))
        if i == 0.0 and n == 0.0:
            return True
        cfg = self.config["action_safety"]
        remaining_i = float(cfg["season_irrigation_soft_limit"]) - float(self.safety_state.cumulative_irrigation)
        remaining_n = float(cfg["season_n_soft_limit"]) - float(self.safety_state.cumulative_n)
        i_min, i_max = [int(x) for x in cfg["irrigation_allowed_dap_range"]]
        n_min, n_max = [int(x) for x in cfg["fertilization_allowed_dap_range"]]
        if i > 0:
            if dap < i_min or dap > i_max:
                return False
            if i - remaining_i > 1e-9:
                return False
            if self.safety_state.last_irrigation_dap is not None and dap - int(self.safety_state.last_irrigation_dap) < int(cfg["min_days_between_irrigation"]):
                return False
        if n > 0:
            if dap < n_min or dap > n_max:
                return False
            if n - remaining_n > 1e-9:
                return False
            if self.safety_state.last_fertilization_dap is not None and dap - int(self.safety_state.last_fertilization_dap) < int(cfg["min_days_between_fertilization"]):
                return False
        return True

    def action_masks(self) -> np.ndarray:
        dap = self._dap()
        mask = np.asarray([self._action_is_legal_without_clipping(raw, dap) for raw in self.grid], dtype=bool)
        mask[0] = True
        if not bool(mask.any()):
            mask[0] = True
        return mask

    def step(self, action):
        prev = dict(self.last_obs_dict)
        dap = self._dap()
        requested_action_index = int(np.asarray(action).item())
        mask = self.action_masks()
        action_index = requested_action_index
        mask_forced_noop = False
        if not bool(mask[action_index]):
            action_index = 0
            mask_forced_noop = True
        raw_real = dict(self.grid[action_index])
        before = ActionSafetyState(**self.safety_state.__dict__)
        safety_result = apply_action_safety(raw_real, dap, self.safety_state, {**self.config["action_safety"], "enabled": True})
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
        unscaled_reward = final_yield_component - resource_cost
        reward_scale = float(reward_cfg.get("reward_scale", 1.0))
        reward = unscaled_reward * reward_scale
        self.last_action_info = {
            "discrete_action_index": action_index,
            "requested_discrete_action_index": requested_action_index,
            "mask_forced_noop": bool(mask_forced_noop),
            "raw_action_amir": raw_real.get("amir", np.nan),
            "raw_action_anfer": raw_real.get("anfer", np.nan),
            "safe_action_amir": safe_i,
            "safe_action_anfer": safe_n,
            "season_cumulative_irrigation": float(self.safety_state.cumulative_irrigation),
            "season_cumulative_n": float(self.safety_state.cumulative_n),
            "action_safety_triggered": safety_result.safety_rule_triggered,
            "mask_valid_action_count": int(self.action_masks().sum()),
            "literature_yield_component": final_yield_component,
            "literature_resource_cost": resource_cost,
            "literature_reward_unscaled": unscaled_reward,
            "reward_scale": reward_scale,
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
    return FreeTimingDiscreteDQNWrapper(base, config)


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


def train(config: dict, env_config: dict) -> pd.DataFrame:
    from stable_baselines3 import DQN

    station = "SYA"
    year = 2014
    seed = int(config["seed"])
    model_path = OUT / "models" / station / "free_timing_discrete_masked_dqn_ncost2x_scaled_reward_seed0.zip"
    env = None
    status = "ok"
    notes = ""
    try:
        env = make_env(config, env_config, station, year, seed, f"{station}_{year}_031_19_train", evaluation=False)
        model = DQN(
            "MlpPolicy",
            env,
            verbose=0,
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
            env.close()
    summary = pd.DataFrame(
        [
            {
                "algorithm": "free_timing_discrete_masked_eval_DQN",
                "station_code": station,
                "train_years": str(year),
                "seed": seed,
                "total_timesteps": int(config["total_timesteps"]),
                "run_status": status,
                "model_path": str(model_path.relative_to(ROOT)) if model_path.exists() else "",
                "notes": notes[-2500:] if notes else "",
            }
        ]
    )
    summary.to_csv(OUT / "evaluation" / "training_run_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def summarize_daily(daily: pd.DataFrame, daily_path: Path, split: str, model_path: Path) -> dict[str, Any]:
    dap = pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce")
    irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0)
    n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0)
    swfac = pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce")
    nstres = pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce")
    final_y = float(pd.to_numeric(daily.get("grnwt", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if len(daily) else np.nan
    total_i = float(irr.sum())
    total_n = float(n.sum())
    nonzero = daily[(irr > 0) | (n > 0)]
    action_sequence = "; ".join(
        f"DAP{int(r.dap)} I{float(r.safe_action_amir):g}/N{float(r.safe_action_anfer):g}"
        for r in nonzero.itertuples(index=False)
    )
    return {
        "algorithm": "free_timing_discrete_masked_eval_DQN",
        "policy_name": "free_discrete_masked_dqn_ncost2x_scaled_reward_20k_seed0",
        "split": split,
        "run_status": "ok" if len(daily) and bool(daily["done"].iloc[-1]) else "failed",
        "episode_length": int(len(daily)),
        "final_grnwt": final_y,
        "total_irrigation": total_i,
        "total_n": total_n,
        "profit_simple": final_y - total_i - 5.0 * total_n if np.isfinite(final_y) else np.nan,
        "PFP_N": final_y / total_n if total_n > 0 and np.isfinite(final_y) else np.nan,
        "literature_reward_sum": float(pd.to_numeric(daily.get("literature_reward", pd.Series(dtype=float)), errors="coerce").sum()) if len(daily) else np.nan,
        "early_dap1_10_irrigation": float(irr[dap <= 10].sum()) if len(daily) else np.nan,
        "early_dap1_10_n": float(n[dap <= 10].sum()) if len(daily) else np.nan,
        "reached_both_caps_by_dap10": bool(float(irr[dap <= 10].sum()) >= 159.999 and float(n[dap <= 10].sum()) >= 249.999) if len(daily) else False,
        "mask_forced_noop_count": int(pd.to_numeric(daily.get("mask_forced_noop", pd.Series(dtype=float)), errors="coerce").fillna(False).astype(bool).sum()) if len(daily) else 0,
        "irrigation_event_count": int((irr > 0).sum()),
        "n_event_count": int((n > 0).sum()),
        "first_irrigation_dap": int(daily.loc[irr > 0, "dap"].iloc[0]) if (irr > 0).any() else np.nan,
        "first_n_dap": int(daily.loc[n > 0, "dap"].iloc[0]) if (n > 0).any() else np.nan,
        "swfac_stress_days_gt_0p05": int((swfac > 0.05).sum()) if len(swfac) else 0,
        "nstres_days_gt_0p05": int((nstres > 0.05).sum()) if len(nstres) else 0,
        "action_sequence": action_sequence,
        "model_path": str(model_path.relative_to(ROOT)),
        "daily_csv_path": str(daily_path.relative_to(ROOT)),
    }


def evaluate(config: dict, env_config: dict, train_summary: pd.DataFrame) -> pd.DataFrame:
    from stable_baselines3 import DQN

    rows: list[dict[str, Any]] = []
    row = train_summary.iloc[0]
    if str(row["run_status"]) != "ok":
        return pd.DataFrame([{"run_status": "failed", "notes": row.get("notes", "")}])
    model_path = ROOT / str(row["model_path"])
    model = DQN.load(str(model_path), device="cpu")
    weather = direct_ppo.weather_for_daily(config)
    station = "SYA"
    year = 2014
    seed = int(config["seed"])
    for split in ["train", "eval"]:
        env = make_env(config, env_config, station, year, seed, f"{station}_{year}_031_19_{split}", evaluation=True)
        records: list[dict[str, Any]] = []
        try:
            obs, info = env.reset()
            done = False
            step_count = 0
            planting = pd.Timestamp(direct_ppo.find_year(env_config, station, year)["planting_date"])
            while not done and step_count < int(config["runtime"]["max_steps"]):
                latest = latest_observation_dict(env, obs, info)
                dap_raw = scalar(latest.get("dap", step_count + 1))
                dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
                mask = env.action_masks()
                action = masked_greedy_action(model, obs, mask)
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                latest = latest_observation_dict(env, obs, info)
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
        daily_path = OUT / "daily_outputs" / station / f"2014_{split}_free_discrete_masked_dqn_ncost2x_scaled_reward_daily.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        row_out = summarize_daily(daily, daily_path, split, model_path)
        row_out["station_code"] = station
        row_out["year"] = year
        row_out["seed"] = seed
        rows.append(row_out)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "evaluation" / "eval_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def masked_greedy_action(model, obs, mask: np.ndarray) -> int:
    with torch.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        q_values = model.q_net(obs_tensor).detach().cpu().numpy()
    q = np.asarray(q_values).reshape(-1)
    legal = np.asarray(mask, dtype=bool)
    if q.shape[0] != legal.shape[0]:
        raise RuntimeError(f"Q shape {q.shape} does not match mask shape {legal.shape}")
    masked_q = q.copy()
    masked_q[~legal] = -np.inf
    if not np.isfinite(masked_q).any():
        return 0
    return int(np.argmax(masked_q))


def load_reference_context() -> pd.DataFrame:
    paths = [
        ROOT / "benchmark_results" / "031_15_literature_dqn_interval7_ncost2x_100k_seed0" / "evaluation" / "eval_summary.csv",
        ROOT / "benchmark_results" / "031_13_literature_dqn_interval7_ncost2x" / "evaluation" / "eval_summary.csv",
        ROOT / "benchmark_results" / "031_12_dqn_nitrogen_counterfactual_audit" / "evaluation" / "031_12_nitrogen_counterfactual_summary.csv",
        ROOT / "benchmark_results" / "031_08_sy2014_free_timing_reward_v3_ppo_smoke" / "031_08_reward_v3_vs_references.csv",
    ]
    frames = []
    for path in paths:
        if path.exists():
            df = pd.read_csv(path)
            df["source_file"] = str(path.relative_to(ROOT))
            frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def write_record(train_summary: pd.DataFrame, eval_summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    lines = [
        "# 031_19 Free-timing discrete masked DQN ncost2x scaled-reward seed0 record",
        "",
        "## Scope",
        "",
        "- SYA2014 seed0 only.",
        "- Daily free timing; no expert DAP windows.",
        "- Algorithm changed from DQN/continuous PPO to discrete DQN.",
        "- Reward/action grid/caps match 031_13/031_15 except for a pure numerical reward scale of 0.001.",
        "- The scale changes optimization numerics only; it does not change the ordering of season returns.",
        "- Training timesteps: 20,000.",
        "",
        "## Training summary",
        "",
        train_summary.to_string(index=False),
        "",
        "## Evaluation summary",
        "",
        eval_summary.to_string(index=False) if not eval_summary.empty else "No evaluation rows.",
        "",
        "## Comparison context",
        "",
        comparison.to_string(index=False) if not comparison.empty else "No comparison rows.",
        "",
        "## Interpretation boundary",
        "",
        "This is a single-seed training-stability smoke. It tests whether PPO value/policy learning is sensitive to reward magnitude under the same free-timing discrete action setup as 031_16.",
        "",
        "Branch rules:",
        "",
        "- If scaled reward materially improves timing/yield while preserving N economy, the next step is a cross-seed repeat of the same fixed configuration.",
        "- If scaled reward only changes reward/loss magnitudes but still front-loads resources or underperforms the known staged reference, the bottleneck is not only numerical reward scale.",
        "- This task must not be expanded to all sites/years from a single seed.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    config = direct_ppo.load_yaml(CONFIG)
    selection = make_sy2014_selection()
    selection.to_csv(OUT / "configs" / "031_19_sy2014_selection.csv", index=False, encoding="utf-8-sig")
    direct_ppo.OUTPUT_ROOT = OUT
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "031_19_resolved_env_config.yaml")

    train_summary = train(config, env_config)
    eval_summary = evaluate(config, env_config, train_summary)
    comparison = eval_summary.copy()
    comparison["source_file"] = "031_19_current"
    refs = load_reference_context()
    if not refs.empty:
        comparison = pd.concat([comparison, refs], ignore_index=True, sort=False)
    comparison_path = OUT / "evaluation" / "031_19_with_reference_context.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    write_record(train_summary, eval_summary, comparison)

    result = {
        "task": "031_19_free_timing_discrete_masked_dqn_ncost2x_scaled_reward_seed0",
        "training_run": True,
        "site_year": "SYA2014",
        "seed": int(config["seed"]),
        "record_md": str(DOC.relative_to(ROOT)),
        "eval_summary": str((OUT / "evaluation" / "eval_summary.csv").relative_to(ROOT)),
        "comparison": str(comparison_path.relative_to(ROOT)),
    }
    (OUT / "031_19_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(eval_summary.to_string(index=False))


if __name__ == "__main__":
    main()



