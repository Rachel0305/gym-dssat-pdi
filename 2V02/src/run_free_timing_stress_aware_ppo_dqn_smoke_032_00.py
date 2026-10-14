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
import run_free_timing_discrete_maskableppo_ncost2x_scaled_reward_031_17 as ppo_base
import run_free_timing_discrete_masked_dqn_ncost2x_scaled_reward_031_19 as dqn_base
from ppo_action_safety import ActionSafetyState, apply_action_safety, normalize_action, update_action_safety_state


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
OUT = ROOT / "benchmark_results" / "032_00_free_timing_stress_aware_ppo_dqn_smoke"
DOC = ROOT / "docs" / "032_00_free_timing_stress_aware_ppo_dqn_smoke_record.md"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"


def ensure_dirs() -> None:
    for rel in [
        "configs",
        "models/LCA",
        "evaluation",
        "daily_outputs/LCA",
        "logs",
        "tensorboard/LCA",
    ]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def latest_observation_dict(env, obs=None, info=None) -> dict[str, Any]:
    return direct_ppo.latest_observation_dict(env, obs, info)


def make_selection(config: dict) -> pd.DataFrame:
    station = str(config["runtime"]["smoke_station"])
    year = int(config["runtime"]["smoke_year"])
    pool = pd.read_csv(POOL)
    row = pool[(pool["station_code"].eq(station)) & (pool["year"].astype(int).eq(year))].copy()
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one {station}{year} row, found {len(row)}")
    row["selected_for_train"] = True
    row["selected_for_eval"] = True
    row["selection_reason"] = "032_00_free_timing_stress_aware_ppo_dqn_smoke"
    return row


def action_grid(config: dict) -> list[dict[str, float]]:
    i_levels = [float(x) for x in config["discrete_actions"]["irrigation_levels"]]
    n_levels = [float(x) for x in config["discrete_actions"]["nitrogen_levels"]]
    return [{"amir": i, "anfer": n} for i in i_levels for n in n_levels]


class StressAwareDiscreteWrapper(gym.Env):
    """Discrete free-timing water/N wrapper with strict masks and stress-relief reward."""

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
        prev_water_stress = scalar(prev.get("swfac"), 0.0)
        cur_water_stress = scalar(latest.get("swfac"), prev_water_stress)
        prev_n_stress = scalar(prev.get("nstres"), 0.0)
        cur_n_stress = scalar(latest.get("nstres"), prev_n_stress)
        water_relief = max(prev_water_stress - cur_water_stress, 0.0)
        nitrogen_relief = max(prev_n_stress - cur_n_stress, 0.0)
        stress_relief_bonus = (
            float(reward_cfg["water_stress_relief_coef"]) * safe_i * water_relief
            + float(reward_cfg["nitrogen_stress_relief_coef"]) * safe_n * nitrogen_relief
        )
        unscaled_reward = final_yield_component - resource_cost + stress_relief_bonus
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
            "yield_component": final_yield_component,
            "resource_cost": resource_cost,
            "water_stress_relief": water_relief,
            "nitrogen_stress_relief": nitrogen_relief,
            "stress_relief_bonus": stress_relief_bonus,
            "reward_unscaled": unscaled_reward,
            "reward_scale": reward_scale,
            "reward_stress_aware": reward,
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
    return StressAwareDiscreteWrapper(base, config)


def ppo_kwargs(config: dict) -> dict[str, Any]:
    return ppo_base.ppo_kwargs(config)


def dqn_kwargs(config: dict) -> dict[str, Any]:
    return dqn_base.dqn_kwargs(config)


def model_filename(algorithm: str, station: str, seed: int) -> Path:
    return OUT / "models" / station / f"{algorithm.lower()}_stress_aware_seed{seed}.zip"


def train_one(config: dict, env_config: dict, algorithm: str) -> pd.DataFrame:
    station = str(config["runtime"]["smoke_station"])
    year = int(config["runtime"]["smoke_year"])
    seed = int(config["seed"])
    model_path = model_filename(algorithm, station, seed)
    env = None
    status = "ok"
    notes = ""
    try:
        env = make_env(config, env_config, station, year, seed, f"{station}_{year}_032_00_{algorithm}_train", evaluation=False)
        if algorithm == "MaskablePPO":
            from sb3_contrib import MaskablePPO

            model = MaskablePPO(
                "MlpPolicy",
                env,
                verbose=0,
                seed=seed,
                tensorboard_log=str(OUT / "tensorboard" / station),
                **ppo_kwargs(config),
            )
        elif algorithm == "DQN":
            from stable_baselines3 import DQN

            model = DQN(
                "MlpPolicy",
                env,
                verbose=0,
                seed=seed,
                tensorboard_log=str(OUT / "tensorboard" / station),
                **dqn_kwargs(config),
            )
        else:
            raise ValueError(algorithm)
        model.learn(total_timesteps=int(config["total_timesteps"]), reset_num_timesteps=True, progress_bar=False)
        model.save(str(model_path.with_suffix("")))
    except Exception:
        status = "failed"
        notes = traceback.format_exc()
    finally:
        if env is not None:
            env.close()
    row = {
        "algorithm": algorithm,
        "station_code": station,
        "train_years": str(year),
        "seed": seed,
        "total_timesteps": int(config["total_timesteps"]),
        "run_status": status,
        "model_path": str(model_path.relative_to(ROOT)) if model_path.exists() else "",
        "notes": notes[-2500:] if notes else "",
    }
    return pd.DataFrame([row])


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
    return int(np.argmax(masked_q)) if np.isfinite(masked_q).any() else 0


def summarize_daily(algorithm: str, daily: pd.DataFrame, daily_path: Path, model_path: Path) -> dict[str, Any]:
    dap = pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce")
    irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0)
    n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0)
    swfac = pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce")
    nstres = pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce")
    stress_bonus = pd.to_numeric(daily.get("stress_relief_bonus", pd.Series(dtype=float)), errors="coerce").fillna(0)
    final_y = float(pd.to_numeric(daily.get("grnwt", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if len(daily) else np.nan
    total_i = float(irr.sum())
    total_n = float(n.sum())
    nonzero = daily[(irr > 0) | (n > 0)]
    action_sequence = "; ".join(
        f"DAP{int(r.dap)} I{float(r.safe_action_amir):g}/N{float(r.safe_action_anfer):g}"
        for r in nonzero.itertuples(index=False)
    )
    return {
        "algorithm": algorithm,
        "policy_name": f"{algorithm.lower()}_stress_aware_5k_seed0",
        "run_status": "ok" if len(daily) and bool(daily["done"].iloc[-1]) else "failed",
        "episode_length": int(len(daily)),
        "final_grnwt": final_y,
        "total_irrigation": total_i,
        "total_n": total_n,
        "profit_simple": final_y - 1.1 * total_i - 1.58 * total_n if np.isfinite(final_y) else np.nan,
        "PFP_N": final_y / total_n if total_n > 0 and np.isfinite(final_y) else np.nan,
        "reward_stress_aware_sum": float(pd.to_numeric(daily.get("reward", pd.Series(dtype=float)), errors="coerce").sum()) if len(daily) else np.nan,
        "stress_relief_bonus_sum_unscaled": float(stress_bonus.sum()) if len(daily) else np.nan,
        "early_dap1_10_irrigation": float(irr[dap <= 10].sum()) if len(daily) else np.nan,
        "early_dap1_10_n": float(n[dap <= 10].sum()) if len(daily) else np.nan,
        "reached_both_caps_by_dap10": bool(float(irr[dap <= 10].sum()) >= 159.999 and float(n[dap <= 10].sum()) >= 249.999) if len(daily) else False,
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


def evaluate_one(config: dict, env_config: dict, train_row: pd.Series) -> pd.DataFrame:
    algorithm = str(train_row["algorithm"])
    if str(train_row["run_status"]) != "ok":
        return pd.DataFrame([{"algorithm": algorithm, "run_status": "failed", "notes": train_row.get("notes", "")}])
    station = str(config["runtime"]["smoke_station"])
    year = int(config["runtime"]["smoke_year"])
    seed = int(config["seed"])
    model_path = ROOT / str(train_row["model_path"])
    if algorithm == "MaskablePPO":
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import get_action_masks

        model = MaskablePPO.load(str(model_path), device="cpu")
    else:
        from stable_baselines3 import DQN

        model = DQN.load(str(model_path), device="cpu")
        get_action_masks = None
    weather = direct_ppo.weather_for_daily(config)
    rows = []
    for split in ["eval"]:
        env = make_env(config, env_config, station, year, seed, f"{station}_{year}_032_00_{algorithm}_{split}", evaluation=True)
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
                if algorithm == "MaskablePPO":
                    mask = get_action_masks(env)
                    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                else:
                    action = masked_greedy_action(model, obs, env.action_masks())
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
                        "algorithm": algorithm,
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
        daily_path = OUT / "daily_outputs" / station / f"{year}_{algorithm.lower()}_stress_aware_eval_daily.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        row = summarize_daily(algorithm, daily, daily_path, model_path)
        row["station_code"] = station
        row["year"] = year
        row["seed"] = seed
        rows.append(row)
    return pd.DataFrame(rows)


def write_record(config: dict, train_summary: pd.DataFrame, eval_summary: pd.DataFrame) -> None:
    lines = [
        "# 032_00 free-timing stress-aware PPO/DQN smoke record",
        "",
        "## Scope",
        "",
        f"- Site-year: {config['runtime']['smoke_station']}{int(config['runtime']['smoke_year'])}.",
        "- Seed0 only.",
        f"- Final model after {int(config['total_timesteps'])} timesteps; no checkpoint selection.",
        "- Algorithms: MaskablePPO and DQN.",
        "- Daily observation; no expert-DAP windows.",
        "- Coarse action grid and 7-day operation interval retained as feasible-operation constraints.",
        "",
        "## Training summary",
        "",
        train_summary.to_string(index=False),
        "",
        "## Evaluation summary",
        "",
        eval_summary.to_string(index=False),
        "",
        "## Interpretation boundary",
        "",
        "- This is an execution/behavior smoke only.",
        "- It does not prove cross-seed, cross-year, or cross-site performance.",
        "- Coefficients are frozen for this task and must not be tuned in-place from these results.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    config = direct_ppo.load_yaml(CONFIG)
    selection = make_selection(config)
    selection.to_csv(OUT / "configs" / "032_00_lc2010_selection.csv", index=False, encoding="utf-8-sig")
    direct_ppo.OUTPUT_ROOT = OUT
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "032_00_resolved_env_config.yaml")

    train_summary = pd.concat(
        [train_one(config, env_config, "MaskablePPO"), train_one(config, env_config, "DQN")],
        ignore_index=True,
    )
    train_summary.to_csv(OUT / "evaluation" / "training_run_summary.csv", index=False, encoding="utf-8-sig")
    eval_summary = pd.concat(
        [evaluate_one(config, env_config, row) for _, row in train_summary.iterrows()],
        ignore_index=True,
        sort=False,
    )
    eval_summary.to_csv(OUT / "evaluation" / "eval_summary.csv", index=False, encoding="utf-8-sig")
    write_record(config, train_summary, eval_summary)
    result = {
        "task": "032_00_free_timing_stress_aware_ppo_dqn_smoke",
        "training_run": True,
        "site_year": f"{config['runtime']['smoke_station']}{int(config['runtime']['smoke_year'])}",
        "seed": int(config["seed"]),
        "record_md": str(DOC.relative_to(ROOT)),
        "eval_summary": str((OUT / "evaluation" / "eval_summary.csv").relative_to(ROOT)),
    }
    (OUT / "032_00_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(eval_summary.to_string(index=False))


if __name__ == "__main__":
    main()

