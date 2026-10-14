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
import run_free_timing_discrete_maskableppo_ncost2x_scaled_reward_031_17 as base
from ppo_action_safety import ActionSafetyState, apply_action_safety, normalize_action, update_action_safety_state


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_41_lc2010_operation_event_cost_maskableppo_smoke.yaml"
OUT = ROOT / "benchmark_results" / "031_41_lc2010_operation_event_cost_maskableppo_smoke"
DOC = ROOT / "docs" / "031_41_lc2010_operation_event_cost_maskableppo_smoke_record.md"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"
REF_03139 = ROOT / "benchmark_results" / "031_39_representative_free_timing_ppo_five_scenario_daily" / "031_39_lca2010_current_free_timing_ppo_five_scenario_summary.csv"


def ensure_dirs() -> None:
    for rel in ["configs", "models/LCA", "daily_outputs/LCA", "evaluation", "logs", "tensorboard/LCA", "figures", "tables"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def json_text(obj: Any, indent: int = 2) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=indent, default=str)


def make_selection(meta: dict[str, Any]) -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    station = str(meta["target_site_year"]["station_code"])
    year = int(meta["target_site_year"]["year"])
    row = pool[(pool["station_code"].eq(station)) & (pool["year"].astype(int).eq(year))].copy()
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one scenario row for {station}{year}, found {len(row)}")
    row["selected_for_train"] = True
    row["selected_for_eval"] = True
    row["selection_reason"] = "031_41_lc2010_operation_event_cost_maskableppo_smoke"
    return row


class OperationEventCostWrapper(gym.Env):
    """Same free-timing discrete wrapper as 031_33, plus fixed per-operation cost."""

    def __init__(self, env, config: dict[str, Any], operation_cost: dict[str, float]):
        super().__init__()
        from gymnasium import spaces

        self.env = env
        self.config = config
        self.operation_cost = operation_cost
        self.grid = base.action_grid(config)
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
        self.last_obs_dict = base.latest_observation_dict(self.env, obs, info)
        return obs, info

    def _dap(self) -> int:
        dap_raw = base.scalar(self.last_obs_dict.get("dap", 1))
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
        action_index = int(np.asarray(action).item())
        if not bool(self.action_masks()[action_index]):
            action_index = 0
        raw_real = dict(self.grid[action_index])
        before = ActionSafetyState(**self.safety_state.__dict__)
        safety_result = apply_action_safety(raw_real, dap, self.safety_state, {**self.config["action_safety"], "enabled": True})
        action_names = list(self.env.formator.action_names)
        safe_norm = normalize_action(action_names, self.env.formator.action_space_dict, safety_result.safe_real_action)
        obs, _env_reward, terminated, truncated, info = self.env.step(safe_norm)
        update_action_safety_state(self.safety_state, safety_result.safe_real_action, dap)
        latest = base.latest_observation_dict(self.env, obs, info)
        self.last_obs_dict = latest

        safe_i = float(safety_result.safe_real_action.get("amir", 0.0))
        safe_n = float(safety_result.safe_real_action.get("anfer", 0.0))
        reward_cfg = self.config["reward"]
        final_yield_component = 0.0
        if bool(terminated or truncated):
            final_yield_component = float(reward_cfg["yield_coef"]) * base.scalar(latest.get("grnwt"), base.scalar(prev.get("grnwt"), 0.0))
        resource_cost = float(reward_cfg["nitrogen_cost"]) * safe_n + float(reward_cfg["water_cost"]) * safe_i
        irrigation_event_cost = float(self.operation_cost["irrigation_event_cost_unscaled"]) if safe_i > 1e-9 else 0.0
        fertilization_event_cost = float(self.operation_cost["fertilization_event_cost_unscaled"]) if safe_n > 1e-9 else 0.0
        operation_event_cost = irrigation_event_cost + fertilization_event_cost
        unscaled_reward = final_yield_component - resource_cost - operation_event_cost
        reward_scale = float(reward_cfg.get("reward_scale", 1.0))
        reward = unscaled_reward * reward_scale
        self.last_action_info = {
            "discrete_action_index": action_index,
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
            "operation_irrigation_event_cost": irrigation_event_cost,
            "operation_fertilization_event_cost": fertilization_event_cost,
            "operation_event_cost": operation_event_cost,
            "operation_adjusted_reward_unscaled": unscaled_reward,
            "reward_scale": reward_scale,
            "operation_adjusted_reward": reward,
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


def make_env(config: dict[str, Any], env_config: dict[str, Any], station: str, year: int, seed: int, run_tag: str, operation_cost: dict[str, float], evaluation: bool = False):
    raw = direct_ppo.make_base_env(env_config, station, year, seed, run_tag, evaluation=evaluation)
    return OperationEventCostWrapper(raw, config, operation_cost)


def checkpoint_path(station: str, year: int, seed: int, step: int) -> Path:
    return OUT / "models" / station / f"{station}_{year}_operation_cost_maskableppo_seed{seed}_ckpt{step}.zip"


class FixedStepCheckpointCallback:
    def __init__(self, station: str, year: int, seed: int, checkpoint_steps: list[int]) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer: FixedStepCheckpointCallback) -> None:
                super().__init__(verbose=0)
                self.outer = outer

            def _on_step(self) -> bool:
                step = int(self.num_timesteps)
                pending = [x for x in self.outer.checkpoint_steps if x <= step and x not in self.outer.saved_steps]
                for target in pending:
                    path = checkpoint_path(self.outer.station, self.outer.year, self.outer.seed, target)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(str(path))
                    self.outer.saved_steps.append(target)
                return True

        self.station = station
        self.year = int(year)
        self.seed = int(seed)
        self.checkpoint_steps = [int(x) for x in checkpoint_steps]
        self.saved_steps: list[int] = []
        self.callback = _Callback(self)


def summarize_daily(daily: pd.DataFrame, daily_path: Path, split: str, model_path: Path) -> dict[str, Any]:
    dap = pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce")
    irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    swfac = pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce")
    nstres = pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce")
    grnwt = pd.to_numeric(daily.get("grnwt", pd.Series(dtype=float)), errors="coerce")
    topwt = pd.to_numeric(daily.get("topwt", pd.Series(dtype=float)), errors="coerce")
    reward = pd.to_numeric(daily.get("operation_adjusted_reward", pd.Series(dtype=float)), errors="coerce")
    reward_unscaled = pd.to_numeric(daily.get("operation_adjusted_reward_unscaled", pd.Series(dtype=float)), errors="coerce")
    yield_component = pd.to_numeric(daily.get("literature_yield_component", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    resource_cost = pd.to_numeric(daily.get("literature_resource_cost", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    event_cost = pd.to_numeric(daily.get("operation_event_cost", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    closure_error = float((reward_unscaled - (yield_component - resource_cost - event_cost)).abs().max()) if len(daily) else np.nan
    final_y = float(grnwt.iloc[-1]) if len(grnwt) else np.nan
    final_biomass = float(topwt.iloc[-1]) if len(topwt) else np.nan
    total_i = float(irr.sum())
    total_n = float(n.sum())
    action_sequence = "; ".join(
        f"DAP{int(r.dap)} I{float(r.safe_action_amir):g}/N{float(r.safe_action_anfer):g}"
        for r in daily[(irr > 0) | (n > 0)].itertuples(index=False)
    )
    return {
        "station_code": str(daily["station_code"].iloc[0]) if len(daily) else "",
        "year": int(daily["year"].iloc[0]) if len(daily) else np.nan,
        "seed": int(daily["seed"].iloc[0]) if len(daily) else np.nan,
        "checkpoint_step": int(daily["checkpoint_step"].iloc[0]) if len(daily) else np.nan,
        "split": split,
        "run_status": "ok" if len(daily) and bool(daily["done"].iloc[-1]) else "failed",
        "episode_length": int(len(daily)),
        "final_grnwt": final_y,
        "final_biomass": final_biomass,
        "total_irrigation": total_i,
        "total_n": total_n,
        "WP_ET_kg_m3": np.nan,
        "yield_per_irrigation_mm_proxy": final_y / total_i if total_i > 0 and np.isfinite(final_y) else np.nan,
        "PFP_N": final_y / total_n if total_n > 0 and np.isfinite(final_y) else np.nan,
        "profit_simple": final_y - total_i - 5.0 * total_n if np.isfinite(final_y) else np.nan,
        "operation_adjusted_reward_sum": float(reward.sum()) if len(reward) else np.nan,
        "operation_event_cost_total_unscaled": float(event_cost.sum()) if len(daily) else np.nan,
        "reward_closure_max_abs_error": closure_error,
        "irrigation_event_count": int((irr > 0).sum()),
        "n_event_count": int((n > 0).sum()),
        "six_mm_irrigation_count": int((irr == 6.0).sum()),
        "small_irrigation_le_6mm_count": int(((irr > 0) & (irr <= 6.0)).sum()),
        "first_irrigation_dap": int(daily.loc[irr > 0, "dap"].iloc[0]) if (irr > 0).any() else np.nan,
        "first_n_dap": int(daily.loc[n > 0, "dap"].iloc[0]) if (n > 0).any() else np.nan,
        "swfac_stress_days_gt_0p05": int((swfac > 0.05).sum()) if len(swfac) else 0,
        "nstres_days_gt_0p05": int((nstres > 0.05).sum()) if len(nstres) else 0,
        "action_sequence": action_sequence,
        "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "daily_csv_path": str(daily_path.relative_to(ROOT)).replace("\\", "/"),
    }


def train_one(config: dict[str, Any], env_config: dict[str, Any], meta: dict[str, Any]) -> pd.DataFrame:
    from sb3_contrib import MaskablePPO

    station = str(meta["target_site_year"]["station_code"])
    year = int(meta["target_site_year"]["year"])
    seed = int(meta["seed"])
    checkpoint_steps = [int(x) for x in meta["checkpoint_steps"]]
    expected = [checkpoint_path(station, year, seed, step) for step in checkpoint_steps]
    rows: list[dict[str, Any]] = []
    if all(p.exists() for p in expected):
        for step, path in zip(checkpoint_steps, expected):
            rows.append({
                "station_code": station,
                "year": year,
                "seed": seed,
                "checkpoint_step": step,
                "run_status": "ok_existing",
                "model_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "model_sha256": sha256_file(path),
            })
        return pd.DataFrame(rows)

    env = None
    try:
        env = make_env(config, env_config, station, year, seed, f"{station}_{year}_031_41_seed{seed}_train", meta["operation_event_cost"], evaluation=False)
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            tensorboard_log=str(OUT / "tensorboard" / station),
            **base.ppo_kwargs(config),
        )
        cb = FixedStepCheckpointCallback(station, year, seed, checkpoint_steps)
        model.learn(total_timesteps=int(meta["max_train_steps"]), reset_num_timesteps=True, progress_bar=False, callback=cb.callback)
        for step, path in zip(checkpoint_steps, expected):
            rows.append({
                "station_code": station,
                "year": year,
                "seed": seed,
                "checkpoint_step": step,
                "run_status": "ok" if path.exists() else "missing",
                "model_path": str(path.relative_to(ROOT)).replace("\\", "/") if path.exists() else "",
                "model_sha256": sha256_file(path) if path.exists() else "",
            })
    except Exception:
        rows.append({
            "station_code": station,
            "year": year,
            "seed": seed,
            "checkpoint_step": "",
            "run_status": "failed",
            "notes": traceback.format_exc()[-4000:],
        })
    finally:
        if env is not None:
            env.close()
    return pd.DataFrame(rows)


def evaluate_checkpoint(config: dict[str, Any], env_config: dict[str, Any], meta: dict[str, Any], step: int) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    station = str(meta["target_site_year"]["station_code"])
    year = int(meta["target_site_year"]["year"])
    seed = int(meta["seed"])
    model_path = checkpoint_path(station, year, seed, int(step))
    daily_path = OUT / "daily_outputs" / station / f"{station}_{year}_seed{seed}_ckpt{step}_operation_cost_daily.csv"
    if not model_path.exists():
        return {"station_code": station, "year": year, "seed": seed, "checkpoint_step": int(step), "run_status": "missing_model"}
    model = MaskablePPO.load(str(model_path), device="cpu")
    weather = direct_ppo.weather_for_daily(config)
    env = make_env(config, env_config, station, year, seed, f"{station}_{year}_031_41_seed{seed}_ckpt{step}_eval", meta["operation_event_cost"], evaluation=True)
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
            records.append({
                "station_code": station,
                "year": year,
                "seed": seed,
                "checkpoint_step": int(step),
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
            })
            step_count += 1
    except Exception:
        return {"station_code": station, "year": year, "seed": seed, "checkpoint_step": int(step), "run_status": "failed", "notes": traceback.format_exc()[-4000:]}
    finally:
        env.close()
    daily = pd.DataFrame(records)
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    return summarize_daily(daily, daily_path, "train_year_eval", model_path)


def select_checkpoint(rows: pd.DataFrame, meta: dict[str, Any]) -> pd.Series:
    ok = rows[rows["run_status"].eq("ok")].copy()
    if ok.empty:
        raise RuntimeError("No successful checkpoint evaluation rows")
    ok = ok.sort_values(
        ["operation_adjusted_reward_sum", "final_grnwt", "irrigation_event_count", "total_irrigation", "total_n", "checkpoint_step"],
        ascending=[False, False, True, True, True, True],
        kind="stable",
    )
    return ok.iloc[0]


def reference_summary() -> pd.DataFrame:
    frames = []
    if REF_03139.exists():
        df = pd.read_csv(REF_03139)
        df["source"] = "031_39_current_frozen_candidate_and_baselines"
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def write_record(meta: dict[str, Any], train_rows: pd.DataFrame, eval_rows: pd.DataFrame, selected: pd.Series, ref: pd.DataFrame) -> None:
    selected_df = selected.to_frame().T
    def frame_text(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "No rows."
        return frame.to_string(index=False)

    lines = [
        "# 031_41 LC2010 operation-event-cost MaskablePPO smoke record",
        "",
        "## Scope",
        "",
        "- LC2010 only, seed1 only.",
        "- Same free-timing MaskablePPO framework as 031_33.",
        "- Only reward accounting changed: fixed operation costs were added.",
        "- Action grid, seasonal caps, min interval, DAP ranges, PPO hyperparameters, weather, and DSSAT inputs were not changed.",
        "",
        "## Operation-event-cost reward",
        "",
        "```text",
        "unscaled_reward = literature_yield_component - literature_resource_cost - operation_event_cost",
        "operation_event_cost = 10 * 1[irrigation > 0] + 20 * 1[nitrogen > 0]",
        "reward = unscaled_reward * 0.001",
        "```",
        "",
        "## Pre-registered config",
        "",
        "```json",
        json_text(meta, indent=2),
        "```",
        "",
        "## Training summary",
        "",
        frame_text(train_rows),
        "",
        "## Checkpoint evaluation summary",
        "",
        frame_text(eval_rows),
        "",
        "## Selected checkpoint",
        "",
        frame_text(selected_df),
        "",
        "## 031_39 reference context",
        "",
        frame_text(ref.head(20)) if not ref.empty else "Reference table not found.",
        "",
        "## Interpretation",
        "",
    ]
    ref_candidate = pd.DataFrame()
    if not ref.empty and "scenario" in ref.columns:
        ref_candidate = ref[ref["scenario"].astype(str).str.contains("PPO|Maskable|candidate", case=False, regex=True, na=False)].copy()
    if not ref_candidate.empty:
        r = ref_candidate.iloc[0]
        lines.extend([
            f"- Reference PPO candidate total irrigation/event count: {r.get('total_irrigation', 'NA')} mm / {r.get('irrigation_event_count', 'NA')} events.",
            f"- 031_41 selected total irrigation/event count: {selected.get('total_irrigation', 'NA')} mm / {selected.get('irrigation_event_count', 'NA')} events.",
            f"- 031_41 selected six-mm irrigation count: {selected.get('six_mm_irrigation_count', 'NA')}.",
        ])
    else:
        lines.extend([
            f"- 031_41 selected total irrigation/event count: {selected.get('total_irrigation', 'NA')} mm / {selected.get('irrigation_event_count', 'NA')} events.",
            f"- 031_41 selected six-mm irrigation count: {selected.get('six_mm_irrigation_count', 'NA')}.",
        ])
    closure = pd.to_numeric(eval_rows.get("reward_closure_max_abs_error", pd.Series(dtype=float)), errors="coerce").max()
    lines.extend([
        f"- Reward closure max absolute error across evaluated checkpoints: {closure}.",
        "- `WP_ET_kg_m3` is intentionally left blank in this smoke summary because formal WP_ET requires ET/ETCP extraction from DSSAT outputs; `yield_per_irrigation_mm_proxy` is only a quick irrigation-efficiency proxy and must not be reported as WP_ET.",
        "",
        "This smoke does not authorize all-site reruns. It only tests whether fixed operation cost is a plausible way to reduce management-unrealistic repeated small irrigation.",
    ])
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    meta = load_yaml(CONFIG)
    base_config = direct_ppo.load_yaml(ROOT / str(meta["base_config"]))
    config = json.loads(json.dumps(base_config))
    station = str(meta["target_site_year"]["station_code"])
    year = int(meta["target_site_year"]["year"])
    seed = int(meta["seed"])
    config["seed"] = seed
    config["total_timesteps"] = int(meta["max_train_steps"])
    config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    direct_ppo.write_yaml(config, OUT / "configs" / "031_41_resolved_training_config.yaml")
    selection = make_selection(meta)
    selection.to_csv(OUT / "configs" / "031_41_lc2010_selection.csv", index=False, encoding="utf-8-sig")
    direct_ppo.OUTPUT_ROOT = OUT
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "031_41_resolved_env_config.yaml")

    train_rows = train_one(config, env_config, meta)
    train_rows.to_csv(OUT / "evaluation" / "031_41_training_run_summary.csv", index=False, encoding="utf-8-sig")
    eval_rows = pd.DataFrame([evaluate_checkpoint(config, env_config, meta, int(step)) for step in meta["checkpoint_steps"]])
    eval_rows.to_csv(OUT / "evaluation" / "031_41_checkpoint_eval_summary.csv", index=False, encoding="utf-8-sig")
    selected = select_checkpoint(eval_rows, meta)
    selected.to_frame().T.to_csv(OUT / "evaluation" / "031_41_selected_checkpoint.csv", index=False, encoding="utf-8-sig")
    ref = reference_summary()
    if not ref.empty:
        ref.to_csv(OUT / "tables" / "031_41_reference_03139_summary.csv", index=False, encoding="utf-8-sig")
    write_record(meta, train_rows, eval_rows, selected, ref)
    print("031_41 complete")
    print(selected.to_json(force_ascii=False))


if __name__ == "__main__":
    main()
