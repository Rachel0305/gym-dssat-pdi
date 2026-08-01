"""040_45: SYA lowIC continuous-action SAC smoke.

This is a candidate-algorithm smoke test, not a replacement for 040_40 PPO.
SAC emits continuous water/N amounts; the same major agronomic safety constraints
used by 040_40 are enforced before actions reach DSSAT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base032
import run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36 as base04036
import run_sya_lowIC_ppo_yield_guardrail_v3_040_40 as base04040
from ppo_action_safety import ActionSafetyState, apply_action_safety, normalize_action, update_action_safety_state


TASK_ID = "040_45"
TASK_NAME = "sya_lowIC_continuous_sac_smoke"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

STATION = "SYA"
SITE = "SY"
SEED = 0
DEFAULT_TOTAL_TIMESTEPS = 200_000
DEFAULT_CHECKPOINT_STEPS = [50_000, 100_000, 150_000, 200_000]
LOWIC_INPUT_ROOT = base04040.LOWIC_INPUT_ROOT

PPO_04040_EVAL = ROOT / "benchmark_results" / "040_40_sya_lowIC_ppo_yield_guardrail_v3" / "evaluation" / "040_40_checkpoint_validation_summary.csv"


def ensure_dirs() -> None:
    for rel in ["configs", "models/SYA", "evaluation", "daily_outputs/SYA", "logs", "tensorboard/SYA", "tables"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def parse_checkpoint_steps(raw: str | None, total_timesteps: int) -> list[int]:
    if not raw:
        return [x for x in DEFAULT_CHECKPOINT_STEPS if x <= int(total_timesteps)]
    steps = [int(x.strip()) for x in raw.split(",") if x.strip()]
    return [x for x in steps if x <= int(total_timesteps)]


def load_config(total_timesteps: int) -> dict[str, Any]:
    cfg = base04040.load_config()
    cfg = json.loads(json.dumps(cfg))
    cfg["seed"] = SEED
    cfg["total_timesteps"] = int(total_timesteps)
    cfg["paths"]["output_root"] = OUT.relative_to(ROOT).as_posix()
    cfg["runtime"]["smoke_station"] = STATION
    cfg["runtime"]["smoke_year"] = 2005
    cfg["algorithm"] = "SAC"
    cfg["continuous_actions"] = {
        "irrigation_low_high_mm": [0.0, 45.0],
        "nitrogen_low_high_kg_ha": [0.0, 120.0],
        "note": "SAC emits continuous real amounts; agronomic constraints are applied before DSSAT execution.",
    }
    cfg["sac"] = {
        "learning_rate": 0.0003,
        "buffer_size": 100_000,
        "learning_starts": 1024,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 1.0,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "net_arch": [64, 64],
    }
    return cfg


def sac_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    sac = dict(config.get("sac", {}))
    net_arch = sac.pop("net_arch", [64, 64])
    return {
        **sac,
        "policy_kwargs": {"net_arch": list(net_arch)},
    }


def load_split() -> pd.DataFrame:
    split = pd.read_csv(base03222.SPLIT_CSV, keep_default_na=False)
    split = split[split["station_code"].astype(str).eq(STATION)].copy()
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    return split.sort_values("year").reset_index(drop=True)


def build_selection(split: pd.DataFrame) -> pd.DataFrame:
    pool = pd.read_csv(base03222.POOL, keep_default_na=False)
    pool["year"] = pd.to_numeric(pool["year"], errors="coerce").astype(int)
    selected = pool.merge(split[["station_code", "year", "split"]], on=["station_code", "year"], how="inner")
    selected["selected_for_train"] = selected["split"].eq("train")
    selected["selected_for_eval"] = True
    selected["selection_reason"] = f"{TASK_ID}_{TASK_NAME}"
    return selected.sort_values(["station_code", "year"]).reset_index(drop=True)


class ContinuousSacSafetyWrapper(gym.Env):
    """Continuous water/N wrapper with 040_40-style reward and safety constraints."""

    def __init__(self, env, config: dict[str, Any], station: str, year: int):
        super().__init__()
        from gymnasium import spaces

        self.env = env
        self.config = config
        self.station = str(station)
        self.year = int(year)
        self.action_space = spaces.Box(
            low=np.asarray([0.0, 0.0], dtype=np.float32),
            high=np.asarray([45.0, 120.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = env.observation_space
        self.metadata = getattr(env, "metadata", {})
        self.safety_state = ActionSafetyState()
        self.last_obs_dict: dict[str, Any] = {}
        self.last_action_info: dict[str, Any] = {}
        self.guardrail_target_yield = base04040.get_target_yield(station, int(year))

    def reset(self, *args, **kwargs):
        self.safety_state = ActionSafetyState()
        self.last_action_info = {}
        obs, info = self.env.reset(*args, **kwargs)
        self.last_obs_dict = direct_ppo.latest_observation_dict(self.env, obs, info)
        return obs, info

    def _dap(self) -> int:
        raw = scalar(self.last_obs_dict.get("dap", 1), 1)
        return int(round(raw)) if np.isfinite(raw) and raw > 0 else 1

    def _stage_clip_irrigation(self, safe: dict[str, float], dap: int, triggers: list[str]) -> dict[str, float]:
        irrigation = float(safe.get("amir", 0.0) or 0.0)
        if irrigation <= 1e-9:
            return safe
        current_i = float(self.safety_state.cumulative_irrigation)
        caps: list[tuple[str, float]] = []
        if dap <= 30:
            caps.append(("early_i75_cap", 75.0))
        if dap <= 60:
            caps.append(("mid_i150_cap", 150.0))
        if dap <= 90:
            caps.append(("pre_late_i195_cap", 195.0))
        for label, cap in caps:
            allowed = max(0.0, cap - current_i)
            if safe["amir"] > allowed + 1e-9:
                triggers.append(label)
                safe["amir"] = allowed
        return safe

    def step(self, action):
        prev = dict(self.last_obs_dict)
        dap = self._dap()
        arr = np.asarray(action, dtype=float).reshape(-1)
        raw_real = {"amir": float(np.clip(arr[0] if len(arr) > 0 else 0.0, 0.0, 45.0)), "anfer": float(np.clip(arr[1] if len(arr) > 1 else 0.0, 0.0, 120.0))}
        before = ActionSafetyState(**self.safety_state.__dict__)
        result = apply_action_safety(raw_real, dap, self.safety_state, {**self.config["action_safety"], "enabled": True})
        safe_real = dict(result.safe_real_action)
        extra_triggers: list[str] = []
        safe_real = self._stage_clip_irrigation(safe_real, dap, extra_triggers)
        action_names = list(self.env.formator.action_names)
        safe_norm = normalize_action(action_names, self.env.formator.action_space_dict, safe_real)
        obs, _env_reward, terminated, truncated, info = self.env.step(safe_norm)
        update_action_safety_state(self.safety_state, safe_real, dap)
        latest = direct_ppo.latest_observation_dict(self.env, obs, info)
        self.last_obs_dict = latest

        safe_i = float(safe_real.get("amir", 0.0))
        safe_n = float(safe_real.get("anfer", 0.0))
        reward_cfg = self.config["reward"]
        final_yield_component = 0.0
        final_yield = scalar(latest.get("grnwt"), scalar(prev.get("grnwt"), np.nan))
        if bool(terminated or truncated) and np.isfinite(final_yield):
            final_yield_component = float(reward_cfg["yield_coef"]) * float(final_yield)
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
        reward_scale = float(reward_cfg.get("reward_scale", 1.0))
        swfac_excess = max(float(cur_water_stress) - 0.05, 0.0) if np.isfinite(cur_water_stress) else 0.0
        swfac_penalty_unscaled = 50.0 * swfac_excess
        yield_deficit = 0.0
        yield_penalty_unscaled = 0.0
        if bool(terminated or truncated) and np.isfinite(final_yield):
            yield_deficit = max(float(self.guardrail_target_yield) - float(final_yield), 0.0)
            yield_penalty_unscaled = float(reward_cfg["yield_coef"]) * yield_deficit
        reward_unscaled = final_yield_component - resource_cost + stress_relief_bonus - swfac_penalty_unscaled - yield_penalty_unscaled
        reward = reward_unscaled * reward_scale
        triggers = [x for x in [result.safety_rule_triggered, *extra_triggers] if x]
        self.last_action_info = {
            "raw_action_amir": raw_real["amir"],
            "raw_action_anfer": raw_real["anfer"],
            "safe_action_amir": safe_i,
            "safe_action_anfer": safe_n,
            "season_cumulative_irrigation": float(self.safety_state.cumulative_irrigation),
            "season_cumulative_n": float(self.safety_state.cumulative_n),
            "action_safety_triggered": ";".join(dict.fromkeys(";".join(triggers).split(";"))) if triggers else "",
            "yield_component": final_yield_component,
            "resource_cost": resource_cost,
            "water_stress_relief": water_relief,
            "nitrogen_stress_relief": nitrogen_relief,
            "stress_relief_bonus": stress_relief_bonus,
            "swfac_guardrail_excess": swfac_excess,
            "swfac_guardrail_penalty_unscaled": swfac_penalty_unscaled,
            "yield_guardrail_target_yield": float(self.guardrail_target_yield),
            "yield_guardrail_final_yield": float(final_yield) if np.isfinite(final_yield) else np.nan,
            "yield_guardrail_deficit": yield_deficit,
            "yield_guardrail_penalty_unscaled": yield_penalty_unscaled,
            "reward_unscaled": reward_unscaled,
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


def make_env(config: dict[str, Any], env_config: dict[str, Any], station: str, year: int, seed: int, run_tag: str, evaluation: bool = False):
    base_env = direct_ppo.make_base_env(env_config, station, year, seed, run_tag, evaluation=evaluation)
    return ContinuousSacSafetyWrapper(base_env, config, station, int(year))


class RandomYearEnv(gym.Env):
    def __init__(self, config: dict[str, Any], env_config: dict[str, Any], station: str, years: list[int], seed: int) -> None:
        super().__init__()
        self.config = config
        self.env_config = env_config
        self.station = station
        self.years = [int(y) for y in years]
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.envs: dict[int, gym.Env] = {}
        self.current_year: int | None = None
        self.reset_counts: Counter[int] = Counter()
        self.switch_log: list[dict[str, Any]] = []
        first = self._env_for(self.years[0])
        self.action_space = first.action_space
        self.observation_space = first.observation_space
        self.metadata = getattr(first, "metadata", {})

    def _env_for(self, year: int):
        year = int(year)
        if year not in self.envs:
            self.envs[year] = make_env(self.config, self.env_config, self.station, year, self.seed, f"{self.station}_{year}_040_45_train", evaluation=False)
        return self.envs[year]

    @property
    def current_env(self):
        if self.current_year is None:
            self.current_year = self.years[0]
        return self._env_for(self.current_year)

    def reset(self, *args, **kwargs):
        year = int(self.rng.choice(self.years))
        self.current_year = year
        self.reset_counts[year] += 1
        self.switch_log.append({"episode_index": int(sum(self.reset_counts.values())), "station_code": self.station, "year": year})
        obs, info = self.current_env.reset(*args, **kwargs)
        info = dict(info or {})
        info["active_year"] = year
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.current_env.step(action)
        info = dict(info or {})
        info["active_year"] = int(self.current_year)
        return obs, reward, terminated, truncated, info

    @property
    def last_action_info(self) -> dict[str, Any]:
        return dict(getattr(self.current_env, "last_action_info", {}))

    def close(self):
        for env in self.envs.values():
            try:
                env.close()
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self.current_env, name)


def model_path(step: int) -> Path:
    return OUT / "models" / STATION / f"{STATION}_lowIC_continuous_sac_seed{SEED}_ckpt{int(step)}.zip"


class FixedStepCheckpointCallback:
    def __init__(self, checkpoint_steps: list[int]) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer: FixedStepCheckpointCallback) -> None:
                super().__init__(verbose=0)
                self.outer = outer

            def _on_step(self) -> bool:
                step_now = int(self.num_timesteps)
                pending = [x for x in self.outer.checkpoint_steps if x <= step_now and x not in self.outer.saved_steps]
                for target in pending:
                    path = model_path(target)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(str(path))
                    self.outer.saved_steps.append(int(target))
                return True

        self.checkpoint_steps = [int(x) for x in checkpoint_steps]
        self.saved_steps: list[int] = []
        self.callback = _Callback(self)


def train(config: dict[str, Any], env_config: dict[str, Any], train_years: list[int], checkpoint_steps: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    from stable_baselines3 import SAC

    expected = [model_path(step) for step in checkpoint_steps]
    if all(path.exists() for path in expected):
        rows = [
            {
                "station_code": STATION,
                "site": SITE,
                "train_years": ",".join(map(str, train_years)),
                "seed": SEED,
                "checkpoint_step": step,
                "run_status": "ok_existing",
                "model_path": path.relative_to(ROOT).as_posix(),
                "model_sha256": sha256_file(path),
            }
            for step, path in zip(checkpoint_steps, expected)
        ]
        reset_path = OUT / "logs" / "040_45_training_year_reset_counts.csv"
        reset_df = pd.read_csv(reset_path) if reset_path.exists() and reset_path.stat().st_size > 0 else pd.DataFrame()
        return pd.DataFrame(rows), reset_df

    env: RandomYearEnv | None = None
    status = "ok"
    notes = ""
    reset_df = pd.DataFrame()
    try:
        env = RandomYearEnv(config, env_config, STATION, train_years, SEED)
        model = SAC(
            "MlpPolicy",
            env,
            verbose=0,
            seed=SEED,
            tensorboard_log=str(OUT / "tensorboard" / STATION),
            **sac_kwargs(config),
        )
        callback = FixedStepCheckpointCallback(checkpoint_steps)
        model.learn(total_timesteps=int(config["total_timesteps"]), reset_num_timesteps=True, progress_bar=False, callback=callback.callback)
    except Exception:
        status = "failed"
        notes = traceback.format_exc()
    finally:
        if env is not None:
            reset_df = pd.DataFrame([{"station_code": STATION, "year": int(year), "episode_count": int(env.reset_counts.get(year, 0))} for year in train_years])
            reset_df.to_csv(OUT / "logs" / "040_45_training_year_reset_counts.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(env.switch_log).to_csv(OUT / "logs" / "040_45_training_year_switch_log.csv", index=False, encoding="utf-8-sig")
            env.close()
    rows = []
    for step, path in zip(checkpoint_steps, expected):
        rows.append(
            {
                "station_code": STATION,
                "site": SITE,
                "train_years": ",".join(map(str, train_years)),
                "seed": SEED,
                "checkpoint_step": int(step),
                "run_status": status if status != "ok" else ("ok" if path.exists() else "missing"),
                "model_path": path.relative_to(ROOT).as_posix() if path.exists() else "",
                "model_sha256": sha256_file(path) if path.exists() else "",
                "notes": notes[-4000:] if notes else "",
            }
        )
    return pd.DataFrame(rows), reset_df


def stress_summary(daily: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in ["swfac", "nstres"]:
        s = pd.to_numeric(daily.get(col, pd.Series(dtype=float)), errors="coerce")
        out[f"max_{col}"] = float(s.max()) if len(s) else np.nan
        out[f"{col}_days_gt_0p05"] = int((s > 0.05).sum()) if len(s) else 0
    return out


def evaluate_checkpoint(config: dict[str, Any], env_config: dict[str, Any], train_row: pd.Series, year: int) -> dict[str, Any]:
    from stable_baselines3 import SAC

    step = int(train_row["checkpoint_step"])
    path = ROOT / str(train_row["model_path"])
    daily_path = OUT / "daily_outputs" / STATION / f"{STATION}_{year}_seed{SEED}_ckpt{step}_daily.csv"
    if not path.exists():
        return {"station_code": STATION, "site": SITE, "year": int(year), "seed": SEED, "checkpoint_step": step, "run_status": "missing_model"}
    if daily_path.exists():
        daily = pd.read_csv(daily_path)
        row = base032.summarize_daily("SAC", daily, daily_path, path)
        row.update({"station_code": STATION, "site": SITE, "year": int(year), "seed": SEED, "checkpoint_step": step, "run_status": "ok_existing"})
        row.update(stress_summary(daily))
        return row

    model = SAC.load(str(path), device="cpu")
    weather = direct_ppo.weather_for_daily(config)
    records: list[dict[str, Any]] = []
    env = make_env(config, env_config, STATION, int(year), SEED, f"{STATION}_{year}_040_45_ckpt{step}_eval", evaluation=True)
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, STATION, int(year))["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = direct_ppo.latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap", step_count + 1), step_count + 1)
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = direct_ppo.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(STATION)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "station_code": STATION,
                    "year": int(year),
                    "seed": SEED,
                    "checkpoint_step": step,
                    "algorithm": "SAC",
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
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    row = base032.summarize_daily("SAC", daily, daily_path, path)
    row.update({"station_code": STATION, "site": SITE, "year": int(year), "seed": SEED, "checkpoint_step": step, "run_status": "ok"})
    row.update(stress_summary(daily))
    return row


def add_ppo_comparison(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty or not PPO_04040_EVAL.exists():
        return eval_df
    ppo = pd.read_csv(PPO_04040_EVAL, keep_default_na=False)
    ppo = ppo[pd.to_numeric(ppo["checkpoint_step"], errors="coerce").eq(100000)].copy()
    keep = ["year", "final_grnwt", "total_irrigation", "total_n", "PFP_N", "max_swfac", "max_nstres"]
    ppo = ppo[[c for c in keep if c in ppo.columns]].copy()
    ppo = ppo.rename(columns={c: f"ppo04040_{c}" for c in ppo.columns if c != "year"})
    out = eval_df.merge(ppo, on="year", how="left")
    for col in ["final_grnwt", "total_irrigation", "total_n", "PFP_N", "max_swfac", "max_nstres"]:
        pcol = f"ppo04040_{col}"
        if col in out.columns and pcol in out.columns:
            out[f"delta_{col}_vs_ppo04040"] = pd.to_numeric(out[col], errors="coerce") - pd.to_numeric(out[pcol], errors="coerce")
    return out


def action_sequence_from_daily(path: Path) -> str:
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    irr = pd.to_numeric(df.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    n = pd.to_numeric(df.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    parts: list[str] = []
    for _, row in df[(irr > 1e-6) | (n > 1e-6)].iterrows():
        dap = int(round(float(row["dap"])))
        i = float(row.get("safe_action_amir", 0.0))
        nf = float(row.get("safe_action_anfer", 0.0))
        if i > 1e-6:
            parts.append(f"DAP{dap}:I{i:.1f}")
        if nf > 1e-6:
            parts.append(f"DAP{dap}:N{nf:.1f}")
    return "; ".join(parts)


def summarize_by_checkpoint(eval_df: pd.DataFrame) -> pd.DataFrame:
    ok = eval_df[eval_df["run_status"].astype(str).str.startswith("ok")].copy() if not eval_df.empty else pd.DataFrame()
    if ok.empty:
        return pd.DataFrame()
    rows = []
    for step, g in ok.groupby("checkpoint_step"):
        seqs_i = []
        seqs_n = []
        for path_str in g.get("daily_csv_path", pd.Series(dtype=str)).astype(str):
            path = ROOT / path_str
            if not path.exists():
                continue
            d = pd.read_csv(path)
            irr_events = d[pd.to_numeric(d.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0.0) > 1e-6]
            n_events = d[pd.to_numeric(d.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0.0) > 1e-6]
            seqs_i.append(";".join(f"{int(r.dap)}:{float(r.safe_action_amir):.1f}" for r in irr_events.itertuples(index=False)))
            seqs_n.append(";".join(f"{int(r.dap)}:{float(r.safe_action_anfer):.1f}" for r in n_events.itertuples(index=False)))
        rows.append(
            {
                "checkpoint_step": int(step),
                "validation_years": int(g["year"].nunique()),
                "mean_final_grnwt": float(pd.to_numeric(g["final_grnwt"], errors="coerce").mean()),
                "mean_total_irrigation": float(pd.to_numeric(g["total_irrigation"], errors="coerce").mean()),
                "mean_total_n": float(pd.to_numeric(g["total_n"], errors="coerce").mean()),
                "mean_PFP_N": float(pd.to_numeric(g["PFP_N"], errors="coerce").mean()),
                "max_swfac": float(pd.to_numeric(g["max_swfac"], errors="coerce").max()),
                "max_nstres": float(pd.to_numeric(g["max_nstres"], errors="coerce").max()),
                "mean_delta_yield_vs_ppo04040": float(pd.to_numeric(g.get("delta_final_grnwt_vs_ppo04040", pd.Series(dtype=float)), errors="coerce").mean()),
                "unique_irrigation_sequence_count": int(pd.Series(seqs_i).nunique()) if seqs_i else 0,
                "unique_nitrogen_sequence_count": int(pd.Series(seqs_n).nunique()) if seqs_n else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("checkpoint_step")


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def write_record(config: dict[str, Any], split: pd.DataFrame, train_df: pd.DataFrame, reset_df: pd.DataFrame, eval_df: pd.DataFrame, by_ckpt: pd.DataFrame) -> None:
    eval_cols = [
        "year",
        "checkpoint_step",
        "run_status",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "PFP_N",
        "max_swfac",
        "max_nstres",
        "delta_final_grnwt_vs_ppo04040",
        "delta_total_irrigation_vs_ppo04040",
        "delta_total_n_vs_ppo04040",
        "delta_PFP_N_vs_ppo04040",
        "action_sequence",
    ]
    lines = [
        "# 040_45 SYA lowIC 连续动作 SAC smoke 记录",
        "",
        "## 结论先说",
        "",
        "- 本任务是候选算法 smoke，不是正式替代 PPO。",
        f"- 输入目录：`{LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix()}`。",
        f"- 算法：SAC，seed={SEED}，timesteps={int(config['total_timesteps'])}。",
        "- SAC 输出连续水氮量，随后由同一安全层和阶段灌溉上限裁剪后进入 DSSAT。",
        "",
        "## 年份划分",
        "",
        md_table(split[["station_code", "site", "year", "split"]], max_rows=40),
        "",
        "## 训练 checkpoint",
        "",
        md_table(train_df, max_rows=20),
        "",
        "## 训练年份采样次数",
        "",
        md_table(reset_df, max_rows=20),
        "",
        "## checkpoint 平均表现及动作年际差异",
        "",
        md_table(by_ckpt, max_rows=20),
        "",
        "## 逐年验证，与 040_40 PPO checkpoint100k 对比",
        "",
        md_table(eval_df[[c for c in eval_cols if c in eval_df.columns]], max_rows=120),
        "",
        "## 解释边界",
        "",
        "- SAC 无原生 action mask；本任务使用安全层裁剪，可能引入“模型输出”和“DSSAT实际执行”之间的差异。",
        "- 若 SAC 指标差或动作仍固定，不能说明 SAC 永远不可用，只说明当前同约束/同 reward smoke 下没有优于 PPO。",
        "- 若 SAC 动作更随年份变化但指标较差，后续需要判断是否值得围绕 SAC 单独调 reward/约束。",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")


def dry_run(config: dict[str, Any], split: pd.DataFrame, checkpoint_steps: list[int]) -> None:
    train_years = split[split["split"].eq("train")]["year"].astype(int).tolist()
    val_years = split[split["split"].eq("validation")]["year"].astype(int).tolist()
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "mode": "dry_run",
        "input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "input_root_exists": LOWIC_INPUT_ROOT.exists(),
        "train_years": train_years,
        "validation_years": val_years,
        "total_timesteps": int(config["total_timesteps"]),
        "checkpoint_steps": checkpoint_steps,
        "action_space": config["continuous_actions"],
        "action_safety": config["action_safety"],
        "sac": config["sac"],
        "next_step_allowed": bool(LOWIC_INPUT_ROOT.exists() and len(train_years) > 0 and len(val_years) > 0),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--checkpoint-steps", type=str, default=None)
    args = parser.parse_args()

    ensure_dirs()
    if PROMPT.exists():
        shutil.copyfile(PROMPT, OUT / "configs" / PROMPT.name)
    config = load_config(args.timesteps)
    checkpoint_steps = parse_checkpoint_steps(args.checkpoint_steps, int(args.timesteps))
    split = load_split()
    if args.dry_run:
        dry_run(config, split, checkpoint_steps)
        return
    selection = build_selection(split)
    selection.to_csv(OUT / "configs" / "040_45_half_split_selection.csv", index=False, encoding="utf-8-sig")
    direct_ppo.OUTPUT_ROOT = OUT
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(config, OUT / "configs" / "040_45_resolved_training_config.yaml")
    direct_ppo.write_yaml(env_config, OUT / "configs" / "040_45_resolved_env_config.yaml")

    train_years = split[split["split"].eq("train")]["year"].astype(int).tolist()
    validation_years = split[split["split"].eq("validation")]["year"].astype(int).tolist()
    train_df, reset_df = train(config, env_config, train_years, checkpoint_steps)
    train_df.to_csv(OUT / "evaluation" / "040_45_training_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
    eval_rows: list[dict[str, Any]] = []
    for _, row in train_df.iterrows():
        if not str(row["run_status"]).startswith("ok"):
            continue
        for year in validation_years:
            try:
                eval_rows.append(evaluate_checkpoint(config, env_config, row, int(year)))
            except Exception:
                eval_rows.append({"station_code": STATION, "site": SITE, "year": int(year), "checkpoint_step": int(row["checkpoint_step"]), "run_status": "failed", "notes": traceback.format_exc()[-4000:]})
    eval_df = pd.DataFrame(eval_rows)
    eval_df = add_ppo_comparison(eval_df)
    for i, row in eval_df.iterrows():
        path_str = str(row.get("daily_csv_path", ""))
        eval_df.loc[i, "action_sequence"] = action_sequence_from_daily(ROOT / path_str) if path_str else ""
    by_ckpt = summarize_by_checkpoint(eval_df)
    eval_df.to_csv(OUT / "evaluation" / "040_45_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
    by_ckpt.to_csv(OUT / "evaluation" / "040_45_validation_summary_by_checkpoint.csv", index=False, encoding="utf-8-sig")
    write_record(config, split, train_df, reset_df, eval_df, by_ckpt)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "algorithm": "SAC",
        "input_root": LOWIC_INPUT_ROOT.relative_to(ROOT).as_posix(),
        "record_md": DOC.relative_to(ROOT).as_posix(),
        "train_inventory": (OUT / "evaluation" / "040_45_training_checkpoint_inventory.csv").relative_to(ROOT).as_posix(),
        "validation_summary": (OUT / "evaluation" / "040_45_checkpoint_validation_summary.csv").relative_to(ROOT).as_posix(),
        "by_checkpoint": (OUT / "evaluation" / "040_45_validation_summary_by_checkpoint.csv").relative_to(ROOT).as_posix(),
        "total_timesteps": int(config["total_timesteps"]),
        "checkpoint_steps": checkpoint_steps,
    }
    (OUT / "040_45_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not by_ckpt.empty:
        print(by_ckpt.to_string(index=False))


if __name__ == "__main__":
    main()

