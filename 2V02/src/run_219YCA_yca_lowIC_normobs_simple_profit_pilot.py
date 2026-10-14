"""219YCA: short YC pilot with observation normalization only.

The experiment keeps the 218YCA data, 16-action grid, simple profit reward,
and safety masks.  The single learning change is VecNormalize(norm_obs=True,
norm_reward=False).  A resource fix releases each DSSAT environment's own
atexit callback after close, based on the 219YCA diagnosis.
"""
from __future__ import annotations

import argparse
import atexit
import copy
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from collections import Counter, deque
from pathlib import Path
from typing import Any

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import gymnasium as gym
import numpy as np
import pandas as pd
import psutil
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import audit_218YCA_action_counterfactual_5k as ref
import audit_219YCA_training_regression as diag
import ppo_safe_rendering


TASK_ID = "219YCA"
TASK_NAME = "yca_lowIC_10y_six_weather_normobs_simple_profit_pilot"
CONFIG_PATH = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
PROMPT_PATH = ROOT / "prompts" / "2026-09-06_219YCA_post218_diagnosis_and_short_pilot.md"
DEFAULT_OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC_PATH = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
STATION = "YCA"
SITE = "YC"
I_LEVELS = [0.0, 15.0, 30.0, 45.0]
N_LEVELS = [0.0, 40.0, 80.0, 120.0]
WATER_N_PREFIXES = ("sw_", "swfac", "nstres", "no3", "nh4", "sno3", "snh4")

ACTIVE_ENV: "ReleasedBalancedRandomYearEnv | None" = None
METRICS_JSONL: Path | None = None
EPISODES_JSONL: Path | None = None
RUN_STARTED = 0.0
MEMORY_STOP_MB = 2048.0

torch.set_num_threads(1)


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return rel(value) if value.is_absolute() and value.resolve().is_relative_to(ROOT) else str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_value), encoding="utf-8")


def append_jsonl(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({k: json_value(v) for k, v in row.items()}, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rss_mb() -> float:
    return float(psutil.Process().memory_info().rss / 1048576.0)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict[str, Any]:
    cfg = load_json(path)
    if cfg["task_id"] != TASK_ID or cfg["task_name"] != TASK_NAME:
        raise ValueError("219YCA config identity mismatch")
    if cfg["station_code"] != STATION or cfg["site"] != SITE:
        raise ValueError("219YCA is fixed to YCA/YC")
    if list(map(float, cfg["actions"]["irrigation_levels_mm"])) != I_LEVELS:
        raise ValueError("219YCA must keep the 16-action irrigation grid")
    if list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"])) != N_LEVELS:
        raise ValueError("219YCA must keep the 16-action nitrogen grid")
    obs = cfg["observation_contract"]
    if not bool(obs.get("normalization_enabled")) or bool(obs.get("weather_forecast_enabled")):
        raise ValueError("219YCA must use norm_obs only, with no forecast features")
    return cfg


def release_env(env: Any) -> None:
    raw = None
    try:
        raw = env.unwrapped
    except Exception:
        raw = None
    try:
        env.close()
    finally:
        callback = getattr(raw, "_cleanup_process", None)
        if callback is not None:
            try:
                atexit.unregister(callback)
            except ValueError:
                pass


def configure_runtime(out: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], pd.DataFrame]:
    source_contract, train_config, env_config = ref.build_env()
    env_config = copy.deepcopy(env_config)
    env_config.setdefault("paths", {})["output_root"] = rel(out)
    ref.runner.base03222.base.StressAwareDiscreteWrapper = ref.ProjectProfitDiscreteWrapper
    ref.ppo_safe_rendering.MULTISITE_INPUT_ROOT = ROOT / source_contract["input_root"]
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = ROOT / source_contract["input_root"]
    manifest = pd.read_csv(ref.MANIFEST_PATH, keep_default_na=False)
    split = ref.runner.base.make_split(manifest)
    split = split[split["station_code"].astype(str).eq(STATION)].copy()
    split["year"] = pd.to_numeric(split["year"], errors="raise").astype(int)
    split["source_year"] = pd.to_numeric(split["source_year"], errors="raise").astype(int)
    return source_contract, train_config, env_config, split


def make_env(config: dict[str, Any], env_config: dict[str, Any], year: int, seed: int, tag: str, evaluation: bool):
    return ref.runner.base03222.base.make_env(config, env_config, STATION, int(year), int(seed), tag, evaluation=evaluation)


class ReleasedBalancedRandomYearEnv(gym.Env):
    def __init__(
        self,
        config: dict[str, Any],
        env_config: dict[str, Any],
        years: list[int],
        scenario_meta: dict[int, dict[str, Any]],
        seed: int,
        cache_limit: int,
    ) -> None:
        super().__init__()
        global ACTIVE_ENV
        self.config = config
        self.env_config = env_config
        self.years = [int(y) for y in years]
        self.scenario_meta = scenario_meta
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.cache_limit = int(cache_limit)
        self.envs: dict[int, gym.Env] = {}
        self.current_year: int | None = None
        self.queue: deque[int] = deque()
        self.cycle_index = 0
        self.reset_counts: Counter[int] = Counter()
        self.switch_log: list[dict[str, Any]] = []
        self.action_counts: Counter[int] = Counter()
        self.safe_pair_counts: Counter[str] = Counter()
        self.training_step_count = 0
        self.env_create_count = 0
        self.env_eviction_count = 0
        self.peak_cached_envs = 0
        self.completed_episodes = 0
        self.recent_episode_rewards: deque[float] = deque(maxlen=20)
        self.recent_episode_lengths: deque[int] = deque(maxlen=20)
        self.recent_episode_irrigation: deque[float] = deque(maxlen=20)
        self.recent_episode_n: deque[float] = deque(maxlen=20)
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_irrigation = 0.0
        self._episode_n = 0.0
        first = self._env_for(self.years[0])
        self.action_space = first.action_space
        self.observation_space = first.observation_space
        self.metadata = getattr(first, "metadata", {})
        ACTIVE_ENV = self

    def _release_cached_env(self, year: int, env: gym.Env) -> None:
        release_env(env)
        self.env_eviction_count += 1

    def _env_for(self, year: int):
        year = int(year)
        if year in self.envs:
            env = self.envs.pop(year)
            self.envs[year] = env
            return env
        while len(self.envs) >= self.cache_limit:
            old_year = next(iter(self.envs))
            old_env = self.envs.pop(old_year)
            self._release_cached_env(old_year, old_env)
        env = make_env(self.config, self.env_config, year, self.seed, f"{STATION}_{year}_{TASK_ID}_train_seed{self.seed}", False)
        self.envs[year] = env
        self.env_create_count += 1
        self.peak_cached_envs = max(self.peak_cached_envs, len(self.envs))
        return env

    @property
    def current_env(self):
        if self.current_year is None:
            self.current_year = self.years[0]
        return self._env_for(self.current_year)

    def _refill_queue(self) -> None:
        source_years = sorted({int(item["source_year"]) for item in self.scenario_meta.values()})
        variants = sorted({str(item["weather_variant"]) for item in self.scenario_meta.values()})
        lookup = {
            (int(item["source_year"]), str(item["weather_variant"])): int(year)
            for year, item in self.scenario_meta.items()
        }
        source_order = [int(x) for x in self.rng.permutation(source_years)]
        variant_order = [str(x) for x in self.rng.permutation(variants)]
        planned: list[int] = []
        for round_index in range(len(variant_order)):
            for source_index, source_year in enumerate(source_order):
                variant = variant_order[(source_index + round_index) % len(variant_order)]
                planned.append(lookup[(source_year, variant)])
        if len(planned) != len(self.years) or len(set(planned)) != len(self.years):
            raise RuntimeError("balanced sampler failed to cover each scenario once")
        self.queue.extend(planned)
        self.cycle_index += 1

    def reset(self, *args, **kwargs):
        if not self.queue:
            self._refill_queue()
        year = int(self.queue.popleft())
        self.current_year = year
        self.reset_counts[year] += 1
        self.switch_log.append(
            {
                "episode_index": int(sum(self.reset_counts.values())),
                "year": year,
                "source_year": self.scenario_meta.get(year, {}).get("source_year"),
                "weather_variant": self.scenario_meta.get(year, {}).get("weather_variant"),
                "sampling_cycle_index": self.cycle_index,
            }
        )
        obs, info = self.current_env.reset(*args, **kwargs)
        info = dict(info or {})
        info["active_year"] = year
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_irrigation = 0.0
        self._episode_n = 0.0
        return obs, info

    def step(self, action):
        index = int(np.asarray(action).reshape(-1)[0])
        obs, reward, terminated, truncated, info = self.current_env.step(index)
        action_info = dict(getattr(self.current_env, "last_action_info", {}) or {})
        safe_i = finite(action_info.get("safe_action_amir"), 0.0)
        safe_n = finite(action_info.get("safe_action_anfer"), 0.0)
        self.training_step_count += 1
        self.action_counts[index] += 1
        self.safe_pair_counts[f"I{safe_i:g}/N{safe_n:g}"] += 1
        self._episode_reward += finite(reward, 0.0)
        self._episode_length += 1
        self._episode_irrigation += safe_i
        self._episode_n += safe_n
        if bool(terminated or truncated):
            self.completed_episodes += 1
            self.recent_episode_rewards.append(self._episode_reward)
            self.recent_episode_lengths.append(self._episode_length)
            self.recent_episode_irrigation.append(self._episode_irrigation)
            self.recent_episode_n.append(self._episode_n)
            append_jsonl(
                EPISODES_JSONL,
                {
                    "seed": self.seed,
                    "episode_index": self.completed_episodes,
                    "pseudo_year": int(self.current_year),
                    "source_year": self.scenario_meta.get(int(self.current_year), {}).get("source_year"),
                    "weather_variant": self.scenario_meta.get(int(self.current_year), {}).get("weather_variant"),
                    "episode_reward": self._episode_reward,
                    "episode_length": self._episode_length,
                    "total_irrigation_mm": self._episode_irrigation,
                    "total_n_kg_ha": self._episode_n,
                },
            )
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self.current_env.action_masks()

    @property
    def last_action_info(self) -> dict[str, Any]:
        return dict(getattr(self.current_env, "last_action_info", {}) or {})

    def snapshot(self) -> dict[str, Any]:
        total_actions = int(sum(self.action_counts.values()))
        probs = np.asarray(list(self.action_counts.values()), dtype=float)
        if probs.size and probs.sum() > 0:
            probs = probs / probs.sum()
            entropy = float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum())
            normalized_entropy = entropy / math.log(max(2, int(self.action_space.n)))
        else:
            normalized_entropy = 0.0
        covered = {int(year) for year, count in self.reset_counts.items() if count > 0}
        return {
            "training_steps_seen": self.training_step_count,
            "episodes_started": int(sum(self.reset_counts.values())),
            "episodes_completed": self.completed_episodes,
            "scenario_coverage": len(covered),
            "source_year_coverage": len({self.scenario_meta[y]["source_year"] for y in covered if y in self.scenario_meta}),
            "variant_coverage": len({self.scenario_meta[y]["weather_variant"] for y in covered if y in self.scenario_meta}),
            "requested_noop_rate": finite(self.action_counts.get(0, 0) / total_actions if total_actions else 0.0),
            "requested_action_entropy_normalized": normalized_entropy,
            "unique_requested_actions": len(self.action_counts),
            "unique_safe_action_pairs": len(self.safe_pair_counts),
            "mean_recent_episode_reward": finite(np.mean(self.recent_episode_rewards), 0.0) if self.recent_episode_rewards else 0.0,
            "mean_recent_episode_length": finite(np.mean(self.recent_episode_lengths), 0.0) if self.recent_episode_lengths else 0.0,
            "mean_recent_episode_irrigation_mm": finite(np.mean(self.recent_episode_irrigation), 0.0) if self.recent_episode_irrigation else 0.0,
            "mean_recent_episode_n_kg_ha": finite(np.mean(self.recent_episode_n), 0.0) if self.recent_episode_n else 0.0,
            "env_cache_size": len(self.envs),
            "peak_cached_envs": self.peak_cached_envs,
            "env_create_count": self.env_create_count,
            "env_eviction_count": self.env_eviction_count,
        }

    def close(self) -> None:
        for year, env in list(self.envs.items()):
            release_env(env)
            self.envs.pop(year, None)

    def __getattr__(self, name: str):
        return getattr(self.current_env, name)


class InstrumentedMaskablePPO(MaskablePPO):
    def train(self) -> None:
        super().train()
        values = dict(getattr(self.logger, "name_to_value", {}) or {})
        row = {
            "elapsed_seconds": time.perf_counter() - RUN_STARTED,
            "num_timesteps": int(self.num_timesteps),
            "n_updates": int(getattr(self, "_n_updates", 0)),
            "learning_rate": values.get("train/learning_rate"),
            "approx_kl": values.get("train/approx_kl"),
            "clip_fraction": values.get("train/clip_fraction"),
            "entropy_loss": values.get("train/entropy_loss"),
            "explained_variance": values.get("train/explained_variance"),
            "policy_gradient_loss": values.get("train/policy_gradient_loss"),
            "value_loss": values.get("train/value_loss"),
            "loss": values.get("train/loss"),
            "rss_mb": rss_mb(),
        }
        if ACTIVE_ENV is not None:
            row.update(ACTIVE_ENV.snapshot())
        append_jsonl(METRICS_JSONL, row)
        print(
            f"[{TASK_ID}][train] step={row['num_timesteps']} kl={finite(row['approx_kl'], float('nan')):.4g} "
            f"ev={finite(row['explained_variance'], float('nan')):.4g} "
            f"cov={int(row.get('scenario_coverage', 0))}/60 rss={row['rss_mb']:.1f}MB",
            flush=True,
        )
        if row["rss_mb"] >= MEMORY_STOP_MB:
            raise MemoryError(f"{TASK_ID} memory stop gate: rss={row['rss_mb']:.1f} MB >= {MEMORY_STOP_MB:.1f} MB")


def ppo_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kwargs = ref.runner.base03222.base.ppo_kwargs(config)
    return kwargs


def normalize_obs(vecnorm: VecNormalize, obs: np.ndarray) -> np.ndarray:
    return np.asarray(vecnorm.normalize_obs(np.asarray(obs, dtype=np.float32).reshape(1, -1)), dtype=np.float32)


def action_signature(daily: pd.DataFrame) -> str:
    events = daily[pd.to_numeric(daily["irrigation"], errors="coerce").fillna(0.0) + pd.to_numeric(daily["nitrogen"], errors="coerce").fillna(0.0) > 0]
    return ";".join(f"DAP{int(r.dap)}:I{finite(r.irrigation):g}/N{finite(r.nitrogen):g}" for r in events.itertuples(index=False))


def evaluate_checkpoint(
    model: MaskablePPO,
    vecnorm: VecNormalize,
    config: dict[str, Any],
    env_config: dict[str, Any],
    out: Path,
    seed: int,
    checkpoint: int,
    years: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    all_raw: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_meta: list[dict[str, Any]] = []
    model.policy.set_training_mode(False)
    for year in years:
        env = make_env(config, env_config, year, seed, f"{TASK_ID}_eval_seed{seed}_ck{checkpoint}_{year}", True)
        try:
            obs, _ = env.reset()
            names = diag.feature_names(env, obs)
            grid = np.asarray([[raw["amir"], raw["anfer"]] for raw in env.grid], dtype=float)
            rows: list[dict[str, Any]] = []
            done = False
            for step_index in range(260):
                dap = int(env._dap())
                mask = np.asarray(env.action_masks(), dtype=bool)
                raw_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                norm_obs = normalize_obs(vecnorm, raw_obs)
                probs, values, sat1, sat2 = diag.forward(model, norm_obs, mask.reshape(1, -1))
                probs = probs[0]
                action = int(np.argmax(probs))
                if not bool(mask[action]):
                    raise RuntimeError(f"illegal action selected: seed={seed}, ck={checkpoint}, year={year}, dap={dap}")
                legal_count = int(mask.sum())
                entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum())
                obs, reward, terminated, truncated, _ = env.step(action)
                info = dict(env.last_action_info)
                record = {
                    "seed": seed,
                    "checkpoint_step": checkpoint,
                    "year": year,
                    "step_index": step_index,
                    "dap": dap,
                    "action": action,
                    "legal_count": legal_count,
                    "top_probability": float(probs.max()),
                    "top_gap": float(np.sort(probs)[-1] - np.sort(probs)[-2]) if probs.size > 1 else np.nan,
                    "entropy_normalized": entropy / math.log(legal_count) if legal_count > 1 else np.nan,
                    "value": float(values[0]),
                    "first_tanh_saturated": float(sat1[0]),
                    "second_tanh_saturated": float(sat2[0]),
                    "irrigation": finite(info.get("safe_action_amir"), 0.0),
                    "nitrogen": finite(info.get("safe_action_anfer"), 0.0),
                    "requested_i": float(grid[action, 0]),
                    "requested_n": float(grid[action, 1]),
                    "forced_noop": bool(info.get("mask_forced_noop", False)),
                    "safety_trigger": str(info.get("action_safety_triggered", "")),
                    "reward": float(reward),
                }
                record.update({f"raw_obs_{name}": float(value) for name, value in zip(names, raw_obs)})
                record.update({f"norm_obs_{name}": float(value) for name, value in zip(names, norm_obs.reshape(-1))})
                rows.append(record)
                all_raw.append(raw_obs.copy())
                all_masks.append(mask.copy())
                all_meta.append({"seed": seed, "checkpoint_step": checkpoint, "year": year, "dap": dap, "step_index": step_index})
                done = bool(terminated or truncated)
                if done:
                    break
            if not done:
                raise RuntimeError(f"episode did not finish: seed={seed}, ck={checkpoint}, year={year}")
            daily = pd.DataFrame(rows)
            daily["return_G"] = daily.reward.iloc[::-1].cumsum().iloc[::-1]
            daily["G_minus_V"] = daily.return_G - daily.value
            final_yield = finite(env.last_obs_dict.get("grnwt"), np.nan)
            snapshot = out / "snapshots" / f"seed{seed}" / f"ck{checkpoint}" / str(year)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(ref.snapshot_from_env(env), snapshot)
            metrics = ref.FIG055.baseline.metrics_from_snapshot(snapshot, final_yield)
            irrigation = float(metrics["actual_irrigation_mm"])
            nitrogen = float(metrics["actual_nitrogen_kg_ha"])
            profit = final_yield - 1.1 * irrigation - 1.58 * nitrogen
            reward_identity_error = float(daily.reward.sum() - profit * 0.001)
            if abs(reward_identity_error) > 1e-4:
                raise RuntimeError(f"reward identity failed: {reward_identity_error}")
            if daily.forced_noop.any() or daily.safety_trigger.astype(str).str.len().gt(0).any():
                raise RuntimeError(f"forced noop or safety clipping in eval: seed={seed}, ck={checkpoint}, year={year}")
            out_daily = out / "daily_outputs" / STATION / f"{STATION}_{year}_seed{seed}_ckpt{checkpoint}_daily.csv"
            out_daily.parent.mkdir(parents=True, exist_ok=True)
            daily.to_csv(out_daily, index=False, encoding="utf-8-sig")
            episode_rows.append(
                {
                    "seed": seed,
                    "checkpoint_step": checkpoint,
                    "year": year,
                    "final_grnwt": final_yield,
                    "total_irrigation": irrigation,
                    "total_n": nitrogen,
                    "simple_profit": profit,
                    "WP_ET_kg_m3": float(metrics["WP_ET_kg_m3"]),
                    "PFP_N": float(metrics["PFP_N_kg_kg"]) if nitrogen > 0 and pd.notna(metrics["PFP_N_kg_kg"]) else np.nan,
                    "irrigation_event_count": int((daily.irrigation > 0).sum()),
                    "n_event_count": int((daily.nitrogen > 0).sum()),
                    "positive_action_rows": int((daily.irrigation.abs() + daily.nitrogen.abs() > 0).sum()),
                    "unique_nonzero_action_pairs": int(daily.loc[daily.irrigation.abs() + daily.nitrogen.abs() > 0, ["irrigation", "nitrogen"]].drop_duplicates().shape[0]),
                    "action_signature": action_signature(daily),
                    "reward_identity_error": reward_identity_error,
                    "daily_csv_path": rel(out_daily),
                    "snapshot_path": rel(snapshot),
                }
            )
            state_rows.extend(daily.to_dict(orient="records"))
            print(
                f"[{TASK_ID}][eval] seed={seed} ck={checkpoint} year={year} "
                f"yield={final_yield:.1f} I={irrigation:g} N={nitrogen:g}",
                flush=True,
            )
        finally:
            release_env(env)
            gc.collect()
    sensitivity = sensitivity_check(model, vecnorm, pd.DataFrame(all_meta), np.stack(all_raw), np.stack(all_masks), names)
    return pd.DataFrame(episode_rows), pd.DataFrame(state_rows), sensitivity


def sensitivity_check(
    model: MaskablePPO,
    vecnorm: VecNormalize,
    meta: pd.DataFrame,
    raw_x: np.ndarray,
    masks: np.ndarray,
    names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    legal = masks.sum(axis=1) >= 2
    if not bool(legal.any()):
        return pd.DataFrame()
    base_meta = meta.loc[legal].reset_index(drop=True)
    base_raw = raw_x[legal].copy()
    base_masks = masks[legal].copy()
    base_norm = np.asarray(vecnorm.normalize_obs(base_raw.copy()), dtype=np.float32)
    base_p, _, _, _ = diag.forward(model, base_norm, base_masks)
    donor = np.arange(len(base_meta))
    for _, idx in base_meta.groupby("dap").groups.items():
        idx_arr = np.asarray(list(idx))
        for k in idx_arr:
            candidates = idx_arr[base_meta.loc[idx_arr, "year"].to_numpy() != base_meta.loc[k, "year"]]
            if len(candidates):
                donor[k] = int(candidates[0])
    eligible = donor != np.arange(len(donor))
    water_n_idx = [i for i, name in enumerate(names) if name.startswith(WATER_N_PREFIXES)]
    for mode in ("all_observed_state_same_dap", "water_n_features_only"):
        altered = base_raw.copy()
        if mode == "all_observed_state_same_dap":
            altered = altered[donor]
        elif water_n_idx:
            altered[:, water_n_idx] = altered[donor][:, water_n_idx]
        else:
            continue
        changed_input = np.any(altered != base_raw, axis=1) & eligible
        altered_norm = np.asarray(vecnorm.normalize_obs(altered), dtype=np.float32)
        alt_p, _, _, _ = diag.forward(model, altered_norm, base_masks)
        changed_action = base_p.argmax(axis=1) != alt_p.argmax(axis=1)
        l1 = np.abs(base_p - alt_p).sum(axis=1)
        rows.append(
            {
                "seed": int(base_meta["seed"].iloc[0]) if len(base_meta) else np.nan,
                "checkpoint_step": int(base_meta["checkpoint_step"].iloc[0]) if len(base_meta) else np.nan,
                "intervention": mode,
                "eligible_rows": int(changed_input.sum()),
                "changed_action_rows": int(changed_action[changed_input].sum()),
                "action_change_rate": float(changed_action[changed_input].mean()) if changed_input.any() else np.nan,
                "mean_probability_l1": float(l1[changed_input].mean()) if changed_input.any() else np.nan,
                "max_probability_l1": float(l1[changed_input].max()) if changed_input.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_checkpoint(episode_df: pd.DataFrame, state_df: pd.DataFrame, sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (seed, ck), group in episode_df.groupby(["seed", "checkpoint_step"]):
        states = state_df[(state_df.seed.eq(seed)) & (state_df.checkpoint_step.eq(ck))]
        choices = states[states.legal_count >= 2]
        sens = sensitivity_df[(sensitivity_df.seed.eq(seed)) & (sensitivity_df.checkpoint_step.eq(ck))]
        water = sens[sens.intervention.eq("water_n_features_only")]
        rows.append(
            {
                "seed": int(seed),
                "checkpoint_step": int(ck),
                "validation_years": int(len(group)),
                "mean_final_grnwt": finite(group.final_grnwt.mean(), np.nan),
                "mean_simple_profit": finite(group.simple_profit.mean(), np.nan),
                "mean_WP_ET_kg_m3": finite(group.WP_ET_kg_m3.mean(), np.nan),
                "mean_PFP_N": finite(group.PFP_N.mean(), np.nan),
                "mean_total_irrigation": finite(group.total_irrigation.mean(), np.nan),
                "mean_total_n": finite(group.total_n.mean(), np.nan),
                "mean_irrigation_events": finite(group.irrigation_event_count.mean(), np.nan),
                "mean_n_events": finite(group.n_event_count.mean(), np.nan),
                "unique_action_signatures": int(group.action_signature.nunique()),
                "unique_total_irrigation": int(group.total_irrigation.nunique()),
                "unique_total_n": int(group.total_n.nunique()),
                "mean_entropy": finite(choices.entropy_normalized.mean(), np.nan),
                "mean_top_probability": finite(choices.top_probability.mean(), np.nan),
                "first_layer_saturation": finite(choices.first_tanh_saturated.mean(), np.nan),
                "critic_G_RMSE": float(np.sqrt(np.square(choices.G_minus_V).mean())) if len(choices) else np.nan,
                "water_n_action_change_rate": finite(water.action_change_rate.iloc[0], np.nan) if len(water) else np.nan,
                "water_n_probability_l1": finite(water.mean_probability_l1.iloc[0], np.nan) if len(water) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["seed", "checkpoint_step"]).reset_index(drop=True)


def save_training_assets(model: MaskablePPO, vecnorm: VecNormalize, out: Path, seed: int, checkpoint: int) -> tuple[Path, Path]:
    model_path = out / "models" / STATION / f"{STATION}_normobs_simple_profit_seed{seed}_ckpt{checkpoint}.zip"
    norm_path = out / "models" / STATION / f"{STATION}_normobs_simple_profit_seed{seed}_ckpt{checkpoint}_vecnormalize.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    vecnorm.save(str(norm_path))
    return model_path, norm_path


def train_seed(
    cfg: dict[str, Any],
    train_config: dict[str, Any],
    env_config: dict[str, Any],
    split: pd.DataFrame,
    out: Path,
    seed: int,
    targets: list[int],
    smoke_target: int,
) -> dict[str, Any]:
    global ACTIVE_ENV, METRICS_JSONL, EPISODES_JSONL, MEMORY_STOP_MB
    runtime = cfg["runtime_safety"]
    MEMORY_STOP_MB = float(runtime["memory_stop_mb"])
    train_split = split[split["split"].eq("train")].copy()
    years = sorted(pd.to_numeric(train_split["year"], errors="raise").astype(int).tolist())
    scenario_meta = {
        int(row.year): {"source_year": int(row.source_year), "weather_variant": str(row.weather_variant)}
        for row in train_split.itertuples(index=False)
    }
    train_env = ReleasedBalancedRandomYearEnv(
        train_config,
        env_config,
        years,
        scenario_meta,
        seed=seed,
        cache_limit=int(runtime["environment_cache_limit"]),
    )
    ACTIVE_ENV = train_env
    vecenv = DummyVecEnv([lambda: train_env])
    vecnorm = VecNormalize(
        vecenv,
        norm_obs=True,
        norm_reward=False,
        clip_obs=float(cfg["observation_contract"]["clip_obs"]),
        training=True,
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = InstrumentedMaskablePPO(
        "MlpPolicy",
        vecnorm,
        verbose=0,
        seed=seed,
        tensorboard_log=str(out / "tensorboard" / STATION / f"seed{seed}"),
        device="cpu",
        **ppo_kwargs(train_config),
    )
    rows: list[dict[str, Any]] = []
    episodes: list[pd.DataFrame] = []
    states: list[pd.DataFrame] = []
    sensitivities: list[pd.DataFrame] = []
    status = "completed"
    error = ""
    try:
        previous = 0
        for target in targets:
            delta = int(target) - int(previous)
            if delta <= 0:
                raise ValueError(f"targets must be increasing: {targets}")
            vecnorm.training = True
            model.learn(total_timesteps=delta, reset_num_timesteps=(previous == 0), progress_bar=False)
            previous = int(model.num_timesteps)
            model_path, norm_path = save_training_assets(model, vecnorm, out, seed, previous)
            eval_years = list(map(int, cfg["pilot"]["smoke_validation_years"])) if previous == smoke_target else list(map(int, cfg["pilot"]["full_validation_years"]))
            ep, st, sen = evaluate_checkpoint(model, vecnorm, train_config, env_config, out, seed, previous, eval_years)
            episodes.append(ep)
            states.append(st)
            sensitivities.append(sen)
            rows.append(
                {
                    "seed": seed,
                    "checkpoint_step": previous,
                    "target_step": target,
                    "model_path": rel(model_path),
                    "model_sha256": sha256_file(model_path),
                    "vecnormalize_path": rel(norm_path),
                    "vecnormalize_sha256": sha256_file(norm_path),
                    "eval_years": ",".join(map(str, eval_years)),
                    "rss_mb": rss_mb(),
                    "status": "completed",
                }
            )
            smoke_ok = bool(
                len(ep) == len(eval_years)
                and ep.total_irrigation.notna().all()
                and ep.total_n.notna().all()
                and not st.forced_noop.any()
                and not st.safety_trigger.astype(str).str.len().gt(0).any()
                and rss_mb() < float(runtime["memory_stop_mb"])
            )
            if previous == smoke_target and not smoke_ok:
                status = "stopped_after_smoke_gate"
                break
    except Exception:
        status = "failed"
        error = traceback.format_exc()
        print(error, flush=True)
    finally:
        try:
            vecenv.close()
        except Exception:
            pass
        train_env.close()
        ACTIVE_ENV = None
        gc.collect()
    pd.DataFrame(train_env.switch_log).to_csv(out / "logs" / f"{TASK_ID}_seed{seed}_training_scenario_switch_log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{"seed": seed, "year": year, "episode_count": int(train_env.reset_counts.get(year, 0))} for year in years]
    ).to_csv(out / "logs" / f"{TASK_ID}_seed{seed}_training_scenario_reset_counts.csv", index=False, encoding="utf-8-sig")
    return {
        "seed": seed,
        "status": status,
        "error": error[-10000:] if error else "",
        "checkpoints": rows,
        "episodes": pd.concat(episodes, ignore_index=True) if episodes else pd.DataFrame(),
        "states": pd.concat(states, ignore_index=True) if states else pd.DataFrame(),
        "sensitivity": pd.concat(sensitivities, ignore_index=True) if sensitivities else pd.DataFrame(),
    }


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "No rows."
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    lines = [header, sep]
    for row in work.to_numpy().tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def record_path(out: Path) -> Path:
    return out / f"{TASK_ID}_record.md"


def write_record(out: Path, cfg: dict[str, Any], diagnosis: dict[str, Any], summary: pd.DataFrame, result: dict[str, Any]) -> None:
    lines = [
        f"# {TASK_ID} YC norm-observation PPO pilot record",
        "",
        "## Boundary",
        "",
        "- Site: YCA/YC lowIC.",
        "- Data: same 10-year x 6 RAIN-augmented weather bank as 218YCA.",
        "- Action grid unchanged: irrigation [0, 15, 30, 45] mm x N [0, 40, 80, 120] kg/ha.",
        "- Reward unchanged: grain yield - 1.1 x irrigation - 1.58 x N, scaled by 0.001.",
        "- Single learning change: VecNormalize observation normalization; no forecast features.",
        "- Resource fix: release each closed DSSAT environment's own atexit callback.",
        "- 2014-2023 is treated as development validation here, not a pristine final test.",
        "",
        "## Why This Change",
        "",
        "The diagnosis found very high first-layer saturation and near-zero action response to water/N-only state swaps under fixed masks.",
        "This is evidence for testing observation normalization, not causal proof that normalization alone will solve the policy problem.",
        "",
        "```json",
        json.dumps(
            {
                "diagnosis_result": cfg["diagnosis_run"],
                "218YCA_reward_identity_passed": diagnosis.get("reward_identity_passed"),
                "218YCA_matched_saved_actions": diagnosis.get("matched_saved_actions"),
                "218YCA_sensitivity": diagnosis.get("sensitivity"),
                "218YCA_memory": diagnosis.get("memory"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## Checkpoint Summary",
        "",
        md_table(summary, 80),
        "",
        "## Result Manifest",
        "",
        "```json",
        json.dumps(result, ensure_ascii=False, indent=2),
        "```",
    ]
    path = record_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg_path: Path, out: Path, seeds: list[int], targets: list[int], dry_run: bool = False) -> dict[str, Any]:
    global METRICS_JSONL, EPISODES_JSONL, RUN_STARTED
    cfg = load_config(cfg_path)
    if not out.resolve().is_relative_to(ROOT):
        raise ValueError("output root must be inside the project")
    if out.exists() and not dry_run:
        raise FileExistsError(f"refuse to overwrite existing output: {rel(out)}")
    diagnosis_path = ROOT / cfg["diagnosis_run"] / "diagnosis_result.json"
    diagnosis = load_json(diagnosis_path)
    source_contract, train_config, env_config, split = configure_runtime(out)
    train_split = split[split["split"].eq("train")]
    issues: list[str] = []
    if len(train_split) != int(cfg["scope"]["expected_train_scenarios"]):
        issues.append(f"expected 60 training scenarios, got {len(train_split)}")
    if sorted(train_split["source_year"].astype(int).unique().tolist()) != list(map(int, cfg["scope"]["train_source_years"])):
        issues.append("train source years mismatch")
    if sorted(train_split["weather_variant"].astype(str).unique().tolist()) != sorted(cfg["scope"]["train_weather_variants"]):
        issues.append("weather variants mismatch")
    if not bool(diagnosis.get("reward_identity_passed")) or not bool(diagnosis.get("matched_saved_actions")):
        issues.append("diagnosis did not verify source replay identity")
    preflight = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "seeds": seeds,
        "targets": targets,
        "source_contract": source_contract,
        "train_scenarios": int(len(train_split)),
        "diagnosis": cfg["diagnosis_run"],
        "normalization": cfg["observation_contract"],
        "issues": issues,
        "next_step_allowed": not issues,
    }
    if dry_run:
        return {**preflight, "mode": "dry_run"}
    if issues:
        raise RuntimeError(json.dumps(preflight, ensure_ascii=False, indent=2))
    for sub in ("configs", "models", "logs", "daily_outputs", "snapshots", "tables", "tensorboard"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, out / "configs" / cfg_path.name)
    if PROMPT_PATH.exists():
        shutil.copy2(PROMPT_PATH, out / "configs" / PROMPT_PATH.name)
    shutil.copy2(diagnosis_path, out / "configs" / diagnosis_path.name)
    split.to_csv(out / "configs" / f"{TASK_ID}_training_validation_split.csv", index=False, encoding="utf-8-sig")
    write_json(out / "configs" / f"{TASK_ID}_preflight.json", preflight)
    effective = {
        "seed_values": seeds,
        "actual_checkpoint_steps": targets,
        "ppo_kwargs": ppo_kwargs(train_config),
        "action_safety": train_config.get("action_safety"),
        "reward": train_config.get("reward"),
        "actions": cfg["actions"],
        "single_algorithm_change": cfg["ppo"]["single_algorithm_change"],
        "resource_fix": cfg["runtime_safety"]["dssat_atexit_callback_release_after_close"],
    }
    write_json(out / "configs" / f"{TASK_ID}_effective_training_contract.json", effective)
    METRICS_JSONL = out / "logs" / f"{TASK_ID}_training_update_metrics.jsonl"
    EPISODES_JSONL = out / "logs" / f"{TASK_ID}_training_episode_metrics.jsonl"
    RUN_STARTED = time.perf_counter()
    seed_results = []
    checkpoint_rows: list[dict[str, Any]] = []
    episode_frames: list[pd.DataFrame] = []
    state_frames: list[pd.DataFrame] = []
    sensitivity_frames: list[pd.DataFrame] = []
    for seed in seeds:
        print(f"[{TASK_ID}] starting seed={seed}", flush=True)
        result = train_seed(cfg, train_config, env_config, split, out, seed, targets, int(cfg["pilot"]["smoke_checkpoint_step"]))
        seed_results.append({k: v for k, v in result.items() if k not in {"episodes", "states", "sensitivity"}})
        checkpoint_rows.extend(result["checkpoints"])
        if not result["episodes"].empty:
            episode_frames.append(result["episodes"])
        if not result["states"].empty:
            state_frames.append(result["states"])
        if not result["sensitivity"].empty:
            sensitivity_frames.append(result["sensitivity"])
        if result["status"] == "failed":
            break
    inventory = pd.DataFrame(checkpoint_rows)
    episode_df = pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame()
    state_df = pd.concat(state_frames, ignore_index=True) if state_frames else pd.DataFrame()
    sensitivity_df = pd.concat(sensitivity_frames, ignore_index=True) if sensitivity_frames else pd.DataFrame()
    summary = summarize_checkpoint(episode_df, state_df, sensitivity_df) if not episode_df.empty else pd.DataFrame()
    inventory.to_csv(out / "tables" / f"{TASK_ID}_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
    episode_df.to_csv(out / "tables" / f"{TASK_ID}_development_validation_episode_metrics.csv", index=False, encoding="utf-8-sig")
    state_df.to_csv(out / "tables" / f"{TASK_ID}_development_validation_decision_states.csv", index=False, encoding="utf-8-sig")
    sensitivity_df.to_csv(out / "tables" / f"{TASK_ID}_development_validation_state_sensitivity.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "tables" / f"{TASK_ID}_checkpoint_summary.csv", index=False, encoding="utf-8-sig")
    if METRICS_JSONL and METRICS_JSONL.exists():
        rows = [json.loads(line) for line in METRICS_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
        pd.DataFrame(rows).to_csv(out / "logs" / f"{TASK_ID}_training_update_metrics.csv", index=False, encoding="utf-8-sig")
    if EPISODES_JSONL and EPISODES_JSONL.exists():
        rows = [json.loads(line) for line in EPISODES_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
        pd.DataFrame(rows).to_csv(out / "logs" / f"{TASK_ID}_training_episode_metrics.csv", index=False, encoding="utf-8-sig")
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "status": "completed" if all(item["status"] == "completed" for item in seed_results) else "partial_or_failed",
        "output_root": rel(out),
        "record_md": rel(record_path(out)),
        "elapsed_seconds": time.perf_counter() - RUN_STARTED,
        "seeds": seeds,
        "actual_checkpoint_steps": targets,
        "primary_final_checkpoint": int(cfg["pilot"]["primary_final_checkpoint"]),
        "preflight": rel(out / "configs" / f"{TASK_ID}_preflight.json"),
        "effective_training_contract": rel(out / "configs" / f"{TASK_ID}_effective_training_contract.json"),
        "checkpoint_inventory": rel(out / "tables" / f"{TASK_ID}_checkpoint_inventory.csv"),
        "checkpoint_summary": rel(out / "tables" / f"{TASK_ID}_checkpoint_summary.csv"),
        "episode_metrics": rel(out / "tables" / f"{TASK_ID}_development_validation_episode_metrics.csv"),
        "decision_states": rel(out / "tables" / f"{TASK_ID}_development_validation_decision_states.csv"),
        "state_sensitivity": rel(out / "tables" / f"{TASK_ID}_development_validation_state_sensitivity.csv"),
        "seed_results": seed_results,
    }
    write_json(out / f"{TASK_ID}_result.json", result)
    write_record(out, cfg, diagnosis, summary, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--targets", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg_for_defaults = load_config(args.config if args.config.is_absolute() else (ROOT / args.config))
    seeds = parse_csv_ints(args.seeds) if args.seeds else list(map(int, cfg_for_defaults["pilot"]["seeds"]))
    targets = parse_csv_ints(args.targets) if args.targets else list(map(int, cfg_for_defaults["pilot"]["actual_checkpoint_steps"]))
    out = args.out if args.out.is_absolute() else ROOT / args.out
    print(json.dumps(run(args.config if args.config.is_absolute() else (ROOT / args.config), out, seeds, targets, args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
