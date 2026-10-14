"""203HLA: paired HLA lowIC PPO with 10 years x 6 weather scenarios.

The shared 054/042/032 training and validation implementation is reused.  This
runner only supplies the 202HLA augmented training split and adds bounded-cache
runtime instrumentation so the 60-scenario experiment remains auditable and
memory safe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import shutil
import sys
import time
import traceback
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine


TASK_ID = "203HLA"
TASK_NAME = "hla_lowIC_10y_six_weather_aug_maskableppo"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
PROMPT = ROOT / "prompts" / "2026-08-24_hla_10y_six_weather_augmentation.md"
STATION = "HLA"
SITE = "HLA"
EXPECTED_VARIANTS = ["original", "early_dry", "mid_dry", "late_dry", "early_wet", "mid_wet"]
EXPECTED_I = [0.0, 15.0, 30.0, 45.0]
EXPECTED_N = [0.0, 40.0, 80.0, 120.0]

ACTIVE_ENV: "BoundedAuditedRandomYearEnv | None" = None
SCENARIO_META: dict[int, dict[str, Any]] = {}
METRICS_JSONL: Path | None = None
METRICS_TEXT: Path | None = None
EPISODES_JSONL: Path | None = None
RUN_STARTED = 0.0
MEMORY_WARNING_MB = 1536.0
MEMORY_STOP_MB = 2048.0
CACHE_LIMIT = 2


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if hasattr(value, "detach"):
        try:
            value = value.detach().cpu().item()
        except Exception:
            return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except Exception:
        return str(value)


def rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024.0 * 1024.0))
    except Exception:
        raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return raw / 1024.0 if raw > 1024.0 * 1024.0 else raw / (1024.0 * 1024.0)


def append_jsonl(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({key: json_value(value) for key, value in row.items()}, ensure_ascii=False) + "\n")


def emit_training_metric(row: dict[str, Any]) -> None:
    append_jsonl(METRICS_JSONL, row)
    line = (
        f"[{TASK_ID}][train] step={int(row.get('num_timesteps', 0))} "
        f"update={int(row.get('n_updates', 0))} "
        f"kl={finite(row.get('approx_kl'), float('nan')):.6g} "
        f"entropy_loss={finite(row.get('entropy_loss'), float('nan')):.6g} "
        f"value_loss={finite(row.get('value_loss'), float('nan')):.6g} "
        f"explained_var={finite(row.get('explained_variance'), float('nan')):.6g} "
        f"coverage={int(row.get('scenario_coverage', 0))}/60 "
        f"noop_rate={finite(row.get('requested_noop_rate'), 0.0):.3f} "
        f"rss_mb={finite(row.get('rss_mb'), 0.0):.1f}"
    )
    print(line, flush=True)
    if METRICS_TEXT is not None:
        METRICS_TEXT.parent.mkdir(parents=True, exist_ok=True)
        with METRICS_TEXT.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("task_id") != TASK_ID or cfg.get("task_name") != TASK_NAME:
        raise ValueError("203HLA task identity mismatch")
    if cfg.get("station_code") != STATION or cfg.get("site") != SITE:
        raise ValueError("203HLA is fixed to HLA")
    if cfg.get("reference_run") != "054_00_hla_lowIC_expanded_action_maskableppo":
        raise ValueError("203HLA must use 054_00 as the paired unaugmented reference")
    if list(map(float, cfg["actions"]["irrigation_levels_mm"])) != EXPECTED_I:
        raise ValueError("irrigation grid differs from 054_00")
    if list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"])) != EXPECTED_N:
        raise ValueError("nitrogen grid differs from 054_00")
    obs = cfg.get("observation_contract", {})
    if obs != {"base": "046_02_raw_observation", "normalization_enabled": False, "weather_forecast_enabled": False}:
        raise ValueError("observation contract differs from 054_00")
    return cfg


def output_root(cfg: dict[str, Any], smoke: bool) -> Path:
    suffix = "_smoke2k" if smoke else ""
    return ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}{suffix}"


def record_path(smoke: bool) -> Path:
    suffix = "_smoke2k" if smoke else ""
    return ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}{suffix}_record.md"


def read_manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / cfg["source_manifest"]
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = pd.read_csv(path, keep_default_na=False)
    selected = manifest[manifest["station_code"].astype(str).eq(STATION)].copy()
    for col in ["source_year", "pseudo_year"]:
        selected[col] = pd.to_numeric(selected[col], errors="raise").astype(int)
    return selected.sort_values(["split", "source_year", "pseudo_year"]).reset_index(drop=True)


def make_split(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        target = ROOT / str(row.target_weather)
        rows.append(
            {
                "station_code": STATION,
                "site": SITE,
                "station_name": SITE,
                "source_year": int(row.source_year),
                "year": int(row.pseudo_year),
                "split": str(row.split),
                "weather_variant": str(row.variant),
                "scenario_type": f"{TASK_ID}_{row.variant}",
                "weather_file": target.name,
                "target_weather": str(row.target_weather),
                "selected_for_train": str(row.split) == "train",
                "selected_for_eval": str(row.split) == "validation",
                "selection_reason": "203HLA_10y_x_6_weather_training" if str(row.split) == "train" else "203HLA_original_lowIC_validation",
                "rain_multiplier": finite(row.rain_multiplier, 1.0),
                "perturbation_window": str(row.window),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "year"]).reset_index(drop=True)


def code_provenance() -> dict[str, str]:
    paths = [
        ROOT / "src" / "run_sya_lowIC_binary_timing_maskableppo_042_10.py",
        ROOT / "src" / "run_five_site_half_split_stress_aware_maskableppo_batch_032_22.py",
        ROOT / "src" / "run_all_year_direct_action_safe_ppo.py",
        ROOT / "src" / "ppo_safe_rendering.py",
        ROOT / "src" / "054_hla_lowIC_site_transfer" / "run_054_00_hla_lowIC_expanded_action_maskableppo.py",
        ROOT / "src" / "weather_scenario_bank_160.py",
        ROOT / "src" / "build_202HLA_hla_lowIC_weather_scenario_bank.py",
        Path(__file__).resolve(),
    ]
    return {rel(path): sha256(path) for path in paths}


def preflight(cfg: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    manifest = read_manifest(cfg)
    split = make_split(manifest)
    train = split[split["split"].eq("train")].copy()
    validation = split[split["split"].eq("validation")].copy()
    expected_source_years = list(map(int, cfg["scope"]["train_source_years"]))
    expected_validation = list(map(int, cfg["scope"]["validation_years"]))
    issues: list[str] = []
    if len(train) != int(cfg["scope"]["expected_train_scenarios"]):
        issues.append(f"training scenario count mismatch: {len(train)}")
    if sorted(train["source_year"].unique().tolist()) != expected_source_years:
        issues.append("training source years mismatch")
    if sorted(train["weather_variant"].unique().tolist()) != sorted(EXPECTED_VARIANTS):
        issues.append("training variants mismatch")
    counts = train.groupby("source_year")["weather_variant"].nunique()
    if len(counts) != 10 or not counts.eq(6).all():
        issues.append(f"not every source year has six variants: {counts.to_dict()}")
    if validation["source_year"].astype(int).tolist() != expected_validation:
        issues.append(f"validation years mismatch: {validation['source_year'].tolist()}")
    if not validation["weather_variant"].eq("validation_original").all():
        issues.append("validation contains perturbed weather")
    if train["target_weather"].duplicated().any():
        issues.append("training target WTH paths are not unique")
    missing_targets = [path for path in split["target_weather"].astype(str) if not (ROOT / path).exists()]
    if missing_targets:
        issues.append(f"missing target weather: {missing_targets}")
    manifest_validation = manifest[manifest["split"].eq("validation")]
    mismatch = manifest_validation[
        manifest_validation["source_sha256"].astype(str) != manifest_validation["target_sha256"].astype(str)
    ]
    if not mismatch.empty:
        issues.append("validation WTH hash mismatch")
    input_root = ROOT / cfg["input_root"]
    source_root = ROOT / cfg["source_input_root"]
    if not input_root.exists() or not source_root.exists():
        issues.append("source or augmented input root missing")
    old_baseline = ROOT / cfg["comparison"]["baseline_run"]
    baseline_eval = old_baseline / "evaluation" / "054_00_checkpoint_validation_summary.csv"
    if not baseline_eval.exists():
        issues.append(f"paired baseline evaluation missing: {rel(baseline_eval)}")
    runtime = cfg["runtime_safety"]
    if int(runtime["environment_cache_limit"]) < 1 or int(runtime["environment_cache_limit"]) > 4:
        issues.append("environment cache limit must be 1..4")
    summary = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "reference_run": cfg["reference_run"],
        "source_input_root": cfg["source_input_root"],
        "augmented_input_root": cfg["input_root"],
        "source_manifest": cfg["source_manifest"],
        "train_source_years": expected_source_years,
        "train_scenario_rows": int(len(train)),
        "train_variant_counts": train["weather_variant"].value_counts().sort_index().to_dict(),
        "validation_original_years": validation["source_year"].astype(int).tolist(),
        "validation_hash_mismatch_rows": int(len(mismatch)),
        "unique_training_target_wth": int(train["target_weather"].nunique()),
        "actions": cfg["actions"],
        "observation_contract": cfg["observation_contract"],
        "reward_and_safety": cfg["scope"]["reward_and_safety"],
        "runtime_safety": runtime,
        "code_sha256": code_provenance(),
        "issues": issues,
        "next_step_allowed": not issues,
    }
    return summary, manifest, split


class BoundedAuditedRandomYearEnv(base03222.RandomYearEnv):
    """Same uniform random-year sampler as 032_22, with a bounded env cache."""

    def __init__(self, *args, **kwargs) -> None:
        global ACTIVE_ENV
        self.cache_limit = int(CACHE_LIMIT)
        self.env_create_count = 0
        self.env_eviction_count = 0
        self.peak_cached_envs = 0
        self.training_step_count = 0
        self.action_counts: Counter[int] = Counter()
        self.safe_pair_counts: Counter[str] = Counter()
        self.safe_nonzero_steps = 0
        self.safe_irrigation_sum = 0.0
        self.safe_n_sum = 0.0
        self.completed_episodes = 0
        self.recent_episode_rewards: deque[float] = deque(maxlen=20)
        self.recent_episode_lengths: deque[int] = deque(maxlen=20)
        self.recent_episode_irrigation: deque[float] = deque(maxlen=20)
        self.recent_episode_n: deque[float] = deque(maxlen=20)
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_irrigation = 0.0
        self._episode_n = 0.0
        super().__init__(*args, **kwargs)
        ACTIVE_ENV = self

    def _env_for(self, year: int):
        year = int(year)
        if year in self.envs:
            env = self.envs.pop(year)
            self.envs[year] = env
            return env
        while len(self.envs) >= self.cache_limit:
            old_year = next(iter(self.envs))
            old_env = self.envs.pop(old_year)
            try:
                old_env.close()
            except Exception:
                pass
            self.env_eviction_count += 1
        env = base03222.base.make_env(
            self.config,
            self.env_config,
            self.station,
            year,
            self.seed,
            f"{self.station}_{year}_032_22_train",
            evaluation=False,
        )
        self.envs[year] = env
        self.env_create_count += 1
        self.peak_cached_envs = max(self.peak_cached_envs, len(self.envs))
        return env

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_irrigation = 0.0
        self._episode_n = 0.0
        return obs, info

    def step(self, action):
        idx = int(np.asarray(action).reshape(-1)[0])
        obs, reward, terminated, truncated, info = super().step(action)
        action_info = dict(getattr(self.current_env, "last_action_info", {}) or {})
        safe_i = finite(action_info.get("safe_action_amir"), 0.0)
        safe_n = finite(action_info.get("safe_action_anfer"), 0.0)
        self.training_step_count += 1
        self.action_counts[idx] += 1
        self.safe_pair_counts[f"I{safe_i:g}/N{safe_n:g}"] += 1
        self.safe_nonzero_steps += int(abs(safe_i) + abs(safe_n) > 1e-9)
        self.safe_irrigation_sum += safe_i
        self.safe_n_sum += safe_n
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
            meta = SCENARIO_META.get(int(self.current_year), {})
            append_jsonl(
                EPISODES_JSONL,
                {
                    "episode_index": self.completed_episodes,
                    "pseudo_year": int(self.current_year),
                    "source_year": meta.get("source_year"),
                    "weather_variant": meta.get("weather_variant"),
                    "episode_reward": self._episode_reward,
                    "episode_length": self._episode_length,
                    "total_irrigation_mm": self._episode_irrigation,
                    "total_n_kg_ha": self._episode_n,
                },
            )
        return obs, reward, terminated, truncated, info

    def snapshot(self) -> dict[str, Any]:
        total_actions = int(sum(self.action_counts.values()))
        probs = np.asarray(list(self.action_counts.values()), dtype=float)
        if probs.size and probs.sum() > 0:
            probs = probs / probs.sum()
            entropy = float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum())
            normalized_entropy = entropy / math.log(max(2, int(self.action_space.n)))
        else:
            normalized_entropy = 0.0
        reset_years = {int(year) for year, count in self.reset_counts.items() if count > 0}
        source_years = {SCENARIO_META[year]["source_year"] for year in reset_years if year in SCENARIO_META}
        variants = {SCENARIO_META[year]["weather_variant"] for year in reset_years if year in SCENARIO_META}
        return {
            "training_steps_seen": self.training_step_count,
            "episodes_started": int(sum(self.reset_counts.values())),
            "episodes_completed": self.completed_episodes,
            "scenario_coverage": len(reset_years),
            "source_year_coverage": len(source_years),
            "variant_coverage": len(variants),
            "requested_noop_rate": finite(self.action_counts.get(0, 0) / total_actions if total_actions else 0.0),
            "requested_action_entropy_normalized": normalized_entropy,
            "safe_nonzero_rate": finite(self.safe_nonzero_steps / total_actions if total_actions else 0.0),
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


def instrumented_ppo_class(original_cls):
    class InstrumentedMaskablePPO(original_cls):
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
            emit_training_metric(row)
            if finite(row["rss_mb"], 0.0) >= MEMORY_STOP_MB:
                raise MemoryError(f"{TASK_ID} memory stop gate: rss={row['rss_mb']:.1f} MB >= {MEMORY_STOP_MB:.1f} MB")

    InstrumentedMaskablePPO.__name__ = "InstrumentedMaskablePPO"
    return InstrumentedMaskablePPO


def copy_task_outputs(out: Path) -> dict[str, str]:
    mapping = {
        "evaluation/042_10_training_checkpoint_inventory.csv": f"evaluation/{TASK_ID}_training_checkpoint_inventory.csv",
        "evaluation/042_10_checkpoint_validation_summary.csv": f"evaluation/{TASK_ID}_checkpoint_validation_summary.csv",
        "evaluation/042_10_validation_summary_by_station_checkpoint.csv": f"evaluation/{TASK_ID}_validation_summary_by_station_checkpoint.csv",
        "logs/032_22_training_year_reset_counts.csv": f"logs/{TASK_ID}_training_scenario_reset_counts.csv",
        "042_10_result.json": f"{TASK_ID}_engine_result.json",
    }
    copied: dict[str, str] = {}
    for source_rel, target_rel in mapping.items():
        source = out / source_rel
        target = out / target_rel
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied[source_rel] = rel(target)
    return copied


def jsonl_to_csv(path: Path, target: Path) -> None:
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if rows:
        pd.DataFrame(rows).to_csv(target, index=False, encoding="utf-8-sig")


def action_signature(daily: pd.DataFrame) -> str:
    events = daily[
        pd.to_numeric(daily.get("safe_action_amir"), errors="coerce").fillna(0.0).abs()
        + pd.to_numeric(daily.get("safe_action_anfer"), errors="coerce").fillna(0.0).abs()
        > 1e-9
    ]
    return "; ".join(
        f"DAP{int(finite(row.dap))}:I{finite(row.safe_action_amir):g}/N{finite(row.safe_action_anfer):g}"
        for row in events.itertuples(index=False)
    )


def audit_validation(out: Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summary_path = out / "evaluation" / f"{TASK_ID}_checkpoint_validation_summary.csv"
    summary = pd.read_csv(summary_path, keep_default_na=False)
    detail_rows: list[dict[str, Any]] = []
    i_grid = np.asarray(EXPECTED_I, dtype=float)
    n_grid = np.asarray(EXPECTED_N, dtype=float)
    for row in summary.itertuples(index=False):
        daily_path = ROOT / str(row.daily_csv_path)
        daily = pd.read_csv(daily_path, keep_default_na=False)
        for col in ["dap", "safe_action_amir", "safe_action_anfer", "discrete_action_index", "swfac", "nstres"]:
            daily[col] = pd.to_numeric(daily.get(col), errors="coerce").fillna(0.0)
        positive = daily["safe_action_amir"].abs() + daily["safe_action_anfer"].abs() > 1e-9
        safe_pairs = set(zip(daily.loc[positive, "safe_action_amir"], daily.loc[positive, "safe_action_anfer"]))
        i_off = ~daily["safe_action_amir"].apply(lambda value: bool(np.isclose(value, i_grid).any()))
        n_off = ~daily["safe_action_anfer"].apply(lambda value: bool(np.isclose(value, n_grid).any()))
        detail_rows.append(
            {
                "checkpoint_step": int(row.checkpoint_step),
                "year": int(row.year),
                "final_grnwt": finite(row.final_grnwt),
                "PFP_N": finite(row.PFP_N),
                "total_irrigation": finite(row.total_irrigation),
                "total_n": finite(row.total_n),
                "reward_stress_aware_sum": finite(row.reward_stress_aware_sum),
                "gap_yield_vs_four_max": finite(getattr(row, "gap_yield_vs_four_max", np.nan), np.nan),
                "gap_pfp_n_vs_four_max": finite(getattr(row, "gap_pfp_n_vs_four_max", np.nan), np.nan),
                "max_swfac": finite(getattr(row, "max_swfac", 0.0)),
                "max_nstres": finite(getattr(row, "max_nstres", 0.0)),
                "positive_action_rows": int(positive.sum()),
                "requested_noop_rate": finite((daily["discrete_action_index"] == 0).mean()),
                "unique_nonzero_safe_pairs": len(safe_pairs),
                "action_signature": action_signature(daily),
                "off_grid_rows": int((i_off | n_off).sum()),
                "daily_csv_path": str(row.daily_csv_path),
            }
        )
    detail = pd.DataFrame(detail_rows).sort_values(["checkpoint_step", "year"]).reset_index(drop=True)
    checkpoint_rows: list[dict[str, Any]] = []
    for checkpoint, group in detail.groupby("checkpoint_step"):
        checkpoint_rows.append(
            {
                "checkpoint_step": int(checkpoint),
                "validation_years": int(len(group)),
                "mean_final_grnwt": finite(group["final_grnwt"].mean()),
                "std_final_grnwt": finite(group["final_grnwt"].std(ddof=0)),
                "mean_PFP_N": finite(group["PFP_N"].mean()),
                "mean_total_irrigation": finite(group["total_irrigation"].mean()),
                "mean_total_n": finite(group["total_n"].mean()),
                "mean_reward": finite(group["reward_stress_aware_sum"].mean()),
                "mean_gap_yield_vs_four_max": finite(group["gap_yield_vs_four_max"].mean(), np.nan),
                "mean_gap_pfp_n_vs_four_max": finite(group["gap_pfp_n_vs_four_max"].mean(), np.nan),
                "unique_full_action_signatures": int(group["action_signature"].nunique()),
                "unique_total_irrigation": int(group["total_irrigation"].nunique()),
                "unique_total_n": int(group["total_n"].nunique()),
                "mean_requested_noop_rate": finite(group["requested_noop_rate"].mean()),
                "mean_positive_action_rows": finite(group["positive_action_rows"].mean()),
                "max_swfac": finite(group["max_swfac"].max()),
                "max_nstres": finite(group["max_nstres"].max()),
                "off_grid_rows": int(group["off_grid_rows"].sum()),
                "all_years_same_action_signature": bool(group["action_signature"].nunique() == 1),
                "all_years_zero_management": bool((group["total_irrigation"].abs() + group["total_n"].abs() <= 1e-9).all()),
            }
        )
    by_checkpoint = pd.DataFrame(checkpoint_rows).sort_values("checkpoint_step").reset_index(drop=True)
    baseline_path = ROOT / cfg["comparison"]["baseline_run"] / "evaluation" / "054_00_checkpoint_validation_summary.csv"
    baseline = pd.read_csv(baseline_path, keep_default_na=False)
    matched = int(cfg["comparison"]["matched_checkpoint"])
    old = baseline[pd.to_numeric(baseline["checkpoint_step"], errors="coerce").eq(matched)].copy()
    for col in ["final_grnwt", "PFP_N", "total_irrigation", "total_n"]:
        old[col] = pd.to_numeric(old[col], errors="coerce")
    new = detail[detail["checkpoint_step"].eq(matched)]
    common_checkpoint_rows: list[dict[str, Any]] = []
    old_steps = set(pd.to_numeric(baseline["checkpoint_step"], errors="coerce").dropna().astype(int))
    new_steps = set(detail["checkpoint_step"].astype(int))
    for checkpoint in sorted(old_steps & new_steps):
        old_group = baseline[pd.to_numeric(baseline["checkpoint_step"], errors="coerce").eq(checkpoint)].copy()
        new_group = detail[detail["checkpoint_step"].eq(checkpoint)].copy()
        for col in ["final_grnwt", "PFP_N", "total_irrigation", "total_n"]:
            old_group[col] = pd.to_numeric(old_group[col], errors="coerce")
        common_checkpoint_rows.append(
            {
                "checkpoint_step": int(checkpoint),
                "old_mean_final_grnwt": finite(old_group["final_grnwt"].mean(), np.nan),
                "new_mean_final_grnwt": finite(new_group["final_grnwt"].mean(), np.nan),
                "delta_mean_final_grnwt": finite(new_group["final_grnwt"].mean() - old_group["final_grnwt"].mean(), np.nan),
                "old_mean_PFP_N": finite(old_group["PFP_N"].mean(), np.nan),
                "new_mean_PFP_N": finite(new_group["PFP_N"].mean(), np.nan),
                "delta_mean_PFP_N": finite(new_group["PFP_N"].mean() - old_group["PFP_N"].mean(), np.nan),
                "old_mean_total_irrigation": finite(old_group["total_irrigation"].mean(), np.nan),
                "new_mean_total_irrigation": finite(new_group["total_irrigation"].mean(), np.nan),
                "old_mean_total_n": finite(old_group["total_n"].mean(), np.nan),
                "new_mean_total_n": finite(new_group["total_n"].mean(), np.nan),
                "old_unique_action_signatures": int(old_group["action_sequence"].astype(str).nunique()),
                "new_unique_action_signatures": int(new_group["action_signature"].astype(str).nunique()),
            }
        )
    comparison = {
        "new_task_id": TASK_ID,
        "matched_checkpoint": matched,
        "old_054_validation_rows": int(len(old)),
        "new_augmented_validation_rows": int(len(new)),
        "old_mean_final_grnwt": finite(old["final_grnwt"].mean(), np.nan),
        "new_mean_final_grnwt": finite(new["final_grnwt"].mean(), np.nan),
        "delta_mean_final_grnwt": finite(new["final_grnwt"].mean() - old["final_grnwt"].mean(), np.nan),
        "old_mean_PFP_N": finite(old["PFP_N"].mean(), np.nan),
        "new_mean_PFP_N": finite(new["PFP_N"].mean(), np.nan),
        "delta_mean_PFP_N": finite(new["PFP_N"].mean() - old["PFP_N"].mean(), np.nan),
        "old_mean_total_irrigation": finite(old["total_irrigation"].mean(), np.nan),
        "new_mean_total_irrigation": finite(new["total_irrigation"].mean(), np.nan),
        "old_mean_total_n": finite(old["total_n"].mean(), np.nan),
        "new_mean_total_n": finite(new["total_n"].mean(), np.nan),
        "common_checkpoint_comparison": common_checkpoint_rows,
        "WP_ET_status": "pending exact Summary.OUT replay; never inferred from daily CSV",
    }
    return detail, by_checkpoint, comparison


def training_coverage(out: Path) -> dict[str, Any]:
    reset_path = out / "logs" / f"{TASK_ID}_training_scenario_reset_counts.csv"
    reset = pd.read_csv(reset_path, keep_default_na=False) if reset_path.exists() else pd.DataFrame()
    if reset.empty:
        return {"scenario_rows": 0, "covered_scenarios": 0, "coverage_ratio": 0.0}
    reset["year"] = pd.to_numeric(reset["year"], errors="coerce").astype(int)
    reset["episode_count"] = pd.to_numeric(reset["episode_count"], errors="coerce").fillna(0).astype(int)
    reset["source_year"] = reset["year"].map(lambda year: SCENARIO_META.get(int(year), {}).get("source_year"))
    reset["weather_variant"] = reset["year"].map(lambda year: SCENARIO_META.get(int(year), {}).get("weather_variant"))
    reset.to_csv(out / "logs" / f"{TASK_ID}_training_scenario_reset_counts_annotated.csv", index=False, encoding="utf-8-sig")
    covered = reset[reset["episode_count"].gt(0)]
    return {
        "scenario_rows": int(len(reset)),
        "covered_scenarios": int(len(covered)),
        "coverage_ratio": finite(len(covered) / len(reset) if len(reset) else 0.0),
        "covered_source_years": int(covered["source_year"].nunique()),
        "covered_variants": int(covered["weather_variant"].nunique()),
        "min_episode_count": int(reset["episode_count"].min()),
        "max_episode_count": int(reset["episode_count"].max()),
        "total_episode_starts": int(reset["episode_count"].sum()),
    }


def write_record(
    cfg: dict[str, Any],
    smoke: bool,
    preflight_result: dict[str, Any],
    effective: dict[str, Any],
    coverage: dict[str, Any],
    by_checkpoint: pd.DataFrame,
    comparison: dict[str, Any],
    gate: dict[str, Any],
    status: str,
    error: str,
) -> None:
    formal_steps = int(cfg["training"]["total_timesteps"])
    phase = "2K smoke" if smoke else f"{formal_steps // 1000}K paired exploration"
    lines = [
        f"# {TASK_ID} HLA lowIC 十年六天气情景 PPO 实验记录",
        "",
        f"- 阶段：`{phase}`；状态：`{status}`。",
        "- 新工作线；复用 054/042/032 的训练、奖励、安全约束和原始验证代码。",
        "- 唯一实验性变化：训练集由 10 个原始天气 episode 扩展为 10×6=60 个天气 episode。",
        "- 为防 OOM，RandomYearEnv 的采样分布保持 uniform random，但环境对象缓存上限为 2；这是资源管理措施，不改变奖励或动作评价。",
        "- WP_ET 只接受 Summary.OUT/ETCP 精确回放；本阶段若尚未回放则明确标为 pending。",
        "",
        "## 数据与代码 provenance",
        "",
        "```json",
        json.dumps(preflight_result, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 实际生效 PPO/奖励/安全配置",
        "",
        "```json",
        json.dumps(effective, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 训练情景覆盖与内存",
        "",
        "```json",
        json.dumps(coverage, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 原始验证集 checkpoint 指标",
        "",
        by_checkpoint.to_markdown(index=False) if not by_checkpoint.empty else "尚无验证结果。",
        "",
        "## 与未增强 054_00 的相同步数配对比较",
        "",
        "```json",
        json.dumps(comparison, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 技术门槛",
        "",
        "```json",
        json.dumps(gate, ensure_ascii=False, indent=2),
        "```",
    ]
    if error:
        lines.extend(["", "## Error", "", "```text", error[-10000:], "```"])
    record_path(smoke).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg_path: Path, *, dry_run: bool, smoke: bool, formal: bool) -> dict[str, Any]:
    global SCENARIO_META, METRICS_JSONL, METRICS_TEXT, EPISODES_JSONL, RUN_STARTED
    global MEMORY_WARNING_MB, MEMORY_STOP_MB, CACHE_LIMIT, ACTIVE_ENV

    if not smoke and not formal and not dry_run:
        raise ValueError("choose --smoke or --formal")
    cfg = load_config(cfg_path)
    preflight_result, manifest, split = preflight(cfg)
    phase_cfg = cfg["smoke"] if smoke else cfg["training"]
    if dry_run:
        return {
            **preflight_result,
            "mode": "dry_run",
            "planned_phase": "smoke" if smoke else "formal",
            "total_timesteps": int(phase_cfg["total_timesteps"]),
            "checkpoint_steps": list(map(int, phase_cfg["checkpoint_steps"])),
        }
    if not preflight_result["next_step_allowed"]:
        raise RuntimeError(json.dumps(preflight_result, ensure_ascii=False, indent=2))
    out = output_root(cfg, smoke)
    if out.exists():
        raise FileExistsError(f"refuse to overwrite existing output: {rel(out)}")
    for sub in ["configs", "evaluation", "logs", "models", "daily_outputs", "audits", "tensorboard"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, out / "configs" / cfg_path.name)
    shutil.copy2(PROMPT, out / "configs" / PROMPT.name)
    manifest.to_csv(out / "configs" / f"{TASK_ID}_selected_manifest.csv", index=False, encoding="utf-8-sig")
    split.to_csv(out / "configs" / f"{TASK_ID}_training_validation_split.csv", index=False, encoding="utf-8-sig")
    (out / "configs" / f"{TASK_ID}_preflight.json").write_text(
        json.dumps(preflight_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    train_split = split[split["split"].eq("train")]
    SCENARIO_META = {
        int(row.year): {"source_year": int(row.source_year), "weather_variant": str(row.weather_variant)}
        for row in train_split.itertuples(index=False)
    }
    runtime = cfg["runtime_safety"]
    CACHE_LIMIT = int(runtime["environment_cache_limit"])
    MEMORY_WARNING_MB = float(runtime["memory_warning_mb"])
    MEMORY_STOP_MB = float(runtime["memory_stop_mb"])
    METRICS_JSONL = out / "logs" / f"{TASK_ID}_training_update_metrics.jsonl"
    METRICS_TEXT = out / "logs" / f"{TASK_ID}_training_update_console.log"
    EPISODES_JSONL = out / "logs" / f"{TASK_ID}_training_episode_metrics.jsonl"
    RUN_STARTED = time.perf_counter()
    ACTIVE_ENV = None

    effective = engine.load_config()
    effective_record = {
        "seed": int(cfg["seed"]),
        "total_timesteps": int(phase_cfg["total_timesteps"]),
        "checkpoint_steps": list(map(int, phase_cfg["checkpoint_steps"])),
        "ppo": effective.get("ppo"),
        "reward": effective.get("reward"),
        "action_safety": effective.get("action_safety"),
        "discrete_actions_after_203_patch": cfg["actions"],
        "reward_changed_by_203": False,
        "action_safety_changed_by_203": False,
        "observation_changed_by_203": False,
    }
    (out / "configs" / f"{TASK_ID}_effective_training_contract.json").write_text(
        json.dumps(effective_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    import sb3_contrib

    original_maskableppo = sb3_contrib.MaskablePPO
    old_engine = {
        key: getattr(engine, key)
        for key in [
            "TASK_ID",
            "TASK_NAME",
            "BASE_OUT",
            "BASE_DOC",
            "PROMPT",
            "LOWIC_INPUT_ROOT",
            "STATION",
            "SITES",
            "BINARY_IRRIGATION_LEVELS",
            "BINARY_NITROGEN_LEVELS",
        ]
    }
    old_base = {
        "SITES": list(base03222.SITES),
        "SITE_NAMES": dict(base03222.SITE_NAMES),
        "SEED": base03222.SEED,
        "OUT": base03222.OUT,
        "DOC": base03222.DOC,
        "PROMPT": base03222.PROMPT,
        "TOTAL_TIMESTEPS": base03222.TOTAL_TIMESTEPS,
        "CHECKPOINT_STEPS": list(base03222.CHECKPOINT_STEPS),
        "load_split": base03222.load_split,
        "build_selection": base03222.build_selection,
        "RandomYearEnv": base03222.RandomYearEnv,
    }
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    status = "completed"
    error = ""
    copied: dict[str, str] = {}
    try:
        sb3_contrib.MaskablePPO = instrumented_ppo_class(original_maskableppo)
        engine.TASK_ID = TASK_ID
        engine.TASK_NAME = TASK_NAME
        engine.BASE_OUT = out
        engine.BASE_DOC = record_path(smoke)
        engine.PROMPT = PROMPT
        engine.LOWIC_INPUT_ROOT = ROOT / cfg["input_root"]
        engine.STATION = STATION
        engine.SITES = [STATION]
        engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
        engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
        base03222.SITES = [STATION]
        base03222.SITE_NAMES.clear()
        base03222.SITE_NAMES.update({STATION: SITE})
        base03222.SEED = int(cfg["seed"])
        base03222.load_split = lambda: split.copy()
        base03222.build_selection = lambda _split: split.copy()
        base03222.RandomYearEnv = BoundedAuditedRandomYearEnv
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = ROOT / cfg["input_root"]
        engine.run_training(
            int(phase_cfg["total_timesteps"]),
            list(map(int, phase_cfg["checkpoint_steps"])),
            suffix="",
        )
        copied = copy_task_outputs(out)
    except Exception:
        status = "failed"
        error = traceback.format_exc()
    finally:
        sb3_contrib.MaskablePPO = original_maskableppo
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root
        for key, value in old_engine.items():
            setattr(engine, key, value)
        for key, value in old_base.items():
            if key == "SITE_NAMES":
                base03222.SITE_NAMES.clear()
                base03222.SITE_NAMES.update(value)
            else:
                setattr(base03222, key, value)

    jsonl_to_csv(METRICS_JSONL, out / "logs" / f"{TASK_ID}_training_update_metrics.csv")
    jsonl_to_csv(EPISODES_JSONL, out / "logs" / f"{TASK_ID}_training_episode_metrics.csv")
    coverage: dict[str, Any] = {}
    detail = pd.DataFrame()
    by_checkpoint = pd.DataFrame()
    comparison: dict[str, Any] = {"WP_ET_status": "not evaluated because training failed"}
    gate: dict[str, Any] = {"status_completed": status == "completed"}
    if status == "completed":
        coverage = training_coverage(out)
        detail, by_checkpoint, comparison = audit_validation(out, cfg)
        detail.to_csv(out / "audits" / f"{TASK_ID}_validation_action_detail.csv", index=False, encoding="utf-8-sig")
        by_checkpoint.to_csv(out / "audits" / f"{TASK_ID}_checkpoint_process_summary.csv", index=False, encoding="utf-8-sig")
        if comparison.get("common_checkpoint_comparison"):
            pd.DataFrame(comparison["common_checkpoint_comparison"]).to_csv(
                out / "audits" / f"{TASK_ID}_paired_054_checkpoint_comparison.csv", index=False, encoding="utf-8-sig"
            )
        metrics_path = out / "logs" / f"{TASK_ID}_training_update_metrics.csv"
        metrics = pd.read_csv(metrics_path, keep_default_na=False) if metrics_path.exists() else pd.DataFrame()
        peak_rss = finite(pd.to_numeric(metrics.get("rss_mb"), errors="coerce").max(), 0.0) if not metrics.empty else 0.0
        final_checkpoint = int(max(phase_cfg["checkpoint_steps"]))
        final = by_checkpoint[by_checkpoint["checkpoint_step"].eq(final_checkpoint)]
        gate = {
            "status_completed": True,
            "final_checkpoint": final_checkpoint,
            "observed_validation_rows": int(len(detail[detail["checkpoint_step"].eq(final_checkpoint)])),
            "expected_validation_rows": len(cfg["scope"]["validation_years"]),
            "training_metric_update_rows": int(len(metrics)),
            "training_scenarios_covered": int(coverage.get("covered_scenarios", 0)),
            "training_scenario_coverage_ratio": finite(coverage.get("coverage_ratio"), 0.0),
            "all_six_variant_types_seen": int(coverage.get("covered_variants", 0)) == 6,
            "all_ten_source_years_seen": int(coverage.get("covered_source_years", 0)) == 10,
            "peak_rss_mb": peak_rss,
            "memory_warning_exceeded": peak_rss >= MEMORY_WARNING_MB,
            "memory_stop_exceeded": peak_rss >= MEMORY_STOP_MB,
            "final_actions_on_grid": bool(len(final) and int(final.iloc[0]["off_grid_rows"]) == 0),
            "final_not_all_zero_management": bool(len(final) and not bool(final.iloc[0]["all_years_zero_management"])),
            "final_unique_action_signatures": int(final.iloc[0]["unique_full_action_signatures"]) if len(final) else 0,
        }
        technical_checks = [
            gate["observed_validation_rows"] == gate["expected_validation_rows"],
            gate["training_metric_update_rows"] > 0,
            gate["all_six_variant_types_seen"],
            gate["final_actions_on_grid"],
            gate["final_not_all_zero_management"],
            not gate["memory_stop_exceeded"],
        ]
        if not smoke:
            technical_checks.extend([gate["all_ten_source_years_seen"], gate["training_scenario_coverage_ratio"] >= 0.80])
        gate["next_step_allowed"] = bool(all(technical_checks))

    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "phase": "smoke2k" if smoke else f"formal{int(phase_cfg['total_timesteps']) // 1000}k",
        "status": status,
        "output_root": rel(out),
        "record_md": rel(record_path(smoke)),
        "elapsed_seconds": time.perf_counter() - RUN_STARTED,
        "preflight": rel(out / "configs" / f"{TASK_ID}_preflight.json"),
        "effective_training_contract": rel(out / "configs" / f"{TASK_ID}_effective_training_contract.json"),
        "training_update_metrics": rel(out / "logs" / f"{TASK_ID}_training_update_metrics.csv") if (out / "logs" / f"{TASK_ID}_training_update_metrics.csv").exists() else "",
        "training_episode_metrics": rel(out / "logs" / f"{TASK_ID}_training_episode_metrics.csv") if (out / "logs" / f"{TASK_ID}_training_episode_metrics.csv").exists() else "",
        "coverage": coverage,
        "matched_054_comparison": comparison,
        "technical_gate": gate,
        "copied_outputs": copied,
        "WP_ET_status": comparison.get("WP_ET_status"),
        "error": error[-10000:] if error else "",
    }
    result_name = f"{TASK_ID}_smoke_result.json" if smoke else f"{TASK_ID}_formal_result.json"
    (out / result_name).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"{TASK_ID}_run_manifest.json").write_text(
        json.dumps({"status": f"completed_{result['phase']}" if status == "completed" else "failed", "config": cfg, "preflight": preflight_result, "result": result}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_record(cfg, smoke, preflight_result, effective_record, coverage, by_checkpoint, comparison, gate, status, error)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if status != "completed":
        raise RuntimeError(error)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--smoke", action="store_true")
    phase.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else (ROOT / args.config).resolve()
    planned_smoke = bool(args.smoke or not args.formal)
    print(json.dumps(run(cfg_path, dry_run=args.dry_run, smoke=planned_smoke, formal=bool(args.formal)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
