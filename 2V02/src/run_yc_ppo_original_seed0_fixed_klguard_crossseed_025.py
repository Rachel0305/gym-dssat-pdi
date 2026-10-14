"""025: independent YC Original PPO seed0 CONTROL vs fixed KL-GUARD.

This file is an experiment orchestrator only.  It imports the existing YC
runtime and the already-frozen KL instrumentation from 019; it does not edit
the PPO, DSSAT, observation, reward, action, or mask implementations.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import inspect
import json
import math
import random
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "src" / "055_yca_lowIC_site_transfer"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

PROMPT = ROOT / "prompt_01" / "025_codex_prompt_025_yc_ppo_original_seed0_fixed_klguard_crossseed.md"
HISTORICAL_CONFIG = ROOT / "benchmark_results" / "055_00_yca_lowIC_expanded_action_maskableppo" / "configs" / "055_00_yca_lowIC_expanded_action_maskableppo.json"
HISTORICAL_METRICS = ROOT / "results" / "yc_seed_instability_checkpoint_metrics.csv"
OUT = ROOT / "benchmark_results" / "025_yc_ppo_original_seed0_fixed_klguard_crossseed"
RESULTS = ROOT / "results"
REPORT = ROOT / "docs" / "yc_ppo_original_seed0_fixed_klguard_crossseed_replication.md"

STATION = "YCA"
SITE = "YC"
SEED = 0
TRAIN_YEARS = list(range(2004, 2014))
VALIDATION_YEARS = list(range(2014, 2024))
CHECKPOINTS = [25_000, 50_000, 75_000, 100_000]
TOTAL_TIMESTEPS = 100_000
SMOKE_TIMESTEPS = 2_000
EFFECTIVE_KL_THRESHOLD = 1.0e-4
CONFIGURED_TARGET_KL = 6.666666667e-05

CONTROL_METRICS_OUT = RESULTS / "yc_ppo_original_seed0_control_training_metrics.csv"
GUARD_METRICS_OUT = RESULTS / "yc_ppo_original_seed0_klguard_training_metrics.csv"
CONTROL_VALIDATION_OUT = RESULTS / "yc_ppo_original_seed0_control_checkpoint_validation.csv"
GUARD_VALIDATION_OUT = RESULTS / "yc_ppo_original_seed0_klguard_checkpoint_validation.csv"
WINDOW_OUT = RESULTS / "yc_ppo_original_seed0_control_vs_klguard_update_windows.csv"
HISTORICAL_OUT = RESULTS / "yc_ppo_original_seed0_historical_vs_new_control.csv"
PAIRED_OUT = RESULTS / "yc_ppo_original_seed0_control_vs_klguard_paired.csv"
DECISION_OUT = RESULTS / "yc_ppo_original_seed0_klguard_decision.csv"

EXPECTED_METRIC_COLUMNS = [
    "global_step", "timestep", "approx_kl", "clip_fraction", "entropy_loss",
    "policy_gradient_loss", "value_loss", "explained_variance", "learning_rate",
    "clip_range", "epochs_attempted", "epochs_completed", "optimizer_steps_completed",
    "kl_guard_triggered", "kl_at_trigger",
]

SOURCE_FILES = [
    HISTORICAL_CONFIG,
    ROOT / "src" / "run_sya_lowIC_binary_timing_maskableppo_042_10.py",
    ROOT / "src" / "run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36.py",
    ROOT / "src" / "run_five_site_half_split_stress_aware_maskableppo_batch_032_22.py",
    ROOT / "src" / "run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py",
    ROOT / "src" / "run_free_timing_discrete_maskableppo_ncost2x_scaled_reward_031_17.py",
    ROOT / "src" / "run_yc_ppo_seed1_kl_guard_paired_019.py",
    ROOT / "my_data" / "MZCER048.CUL",
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def set_all_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_data(path: Path) -> dict[str, Any]:
    import zipfile

    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("data"))


def read_historical_config() -> dict[str, Any]:
    return json.loads(HISTORICAL_CONFIG.read_text(encoding="utf-8"))


def contract_snapshot(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "station_code": cfg.get("station_code"),
        "site": cfg.get("site"),
        "input_profile": cfg.get("input_profile"),
        "seed": cfg.get("seed"),
        "actions": cfg.get("actions"),
        "observation_contract": cfg.get("observation_contract"),
        "train_years": cfg.get("scope", {}).get("train_years"),
        "validation_years": cfg.get("scope", {}).get("validation_years"),
        "reward_and_safety": cfg.get("scope", {}).get("reward_and_safety"),
        "training": cfg.get("training"),
    }


def runtime_setup(output_root: Path):
    """Patch only in-process module globals, exactly as the 055 runner does."""
    import ppo_safe_rendering
    import run_055_03_yca_lowIC_five_scenario_figures as replay

    engine = replay.engine
    base03222 = replay.base03222
    direct = replay.direct_ppo
    lowic_root = replay.INPUT_PROFILES["lowIC"]

    # 055's existing 042_10 patch selects the historical LateIrrigationReserve
    # wrapper and the same 032_22 PPO configuration.  This is runtime setup,
    # not a source edit.  Set the engine globals before patching because the
    # patch copies the station list into base03222.
    engine.TASK_ID = "025"
    engine.TASK_NAME = "yc_ppo_original_seed0_fixed_klguard_crossseed"
    engine.BASE_OUT = output_root
    engine.BASE_DOC = REPORT
    engine.LOWIC_INPUT_ROOT = lowic_root
    engine.STATION = STATION
    engine.SITES = [STATION]
    engine.BINARY_IRRIGATION_LEVELS = [0.0, 15.0, 30.0, 45.0]
    engine.BINARY_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
    engine.patch_base_module(output_root, REPORT, TOTAL_TIMESTEPS, CHECKPOINTS)
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = lowic_root
    base03222.SITE_NAMES[STATION] = SITE

    config = base03222.load_config()
    config["seed"] = SEED
    config["total_timesteps"] = TOTAL_TIMESTEPS
    selection = base03222.build_selection(base03222.load_split())
    env_config = direct.build_env_config(config, selection)
    env_config["paths"]["output_root"] = rel(output_root)
    return replay, base03222, direct, config, env_config


def verify_guard_semantics() -> dict[str, Any]:
    from sb3_contrib import MaskablePPO

    source = inspect.getsource(MaskablePPO.train)
    match = re.search(r"if self\.target_kl is not None and approx_kl_div > ([^:]+):", source)
    condition = match.group(0) if match else "unknown / insufficient evidence"
    uses_one_point_five = "1.5 * self.target_kl" in source
    effective = CONFIGURED_TARGET_KL * 1.5 if uses_one_point_five else math.nan
    return {
        "source_file": str(inspect.getsourcefile(MaskablePPO.train) or "unknown / insufficient evidence"),
        "source_code_condition": condition,
        "configured_target_kl": CONFIGURED_TARGET_KL,
        "effective_threshold": effective,
        "required_effective_threshold": EFFECTIVE_KL_THRESHOLD,
        "pass": bool(match and uses_one_point_five and math.isclose(effective, EFFECTIVE_KL_THRESHOLD, rel_tol=0.0, abs_tol=1e-10)),
    }


def preflight() -> dict[str, Any]:
    cfg = read_historical_config()
    issues: list[str] = []
    expected_contract = {
        "station_code": STATION,
        "site": SITE,
        "input_profile": "lowIC",
        "seed": SEED,
        "actions": {
            "irrigation_levels_mm": [0.0, 15.0, 30.0, 45.0],
            "nitrogen_levels_kg_ha": [0.0, 40.0, 80.0, 120.0],
        },
        "train_years": TRAIN_YEARS,
        "validation_years": VALIDATION_YEARS,
    }
    if cfg.get("station_code") != expected_contract["station_code"] or cfg.get("site") != expected_contract["site"]:
        issues.append("historical 055 station/site contract mismatch")
    if cfg.get("input_profile") != expected_contract["input_profile"] or cfg.get("seed") != SEED:
        issues.append("historical 055 input/seed contract mismatch")
    if cfg.get("actions") != expected_contract["actions"]:
        issues.append("historical 055 action grid mismatch")
    if cfg.get("scope", {}).get("train_years") != TRAIN_YEARS:
        issues.append("historical 055 train years mismatch")
    if cfg.get("scope", {}).get("validation_years") != VALIDATION_YEARS:
        issues.append("historical 055 validation years mismatch")
    if cfg.get("observation_contract") != {
        "base": "046_02_raw_observation",
        "normalization_enabled": False,
        "weather_forecast_enabled": False,
    }:
        issues.append("historical 055 observation contract mismatch")
    if not PROMPT.exists():
        issues.append(f"prompt missing: {rel(PROMPT)}")
    for path in SOURCE_FILES:
        if not path.exists():
            issues.append(f"source missing: {path}")
    historical_root = ROOT / "benchmark_results" / "055_00_yca_lowIC_expanded_action_maskableppo"
    historical_models = [
        historical_root / "models" / "YCA" / f"YCA_half_split_stress_aware_maskableppo_seed0_ckpt{x}.zip"
        for x in CHECKPOINTS
    ]
    missing_historical = [rel(path) for path in historical_models if not path.exists()]
    if missing_historical:
        issues.append(f"historical seed0 checkpoints missing: {missing_historical}")
    guard = verify_guard_semantics()
    if not guard["pass"]:
        issues.append("MaskablePPO source KL guard semantics did not verify")

    payload = {
        "status": "PASS" if not issues else "FAIL",
        "prompt": rel(PROMPT),
        "historical_config": rel(HISTORICAL_CONFIG),
        "historical_contract": contract_snapshot(cfg),
        "new_experiment": {
            "branches": ["control", "klguard"],
            "training_seed": SEED,
            "training_from_scratch": True,
            "total_timesteps": TOTAL_TIMESTEPS,
            "checkpoints": CHECKPOINTS,
            "configured_target_kl_control": None,
            "configured_target_kl_klguard": CONFIGURED_TARGET_KL,
            "effective_kl_threshold": EFFECTIVE_KL_THRESHOLD,
        },
        "guard_semantics": guard,
        "historical_checkpoint_paths": [rel(path) for path in historical_models],
        "source_sha256": {rel(path): sha256_file(path) for path in SOURCE_FILES if path.exists()},
        "issues": issues,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "preflight.json", payload)
    return payload


def smoke_branch(branch: str, base03222: Any, config: dict[str, Any], env_config: dict[str, Any]) -> dict[str, Any]:
    from stable_baselines3.common.logger import configure

    InstrumentedPPO = __import__("run_yc_ppo_seed1_kl_guard_paired_019", fromlist=["make_instrumented_ppo_class"]).make_instrumented_ppo_class()
    branch_root = OUT / "smoke" / branch
    branch_root.mkdir(parents=True, exist_ok=True)
    branch_env_config = copy.deepcopy(env_config)
    branch_env_config["paths"]["output_root"] = rel(branch_root / "runtime")
    smoke_config = copy.deepcopy(config)
    smoke_config["total_timesteps"] = SMOKE_TIMESTEPS
    env = None
    result: dict[str, Any] = {"branch": branch, "status": "FAIL"}
    try:
        set_all_seeds(SEED)
        env = base03222.RandomYearEnv(smoke_config, branch_env_config, STATION, TRAIN_YEARS, SEED)
        obs, info = env.reset()
        mask = np.asarray(env.action_masks(), dtype=bool)
        kwargs = base03222.base.ppo_kwargs(smoke_config)
        kwargs["target_kl"] = None if branch == "control" else CONFIGURED_TARGET_KL
        model = InstrumentedPPO("MlpPolicy", env, verbose=0, seed=SEED, device="cpu", **kwargs)
        model.set_logger(configure(str(branch_root / "tensorboard"), ["tensorboard"]))
        model.learn(total_timesteps=SMOKE_TIMESTEPS, reset_num_timesteps=True, progress_bar=False)
        model.logger.close()
        result.update({
            "status": "PASS",
            "timesteps": int(model.num_timesteps),
            "observation_shape": list(env.observation_space.shape or []),
            "action_space_n": int(env.action_space.n),
            "initial_legal_action_count": int(mask.sum()),
            "initial_mask_has_noop": bool(mask[0]),
            "target_kl": kwargs["target_kl"],
        })
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        if env is not None:
            env.close()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return result


def run_smoke() -> dict[str, Any]:
    pf = preflight()
    if pf["status"] != "PASS":
        raise RuntimeError("Preflight failed: " + "; ".join(pf["issues"]))
    _, base03222, _, config, env_config = runtime_setup(OUT)
    rows = [smoke_branch(branch, base03222, config, env_config) for branch in ["control", "klguard"]]
    payload = {
        "status": "PASS" if all(row.get("status") == "PASS" for row in rows) else "FAIL",
        "rows": rows,
        "next_step_allowed": all(row.get("status") == "PASS" for row in rows),
    }
    write_json(OUT / "smoke_manifest.json", payload)
    return payload


class DenseCheckpointCallbackFactory:
    def __init__(self, branch: str, checkpoints: list[int]):
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class Callback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0)

            def _on_step(self) -> bool:
                now = int(self.model.num_timesteps)
                for target in outer.checkpoints:
                    if target <= now and target not in outer.saved:
                        path = outer.path_for(target)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        self.model.save(str(path.with_suffix("")))
                        outer.saved.append(target)
                        outer.actual_steps[target] = int(self.model.num_timesteps)
                return True

        self.branch = branch
        self.checkpoints = [int(x) for x in checkpoints]
        self.saved: list[int] = []
        self.actual_steps: dict[int, int] = {}
        self.callback = Callback()

    def path_for(self, checkpoint: int) -> Path:
        return OUT / self.branch / "models" / "YCA" / f"yc_ppo_original_seed0_{self.branch}_ckpt{int(checkpoint)}.zip"


def branch_checkpoint_path(branch: str, checkpoint: int) -> Path:
    return OUT / branch / "models" / "YCA" / f"yc_ppo_original_seed0_{branch}_ckpt{int(checkpoint)}.zip"


def smoke_is_passed() -> bool:
    path = OUT / "smoke_manifest.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("status") == "PASS" and payload.get("next_step_allowed") is True
    except Exception:
        return False


def train_branch(branch: str) -> dict[str, Any]:
    if not smoke_is_passed():
        raise RuntimeError("025 formal training is blocked until the two-branch smoke manifest passes")
    branch_root = OUT / branch
    manifest_path = branch_root / "training_manifest.json"
    expected = {checkpoint: branch_checkpoint_path(branch, checkpoint) for checkpoint in CHECKPOINTS}
    if manifest_path.exists() and all(path.exists() for path in expected.values()):
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    existing = [rel(path) for path in expected.values() if path.exists()]
    if existing:
        raise RuntimeError(f"{branch}: partial formal outputs exist; refusing to overwrite: {existing}")

    import torch
    from stable_baselines3.common.logger import configure

    _, base03222, _, config, env_config = runtime_setup(OUT)
    branch_env_config = copy.deepcopy(env_config)
    branch_env_config["paths"]["output_root"] = rel(branch_root / "runtime")
    branch_root.mkdir(parents=True, exist_ok=True)
    (branch_root / "configs").mkdir(parents=True, exist_ok=True)
    write_json(branch_root / "configs" / "resolved_runtime_config.json", config)
    write_json(branch_root / "configs" / "resolved_env_config.json", branch_env_config)
    InstrumentedPPO = __import__("run_yc_ppo_seed1_kl_guard_paired_019", fromlist=["make_instrumented_ppo_class"]).make_instrumented_ppo_class()
    target_kl = None if branch == "control" else CONFIGURED_TARGET_KL
    env = None
    model = None
    callback_factory = DenseCheckpointCallbackFactory(branch, CHECKPOINTS)
    start = pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
    failure_path = branch_root / "training_failure_manifest.json"
    try:
        set_all_seeds(SEED)
        env = base03222.RandomYearEnv(config, branch_env_config, STATION, TRAIN_YEARS, SEED)
        kwargs = base03222.base.ppo_kwargs(config)
        kwargs["target_kl"] = target_kl
        model = InstrumentedPPO("MlpPolicy", env, verbose=0, seed=SEED, device="cpu", **kwargs)
        model.set_logger(configure(str(branch_root / "tensorboard"), ["tensorboard"]))
        model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=True, progress_bar=False, callback=callback_factory.callback)
        final_steps = int(model.num_timesteps)
        model.logger.close()
    except Exception:
        payload = {
            "branch": branch,
            "status": "FAILED",
            "started_at": start,
            "finished_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            "target_kl": target_kl,
            "saved_checkpoints": callback_factory.saved,
            "actual_steps": callback_factory.actual_steps,
            "error": traceback.format_exc(),
        }
        write_json(failure_path, payload)
        raise
    finally:
        if env is not None:
            env.close()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    missing = [checkpoint for checkpoint, path in expected.items() if not path.exists()]
    if missing:
        payload = {
            "branch": branch,
            "status": "FAILED_MISSING_CHECKPOINTS",
            "saved_checkpoints": callback_factory.saved,
            "actual_steps": callback_factory.actual_steps,
            "missing": missing,
        }
        write_json(failure_path, payload)
        raise RuntimeError(f"{branch}: missing checkpoints {missing}")
    manifest = {
        "branch": branch,
        "status": "completed",
        "training_seed": SEED,
        "training_from_scratch": True,
        "total_timesteps_requested": TOTAL_TIMESTEPS,
        "final_num_timesteps": final_steps,
        "target_kl": target_kl,
        "effective_kl_threshold": None if target_kl is None else target_kl * 1.5,
        "checkpoint_paths": {str(k): rel(v) for k, v in expected.items()},
        "checkpoint_actual_num_timesteps": {str(k): checkpoint_data(v).get("num_timesteps") for k, v in expected.items()},
        "checkpoint_callback_observed_steps": {str(k): v for k, v in callback_factory.actual_steps.items()},
        "ppo_kwargs": base03222.base.ppo_kwargs(config),
        "runtime_wrapper": "LateIrrigationReserveMaskWrapper via 040_28/040_36 chain",
        "scientific_contract_unchanged": True,
        "instrumentation_only": True,
        "started_at": start,
        "finished_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
    }
    write_json(manifest_path, manifest)
    return manifest


def extract_training_metrics(branch: str) -> pd.DataFrame:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = sorted((OUT / branch / "tensorboard").glob("**/events*"))
    if len(event_files) != 1:
        raise RuntimeError(f"{branch}: expected exactly one TensorBoard event file, got {len(event_files)}")
    accumulator = EventAccumulator(str(event_files[0]))
    accumulator.Reload()
    mapping = {
        "train/approx_kl": "approx_kl", "train/clip_fraction": "clip_fraction",
        "train/entropy_loss": "entropy_loss", "train/policy_gradient_loss": "policy_gradient_loss",
        "train/value_loss": "value_loss", "train/explained_variance": "explained_variance",
        "train/learning_rate": "learning_rate", "train/clip_range": "clip_range",
        "train/epochs_attempted": "epochs_attempted", "train/epochs_completed": "epochs_completed",
        "train/optimizer_steps_completed": "optimizer_steps_completed",
        "train/kl_guard_triggered": "kl_guard_triggered", "train/kl_at_trigger": "kl_at_trigger",
    }
    records: dict[int, dict[str, Any]] = {}
    for tag, name in mapping.items():
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        for event in accumulator.Scalars(tag):
            step = int(event.step)
            records.setdefault(step, {"global_step": step, "timestep": step})[name] = float(event.value)
    metrics = pd.DataFrame(sorted(records.values(), key=lambda item: item["global_step"]))
    for column in EXPECTED_METRIC_COLUMNS:
        if column not in metrics.columns:
            metrics[column] = np.nan
    metrics.insert(0, "branch", branch)
    metrics = metrics[["branch"] + EXPECTED_METRIC_COLUMNS]
    output = CONTROL_METRICS_OUT if branch == "control" else GUARD_METRICS_OUT
    metrics.to_csv(output, index=False, encoding="utf-8-sig")
    write_json(OUT / branch / "tensorboard_manifest.json", {
        "event_file": rel(event_files[0]),
        "event_file_sha256": sha256_file(event_files[0]),
        "scalar_tags": accumulator.Tags().get("scalars", []),
        "rows": int(len(metrics)),
        "missing_required_columns": [column for column in EXPECTED_METRIC_COLUMNS if column not in metrics or metrics[column].isna().all()],
    })
    return metrics


def diagnostic_snapshot(env: Any) -> Path:
    folder = getattr(env.unwrapped, "_tmp_folder", None)
    if not folder:
        raise RuntimeError("DSSAT temporary folder unavailable")
    snapshot = Path(folder)
    if not (snapshot / "Summary.OUT").exists():
        raise FileNotFoundError(snapshot / "Summary.OUT")
    return snapshot


def validation_daily(model: Any, base: Any, direct: Any, config: dict[str, Any], env_config: dict[str, Any], year: int, checkpoint: int, branch: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sb3_contrib.common.maskable.utils import get_action_masks
    import run_yc_ppo_seed1_kl_guard_paired_019 as kl019

    weather = direct.weather_for_daily(config)
    records: list[dict[str, Any]] = []
    env = base.make_env(config, env_config, STATION, int(year), SEED, f"025_{branch}_{year}_ckpt{checkpoint}_eval", evaluation=True)
    planting = pd.Timestamp(direct.find_year(env_config, STATION, int(year))["planting_date"])
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = base.latest_observation_dict(env, obs, info)
            dap_raw = direct.scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = base.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(STATION)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append({
                "station_code": STATION, "site": SITE, "year": int(year), "seed": SEED,
                "checkpoint_step": int(checkpoint), "checkpoint": int(checkpoint), "branch": branch,
                "algorithm": "MaskablePPO", "date": date.strftime("%Y-%m-%d"), "doy": int(date.dayofyear),
                "dap": dap, "rain": direct.scalar(wrow.get("rain"), np.nan), "srad": direct.scalar(wrow.get("srad"), np.nan),
                "tmax": direct.scalar(wrow.get("tmax"), np.nan), "tmin": direct.scalar(wrow.get("tmin"), np.nan),
                "swfac": direct.scalar(latest.get("swfac")), "nstres": direct.scalar(latest.get("nstres")),
                "topwt": direct.scalar(latest.get("topwt")), "grnwt": direct.scalar(latest.get("grnwt")),
                "xlai": direct.scalar(latest.get("xlai")), "reward": float(reward),
                **dict(env.last_action_info), "done": done,
                "info": json.dumps(info, ensure_ascii=False, default=str),
            })
            step_count += 1
        if not done:
            raise RuntimeError(f"validation did not finish: {branch} checkpoint={checkpoint} year={year}")
        snapshot = diagnostic_snapshot(env)
        daily = pd.DataFrame(records)
        final_yield = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
        total_i = float(pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0).sum())
        total_n = float(pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0).sum())
        strict = kl019.strict_metrics(snapshot, final_yield, total_i, total_n)
    finally:
        env.close()
    return pd.DataFrame(records), strict


def validate_branch(branch: str) -> pd.DataFrame:
    from sb3_contrib import MaskablePPO

    _, base03222, direct, config, env_config = runtime_setup(OUT)
    branch_root = OUT / branch
    validation_env_config = copy.deepcopy(env_config)
    validation_env_config["paths"]["output_root"] = rel(branch_root / "runtime_validation")
    daily_root = branch_root / "daily_outputs" / STATION
    daily_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        model_path = branch_checkpoint_path(branch, checkpoint)
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model = MaskablePPO.load(str(model_path), device="cpu")
        try:
            for year in VALIDATION_YEARS:
                daily, strict = validation_daily(model, base03222.base, direct, config, validation_env_config, year, checkpoint, branch)
                daily_path = daily_root / f"YCA_{year}_seed0_ckpt{checkpoint}_daily.csv"
                daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
                invalid = int((pd.to_numeric(daily["requested_discrete_action_index"], errors="coerce") != pd.to_numeric(daily["discrete_action_index"], errors="coerce")).sum())
                fallback = int(pd.Series(daily["mask_forced_noop"]).astype(str).str.lower().eq("true").sum())
                final_yield = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
                total_i = float(pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0).sum())
                total_n = float(pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0).sum())
                rows.append({
                    "row_type": "year", "branch": branch, "checkpoint_step": checkpoint, "checkpoint": checkpoint,
                    "model_num_timesteps": checkpoint_data(model_path).get("num_timesteps"), "year": year,
                    "yield_kg_ha": final_yield, "I_mm": total_i, "N_kg_ha": total_n,
                    "PFP_N": strict["PFP_N_kg_kg"], "WP_ET": strict["WP_ET_kg_m3"], "ETCP_mm": strict["etcp_mm"],
                    "invalid": invalid, "fallback": fallback, "invalid_count": invalid, "fallback_count": fallback,
                    "daily_path": rel(daily_path), "status": "PASS",
                })
        finally:
            del model
            gc.collect()
    detail = pd.DataFrame(rows)
    aggregates: list[dict[str, Any]] = []
    for checkpoint, group in detail.groupby("checkpoint_step", sort=True):
        aggregates.append({
            "row_type": "aggregate", "branch": branch, "checkpoint_step": int(checkpoint), "checkpoint": int(checkpoint),
            "model_num_timesteps": int(pd.to_numeric(group["model_num_timesteps"], errors="coerce").max()),
            "year": "ALL_2014_2023", "validation_year_count": int(len(group)),
            "mean_yield_kg_ha": float(group["yield_kg_ha"].mean()), "sd_yield_kg_ha": float(group["yield_kg_ha"].std(ddof=1)),
            "mean_I_mm": float(group["I_mm"].mean()), "mean_N_kg_ha": float(group["N_kg_ha"].mean()),
            "mean_PFP_N": float(group["PFP_N"].mean()), "mean_WP_ET": float(group["WP_ET"].mean()),
            "invalid": int(group["invalid"].sum()), "fallback": int(group["fallback"].sum()),
            "invalid_count_total": int(group["invalid"].sum()), "fallback_count_total": int(group["fallback"].sum()),
            "status": "PASS" if len(group) == len(VALIDATION_YEARS) else "FAIL",
        })
    output = CONTROL_VALIDATION_OUT if branch == "control" else GUARD_VALIDATION_OUT
    combined = pd.concat([detail, pd.DataFrame(aggregates)], ignore_index=True, sort=False)
    combined.to_csv(output, index=False, encoding="utf-8-sig")
    detail.to_csv(branch_root / "validation_by_year.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(aggregates)


def read_aggregate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    df = df[df["row_type"].eq("aggregate")].copy()
    for col in ["checkpoint_step", "mean_yield_kg_ha", "sd_yield_kg_ha", "mean_I_mm", "mean_N_kg_ha", "mean_PFP_N", "mean_WP_ET", "invalid_count_total", "fallback_count_total"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("checkpoint_step").reset_index(drop=True)


def make_windows(control: pd.DataFrame, guard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frames = {"control": control, "klguard": guard}
    for branch, df in frames.items():
        for lower, upper, label in [(0, 25_000, "0-25K"), (25_000, 50_000, "25-50K"), (50_000, 75_000, "50-75K"), (75_000, 100_000, "75-100K")]:
            group = df[df["global_step"].between(lower + 1, upper)]
            row: dict[str, Any] = {
                "branch": branch, "window": label, "lower_step": lower, "upper_step": upper,
                "update_count": int(len(group)),
            }
            for col in ["approx_kl", "clip_fraction", "entropy_loss", "policy_gradient_loss", "value_loss", "explained_variance"]:
                values = pd.to_numeric(group[col], errors="coerce").dropna() if col in group else pd.Series(dtype=float)
                row[f"{col}_mean"] = float(values.mean()) if len(values) else np.nan
                row[f"{col}_median"] = float(values.median()) if len(values) else np.nan
            for col in ["kl_guard_triggered", "epochs_completed", "optimizer_steps_completed"]:
                values = pd.to_numeric(group[col], errors="coerce").dropna() if col in group else pd.Series(dtype=float)
                row[f"{col}_mean"] = float(values.mean()) if len(values) else np.nan
            row["optimizer_step_fraction"] = float((pd.to_numeric(group["optimizer_steps_completed"], errors="coerce") > 0).mean()) if len(group) else np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(WINDOW_OUT, index=False, encoding="utf-8-sig")
    return out


def make_historical(control: pd.DataFrame) -> pd.DataFrame:
    historical = pd.read_csv(HISTORICAL_METRICS, keep_default_na=False)
    historical = historical[(historical["method"].eq("Original")) & (pd.to_numeric(historical["seed"], errors="coerce").eq(SEED))].copy()
    historical["checkpoint_step"] = pd.to_numeric(historical["checkpoint"], errors="coerce").astype(int)
    historical = historical[historical["checkpoint_step"].isin(CHECKPOINTS)]
    rows: list[dict[str, Any]] = []
    for _, old in historical.sort_values("checkpoint_step").iterrows():
        new = control[control["checkpoint_step"].eq(int(old["checkpoint_step"]))]
        if len(new) != 1:
            raise RuntimeError(f"Expected one new CONTROL aggregate at {old['checkpoint_step']}, got {len(new)}")
        new = new.iloc[0]
        rows.append({
            "checkpoint": int(old["checkpoint_step"]),
            "historical_mean_yield_kg_ha": float(old["mean_yield"]), "new_control_mean_yield_kg_ha": float(new["mean_yield_kg_ha"]),
            "delta_new_minus_historical_yield_kg_ha": float(new["mean_yield_kg_ha"]) - float(old["mean_yield"]),
            "historical_mean_I_mm": float(old["mean_irrigation"]), "new_control_mean_I_mm": float(new["mean_I_mm"]),
            "delta_new_minus_historical_I_mm": float(new["mean_I_mm"]) - float(old["mean_irrigation"]),
            "historical_mean_N_kg_ha": float(old["mean_n"]), "new_control_mean_N_kg_ha": float(new["mean_N_kg_ha"]),
            "delta_new_minus_historical_N_kg_ha": float(new["mean_N_kg_ha"]) - float(old["mean_n"]),
            "historical_validation_source": old["validation_source_file"],
            "historical_status": old["status"],
        })
    out = pd.DataFrame(rows)
    if len(out) != len(CHECKPOINTS):
        raise RuntimeError(f"Historical seed0 comparison incomplete: {len(out)} rows")
    out.to_csv(HISTORICAL_OUT, index=False, encoding="utf-8-sig")
    return out


def paired_summary(control: pd.DataFrame, guard: pd.DataFrame, control_metrics: pd.DataFrame, guard_metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for branch, validation, metrics in [("CONTROL", control, control_metrics), ("KL-GUARD", guard, guard_metrics)]:
        values = {int(row.checkpoint_step): float(row.mean_yield_kg_ha) for row in validation.itertuples()}
        best_pre = max(values[x] for x in [25_000, 50_000, 75_000])
        y100 = values[100_000]
        metric_late = metrics[metrics["global_step"].between(75_001, 100_000)]
        optimized = pd.to_numeric(metrics["optimizer_steps_completed"], errors="coerce")
        trigger = pd.to_numeric(metrics["kl_guard_triggered"], errors="coerce")
        row = {
            "branch": branch,
            "yield_25K": values[25_000], "yield_50K": values[50_000], "yield_75K": values[75_000], "yield_100K": y100,
            "best_pre100_yield": best_pre, "pre100_best_drop": best_pre - y100,
            "full_collapse": bool(y100 < 6500.0 or best_pre - y100 > 1500.0),
            "I_100K": float(validation.loc[validation["checkpoint_step"].eq(100_000), "mean_I_mm"].iloc[0]),
            "N_100K": float(validation.loc[validation["checkpoint_step"].eq(100_000), "mean_N_kg_ha"].iloc[0]),
            "approx_kl_75_100_mean": float(pd.to_numeric(metric_late["approx_kl"], errors="coerce").mean()),
            "approx_kl_75_100_median": float(pd.to_numeric(metric_late["approx_kl"], errors="coerce").median()),
            "trigger_rate": float(trigger.mean()),
            "optimizer_step_fraction": float((optimized > 0).mean()),
            "mean_completed_epochs": float(pd.to_numeric(metric_late["epochs_completed"], errors="coerce").mean()),
            "mean_optimizer_steps": float(optimized.mean()),
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    control_row = summary[summary["branch"].eq("CONTROL")].iloc[0]
    guard_row = summary[summary["branch"].eq("KL-GUARD")].iloc[0]
    delta_yield = float(guard_row["yield_100K"] - control_row["yield_100K"])
    delta_drop = float(guard_row["pre100_best_drop"] - control_row["pre100_best_drop"])
    summary["delta_yield_100K"] = delta_yield
    summary["delta_drop"] = delta_drop
    summary["comparison_role"] = "paired_training_seed0"
    summary.to_csv(PAIRED_OUT, index=False, encoding="utf-8-sig")
    return summary, {"delta_yield_100K": delta_yield, "delta_drop": delta_drop}


def decide(control: pd.DataFrame, guard: pd.DataFrame, guard_metrics: pd.DataFrame) -> dict[str, Any]:
    c = control.iloc[0]
    g = guard.iloc[0]
    c_noncollapse = not bool(c["full_collapse"])
    guard_frozen = float(g["optimizer_step_fraction"]) <= 0.0 or float(g["mean_optimizer_steps"]) <= 0.0
    harm_1 = bool(c_noncollapse and bool(g["full_collapse"]))
    harm_2 = bool(float(g["yield_100K"]) < 0.95 * float(c["yield_100K"]))
    harm_3 = bool(float(g["pre100_best_drop"]) > 700.0 and float(c["pre100_best_drop"]) <= 700.0)
    material_harm = bool(harm_1 or harm_2 or harm_3)
    guard_protection = bool(
        bool(c["full_collapse"]) and not bool(g["full_collapse"]) and
        float(g["yield_100K"]) >= 7500.0 and float(g["pre100_best_drop"]) <= 700.0 and not guard_frozen
    )
    improvement = float(g["yield_100K"] - c["yield_100K"])
    if guard_frozen or material_harm:
        decision = "SEED0_KL_GUARD_NOT_SUPPORTED"
    elif bool(c["full_collapse"]) and guard_protection:
        decision = "SEED0_KL_GUARD_CROSSSEED_PROMISING"
    elif bool(c["full_collapse"]) and improvement >= 1000.0:
        decision = "SEED0_KL_GUARD_PARTIAL_RESCUE"
    elif c_noncollapse:
        decision = "SEED0_CONTROL_NONCOLLAPSE_INCONCLUSIVE"
    else:
        decision = "SEED0_KL_GUARD_NOT_SUPPORTED"
    payload = {
        "decision": decision,
        "control_completed_to_100K": True,
        "klguard_completed_to_100K": True,
        "control_100K_mean_yield": float(c["yield_100K"]),
        "klguard_100K_mean_yield": float(g["yield_100K"]),
        "delta_yield_100K": improvement,
        "control_full_collapse": bool(c["full_collapse"]),
        "klguard_full_collapse": bool(g["full_collapse"]),
        "control_pre100_best_drop": float(c["pre100_best_drop"]),
        "klguard_pre100_best_drop": float(g["pre100_best_drop"]),
        "klguard_trigger_rate": float(g["trigger_rate"]),
        "klguard_mean_completed_epochs": float(g["mean_completed_epochs"]),
        "klguard_mean_optimizer_steps": float(g["mean_optimizer_steps"]),
        "klguard_optimizer_step_fraction": float(g["optimizer_step_fraction"]),
        "klguard_frozen": guard_frozen,
        "material_harm": material_harm,
        "material_harm_components": {"control_noncollapse_guard_collapse": harm_1, "guard_yield_below_95pct_control": harm_2, "guard_drop_gt700_control_drop_not_gt700": harm_3},
        "guard_protection_gate": guard_protection,
        "effective_kl_threshold": EFFECTIVE_KL_THRESHOLD,
        "configured_target_kl": CONFIGURED_TARGET_KL,
        "decision_rules_source": rel(PROMPT),
    }
    DECISION_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([payload]).to_csv(DECISION_OUT, index=False, encoding="utf-8-sig")
    write_json(OUT / "decision.json", payload)
    return payload


def classify_historical_relation(history: pd.DataFrame) -> str:
    yield_diff = pd.to_numeric(history["delta_new_minus_historical_yield_kg_ha"], errors="coerce").abs().max()
    i_diff = pd.to_numeric(history["delta_new_minus_historical_I_mm"], errors="coerce").abs().max()
    n_diff = pd.to_numeric(history["delta_new_minus_historical_N_kg_ha"], errors="coerce").abs().max()
    if yield_diff <= 1.0 and i_diff <= 1.0 and n_diff <= 1.0:
        return "exact-like"
    hist_y = history.sort_values("checkpoint")["historical_mean_yield_kg_ha"].to_numpy(float)
    new_y = history.sort_values("checkpoint")["new_control_mean_yield_kg_ha"].to_numpy(float)
    both_shape = bool(hist_y[0] > 7500 and hist_y[-1] < hist_y[0] - 1500 and new_y[0] > 7500 and new_y[-1] < new_y[0] - 1500)
    return "phenotype-consistent" if both_shape else "nonreproduced"


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    work = df[cols].copy()
    for col in work.columns:
        if pd.api.types.is_numeric_dtype(work[col]):
            work[col] = work[col].round(6)
    return work.to_markdown(index=False)


def write_report(decision: dict[str, Any], control: pd.DataFrame, guard: pd.DataFrame, windows: pd.DataFrame, history: pd.DataFrame, paired: pd.DataFrame, control_metrics: pd.DataFrame, guard_metrics: pd.DataFrame) -> None:
    relation = classify_historical_relation(history)
    late_c = windows[(windows["branch"] == "control") & (windows["window"] == "75-100K")].iloc[0]
    late_g = windows[(windows["branch"] == "klguard") & (windows["window"] == "75-100K")].iloc[0]
    guard_not_frozen = not bool(decision["klguard_frozen"])
    lines = [
        "# YC PPO Original seed0 fixed KL-GUARD cross-training-seed replication",
        "",
        "## Status",
        "",
        f"- Primary decision: `{decision['decision']}`.",
        "- `YC_PPO_ORIGINAL_SEED0_FIXED_KLGUARD_CROSSSEED_COMPLETE`",
        "- This is a paired-by-training-seed comparison: both branches start from a new PPO seed0 training run; after a KL early-stop intervention changes epoch count, exact stepwise RNG parity is not claimed.",
        "",
        "## Contract and provenance",
        "",
        f"- Prompt: `{rel(PROMPT)}`.",
        f"- Historical Original seed0 contract/config source: `{rel(HISTORICAL_CONFIG)}`.",
        f"- Runtime wrapper: `LateIrrigationReserveMaskWrapper via 040_28/040_36 chain`; station/site `{STATION}/{SITE}`; input profile `lowIC`.",
        f"- Training seed: `{SEED}`; train years `{TRAIN_YEARS}`; validation years `{VALIDATION_YEARS}`.",
        "- Observation: raw `046_02_raw_observation`, no normalization and no weather forecast. Action grid: irrigation `[0,15,30,45]` mm × nitrogen `[0,40,80,120]` kg/ha. Reward, safety masks, irrigation/fertilizer constraints, IC, cultivar, network and PPO hyperparameters are inherited from the existing 055/042_10/032_22 path.",
        f"- Fixed guard: configured `target_kl={CONFIGURED_TARGET_KL:.10g}`; source condition `{verify_guard_semantics()['source_code_condition']}`; effective threshold `{EFFECTIVE_KL_THRESHOLD:.1e}`.",
        "- The only scientific intervention is fixed KL-based update early stop. The new file is an orchestration/provenance layer; existing PPO/DSSAT source files were not edited.",
        "",
        "## Smoke gate",
        "",
        f"- Evidence: `{rel(OUT / 'smoke_manifest.json')}`; both branches passed the isolated 2K smoke before formal training.",
        "",
        "## Formal training",
        "",
        f"- CONTROL manifest: `{rel(OUT / 'control' / 'training_manifest.json')}`; metrics `{rel(CONTROL_METRICS_OUT)}` ({len(control_metrics)} rows).",
        f"- KL-GUARD manifest: `{rel(OUT / 'klguard' / 'training_manifest.json')}`; metrics `{rel(GUARD_METRICS_OUT)}` ({len(guard_metrics)} rows).",
        f"- New CONTROL completed to 100K: `{decision['control_completed_to_100K']}`. New KL-GUARD completed to 100K: `{decision['klguard_completed_to_100K']}`.",
        f"- KL-GUARD trigger rate `{decision['klguard_trigger_rate']:.6f}`; mean completed epochs `{decision['klguard_mean_completed_epochs']:.6f}`; mean optimizer steps `{decision['klguard_mean_optimizer_steps']:.6f}`; optimizer-step fraction `{decision['klguard_optimizer_step_fraction']:.6f}`; real optimizer updates/non-frozen `{guard_not_frozen}`.",
        "",
        "## Checkpoint validation",
        "",
        f"- CONTROL validation: `{rel(CONTROL_VALIDATION_OUT)}`.",
        f"- KL-GUARD validation: `{rel(GUARD_VALIDATION_OUT)}`.",
        "",
        md_table(pd.concat([control.assign(branch="CONTROL"), guard.assign(branch="KL-GUARD")], ignore_index=True), ["branch", "checkpoint_step", "mean_yield_kg_ha", "sd_yield_kg_ha", "mean_I_mm", "mean_N_kg_ha", "mean_PFP_N", "mean_WP_ET", "invalid", "fallback"]),
        "",
        "## Collapse and paired result",
        "",
        f"- CONTROL 100K mean yield `{decision['control_100K_mean_yield']:.3f}` kg/ha; full collapse `{decision['control_full_collapse']}`; pre-100K best drop `{decision['control_pre100_best_drop']:.3f}` kg/ha.",
        f"- KL-GUARD 100K mean yield `{decision['klguard_100K_mean_yield']:.3f}` kg/ha; full collapse `{decision['klguard_full_collapse']}`; pre-100K best drop `{decision['klguard_pre100_best_drop']:.3f}` kg/ha.",
        f"- 100K yield improvement `KL-GUARD - CONTROL = {decision['delta_yield_100K']:.3f}` kg/ha.",
        f"- Material harm `{decision['material_harm']}`; guard frozen `{decision['klguard_frozen']}`.",
        f"- Paired summary: `{rel(PAIRED_OUT)}`.",
        "",
        md_table(paired, ["branch", "yield_25K", "yield_50K", "yield_75K", "yield_100K", "best_pre100_yield", "pre100_best_drop", "full_collapse", "I_100K", "N_100K", "approx_kl_75_100_mean", "approx_kl_75_100_median", "trigger_rate", "optimizer_step_fraction", "delta_yield_100K", "delta_drop"]),
        "",
        "## Update-window description",
        "",
        f"- Window output: `{rel(WINDOW_OUT)}`.",
        f"- In 75–100K, CONTROL approx_kl mean/median `{late_c['approx_kl_mean']:.8f}/{late_c['approx_kl_median']:.8f}` versus KL-GUARD `{late_g['approx_kl_mean']:.8f}/{late_g['approx_kl_median']:.8f}`; clip_fraction mean/median `{late_c['clip_fraction_mean']:.8f}/{late_c['clip_fraction_median']:.8f}` versus `{late_g['clip_fraction_mean']:.8f}/{late_g['clip_fraction_median']:.8f}`.",
        "- These are descriptive logged update metrics. Any reduction in update magnitude is not treated as proof that KL is the unique causal root of collapse.",
        "",
        "## Historical seed0 consistency check",
        "",
        f"- Historical comparison output: `{rel(HISTORICAL_OUT)}`; source `{rel(HISTORICAL_METRICS)}`.",
        f"- Relation classification: `{relation}`. This compares historical Original seed0 against the new CONTROL only; the primary paired comparator remains new CONTROL versus new KL-GUARD.",
        "- Classification rule used here: `exact-like` requires all four yield, I and N absolute differences to be ≤1 in their native units; otherwise `phenotype-consistent` requires both trajectories to start above 7500 kg/ha and lose >1500 kg/ha by 100K; otherwise `nonreproduced`.",
        "",
        "## Required answers",
        "",
        f"1. New CONTROL seed0 complete to 100K: `{decision['control_completed_to_100K']}`.",
        f"2. New KL-GUARD seed0 complete to 100K: `{decision['klguard_completed_to_100K']}`.",
        f"3. New CONTROL reproduces early-high/late-degraded phenotype: `{relation != 'nonreproduced'}` (`{relation}`).",
        f"4. New CONTROL 100K yield/full-collapse: `{decision['control_100K_mean_yield']:.3f}` kg/ha / `{decision['control_full_collapse']}`.",
        f"5. KL-GUARD 100K yield/full-collapse: `{decision['klguard_100K_mean_yield']:.3f}` kg/ha / `{decision['klguard_full_collapse']}`.",
        f"6. 100K yield improvement: `{decision['delta_yield_100K']:.3f}` kg/ha.",
        f"7. KL-GUARD triggered and not frozen: trigger rate `{decision['klguard_trigger_rate']:.6f}`, non-frozen `{guard_not_frozen}`.",
        f"8. 75–100K update magnitude lowered: see paired descriptive window means/medians above; approx_kl mean delta (guard-control) `{late_g['approx_kl_mean'] - late_c['approx_kl_mean']:.8f}` and median delta `{late_g['approx_kl_median'] - late_c['approx_kl_median']:.8f}`.",
        f"9. Historical seed0 versus new CONTROL: `{relation}`.",
        f"10. Final allowed decision: `{decision['decision']}`.",
        "",
        "## Stop boundary",
        "",
        "No Original seed2, Augmented, other site, RNG expansion, KL-threshold tuning, PPO hyperparameter tuning, or new intervention was started by this task.",
        "",
        "YC_PPO_ORIGINAL_SEED0_FIXED_KLGUARD_CROSSSEED_COMPLETE",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_report() -> dict[str, Any]:
    control_metrics = extract_training_metrics("control")
    guard_metrics = extract_training_metrics("klguard")
    control = read_aggregate(CONTROL_VALIDATION_OUT)
    guard = read_aggregate(GUARD_VALIDATION_OUT)
    if len(control) != 4 or len(guard) != 4:
        raise RuntimeError("Validation aggregate rows must contain exactly four checkpoints per branch")
    windows = make_windows(control_metrics, guard_metrics)
    history = make_historical(control)
    paired, _ = paired_summary(control, guard, control_metrics, guard_metrics)
    decision = decide(paired[paired["branch"].eq("CONTROL")], paired[paired["branch"].eq("KL-GUARD")], guard_metrics)
    write_report(decision, control, guard, windows, history, paired, control_metrics, guard_metrics)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["preflight", "smoke", "train", "validate", "report", "all"], default="all")
    parser.add_argument("--branch", choices=["control", "klguard", "all"], default="all")
    args = parser.parse_args()
    if args.phase == "preflight":
        print(json.dumps(preflight(), ensure_ascii=False, indent=2))
        return 0
    if args.phase == "smoke":
        print(json.dumps(run_smoke(), ensure_ascii=False, indent=2))
        return 0
    if args.phase == "train":
        branches = ["control", "klguard"] if args.branch == "all" else [args.branch]
        results = {branch: train_branch(branch) for branch in branches}
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    if args.phase == "validate":
        branches = ["control", "klguard"] if args.branch == "all" else [args.branch]
        results = {branch: len(validate_branch(branch)) for branch in branches}
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    if args.phase == "report":
        print(json.dumps(run_report(), ensure_ascii=False, indent=2))
        return 0
    # all is deliberately explicit and serial to limit memory pressure.
    run_smoke()
    train_branch("control")
    train_branch("klguard")
    validate_branch("control")
    validate_branch("klguard")
    print(json.dumps(run_report(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
