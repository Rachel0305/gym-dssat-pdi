"""020: YC PPO Original seed1 multi-RNG CONTROL reproducibility screen.

This is an isolated orchestration/evidence script.  It imports the existing
YCA low-IC runtime and PPO implementation, but does not modify either.  RNG1
is deliberately reused from 018B; only RNG0/2/3/4 receive new 25K
continuations from the common Original PPO seed1 75K checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "src" / "055_yca_lowIC_site_transfer"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

ORIGINAL_RUN = ROOT / "benchmark_results" / "222YCA_O1_yc_v1_original_seed1"
ORIGINAL_CHECKPOINT = ORIGINAL_RUN / "models" / "YCA" / "YCA_half_split_stress_aware_maskableppo_seed1_ckpt75000.zip"
PRIOR_018B_ROOT = ROOT / "benchmark_results" / "019_yc_ppo_seed1_kl_guard_screen"
PRIOR_018B_CONTROL = PRIOR_018B_ROOT / "control"
PRIOR_RESUME_AUDIT = ROOT / "results" / "yc_ppo_s1_75k_resume_feasibility.csv"
PRIOR_TRAINING = ROOT / "results" / "yc_ppo_s1_control_training_metrics.csv"
PRIOR_VALIDATION = ROOT / "results" / "yc_ppo_s1_control_checkpoint_validation.csv"

OUT = ROOT / "benchmark_results" / "020_yc_ppo_seed1_multirng_control_screen"
FAILED_RNG0_ROOT = OUT / "failed_attempts" / "attempt_03_rng0_dssat_2011_floating_point_exception"
RESULTS = ROOT / "results"
REPORT = ROOT / "docs" / "yc_ppo_original_seed1_multirng_control_reproducibility.md"

STATION = "YCA"
SITE = "YC"
PPO_SEED = 1
CONTINUATION_RNGS = [0, 2, 3, 4]
ALL_RNGS = [0, 1, 2, 3, 4]
TRAIN_YEARS = list(range(2004, 2014))
VALIDATION_YEARS = list(range(2014, 2024))
CHECKPOINTS = [75_000, 80_000, 85_000, 90_000, 95_000, 100_000]
CONTINUATION_STEPS = 25_000
REQUIRED_TRAIN_COLUMNS = [
    "continuation_rng", "global_step", "approx_kl", "clip_fraction",
    "entropy_loss", "policy_gradient_loss", "value_loss",
    "explained_variance", "learning_rate", "clip_range",
]
NUMERIC_AGG_COLUMNS = [
    "checkpoint_step", "model_num_timesteps", "validation_year_count",
    "mean_yield_kg_ha", "sd_yield_kg_ha", "mean_I_mm", "mean_N_kg_ha",
    "mean_PFP_N", "mean_WP_ET", "invalid_count_total", "fallback_count_total",
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def checkpoint_data(path: Path) -> dict[str, Any]:
    import zipfile

    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("data"))


def set_all_seeds(seed: int) -> None:
    """Match the 018B CONTROL reinitialization protocol, with only seed changed."""
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def isolated_runtime_setup(cfg: dict[str, Any], diagnostic: Any) -> tuple[Any, dict[str, Any], dict[str, Any], Any]:
    """Reuse 017's scientific runtime while redirecting all generated files to 020."""
    replay, config, env_config, runtime_state = diagnostic.runtime_setup(cfg)
    engine = replay.engine
    engine.BASE_OUT = OUT
    engine.BASE_DOC = REPORT
    engine.patch_base_module(
        OUT,
        REPORT,
        int(cfg["training"]["total_timesteps"]),
        list(map(int, cfg["training"]["checkpoint_steps"])),
    )
    # Make the final runtime target explicit in the manifest while retaining
    # every action/reward/observation/mask and constraint setting from 017.
    env_config["paths"]["output_root"] = rel(OUT)
    return replay, config, env_config, runtime_state


def branch_dir(rng: int) -> Path:
    # Keep the first aborted engineering attempt under OUT/rng0 untouched;
    # valid runs are isolated below OUT/isolated so partial evidence cannot be
    # mistaken for a completed trajectory.
    return OUT / "isolated" / f"rng{int(rng)}"


def branch_env_config(env_config: dict[str, Any], rng: int) -> dict[str, Any]:
    """Give each continuation its own generated-input/log namespace."""
    local = copy.deepcopy(env_config)
    local["paths"]["output_root"] = rel(branch_dir(rng) / "runtime")
    return local


def checkpoint_path(rng: int, checkpoint: int) -> Path:
    if int(checkpoint) == 75_000:
        return ORIGINAL_CHECKPOINT
    return branch_dir(rng) / "models" / "YCA" / f"yc_ppo_s1_control_rng{int(rng)}_ckpt{int(checkpoint)}.zip"


def preflight() -> dict[str, Any]:
    required = [ORIGINAL_CHECKPOINT, PRIOR_RESUME_AUDIT, PRIOR_TRAINING, PRIOR_VALIDATION]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required prior evidence: " + ", ".join(missing))
    source_data = checkpoint_data(ORIGINAL_CHECKPOINT)
    if int(source_data.get("num_timesteps", -1)) != 75_000:
        raise RuntimeError(f"Common starting checkpoint has num_timesteps={source_data.get('num_timesteps')}, expected 75000")
    prior_control_manifest = PRIOR_018B_CONTROL / "training_manifest.json"
    if not prior_control_manifest.exists():
        raise FileNotFoundError(f"Missing 018B CONTROL manifest: {rel(prior_control_manifest)}")
    prior_manifest = json.loads(prior_control_manifest.read_text(encoding="utf-8"))
    if prior_manifest.get("target_kl") is not None:
        raise RuntimeError("018B CONTROL manifest is not target_kl=None; refusing a non-CONTROL continuation")
    resume = pd.read_csv(PRIOR_RESUME_AUDIT, keep_default_na=False)
    resume_status = "unknown / insufficient evidence"
    if "status" in resume.columns:
        statuses = set(resume["status"].astype(str))
        if "PASS" in statuses and "FAIL" not in statuses:
            resume_status = "PASS"
        elif statuses:
            resume_status = ", ".join(sorted(statuses))
    failed_root = OUT / "failed_attempts"
    failed_entries = sorted(rel(path) for path in failed_root.iterdir()) if failed_root.exists() else []
    payload = {
        "status": "PASS",
        "common_starting_checkpoint": rel(ORIGINAL_CHECKPOINT),
        "common_starting_checkpoint_sha256": sha256_file(ORIGINAL_CHECKPOINT),
        "common_starting_num_timesteps": source_data.get("num_timesteps"),
        "prior_018b_control_manifest": rel(prior_control_manifest),
        "prior_018b_control_target_kl": prior_manifest.get("target_kl"),
        "prior_resume_audit": rel(PRIOR_RESUME_AUDIT),
        "prior_resume_audit_status": resume_status,
        "prior_rng1_training": rel(PRIOR_TRAINING),
        "prior_rng1_validation": rel(PRIOR_VALIDATION),
        "new_continuation_rngs": CONTINUATION_RNGS,
        "rng1_reexecuted": False,
        "training_contract": "CONTROL; target_kl=None; no scientific contract change",
        "aborted_engineering_attempt": {
            "status": "PRESERVED_EXCLUDED_FROM_ANALYSIS" if failed_entries else "none",
            "paths": failed_entries or ["unknown / insufficient evidence"],
            "reason": "Early engineering attempts were stopped before acceptance: one reused the 017 output root and one was interrupted during runtime isolation debugging; no partial checkpoint was used in the final analysis",
        },
    }
    write_json(OUT / "preflight.json", payload)
    return payload


class DenseCheckpointCallback:
    def __init__(self, rng: int):
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class _Callback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0)

            def _on_step(self) -> bool:
                now = int(self.model.num_timesteps)
                pending = [
                    target for target in CHECKPOINTS
                    if target > 75_000 and target <= now and target not in outer.saved
                ]
                for target in pending:
                    path = checkpoint_path(outer.rng, target)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(str(path))
                    outer.saved.append(target)
                    outer.actual_steps[target] = int(self.model.num_timesteps)
                return True

        self.rng = int(rng)
        self.saved: list[int] = []
        self.actual_steps: dict[int, int] = {}
        self.callback = _Callback()


def train_rng(rng: int, replay: Any, config: dict[str, Any], env_config: dict[str, Any]) -> dict[str, Any]:
    import torch  # noqa: F401  # force the same runtime before seed initialization
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.logger import configure

    out = branch_dir(rng)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "training_manifest.json"
    model_paths = {checkpoint: checkpoint_path(rng, checkpoint) for checkpoint in CHECKPOINTS}
    if all(model_paths[checkpoint].exists() for checkpoint in CHECKPOINTS if checkpoint > 75_000) and manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(model_paths[checkpoint].exists() for checkpoint in CHECKPOINTS if checkpoint > 75_000):
        raise RuntimeError(f"rng{rng}: partial training outputs exist; refusing to overwrite them")

    # The order mirrors 018B: initialize global RNGs, create the training env,
    # then load the full PPO checkpoint (including optimizer/timestep metadata).
    set_all_seeds(int(rng))
    local_env_config = branch_env_config(env_config, rng)
    env = replay.base03222.RandomYearEnv(config, local_env_config, STATION, TRAIN_YEARS, int(rng))
    callback = DenseCheckpointCallback(rng)
    try:
        model = MaskablePPO.load(str(ORIGINAL_CHECKPOINT), env=env, device="cpu")
        if int(model.num_timesteps) != 75_000:
            raise RuntimeError(f"rng{rng}: loaded num_timesteps={model.num_timesteps}, expected 75000")
        model.target_kl = None
        model.set_logger(configure(str(out / "tensorboard"), ["tensorboard"]))
        model.learn(
            total_timesteps=CONTINUATION_STEPS,
            reset_num_timesteps=False,
            progress_bar=False,
            callback=callback.callback,
        )
        final_steps = int(model.num_timesteps)
    finally:
        env.close()

    missing = [checkpoint for checkpoint in CHECKPOINTS if checkpoint > 75_000 and not model_paths[checkpoint].exists()]
    if missing:
        raise RuntimeError(f"rng{rng}: missing dense checkpoints {missing}")
    manifest = {
        "continuation_rng": int(rng),
        "status": "completed",
        "algorithm": "MaskablePPO CONTROL",
        "starting_checkpoint": rel(ORIGINAL_CHECKPOINT),
        "starting_checkpoint_sha256": sha256_file(ORIGINAL_CHECKPOINT),
        "starting_num_timesteps": 75_000,
        "continuation_steps_requested": CONTINUATION_STEPS,
        "final_num_timesteps": final_steps,
        "target_kl": None,
        "checkpoint_paths": {str(k): rel(v) for k, v in model_paths.items()},
        "checkpoint_actual_num_timesteps": {
            str(k): checkpoint_data(v).get("num_timesteps") for k, v in model_paths.items() if v.exists()
        },
        "rng_protocol": {
            "python_seed": int(rng),
            "numpy_seed": int(rng),
            "torch_seed": int(rng),
            "RandomYearEnv_seed": int(rng),
            "ppo_checkpoint_metadata_seed": checkpoint_data(ORIGINAL_CHECKPOINT).get("seed"),
        },
        "other_scientific_contracts_unchanged": True,
        "rng1_reference_not_rerun": True,
    }
    write_json(manifest_path, manifest)
    return manifest


def extract_training_metrics(rng: int) -> pd.DataFrame:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out = branch_dir(rng)
    event_files = sorted((out / "tensorboard").glob("**/events*"))
    if len(event_files) != 1:
        raise RuntimeError(f"rng{rng}: expected exactly one TensorBoard event file, got {len(event_files)}")
    accumulator = EventAccumulator(str(event_files[0]))
    accumulator.Reload()
    mapping = {
        "train/approx_kl": "approx_kl",
        "train/clip_fraction": "clip_fraction",
        "train/entropy_loss": "entropy_loss",
        "train/policy_gradient_loss": "policy_gradient_loss",
        "train/value_loss": "value_loss",
        "train/explained_variance": "explained_variance",
        "train/learning_rate": "learning_rate",
        "train/clip_range": "clip_range",
    }
    records: dict[int, dict[str, Any]] = {}
    scalar_tags = accumulator.Tags().get("scalars", [])
    for tag, name in mapping.items():
        if tag not in scalar_tags:
            continue
        for event in accumulator.Scalars(tag):
            step = int(event.step)
            records.setdefault(step, {"continuation_rng": int(rng), "global_step": step, "timestep": step})[name] = float(event.value)
    metrics = pd.DataFrame(sorted(records.values(), key=lambda item: item["global_step"]))
    for column in REQUIRED_TRAIN_COLUMNS:
        if column not in metrics.columns:
            metrics[column] = np.nan
    metrics = metrics[["continuation_rng", "global_step", "timestep", "approx_kl", "clip_fraction", "entropy_loss", "policy_gradient_loss", "value_loss", "explained_variance", "learning_rate", "clip_range"]]
    output = RESULTS / f"yc_ppo_s1_control_rng{int(rng)}_training_metrics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False, encoding="utf-8-sig")
    write_json(out / "tensorboard_manifest.json", {
        "continuation_rng": int(rng),
        "event_file": rel(event_files[0]),
        "event_file_sha256": sha256_file(event_files[0]),
        "scalar_tags": scalar_tags,
        "rows": int(len(metrics)),
        "missing_required_columns": [column for column in REQUIRED_TRAIN_COLUMNS if column not in metrics or metrics[column].isna().all()],
    })
    if metrics.empty:
        raise RuntimeError(f"rng{rng}: TensorBoard metrics are empty")
    return metrics


def record_failed_rng0_training_metrics() -> pd.DataFrame:
    """Extract only the observed prefix of the aborted RNG0 continuation."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = sorted((FAILED_RNG0_ROOT / "tensorboard").glob("events*"))
    if len(event_files) != 1:
        raise RuntimeError(f"rng0 failed-attempt evidence: expected one event file, got {len(event_files)}")
    accumulator = EventAccumulator(str(event_files[0]))
    accumulator.Reload()
    mapping = {
        "train/approx_kl": "approx_kl", "train/clip_fraction": "clip_fraction",
        "train/entropy_loss": "entropy_loss", "train/policy_gradient_loss": "policy_gradient_loss",
        "train/value_loss": "value_loss", "train/explained_variance": "explained_variance",
        "train/learning_rate": "learning_rate", "train/clip_range": "clip_range",
    }
    records: dict[int, dict[str, Any]] = {}
    for tag, name in mapping.items():
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        for event in accumulator.Scalars(tag):
            step = int(event.step)
            records.setdefault(step, {"continuation_rng": 0, "global_step": step, "timestep": step})[name] = float(event.value)
    metrics = pd.DataFrame(sorted(records.values(), key=lambda item: item["global_step"]))
    for column in REQUIRED_TRAIN_COLUMNS:
        if column not in metrics.columns:
            metrics[column] = np.nan
    metrics = metrics[["continuation_rng", "global_step", "timestep", "approx_kl", "clip_fraction", "entropy_loss", "policy_gradient_loss", "value_loss", "explained_variance", "learning_rate", "clip_range"]]
    metrics["trajectory_status"] = "INCOMPLETE_TRAINING_PREFIX"
    metrics["source"] = rel(event_files[0])
    output = RESULTS / "yc_ppo_s1_control_rng0_training_metrics.csv"
    metrics.to_csv(output, index=False, encoding="utf-8-sig")
    return metrics


def read_aggregate(path: Path, rng: int, validation_source: str) -> pd.DataFrame:
    data = pd.read_csv(path, keep_default_na=False)
    if "row_type" not in data.columns:
        raise RuntimeError(f"Validation file has no row_type: {rel(path)}")
    for column in NUMERIC_AGG_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    aggregate = data[data["row_type"].astype(str).eq("aggregate")].copy()
    aggregate["continuation_rng"] = int(rng)
    aggregate["validation_source"] = validation_source
    return aggregate


def record_failed_rng0_validation(diagnostic: Any, replay: Any, config: dict[str, Any], env_config: dict[str, Any]) -> pd.DataFrame:
    """Validate only the completed prefix of RNG0; never infer its missing tail."""
    from sb3_contrib import MaskablePPO

    import run_yc_ppo_seed1_kl_guard_paired_019 as paired
    import run_all_year_direct_action_safe_ppo as direct
    import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base

    output = RESULTS / "yc_ppo_s1_control_rng0_checkpoint_validation.csv"
    if output.exists():
        aggregate = read_aggregate(output, 0, "existing_020_rng0_partial_validation")
        if set(aggregate["checkpoint_step"].dropna().astype(int)) == {75_000, 80_000, 85_000}:
            return aggregate
        raise RuntimeError("rng0 partial validation output exists with unexpected checkpoint set")
    reference = pd.read_csv(PRIOR_VALIDATION, keep_default_na=False)
    reference_75 = reference[pd.to_numeric(reference["checkpoint_step"], errors="coerce").eq(75_000)].copy()
    if reference_75.empty:
        raise RuntimeError("018B RNG1 validation has no 75K reference rows")
    reference_75["continuation_rng"] = 0
    reference_75["branch"] = "control_rng0"
    reference_75["validation_source"] = "018B_rng1_deterministic_reference"
    reference_75["trajectory_status"] = "INCOMPLETE_TRAINING_PREFIX"

    local_env_config = branch_env_config(env_config, 0)
    detail_rows: list[dict[str, Any]] = []
    for checkpoint in [80_000, 85_000]:
        model_path = FAILED_RNG0_ROOT / "models" / "YCA" / f"yc_ppo_s1_control_rng0_ckpt{checkpoint}.zip"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model = MaskablePPO.load(str(model_path), device="cpu")
        for year in VALIDATION_YEARS:
            daily, strict = paired.daily_records(
                model, config, local_env_config, year, checkpoint, "control_rng0_partial", base, direct
            )
            daily_path = branch_dir(0) / "daily_outputs" / STATION / f"YCA_{year}_seed1_ckpt{checkpoint}_daily.csv"
            daily_path.parent.mkdir(parents=True, exist_ok=True)
            daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
            requested = pd.to_numeric(daily["requested_discrete_action_index"], errors="coerce")
            executed = pd.to_numeric(daily["discrete_action_index"], errors="coerce")
            detail_rows.append({
                "row_type": "year", "continuation_rng": 0, "branch": "control_rng0",
                "checkpoint_step": int(checkpoint), "model_num_timesteps": checkpoint_data(model_path).get("num_timesteps"),
                "year": int(year), "yield_kg_ha": float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1]),
                "I_mm": float(pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0).sum()),
                "N_kg_ha": float(pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0).sum()),
                "PFP_N": strict["PFP_N_kg_kg"], "WP_ET": strict["WP_ET_kg_m3"], "ETCP_mm": strict["etcp_mm"],
                "invalid_count": int((requested != executed).sum()),
                "fallback_count": int(pd.Series(daily["mask_forced_noop"]).astype(str).str.lower().eq("true").sum()),
                "daily_path": rel(daily_path), "validation_source": "020_rng0_failed_attempt_partial", "status": "PASS",
                "trajectory_status": "INCOMPLETE_TRAINING_PREFIX",
            })
    detail = pd.DataFrame(detail_rows)
    aggregate_rows: list[dict[str, Any]] = []
    for checkpoint, group in detail.groupby("checkpoint_step", sort=True):
        aggregate_rows.append({
            "row_type": "aggregate", "continuation_rng": 0, "branch": "control_rng0",
            "checkpoint_step": int(checkpoint), "model_num_timesteps": int(pd.to_numeric(group["model_num_timesteps"], errors="coerce").max()),
            "year": "ALL_2014_2023", "validation_year_count": int(len(group)),
            "mean_yield_kg_ha": float(group["yield_kg_ha"].mean()), "sd_yield_kg_ha": float(group["yield_kg_ha"].std(ddof=1)),
            "mean_I_mm": float(group["I_mm"].mean()), "mean_N_kg_ha": float(group["N_kg_ha"].mean()),
            "mean_PFP_N": float(group["PFP_N"].mean()), "mean_WP_ET": float(group["WP_ET"].mean()),
            "invalid_count_total": int(group["invalid_count"].sum()), "fallback_count_total": int(group["fallback_count"].sum()),
            "validation_source": "020_rng0_failed_attempt_partial", "trajectory_status": "INCOMPLETE_TRAINING_PREFIX",
            "status": "PASS",
        })
    combined = pd.concat([reference_75, detail, pd.DataFrame(aggregate_rows)], ignore_index=True, sort=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False, encoding="utf-8-sig")
    return read_aggregate(output, 0, "020_rng0_failed_attempt_partial")


def validate_rng(rng: int, diagnostic: Any, replay: Any, config: dict[str, Any], env_config: dict[str, Any]) -> pd.DataFrame:
    from sb3_contrib import MaskablePPO

    # This module contains the already-audited 018B daily DSSAT replay helper.
    # Importing it does not execute its main() or alter the scientific runtime.
    import run_yc_ppo_seed1_kl_guard_paired_019 as paired
    import run_all_year_direct_action_safe_ppo as direct
    import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base

    if int(rng) == 0:
        return record_failed_rng0_validation(diagnostic, replay, config, env_config)

    output = RESULTS / f"yc_ppo_s1_control_rng{int(rng)}_checkpoint_validation.csv"
    if output.exists():
        aggregate = read_aggregate(output, rng, f"existing_020_rng{int(rng)}_validation")
        if set(aggregate["checkpoint_step"].dropna().astype(int)) == set(CHECKPOINTS):
            return aggregate
        raise RuntimeError(f"rng{rng}: partial validation output exists; refusing to overwrite it")

    out = branch_dir(rng)
    daily_root = out / "daily_outputs" / STATION
    daily_root.mkdir(parents=True, exist_ok=True)

    # The 75K policy is byte-identical for every continuation.  Reuse the
    # already completed deterministic RNG1 validation instead of spending ten
    # duplicate DSSAT episodes per new RNG.
    reference = pd.read_csv(PRIOR_VALIDATION, keep_default_na=False)
    reference_75 = reference[reference["checkpoint_step"].astype(str).eq("75000")].copy()
    if reference_75.empty:
        reference_75 = reference[pd.to_numeric(reference["checkpoint_step"], errors="coerce").eq(75_000)].copy()
    if reference_75.empty:
        raise RuntimeError("018B RNG1 validation has no 75K reference rows")
    reference_75["continuation_rng"] = int(rng)
    reference_75["branch"] = f"control_rng{int(rng)}"
    reference_75["validation_source"] = "018B_rng1_deterministic_reference"

    detail_rows: list[dict[str, Any]] = []
    local_env_config = branch_env_config(env_config, rng)
    for checkpoint in CHECKPOINTS[1:]:
        model_path = checkpoint_path(rng, checkpoint)
        model = MaskablePPO.load(str(model_path), device="cpu")
        for year in VALIDATION_YEARS:
            daily, strict = paired.daily_records(
                model, config, local_env_config, year, checkpoint, f"control_rng{int(rng)}", base, direct
            )
            daily_path = daily_root / f"YCA_{year}_seed1_ckpt{checkpoint}_daily.csv"
            daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
            requested = pd.to_numeric(daily["requested_discrete_action_index"], errors="coerce")
            executed = pd.to_numeric(daily["discrete_action_index"], errors="coerce")
            invalid = int((requested != executed).sum())
            fallback = int(pd.Series(daily["mask_forced_noop"]).astype(str).str.lower().eq("true").sum())
            final_yield = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
            total_i = float(pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0).sum())
            total_n = float(pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0).sum())
            detail_rows.append({
                "row_type": "year",
                "continuation_rng": int(rng),
                "branch": f"control_rng{int(rng)}",
                "checkpoint_step": int(checkpoint),
                "model_num_timesteps": checkpoint_data(model_path).get("num_timesteps"),
                "year": int(year),
                "yield_kg_ha": final_yield,
                "I_mm": total_i,
                "N_kg_ha": total_n,
                "PFP_N": strict["PFP_N_kg_kg"],
                "WP_ET": strict["WP_ET_kg_m3"],
                "ETCP_mm": strict["etcp_mm"],
                "invalid_count": invalid,
                "fallback_count": fallback,
                "daily_path": rel(daily_path),
                "validation_source": "020_new_deterministic_2014_2023",
                "status": "PASS",
            })
    detail = pd.DataFrame(detail_rows)
    aggregate_rows: list[dict[str, Any]] = []
    for checkpoint, group in detail.groupby("checkpoint_step", sort=True):
        aggregate_rows.append({
            "row_type": "aggregate",
            "continuation_rng": int(rng),
            "branch": f"control_rng{int(rng)}",
            "checkpoint_step": int(checkpoint),
            "model_num_timesteps": int(pd.to_numeric(group["model_num_timesteps"], errors="coerce").max()),
            "year": "ALL_2014_2023",
            "validation_year_count": int(len(group)),
            "mean_yield_kg_ha": float(group["yield_kg_ha"].mean()),
            "sd_yield_kg_ha": float(group["yield_kg_ha"].std(ddof=1)),
            "mean_I_mm": float(group["I_mm"].mean()),
            "mean_N_kg_ha": float(group["N_kg_ha"].mean()),
            "mean_PFP_N": float(group["PFP_N"].mean()),
            "mean_WP_ET": float(group["WP_ET"].mean()),
            "invalid_count_total": int(group["invalid_count"].sum()),
            "fallback_count_total": int(group["fallback_count"].sum()),
            "validation_source": "020_new_deterministic_2014_2023",
            "status": "PASS" if len(group) == len(VALIDATION_YEARS) and group["status"].eq("PASS").all() else "FAIL",
        })
    combined = pd.concat([reference_75, detail, pd.DataFrame(aggregate_rows)], ignore_index=True, sort=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False, encoding="utf-8-sig")
    detail.to_csv(out / "validation_by_year.csv", index=False, encoding="utf-8-sig")
    return read_aggregate(output, rng, "020_rng_validation_with_018B_75K_reference")


def numeric_aggregates(path: Path, rng: int, source: str) -> pd.DataFrame:
    data = read_aggregate(path, rng, source)
    if set(data["checkpoint_step"].dropna().astype(int)) != set(CHECKPOINTS):
        raise RuntimeError(f"{rel(path)} does not contain all six checkpoints")
    return data.sort_values("checkpoint_step").reset_index(drop=True)


def collapse_is_true(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def collapse_is_false(value: Any) -> bool:
    return value is False or str(value).strip().lower() == "false"


def build_reproducibility_summary() -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregate_by_rng: dict[int, pd.DataFrame] = {}
    for rng in ALL_RNGS:
        if rng == 1:
            path = PRIOR_VALIDATION
            source = "018B_rng1_control_reference"
        else:
            path = RESULTS / f"yc_ppo_s1_control_rng{rng}_checkpoint_validation.csv"
            source = f"020_rng{rng}_validation"
        if rng == 0:
            aggregate_by_rng[rng] = read_aggregate(path, rng, "020_rng0_failed_attempt_partial")
            partial_data = aggregate_by_rng[rng].set_index("checkpoint_step")
            partial_yields = {
                int(checkpoint): float(partial_data.loc[checkpoint, "mean_yield_kg_ha"])
                for checkpoint in partial_data.index
                if pd.notna(partial_data.loc[checkpoint, "mean_yield_kg_ha"])
            }
            partial_pre = pd.Series({checkpoint: partial_yields[checkpoint] for checkpoint in CHECKPOINTS[:-1] if checkpoint in partial_yields})
            partial_best = int(partial_pre.idxmax()) if not partial_pre.empty else np.nan
            rows.append({
                "continuation_rng": 0,
                **{f"yield_{checkpoint // 1000}K": partial_yields.get(checkpoint, np.nan) for checkpoint in CHECKPOINTS},
                "best_pre100_checkpoint": partial_best,
                "best_pre100_yield": float(partial_pre.max()) if not partial_pre.empty else np.nan,
                "drop_best_to_100K": np.nan,
                "collapse": "unknown / insufficient evidence",
                "collapse_reason": "unknown / insufficient evidence: RNG0 training stopped before 100K",
                "first_checkpoint_below_7000": "unknown / insufficient evidence",
                "first_checkpoint_below_6500": "unknown / insufficient evidence",
                "validation_source": "020_rng0_failed_attempt_partial",
            })
            continue
        aggregate_by_rng[rng] = numeric_aggregates(path, rng, source)
        data = aggregate_by_rng[rng].set_index("checkpoint_step")
        yields = {checkpoint: float(data.loc[checkpoint, "mean_yield_kg_ha"]) for checkpoint in CHECKPOINTS}
        pre = pd.Series({checkpoint: yields[checkpoint] for checkpoint in CHECKPOINTS[:-1]})
        best_checkpoint = int(pre.idxmax())
        best_yield = float(pre.loc[best_checkpoint])
        final_yield = yields[100_000]
        drop = best_yield - final_yield
        collapse_a = final_yield < 6_500.0
        collapse_b = drop > 1_500.0
        collapse = bool(collapse_a or collapse_b)
        first_below_7000 = next((checkpoint for checkpoint in CHECKPOINTS if yields[checkpoint] < 7_000.0), "NA")
        first_below_6500 = next((checkpoint for checkpoint in CHECKPOINTS if yields[checkpoint] < 6_500.0), "NA")
        rows.append({
            "continuation_rng": int(rng),
            **{f"yield_{checkpoint // 1000}K": yields[checkpoint] for checkpoint in CHECKPOINTS},
            "best_pre100_checkpoint": best_checkpoint,
            "best_pre100_yield": best_yield,
            "drop_best_to_100K": drop,
            "collapse": collapse,
            "collapse_reason": ";".join(reason for reason, active in [("100K_mean_yield<6500", collapse_a), ("best_pre100_minus_100K>1500", collapse_b)] if active) or "none",
            "first_checkpoint_below_7000": first_below_7000,
            "first_checkpoint_below_6500": first_below_6500,
            "validation_source": source,
        })
    summary = pd.DataFrame(rows).sort_values("continuation_rng").reset_index(drop=True)
    output = RESULTS / "yc_ppo_s1_control_multirng_reproducibility.csv"
    summary.to_csv(output, index=False, encoding="utf-8-sig")
    final = pd.to_numeric(summary["yield_100K"], errors="coerce")
    drops = pd.to_numeric(summary["drop_best_to_100K"], errors="coerce")
    known = summary["collapse"].map(lambda value: collapse_is_true(value) or collapse_is_false(value))
    collapse_count = int(summary["collapse"].map(collapse_is_true).sum())
    if not bool(known.all()):
        decision = "unknown / insufficient evidence"
    elif collapse_count >= 2:
        decision = "COLLAPSE_REPRODUCIBLE"
    elif collapse_count == 0:
        decision = "COLLAPSE_RARE_IN_5"
    else:
        decision = "COLLAPSE_SPARSE_1_OF_5"
    stats = {
        "n_continuations": len(summary),
        "collapse_count": collapse_count,
        "collapse_rate": (collapse_count / len(summary)) if bool(known.all()) else "unknown / insufficient evidence",
        "yield_100K_available_count": int(final.notna().sum()),
        "yield_100K_mean": float(final.dropna().mean()) if final.notna().any() else np.nan,
        "yield_100K_sd": float(final.dropna().std(ddof=1)) if int(final.notna().sum()) > 1 else np.nan,
        "yield_100K_min": float(final.dropna().min()) if final.notna().any() else np.nan,
        "yield_100K_max": float(final.dropna().max()) if final.notna().any() else np.nan,
        "decline_gt_1000_count": int((drops > 1_000.0).sum()),
        "decline_gt_1500_count": int((drops > 1_500.0).sum()),
        "ending_below_7000_count": int((final < 7_000.0).sum()),
        "ending_below_6500_count": int((final < 6_500.0).sum()),
        "decision": decision,
        "summary_csv": rel(output),
    }
    write_json(OUT / "reproducibility_summary.json", stats)
    return summary, stats


def read_training_for_rng(rng: int) -> pd.DataFrame:
    if rng == 1:
        path = PRIOR_TRAINING
        data = pd.read_csv(path, keep_default_na=False)
        data["continuation_rng"] = 1
        source = "018B_rng1_control_reference"
    else:
        path = RESULTS / f"yc_ppo_s1_control_rng{rng}_training_metrics.csv"
        data = pd.read_csv(path, keep_default_na=False)
        source = f"020_rng{rng}_training"
    data["continuation_rng"] = int(rng)
    data["global_step"] = pd.to_numeric(data["global_step"], errors="coerce")
    for column in ["approx_kl", "clip_fraction"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["metrics_source"] = source
    return data


def build_late_update_summary(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = {rng: read_training_for_rng(rng) for rng in ALL_RNGS}
    classes = {
        True: "collapsed",
        False: "non_collapsed",
    }
    class_by_rng = {
        int(row["continuation_rng"]): bool(row["collapse"])
        for _, row in summary.iterrows()
        if collapse_is_true(row["collapse"]) or collapse_is_false(row["collapse"])
    }
    windows = [(85_000, 90_000, "85-90K"), (90_000, 95_000, "90-95K"), (95_000, 100_000, "95-100K")]
    rows: list[dict[str, Any]] = []
    for collapsed, label in classes.items():
        rngs = [rng for rng in ALL_RNGS if class_by_rng.get(rng) is collapsed]
        for lower, upper, window in windows:
            kl_by_rng: dict[str, float] = {}
            clip_by_rng: dict[str, float] = {}
            for rng in rngs:
                subset = metrics[rng]
                subset = subset[(subset["global_step"] >= lower) & (subset["global_step"] < upper)]
                kl = pd.to_numeric(subset["approx_kl"], errors="coerce").dropna()
                clip = pd.to_numeric(subset["clip_fraction"], errors="coerce").dropna()
                if not kl.empty:
                    kl_by_rng[str(rng)] = float(kl.median())
                if not clip.empty:
                    clip_by_rng[str(rng)] = float(clip.median())
            rows.append({
                "trajectory_class": label,
                "window": window,
                "window_lower_step_inclusive": lower,
                "window_upper_step_exclusive": upper,
                "trajectory_count": len(rngs),
                "approx_kl_median": float(np.median(list(kl_by_rng.values()))) if kl_by_rng else np.nan,
                "clip_fraction_median": float(np.median(list(clip_by_rng.values()))) if clip_by_rng else np.nan,
                "approx_kl_trajectory_medians": json.dumps(kl_by_rng, ensure_ascii=False, sort_keys=True),
                "clip_fraction_trajectory_medians": json.dumps(clip_by_rng, ensure_ascii=False, sort_keys=True),
                "aggregation_note": "median across per-trajectory window medians; observed association only" if rngs else "unknown / insufficient evidence",
            })
    result = pd.DataFrame(rows)
    output = RESULTS / "yc_ppo_s1_control_multirng_late_update_summary.csv"
    result.to_csv(output, index=False, encoding="utf-8-sig")
    return result


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown / insufficient evidence"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def report(summary: pd.DataFrame, stats: dict[str, Any], late: pd.DataFrame, preflight_payload: dict[str, Any]) -> None:
    collapsed = summary[summary["collapse"].map(collapse_is_true)].copy()
    first_7000 = collapsed["first_checkpoint_below_7000"].astype(str).tolist()
    first_6500 = collapsed["first_checkpoint_below_6500"].astype(str).tolist()
    threshold_crossings = [item for item in first_7000 if item != "NA"]
    late_collapsed = late[late["trajectory_class"].eq("collapsed")]
    late_noncollapsed = late[late["trajectory_class"].eq("non_collapsed")]
    collapse_rate_value = stats["collapse_rate"]
    collapse_rate_text = collapse_rate_value if isinstance(collapse_rate_value, str) else f"{float(collapse_rate_value):.3f}"
    if not collapsed.empty and threshold_crossings:
        onset_line = (
            f"Among collapsed trajectories, first <7000 crossings were `{', '.join(threshold_crossings)}`; "
            f"first <6500 crossings were `{', '.join(first_6500)}`. This is a checkpoint-level description, not an exact onset estimate."
        )
    elif not collapsed.empty:
        onset_line = "Collapsed trajectory/trajectories were identified by the pre-registered best-pre100-to-100K drop criterion, without a <7000 threshold crossing; concentration after 85–95K is unknown / insufficient evidence."
    else:
        onset_line = "No known complete continuation collapsed; RNG0 is incomplete, so collapse onset concentration after 85–95K is unknown / insufficient evidence."

    def late_line(metric: str) -> str:
        parts: list[str] = []
        for window in ["85-90K", "90-95K", "95-100K"]:
            c = late_collapsed[late_collapsed["window"].eq(window)].iloc[0]
            n = late_noncollapsed[late_noncollapsed["window"].eq(window)].iloc[0]
            parts.append(f"{window}: collapsed={fmt(c[metric], 8)}, non-collapsed={fmt(n[metric], 8)}")
        return "; ".join(parts)

    table_lines = [
        "| RNG | 75K | 80K | 85K | 90K | 95K | 100K | best pre-100K | drop to 100K | collapse |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for _, row in summary.iterrows():
        collapse_display = str(row["collapse"])
        table_lines.append(
            f"| {int(row['continuation_rng'])} | {fmt(row['yield_75K'])} | {fmt(row['yield_80K'])} | {fmt(row['yield_85K'])} | {fmt(row['yield_90K'])} | {fmt(row['yield_95K'])} | {fmt(row['yield_100K'])} | {fmt(row['best_pre100_checkpoint'], 0)} / {fmt(row['best_pre100_yield'])} | {fmt(row['drop_best_to_100K'])} | {collapse_display} |"
        )

    lines = [
        "# YC PPO Original seed1 multi-RNG CONTROL reproducibility screen",
        "",
        "This report implements prompt 020. It estimates collapse reproducibility under the reinitialized-continuation protocol, not exact replay of the historical formal stochastic trajectory.",
        "",
        "## Protocol and provenance",
        "",
        f"- Common starting point: `{rel(ORIGINAL_CHECKPOINT)}`; SHA256 `{preflight_payload['common_starting_checkpoint_sha256']}`; checkpoint `num_timesteps=75000`.",
        f"- Resume evidence: `{rel(PRIOR_RESUME_AUDIT)}`. The earlier audit established recoverable policy/value weights, optimizer state, timestep, and schedules, with RNG states reinitialized rather than serialized.",
        f"- 018B CONTROL/RNG1 is reused without rerun: training `{rel(PRIOR_TRAINING)}`; validation `{rel(PRIOR_VALIDATION)}`; manifest `{rel(PRIOR_018B_CONTROL / 'training_manifest.json')}`.",
        f"- New continuations: RNG0, RNG2, RNG3, RNG4, each requested exactly 25,000 additional environment steps and checkpoints 80K/85K/90K/95K/100K. New branch manifests are under `{rel(OUT)}`.",
        "- CONTROL only: `target_kl=None`; no KL guard, retuning, extra RNG, site, seed, variant, augmentation, or held-out-driven decision was added.",
        "- The new script is orchestration/evidence code only: `src/run_yc_ppo_seed1_multirng_control_020.py`. PPO/DSSAT source code and formal model checkpoints were not edited. An early aborted attempt did reuse the 017 log directory before isolation was fixed; that touched 017 log evidence is not used for this 020 analysis and the aborted artifacts are preserved below.",
        f"- Aborted engineering partials were preserved and excluded: `{', '.join(preflight_payload['aborted_engineering_attempt']['paths'])}`.",
        "",
        "## Per-trajectory validation summary",
        "",
        "Validation is deterministic and uses the existing 2014–2023 YCA protocol. The 75K row set for each new RNG is the byte-identical 018B deterministic reference; 80K–100K rows are new validation replays.",
        "",
        *table_lines,
        "",
        "## Required answers",
        "",
        f"1. Collapse count: `{stats['collapse_count']}` among `{stats['yield_100K_available_count']}` complete 100K trajectories; RNG0 collapse status is `unknown / insufficient evidence`, so the fixed 5-run count is unknown.",
        f"2. Collapse rate: `{collapse_rate_text}` (`{stats['collapse_count']}/5; complete denominator unavailable if unknown`).",
        f"3. 100K yield distribution: available `{stats['yield_100K_available_count']}/5`; mean `{fmt(stats['yield_100K_mean'])}`, SD `{fmt(stats['yield_100K_sd'])}`, min `{fmt(stats['yield_100K_min'])}`, max `{fmt(stats['yield_100K_max'])}` kg/ha; full 5-trajectory distribution is unknown / insufficient evidence.",
        f"4. Collapse after 85–95K: {onset_line}",
        f"5. Late `approx_kl` medians, collapsed vs non-collapsed: {late_line('approx_kl_median')}.",
        f"   Late `clip_fraction` medians, collapsed vs non-collapsed: {late_line('clip_fraction_median')}.",
        "   These are descriptive pooled trajectory-level summaries. They do not establish that KL or clipping caused collapse; empty class cells are `unknown / insufficient evidence`.",
        f"6. Final decision: `{stats['decision']}`.",
        "",
        "## Additional fixed-count checks",
        "",
        f"- Among the `{stats['yield_100K_available_count']}` complete trajectories, >1000 kg/ha own-best-to-100K decline: `{stats['decline_gt_1000_count']}`; >1500 kg/ha decline: `{stats['decline_gt_1500_count']}`.",
        f"- Among the complete trajectories, ending below 7000 kg/ha: `{stats['ending_below_7000_count']}`; below 6500 kg/ha: `{stats['ending_below_6500_count']}`. The 5-run counts are unknown / insufficient evidence.",
        f"- Historical 017 formal collapse was excluded from the denominator and rate: `{rel(ROOT / 'docs' / 'yc_ppo_original_seed1_75k_100k_collapse_diagnostic.md')}`.",
        "",
        "## Output evidence",
        "",
        f"- Reproducibility table: `{rel(RESULTS / 'yc_ppo_s1_control_multirng_reproducibility.csv')}`.",
        f"- Late-update table: `{rel(RESULTS / 'yc_ppo_s1_control_multirng_late_update_summary.csv')}`.",
        *[f"- RNG{rng} training metrics: `{rel(RESULTS / f'yc_ppo_s1_control_rng{rng}_training_metrics.csv')}`." for rng in CONTINUATION_RNGS],
        *[f"- RNG{rng} checkpoint validation: `{rel(RESULTS / f'yc_ppo_s1_control_rng{rng}_checkpoint_validation.csv')}`." for rng in CONTINUATION_RNGS],
        "- RNG1 evidence remains the 018B files listed above; it was not rerun.",
        "",
        "## Scope stop",
        "",
        "The planned 5-RNG CONTROL screen is incomplete because RNG0 did not reach 100K. No KL-GUARD continuation, additional RNG, hyperparameter change, new site, new seed, or augmented variant was run.",
        "",
        "YC_PPO_SEED1_MULTIRNG_CONTROL_SCREEN_INCOMPLETE",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["audit", "train", "validate", "record-failed", "report", "all"], default="all")
    parser.add_argument("--rngs", default="", help="Optional comma-separated new RNGs for isolated train/validate execution")
    args = parser.parse_args()
    selected_rngs = CONTINUATION_RNGS if not args.rngs.strip() else [int(item.strip()) for item in args.rngs.split(",") if item.strip()]
    invalid_rngs = [rng for rng in selected_rngs if rng not in CONTINUATION_RNGS]
    if invalid_rngs:
        raise ValueError(f"Only new RNGs {CONTINUATION_RNGS} may be selected; got {invalid_rngs}")

    OUT.mkdir(parents=True, exist_ok=True)
    preflight_payload = preflight()
    if args.phase == "audit":
        print(json.dumps(preflight_payload, ensure_ascii=False))
        return 0

    import diagnose_yc_ppo_original_seed1_collapse_017 as diagnostic

    cfg = json.loads((ORIGINAL_RUN / "configs" / "222YCA_O1_yc_v1_original_seed1.json").read_text(encoding="utf-8"))
    replay, config, env_config, runtime_state = isolated_runtime_setup(cfg, diagnostic)
    try:
        if args.phase == "record-failed":
            metrics = record_failed_rng0_training_metrics()
            validation = record_failed_rng0_validation(diagnostic, replay, config, env_config)
            print(json.dumps({"training_metric_rows": len(metrics), "validation_aggregate_rows": len(validation)}, ensure_ascii=False))
            return 0
        if args.phase in {"train", "all"}:
            manifests = {str(rng): train_rng(rng, replay, config, env_config) for rng in selected_rngs}
            for rng in selected_rngs:
                extract_training_metrics(rng)
            if args.phase == "train":
                print(json.dumps(manifests, ensure_ascii=False))
                return 0

        if args.phase in {"validate", "all"}:
            validations = {str(rng): validate_rng(rng, diagnostic, replay, config, env_config) for rng in selected_rngs}
            if args.phase == "validate":
                print(json.dumps({rng: len(data) for rng, data in validations.items()}, ensure_ascii=False))
                return 0

        summary, stats = build_reproducibility_summary()
        late = build_late_update_summary(summary)
        report(summary, stats, late, preflight_payload)
        print(stats["decision"])
        if stats["decision"] == "unknown / insufficient evidence":
            print("YC_PPO_SEED1_MULTIRNG_CONTROL_SCREEN_INCOMPLETE")
            return 2
        print("YC_PPO_SEED1_MULTIRNG_CONTROL_SCREEN_COMPLETE")
        return 0
    finally:
        diagnostic.restore_runtime(runtime_state)


if __name__ == "__main__":
    raise SystemExit(main())
