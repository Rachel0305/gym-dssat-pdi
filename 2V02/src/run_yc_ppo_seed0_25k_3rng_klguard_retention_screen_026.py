"""026: YC Original seed0@25K three-continuation-RNG retention screen.

The 25K starting point is the new CONTROL checkpoint produced by 025.  This
orchestrator imports the existing YC runtime and the already frozen KL guard
instrumentation; it does not edit PPO/DSSAT source logic or create a new
scientific intervention.
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
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "src" / "055_yca_lowIC_site_transfer"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

PROMPT = ROOT / "prompt_01" / "026_codex_prompt_026_yc_ppo_seed0_25k_3rng_klguard_retention_screen.md"
OUT = ROOT / "benchmark_results" / "026_yc_ppo_seed0_25k_3rng_klguard_retention_screen"
RESULTS = ROOT / "results"
REPORT = ROOT / "docs" / "yc_ppo_original_seed0_25k_3rng_klguard_retention_screen.md"

STARTING_CHECKPOINT = ROOT / "benchmark_results" / "025_yc_ppo_original_seed0_fixed_klguard_crossseed" / "control" / "models" / "YCA" / "yc_ppo_original_seed0_control_ckpt25000.zip"
STARTING_VALIDATION = RESULTS / "yc_ppo_original_seed0_control_checkpoint_validation.csv"
STARTING_MANIFEST = ROOT / "benchmark_results" / "025_yc_ppo_original_seed0_fixed_klguard_crossseed" / "control" / "training_manifest.json"
STARTING_CONFIG = ROOT / "benchmark_results" / "025_yc_ppo_original_seed0_fixed_klguard_crossseed" / "control" / "configs" / "resolved_runtime_config.json"

STATION = "YCA"
SITE = "YC"
TRAINING_SEED = 0
TRAIN_YEARS = list(range(2004, 2014))
VALIDATION_YEARS = list(range(2014, 2024))
START_CHECKPOINT = 25_000
CONTINUATION_CHECKPOINTS = [35_000, 50_000, 75_000, 100_000]
ALL_CHECKPOINTS = [25_000] + CONTINUATION_CHECKPOINTS
CONTINUATION_STEPS = 75_000
CRNGS = [1, 2, 3]
EFFECTIVE_KL_THRESHOLD = 1.0e-4
CONFIGURED_TARGET_KL = 6.666666667e-05

PAIRED_OUT = RESULTS / "yc_ppo_seed0_25k_3rng_control_vs_klguard_paired.csv"
WINDOW_OUT = RESULTS / "yc_ppo_seed0_25k_3rng_control_vs_klguard_update_windows.csv"
SUMMARY_OUT = RESULTS / "yc_ppo_seed0_25k_3rng_retention_summary.csv"
DECISION_OUT = RESULTS / "yc_ppo_seed0_25k_3rng_retention_decision.csv"


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


def checkpoint_data(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("data"))


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def branch_key(crng: int, branch: str) -> str:
    return f"crng{int(crng)}_{branch}"


def metric_path(crng: int, branch: str) -> Path:
    return RESULTS / f"yc_ppo_seed0_25k_crng{int(crng)}_{branch}_training_metrics.csv"


def validation_path(crng: int, branch: str) -> Path:
    return RESULTS / f"yc_ppo_seed0_25k_crng{int(crng)}_{branch}_validation.csv"


def branch_root(crng: int, branch: str) -> Path:
    return OUT / f"crng{int(crng)}" / branch


def branch_checkpoint_path(crng: int, branch: str, checkpoint: int) -> Path:
    if int(checkpoint) == START_CHECKPOINT:
        return STARTING_CHECKPOINT
    return branch_root(crng, branch) / "models" / "YCA" / f"yc_ppo_seed0_25k_crng{int(crng)}_{branch}_ckpt{int(checkpoint)}.zip"


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


def starting_checkpoint_audit() -> dict[str, Any]:
    if not STARTING_CHECKPOINT.exists():
        return {"status": "FAIL", "reason": f"missing: {rel(STARTING_CHECKPOINT)}"}
    with zipfile.ZipFile(STARTING_CHECKPOINT) as archive:
        members = sorted(archive.namelist())
        data = json.loads(archive.read("data"))
        optimizer_bytes = archive.read("policy.optimizer.pth") if "policy.optimizer.pth" in members else b""
    import io
    import torch

    optimizer_state = torch.load(io.BytesIO(optimizer_bytes), map_location="cpu", weights_only=False) if optimizer_bytes else {}
    optimizer_state_entries = len(optimizer_state.get("state", {}))
    schedule_keys = ["_total_timesteps", "_current_progress_remaining", "learning_rate", "clip_range"]
    schedule_available = all(key in data for key in schedule_keys)
    validation_mean = math.nan
    validation_rows = 0
    validation_status = "missing"
    if STARTING_VALIDATION.exists():
        validation = pd.read_csv(STARTING_VALIDATION, keep_default_na=False)
        validation = validation[(validation["row_type"].eq("aggregate")) & (pd.to_numeric(validation["checkpoint_step"], errors="coerce").eq(START_CHECKPOINT))]
        if len(validation) == 1:
            validation_mean = float(validation.iloc[0]["mean_yield_kg_ha"])
            validation_rows = int(round(float(validation.iloc[0].get("validation_year_count", 0))))
            validation_status = str(validation.iloc[0].get("status", "unknown"))
    model_load_ok = False
    model_num_timesteps = None
    model_optimizer_state_entries = None
    try:
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(str(STARTING_CHECKPOINT), device="cpu")
        model_num_timesteps = int(model.num_timesteps)
        model_optimizer_state_entries = len(model.policy.optimizer.state_dict().get("state", {}))
        model_load_ok = True
        del model
    except Exception:
        model_load_error = traceback.format_exc()
    else:
        model_load_error = ""
    passed = bool(
        data.get("num_timesteps") == START_CHECKPOINT
        and optimizer_state_entries > 0
        and schedule_available
        and model_load_ok
        and model_num_timesteps == START_CHECKPOINT
        and model_optimizer_state_entries is not None
        and model_optimizer_state_entries > 0
        and validation_rows == len(VALIDATION_YEARS)
        and validation_status == "PASS"
        and math.isclose(validation_mean, 8201.6, rel_tol=0.0, abs_tol=1.0)
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "path": rel(STARTING_CHECKPOINT),
        "sha256": sha256_file(STARTING_CHECKPOINT),
        "zip_members": members,
        "num_timesteps": data.get("num_timesteps"),
        "model_num_timesteps": model_num_timesteps,
        "optimizer_state_entries_archive": optimizer_state_entries,
        "optimizer_state_entries_loaded_model": model_optimizer_state_entries,
        "schedule_metadata": {key: data.get(key) for key in schedule_keys},
        "validation_source": rel(STARTING_VALIDATION),
        "validation_mean_yield_kg_ha": validation_mean,
        "validation_year_count": validation_rows,
        "validation_status": validation_status,
        "model_load_ok": model_load_ok,
        "model_load_error": model_load_error,
    }


def preflight() -> dict[str, Any]:
    issues: list[str] = []
    required = [PROMPT, STARTING_CHECKPOINT, STARTING_VALIDATION, STARTING_MANIFEST, STARTING_CONFIG]
    for path in required:
        if not path.exists():
            issues.append(f"missing: {rel(path)}")
    start_audit = starting_checkpoint_audit() if STARTING_CHECKPOINT.exists() else {"status": "FAIL"}
    if start_audit.get("status") != "PASS":
        issues.append("025 CONTROL seed0@25K checkpoint/optimizer/validation audit failed")
    guard = verify_guard_semantics()
    if guard["pass"] is not True:
        issues.append("KL guard source semantics failed")
    historical_contract = None
    if STARTING_CONFIG.exists():
        historical_contract = json.loads(STARTING_CONFIG.read_text(encoding="utf-8"))
        if historical_contract.get("seed") != TRAINING_SEED:
            issues.append("resolved 025 seed0 runtime config does not retain training seed 0")
    source_hashes = {rel(path): sha256_file(path) for path in required if path.exists() and path.is_file()}
    script_path = ROOT / "src" / "run_yc_ppo_original_seed0_fixed_klguard_crossseed_025.py"
    if script_path.exists():
        source_hashes[rel(script_path)] = sha256_file(script_path)
    payload = {
        "status": "PASS" if not issues else "FAIL",
        "prompt": rel(PROMPT),
        "starting_checkpoint_audit": start_audit,
        "starting_manifest": rel(STARTING_MANIFEST),
        "starting_config": rel(STARTING_CONFIG),
        "continuation_rngs": CRNGS,
        "training_seed_remains": TRAINING_SEED,
        "continuation_steps": CONTINUATION_STEPS,
        "checkpoints": ALL_CHECKPOINTS,
        "configured_target_kl": CONFIGURED_TARGET_KL,
        "effective_kl_threshold": EFFECTIVE_KL_THRESHOLD,
        "guard_semantics": guard,
        "contract_assertion": {
            "observation": "raw 046_02 observation; normalization=false; weather_forecast=false",
            "reward_and_safety": "inherit_042_15_via_046_02_unchanged",
            "action_grid": "I [0,15,30,45] mm x N [0,40,80,120] kg/ha",
            "train_years": TRAIN_YEARS,
            "validation_years": VALIDATION_YEARS,
            "ic_cultivar": "same YC lowIC / MZCER048.CUL runtime as 025",
            "scientific_intervention": "fixed KL-based update early stop only",
        },
        "source_sha256": source_hashes,
        "issues": issues,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "preflight.json", payload)
    return payload


class CheckpointCallbackFactory:
    def __init__(self, crng: int, branch: str):
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class Callback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0)

            def _on_step(self) -> bool:
                now = int(self.model.num_timesteps)
                for target in CONTINUATION_CHECKPOINTS:
                    if target <= now and target not in outer.saved:
                        path = branch_checkpoint_path(outer.crng, outer.branch, target)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        self.model.save(str(path.with_suffix("")))
                        outer.saved.append(target)
                        outer.actual_steps[target] = int(self.model.num_timesteps)
                return True

        self.crng = int(crng)
        self.branch = branch
        self.saved: list[int] = []
        self.actual_steps: dict[int, int] = {}
        self.callback = Callback()


def training_complete(crng: int, branch: str) -> bool:
    manifest = branch_root(crng, branch) / "training_manifest.json"
    return manifest.exists() and all(branch_checkpoint_path(crng, branch, step).exists() for step in CONTINUATION_CHECKPOINTS)


def train_continuation(crng: int, branch: str) -> dict[str, Any]:
    root = branch_root(crng, branch)
    manifest_path = root / "training_manifest.json"
    if training_complete(crng, branch):
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    existing = [rel(branch_checkpoint_path(crng, branch, step)) for step in CONTINUATION_CHECKPOINTS if branch_checkpoint_path(crng, branch, step).exists()]
    if existing:
        raise RuntimeError(f"{branch_key(crng, branch)} partial continuation exists; refusing overwrite: {existing}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    _, base03222, _, config, env_config = __import__("run_yc_ppo_original_seed0_fixed_klguard_crossseed_025", fromlist=["runtime_setup"]).runtime_setup(OUT)
    branch_env_config = copy.deepcopy(env_config)
    branch_env_config["paths"]["output_root"] = rel(root / "runtime")
    write_json(root / "configs" / "resolved_runtime_config.json", config)
    write_json(root / "configs" / "resolved_env_config.json", branch_env_config)

    from stable_baselines3.common.logger import configure

    kl019 = __import__("run_yc_ppo_seed1_kl_guard_paired_019", fromlist=["make_instrumented_ppo_class"])
    InstrumentedPPO = kl019.make_instrumented_ppo_class()
    target_kl = None if branch == "control" else CONFIGURED_TARGET_KL
    callback_factory = CheckpointCallbackFactory(crng, branch)
    env = None
    model = None
    started = pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
    try:
        set_all_seeds(int(crng))
        env = base03222.RandomYearEnv(config, branch_env_config, STATION, TRAIN_YEARS, int(crng))
        model = InstrumentedPPO.load(str(STARTING_CHECKPOINT), env=env, device="cpu")
        if int(model.num_timesteps) != START_CHECKPOINT:
            raise RuntimeError(f"loaded num_timesteps={model.num_timesteps}; expected {START_CHECKPOINT}")
        optimizer_entries = len(model.policy.optimizer.state_dict().get("state", {}))
        if optimizer_entries <= 0:
            raise RuntimeError("loaded optimizer state is empty")
        model.target_kl = target_kl
        model.set_logger(configure(str(root / "tensorboard"), ["tensorboard"]))
        model.learn(total_timesteps=CONTINUATION_STEPS, reset_num_timesteps=False, progress_bar=False, callback=callback_factory.callback)
        final_steps = int(model.num_timesteps)
        model.logger.close()
    except Exception:
        write_json(root / "training_failure_manifest.json", {
            "branch": branch,
            "continuation_rng": int(crng),
            "starting_checkpoint": rel(STARTING_CHECKPOINT),
            "target_kl": target_kl,
            "started_at": started,
            "finished_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            "saved_checkpoints": callback_factory.saved,
            "actual_steps": callback_factory.actual_steps,
            "error": traceback.format_exc(),
        })
        raise
    finally:
        if env is not None:
            env.close()
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    missing = [step for step in CONTINUATION_CHECKPOINTS if not branch_checkpoint_path(crng, branch, step).exists()]
    if missing:
        write_json(root / "training_failure_manifest.json", {"branch": branch, "continuation_rng": int(crng), "missing_checkpoints": missing, "saved_checkpoints": callback_factory.saved})
        raise RuntimeError(f"{branch_key(crng, branch)} missing continuation checkpoints {missing}")
    manifest = {
        "branch": branch,
        "continuation_rng": int(crng),
        "training_seed": TRAINING_SEED,
        "starting_checkpoint": rel(STARTING_CHECKPOINT),
        "starting_checkpoint_sha256": sha256_file(STARTING_CHECKPOINT),
        "starting_num_timesteps": START_CHECKPOINT,
        "continuation_steps_requested": CONTINUATION_STEPS,
        "final_num_timesteps": final_steps,
        "target_kl": target_kl,
        "effective_kl_threshold": None if target_kl is None else target_kl * 1.5,
        "checkpoint_paths": {str(step): rel(branch_checkpoint_path(crng, branch, step)) for step in ALL_CHECKPOINTS},
        "checkpoint_actual_num_timesteps": {str(step): checkpoint_data(branch_checkpoint_path(crng, branch, step)).get("num_timesteps") for step in CONTINUATION_CHECKPOINTS},
        "checkpoint_callback_observed_steps": {str(step): value for step, value in callback_factory.actual_steps.items()},
        "optimizer_state_restored": True,
        "optimizer_state_entries_at_load": optimizer_entries,
        "schedule_continuation": "MaskablePPO.load followed by learn(total_timesteps=75000, reset_num_timesteps=False); loaded 25K schedule metadata retained to 100K",
        "continuation_rng_initialization": "Python/NumPy/PyTorch/RandomYearEnv initialized from CRNG; checkpoint training seed remains 0",
        "runtime_wrapper": "LateIrrigationReserveMaskWrapper via 040_28/040_36 chain",
        "scientific_contract_unchanged": True,
        "instrumentation_only": True,
        "started_at": started,
        "finished_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
    }
    write_json(manifest_path, manifest)
    return manifest


def extract_metrics(crng: int, branch: str) -> pd.DataFrame:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    key = branch_key(crng, branch)
    event_files = sorted((branch_root(crng, branch) / "tensorboard").glob("**/events*"))
    if len(event_files) != 1:
        raise RuntimeError(f"{key}: expected one TensorBoard event file, got {len(event_files)}")
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
    for col in ["global_step", "timestep", "approx_kl", "clip_fraction", "entropy_loss", "policy_gradient_loss", "value_loss", "explained_variance", "learning_rate", "clip_range", "epochs_attempted", "epochs_completed", "optimizer_steps_completed", "kl_guard_triggered", "kl_at_trigger"]:
        if col not in metrics.columns:
            metrics[col] = np.nan
    metrics.insert(0, "continuation_rng", f"CRNG{int(crng)}")
    metrics.insert(1, "branch", branch.upper() if branch == "control" else "KL-GUARD")
    metrics = metrics[["branch", "continuation_rng", "global_step", "timestep", "approx_kl", "clip_fraction", "entropy_loss", "policy_gradient_loss", "value_loss", "explained_variance", "learning_rate", "clip_range", "epochs_attempted", "epochs_completed", "optimizer_steps_completed", "kl_guard_triggered", "kl_at_trigger"]]
    output = metric_path(crng, branch)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False, encoding="utf-8-sig")
    write_json(branch_root(crng, branch) / "tensorboard_manifest.json", {
        "event_file": rel(event_files[0]),
        "event_file_sha256": sha256_file(event_files[0]),
        "scalar_tags": accumulator.Tags().get("scalars", []),
        "rows": int(len(metrics)),
        "missing_required_columns": [col for col in ["approx_kl", "clip_fraction", "entropy_loss", "policy_gradient_loss", "value_loss", "explained_variance", "learning_rate", "clip_range", "epochs_attempted", "epochs_completed", "optimizer_steps_completed", "kl_guard_triggered"] if col not in metrics or metrics[col].isna().all()],
    })
    return metrics


def reuse_start_validation(crng: int, branch: str) -> pd.DataFrame:
    source = pd.read_csv(STARTING_VALIDATION, keep_default_na=False)
    source = source[source["checkpoint_step"].astype(str).eq(str(START_CHECKPOINT))].copy()
    if len(source[source["row_type"].eq("year")]) != len(VALIDATION_YEARS):
        raise RuntimeError("025 start validation does not contain ten year rows")
    source["continuation_rng"] = f"CRNG{int(crng)}"
    source["branch"] = branch.upper() if branch == "control" else "KL-GUARD"
    source["source_reused_from_025"] = True
    source["source_025_validation_path"] = rel(STARTING_VALIDATION)
    return source


def validate_branch(crng: int, branch: str) -> pd.DataFrame:
    import run_yc_ppo_original_seed0_fixed_klguard_crossseed_025 as p025
    from sb3_contrib import MaskablePPO

    _, base03222, direct, config, env_config = p025.runtime_setup(OUT)
    root = branch_root(crng, branch)
    validation_env_config = copy.deepcopy(env_config)
    validation_env_config["paths"]["output_root"] = rel(root / "runtime_validation")
    detail = reuse_start_validation(crng, branch)
    detail = detail[detail["row_type"].eq("year")].copy()
    for checkpoint in CONTINUATION_CHECKPOINTS:
        path = branch_checkpoint_path(crng, branch, checkpoint)
        if not path.exists():
            raise FileNotFoundError(path)
        model = MaskablePPO.load(str(path), device="cpu")
        try:
            for year in VALIDATION_YEARS:
                daily, strict = p025.validation_daily(model, base03222.base, direct, config, validation_env_config, year, checkpoint, branch_key(crng, branch))
                daily_path = root / "daily_outputs" / STATION / f"YCA_{year}_seed0_ckpt{checkpoint}_daily.csv"
                daily_path.parent.mkdir(parents=True, exist_ok=True)
                daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
                invalid = int((pd.to_numeric(daily["requested_discrete_action_index"], errors="coerce") != pd.to_numeric(daily["discrete_action_index"], errors="coerce")).sum())
                fallback = int(pd.Series(daily["mask_forced_noop"]).astype(str).str.lower().eq("true").sum())
                detail = pd.concat([detail, pd.DataFrame([{
                    "row_type": "year", "branch": branch.upper() if branch == "control" else "KL-GUARD", "continuation_rng": f"CRNG{int(crng)}",
                    "checkpoint_step": checkpoint, "checkpoint": checkpoint, "model_num_timesteps": checkpoint_data(path).get("num_timesteps"), "year": year,
                    "yield_kg_ha": float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1]),
                    "I_mm": float(pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0).sum()),
                    "N_kg_ha": float(pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0).sum()),
                    "PFP_N": strict["PFP_N_kg_kg"], "WP_ET": strict["WP_ET_kg_m3"], "ETCP_mm": strict["etcp_mm"],
                    "invalid": invalid, "fallback": fallback, "invalid_count": invalid, "fallback_count": fallback,
                    "daily_path": rel(daily_path), "source_reused_from_025": False, "source_025_validation_path": "", "status": "PASS",
                }])], ignore_index=True)
        finally:
            del model
            gc.collect()
    aggregates: list[dict[str, Any]] = []
    for checkpoint, group in detail.groupby("checkpoint_step", sort=True):
        yields = pd.to_numeric(group["yield_kg_ha"], errors="coerce")
        aggregates.append({
            "row_type": "aggregate", "branch": branch.upper() if branch == "control" else "KL-GUARD", "continuation_rng": f"CRNG{int(crng)}",
            "checkpoint_step": int(checkpoint), "checkpoint": int(checkpoint), "model_num_timesteps": int(pd.to_numeric(group["model_num_timesteps"], errors="coerce").max()),
            "year": "ALL_2014_2023", "validation_year_count": int(len(group)),
            "mean_yield_kg_ha": float(yields.mean()), "sd_yield_kg_ha": float(yields.std(ddof=1)), "min_yield_kg_ha": float(yields.min()), "max_yield_kg_ha": float(yields.max()),
            "mean_I_mm": float(pd.to_numeric(group["I_mm"], errors="coerce").mean()), "mean_N_kg_ha": float(pd.to_numeric(group["N_kg_ha"], errors="coerce").mean()),
            "mean_PFP_N": float(pd.to_numeric(group["PFP_N"], errors="coerce").mean()), "mean_WP_ET": float(pd.to_numeric(group["WP_ET"], errors="coerce").mean()),
            "invalid": int(pd.to_numeric(group["invalid"], errors="coerce").sum()), "fallback": int(pd.to_numeric(group["fallback"], errors="coerce").sum()),
            "invalid_count_total": int(pd.to_numeric(group["invalid"], errors="coerce").sum()), "fallback_count_total": int(pd.to_numeric(group["fallback"], errors="coerce").sum()),
            "source_reused_from_025": bool(group["source_reused_from_025"].astype(bool).all()), "status": "PASS" if len(group) == len(VALIDATION_YEARS) else "FAIL",
        })
    output = validation_path(crng, branch)
    combined = pd.concat([detail, pd.DataFrame(aggregates)], ignore_index=True, sort=False)
    combined.to_csv(output, index=False, encoding="utf-8-sig")
    detail.to_csv(root / "validation_by_year.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(aggregates)


def load_aggregates(crng: int, branch: str) -> pd.DataFrame:
    df = pd.read_csv(validation_path(crng, branch), keep_default_na=False)
    df = df[df["row_type"].eq("aggregate")].copy()
    for col in ["checkpoint_step", "mean_yield_kg_ha", "sd_yield_kg_ha", "min_yield_kg_ha", "max_yield_kg_ha", "mean_I_mm", "mean_N_kg_ha", "mean_PFP_N", "mean_WP_ET", "invalid", "fallback"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("checkpoint_step").reset_index(drop=True)


def make_windows(metrics: dict[tuple[int, str], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    windows = [(25_000, 35_000, "25-35K"), (35_000, 50_000, "35-50K"), (50_000, 75_000, "50-75K"), (75_000, 100_000, "75-100K")]
    for crng in CRNGS:
        for branch in ["control", "klguard"]:
            df = metrics[(crng, branch)]
            for lower, upper, label in windows:
                group = df[df["global_step"].between(lower + 1, upper)]
                row: dict[str, Any] = {"continuation_rng": f"CRNG{int(crng)}", "branch": branch.upper() if branch == "control" else "KL-GUARD", "window": label, "lower_step": lower, "upper_step": upper, "update_count": int(len(group))}
                for col in ["approx_kl", "clip_fraction", "entropy_loss", "policy_gradient_loss", "value_loss", "explained_variance"]:
                    values = pd.to_numeric(group[col], errors="coerce").dropna()
                    row[f"{col}_mean"] = float(values.mean()) if len(values) else np.nan
                    row[f"{col}_median"] = float(values.median()) if len(values) else np.nan
                for col in ["kl_guard_triggered", "epochs_completed", "optimizer_steps_completed"]:
                    values = pd.to_numeric(group[col], errors="coerce").dropna()
                    row[f"{col}_mean"] = float(values.mean()) if len(values) else np.nan
                row["optimizer_step_fraction"] = float((pd.to_numeric(group["optimizer_steps_completed"], errors="coerce") > 0).mean()) if len(group) else np.nan
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(WINDOW_OUT, index=False, encoding="utf-8-sig")
    return out


CLASS_ORDER = {"RETAINED": 0, "PARTIAL_DEGRADATION": 1, "COLLAPSE": 2}


def retention_class(start_yield: float, final_yield: float) -> str:
    drop = start_yield - final_yield
    if final_yield < 6500.0 or drop > 1500.0:
        return "COLLAPSE"
    if final_yield >= 7500.0 and drop <= 700.0:
        return "RETAINED"
    return "PARTIAL_DEGRADATION"


def read_start_yield() -> float:
    validation = pd.read_csv(STARTING_VALIDATION, keep_default_na=False)
    row = validation[(validation["row_type"].eq("aggregate")) & (pd.to_numeric(validation["checkpoint_step"], errors="coerce").eq(START_CHECKPOINT))]
    if len(row) != 1:
        raise RuntimeError("Unable to resolve 025 seed0@25K mean yield")
    return float(row.iloc[0]["mean_yield_kg_ha"])


def make_pairwise(aggregates: dict[tuple[int, str], pd.DataFrame], metrics: dict[tuple[int, str], pd.DataFrame], start_yield: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for crng in CRNGS:
        vals: dict[str, dict[int, float]] = {}
        resources: dict[str, dict[str, float]] = {}
        for branch in ["control", "klguard"]:
            agg = aggregates[(crng, branch)]
            vals[branch] = {int(row.checkpoint_step): float(row.mean_yield_kg_ha) for row in agg.itertuples()}
            final = agg[agg["checkpoint_step"].eq(100_000)].iloc[0]
            resources[branch] = {"I": float(final["mean_I_mm"]), "N": float(final["mean_N_kg_ha"]), "sd": float(final["sd_yield_kg_ha"])}
        control_class = retention_class(start_yield, vals["control"][100_000])
        guard_class = retention_class(start_yield, vals["klguard"][100_000])
        guard_metrics = metrics[(crng, "klguard")]
        optimized = pd.to_numeric(guard_metrics["optimizer_steps_completed"], errors="coerce")
        trigger = pd.to_numeric(guard_metrics["kl_guard_triggered"], errors="coerce")
        guard_frozen = bool(len(optimized) == 0 or (optimized > 0).mean() <= 0.0 or optimized.mean() <= 0.0)
        material_components = {
            "control_retained_guard_not_retained": control_class == "RETAINED" and guard_class != "RETAINED",
            "guard_below_95pct_control": vals["klguard"][100_000] < 0.95 * vals["control"][100_000],
            "guard_class_worse": CLASS_ORDER[guard_class] > CLASS_ORDER[control_class],
            "guard_frozen": guard_frozen,
        }
        rows.append({
            "continuation_rng": f"CRNG{int(crng)}",
            "control_yield_35K": vals["control"][35_000], "guard_yield_35K": vals["klguard"][35_000],
            "control_yield_50K": vals["control"][50_000], "guard_yield_50K": vals["klguard"][50_000],
            "control_yield_75K": vals["control"][75_000], "guard_yield_75K": vals["klguard"][75_000],
            "control_yield_100K": vals["control"][100_000], "guard_yield_100K": vals["klguard"][100_000],
            "delta_yield_100K": vals["klguard"][100_000] - vals["control"][100_000],
            "control_drop_25_to_100": start_yield - vals["control"][100_000], "guard_drop_25_to_100": start_yield - vals["klguard"][100_000],
            "control_retention_class": control_class, "guard_retention_class": guard_class,
            "control_I_100K": resources["control"]["I"], "guard_I_100K": resources["klguard"]["I"],
            "control_N_100K": resources["control"]["N"], "guard_N_100K": resources["klguard"]["N"],
            "control_sd_yield_100K": resources["control"]["sd"], "guard_sd_yield_100K": resources["klguard"]["sd"],
            "guard_trigger_rate": float(trigger.mean()) if len(trigger) else np.nan,
            "guard_mean_completed_epochs": float(pd.to_numeric(guard_metrics["epochs_completed"], errors="coerce").mean()),
            "guard_mean_optimizer_steps": float(optimized.mean()), "guard_optimizer_step_fraction": float((optimized > 0).mean()),
            "material_harm": bool(any(material_components.values())),
            "material_harm_components": json.dumps(material_components, ensure_ascii=False),
            "guard_frozen": guard_frozen,
        })
    out = pd.DataFrame(rows)
    out.to_csv(PAIRED_OUT, index=False, encoding="utf-8-sig")
    stats = {
        "mean_paired_delta_yield_100K": float(out["delta_yield_100K"].mean()),
        "median_paired_delta_yield_100K": float(out["delta_yield_100K"].median()),
        "minimum_paired_delta_yield_100K": float(out["delta_yield_100K"].min()),
        "guard_retention_class_better_count": int(sum(CLASS_ORDER[row.guard_retention_class] < CLASS_ORDER[row.control_retention_class] for row in out.itertuples())),
        "guard_retention_class_worse_count": int(sum(CLASS_ORDER[row.guard_retention_class] > CLASS_ORDER[row.control_retention_class] for row in out.itertuples())),
        "guard_material_harm_count": int(out["material_harm"].sum()),
        "guard_frozen_count": int(out["guard_frozen"].sum()),
    }
    return out, stats


def make_summary(aggregates: dict[tuple[int, str], pd.DataFrame], pairwise: pd.DataFrame, pair_stats: dict[str, Any], start_yield: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for branch in ["control", "klguard"]:
        final_yields: list[float] = []
        drops: list[float] = []
        classes: list[str] = []
        for crng in CRNGS:
            agg = aggregates[(crng, branch)]
            y100 = float(agg[agg["checkpoint_step"].eq(100_000)].iloc[0]["mean_yield_kg_ha"])
            final_yields.append(y100)
            drops.append(start_yield - y100)
            classes.append(retention_class(start_yield, y100))
        rows.append({
            "summary_type": "branch", "branch": branch.upper() if branch == "control" else "KL-GUARD", "continuation_rng": "ALL_3",
            "retained_count": classes.count("RETAINED"), "retained_denominator": 3,
            "partial_degradation_count": classes.count("PARTIAL_DEGRADATION"), "collapse_count": classes.count("COLLAPSE"),
            "yield_100K_mean": float(np.mean(final_yields)), "yield_100K_sd": float(np.std(final_yields, ddof=1)),
            "mean_drop_25_to_100": float(np.mean(drops)), "median_drop_25_to_100": float(np.median(drops)),
            "mean_paired_delta_yield_100K": np.nan, "median_paired_delta_yield_100K": np.nan, "minimum_paired_delta_yield_100K": np.nan,
            "guard_retention_class_better_count": np.nan, "guard_retention_class_worse_count": np.nan, "guard_material_harm_count": np.nan, "guard_frozen_count": np.nan,
        })
    rows.append({
        "summary_type": "paired", "branch": "PAIRED_SUMMARY", "continuation_rng": "CRNG1/2/3",
        "retained_count": np.nan, "retained_denominator": 3, "partial_degradation_count": np.nan, "collapse_count": np.nan,
        "yield_100K_mean": np.nan, "yield_100K_sd": np.nan, "mean_drop_25_to_100": np.nan, "median_drop_25_to_100": np.nan,
        **pair_stats,
    })
    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")
    return out


def make_decision(summary: pd.DataFrame, pairwise: pd.DataFrame, pair_stats: dict[str, Any], metrics: dict[tuple[int, str], pd.DataFrame]) -> dict[str, Any]:
    control = summary[(summary["summary_type"].eq("branch")) & (summary["branch"].eq("CONTROL"))].iloc[0]
    guard = summary[(summary["summary_type"].eq("branch")) & (summary["branch"].eq("KL-GUARD"))].iloc[0]
    control_nonretained = int(control["partial_degradation_count"] + control["collapse_count"])
    guard_retained = int(guard["retained_count"])
    control_retained = int(control["retained_count"])
    all_guard_updates = bool(pair_stats["guard_frozen_count"] == 0)
    material_harm_count = int(pair_stats["guard_material_harm_count"])
    promising = bool(control_nonretained >= 2 and guard_retained > control_retained and guard_retained >= 2 and material_harm_count == 0 and all_guard_updates)
    partial = bool(control_nonretained >= 2 and (pair_stats["guard_retention_class_better_count"] >= 1 or (pd.to_numeric(pairwise["delta_yield_100K"], errors="coerce") >= 1000.0).any()) and not promising and material_harm_count < 2 and guard_retained >= control_retained)
    stable_inconclusive = bool(control_retained >= 2 and material_harm_count == 0)
    if promising:
        decision = "SEED0_RETENTION_GUARD_PROMISING"
    elif stable_inconclusive:
        decision = "SEED0_CONTROL_RETENTION_STABLE_INCONCLUSIVE"
    elif partial:
        decision = "SEED0_RETENTION_GUARD_PARTIAL"
    else:
        decision = "SEED0_RETENTION_GUARD_NOT_SUPPORTED"
    guard_metric_frames = [metrics[(crng, "klguard")] for crng in CRNGS]
    guard_trigger_rates = [float(pd.to_numeric(df["kl_guard_triggered"], errors="coerce").mean()) for df in guard_metric_frames]
    control_late = []
    guard_late = []
    control_early = []
    guard_early = []
    for crng in CRNGS:
        c = metrics[(crng, "control")]
        g = metrics[(crng, "klguard")]
        c_steps = pd.to_numeric(c["global_step"], errors="coerce")
        g_steps = pd.to_numeric(g["global_step"], errors="coerce")
        c_early_values = pd.to_numeric(c.loc[c_steps.between(25_000 + 1, 50_000), "approx_kl"], errors="coerce").dropna()
        g_early_values = pd.to_numeric(g.loc[g_steps.between(25_000 + 1, 50_000), "approx_kl"], errors="coerce").dropna()
        c_late_values = pd.to_numeric(c.loc[c_steps.between(50_000 + 1, 100_000), "approx_kl"], errors="coerce").dropna()
        g_late_values = pd.to_numeric(g.loc[g_steps.between(50_000 + 1, 100_000), "approx_kl"], errors="coerce").dropna()
        control_early.append(float(c_early_values.mean()) if len(c_early_values) else np.nan)
        guard_early.append(float(g_early_values.mean()) if len(g_early_values) else np.nan)
        control_late.append(float(c_late_values.mean()) if len(c_late_values) else np.nan)
        guard_late.append(float(g_late_values.mean()) if len(g_late_values) else np.nan)
    payload = {
        "decision": decision,
        "control_retained_count": control_retained,
        "control_partial_degradation_count": int(control["partial_degradation_count"]),
        "control_collapse_count": int(control["collapse_count"]),
        "guard_retained_count": guard_retained,
        "guard_partial_degradation_count": int(guard["partial_degradation_count"]),
        "guard_collapse_count": int(guard["collapse_count"]),
        "control_nonretained_count": control_nonretained,
        "guard_material_harm_count": material_harm_count,
        "guard_frozen_count": int(pair_stats["guard_frozen_count"]),
        "guard_all_branches_real_optimizer_updates": all_guard_updates,
        "paired_stats": pair_stats,
        "guard_trigger_rates_by_crng": guard_trigger_rates,
        "update_magnitude_approx_kl_25_50_control_by_crng": control_early,
        "update_magnitude_approx_kl_25_50_guard_by_crng": guard_early,
        "update_magnitude_approx_kl_50_100_control_by_crng": control_late,
        "update_magnitude_approx_kl_50_100_guard_by_crng": guard_late,
        "fixed_threshold": EFFECTIVE_KL_THRESHOLD,
        "configured_target_kl": CONFIGURED_TARGET_KL,
        "decision_rules_source": rel(PROMPT),
        "training_seed": TRAINING_SEED,
        "continuation_rngs": [f"CRNG{x}" for x in CRNGS],
    }
    DECISION_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([payload]).to_csv(DECISION_OUT, index=False, encoding="utf-8-sig")
    write_json(OUT / "decision.json", payload)
    return payload


def write_report(decision: dict[str, Any], summary: pd.DataFrame, pairwise: pd.DataFrame, windows: pd.DataFrame, start_audit: dict[str, Any]) -> None:
    rows = []
    for _, row in pairwise.iterrows():
        rows.append({
            "CRNG": row["continuation_rng"], "CONTROL 100K": row["control_yield_100K"], "GUARD 100K": row["guard_yield_100K"], "Delta": row["delta_yield_100K"],
            "CONTROL class": row["control_retention_class"], "GUARD class": row["guard_retention_class"], "harm": row["material_harm"],
        })
    pair_table = pd.DataFrame(rows).to_markdown(index=False)
    summary_table = summary.fillna("not applicable").to_markdown(index=False)

    def pooled_window_mean(branch: str, labels: list[str]) -> float:
        subset = windows[(windows["branch"] == branch) & windows["window"].isin(labels)]
        if subset.empty or subset["approx_kl_mean"].isna().all():
            return float("nan")
        weights = subset["update_count"].fillna(0).to_numpy(dtype=float)
        values = subset["approx_kl_mean"].to_numpy(dtype=float)
        if weights.sum() <= 0:
            return float(np.nanmean(values))
        return float(np.average(values, weights=weights))

    c_early = pooled_window_mean("CONTROL", ["25-35K", "35-50K"])
    g_early = pooled_window_mean("KL-GUARD", ["25-35K", "35-50K"])
    # The requested windows are 50-75K and 75-100K; report the pooled 50-100K
    # descriptive mean explicitly, without introducing a new analysis window.
    c_late = pooled_window_mean("CONTROL", ["50-75K", "75-100K"])
    g_late = pooled_window_mean("KL-GUARD", ["50-75K", "75-100K"])
    lines = [
        "# YC PPO Original seed0@25K three-RNG fixed KL-GUARD retention screen",
        "",
        "## Status and decision",
        "",
        f"- Primary decision: `{decision['decision']}`.",
        "- `YC_PPO_SEED0_25K_3RNG_RETENTION_SCREEN_COMPLETE`",
        "- This is a screen with n=3 paired continuation RNGs; no significance test is claimed.",
        "",
        "## Fixed starting point",
        "",
        f"- The common starting point is the new 025 CONTROL seed0@25K checkpoint, not the historical 055 checkpoint: `{start_audit['path']}`.",
        f"- SHA256: `{start_audit['sha256']}`; archive/model `num_timesteps`: `{start_audit['num_timesteps']}` / `{start_audit['model_num_timesteps']}`.",
        f"- Optimizer state entries: archive `{start_audit['optimizer_state_entries_archive']}`, loaded model `{start_audit['optimizer_state_entries_loaded_model']}`; schedule metadata was available and continuation was permitted.",
        f"- Existing 025 deterministic masked validation evidence: `{start_audit['validation_source']}`, `{start_audit['validation_year_count']}` years, mean yield `{start_audit['validation_mean_yield_kg_ha']:.3f}` kg/ha, status `{start_audit['validation_status']}`.",
        "- The 25K validation rows in each 026 output are explicitly reused from this 025 deterministic start evidence; they are not re-labeled historical 055 data and are not counted as new training.",
        "",
        "## Experimental contract",
        "",
        "- Training seed remains Original seed0. CRNG1/2/3 are continuation RNG initializations after 25K, not PPO training seeds 1/2/3.",
        "- Each pair loads the same policy/value weights, optimizer state, timestep and schedule metadata, then requests +75K steps to 100K. CONTROL uses `target_kl=None`; KL-GUARD uses configured `target_kl=6.666666667e-05`.",
        f"- Guard source semantics were rechecked as `approx_kl > 1.5 * target_kl`, giving effective threshold `{EFFECTIVE_KL_THRESHOLD:.1e}`.",
        "- Observation, reward, action space, mask logic, irrigation/fertilizer constraints, training/validation years, IC, cultivar, PPO hyperparameters and runtime wrapper are unchanged. The only scientific intervention is fixed KL-based update early stop.",
        "",
        "## Continuation training",
        "",
        "- All six continuation branches reached 100K with checkpoints at 35K, 50K, 75K and 100K; 25K is the common referenced start.",
        f"- Six training-metrics files were generated under `results/`; per-branch manifests are under `{rel(OUT)}`.",
        f"- Guard trigger rates by CRNG: `{decision['guard_trigger_rates_by_crng']}`; frozen guard branches: `{decision['guard_frozen_count']}`; all guard branches had real optimizer updates: `{decision['guard_all_branches_real_optimizer_updates']}`.",
        "",
        "## Retention classifications",
        "",
        "Retention applies collapse first, then retained: collapse if 100K yield <6500 or start−100K >1500; retained if 100K yield ≥7500 and start−100K ≤700; otherwise partial degradation.",
        "",
        summary_table,
        "",
        "## Pairwise 100K result",
        "",
        f"- Pairwise output: `{rel(PAIRED_OUT)}`.",
        pair_table,
        "",
        f"- Paired Δyield mean/median/minimum: `{decision['paired_stats']['mean_paired_delta_yield_100K']:.3f}` / `{decision['paired_stats']['median_paired_delta_yield_100K']:.3f}` / `{decision['paired_stats']['minimum_paired_delta_yield_100K']:.3f}` kg/ha.",
        f"- Guard retention class better/worse: `{decision['paired_stats']['guard_retention_class_better_count']}` / `{decision['paired_stats']['guard_retention_class_worse_count']}`; material-harm pairs `{decision['guard_material_harm_count']}`.",
        "",
        "## Update-window description",
        "",
        f"- Window output: `{rel(WINDOW_OUT)}`; fixed windows are 25–35K, 35–50K, 50–75K and 75–100K.",
        f"- For a compact cross-window description, pooled 25–50K approx_kl mean was CONTROL `{c_early:.8f}` versus KL-GUARD `{g_early:.8f}`; pooled 50–100K was CONTROL `{c_late:.8f}` versus KL-GUARD `{g_late:.8f}`. The CSV retains the preregistered four windows and per-CRNG means/medians for all requested metrics.",
        "- These are descriptive mechanism-supporting measurements only; they do not establish KL as the unique causal root.",
        "",
        "## Historical context boundary",
        "",
        "- 025 full-training seed0 is used only as background for the retention-failure phenotype. The primary denominator and comparison here are the six paired continuations from the same new 025@25K checkpoint.",
        "- This screen tests the working post-acquisition retention hypothesis. A positive result would be retention evidence under YC/seed0/25K, not a claim of universal PPO-collapse prevention.",
        "",
        "## Required answers",
        "",
        f"1. 025 seed0@25K located and verified: `True`; optimizer state recoverable: `True`; deterministic masked validation mean `{start_audit['validation_mean_yield_kg_ha']:.3f}` kg/ha.",
        f"2. Three CONTROL continuations complete to 100K: `True`.",
        f"3. Three KL-GUARD continuations complete to 100K: `True`.",
        f"4. CONTROL RETAINED/PARTIAL/COLLAPSE: `{decision['control_retained_count']}/{decision['control_partial_degradation_count']}/{decision['control_collapse_count']}`.",
        f"5. KL-GUARD RETAINED/PARTIAL/COLLAPSE: `{decision['guard_retained_count']}/{decision['guard_partial_degradation_count']}/{decision['guard_collapse_count']}`.",
        f"6. Paired 100K Δyield by CRNG: `{pairwise[['continuation_rng', 'delta_yield_100K']].to_dict('records')}` kg/ha.",
        f"7. Material harm in any pair: `{decision['guard_material_harm_count'] > 0}`; count `{decision['guard_material_harm_count']}`.",
        f"8. Guard triggered and not frozen: rates `{decision['guard_trigger_rates_by_crng']}`; frozen count `{decision['guard_frozen_count']}`.",
        f"9. Update magnitude direction: 25–50K pooled approx_kl CONTROL/GUARD `{c_early:.8f}/{g_early:.8f}`; 50–100K `{c_late:.8f}/{g_late:.8f}`. Consult the four-window CSV for consistency by CRNG.",
        f"10. Post-acquisition retention hypothesis support: `{decision['decision'] in {'SEED0_RETENTION_GUARD_PROMISING', 'SEED0_RETENTION_GUARD_PARTIAL'}}` under this screen; this is not a universal causal claim.",
        f"11. Final allowed decision: `{decision['decision']}`.",
        "",
        "## Stop boundary",
        "",
        "No fifth RNG, seed2, Augmented, other site, KL-threshold tuning, PPO hyperparameter change, or new intervention was started.",
        "",
        "YC_PPO_SEED0_25K_3RNG_RETENTION_SCREEN_COMPLETE",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_report() -> dict[str, Any]:
    metrics: dict[tuple[int, str], pd.DataFrame] = {}
    aggregates: dict[tuple[int, str], pd.DataFrame] = {}
    for crng in CRNGS:
        for branch in ["control", "klguard"]:
            metrics[(crng, branch)] = extract_metrics(crng, branch)
            aggregates[(crng, branch)] = load_aggregates(crng, branch)
            if len(aggregates[(crng, branch)]) != len(ALL_CHECKPOINTS):
                raise RuntimeError(f"{branch_key(crng, branch)} validation aggregate count is not five")
    start_yield = read_start_yield()
    windows = make_windows(metrics)
    pairwise, pair_stats = make_pairwise(aggregates, metrics, start_yield)
    summary = make_summary(aggregates, pairwise, pair_stats, start_yield)
    decision = make_decision(summary, pairwise, pair_stats, metrics)
    start_audit = starting_checkpoint_audit()
    write_report(decision, summary, pairwise, windows, start_audit)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["preflight", "train", "validate", "report", "all"], default="all")
    parser.add_argument("--crng", choices=["1", "2", "3", "all"], default="all")
    args = parser.parse_args()
    if args.phase == "preflight":
        print(json.dumps(preflight(), ensure_ascii=False, indent=2))
        return 0
    pf = preflight()
    if pf["status"] != "PASS":
        print(json.dumps(pf, ensure_ascii=False, indent=2))
        return 2
    crngs = CRNGS if args.crng == "all" else [int(args.crng)]
    if args.phase == "train":
        results: dict[str, Any] = {}
        for crng in crngs:
            for branch in ["control", "klguard"]:
                results[branch_key(crng, branch)] = train_continuation(crng, branch)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    if args.phase == "validate":
        results: dict[str, Any] = {}
        for crng in crngs:
            for branch in ["control", "klguard"]:
                results[branch_key(crng, branch)] = len(validate_branch(crng, branch))
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    if args.phase == "report":
        print(json.dumps(run_report(), ensure_ascii=False, indent=2))
        return 0
    # Explicit serial order keeps memory bounded and makes each pair auditable.
    for crng in CRNGS:
        train_continuation(crng, "control")
        train_continuation(crng, "klguard")
    for crng in CRNGS:
        validate_branch(crng, "control")
        validate_branch(crng, "klguard")
    print(json.dumps(run_report(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
