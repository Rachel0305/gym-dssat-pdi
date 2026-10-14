"""021: complete the missing YC PPO Original seed1 RNG0 CONTROL continuation.

This script is deliberately an evidence/orchestration layer.  It reuses the
audited 017/018B/019/020 runtime and does not edit PPO, DSSAT, observation,
reward, action, mask, or scientific-input code.  The only new continuation is
the prompt-authorized RNG0 retry from the common 75K checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
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
SCREEN020_OUT = ROOT / "benchmark_results" / "020_yc_ppo_seed1_multirng_control_screen"

OUT = ROOT / "benchmark_results" / "021_yc_ppo_seed1_rng0_completion"
RESULTS = ROOT / "results"
RECHECK_OUT = RESULTS / "yc_ppo_s1_rng0_dssat_2011_recheck.csv"
RETRY_METRICS_OUT = RESULTS / "yc_ppo_s1_control_rng0_training_metrics_retry.csv"
RETRY_VALIDATION_OUT = RESULTS / "yc_ppo_s1_control_rng0_checkpoint_validation_retry.csv"
SUMMARY_OUT = RESULTS / "yc_ppo_s1_control_multirng_reproducibility.csv"
LATE_OUT = RESULTS / "yc_ppo_s1_control_multirng_late_update_summary.csv"
REPORT = ROOT / "docs" / "yc_ppo_original_seed1_rng0_completion.md"

STATION = "YCA"
SITE = "YC"
PPO_SEED = 1
RNG0 = 0
ALL_RNGS = [0, 1, 2, 3, 4]
TRAIN_YEARS = list(range(2004, 2014))
VALIDATION_YEARS = list(range(2014, 2024))
CHECKPOINTS = [75_000, 80_000, 85_000, 90_000, 95_000, 100_000]
CONTINUATION_STEPS = 25_000
RECHECK_STAGES = ["environment_reset", "dssat_launch", "first_step_serving"]
REQUIRED_METRICS = [
    "global_step", "approx_kl", "clip_fraction", "entropy_loss",
    "policy_gradient_loss", "value_loss", "explained_variance",
    "learning_rate", "clip_range",
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_data(path: Path) -> dict[str, Any]:
    import zipfile

    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("data"))


def import_screen020() -> Any:
    import run_yc_ppo_seed1_multirng_control_020 as screen020

    # The imported module contains only functions/constants; its main() is not
    # executed.  Redirect its helper functions to this isolated 021 root.
    screen020.OUT = OUT
    screen020.REPORT = REPORT
    screen020.RESULTS = RESULTS
    return screen020


def load_context() -> tuple[Any, Any, dict[str, Any], dict[str, Any], Any, Any]:
    """Load the exact 017 scientific runtime with a 021 output namespace."""
    import diagnose_yc_ppo_original_seed1_collapse_017 as diagnostic

    screen020 = import_screen020()
    cfg = json.loads((ORIGINAL_RUN / "configs" / "222YCA_O1_yc_v1_original_seed1.json").read_text(encoding="utf-8"))
    replay, config, env_config, runtime_state = screen020.isolated_runtime_setup(cfg, diagnostic)
    return diagnostic, screen020, replay, config, env_config, runtime_state


def preflight() -> dict[str, Any]:
    required = [ORIGINAL_CHECKPOINT, PRIOR_RESUME_AUDIT, PRIOR_TRAINING, PRIOR_VALIDATION]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required prior evidence: " + ", ".join(missing))
    source = checkpoint_data(ORIGINAL_CHECKPOINT)
    if int(source.get("num_timesteps", -1)) != 75_000:
        raise RuntimeError(f"Original checkpoint num_timesteps={source.get('num_timesteps')}, expected 75000")
    control_manifest = PRIOR_018B_CONTROL / "training_manifest.json"
    if not control_manifest.exists():
        raise FileNotFoundError(rel(control_manifest))
    manifest = json.loads(control_manifest.read_text(encoding="utf-8"))
    if manifest.get("target_kl") is not None:
        raise RuntimeError("018B CONTROL manifest is not target_kl=None")
    old_failed = sorted(rel(path) for path in (SCREEN020_OUT / "failed_attempts").rglob("*") if path.is_file())
    payload = {
        "status": "PASS",
        "prompt": rel(ROOT / "prompt_01" / "021_codex_prompt_020_yc_ppo_seed1_rng0_completion.md"),
        "common_starting_checkpoint": rel(ORIGINAL_CHECKPOINT),
        "common_starting_checkpoint_sha256": sha256_file(ORIGINAL_CHECKPOINT),
        "common_starting_num_timesteps": source.get("num_timesteps"),
        "prior_resume_audit": rel(PRIOR_RESUME_AUDIT),
        "prior_018b_control_manifest": rel(control_manifest),
        "prior_018b_control_target_kl": manifest.get("target_kl"),
        "prior_multirng_report": rel(ROOT / "docs" / "yc_ppo_original_seed1_multirng_control_reproducibility.md"),
        "prior_rng0_failed_attempt_evidence": old_failed or ["unknown / insufficient evidence"],
        "scientific_contract": "Original YC/YCA low-IC CONTROL; unchanged; continuation RNG=0",
        "formal_code_modified": False,
    }
    write_json(OUT / "preflight.json", payload)
    return payload


def _json_from_output(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _probe_log_text(stage: str) -> str:
    root = OUT / "recheck_runtime" / stage
    snippets: list[str] = []
    if root.exists():
        for path in sorted(root.rglob("*.log")):
            try:
                snippets.append(f"{rel(path)}: " + path.read_text(encoding="utf-8", errors="replace")[-2000:])
            except OSError as exc:
                snippets.append(f"{rel(path)}: log read failed: {exc}")
    return "\n".join(snippets)


def probe_child(stage: str) -> int:
    """Run one probe in a kill-isolated child; native DSSAT FPE cannot be caught."""
    if stage not in RECHECK_STAGES:
        raise ValueError(stage)
    diagnostic, _screen020, replay, config, env_config, runtime_state = load_context()
    import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base

    local = copy.deepcopy(env_config)
    local["paths"]["output_root"] = rel(OUT / "recheck_runtime" / stage)
    env = None
    try:
        env = base.make_env(
            config, local, STATION, 2011, RNG0,
            f"021_rng0_recheck_{stage}", evaluation=False,
        )
        obs, info = env.reset()
        notes = [f"reset_observation_shape={tuple(np.asarray(obs).shape)}"]
        if stage == "dssat_launch":
            folder = getattr(env.unwrapped, "_tmp_folder", None)
            notes.append(f"tmp_folder_present={bool(folder)}")
        elif stage == "first_step_serving":
            from sb3_contrib.common.maskable.utils import get_action_masks

            mask = np.asarray(get_action_masks(env), dtype=bool)
            if mask.size == 0 or not bool(mask.any()):
                raise RuntimeError("first-step action mask is empty")
            action = 0
            if not bool(mask[action]):
                action = int(np.flatnonzero(mask)[0])
            step_result = env.step(action)
            notes.extend([f"mask_size={int(mask.size)}", f"selected_action={action}", f"step_arity={len(step_result)}"])
        print(json.dumps({"stage": stage, "status": "PASS", "notes": "; ".join(notes)}, ensure_ascii=False))
        return 0
    except BaseException as exc:
        print(json.dumps({"stage": stage, "status": "FAIL", "notes": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass
        diagnostic.restore_runtime(runtime_state)


def run_recheck() -> tuple[pd.DataFrame, bool]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for stage in RECHECK_STAGES:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--probe-stage", stage],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        payload = _json_from_output(proc.stdout) or _json_from_output(proc.stderr)
        combined = (proc.stdout + "\n" + proc.stderr + "\n" + _probe_log_text(stage)).lower()
        fpe = bool(
            "floating point exception" in combined
            or "floating_point_exception" in combined
            or proc.returncode in {-8, 8, 134}
        )
        status = "PASS" if proc.returncode == 0 and payload.get("status") == "PASS" and not fpe else "FAIL"
        notes = str(payload.get("notes", "unknown / insufficient evidence"))
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().replace("\n", " ")[-1200:]
            notes = f"{notes}; child_output={tail or 'unknown / insufficient evidence'}"
        rows.append({
            "stage": stage,
            "status": status,
            "exit_code": int(proc.returncode),
            "floating_point_exception": bool(fpe),
            "notes": notes,
        })
    result = pd.DataFrame(rows, columns=["stage", "status", "exit_code", "floating_point_exception", "notes"])
    RECHECK_OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(RECHECK_OUT, index=False, encoding="utf-8-sig")
    passed = bool(result["status"].eq("PASS").all()) and not bool(result["floating_point_exception"].any())
    write_json(OUT / "recheck_manifest.json", {
        "status": "PASS" if passed else "FAIL",
        "result_csv": rel(RECHECK_OUT),
        "stages": rows,
        "infrastructure_failure_reproducible": bool(not passed and result["floating_point_exception"].all()),
        "scientific_input_change": False,
    })
    return result, passed


def _run_worker(worker: str) -> tuple[int, dict[str, Any], str]:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), f"--{worker}"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    payload = _json_from_output(proc.stdout) or _json_from_output(proc.stderr)
    output = (proc.stdout + "\n" + proc.stderr).strip()
    return int(proc.returncode), payload, output


def train_worker() -> int:
    diagnostic, screen020, replay, config, env_config, runtime_state = load_context()
    try:
        manifest = screen020.train_rng(RNG0, replay, config, env_config)
        print(json.dumps({"status": "PASS", "manifest": manifest}, ensure_ascii=False))
        return 0
    finally:
        diagnostic.restore_runtime(runtime_state)


def extract_retry_metrics(trajectory_status: str = "COMPLETE") -> pd.DataFrame:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = sorted((OUT / "isolated" / "rng0" / "tensorboard").glob("**/events*"))
    if len(event_files) != 1:
        raise RuntimeError(f"Expected exactly one RNG0 retry event file, got {len(event_files)}")
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
    for tag, name in mapping.items():
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        for event in accumulator.Scalars(tag):
            step = int(event.step)
            records.setdefault(step, {"continuation_rng": RNG0, "global_step": step, "timestep": step})[name] = float(event.value)
    metrics = pd.DataFrame(sorted(records.values(), key=lambda row: row["global_step"]))
    for column in REQUIRED_METRICS:
        if column not in metrics.columns:
            metrics[column] = np.nan
    metrics = metrics[["continuation_rng", *REQUIRED_METRICS, "timestep"]]
    metrics["trajectory_status"] = trajectory_status
    metrics["source"] = rel(event_files[0])
    RETRY_METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RETRY_METRICS_OUT, index=False, encoding="utf-8-sig")
    write_json(OUT / "isolated" / "rng0" / "tensorboard_manifest_retry.json", {
        "continuation_rng": RNG0,
        "event_file": rel(event_files[0]),
        "event_file_sha256": sha256_file(event_files[0]),
        "scalar_tags": accumulator.Tags().get("scalars", []),
        "rows": int(len(metrics)),
        "missing_required_columns": [column for column in REQUIRED_METRICS if metrics[column].isna().all()],
    })
    if metrics.empty:
        raise RuntimeError("RNG0 retry TensorBoard metrics are empty")
    return metrics


def record_failure_evidence() -> pd.DataFrame:
    """Record the observed event prefix without treating it as a complete run."""
    metrics = extract_retry_metrics("INCOMPLETE_TRAINING_PREFIX")
    log_path = OUT / "isolated" / "rng0" / "runtime" / "logs" / "YCA" / "YCA_2011_YCA_2011_032_22_train.log"
    write_json(OUT / "observed_fpe_manifest.json", {
        "status": "INCOMPLETE_TRAINING_PREFIX",
        "continuation_rng": RNG0,
        "last_tensorboard_step": int(pd.to_numeric(metrics["global_step"], errors="coerce").max()),
        "metrics_csv": rel(RETRY_METRICS_OUT),
        "fpe_log": rel(log_path) if log_path.exists() else "unknown / insufficient evidence",
        "fpe_log_contains_floating_point_exception": bool(log_path.exists() and "floating point exception" in log_path.read_text(encoding="utf-8", errors="replace").lower()),
        "not_used_for_complete_5_rng_summary": True,
    })
    return metrics


def run_training() -> tuple[pd.DataFrame, dict[str, Any]]:
    code, payload, output = _run_worker("train-worker")
    if code != 0:
        write_json(OUT / "training_failure_manifest.json", {
            "status": "FAIL",
            "exit_code": code,
            "child_payload": payload,
            "child_output_tail": output[-5000:],
            "starting_checkpoint": rel(ORIGINAL_CHECKPOINT),
            "continuation_rng": RNG0,
            "scientific_contract_change": False,
        })
        raise RuntimeError(f"RNG0 retry training worker failed with exit_code={code}")
    metrics = extract_retry_metrics()
    manifest_path = OUT / "isolated" / "rng0" / "training_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(rel(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["retry_result_metrics"] = rel(RETRY_METRICS_OUT)
    manifest["recheck_result"] = rel(RECHECK_OUT)
    manifest["formal_completion_scope"] = "Only missing RNG0 CONTROL continuation from original 75K"
    write_json(OUT / "isolated" / "rng0" / "retry_manifest.json", manifest)
    return metrics, manifest


def _validation_rows_for_checkpoint(model: Any, config: dict[str, Any], env_config: dict[str, Any], checkpoint: int, branch: str, paired: Any, base: Any, direct: Any, daily_root: Path, screen020: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year in VALIDATION_YEARS:
        daily, strict = paired.daily_records(model, config, env_config, year, checkpoint, branch, base, direct)
        daily_path = daily_root / f"YCA_{year}_seed1_ckpt{checkpoint}_daily.csv"
        daily_path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        requested = pd.to_numeric(daily["requested_discrete_action_index"], errors="coerce")
        executed = pd.to_numeric(daily["discrete_action_index"], errors="coerce")
        rows.append({
            "row_type": "year", "continuation_rng": RNG0, "branch": branch,
            "checkpoint_step": int(checkpoint), "model_num_timesteps": screen020.checkpoint_data(screen020.checkpoint_path(RNG0, checkpoint)).get("num_timesteps"),
            "year": int(year),
            "yield_kg_ha": float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1]),
            "I_mm": float(pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0).sum()),
            "N_kg_ha": float(pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0).sum()),
            "PFP_N": strict["PFP_N_kg_kg"], "WP_ET": strict["WP_ET_kg_m3"], "ETCP_mm": strict["etcp_mm"],
            "invalid_count": int((requested != executed).sum()),
            "fallback_count": int(pd.Series(daily["mask_forced_noop"]).astype(str).str.lower().eq("true").sum()),
            "daily_path": rel(daily_path), "validation_source": "021_rng0_retry_deterministic_2014_2023", "status": "PASS",
        })
    return rows


def validate_worker() -> int:
    diagnostic, screen020, replay, config, env_config, runtime_state = load_context()
    try:
        from sb3_contrib import MaskablePPO

        import run_all_year_direct_action_safe_ppo as direct
        import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base
        import run_yc_ppo_seed1_kl_guard_paired_019 as paired

        if RETRY_VALIDATION_OUT.exists():
            old = pd.read_csv(RETRY_VALIDATION_OUT, keep_default_na=False)
            agg = old[old["row_type"].astype(str).eq("aggregate")]
            if set(pd.to_numeric(agg["checkpoint_step"], errors="coerce").dropna().astype(int)) == set(CHECKPOINTS):
                print(json.dumps({"status": "PASS", "reused": True, "aggregate_rows": len(agg)}, ensure_ascii=False))
                return 0
            raise RuntimeError("Partial RNG0 retry validation output exists; refusing overwrite")

        reference = pd.read_csv(PRIOR_VALIDATION, keep_default_na=False)
        reference_75 = reference[pd.to_numeric(reference["checkpoint_step"], errors="coerce").eq(75_000)].copy()
        if reference_75.empty:
            raise RuntimeError("018B deterministic validation has no 75K reference")
        reference_75["continuation_rng"] = RNG0
        reference_75["branch"] = "control_rng0_retry"
        reference_75["validation_source"] = "018B_rng1_deterministic_reference"
        reference_75["trajectory_status"] = "COMPLETE"

        local_env_config = screen020.branch_env_config(env_config, RNG0)
        daily_root = OUT / "isolated" / "rng0" / "daily_outputs" / STATION
        detail_rows: list[dict[str, Any]] = []
        for checkpoint in CHECKPOINTS[1:]:
            model_path = screen020.checkpoint_path(RNG0, checkpoint)
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            model = MaskablePPO.load(str(model_path), device="cpu")
            detail_rows.extend(_validation_rows_for_checkpoint(
                model, config, local_env_config, checkpoint, "control_rng0_retry",
                paired, base, direct, daily_root, screen020,
            ))
        detail = pd.DataFrame(detail_rows)
        aggregate_rows: list[dict[str, Any]] = []
        for checkpoint, group in detail.groupby("checkpoint_step", sort=True):
            aggregate_rows.append({
                "row_type": "aggregate", "continuation_rng": RNG0, "branch": "control_rng0_retry",
                "checkpoint_step": int(checkpoint), "model_num_timesteps": int(pd.to_numeric(group["model_num_timesteps"], errors="coerce").max()),
                "year": "ALL_2014_2023", "validation_year_count": int(len(group)),
                "mean_yield_kg_ha": float(group["yield_kg_ha"].mean()), "sd_yield_kg_ha": float(group["yield_kg_ha"].std(ddof=1)),
                "mean_I_mm": float(group["I_mm"].mean()), "mean_N_kg_ha": float(group["N_kg_ha"].mean()),
                "mean_PFP_N": float(group["PFP_N"].mean()), "mean_WP_ET": float(group["WP_ET"].mean()),
                "invalid_count_total": int(group["invalid_count"].sum()), "fallback_count_total": int(group["fallback_count"].sum()),
                "validation_source": "021_rng0_retry_deterministic_2014_2023",
                "trajectory_status": "COMPLETE", "status": "PASS" if len(group) == len(VALIDATION_YEARS) else "FAIL",
            })
        combined = pd.concat([reference_75, detail, pd.DataFrame(aggregate_rows)], ignore_index=True, sort=False)
        RETRY_VALIDATION_OUT.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(RETRY_VALIDATION_OUT, index=False, encoding="utf-8-sig")
        detail.to_csv(OUT / "isolated" / "rng0" / "validation_by_year.csv", index=False, encoding="utf-8-sig")
        write_json(OUT / "isolated" / "rng0" / "validation_manifest.json", {
            "status": "PASS" if len(detail) == 5 * len(VALIDATION_YEARS) else "FAIL",
            "validation_years": VALIDATION_YEARS,
            "checkpoints": CHECKPOINTS,
            "output": rel(RETRY_VALIDATION_OUT),
            "reference_75k": "018B_rng1_deterministic_reference",
        })
        print(json.dumps({"status": "PASS", "aggregate_rows": len(aggregate_rows), "detail_rows": len(detail)}, ensure_ascii=False))
        return 0
    finally:
        diagnostic.restore_runtime(runtime_state)


def validate_rng_aggregate(path: Path, rng: int, source: str) -> pd.DataFrame:
    data = pd.read_csv(path, keep_default_na=False)
    if "row_type" not in data.columns:
        raise RuntimeError(f"No row_type in {rel(path)}")
    aggregate = data[data["row_type"].astype(str).eq("aggregate")].copy()
    aggregate["checkpoint_step"] = pd.to_numeric(aggregate["checkpoint_step"], errors="coerce")
    if set(aggregate["checkpoint_step"].dropna().astype(int)) != set(CHECKPOINTS):
        raise RuntimeError(f"{rel(path)} does not contain all six checkpoints")
    aggregate["continuation_rng"] = int(rng)
    aggregate["validation_source"] = source
    return aggregate.sort_values("checkpoint_step").reset_index(drop=True)


def collapse_bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def build_summary() -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = {
        0: (RETRY_VALIDATION_OUT, "021_rng0_retry_deterministic_2014_2023"),
        1: (PRIOR_VALIDATION, "018B_rng1_control_reference"),
        2: (RESULTS / "yc_ppo_s1_control_rng2_checkpoint_validation.csv", "020_rng2_validation"),
        3: (RESULTS / "yc_ppo_s1_control_rng3_checkpoint_validation.csv", "020_rng3_validation"),
        4: (RESULTS / "yc_ppo_s1_control_rng4_checkpoint_validation.csv", "020_rng4_validation"),
    }
    rows: list[dict[str, Any]] = []
    for rng in ALL_RNGS:
        path, source = paths[rng]
        data = validate_rng_aggregate(path, rng, source).set_index("checkpoint_step")
        yields = {checkpoint: float(data.loc[checkpoint, "mean_yield_kg_ha"]) for checkpoint in CHECKPOINTS}
        pre = pd.Series({checkpoint: yields[checkpoint] for checkpoint in CHECKPOINTS[:-1]})
        best_checkpoint = int(pre.idxmax())
        best_yield = float(pre.max())
        final_yield = yields[100_000]
        drop = best_yield - final_yield
        criterion_a = final_yield < 6_500.0
        criterion_b = drop > 1_500.0
        collapse = bool(criterion_a or criterion_b)
        first_7000 = next((checkpoint for checkpoint in CHECKPOINTS if yields[checkpoint] < 7_000.0), "NA")
        first_6500 = next((checkpoint for checkpoint in CHECKPOINTS if yields[checkpoint] < 6_500.0), "NA")
        rows.append({
            "continuation_rng": rng,
            **{f"yield_{checkpoint // 1000}K": yields[checkpoint] for checkpoint in CHECKPOINTS},
            "best_pre100_checkpoint": best_checkpoint, "best_pre100_yield": best_yield,
            "drop_best_to_100K": drop, "collapse": collapse,
            "collapse_reason": ";".join(reason for reason, active in [
                ("100K_mean_yield<6500", criterion_a), ("best_pre100_minus_100K>1500", criterion_b)
            ] if active) or "none",
            "first_checkpoint_below_7000": first_7000,
            "first_checkpoint_below_6500": first_6500,
            "validation_source": source,
        })
    summary = pd.DataFrame(rows).sort_values("continuation_rng").reset_index(drop=True)
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")
    finals = pd.to_numeric(summary["yield_100K"], errors="coerce")
    drops = pd.to_numeric(summary["drop_best_to_100K"], errors="coerce")
    count = int(summary["collapse"].map(collapse_bool).sum())
    if count >= 2:
        decision = "COLLAPSE_REPRODUCIBLE"
    elif count == 1:
        decision = "COLLAPSE_SPARSE_1_OF_5"
    else:
        decision = "COLLAPSE_RARE_IN_5"
    stats = {
        "n_continuations": 5, "collapse_count": count, "collapse_rate": count / 5,
        "yield_100K_mean": float(finals.mean()), "yield_100K_sd": float(finals.std(ddof=1)),
        "yield_100K_min": float(finals.min()), "yield_100K_max": float(finals.max()),
        "decline_gt_1000_count": int((drops > 1_000).sum()), "decline_gt_1500_count": int((drops > 1_500).sum()),
        "ending_below_7000_count": int((finals < 7_000).sum()), "ending_below_6500_count": int((finals < 6_500).sum()),
        "decision": decision, "summary_csv": rel(SUMMARY_OUT),
    }
    write_json(OUT / "reproducibility_summary.json", stats)
    return summary, stats


def metric_path(rng: int) -> tuple[Path, str]:
    if rng == 0:
        return RETRY_METRICS_OUT, "021_rng0_retry_training"
    if rng == 1:
        return PRIOR_TRAINING, "018B_rng1_control_reference"
    return RESULTS / f"yc_ppo_s1_control_rng{rng}_training_metrics.csv", f"020_rng{rng}_training"


def build_late_summary(summary: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    windows = [(85_000, 90_000, "85-90K"), (90_000, 95_000, "90-95K"), (95_000, 100_000, "95-100K")]
    class_by_rng = {int(row["continuation_rng"]): bool(row["collapse"]) for _, row in summary.iterrows()}
    metrics: dict[int, pd.DataFrame] = {}
    for rng in ALL_RNGS:
        path, source = metric_path(rng)
        data = pd.read_csv(path, keep_default_na=False)
        data["global_step"] = pd.to_numeric(data["global_step"], errors="coerce")
        for column in ["approx_kl", "clip_fraction"]:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data["continuation_rng"] = rng
        data["metrics_source"] = source
        metrics[rng] = data
    rows: list[dict[str, Any]] = []
    for collapsed, label in [(True, "collapsed"), (False, "non_collapsed")]:
        rngs = [rng for rng in ALL_RNGS if class_by_rng[rng] is collapsed]
        for lower, upper, window in windows:
            kl_by_rng: dict[str, float] = {}
            clip_by_rng: dict[str, float] = {}
            for rng in rngs:
                subset = metrics[rng]
                subset = subset[(subset["global_step"] >= lower) & (subset["global_step"] < upper)]
                kl = subset["approx_kl"].dropna()
                clip = subset["clip_fraction"].dropna()
                if not kl.empty:
                    kl_by_rng[str(rng)] = float(kl.median())
                if not clip.empty:
                    clip_by_rng[str(rng)] = float(clip.median())
            rows.append({
                "trajectory_class": label, "window": window,
                "window_lower_step_inclusive": lower, "window_upper_step_exclusive": upper,
                "trajectory_count": len(rngs),
                "approx_kl_median": float(np.median(list(kl_by_rng.values()))) if kl_by_rng else np.nan,
                "clip_fraction_median": float(np.median(list(clip_by_rng.values()))) if clip_by_rng else np.nan,
                "approx_kl_trajectory_medians": json.dumps(kl_by_rng, sort_keys=True),
                "clip_fraction_trajectory_medians": json.dumps(clip_by_rng, sort_keys=True),
                "aggregation_note": "median across per-trajectory window medians; observed association only",
            })
    result = pd.DataFrame(rows)
    result.to_csv(LATE_OUT, index=False, encoding="utf-8-sig")
    comparisons: list[bool] = []
    for _, row in result[result["trajectory_class"].eq("collapsed")].iterrows():
        other = result[(result["trajectory_class"].eq("non_collapsed")) & result["window"].eq(row["window"])]
        if len(other) == 1 and pd.notna(row["approx_kl_median"]) and pd.notna(other.iloc[0]["approx_kl_median"]):
            comparisons.append(float(row["approx_kl_median"]) > float(other.iloc[0]["approx_kl_median"]))
    pattern = "YES" if comparisons and all(comparisons) else ("NO" if comparisons else "unknown / insufficient evidence")
    return result, pattern


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "unknown / insufficient evidence"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "unknown / insufficient evidence" if math.isnan(number) else f"{number:.{digits}f}"


def write_failure_report(preflight_payload: dict[str, Any], recheck: pd.DataFrame, reason: str, decision: str) -> None:
    lines = [
        "# YC PPO Original seed1 RNG0 completion — infrastructure stop",
        "",
        "本报告记录 prompt 021 的最小 infrastructure recheck。该任务只允许补完 019/020 缺失的 RNG0 CONTROL continuation；本次未启动训练。",
        "",
        "## Recheck",
        "",
        f"- Common 75K checkpoint: `{rel(ORIGINAL_CHECKPOINT)}`; SHA256 `{preflight_payload['common_starting_checkpoint_sha256']}`.",
        f"- Recheck CSV: `{rel(RECHECK_OUT)}`.",
        "",
        recheck.to_markdown(index=False),
        "",
        f"- Result: `{reason}`.",
        f"- Decision: `{decision}`.",
        "- No PPO/DSSAT code, scientific input, hyperparameter, seed/site/year scope, or formal experiment was changed.",
        "- The earlier 020 RNG0 failed-attempt artifacts remain preserved under `benchmark_results/020_yc_ppo_seed1_multirng_control_screen/failed_attempts/`.",
        "",
        decision,
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_success_report(preflight_payload: dict[str, Any], recheck: pd.DataFrame, summary: pd.DataFrame, stats: dict[str, Any], late: pd.DataFrame, pattern: str, manifest: dict[str, Any]) -> None:
    table = [
        "| RNG | 75K | 80K | 85K | 90K | 95K | 100K | best pre-100K | drop to 100K | collapse |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for _, row in summary.iterrows():
        table.append(
            f"| {int(row['continuation_rng'])} | {fmt(row['yield_75K'])} | {fmt(row['yield_80K'])} | {fmt(row['yield_85K'])} | {fmt(row['yield_90K'])} | {fmt(row['yield_95K'])} | {fmt(row['yield_100K'])} | {fmt(row['best_pre100_checkpoint'], 0)} / {fmt(row['best_pre100_yield'])} | {fmt(row['drop_best_to_100K'])} | {row['collapse']} |"
        )
    late_lines: list[str] = []
    for window in ["85-90K", "90-95K", "95-100K"]:
        c = late[(late["trajectory_class"].eq("collapsed")) & late["window"].eq(window)].iloc[0]
        n = late[(late["trajectory_class"].eq("non_collapsed")) & late["window"].eq(window)].iloc[0]
        late_lines.append(
            f"- `{window}`: approx_kl collapsed `{fmt(c['approx_kl_median'], 8)}` vs non-collapsed `{fmt(n['approx_kl_median'], 8)}`; clip_fraction collapsed `{fmt(c['clip_fraction_median'], 8)}` vs non-collapsed `{fmt(n['clip_fraction_median'], 8)}`."
        )
    failed_paths = preflight_payload.get("prior_rng0_failed_attempt_evidence", ["unknown / insufficient evidence"])
    lines = [
        "# YC PPO Original seed1 RNG0 completion",
        "",
        "本报告执行并记录 prompt 021：补完 019/020 中缺失的第 5 条 YC PPO Original seed1 CONTROL continuation（RNG0）。",
        "",
        "## Scope and contract",
        "",
        "- 这是 RNG0 continuation completion，不是 KL-GUARD 实验，不是新 intervention，也不是 hyperparameter test。",
        f"- Common starting point: `{rel(ORIGINAL_CHECKPOINT)}`; SHA256 `{preflight_payload['common_starting_checkpoint_sha256']}`; `num_timesteps=75000`.",
        "- RNG0 was reinitialized from 75K with Python/NumPy/Torch/RandomYearEnv seed 0. It was not resumed from the failed 85K prefix because the original RNG state was not serialized.",
        "- CONTROL contract: `target_kl=None`, no KL guard, entropy/clip/lr/GAE/gamma/network/reward/state/action/mask change, and no augmentation or site/year deletion.",
        "- The only newly added source is the orchestration/evidence script `src/run_yc_ppo_seed1_rng0_completion_021.py`; formal PPO/DQN/DSSAT code was not modified.",
        "",
        "## 1. YCA 2011 infrastructure recheck",
        "",
        f"- Evidence: `{rel(RECHECK_OUT)}`.",
        "- The same original YCA 2011 input path/protocol was used. No IC, cultivar, weather year, management, reward, action, or mask setting was changed.",
        "",
        recheck.to_markdown(index=False),
        "",
        "## 2. RNG0 completion",
        "",
        f"- Training manifest: `{rel(OUT / 'isolated' / 'rng0' / 'retry_manifest.json')}`.",
        f"- Training metrics: `{rel(RETRY_METRICS_OUT)}`.",
        f"- Checkpoint validation: `{rel(RETRY_VALIDATION_OUT)}`; deterministic masked validation years 2014–2023.",
        f"- Checkpoints: `{', '.join(str(x // 1000) + 'K' for x in CHECKPOINTS)}`; final `model_num_timesteps={manifest.get('final_num_timesteps', 'unknown / insufficient evidence')}`.",
        "- Restore/resume contents were the original checkpoint’s policy/value weights, optimizer state, timestep, learning-rate schedule, clip-range schedule, and recoverable training metadata. No 85K partial checkpoint was used.",
        "",
        "## 3. Complete 5-RNG CONTROL summary",
        "",
        *table,
        "",
        f"- `collapse_count = {stats['collapse_count']}/5`; `collapse_rate = {stats['collapse_rate']:.3f}`.",
        f"- 100K yield distribution: mean `{fmt(stats['yield_100K_mean'])}`, SD `{fmt(stats['yield_100K_sd'])}`, min `{fmt(stats['yield_100K_min'])}`, max `{fmt(stats['yield_100K_max'])}` kg/ha.",
        f"- Summary CSV updated in place as required: `{rel(SUMMARY_OUT)}`.",
        "",
        "## 4. Late-window secondary diagnostic",
        "",
        *late_lines,
        f"- After adding RNG0, the descriptive pattern ‘collapsed trajectories have higher late-window approx_kl’ remains: `{pattern}`.",
        f"- Late summary CSV: `{rel(LATE_OUT)}`.",
        "- This is a descriptive association only; it is not a causal claim and no KL guard or intervention was run.",
        "",
        "## 5. Attempt provenance and preservation",
        "",
        "- The earlier RNG0 failure was not discarded. Its partial checkpoints, TensorBoard event file, logs, and manifests remain under:",
        *[f"  - `{path}`" for path in failed_paths],
        "- Those partial artifacts were excluded from the final 5-RNG denominator and were not used as the retry starting point.",
        "- The retry began from the same original 75K checkpoint and used the same CONTROL/scientific contract; it changed only the continuation RNG initialization and isolated output namespace required by the prompt.",
        "",
        "## Required answers",
        "",
        "1. YCA 2011 infrastructure failure reproducible? **NO** — the required base recheck stages passed; the prior failure artifact remains preserved.",
        "2. RNG0 from 75K through 100K? **YES**.",
        f"3. RNG0 collapse? **{str(bool(summary.loc[summary['continuation_rng'].eq(0), 'collapse'].iloc[0])).upper()}** under the preregistered rule.",
        f"4. Complete 5-RNG collapse_count? **{stats['collapse_count']}**.",
        f"5. collapse_rate? **{stats['collapse_rate']:.3f} ({stats['collapse_count']}/5)**.",
        f"6. Late approx_kl pattern still aligned with collapse? **{pattern}**, descriptive only.",
        f"7. Final decision: **{stats['decision']}**.",
        "",
        "No KL-GUARD, RNG5+, threshold change, PPO hyperparameter change, YCA 2011 skip, scientific-input modification, extra seed/site, or augmented run was started.",
        "",
        "YC_PPO_SEED1_RNG0_COMPLETION_COMPLETE",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["recheck", "train", "validate", "record-failure", "all"], default="all")
    parser.add_argument("--probe-stage", choices=RECHECK_STAGES, default=None)
    parser.add_argument("--train-worker", action="store_true")
    parser.add_argument("--validate-worker", action="store_true")
    args = parser.parse_args()

    if args.probe_stage:
        return probe_child(args.probe_stage)
    if args.train_worker:
        return train_worker()
    if args.validate_worker:
        return validate_worker()

    OUT.mkdir(parents=True, exist_ok=True)
    preflight_payload = preflight()
    recheck, passed = run_recheck()
    if not passed:
        fpe_reproducible = bool(recheck["floating_point_exception"].all())
        decision = "RNG0_INFRA_FAILURE_REPRODUCIBLE" if fpe_reproducible else "unknown / insufficient evidence"
        write_failure_report(preflight_payload, recheck, "RNG0_INFRA_FAILURE_REPRODUCIBLE" if fpe_reproducible else "infrastructure recheck failed without a stable FPE classification", decision)
        print(decision)
        return 2
    if args.phase == "recheck":
        print("RNG0_INFRA_RECHECK_PASS")
        return 0
    if args.phase == "record-failure":
        metrics = record_failure_evidence()
        print(json.dumps({"status": "PASS", "trajectory_status": "INCOMPLETE_TRAINING_PREFIX", "rows": len(metrics)}, ensure_ascii=False))
        return 0

    try:
        if args.phase in {"train", "all"}:
            _metrics, _manifest = run_training()
            if args.phase == "train":
                print("RNG0_TRAINING_COMPLETE")
                return 0
        if args.phase in {"validate", "all"}:
            code, payload, output = _run_worker("validate-worker")
            if code != 0:
                write_json(OUT / "validation_failure_manifest.json", {"status": "FAIL", "exit_code": code, "child_payload": payload, "child_output_tail": output[-5000:]})
                raise RuntimeError(f"RNG0 validation worker failed with exit_code={code}")
            if args.phase == "validate":
                print("RNG0_VALIDATION_COMPLETE")
                return 0
        summary, stats = build_summary()
        late, pattern = build_late_summary(summary)
        manifest = json.loads((OUT / "isolated" / "rng0" / "retry_manifest.json").read_text(encoding="utf-8"))
        write_success_report(preflight_payload, recheck, summary, stats, late, pattern, manifest)
        print(stats["decision"])
        print("YC_PPO_SEED1_RNG0_COMPLETION_COMPLETE")
        return 0
    except Exception as exc:
        write_json(OUT / "completion_failure_manifest.json", {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
        print(f"YC_PPO_SEED1_RNG0_COMPLETION_INCOMPLETE: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
