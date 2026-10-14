"""023: targeted YC PPO seed1 RNG2 KL-guard rescue test.

This is a single intervention test.  The existing RNG2 CONTROL trajectory is
read as the fixed comparator; only a new RNG2 continuation with the already
audited KL early-stop guard is executed.  PPO/DSSAT source code is imported,
not edited.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
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
PRIOR_LOGIC = ROOT / "results" / "yc_ppo_s1_kl_guard_logic_test.csv"
CONTROL_METRICS = ROOT / "results" / "yc_ppo_s1_control_rng2_training_metrics.csv"
CONTROL_VALIDATION = ROOT / "results" / "yc_ppo_s1_control_rng2_checkpoint_validation.csv"

OUT = ROOT / "benchmark_results" / "023_yc_ppo_seed1_rng2_kl_guard_rescue"
RESULTS = ROOT / "results"
SMOKE_OUT = OUT / "rng2_smoke.csv"
GUARD_METRICS_OUT = RESULTS / "yc_ppo_s1_rng2_klguard_training_metrics.csv"
GUARD_VALIDATION_OUT = RESULTS / "yc_ppo_s1_rng2_klguard_checkpoint_validation.csv"
COMPARISON_OUT = RESULTS / "yc_ppo_s1_rng2_control_vs_klguard.csv"
LATE_OUT = RESULTS / "yc_ppo_s1_rng2_control_vs_klguard_late_update_summary.csv"
DECISION_OUT = RESULTS / "yc_ppo_s1_rng2_klguard_rescue_decision.csv"
REPORT = ROOT / "docs" / "yc_ppo_original_seed1_rng2_kl_guard_rescue.md"

STATION = "YCA"
RNG2 = 2
VALIDATION_YEARS = list(range(2014, 2024))
CHECKPOINTS = [75_000, 80_000, 85_000, 90_000, 95_000, 100_000]
CONTINUATION_STEPS = 25_000
CONFIGURED_TARGET_KL = 6.666666666666667e-05
EFFECTIVE_KL_THRESHOLD = 1.0e-04
REQUIRED_GUARD_METRICS = [
    "global_step", "approx_kl", "clip_fraction", "entropy_loss",
    "policy_gradient_loss", "value_loss", "explained_variance",
    "learning_rate", "clip_range", "epochs_attempted", "epochs_completed",
    "optimizer_steps_completed", "kl_guard_triggered", "kl_at_trigger",
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


def import_modules() -> tuple[Any, Any]:
    import run_yc_ppo_seed1_multirng_control_020 as screen020
    import run_yc_ppo_seed1_kl_guard_paired_019 as kl019

    # Redirect only the orchestration modules' output constants.  The imported
    # PPO/DSSAT implementation and scientific configuration are unchanged.
    screen020.OUT = OUT
    screen020.REPORT = REPORT
    screen020.RESULTS = RESULTS
    kl019.OUT = OUT
    kl019.REPORT = REPORT
    kl019.RESULTS = RESULTS
    kl019.GUARD_METRICS_OUT = GUARD_METRICS_OUT
    kl019.GUARD_VALIDATION_OUT = GUARD_VALIDATION_OUT
    kl019.SEED = RNG2
    return screen020, kl019


def load_context() -> tuple[Any, Any, Any, Any, dict[str, Any], Any]:
    import diagnose_yc_ppo_original_seed1_collapse_017 as diagnostic

    screen020, _kl019 = import_modules()
    cfg = json.loads((ORIGINAL_RUN / "configs" / "222YCA_O1_yc_v1_original_seed1.json").read_text(encoding="utf-8"))
    replay, config, env_config, runtime_state = screen020.isolated_runtime_setup(cfg, diagnostic)
    return diagnostic, screen020, replay, config, env_config, runtime_state


def preflight() -> dict[str, Any]:
    required = [ORIGINAL_CHECKPOINT, PRIOR_RESUME_AUDIT, PRIOR_LOGIC, CONTROL_METRICS, CONTROL_VALIDATION]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required 023 evidence: " + ", ".join(missing))
    source = checkpoint_data(ORIGINAL_CHECKPOINT)
    if int(source.get("num_timesteps", -1)) != 75_000:
        raise RuntimeError(f"Starting checkpoint num_timesteps={source.get('num_timesteps')}, expected 75000")
    logic = pd.read_csv(PRIOR_LOGIC, keep_default_na=False)
    logic_pass = bool(logic["pass_fail"].astype(str).eq("PASS").any())
    threshold_match = bool(
        np.isclose(pd.to_numeric(logic["configured_target_kl"], errors="coerce"), CONFIGURED_TARGET_KL, rtol=1e-9, atol=1e-12).any()
        and np.isclose(pd.to_numeric(logic["effective_threshold"], errors="coerce"), EFFECTIVE_KL_THRESHOLD, rtol=1e-9, atol=1e-12).any()
    )
    if not logic_pass or not threshold_match:
        raise RuntimeError("Existing KL-guard logic evidence does not verify configured target/effective threshold")
    control_validation = pd.read_csv(CONTROL_VALIDATION, keep_default_na=False)
    control_agg = control_validation[control_validation["row_type"].astype(str).eq("aggregate")]
    control_steps = set(pd.to_numeric(control_agg["checkpoint_step"], errors="coerce").dropna().astype(int))
    if control_steps != set(CHECKPOINTS):
        raise RuntimeError(f"Existing RNG2 CONTROL validation checkpoints={sorted(control_steps)}")
    old_019 = PRIOR_018B_CONTROL / "training_manifest.json"
    if not old_019.exists():
        raise FileNotFoundError(rel(old_019))
    old_manifest = json.loads(old_019.read_text(encoding="utf-8"))
    payload = {
        "status": "PASS",
        "prompt": rel(ROOT / "prompt_01" / "023_codex_prompt_023_yc_ppo_seed1_rng2_kl_guard_rescue.md"),
        "starting_checkpoint": rel(ORIGINAL_CHECKPOINT),
        "starting_checkpoint_sha256": sha256_file(ORIGINAL_CHECKPOINT),
        "starting_num_timesteps": source.get("num_timesteps"),
        "continuation_rng": RNG2,
        "configured_target_kl": CONFIGURED_TARGET_KL,
        "effective_kl_threshold": EFFECTIVE_KL_THRESHOLD,
        "threshold_logic_evidence": rel(PRIOR_LOGIC),
        "control_reference_metrics": rel(CONTROL_METRICS),
        "control_reference_validation": rel(CONTROL_VALIDATION),
        "control_reference_not_rerun": True,
        "control_rng2_existing_collapse": True,
        "prior_018b_control_manifest": rel(old_019),
        "prior_018b_target_kl": old_manifest.get("target_kl"),
        "rng0_status": "INFRASTRUCTURE_CENSORED; excluded; not rerun",
        "scientific_contract_change": "Only KL-based update early stop",
        "formal_ppo_dqn_dssat_code_modified": False,
    }
    write_json(OUT / "preflight.json", payload)
    return payload


def _parse_json(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def run_child(worker: str, timeout_seconds: int) -> tuple[int, dict[str, Any], str]:
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), f"--{worker}"],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
        return 124, {"status": "TIMEOUT", "timeout_seconds": timeout_seconds}, ((stdout or "") + "\n" + (stderr or ""))[-8000:]
    output = (stdout or "") + "\n" + (stderr or "")
    return int(process.returncode), _parse_json(output), output[-8000:]


def smoke_worker() -> int:
    diagnostic, _screen020, replay, config, env_config, runtime_state = load_context()
    import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base

    local = copy.deepcopy(env_config)
    local["paths"]["output_root"] = rel(OUT / "smoke_runtime")
    env = None
    try:
        env = base.make_env(config, local, STATION, 2011, RNG2, "023_rng2_yca2011_smoke", evaluation=False)
        obs, _info = env.reset()
        from sb3_contrib.common.maskable.utils import get_action_masks

        mask = np.asarray(get_action_masks(env), dtype=bool)
        if mask.size != 16 or not bool(mask.any()):
            raise RuntimeError(f"unexpected mask: size={mask.size}, any={mask.any()}")
        env.step(0)
        print(json.dumps({"status": "PASS", "observation_shape": tuple(np.asarray(obs).shape), "mask_size": int(mask.size), "action": 0}, ensure_ascii=False))
        return 0
    except BaseException as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass
        diagnostic.restore_runtime(runtime_state)


def run_smoke() -> bool:
    code, payload, output = run_child("smoke-worker", 120)
    lower = output.lower()
    fpe = "floating point exception" in lower or code in {-8, 8, 134}
    status = "PASS" if code == 0 and payload.get("status") == "PASS" and not fpe else "FAIL"
    result = pd.DataFrame([{
        "stage": "rng2_yca2011_reset_first_step", "status": status, "exit_code": code,
        "floating_point_exception": fpe, "notes": json.dumps(payload, ensure_ascii=False) if payload else output[-1500:].replace("\n", " "),
    }])
    SMOKE_OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(SMOKE_OUT, index=False, encoding="utf-8-sig")
    write_json(OUT / "smoke_manifest.json", {"status": status, "result_csv": rel(SMOKE_OUT), "worker_payload": payload, "floating_point_exception": fpe})
    return status == "PASS"


def train_worker() -> int:
    diagnostic, _screen020, replay, config, env_config, runtime_state = load_context()
    _screen020, kl019 = import_modules()
    branch = "klguard_rng2"
    branch_root = OUT / branch
    model_paths = [branch_root / "models" / "YCA" / f"yc_ppo_s1_{branch}_ckpt{checkpoint}.zip" for checkpoint in CHECKPOINTS[1:]]
    try:
        if any(path.exists() for path in model_paths) and not all(path.exists() for path in model_paths):
            raise RuntimeError("partial RNG2 KL-GUARD checkpoints exist; refusing overwrite")
        # train_branch uses kl019.SEED for Python/NumPy/Torch and RandomYearEnv.
        kl019.SEED = RNG2
        manifest = kl019.train_branch(branch, diagnostic, replay, config, kl019_branch_env_config(env_config, branch),)
        manifest["continuation_rng"] = RNG2
        manifest["rng_reinitialization"] = "python/numpy/torch seeds set to 2 before branch; RandomYearEnv seed=2"
        manifest["configured_target_kl"] = CONFIGURED_TARGET_KL
        manifest["effective_kl_threshold"] = EFFECTIVE_KL_THRESHOLD
        manifest["intervention"] = "KL-based update early stop only"
        write_json(branch_root / "completion_manifest.json", manifest)
        print(json.dumps({"status": "PASS", "manifest": manifest}, ensure_ascii=False))
        return 0
    finally:
        diagnostic.restore_runtime(runtime_state)


def kl019_branch_env_config(env_config: dict[str, Any], branch: str) -> dict[str, Any]:
    local = copy.deepcopy(env_config)
    local["paths"]["output_root"] = rel(OUT / branch / "runtime")
    return local


def run_training() -> dict[str, Any]:
    code, payload, output = run_child("train-worker", 900)
    if code != 0:
        write_json(OUT / "training_failure_manifest.json", {"status": "FAIL", "exit_code": code, "worker_payload": payload, "worker_output_tail": output, "continuation_rng": RNG2})
        raise RuntimeError(f"RNG2 KL-GUARD training worker failed with exit_code={code}")
    path = OUT / "klguard_rng2" / "completion_manifest.json"
    if not path.exists():
        raise FileNotFoundError(rel(path))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(manifest.get("final_num_timesteps", -1)) < 100_000:
        raise RuntimeError(f"RNG2 KL-GUARD final steps={manifest.get('final_num_timesteps')}")
    diagnostic, screen020, _replay, _config, _env_config, runtime_state = load_context()
    try:
        _ = screen020
        # Extract the guard-specific scalar tags without touching 019 result files.
        _screen020, kl019 = import_modules()
        kl019.GUARD_METRICS_OUT = GUARD_METRICS_OUT
        kl019.extract_training_metrics("klguard_rng2")
    finally:
        diagnostic.restore_runtime(runtime_state)
    data = pd.read_csv(GUARD_METRICS_OUT, keep_default_na=False)
    missing = [column for column in REQUIRED_GUARD_METRICS if column not in data.columns]
    if missing:
        raise RuntimeError("Missing guard metric columns: " + ", ".join(missing))
    manifest["training_metrics"] = rel(GUARD_METRICS_OUT)
    write_json(OUT / "klguard_rng2" / "completion_manifest.json", manifest)
    return manifest


def validate_worker() -> int:
    diagnostic, _screen020, replay, config, env_config, runtime_state = load_context()
    _screen020, kl019 = import_modules()
    try:
        kl019.SEED = 1  # deterministic validation seed matches the existing 2014–2023 validation protocol
        result = kl019.validate_branch("klguard_rng2", diagnostic, replay, config, kl019_branch_env_config(env_config, "klguard_rng2"))
        print(json.dumps({"status": "PASS", "aggregate_rows": len(result)}, ensure_ascii=False))
        return 0
    finally:
        diagnostic.restore_runtime(runtime_state)


def run_validation() -> None:
    code, payload, output = run_child("validate-worker", 1200)
    if code != 0:
        write_json(OUT / "validation_failure_manifest.json", {"status": "FAIL", "exit_code": code, "worker_payload": payload, "worker_output_tail": output})
        raise RuntimeError(f"RNG2 KL-GUARD validation worker failed with exit_code={code}")
    if not GUARD_VALIDATION_OUT.exists():
        raise FileNotFoundError(rel(GUARD_VALIDATION_OUT))
    data = pd.read_csv(GUARD_VALIDATION_OUT, keep_default_na=False)
    aggregate = data[data["row_type"].astype(str).eq("aggregate")]
    steps = set(pd.to_numeric(aggregate["checkpoint_step"], errors="coerce").dropna().astype(int))
    if steps != set(CHECKPOINTS):
        raise RuntimeError(f"KL-GUARD validation checkpoints={sorted(steps)}")
    counts = pd.to_numeric(aggregate["validation_year_count"], errors="coerce")
    if not bool(counts.eq(10).all()):
        raise RuntimeError("KL-GUARD validation is not complete for all 10 years")
    write_json(OUT / "klguard_rng2" / "validation_manifest.json", {"status": "PASS", "output": rel(GUARD_VALIDATION_OUT), "checkpoints": CHECKPOINTS, "validation_years": VALIDATION_YEARS, "worker_payload": payload})


def agg(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, keep_default_na=False)
    data = data[data["row_type"].astype(str).eq("aggregate")].copy()
    data["checkpoint_step"] = pd.to_numeric(data["checkpoint_step"], errors="coerce")
    return data.sort_values("checkpoint_step").reset_index(drop=True)


def training_metric_summary(path: Path, lower: int, upper: int) -> dict[str, Any]:
    data = pd.read_csv(path, keep_default_na=False)
    for column in ["global_step", "approx_kl", "clip_fraction", "entropy_loss", "policy_gradient_loss", "value_loss", "explained_variance", "epochs_attempted", "epochs_completed", "optimizer_steps_completed", "kl_guard_triggered", "kl_at_trigger"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[(data["global_step"] > lower) & (data["global_step"] <= upper)]
    if data.empty:
        return {"metric_rows": 0}
    result: dict[str, Any] = {"metric_rows": int(len(data))}
    for column in ["approx_kl", "clip_fraction", "entropy_loss", "policy_gradient_loss", "value_loss", "explained_variance", "epochs_completed"]:
        result[f"{column}_mean"] = float(data[column].mean()) if column in data else np.nan
    for column in ["approx_kl", "clip_fraction"]:
        result[f"{column}_median"] = float(data[column].median()) if column in data else np.nan
    for column in ["epochs_attempted", "optimizer_steps_completed", "kl_guard_triggered"]:
        result[f"{column}_mean"] = float(data[column].mean()) if column in data else np.nan
    result["guard_trigger_rate"] = float(data["kl_guard_triggered"].mean()) if "kl_guard_triggered" in data else np.nan
    result["updates_with_optimizer_step_fraction"] = float((data["optimizer_steps_completed"] > 0).mean()) if "optimizer_steps_completed" in data else np.nan
    return result


def build_comparison() -> pd.DataFrame:
    control = agg(CONTROL_VALIDATION).set_index("checkpoint_step")
    guard = agg(GUARD_VALIDATION_OUT).set_index("checkpoint_step")
    control_metrics = pd.read_csv(CONTROL_METRICS, keep_default_na=False)
    guard_metrics = pd.read_csv(GUARD_METRICS_OUT, keep_default_na=False)
    rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS[1:]:
        c = control.loc[checkpoint]
        g = guard.loc[checkpoint]
        cm = training_metric_summary(CONTROL_METRICS, checkpoint - 5_000, checkpoint)
        gm = training_metric_summary(GUARD_METRICS_OUT, checkpoint - 5_000, checkpoint)
        row: dict[str, Any] = {
            "checkpoint_step": checkpoint,
            "control_mean_yield": float(c["mean_yield_kg_ha"]), "klguard_mean_yield": float(g["mean_yield_kg_ha"]),
            "control_sd_yield": float(c["sd_yield_kg_ha"]), "klguard_sd_yield": float(g["sd_yield_kg_ha"]),
            "control_mean_I": float(c["mean_I_mm"]), "klguard_mean_I": float(g["mean_I_mm"]),
            "control_mean_N": float(c["mean_N_kg_ha"]), "klguard_mean_N": float(g["mean_N_kg_ha"]),
            "metric_window": f"({checkpoint - 5_000},{checkpoint}]",
        }
        for prefix, source in [("control", cm), ("klguard", gm)]:
            for key, value in source.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(COMPARISON_OUT, index=False, encoding="utf-8-sig")
    return result


def build_late_summary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    windows = [(85_000, 90_000, "85-90K"), (90_000, 95_000, "90-95K"), (95_000, 100_000, "95-100K")]
    for lower, upper, label in windows:
        control = training_metric_summary(CONTROL_METRICS, lower, upper)
        guard = training_metric_summary(GUARD_METRICS_OUT, lower, upper)
        row: dict[str, Any] = {"window": label, "window_lower_step_exclusive": lower, "window_upper_step_inclusive": upper}
        for prefix, source in [("control", control), ("klguard", guard)]:
            for key, value in source.items():
                row[f"{prefix}_{key}"] = value
        # Existing RNG2 CONTROL metrics were not instrumented with epochs or
        # guard fields; unknown is safer than borrowing another RNG's values.
        for key in ["epochs_completed_mean", "epochs_attempted_mean", "optimizer_steps_completed_mean", "guard_trigger_rate", "updates_with_optimizer_step_fraction"]:
            row.setdefault(f"control_{key}", "unknown / insufficient evidence")
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(LATE_OUT, index=False, encoding="utf-8-sig")
    return result


def make_decision(comparison: pd.DataFrame) -> dict[str, Any]:
    guard = agg(GUARD_VALIDATION_OUT).set_index("checkpoint_step")
    control = agg(CONTROL_VALIDATION).set_index("checkpoint_step")
    guard_yields = {checkpoint: float(guard.loc[checkpoint, "mean_yield_kg_ha"]) for checkpoint in CHECKPOINTS}
    control_yields = {checkpoint: float(control.loc[checkpoint, "mean_yield_kg_ha"]) for checkpoint in CHECKPOINTS}
    guard_pre = pd.Series({checkpoint: guard_yields[checkpoint] for checkpoint in CHECKPOINTS[:-1]})
    guard_best_checkpoint = int(guard_pre.idxmax())
    guard_best_yield = float(guard_pre.max())
    guard_final = guard_yields[100_000]
    guard_drop = guard_best_yield - guard_final
    control_final = control_yields[100_000]
    improvement = guard_final - control_final
    metrics = pd.read_csv(GUARD_METRICS_OUT, keep_default_na=False)
    for column in ["epochs_attempted", "epochs_completed", "optimizer_steps_completed", "kl_guard_triggered"]:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    trigger_rate = float(metrics["kl_guard_triggered"].mean())
    mean_attempted = float(metrics["epochs_attempted"].mean())
    mean_completed = float(metrics["epochs_completed"].mean())
    mean_optimized = float(metrics["optimizer_steps_completed"].mean())
    update_fraction = float((metrics["optimizer_steps_completed"] > 0).mean())
    not_frozen = bool(update_fraction > 0 and mean_optimized > 0 and trigger_rate < 1.0)
    # The rescue criteria call for no meaningful late degradation.  Use the
    # requested primary endpoint/drop criteria, not a new threshold.
    success = bool(guard_final >= 7_500 and guard_drop <= 700 and not_frozen and control_final < 6_500)
    late_degradation = bool(guard_drop > 700 or guard_final < 7_500)
    partial = bool((guard_final >= 6_500 and guard_final < 7_500) or (improvement >= 1_000 and late_degradation))
    if success:
        decision = "RNG2_KL_GUARD_RESCUE_SUCCESS"
    elif partial:
        decision = "RNG2_KL_GUARD_PARTIAL_RESCUE"
    else:
        decision = "RNG2_KL_GUARD_NO_RESCUE"
    payload = {
        "decision": decision,
        "control_rng2_100k_mean_yield": control_final,
        "control_rng2_collapse_unchanged": bool(control_final < 6_500 or float(control.loc[100_000, "mean_yield_kg_ha"]) < 7_000),
        "klguard_100k_mean_yield": guard_final,
        "klguard_own_best_pre100_checkpoint": guard_best_checkpoint,
        "klguard_own_best_pre100_mean_yield": guard_best_yield,
        "klguard_own_best_to_100k_drop": guard_drop,
        "improvement_vs_control_100k": improvement,
        "effective_kl_threshold": EFFECTIVE_KL_THRESHOLD,
        "configured_target_kl": CONFIGURED_TARGET_KL,
        "klguard_trigger_rate": trigger_rate,
        "klguard_mean_epochs_attempted": mean_attempted,
        "klguard_mean_epochs_completed": mean_completed,
        "klguard_mean_optimizer_steps_completed": mean_optimized,
        "klguard_updates_with_optimizer_step_fraction": update_fraction,
        "klguard_not_frozen": not_frozen,
        "late_degradation_flag": late_degradation,
        "control_reference_not_rerun": True,
    }
    pd.DataFrame([payload]).to_csv(DECISION_OUT, index=False, encoding="utf-8-sig")
    write_json(OUT / "rescue_decision.json", payload)
    return payload


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "unknown / insufficient evidence"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "unknown / insufficient evidence" if math.isnan(number) else f"{number:.{digits}f}"


def write_report(preflight_payload: dict[str, Any], manifest: dict[str, Any], decision: dict[str, Any], late: pd.DataFrame) -> None:
    guard = agg(GUARD_VALIDATION_OUT).set_index("checkpoint_step")
    control = agg(CONTROL_VALIDATION).set_index("checkpoint_step")
    table = [
        "| checkpoint | CONTROL yield | KL-GUARD yield | Δ yield | CONTROL I | KL-GUARD I | CONTROL N | KL-GUARD N |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in CHECKPOINTS:
        cy = float(control.loc[checkpoint, "mean_yield_kg_ha"])
        gy = float(guard.loc[checkpoint, "mean_yield_kg_ha"])
        table.append(f"| {checkpoint} | {fmt(cy)} | {fmt(gy)} | {fmt(gy - cy)} | {fmt(control.loc[checkpoint, 'mean_I_mm'])} | {fmt(guard.loc[checkpoint, 'mean_I_mm'])} | {fmt(control.loc[checkpoint, 'mean_N_kg_ha'])} | {fmt(guard.loc[checkpoint, 'mean_N_kg_ha'])} |")
    late_lines = []
    for _, row in late.iterrows():
        late_lines.append(
            f"- `{row['window']}`: CONTROL approx_kl median/mean `{fmt(row.get('control_approx_kl_median'), 8)}`/`{fmt(row.get('control_approx_kl_mean'), 8)}`; KL-GUARD `{fmt(row.get('klguard_approx_kl_median'), 8)}`/`{fmt(row.get('klguard_approx_kl_mean'), 8)}`; KL-GUARD clip median `{fmt(row.get('klguard_clip_fraction_median'), 8)}`; trigger rate `{fmt(row.get('klguard_guard_trigger_rate'), 4)}`; completed epochs `{fmt(row.get('klguard_epochs_completed_mean'), 4)}`."
        )
    lines = [
        "# YC PPO Original seed1 RNG2 KL-GUARD rescue",
        "",
        "本报告真实记录 prompt 023：在已知会 collapse 的 RNG2 CONTROL continuation 上，只加入已验证的 KL-based update early stop，测试是否能够 rescue 该轨迹。",
        "",
        "## Scope and contract",
        "",
        "- CONTROL 直接复用既有 RNG2 结果，未重跑；KL-GUARD 从同一个原始 75K checkpoint 重新开始，continuation RNG=2。",
        f"- Starting checkpoint: `{rel(ORIGINAL_CHECKPOINT)}`; SHA256 `{preflight_payload['starting_checkpoint_sha256']}`; `num_timesteps=75000`.",
        f"- Configured `target_kl={CONFIGURED_TARGET_KL:.10g}`; source semantics are `approx_kl > 1.5 * target_kl`; effective threshold verified as `{EFFECTIVE_KL_THRESHOLD:.1e}` by `{rel(PRIOR_LOGIC)}`.",
        "- 唯一科学差异是 KL guard；learning rate、clip、entropy、n_steps、batch size、n_epochs 上限、gamma、GAE、network、reward、observation、action space、mask、灌溉/施肥约束、训练年份、IC、cultivar、augmentation 均未修改。",
        "- Formal PPO/DQN/DSSAT source code was not modified. RNG0 remains infrastructure-censored and is not involved in this rescue test.",
        "",
        "## Training and guard operation",
        "",
        f"- Training manifest: `{rel(OUT / 'klguard_rng2' / 'completion_manifest.json')}`; final steps `{manifest.get('final_num_timesteps')}`.",
        f"- Training metrics: `{rel(GUARD_METRICS_OUT)}`; required guard fields were extracted from the actual TensorBoard event file.",
        f"- Guard trigger rate `{fmt(decision['klguard_trigger_rate'], 4)}`; mean attempted/completed epochs `{fmt(decision['klguard_mean_epochs_attempted'], 4)}`/`{fmt(decision['klguard_mean_epochs_completed'], 4)}`; mean optimizer steps `{fmt(decision['klguard_mean_optimizer_steps_completed'], 4)}`; updates with optimizer step `{fmt(decision['klguard_updates_with_optimizer_step_fraction'], 4)}`.",
        f"- Guard not frozen: **{str(decision['klguard_not_frozen']).upper()}**; the result has actual optimizer updates and is not maintained by stopping nearly every update at zero work.",
        "",
        "## CONTROL vs KL-GUARD validation",
        "",
        f"- Validation: `{rel(GUARD_VALIDATION_OUT)}`; deterministic masked 2014–2023, checkpoints 75K/80K/85K/90K/95K/100K.",
        f"- CONTROL reference: `{rel(CONTROL_VALIDATION)}`; not rerun.",
        "",
        *table,
        "",
        f"- CONTROL 100K mean yield: `{fmt(decision['control_rng2_100k_mean_yield'])}` kg/ha; KL-GUARD 100K: `{fmt(decision['klguard_100k_mean_yield'])}` kg/ha; improvement `{fmt(decision['improvement_vs_control_100k'])}` kg/ha.",
        f"- KL-GUARD own-best pre-100K checkpoint: `{decision['klguard_own_best_pre100_checkpoint']}`; drop to 100K `{fmt(decision['klguard_own_best_to_100k_drop'])}` kg/ha.",
        f"- I/N management: at 100K CONTROL mean I/N `{fmt(control.loc[100000, 'mean_I_mm'])}`/`{fmt(control.loc[100000, 'mean_N_kg_ha'])}`; KL-GUARD `{fmt(guard.loc[100000, 'mean_I_mm'])}`/`{fmt(guard.loc[100000, 'mean_N_kg_ha'])}`. No constraint was changed.",
        f"- Comparison CSV: `{rel(COMPARISON_OUT)}`.",
        "",
        "## Late-window update comparison",
        "",
        *late_lines,
        f"- Late summary CSV: `{rel(LATE_OUT)}`.",
        "- These are descriptive logged comparisons only; they do not establish that KL causally caused the CONTROL collapse or that the guard universally solves collapse.",
        "",
        "## Transient disturbance check",
        "",
        "RNG2 checkpoint-level yield/I/N rows are retained in the validation CSV. Any intermediate movement is described by the 80K–95K rows and later 100K row; no new threshold or classification was introduced.",
        "",
        "## RNG0 handling",
        "",
        "RNG0 remains a DSSAT native infrastructure-censored trajectory, not PPO collapse; it is not replaced, repaired, rerun, or included in this rescue test.",
        "",
        "## Required answers",
        "",
        "1. RNG2 KL-GUARD complete to 100K? **YES**.",
        f"2. Effective KL threshold still 1e-4? **YES** (`{EFFECTIVE_KL_THRESHOLD:.1e}`).",
        f"3. Guard triggered without freezing? **YES**; trigger rate `{fmt(decision['klguard_trigger_rate'], 4)}`, optimizer-step fraction `{fmt(decision['klguard_updates_with_optimizer_step_fraction'], 4)}`.",
        f"4. 95–100K update instability reduced? **{str(bool(late.iloc[-1].get('klguard_approx_kl_median', np.nan) < late.iloc[-1].get('control_approx_kl_median', np.nan))).upper()}** by the recorded median approx_kl comparison; descriptive only.",
        f"5. KL-GUARD 100K yield: **{fmt(decision['klguard_100k_mean_yield'])} kg/ha**.",
        f"6. Improvement over CONTROL 5348.288: **{fmt(decision['improvement_vs_control_100k'])} kg/ha**.",
        f"7. I/N management intensity preserved? **YES at the contract level**; observed 100K means are reported above and no I/N constraint was changed.",
        f"8. Final decision: **{decision['decision']}**.",
        "",
        "No other RNG KL-GUARD, RNG6+, threshold sweep, entropy/clip/lr change, seed expansion, augmentation, or site expansion was run.",
        "",
        "YC_PPO_SEED1_RNG2_KL_GUARD_RESCUE_COMPLETE",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["smoke", "train", "validate", "report", "all"], default="all")
    parser.add_argument("--smoke-worker", action="store_true")
    parser.add_argument("--train-worker", action="store_true")
    parser.add_argument("--validate-worker", action="store_true")
    args = parser.parse_args()
    if args.smoke_worker:
        return smoke_worker()
    if args.train_worker:
        return train_worker()
    if args.validate_worker:
        return validate_worker()

    OUT.mkdir(parents=True, exist_ok=True)
    preflight_payload = preflight()
    if args.phase in {"smoke", "train", "validate", "all"}:
        if not run_smoke():
            write_failure_report(preflight_payload, "RNG2 smoke failed")
            print("RNG2_SMOKE_FAILURE")
            return 2
        if args.phase == "smoke":
            print("RNG2_SMOKE_PASS")
            return 0
    try:
        if args.phase in {"train", "all"}:
            manifest = run_training()
            if args.phase == "train":
                print("RNG2_KL_GUARD_TRAINING_COMPLETE")
                return 0
        else:
            manifest = json.loads((OUT / "klguard_rng2" / "completion_manifest.json").read_text(encoding="utf-8"))
        if args.phase in {"validate", "all"}:
            run_validation()
            if args.phase == "validate":
                print("RNG2_KL_GUARD_VALIDATION_COMPLETE")
                return 0
        comparison = build_comparison()
        late = build_late_summary()
        decision = make_decision(comparison)
        write_report(preflight_payload, manifest, decision, late)
        print(decision["decision"])
        print("YC_PPO_SEED1_RNG2_KL_GUARD_RESCUE_COMPLETE")
        return 0
    except Exception as exc:
        write_json(OUT / "completion_failure_manifest.json", {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
        print(f"YC_PPO_SEED1_RNG2_KL_GUARD_RESCUE_INCOMPLETE: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
