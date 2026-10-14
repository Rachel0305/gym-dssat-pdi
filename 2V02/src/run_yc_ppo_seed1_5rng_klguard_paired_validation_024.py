"""024: YC Original seed1 5-RNG fixed KL-guard paired validation.

This orchestration file reuses the audited 019 MaskablePPO KL-guard
instrumentation and the 017/020 YC runtime.  It adds no PPO/DSSAT source
logic.  RNG2 is reused from 023; only RNG1/3/4/5 are newly continued.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import zipfile
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
CONFIG_PATH = ORIGINAL_RUN / "configs" / "222YCA_O1_yc_v1_original_seed1.json"
OUT = ROOT / "benchmark_results" / "024_yc_ppo_seed1_5rng_klguard_paired_validation"
RESULTS = ROOT / "results"
REPORT = ROOT / "docs" / "yc_ppo_original_seed1_5rng_klguard_paired_validation.md"

NEW_RNGS = [1, 3, 4, 5]
ALL_RNGS = [1, 2, 3, 4, 5]
STATION = "YCA"
SITE = "YC"
TRAIN_YEARS = list(range(2004, 2014))
VALIDATION_YEARS = list(range(2014, 2024))
CHECKPOINTS = [75_000, 80_000, 85_000, 90_000, 95_000, 100_000]
CONTINUATION_STEPS = 25_000
CONFIGURED_TARGET_KL = 6.666666666666667e-05
EFFECTIVE_KL_THRESHOLD = 1.0e-04

RNG2_GUARD_METRICS = RESULTS / "yc_ppo_s1_rng2_klguard_training_metrics.csv"
RNG2_GUARD_VALIDATION = RESULTS / "yc_ppo_s1_rng2_klguard_checkpoint_validation.csv"

CONTROL_SOURCES = {
    1: (RESULTS / "yc_ppo_s1_control_training_metrics.csv", RESULTS / "yc_ppo_s1_control_checkpoint_validation.csv"),
    2: (RESULTS / "yc_ppo_s1_control_rng2_training_metrics.csv", RESULTS / "yc_ppo_s1_control_rng2_checkpoint_validation.csv"),
    3: (RESULTS / "yc_ppo_s1_control_rng3_training_metrics.csv", RESULTS / "yc_ppo_s1_control_rng3_checkpoint_validation.csv"),
    4: (RESULTS / "yc_ppo_s1_control_rng4_training_metrics.csv", RESULTS / "yc_ppo_s1_control_rng4_checkpoint_validation.csv"),
    5: (RESULTS / "yc_ppo_s1_control_rng5_training_metrics.csv", RESULTS / "yc_ppo_s1_control_rng5_checkpoint_validation.csv"),
}

GUARD_REQUIRED_COLUMNS = [
    "continuation_rng", "global_step", "approx_kl", "clip_fraction", "entropy_loss",
    "policy_gradient_loss", "value_loss", "explained_variance", "learning_rate",
    "clip_range", "epochs_attempted", "epochs_completed", "optimizer_steps_completed",
    "kl_guard_triggered", "kl_at_trigger",
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_data(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("data"))


def guard_metrics_path(rng: int) -> Path:
    return RESULTS / f"yc_ppo_s1_rng{int(rng)}_klguard_training_metrics.csv"


def guard_validation_path(rng: int) -> Path:
    return RESULTS / f"yc_ppo_s1_rng{int(rng)}_klguard_checkpoint_validation.csv"


def branch_name(rng: int) -> str:
    return f"klguard_rng{int(rng)}"


def branch_dir(rng: int) -> Path:
    return OUT / branch_name(rng)


def branch_checkpoint_path(rng: int, checkpoint: int) -> Path:
    if int(checkpoint) == 75_000:
        return ORIGINAL_CHECKPOINT
    return branch_dir(rng) / "models" / "YCA" / f"yc_ppo_s1_klguard_rng{int(rng)}_ckpt{int(checkpoint)}.zip"


def import_modules(rng: int) -> tuple[Any, Any]:
    import run_yc_ppo_seed1_multirng_control_020 as screen020
    import run_yc_ppo_seed1_kl_guard_paired_019 as kl019

    # Redirect orchestration outputs only.  The imported PPO/DSSAT
    # implementation and scientific configuration are not edited.
    screen020.OUT = OUT
    screen020.REPORT = REPORT
    screen020.RESULTS = RESULTS
    kl019.OUT = OUT
    kl019.REPORT = REPORT
    kl019.RESULTS = RESULTS
    kl019.SEED = int(rng)
    kl019.GUARD_METRICS_OUT = guard_metrics_path(rng)
    kl019.GUARD_VALIDATION_OUT = guard_validation_path(rng)
    return screen020, kl019


def load_context(rng: int) -> tuple[Any, Any, Any, Any, dict[str, Any], Any, Any]:
    import diagnose_yc_ppo_original_seed1_collapse_017 as diagnostic

    screen020, kl019 = import_modules(rng)
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    replay, config, env_config, runtime_state = screen020.isolated_runtime_setup(cfg, diagnostic)
    return diagnostic, screen020, kl019, replay, config, env_config, runtime_state


def preflight() -> dict[str, Any]:
    required = [ORIGINAL_CHECKPOINT, CONFIG_PATH]
    for rng in ALL_RNGS:
        required.extend(CONTROL_SOURCES[rng])
    required.extend([RNG2_GUARD_METRICS, RNG2_GUARD_VALIDATION])
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prior evidence: " + ", ".join(missing))

    source = checkpoint_data(ORIGINAL_CHECKPOINT)
    if int(source.get("num_timesteps", -1)) != 75_000:
        raise RuntimeError(f"Common checkpoint num_timesteps={source.get('num_timesteps')}, expected 75000")

    logic_path = RESULTS / "yc_ppo_s1_kl_guard_logic_test.csv"
    if not logic_path.exists():
        raise FileNotFoundError(rel(logic_path))
    logic = pd.read_csv(logic_path, keep_default_na=False)
    logic_ok = bool(logic["pass_fail"].astype(str).eq("PASS").any())
    threshold_ok = bool(
        np.isclose(pd.to_numeric(logic["configured_target_kl"], errors="coerce"), CONFIGURED_TARGET_KL, rtol=1e-9, atol=1e-12).any()
        and np.isclose(pd.to_numeric(logic["effective_threshold"], errors="coerce"), EFFECTIVE_KL_THRESHOLD, rtol=1e-9, atol=1e-12).any()
    )
    if not logic_ok or not threshold_ok:
        raise RuntimeError("Existing KL-guard logic evidence does not verify the fixed threshold")

    source_checks: dict[str, Any] = {}
    for rng in ALL_RNGS:
        metrics_path, validation_path = CONTROL_SOURCES[rng]
        validation = pd.read_csv(validation_path, keep_default_na=False)
        aggregate = validation[validation["row_type"].astype(str).eq("aggregate")].copy()
        steps = set(pd.to_numeric(aggregate["checkpoint_step"], errors="coerce").dropna().astype(int))
        counts = pd.to_numeric(aggregate["validation_year_count"], errors="coerce")
        if steps != set(CHECKPOINTS) or not bool(counts.eq(10).all()):
            raise RuntimeError(f"CONTROL RNG{rng} validation incomplete: checkpoints={sorted(steps)} counts={counts.tolist()}")
        metrics = pd.read_csv(metrics_path, keep_default_na=False)
        if metrics.empty or int(pd.to_numeric(metrics["global_step"], errors="coerce").max()) < 100_000:
            raise RuntimeError(f"CONTROL RNG{rng} metrics do not reach 100K")
        source_checks[str(rng)] = {
            "training_metrics": rel(metrics_path),
            "validation": rel(validation_path),
            "training_rows": int(len(metrics)),
            "validation_aggregate_rows": int(len(aggregate)),
        }

    rng2_guard = pd.read_csv(RNG2_GUARD_VALIDATION, keep_default_na=False)
    rng2_agg = rng2_guard[rng2_guard["row_type"].astype(str).eq("aggregate")]
    rng2_steps = set(pd.to_numeric(rng2_agg["checkpoint_step"], errors="coerce").dropna().astype(int))
    if rng2_steps != set(CHECKPOINTS) or not bool(pd.to_numeric(rng2_agg["validation_year_count"], errors="coerce").eq(10).all()):
        raise RuntimeError("023 RNG2 KL-GUARD validation is incomplete")
    rng2_metrics = pd.read_csv(RNG2_GUARD_METRICS, keep_default_na=False)
    if rng2_metrics.empty or int(pd.to_numeric(rng2_metrics["global_step"], errors="coerce").max()) < 100_000:
        raise RuntimeError("023 RNG2 KL-GUARD metrics do not reach 100K")

    partial = []
    for rng in NEW_RNGS:
        paths = [branch_checkpoint_path(rng, cp) for cp in CHECKPOINTS[1:]]
        present = [path.exists() for path in paths]
        manifest = branch_dir(rng) / "training_manifest.json"
        if any(present) and not (all(present) and manifest.exists()):
            partial.append({"rng": rng, "paths": [rel(path) for path, exists in zip(paths, present) if exists]})
    if partial:
        raise RuntimeError("Partial 024 checkpoints exist; refusing overwrite: " + json.dumps(partial))

    payload = {
        "status": "PASS",
        "prompt": rel(ROOT / "prompt_01" / "024_codex_prompt_024_yc_ppo_seed1_5rng_klguard_paired_validation.md"),
        "common_starting_checkpoint": rel(ORIGINAL_CHECKPOINT),
        "common_starting_checkpoint_sha256": sha256_file(ORIGINAL_CHECKPOINT),
        "common_starting_num_timesteps": source.get("num_timesteps"),
        "new_rngs": NEW_RNGS,
        "reused_rng": 2,
        "configured_target_kl": CONFIGURED_TARGET_KL,
        "effective_kl_threshold": EFFECTIVE_KL_THRESHOLD,
        "threshold_logic_evidence": rel(logic_path),
        "control_sources": source_checks,
        "rng2_guard_metrics": rel(RNG2_GUARD_METRICS),
        "rng2_guard_validation": rel(RNG2_GUARD_VALIDATION),
        "rng0_status": "INFRASTRUCTURE_CENSORED; excluded; not rerun",
        "scientific_contract_change": "Only fixed KL-based update early stop in new guard branches",
        "formal_ppo_dqn_dssat_code_modified": False,
    }
    write_json(OUT / "preflight.json", payload)
    return payload


def branch_env_config(env_config: dict[str, Any], rng: int, suffix: str = "runtime") -> dict[str, Any]:
    local = copy.deepcopy(env_config)
    local["paths"]["output_root"] = rel(branch_dir(rng) / suffix)
    return local


def smoke_worker(rng: int) -> int:
    diagnostic, _screen020, _kl019, replay, config, env_config, runtime_state = load_context(rng)
    import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base
    from sb3_contrib.common.maskable.utils import get_action_masks

    env = None
    try:
        local = branch_env_config(env_config, rng, "smoke_runtime")
        env = base.make_env(config, local, STATION, 2011, int(rng), f"024_klguard_rng{rng}_smoke", evaluation=False)
        obs, _info = env.reset()
        mask = np.asarray(get_action_masks(env), dtype=bool)
        if mask.size != 16 or not bool(mask.any()):
            raise RuntimeError(f"unexpected mask size/availability: {mask.size}/{mask.any()}")
        env.step(0)
        payload = {"status": "PASS", "rng": rng, "observation_shape": list(np.asarray(obs).shape), "mask_size": int(mask.size)}
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    except BaseException as exc:
        print(json.dumps({"status": "FAIL", "rng": rng, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass
        diagnostic.restore_runtime(runtime_state)


def normalize_guard_metrics(path: Path, rng: int, branch: str) -> pd.DataFrame:
    data = pd.read_csv(path, keep_default_na=False)
    if "continuation_rng" not in data.columns:
        data.insert(0, "continuation_rng", int(rng))
    else:
        data["continuation_rng"] = int(rng)
    if "branch" not in data.columns:
        data.insert(1, "branch", branch)
    for column in GUARD_REQUIRED_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan
    ordered = ["continuation_rng", "branch"] + [c for c in GUARD_REQUIRED_COLUMNS if c != "continuation_rng"]
    extras = [c for c in data.columns if c not in ordered]
    data = data[ordered + extras]
    data.to_csv(path, index=False, encoding="utf-8-sig")
    missing = [c for c in GUARD_REQUIRED_COLUMNS if c not in data.columns or data[c].isna().all()]
    if missing:
        raise RuntimeError(f"RNG{rng} guard metrics missing/empty: {missing}")
    if int(pd.to_numeric(data["global_step"], errors="coerce").max()) < 100_000:
        raise RuntimeError(f"RNG{rng} guard metrics do not reach 100K")
    return data


def train_worker(rng: int) -> int:
    diagnostic, _screen020, kl019, replay, config, env_config, runtime_state = load_context(rng)
    branch = branch_name(rng)
    try:
        kl019.SEED = int(rng)
        manifest = kl019.train_branch(branch, diagnostic, replay, config, branch_env_config(env_config, rng))
        # Extract metrics from the real TensorBoard event file after training
        # (or after discovering a complete prior continuation).  This is
        # post-processing and does not rerun or alter the continuation.
        kl019.extract_training_metrics(branch)
        metrics = normalize_guard_metrics(guard_metrics_path(rng), rng, branch)
        manifest.update({
            "continuation_rng": int(rng),
            "configured_target_kl": CONFIGURED_TARGET_KL,
            "effective_kl_threshold": EFFECTIVE_KL_THRESHOLD,
            "intervention": "fixed KL-based update early stop only",
            "training_metrics": rel(guard_metrics_path(rng)),
            "training_metric_rows": int(len(metrics)),
            "rng_reinitialization": f"python/numpy/torch seeds set to {rng}; RandomYearEnv seed={rng}",
        })
        write_json(branch_dir(rng) / "completion_manifest.json", manifest)
        print(json.dumps({"status": "PASS", "rng": rng, "final_num_timesteps": manifest.get("final_num_timesteps")}, ensure_ascii=False))
        return 0
    except BaseException as exc:
        write_json(branch_dir(rng) / "training_failure_manifest.json", {"status": "FAIL", "rng": rng, "error": f"{type(exc).__name__}: {exc}"})
        print(json.dumps({"status": "FAIL", "rng": rng, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    finally:
        diagnostic.restore_runtime(runtime_state)


def normalize_guard_validation(path: Path, rng: int, branch: str) -> pd.DataFrame:
    data = pd.read_csv(path, keep_default_na=False)
    if "continuation_rng" not in data.columns:
        data.insert(0, "continuation_rng", int(rng))
    else:
        data["continuation_rng"] = int(rng)
    if "validation_source" not in data.columns:
        data["validation_source"] = "024_new_deterministic_masked_2014_2023"
    data["branch"] = branch
    data.to_csv(path, index=False, encoding="utf-8-sig")
    aggregate = data[data["row_type"].astype(str).eq("aggregate")]
    steps = set(pd.to_numeric(aggregate["checkpoint_step"], errors="coerce").dropna().astype(int))
    if steps != set(CHECKPOINTS):
        raise RuntimeError(f"RNG{rng} guard validation checkpoints={sorted(steps)}")
    if not bool(pd.to_numeric(aggregate["validation_year_count"], errors="coerce").eq(10).all()):
        raise RuntimeError(f"RNG{rng} guard validation is not complete for all ten years")
    return data


def validate_worker(rng: int) -> int:
    diagnostic, _screen020, kl019, replay, config, env_config, runtime_state = load_context(rng)
    branch = branch_name(rng)
    try:
        # Validation protocol is deterministic and uses seed=1 for every branch.
        kl019.SEED = 1
        result = kl019.validate_branch(branch, diagnostic, replay, config, branch_env_config(env_config, rng))
        data = normalize_guard_validation(guard_validation_path(rng), rng, branch)
        write_json(branch_dir(rng) / "validation_manifest.json", {
            "status": "PASS", "continuation_rng": int(rng), "validation_seed": 1,
            "validation_years": VALIDATION_YEARS, "checkpoints": CHECKPOINTS,
            "output": rel(guard_validation_path(rng)), "aggregate_rows": int(len(result)),
            "rows": int(len(data)),
        })
        print(json.dumps({"status": "PASS", "rng": rng, "rows": len(data)}, ensure_ascii=False))
        return 0
    except BaseException as exc:
        write_json(branch_dir(rng) / "validation_failure_manifest.json", {"status": "FAIL", "rng": rng, "error": f"{type(exc).__name__}: {exc}"})
        print(json.dumps({"status": "FAIL", "rng": rng, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    finally:
        diagnostic.restore_runtime(runtime_state)


def run_child(worker: str, rng: int, timeout_seconds: int) -> tuple[int, str]:
    command = [sys.executable, str(Path(__file__).resolve()), f"--{worker}", "--rng", str(int(rng))]
    process = subprocess.Popen(
        command, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except BaseException:
            process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except BaseException:
                process.kill()
            stdout, stderr = process.communicate()
        return 124, ((stdout or "") + "\n" + (stderr or ""))[-12000:]
    return int(process.returncode), ((stdout or "") + "\n" + (stderr or ""))[-12000:]


def run_smoke(rng: int) -> None:
    code, output = run_child("smoke-worker", rng, 180)
    result = pd.DataFrame([{"rng": rng, "status": "PASS" if code == 0 else "FAIL", "exit_code": code, "output_tail": output}])
    result.to_csv(OUT / f"smoke_rng{rng}.csv", index=False, encoding="utf-8-sig")
    if code != 0:
        raise RuntimeError(f"RNG{rng} smoke failed: {output[-2000:]}")


def run_training(rng: int) -> None:
    manifest_path = branch_dir(rng) / "completion_manifest.json"
    expected = [branch_checkpoint_path(rng, cp) for cp in CHECKPOINTS[1:]]
    if manifest_path.exists() and all(path.exists() for path in expected) and guard_metrics_path(rng).exists():
        return
    code, output = run_child("train-worker", rng, 1800)
    if code != 0:
        raise RuntimeError(f"RNG{rng} KL-GUARD training failed: {output[-4000:]}")
    if not manifest_path.exists() or not all(path.exists() for path in expected):
        raise RuntimeError(f"RNG{rng} training did not produce all checkpoints/manifest")


def run_validation(rng: int) -> None:
    output = guard_validation_path(rng)
    if output.exists():
        data = pd.read_csv(output, keep_default_na=False)
        aggregate = data[data["row_type"].astype(str).eq("aggregate")]
        steps = set(pd.to_numeric(aggregate["checkpoint_step"], errors="coerce").dropna().astype(int))
        if steps == set(CHECKPOINTS) and bool(pd.to_numeric(aggregate["validation_year_count"], errors="coerce").eq(10).all()):
            return
    code, output_tail = run_child("validate-worker", rng, 3000)
    if code != 0:
        raise RuntimeError(f"RNG{rng} KL-GUARD validation failed: {output_tail[-4000:]}")
    if not guard_validation_path(rng).exists():
        raise RuntimeError(f"RNG{rng} validation CSV missing after worker completion")


def read_aggregates(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, keep_default_na=False)
    data = data[data["row_type"].astype(str).eq("aggregate")].copy()
    for column in ["checkpoint_step", "mean_yield_kg_ha", "sd_yield_kg_ha", "mean_I_mm", "mean_N_kg_ha", "mean_PFP_N", "mean_WP_ET", "invalid_count_total", "fallback_count_total"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.sort_values("checkpoint_step").reset_index(drop=True)


def read_metrics(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, keep_default_na=False)
    for column in ["global_step", "approx_kl", "clip_fraction", "entropy_loss", "policy_gradient_loss", "value_loss", "explained_variance", "learning_rate", "clip_range", "epochs_attempted", "epochs_completed", "optimizer_steps_completed", "kl_guard_triggered", "kl_at_trigger"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def window_summary(path: Path, lower: int, upper: int) -> dict[str, Any]:
    data = read_metrics(path)
    data = data[(data["global_step"] > lower) & (data["global_step"] <= upper)]
    result: dict[str, Any] = {"metric_rows": int(len(data))}
    for column in ["approx_kl", "clip_fraction", "entropy_loss", "policy_gradient_loss", "value_loss", "explained_variance", "epochs_completed"]:
        result[f"{column}_mean"] = float(data[column].mean()) if column in data and not data[column].dropna().empty else "unknown / insufficient evidence"
    for column in ["approx_kl", "clip_fraction"]:
        result[f"{column}_median"] = float(data[column].median()) if column in data and not data[column].dropna().empty else "unknown / insufficient evidence"
    for column in ["epochs_attempted", "optimizer_steps_completed", "kl_guard_triggered"]:
        result[f"{column}_mean"] = float(data[column].mean()) if column in data and not data[column].dropna().empty else "unknown / insufficient evidence"
    if "kl_guard_triggered" in data and not data["kl_guard_triggered"].dropna().empty:
        result["guard_trigger_rate"] = float(data["kl_guard_triggered"].mean())
    else:
        result["guard_trigger_rate"] = "unknown / insufficient evidence"
    if "optimizer_steps_completed" in data and not data["optimizer_steps_completed"].dropna().empty:
        result["updates_with_optimizer_step_fraction"] = float((data["optimizer_steps_completed"] > 0).mean())
    else:
        result["updates_with_optimizer_step_fraction"] = "unknown / insufficient evidence"
    return result


def collapse_flag(aggregates: pd.DataFrame) -> tuple[bool, int, float, float]:
    indexed = aggregates.set_index("checkpoint_step")
    final_yield = float(indexed.loc[100_000, "mean_yield_kg_ha"])
    pre = indexed.loc[[75_000, 80_000, 85_000, 90_000, 95_000], "mean_yield_kg_ha"]
    best_checkpoint = int(pre.idxmax())
    best_yield = float(pre.max())
    drop = best_yield - final_yield
    collapsed = bool(final_yield < 6500.0 or drop > 1500.0)
    return collapsed, best_checkpoint, best_yield, drop


def build_pairwise() -> pd.DataFrame:
    rows = []
    for rng in ALL_RNGS:
        control = read_aggregates(CONTROL_SOURCES[rng][1]).set_index("checkpoint_step")
        guard_path = RNG2_GUARD_VALIDATION if rng == 2 else guard_validation_path(rng)
        guard = read_aggregates(guard_path).set_index("checkpoint_step")
        c_collapsed, c_best_cp, c_best, c_drop = collapse_flag(control.reset_index())
        g_collapsed, g_best_cp, g_best, g_drop = collapse_flag(guard.reset_index())
        c100 = control.loc[100_000]
        g100 = guard.loc[100_000]
        material_harm = False if rng == 2 else bool(
            g_collapsed
            or float(g100["mean_yield_kg_ha"]) < 0.95 * float(c100["mean_yield_kg_ha"])
            or g_drop > 700.0
        )
        rows.append({
            "rng": rng,
            "control_yield_100K": float(c100["mean_yield_kg_ha"]),
            "guard_yield_100K": float(g100["mean_yield_kg_ha"]),
            "delta_yield": float(g100["mean_yield_kg_ha"] - c100["mean_yield_kg_ha"]),
            "control_best_pre100": float(c_best),
            "guard_best_pre100": float(g_best),
            "control_best_pre100_checkpoint": c_best_cp,
            "guard_best_pre100_checkpoint": g_best_cp,
            "control_drop_to_100K": float(c_drop),
            "guard_drop_to_100K": float(g_drop),
            "control_I_100K": float(c100["mean_I_mm"]),
            "guard_I_100K": float(g100["mean_I_mm"]),
            "control_N_100K": float(c100["mean_N_kg_ha"]),
            "guard_N_100K": float(g100["mean_N_kg_ha"]),
            "control_collapse": c_collapsed,
            "guard_collapse": g_collapsed,
            "material_harm": material_harm,
            "rng2_status": "REUSED_023" if rng == 2 else "NEW_024",
        })
    result = pd.DataFrame(rows).sort_values("rng")
    result.to_csv(RESULTS / "yc_ppo_s1_5rng_control_vs_klguard_paired.csv", index=False, encoding="utf-8-sig")
    return result


def build_late_summary() -> pd.DataFrame:
    rows = []
    windows = [(85_000, 90_000, "85-90K"), (90_000, 95_000, "90-95K"), (95_000, 100_000, "95-100K")]
    for rng in ALL_RNGS:
        guard_path = RNG2_GUARD_METRICS if rng == 2 else guard_metrics_path(rng)
        control_path = CONTROL_SOURCES[rng][0]
        for lower, upper, label in windows:
            c = window_summary(control_path, lower, upper)
            g = window_summary(guard_path, lower, upper)
            row: dict[str, Any] = {"rng": rng, "window": label, "window_lower_step_exclusive": lower, "window_upper_step_inclusive": upper}
            for prefix, values in [("control", c), ("klguard", g)]:
                for key, value in values.items():
                    row[f"{prefix}_{key}"] = value
            for key in ["approx_kl_mean", "approx_kl_median", "clip_fraction_mean", "clip_fraction_median"]:
                c_value = c.get(key)
                g_value = g.get(key)
                row[f"klguard_minus_control_{key}"] = float(g_value - c_value) if isinstance(c_value, (int, float)) and isinstance(g_value, (int, float)) else "unknown / insufficient evidence"
            rows.append(row)
    result = pd.DataFrame(rows).sort_values(["rng", "window_lower_step_exclusive"])
    result.to_csv(RESULTS / "yc_ppo_s1_5rng_control_vs_klguard_late_update_summary.csv", index=False, encoding="utf-8-sig")
    return result


def optimizer_summary(rng: int) -> dict[str, Any]:
    path = RNG2_GUARD_METRICS if rng == 2 else guard_metrics_path(rng)
    data = read_metrics(path)
    optimized = data["optimizer_steps_completed"]
    completed = data["epochs_completed"]
    attempted = data["epochs_attempted"]
    triggered = data["kl_guard_triggered"]
    return {
        "trigger_rate": float(triggered.mean()),
        "mean_epochs_attempted": float(attempted.mean()),
        "mean_epochs_completed": float(completed.mean()),
        "mean_optimizer_steps_completed": float(optimized.mean()),
        "updates_with_optimizer_step_fraction": float((optimized > 0).mean()),
        "not_frozen": bool(len(data) > 0 and float(optimized.mean()) > 0.0 and float((optimized > 0).mean()) > 0.0),
        "metric_rows": int(len(data)),
    }


def build_group_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    control_yields = pairwise["control_yield_100K"].astype(float)
    guard_yields = pairwise["guard_yield_100K"].astype(float)
    stable = pairwise[pairwise["rng"].isin([1, 3, 4, 5])]
    rows = [
        {"group": "CONTROL", "collapse_count": int(pairwise["control_collapse"].sum()), "collapse_rate": float(pairwise["control_collapse"].mean()), "yield_mean": float(control_yields.mean()), "yield_sd": float(control_yields.std(ddof=1)), "yield_min": float(control_yields.min()), "yield_max": float(control_yields.max())},
        {"group": "KL-GUARD", "collapse_count": int(pairwise["guard_collapse"].sum()), "collapse_rate": float(pairwise["guard_collapse"].mean()), "yield_mean": float(guard_yields.mean()), "yield_sd": float(guard_yields.std(ddof=1)), "yield_min": float(guard_yields.min()), "yield_max": float(guard_yields.max())},
        {"group": "PAIRED", "mean_paired_delta_yield": float(pairwise["delta_yield"].mean()), "median_paired_delta_yield": float(pairwise["delta_yield"].median()), "minimum_paired_delta_yield": float(pairwise["delta_yield"].min()), "positive_delta_count": int((pairwise["delta_yield"] > 0).sum()), "material_harm_count_stable_rngs": int(stable["material_harm"].sum()), "rng2_guard_collapse": bool(pairwise.loc[pairwise["rng"].eq(2), "guard_collapse"].iloc[0]), "rng2_control_collapse": bool(pairwise.loc[pairwise["rng"].eq(2), "control_collapse"].iloc[0])},
    ]
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "yc_ppo_s1_5rng_klguard_summary.csv", index=False, encoding="utf-8-sig")
    return result


def build_decision(pairwise: pd.DataFrame) -> dict[str, Any]:
    optimizer = {str(rng): optimizer_summary(rng) for rng in ALL_RNGS}
    stable_harm_count = int(pairwise.loc[pairwise["rng"].isin([1, 3, 4, 5]), "material_harm"].sum())
    guard_collapse_count = int(pairwise["guard_collapse"].sum())
    control_collapse_count = int(pairwise["control_collapse"].sum())
    rng2_guard_collapse = bool(pairwise.loc[pairwise["rng"].eq(2), "guard_collapse"].iloc[0])
    all_not_frozen = bool(all(item["not_frozen"] for item in optimizer.values()))
    if guard_collapse_count < control_collapse_count and not rng2_guard_collapse and stable_harm_count == 0 and all_not_frozen:
        decision = "KL_GUARD_MULTIRNG_PROMISING"
    elif stable_harm_count >= 2:
        decision = "KL_GUARD_HARMS_STABLE_TRAJECTORIES"
    else:
        decision = "KL_GUARD_MULTIRNG_INSUFFICIENT"
    payload = {
        "decision": decision,
        "control_collapse_count": control_collapse_count,
        "control_collapse_rate": control_collapse_count / 5.0,
        "klguard_collapse_count": guard_collapse_count,
        "klguard_collapse_rate": guard_collapse_count / 5.0,
        "rng2_control_collapse": bool(pairwise.loc[pairwise["rng"].eq(2), "control_collapse"].iloc[0]),
        "rng2_guard_collapse": rng2_guard_collapse,
        "rng2_guard_status": "non-collapse / partial degradation" if not rng2_guard_collapse else "full-collapse",
        "stable_rng_material_harm_count": stable_harm_count,
        "all_guard_branches_not_frozen": all_not_frozen,
        "optimizer_summary": optimizer,
        "rng0_status": "INFRASTRUCTURE_CENSORED; excluded from n=5 denominator",
        "n": 5,
        "note": "Descriptive n=5 continuation set; no significance test and no population-level collapse-rate claim.",
    }
    pd.DataFrame([payload]).to_csv(RESULTS / "yc_ppo_s1_5rng_klguard_decision.csv", index=False, encoding="utf-8-sig")
    write_json(OUT / "decision.json", payload)
    return payload


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or isinstance(value, str):
        return "unknown / insufficient evidence" if value is None else value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "unknown / insufficient evidence" if math.isnan(number) else f"{number:.{digits}f}"


def write_report(decision: dict[str, Any], pairwise: pd.DataFrame, late: pd.DataFrame, group: pd.DataFrame) -> None:
    rows = [
        "# YC PPO Original seed1 5-RNG fixed KL-GUARD paired validation",
        "",
        "## Decision",
        "",
        f"Primary decision: **{decision['decision']}**.",
        "",
        "This is a descriptive n=5 continuation-set validation. RNG0 remains infrastructure-censored and is excluded; no significance test or population-level collapse-rate claim is made.",
        "",
        "## Contract and provenance",
        "",
        f"- Common 75K checkpoint: `{rel(ORIGINAL_CHECKPOINT)}`; SHA256 `{sha256_file(ORIGINAL_CHECKPOINT)}`; `num_timesteps=75000`.",
        f"- New KL-GUARD continuations: RNG1, RNG3, RNG4, RNG5; each requested 25,000 environment steps and checkpoints 80K/85K/90K/95K/100K. RNG2 is reused from 023 and was not rerun.",
        f"- Fixed configured `target_kl={CONFIGURED_TARGET_KL:.10g}`; source semantics give effective threshold `{EFFECTIVE_KL_THRESHOLD:.1e}`. No threshold sweep or hyperparameter tuning was performed.",
        "- The only scientific intervention in new branches was the fixed KL-based update early stop. Observation, reward, action space, mask logic, irrigation/fertilizer constraints, training years, IC, cultivar, augmentation, network, learning rate, clip range, n_steps, batch size, n_epochs, gamma, and GAE were not changed.",
        "- PPO/DQN/DSSAT source logic was not modified. Validation is deterministic masked 2014–2023.",
        "",
        "## Completion and group summary",
        "",
        f"- All four new KL-GUARD branches reached 100K: **YES**.",
        f"- CONTROL collapse count/rate: `{decision['control_collapse_count']}/5` (`{decision['control_collapse_rate']:.3f}`).",
        f"- KL-GUARD collapse count/rate: `{decision['klguard_collapse_count']}/5` (`{decision['klguard_collapse_rate']:.3f}`).",
        f"- RNG2: CONTROL full-collapse=`{decision['rng2_control_collapse']}`; KL-GUARD full-collapse=`{decision['rng2_guard_collapse']}`; status=`{decision['rng2_guard_status']}`.",
        f"- RNG1/3/4/5 material-harm count: `{decision['stable_rng_material_harm_count']}`.",
        f"- All KL-GUARD branches had real optimizer updates/non-freezing: **{str(decision['all_guard_branches_not_frozen']).upper()}**.",
        "",
        "### Paired 100K yields",
        "",
        "| RNG | CONTROL 100K | KL-GUARD 100K | Δ yield | CONTROL collapse | KL-GUARD collapse | material harm |",
        "|---:|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for _, row in pairwise.sort_values("rng").iterrows():
        rows.append(f"| {int(row['rng'])} | {fmt(row['control_yield_100K'])} | {fmt(row['guard_yield_100K'])} | {fmt(row['delta_yield'])} | {row['control_collapse']} | {row['guard_collapse']} | {row['material_harm']} |")
    rows.extend([
        "",
        f"- Pairwise mean/median/min Δyield: `{fmt(pairwise['delta_yield'].mean())}` / `{fmt(pairwise['delta_yield'].median())}` / `{fmt(pairwise['delta_yield'].min())}` kg/ha.",
        "",
        "## Training evidence",
        "",
    ])
    for rng in ALL_RNGS:
        metrics_path = RNG2_GUARD_METRICS if rng == 2 else guard_metrics_path(rng)
        manifest_path = OUT / "023_reused_rng2" / "manifest.json" if rng == 2 else branch_dir(rng) / "completion_manifest.json"
        opt = decision["optimizer_summary"][str(rng)]
        rows.append(f"- RNG{rng}: metrics `{rel(metrics_path)}`; trigger rate `{fmt(opt['trigger_rate'], 4)}`; mean completed epochs `{fmt(opt['mean_epochs_completed'], 4)}`; mean optimizer steps `{fmt(opt['mean_optimizer_steps_completed'], 4)}`; optimizer-step fraction `{fmt(opt['updates_with_optimizer_step_fraction'], 4)}`; not frozen=`{opt['not_frozen']}`.")
    rows.extend([
        "",
        "## Preserved engineering attempt",
        "",
        f"- RNG1 的第一次 024 编排 attempt 已保留在 `{rel(branch_dir(1) / 'training_failure_manifest.json')}`。训练本体与 5 个 checkpoint 已完成；失败发生在 post-training TensorBoard metrics 抽取调用遗漏，未发生 PPO/DSSAT 逻辑失败。修复后仅补做 metrics 抽取，未重跑 RNG1 训练，也未覆盖 checkpoint。",
        "",
        "## Checkpoint validation evidence",
        "",
    ])
    for rng in ALL_RNGS:
        validation_path = RNG2_GUARD_VALIDATION if rng == 2 else guard_validation_path(rng)
        label = "reused from 023" if rng == 2 else "new 024 validation"
        rows.append(f"- RNG{rng} ({label}): `{rel(validation_path)}`; checkpoints 75K/80K/85K/90K/95K/100K; 10 validation years per checkpoint.")
    rows.extend([
        "",
        "## Late-window update comparison",
        "",
        "The following rows are descriptive. They compare the recorded CONTROL and KL-GUARD update metrics for 85–90K, 90–95K and 95–100K; missing CONTROL instrumentation is reported as `unknown / insufficient evidence`.",
        "",
        f"- Late summary: `{rel(RESULTS / 'yc_ppo_s1_5rng_control_vs_klguard_late_update_summary.csv')}`.",
    ])
    for label, group_window in late.groupby("window", sort=False):
        valid = group_window[pd.to_numeric(group_window["control_approx_kl_mean"], errors="coerce").notna() & pd.to_numeric(group_window["klguard_approx_kl_mean"], errors="coerce").notna()]
        if len(valid):
            cmean = pd.to_numeric(valid["control_approx_kl_mean"], errors="coerce").mean()
            gmean = pd.to_numeric(valid["klguard_approx_kl_mean"], errors="coerce").mean()
            cmed = pd.to_numeric(valid["control_approx_kl_median"], errors="coerce").mean()
            gmed = pd.to_numeric(valid["klguard_approx_kl_median"], errors="coerce").mean()
            rows.append(f"- `{label}`: available-RNG mean of per-RNG means CONTROL/KL-GUARD approx_kl `{cmean:.8f}`/`{gmean:.8f}`; mean of per-RNG medians `{cmed:.8f}`/`{gmed:.8f}`; guard lower mean=`{gmean < cmean}`, guard lower median=`{gmed < cmed}`.")
    rows.extend([
        "- Interpretation: guard appears to lower late-window update magnitude only where the paired descriptive means/medians are lower; this does not establish KL causality for collapse.",
        "- RNG2’s 95–100K high-KL behavior is compared in the same file against the reused 023 metrics. RNG5’s transient disturbance is compared descriptively; no causal claim is made.",
        "",
        "## Output files",
        "",
        f"- New training metrics: `{', '.join(rel(guard_metrics_path(rng)) for rng in NEW_RNGS)}`.",
        f"- New checkpoint validation: `{', '.join(rel(guard_validation_path(rng)) for rng in NEW_RNGS)}`.",
        f"- Paired comparison: `{rel(RESULTS / 'yc_ppo_s1_5rng_control_vs_klguard_paired.csv')}`.",
        f"- Late summary: `{rel(RESULTS / 'yc_ppo_s1_5rng_control_vs_klguard_late_update_summary.csv')}`.",
        f"- Group summary: `{rel(RESULTS / 'yc_ppo_s1_5rng_klguard_summary.csv')}`.",
        f"- Decision: `{rel(RESULTS / 'yc_ppo_s1_5rng_klguard_decision.csv')}`.",
        "",
        "## Stop boundary",
        "",
        "No RNG6+, threshold change, PPO hyperparameter change, seed expansion, augmentation, site expansion, or new intervention was run.",
        "",
        "YC_PPO_SEED1_5RNG_KL_GUARD_VALIDATION_COMPLETE",
    ])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["preflight", "smoke", "train", "validate", "report", "all"], default="all")
    parser.add_argument("--rng", type=int, choices=NEW_RNGS)
    parser.add_argument("--smoke-worker", action="store_true")
    parser.add_argument("--train-worker", action="store_true")
    parser.add_argument("--validate-worker", action="store_true")
    args = parser.parse_args()
    if args.rng is None and (args.smoke_worker or args.train_worker or args.validate_worker):
        raise SystemExit("--rng is required for worker mode")
    if args.smoke_worker:
        return smoke_worker(args.rng)
    if args.train_worker:
        return train_worker(args.rng)
    if args.validate_worker:
        return validate_worker(args.rng)

    OUT.mkdir(parents=True, exist_ok=True)
    try:
        payload = preflight()
        if args.phase == "preflight":
            print("YC_PPO_SEED1_5RNG_KL_GUARD_PREFLIGHT_PASS")
            return 0
        if args.phase in {"smoke", "train", "validate", "all"}:
            for rng in NEW_RNGS:
                run_smoke(rng)
        if args.phase == "smoke":
            print("YC_PPO_SEED1_5RNG_KL_GUARD_SMOKE_PASS")
            return 0
        if args.phase in {"train", "all"}:
            for rng in NEW_RNGS:
                run_training(rng)
        if args.phase == "train":
            print("YC_PPO_SEED1_5RNG_KL_GUARD_TRAINING_COMPLETE")
            return 0
        if args.phase in {"validate", "all"}:
            for rng in NEW_RNGS:
                run_validation(rng)
        if args.phase == "validate":
            print("YC_PPO_SEED1_5RNG_KL_GUARD_VALIDATION_COMPLETE")
            return 0
        pairwise = build_pairwise()
        late = build_late_summary()
        group = build_group_summary(pairwise)
        decision = build_decision(pairwise)
        write_report(decision, pairwise, late, group)
        print(decision["decision"])
        print("YC_PPO_SEED1_5RNG_KL_GUARD_VALIDATION_COMPLETE")
        return 0
    except BaseException as exc:
        write_json(OUT / "completion_failure_manifest.json", {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
        print(f"YC_PPO_SEED1_5RNG_KL_GUARD_VALIDATION_INCOMPLETE: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
