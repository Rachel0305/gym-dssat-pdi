"""022: YC PPO Original seed1 RNG5 CONTROL completion.

Only the prompt-authorized RNG5 continuation is new here.  RNG0 is kept as
infrastructure-censored and is never used as a starting point or denominator.
The PPO/DSSAT runtime is imported from the audited 017/020 implementation;
this file only orchestrates isolated execution and evidence aggregation.
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
PRIOR_TRAINING = ROOT / "results" / "yc_ppo_s1_control_training_metrics.csv"
PRIOR_VALIDATION = ROOT / "results" / "yc_ppo_s1_control_checkpoint_validation.csv"
SCREEN020_OUT = ROOT / "benchmark_results" / "020_yc_ppo_seed1_multirng_control_screen"

OUT = ROOT / "benchmark_results" / "022_yc_ppo_seed1_rng5_control_completion"
RESULTS = ROOT / "results"
SMOKE_OUT = OUT / "rng5_smoke.csv"
TRAINING_OUT = RESULTS / "yc_ppo_s1_control_rng5_training_metrics.csv"
VALIDATION_OUT = RESULTS / "yc_ppo_s1_control_rng5_checkpoint_validation.csv"
SUMMARY_OUT = RESULTS / "yc_ppo_s1_control_5complete_reproducibility.csv"
LATE_OUT = RESULTS / "yc_ppo_s1_control_5complete_late_update_summary.csv"
REPORT = ROOT / "docs" / "yc_ppo_original_seed1_rng5_control_completion.md"

STATION = "YCA"
SITE = "YC"
RNG5 = 5
COMPLETE_RNGS = [1, 2, 3, 4, 5]
TRAIN_YEARS = list(range(2004, 2014))
VALIDATION_YEARS = list(range(2014, 2024))
CHECKPOINTS = [75_000, 80_000, 85_000, 90_000, 95_000, 100_000]
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

    screen020.OUT = OUT
    screen020.REPORT = REPORT
    screen020.RESULTS = RESULTS
    return screen020


def load_context() -> tuple[Any, Any, Any, Any, dict[str, Any], Any]:
    import diagnose_yc_ppo_original_seed1_collapse_017 as diagnostic

    screen020 = import_screen020()
    cfg = json.loads((ORIGINAL_RUN / "configs" / "222YCA_O1_yc_v1_original_seed1.json").read_text(encoding="utf-8"))
    replay, config, env_config, runtime_state = screen020.isolated_runtime_setup(cfg, diagnostic)
    return diagnostic, screen020, replay, config, env_config, runtime_state


def preflight() -> dict[str, Any]:
    required = [ORIGINAL_CHECKPOINT, PRIOR_RESUME_AUDIT, PRIOR_TRAINING, PRIOR_VALIDATION]
    required.extend(
        RESULTS / f"yc_ppo_s1_control_rng{rng}_checkpoint_validation.csv"
        for rng in [2, 3, 4]
    )
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required complete-control evidence: " + ", ".join(missing))
    source = checkpoint_data(ORIGINAL_CHECKPOINT)
    if int(source.get("num_timesteps", -1)) != 75_000:
        raise RuntimeError(f"Original checkpoint num_timesteps={source.get('num_timesteps')}, expected 75000")
    control_manifest = PRIOR_018B_CONTROL / "training_manifest.json"
    if not control_manifest.exists():
        raise FileNotFoundError(rel(control_manifest))
    manifest = json.loads(control_manifest.read_text(encoding="utf-8"))
    if manifest.get("target_kl") is not None:
        raise RuntimeError("018B CONTROL target_kl is not None")
    old_failed = sorted(rel(path) for path in (SCREEN020_OUT / "failed_attempts").rglob("*") if path.is_file())
    payload = {
        "status": "PASS",
        "prompt": rel(ROOT / "prompt_01" / "022_codex_prompt_022_yc_ppo_seed1_rng5_control_completion.md"),
        "starting_checkpoint": rel(ORIGINAL_CHECKPOINT),
        "starting_checkpoint_sha256": sha256_file(ORIGINAL_CHECKPOINT),
        "starting_num_timesteps": source.get("num_timesteps"),
        "continuation_rng": RNG5,
        "complete_denominator_rngs": COMPLETE_RNGS,
        "rng0_status": "INFRASTRUCTURE_CENSORED",
        "prior_resume_audit": rel(PRIOR_RESUME_AUDIT),
        "prior_018b_control_manifest": rel(control_manifest),
        "prior_018b_target_kl": manifest.get("target_kl"),
        "prior_complete_validation": {str(rng): rel(PRIOR_VALIDATION if rng == 1 else RESULTS / f"yc_ppo_s1_control_rng{rng}_checkpoint_validation.csv") for rng in COMPLETE_RNGS if rng != 5},
        "preserved_rng0_failure_evidence": old_failed or ["unknown / insufficient evidence"],
        "scientific_contract_change": False,
        "formal_ppo_dqn_code_modified": False,
    }
    write_json(OUT / "preflight.json", payload)
    return payload


def _parse_json_line(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def run_child(worker: str, timeout_seconds: int) -> tuple[int, dict[str, Any], str]:
    """Run a worker in its own process group so DSSAT FPE/hangs are bounded."""
    command = [sys.executable, str(Path(__file__).resolve()), f"--{worker}"]
    process = subprocess.Popen(
        command, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
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
        text = (stdout or "") + "\n" + (stderr or "")
        return 124, {"status": "TIMEOUT", "timeout_seconds": timeout_seconds}, text[-8000:]
    output = (stdout or "") + "\n" + (stderr or "")
    return int(process.returncode), _parse_json_line(output), output[-8000:]


def smoke_worker() -> int:
    diagnostic, _screen020, replay, config, env_config, runtime_state = load_context()
    import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base

    local = copy.deepcopy(env_config)
    local["paths"]["output_root"] = rel(OUT / "smoke_runtime")
    env = None
    try:
        env = base.make_env(config, local, STATION, 2011, RNG5, "022_rng5_yca2011_smoke", evaluation=False)
        obs, _info = env.reset()
        from sb3_contrib.common.maskable.utils import get_action_masks

        mask = np.asarray(get_action_masks(env), dtype=bool)
        if mask.size != 16 or not bool(mask.any()):
            raise RuntimeError(f"unexpected action mask size/validity: size={mask.size}, any={mask.any()}")
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
    notes = json.dumps(payload, ensure_ascii=False) if payload else output[-1500:].replace("\n", " ")
    result = pd.DataFrame([{
        "stage": "rng5_yca2011_reset_first_step", "status": status, "exit_code": code,
        "floating_point_exception": bool(fpe), "notes": notes,
    }])
    SMOKE_OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(SMOKE_OUT, index=False, encoding="utf-8-sig")
    write_json(OUT / "smoke_manifest.json", {
        "status": status, "result_csv": rel(SMOKE_OUT), "worker_payload": payload,
        "floating_point_exception": bool(fpe), "contract_change": False,
    })
    return status == "PASS"


def train_worker() -> int:
    diagnostic, screen020, replay, config, env_config, runtime_state = load_context()
    try:
        manifest = screen020.train_rng(RNG5, replay, config, env_config)
        print(json.dumps({"status": "PASS", "manifest": manifest}, ensure_ascii=False))
        return 0
    finally:
        diagnostic.restore_runtime(runtime_state)


def run_training() -> dict[str, Any]:
    code, payload, output = run_child("train-worker", 900)
    if code != 0:
        write_json(OUT / "training_failure_manifest.json", {
            "status": "FAIL", "exit_code": code, "worker_payload": payload,
            "worker_output_tail": output, "starting_checkpoint": rel(ORIGINAL_CHECKPOINT),
            "continuation_rng": RNG5, "contract_change": False,
        })
        raise RuntimeError(f"RNG5 training worker failed with exit_code={code}")
    manifest_path = OUT / "isolated" / "rng5" / "training_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(rel(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("final_num_timesteps", -1)) < 100_000:
        raise RuntimeError(f"RNG5 final_num_timesteps={manifest.get('final_num_timesteps')}, expected >=100000")
    metrics = import_training_metrics()
    manifest["metrics_csv"] = rel(TRAINING_OUT)
    manifest["smoke_csv"] = rel(SMOKE_OUT)
    write_json(OUT / "isolated" / "rng5" / "completion_manifest.json", manifest)
    return manifest


def import_training_metrics() -> pd.DataFrame:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = sorted((OUT / "isolated" / "rng5" / "tensorboard").glob("**/events*"))
    if len(event_files) != 1:
        raise RuntimeError(f"Expected exactly one RNG5 event file, got {len(event_files)}")
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
            records.setdefault(step, {"continuation_rng": RNG5, "global_step": step, "timestep": step})[name] = float(event.value)
    metrics = pd.DataFrame(sorted(records.values(), key=lambda row: row["global_step"]))
    for column in REQUIRED_METRICS:
        if column not in metrics.columns:
            metrics[column] = np.nan
    metrics = metrics[["continuation_rng", *REQUIRED_METRICS, "timestep"]]
    metrics["trajectory_status"] = "COMPLETE"
    metrics["source"] = rel(event_files[0])
    TRAINING_OUT.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TRAINING_OUT, index=False, encoding="utf-8-sig")
    write_json(OUT / "isolated" / "rng5" / "tensorboard_manifest.json", {
        "event_file": rel(event_files[0]), "event_file_sha256": sha256_file(event_files[0]),
        "rows": len(metrics), "required_columns": REQUIRED_METRICS,
        "missing_required_columns": [column for column in REQUIRED_METRICS if metrics[column].isna().all()],
    })
    if metrics.empty:
        raise RuntimeError("RNG5 training metrics are empty")
    return metrics


def validate_worker() -> int:
    diagnostic, screen020, replay, config, env_config, runtime_state = load_context()
    try:
        aggregate = screen020.validate_rng(RNG5, diagnostic, replay, config, env_config)
        print(json.dumps({"status": "PASS", "aggregate_rows": len(aggregate)}, ensure_ascii=False))
        return 0
    finally:
        diagnostic.restore_runtime(runtime_state)


def run_validation() -> None:
    code, payload, output = run_child("validate-worker", 1200)
    if code != 0:
        write_json(OUT / "validation_failure_manifest.json", {
            "status": "FAIL", "exit_code": code, "worker_payload": payload,
            "worker_output_tail": output,
        })
        raise RuntimeError(f"RNG5 validation worker failed with exit_code={code}")
    if not VALIDATION_OUT.exists():
        raise FileNotFoundError(rel(VALIDATION_OUT))
    data = pd.read_csv(VALIDATION_OUT, keep_default_na=False)
    aggregate = data[data["row_type"].astype(str).eq("aggregate")]
    checkpoints = set(pd.to_numeric(aggregate["checkpoint_step"], errors="coerce").dropna().astype(int))
    if checkpoints != set(CHECKPOINTS):
        raise RuntimeError(f"RNG5 validation checkpoint set={sorted(checkpoints)}, expected={CHECKPOINTS}")
    if not bool(pd.to_numeric(aggregate["validation_year_count"], errors="coerce").eq(10).all()):
        raise RuntimeError("RNG5 validation does not contain ten years at every checkpoint")
    write_json(OUT / "isolated" / "rng5" / "completion_validation_manifest.json", {
        "status": "PASS", "output": rel(VALIDATION_OUT), "checkpoints": CHECKPOINTS,
        "validation_years": VALIDATION_YEARS, "worker_payload": payload,
    })


def read_aggregate(path: Path, rng: int, source: str) -> pd.DataFrame:
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


def build_complete_summary() -> tuple[pd.DataFrame, dict[str, Any]]:
    sources = {
        1: (PRIOR_VALIDATION, "018B_rng1_control_reference"),
        2: (RESULTS / "yc_ppo_s1_control_rng2_checkpoint_validation.csv", "020_rng2_validation"),
        3: (RESULTS / "yc_ppo_s1_control_rng3_checkpoint_validation.csv", "020_rng3_validation"),
        4: (RESULTS / "yc_ppo_s1_control_rng4_checkpoint_validation.csv", "020_rng4_validation"),
        5: (VALIDATION_OUT, "022_rng5_validation"),
    }
    rows: list[dict[str, Any]] = []
    for rng in COMPLETE_RNGS:
        path, source = sources[rng]
        data = read_aggregate(path, rng, source).set_index("checkpoint_step")
        yields = {checkpoint: float(data.loc[checkpoint, "mean_yield_kg_ha"]) for checkpoint in CHECKPOINTS}
        pre = pd.Series({checkpoint: yields[checkpoint] for checkpoint in CHECKPOINTS[:-1]})
        best_checkpoint = int(pre.idxmax())
        best_yield = float(pre.max())
        final_yield = yields[100_000]
        drop = best_yield - final_yield
        a = final_yield < 6_500.0
        b = drop > 1_500.0
        rows.append({
            "continuation_rng": rng,
            **{f"yield_{checkpoint // 1000}K": yields[checkpoint] for checkpoint in CHECKPOINTS},
            "best_pre100_checkpoint": best_checkpoint, "best_pre100_yield": best_yield,
            "drop_best_to_100K": drop, "collapse": bool(a or b),
            "collapse_reason": ";".join(reason for reason, active in [
                ("100K_mean_yield<6500", a), ("best_pre100_minus_100K>1500", b)
            ] if active) or "none",
            "validation_source": source,
        })
    summary = pd.DataFrame(rows).sort_values("continuation_rng").reset_index(drop=True)
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")
    final = pd.to_numeric(summary["yield_100K"], errors="coerce")
    drops = pd.to_numeric(summary["drop_best_to_100K"], errors="coerce")
    collapse_count = int(summary["collapse"].astype(bool).sum())
    decision = "COLLAPSE_REPRODUCIBLE" if collapse_count >= 2 else "COLLAPSE_SPARSE_1_OF_5"
    stats = {
        "denominator": 5, "denominator_rngs": COMPLETE_RNGS, "rng0_excluded_as": "INFRASTRUCTURE_CENSORED",
        "collapse_count": collapse_count, "collapse_rate": collapse_count / 5,
        "yield_100K_mean": float(final.mean()), "yield_100K_sd": float(final.std(ddof=1)),
        "yield_100K_min": float(final.min()), "yield_100K_max": float(final.max()),
        "decline_gt_1000_count": int((drops > 1_000).sum()), "decline_gt_1500_count": int((drops > 1_500).sum()),
        "decision": decision, "summary_csv": rel(SUMMARY_OUT),
    }
    write_json(OUT / "5complete_reproducibility_summary.json", stats)
    return summary, stats


def metric_path(rng: int) -> tuple[Path, str]:
    if rng == 1:
        return PRIOR_TRAINING, "018B_rng1_control_reference"
    if rng == 5:
        return TRAINING_OUT, "022_rng5_training"
    return RESULTS / f"yc_ppo_s1_control_rng{rng}_training_metrics.csv", f"020_rng{rng}_training"


def build_late_summary(summary: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    metrics: dict[int, pd.DataFrame] = {}
    for rng in COMPLETE_RNGS:
        path, source = metric_path(rng)
        data = pd.read_csv(path, keep_default_na=False)
        data["global_step"] = pd.to_numeric(data["global_step"], errors="coerce")
        for column in ["approx_kl", "clip_fraction"]:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data["metrics_source"] = source
        metrics[rng] = data
    class_by_rng = {int(row["continuation_rng"]): bool(row["collapse"]) for _, row in summary.iterrows()}
    windows = [(85_000, 90_000, "85-90K"), (90_000, 95_000, "90-95K"), (95_000, 100_000, "95-100K")]
    rows: list[dict[str, Any]] = []
    for collapsed, label in [(True, "collapsed"), (False, "non_collapsed")]:
        rngs = [rng for rng in COMPLETE_RNGS if class_by_rng[rng] is collapsed]
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


def write_failure_report(preflight_payload: dict[str, Any], reason: str) -> None:
    lines = [
        "# YC PPO Original seed1 RNG5 CONTROL completion",
        "",
        "本报告记录 022 的 RNG5 CONTROL 补齐尝试；RNG0 仍按既定规则作为 infrastructure-censored，不纳入分母且未重跑。",
        "",
        f"- Starting checkpoint: `{rel(ORIGINAL_CHECKPOINT)}`; SHA256 `{preflight_payload['starting_checkpoint_sha256']}`; RNG5 continuation only.",
        f"- Failure: `{reason}`.",
        f"- Smoke evidence: `{rel(SMOKE_OUT)}`; training evidence is under `{rel(OUT)}`.",
        "- No PPO/DQN/DSSAT scientific code, contract, hyperparameter, site/year scope, or RNG0 artifact was changed or used as a starting point.",
        "",
        "RNG5_COLLAPSE=unknown / insufficient evidence",
        "COLLAPSE decision=unknown / insufficient evidence",
        "YC_PPO_SEED1_RNG5_CONTROL_COMPLETION_INCOMPLETE",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_success_report(preflight_payload: dict[str, Any], summary: pd.DataFrame, stats: dict[str, Any], late: pd.DataFrame, pattern: str, manifest: dict[str, Any]) -> None:
    table = [
        "| RNG | 75K | 80K | 85K | 90K | 95K | 100K | best pre-100K | drop | collapse |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for _, row in summary.iterrows():
        table.append(
            f"| {int(row['continuation_rng'])} | {fmt(row['yield_75K'])} | {fmt(row['yield_80K'])} | {fmt(row['yield_85K'])} | {fmt(row['yield_90K'])} | {fmt(row['yield_95K'])} | {fmt(row['yield_100K'])} | {fmt(row['best_pre100_checkpoint'], 0)} / {fmt(row['best_pre100_yield'])} | {fmt(row['drop_best_to_100K'])} | {row['collapse']} |"
        )
    late_lines: list[str] = []
    for window in ["85-90K", "90-95K", "95-100K"]:
        collapsed = late[(late["trajectory_class"].eq("collapsed")) & late["window"].eq(window)].iloc[0]
        noncollapsed = late[(late["trajectory_class"].eq("non_collapsed")) & late["window"].eq(window)].iloc[0]
        late_lines.append(
            f"- `{window}`: approx_kl collapsed `{fmt(collapsed['approx_kl_median'], 8)}` vs non-collapsed `{fmt(noncollapsed['approx_kl_median'], 8)}`; clip_fraction collapsed `{fmt(collapsed['clip_fraction_median'], 8)}` vs non-collapsed `{fmt(noncollapsed['clip_fraction_median'], 8)}`."
        )
    lines = [
        "# YC PPO Original seed1 RNG5 CONTROL completion",
        "",
        "本报告执行 prompt 022：只新增一条 RNG5 CONTROL continuation，以补足完整 5-trajectory reproducibility baseline。",
        "",
        "## Scope and fixed contract",
        "",
        "- 本轮只新增 RNG5 CONTROL；不是 intervention、不是 KL-GUARD、不是 hyperparameter tuning。",
        f"- Starting point: `{rel(ORIGINAL_CHECKPOINT)}`; SHA256 `{preflight_payload['starting_checkpoint_sha256']}`; `num_timesteps=75000`.",
        "- 恢复 policy/value weights、optimizer state、num_timesteps、learning-rate/clip-range schedule 及可恢复 metadata；仅重新初始化 continuation RNG=5。",
        "- CONTROL: `target_kl=None`; no KL guard, entropy/clip/lr/GAE/gamma/network/reward/observation/action/mask/irrigation-fertilizer constraint changes.",
        "- No training-year deletion/replacement, IC/cultivar change, augmentation, extra site/seed, or scientific-input change.",
        "- Formal PPO/DQN/DSSAT code was not modified; the new 022 file is orchestration/evidence code only.",
        "",
        "## RNG5 smoke and training",
        "",
        f"- Minimal YCA 2011 reset/first-step smoke: `{rel(SMOKE_OUT)}`; it passed before training.",
        f"- RNG5 completion manifest: `{rel(OUT / 'isolated' / 'rng5' / 'completion_manifest.json')}`.",
        f"- Training metrics: `{rel(TRAINING_OUT)}`; required scalar fields are present in the recorded event-derived table.",
        f"- Checkpoint validation: `{rel(VALIDATION_OUT)}`; deterministic masked validation over 2014–2023 for 75K/80K/85K/90K/95K/100K.",
        f"- Final `model_num_timesteps`: `{manifest.get('final_num_timesteps', 'unknown / insufficient evidence')}`.",
        "",
        "## RNG0 handling",
        "",
        "- RNG0 is `INFRASTRUCTURE_CENSORED`: its DSSAT native SIGFPE failures are not PPO collapse, RNG0 is not in the denominator, and RNG0 was not replaced, repaired, or rerun.",
        "- Preserved RNG0 evidence remains under `benchmark_results/020_yc_ppo_seed1_multirng_control_screen/failed_attempts/` and the 021 completion directory.",
        "",
        "## Complete 5-trajectory CONTROL baseline",
        "",
        "分母固定为 RNG1/RNG2/RNG3/RNG4/RNG5；RNG0 不计入。",
        "",
        *table,
        "",
        f"- `collapse_count = {stats['collapse_count']}/5`; `collapse_rate = {stats['collapse_rate']:.3f}`.",
        f"- 100K yield distribution: mean `{fmt(stats['yield_100K_mean'])}`, SD `{fmt(stats['yield_100K_sd'])}`, min `{fmt(stats['yield_100K_min'])}`, max `{fmt(stats['yield_100K_max'])}` kg/ha.",
        f"- Reproducibility CSV: `{rel(SUMMARY_OUT)}`.",
        "",
        "## Late-KL secondary diagnostic",
        "",
        *late_lines,
        f"- 加入 RNG5 后，collapsed trajectories 是否仍伴随更高 late-window approx_kl：**{pattern}**。",
        f"- Late summary CSV: `{rel(LATE_OUT)}`.",
        "- This is descriptive only; it does not establish causality and no KL guard/intervention was run.",
        "",
        "## Required answers",
        "",
        "1. RNG5 是否完整跑到 100K？**YES**。",
        f"2. RNG5 是否 collapse？**{str(bool(summary.loc[summary['continuation_rng'].eq(5), 'collapse'].iloc[0])).upper()}**。",
        f"3. 完整 5 条中的 collapse_count：**{stats['collapse_count']}**。",
        f"4. collapse_rate：**{stats['collapse_rate']:.3f} ({stats['collapse_count']}/5)**。",
        f"5. 100K yield 分布：**mean={fmt(stats['yield_100K_mean'])}, SD={fmt(stats['yield_100K_sd'])}, min={fmt(stats['yield_100K_min'])}, max={fmt(stats['yield_100K_max'])} kg/ha**。",
        f"6. collapsed trajectories 是否仍伴随更高 late approx_kl？**{pattern}**，仅为描述性结果。",
        f"7. 最终决策：**{stats['decision']}**。",
        "",
        "本轮结束；禁止自动继续 KL-GUARD、RNG6+、KL threshold 调整、PPO 调参、扩 seed、扩 Augmented 或扩 site。",
        "",
        "YC_PPO_SEED1_RNG5_CONTROL_COMPLETION_COMPLETE",
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
            write_failure_report(preflight_payload, "RNG5 smoke failed")
            print("RNG5_SMOKE_FAILURE")
            return 2
        if args.phase == "smoke":
            print("RNG5_SMOKE_PASS")
            return 0
    try:
        if args.phase in {"train", "all"}:
            manifest = run_training()
            if args.phase == "train":
                print("RNG5_TRAINING_COMPLETE")
                return 0
        else:
            manifest_path = OUT / "isolated" / "rng5" / "completion_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        if args.phase in {"validate", "all"}:
            run_validation()
            if args.phase == "validate":
                print("RNG5_VALIDATION_COMPLETE")
                return 0
        summary, stats = build_complete_summary()
        late, pattern = build_late_summary(summary)
        write_success_report(preflight_payload, summary, stats, late, pattern, manifest)
        print(stats["decision"])
        print("YC_PPO_SEED1_RNG5_CONTROL_COMPLETION_COMPLETE")
        return 0
    except Exception as exc:
        write_failure_report(preflight_payload, f"{type(exc).__name__}: {exc}")
        write_json(OUT / "completion_failure_manifest.json", {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
        print(f"YC_PPO_SEED1_RNG5_CONTROL_COMPLETION_INCOMPLETE: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
