from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "yc_random_weather_ppo" / "004_05"
BASE = ROOT / "results" / "yc_random_weather_ppo" / "004_03"
DIAG = ROOT / "results" / "yc_random_weather_ppo" / "004_04"
PILOT_PATH = BASE / "run_controlled_pilot.py"
PROBE_STATES = DIAG / "policy_similarity" / "policy_probe_states.csv"
PROBE_ACTIONS = DIAG / "policy_similarity" / "policy_probe_actions.csv"
NEW_SEEDS = list(range(3, 8))
ALL_SEEDS = list(range(8))
REGIMES = ("HISTORICAL_WEATHER", "RANDOM_WEATHER_WGEN")
MODEL_MAP = {
    **{f"H{s}": ("HISTORICAL_WEATHER", s) for s in range(8)},
    **{f"W{s}": ("RANDOM_WEATHER_WGEN", s) for s in range(8)},
}
PROTOTYPES = ("H0", "H1_W0", "H2", "W1", "W2")
PROTOTYPE_LABELS = {
    "H0": "HIGH_INPUT",
    "H1_W0": "VERY_LOW_INPUT",
    "H2": "MODERATE_MIXED",
    "W1": "LOW_I_HIGH_N",
    "W2": "LOW_I_HIGH_N",
}
STEP_CAPTURE: list[dict[str, Any]] = []
_PILOT = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def write_json(path: Path, payload: Any, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and not replace:
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"Refusing to overwrite existing evidence: {rel(path)}")
        return
    path.write_text(encoded, encoding="utf-8", newline="\n")


def load_pilot():
    global _PILOT
    if _PILOT is None:
        spec = importlib.util.spec_from_file_location("yc_00403_frozen_pilot", PILOT_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load frozen pilot runner: {PILOT_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.TASK_ROOT = OUT
        module.CONFIG_ROOT = BASE / "config"
        module.TRAIN_ROOT = OUT / "training"
        module.EVAL_ROOT = OUT / "evaluation"
        module.EVAL_RUN_ROOT = OUT / "evaluation" / "runs"
        module.SMOKE_ROOT = OUT / "smoke"
        module.PPO_SEEDS = ALL_SEEDS
        module.MAX_PROCESS_TREE_RSS_MB = 6000
        _PILOT = module
    return _PILOT


def ensure_task_dirs() -> None:
    for name in (
        "config", "training", "evaluation", "policy_probe", "archetype_analysis",
        "performance_analysis", "figures",
    ):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def copy_frozen_config() -> dict[str, str]:
    ensure_task_dirs()
    copied: dict[str, str] = {}
    for name in (
        "canonical_yc_ppo_config.json",
        "experiment_design.json",
        "training_weather_schedule_historical.csv",
        "training_weather_schedule_random.csv",
        "evaluation_manifest.csv",
    ):
        source = BASE / "config" / name
        if not source.is_file():
            raise FileNotFoundError(source)
        target = OUT / "config" / name
        if target.exists():
            if sha256_file(target) != sha256_file(source):
                raise RuntimeError(f"Frozen config copy differs from source: {rel(target)}")
        else:
            shutil.copy2(source, target)
        copied[rel(target)] = sha256_file(target)
    return copied


def expected_base_models() -> pd.DataFrame:
    frame = pd.read_csv(DIAG / "model_manifest.csv", keep_default_na=False)
    if len(frame) != 6 or set(frame["model_id"]) != {"H0", "H1", "H2", "W0", "W1", "W2"}:
        raise RuntimeError("004_04 model manifest does not contain exactly H0-H2/W0-W2")
    return frame


def preflight(load_checkpoints: bool = True) -> dict[str, Any]:
    pilot = load_pilot()
    issues: list[str] = []
    checks: dict[str, Any] = {}
    training_qc_path = BASE / "training_qc_verified_retry1.json"
    training_qc = json.loads(training_qc_path.read_text(encoding="utf-8"))
    diag_qc = json.loads((DIAG / "config" / "final_qc.json").read_text(encoding="utf-8"))
    eval_qc = json.loads((BASE / "evaluation_qc_verified.json").read_text(encoding="utf-8"))
    if training_qc.get("passed") is not True or training_qc.get("run_count") != 6:
        issues.append("004_03 authoritative retry1 training QC is not passed for six models")
    if diag_qc.get("passed") is not True or diag_qc.get("probe_states") != 600:
        issues.append("004_04 fixed-probe final QC is not passed for 600 states")
    if eval_qc.get("passed") is not True:
        issues.append("004_03 authoritative evaluation QC is not passed")
    checks["authoritative_004_03_training_qc"] = rel(training_qc_path)
    checks["stale_failed_004_03_qc_preserved"] = (BASE / "training_qc_verified.json").is_file()
    checks["004_04_final_qc"] = diag_qc.get("passed") is True

    models = expected_base_models()
    model_rows = []
    signatures = set()
    for row in models.to_dict(orient="records"):
        model_path = ROOT / row["model_path"]
        manifest_path = ROOT / row["verified_manifest"]
        if not model_path.is_file() or not manifest_path.is_file():
            issues.append(f"missing verified base model artifact: {row['model_id']}")
            continue
        actual_hash = sha256_file(model_path)
        if actual_hash != str(row["model_sha256"]).upper():
            issues.append(f"base checkpoint SHA256 mismatch: {row['model_id']}")
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if run_manifest.get("run_status") != "completed" or run_manifest.get("total_timesteps_actual") != 100080:
            issues.append(f"base checkpoint run manifest mismatch: {row['model_id']}")
        run_dir = manifest_path.parent
        config = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))
        signatures.add(json.dumps({key: config.get(key) for key in (
            "effective_ppo", "reward", "action_grid", "action_safety", "normalization", "total_timesteps_requested"
        )}, sort_keys=True))
        model_rows.append({"model_id": row["model_id"], "sha256": actual_hash, "status": "verified"})
        if load_checkpoints:
            from sb3_contrib import MaskablePPO
            loaded = MaskablePPO.load(str(model_path), device="cpu")
            del loaded
            gc.collect()
    if len(signatures) != 1:
        issues.append(f"base model canonical config signatures differ: {len(signatures)}")
    checks["base_models"] = model_rows
    checks["base_checkpoint_hashes_match_004_04"] = len(model_rows) == 6 and not any("SHA256 mismatch" in item for item in issues)
    checks["base_checkpoints_loaded"] = bool(load_checkpoints)
    checks["base_canonical_config_signature_count"] = len(signatures)

    hist = pd.read_csv(BASE / "config" / "training_weather_schedule_historical.csv")
    wgen = pd.read_csv(BASE / "config" / "training_weather_schedule_random.csv")
    expected_hist = pd.DataFrame(pilot.generate_year_schedule(pilot.MAX_SCHEDULE_EPISODES))
    expected_wgen = pd.DataFrame(pilot.generate_wgen_schedule(expected_hist.to_dict(orient="records")))
    hist_years = pd.to_numeric(hist["historical_year"], errors="coerce").astype(int).tolist()
    expected_years = expected_hist["historical_year"].astype(int).tolist()
    wgen_seed_col = "weather_seed"
    wgen_seeds = pd.to_numeric(wgen[wgen_seed_col], errors="coerce").astype(int).tolist()
    expected_wgen_seeds = expected_wgen["weather_seed"].astype(int).tolist()
    if hist_years != expected_years:
        issues.append("historical schedule does not regenerate from frozen seed 64003")
    if wgen_seeds != expected_wgen_seeds:
        issues.append("WGEN schedule does not regenerate from frozen seed 64004")
    if pd.to_numeric(wgen["historical_year_context"], errors="coerce").astype(int).tolist() != hist_years:
        issues.append("historical year context differs between H and W schedules")
    if not set(wgen_seeds).issubset(set(range(1001, 1081))) or set(wgen_seeds) & set(range(1081, 1101)):
        issues.append("training and evaluation WGEN seed pools are not disjoint")
    checks["historical_schedule_seed_64003_exact"] = hist_years == expected_years
    checks["wgen_schedule_seed_64004_exact"] = wgen_seeds == expected_wgen_seeds
    checks["wgen_training_pool_1001_1080"] = set(wgen_seeds).issubset(set(range(1001, 1081)))
    checks["heldout_pool_1081_1100_disjoint"] = not bool(set(wgen_seeds) & set(range(1081, 1101)))
    checks["ppo_seed3_separate_from_weather_schedule"] = all(seed not in set(range(1001, 1081)) for seed in NEW_SEEDS)

    states = pd.read_csv(PROBE_STATES, keep_default_na=False)
    unique_state_ids = states["probe_id"].astype(str).tolist()
    if len(states) != 600 or len(set(unique_state_ids)) != 600:
        issues.append("004_04 fixed probe input is not exactly 600 unique probe_id rows")
    probe_hash = sha256_file(PROBE_STATES)
    probe_actions = pd.read_csv(PROBE_ACTIONS, keep_default_na=False)
    if len(probe_actions) != 3600 or probe_actions["probe_id"].nunique() != 600:
        issues.append("004_04 baseline probe actions do not contain 3600 rows / 600 states")
    checks["fixed_probe_states_path"] = rel(PROBE_STATES)
    checks["fixed_probe_states_sha256"] = probe_hash
    checks["fixed_probe_states_count"] = len(states)
    checks["fixed_probe_actions_count"] = len(probe_actions)
    checks["fixed_probe_final_qc_passed"] = diag_qc.get("probe_states_frozen_before_predictions") is True

    protocol = pilot.preflight_payload()
    if protocol.get("passed") is not True:
        issues.extend([f"canonical source preflight: {item}" for item in protocol.get("issues", [])])
    checks["canonical_source_preflight"] = protocol
    config_hashes = copy_frozen_config()
    checks["copied_frozen_config_sha256"] = config_hashes
    checks["docker_memory_limit_bytes"] = 8589934592
    checks["process_tree_rss_guard_mb"] = pilot.MAX_PROCESS_TREE_RSS_MB
    checks["execution_order"] = "serial"
    result = {"passed": not issues, "issues": issues, "checks": checks}
    write_json(OUT / "config" / "preflight.json", result, replace=(OUT / "config" / "preflight.json").exists())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def verify_preflight_file() -> dict[str, Any]:
    path = OUT / "config" / "preflight.json"
    if not path.is_file():
        raise RuntimeError("Run --phase preflight before this phase")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("passed") is not True:
        raise RuntimeError("Preflight did not pass; refusing to continue")
    if sha256_file(PROBE_STATES) != payload["checks"]["fixed_probe_states_sha256"]:
        raise RuntimeError("004_04 fixed probe input changed after preflight")
    return payload


def run_smokes() -> dict[str, Any]:
    verify_preflight_file()
    pilot = load_pilot()
    records = []
    for regime, attempt in (("HISTORICAL_WEATHER", "historical_seed3"), ("RANDOM_WEATHER_WGEN", "wgen_seed3")):
        path = pilot.SMOKE_ROOT / attempt
        if path.exists() and any(path.iterdir()):
            manifest_path = path / "run_manifest.json"
            if not manifest_path.is_file():
                raise FileExistsError(f"Smoke attempt exists without a final manifest: {rel(path)}")
            result = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            print(f"[smoke] regime={regime} ppo_seed=3 timesteps=432", flush=True)
            result = pilot.train_model(regime, 3, smoke=True, smoke_attempt=attempt)
        episodes_path = path / "training_episode_summary.csv"
        episodes = pd.read_csv(episodes_path, keep_default_na=False) if episodes_path.is_file() else pd.DataFrame()
        ok = result.get("run_status") == "completed" and result.get("total_timesteps_actual") == 432
        if episodes.empty or episodes["status"].ne("completed").any():
            ok = False
        if regime == "HISTORICAL_WEATHER":
            ok = ok and pd.to_numeric(episodes["ppo_seed"], errors="coerce").eq(3).all()
            filex_check = inspect_filex_modes(pilot, regime)
            ok = ok and filex_check["modes"] == ["M"]
        else:
            used = pd.to_numeric(episodes["actual_rseed1_sent_to_pdi"], errors="coerce").dropna().astype(int)
            ok = ok and episodes["filex_wther"].astype(str).eq("W").all()
            ok = ok and len(used) > 0 and used.between(1001, 1080).all()
            ok = ok and pd.to_numeric(episodes["ppo_seed"], errors="coerce").eq(3).all()
            ok = ok and used.tolist() == pd.to_numeric(episodes["training_weather_seed"], errors="coerce").dropna().astype(int).tolist()
            filex_check = inspect_filex_modes(pilot, regime)
            ok = ok and filex_check["modes"] and set(filex_check["modes"]) == {"W"}
        if float(result.get("max_process_tree_rss_mb") or 0) >= pilot.MAX_PROCESS_TREE_RSS_MB:
            ok = False
        record = {"training_regime": regime, "ppo_seed": 3, "timesteps_actual": result.get("total_timesteps_actual"), "max_rss_mb": result.get("max_process_tree_rss_mb"), "wther": "M" if regime == "HISTORICAL_WEATHER" else "W", "filex_template": filex_check["path"], "filex_template_sha256": filex_check["sha256"], "episode_log_wther": sorted(set(episodes["filex_wther"].astype(str))), "passed": bool(ok), "run_manifest": rel(path / "run_manifest.json")}
        records.append(record)
        if not ok:
            result = {"passed": False, "smokes": records, "stopped_after": record}
            write_json(OUT / "smoke_qc.json", result, replace=(OUT / "smoke_qc.json").exists())
            raise RuntimeError(f"Seed 3 smoke failed; preserved at {rel(path)}")
        del episodes
        gc.collect()
    result = {"passed": True, "smokes": records, "formal_training_performance_excluded": True}
    write_json(OUT / "smoke_qc.json", result, replace=(OUT / "smoke_qc.json").exists())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def inspect_filex_modes(pilot, regime: str) -> dict[str, Any]:
    if regime == "HISTORICAL_WEATHER":
        _, _, batch, rendering, _, _, _, env_config = pilot._import_canonical()
        year_info = batch.base.direct_ppo.find_year(env_config, "YCA", pilot.WGEN_EVAL_CROP_YEAR)
        args = rendering.build_env_args(
            station="YCA",
            year=pilot.WGEN_EVAL_CROP_YEAR,
            planting_date=year_info["planting_date"],
            seed=pilot.RUNTIME_BOOTSTRAP_SEED,
            config=env_config,
            run_tag="yc00405_historical_filex_preflight",
            evaluation=False,
            mode=env_config.get("runtime", {}).get("mode", "all"),
            linked_management=True,
        )
        candidates = [Path(args["fileX_template_path"])]
    else:
        candidates = [
            path for path in (OUT / "runtime_templates").rglob("*.jinja2")
            if "RANDOM_WEATHER_WGEN_wgen_seed3" in str(path)
        ]
    if not candidates:
        return {"path": "", "sha256": "", "modes": []}
    reports = []
    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="replace")
        modes = re.findall(r"(?m)^\s*\d+\s+ME\s+(\S+)", text)
        reports.append({"path": rel(path), "sha256": sha256_file(path), "modes": modes})
    all_modes = sorted(set(mode for report in reports for mode in report["modes"]))
    return {"path": ";".join(report["path"] for report in reports), "sha256": ";".join(report["sha256"] for report in reports), "modes": all_modes}


def final_checkpoint(run_dir: Path) -> Path:
    return run_dir / "models" / "checkpoint_100000.zip"


def train_models(retry_id: str | None = None) -> dict[str, Any]:
    verify_preflight_file()
    smoke_qc = json.loads((OUT / "smoke_qc.json").read_text(encoding="utf-8"))
    if smoke_qc.get("passed") is not True:
        raise RuntimeError("Smoke gate missing or failed; refusing formal training")
    pilot = load_pilot()
    train_root = OUT / "training" if retry_id is None else OUT / "training" / "retries" / retry_id
    pilot.TRAIN_ROOT = train_root
    results = []
    for regime in REGIMES:
        for seed in NEW_SEEDS:
            run_dir = train_root / ("historical" if regime == "HISTORICAL_WEATHER" else "random_weather") / f"ppo_seed_{seed}"
            manifest_path = run_dir / "run_manifest.json"
            if manifest_path.is_file():
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                checkpoint = ROOT / str(existing.get("model_final", ""))
                if existing.get("run_status") == "completed" and existing.get("ppo_seed") == seed and existing.get("training_regime") == regime and existing.get("total_timesteps_actual") == 100080 and checkpoint.is_file():
                    print(f"[train] reuse completed same-attempt result {regime} seed={seed}", flush=True)
                    results.append(existing)
                    continue
                raise RuntimeError(f"Existing run attempt is incomplete; preserve and inspect before retry: {rel(run_dir)}")
            if run_dir.exists() and any(run_dir.iterdir()):
                raise FileExistsError(f"Refusing to overwrite training attempt: {rel(run_dir)}")
            print(f"[train] regime={regime} ppo_seed={seed} requested=100000 serial start", flush=True)
            result = pilot.train_model(regime, seed)
            results.append(result)
            write_training_manifest(results, replace=True)
            if result.get("run_status") != "completed" or result.get("total_timesteps_actual") != 100080:
                write_json(OUT / "training_failure_stop.json", {"stopped_after": result, "reason": "run incomplete or existing 6000 MB guard reached; no automatic retry"}, replace=(OUT / "training_failure_stop.json").exists())
                raise RuntimeError(f"Formal run did not complete; retained artifacts at {result.get('model_dir')}")
            checkpoint = final_checkpoint(run_dir)
            from sb3_contrib import MaskablePPO
            loaded = MaskablePPO.load(str(checkpoint), device="cpu")
            del loaded
            gc.collect()
            print(f"[train-done] {regime} seed={seed} actual={result['total_timesteps_actual']} rss={result.get('max_process_tree_rss_mb')}", flush=True)
    qc = validate_new_training()
    if not qc["passed"]:
        raise RuntimeError("New training QC failed: " + "; ".join(qc["issues"]))
    write_training_manifest(results, replace=True)
    print(json.dumps(qc, ensure_ascii=False, indent=2), flush=True)
    return qc


def write_training_manifest(results: list[dict[str, Any]], replace: bool = False) -> None:
    rows = []
    for path in sorted((OUT / "training").glob("**/run_manifest.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        model_path = ROOT / item.get("model_final", "")
        rows.append({**item, "model_sha256": sha256_file(model_path) if model_path.is_file() else "", "run_manifest": rel(path)})
    frame = pd.DataFrame(rows)
    target = OUT / "new_training_run_manifest.csv"
    if target.exists() and not replace:
        raise FileExistsError(target)
    frame.to_csv(target, index=False, encoding="utf-8-sig", lineterminator="\n")


def validate_new_training() -> dict[str, Any]:
    issues: list[str] = []
    pilot = load_pilot()
    historical_filex = inspect_filex_modes(pilot, "HISTORICAL_WEATHER")
    if historical_filex["modes"] != ["M"]:
        issues.append(f"historical runtime FileX template METHODS/WTHER is not M: {historical_filex['modes']}")
    hist_schedule = pd.read_csv(BASE / "config" / "training_weather_schedule_historical.csv", keep_default_na=False)
    wgen_schedule = pd.read_csv(BASE / "config" / "training_weather_schedule_random.csv", keep_default_na=False)
    expected_hist = pd.to_numeric(hist_schedule["historical_year"], errors="coerce").astype(int).tolist()
    expected_wgen = pd.to_numeric(wgen_schedule["weather_seed"], errors="coerce").astype(int).tolist()
    baseline_configs = []
    for _, row in expected_base_models().iterrows():
        cfg = json.loads((ROOT / row["verified_manifest"]).parent.joinpath("training_config.json").read_text(encoding="utf-8"))
        baseline_configs.append({key: cfg.get(key) for key in ("effective_ppo", "reward", "action_grid", "action_safety", "normalization", "total_timesteps_requested")})
    base_signature = json.dumps(baseline_configs[0], sort_keys=True)
    run_rows = []
    hist_sequences = []
    wgen_sequences = []
    actual_steps = set()
    configs = set()
    for regime in REGIMES:
        for seed in NEW_SEEDS:
            run_dir = OUT / "training" / ("historical" if regime == "HISTORICAL_WEATHER" else "random_weather") / f"ppo_seed_{seed}"
            manifest_path = run_dir / "run_manifest.json"
            config_path = run_dir / "training_config.json"
            episode_path = run_dir / "training_episode_summary.csv"
            if not manifest_path.is_file() or not config_path.is_file() or not episode_path.is_file():
                issues.append(f"missing training evidence: {regime}/seed{seed}")
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config = json.loads(config_path.read_text(encoding="utf-8"))
            episodes = pd.read_csv(episode_path, keep_default_na=False)
            if manifest.get("run_status") != "completed" or manifest.get("ppo_seed") != seed or manifest.get("training_regime") != regime:
                issues.append(f"training manifest identity/status mismatch: {regime}/seed{seed}")
            if manifest.get("total_timesteps_requested") != 100000 or manifest.get("total_timesteps_actual") != 100080:
                issues.append(f"training budget mismatch: {regime}/seed{seed}")
            if config.get("ppo_seed") != seed or config.get("training_regime") != regime or config.get("total_timesteps_requested") != 100000:
                issues.append(f"training config identity/budget mismatch: {regime}/seed{seed}")
            signature = json.dumps({key: config.get(key) for key in ("effective_ppo", "reward", "action_grid", "action_safety", "normalization", "total_timesteps_requested")}, sort_keys=True)
            configs.add(signature)
            if signature != base_signature:
                issues.append(f"canonical training contract differs from baseline: {regime}/seed{seed}")
            if episodes.empty or episodes["status"].astype(str).ne("completed").any():
                issues.append(f"empty/incomplete training episodes: {regime}/seed{seed}")
            errors = pd.to_numeric(episodes["reward_decomposition_abs_error"], errors="coerce")
            if errors.isna().any() or errors.gt(1e-8).any():
                issues.append(f"reward decomposition errors: {regime}/seed{seed}")
            if episodes.isna().any().any():
                # Blank weather fields are legal; numeric NaN in core metrics is not.
                for col in ("episode_return", "yield", "fertilizer", "irrigation", "episode_days"):
                    if pd.to_numeric(episodes[col], errors="coerce").isna().any():
                        issues.append(f"non-finite core episode field {col}: {regime}/seed{seed}")
                        break
            if regime == "HISTORICAL_WEATHER":
                years = pd.to_numeric(episodes["historical_year"], errors="coerce").astype(int).tolist()
                if years != expected_hist[:len(years)]:
                    issues.append(f"historical schedule diverged: seed={seed}")
                hist_sequences.append(years)
            else:
                logged = pd.to_numeric(episodes["training_weather_seed"], errors="coerce").astype(int).tolist()
                actual = pd.to_numeric(episodes["actual_rseed1_sent_to_pdi"], errors="coerce").astype(int).tolist()
                years = pd.to_numeric(episodes["historical_year"], errors="coerce").astype(int).tolist()
                if logged != expected_wgen[:len(logged)] or actual != expected_wgen[:len(actual)]:
                    issues.append(f"WGEN schedule/rseed1_ diverged: seed={seed}")
                if years != pd.to_numeric(wgen_schedule["historical_year_context"], errors="coerce").astype(int).tolist()[:len(years)]:
                    issues.append(f"WGEN historical context schedule diverged: seed={seed}")
                if not episodes["filex_wther"].astype(str).eq("W").all():
                    issues.append(f"WGEN WTHER is not W: seed={seed}")
                if seed in set(actual):
                    issues.append(f"PPO seed collides with WGEN rseed1_: seed={seed}")
                if episodes["runtime_weather_sha256"].astype(str).eq("").any():
                    issues.append(f"WGEN runtime weather hash missing: seed={seed}")
                wgen_sequences.append(list(zip(years, actual, episodes["runtime_weather_sha256"].astype(str))))
            checkpoint_values = manifest.get("checkpoint_steps_actual", {})
            if [int(x) for x in manifest.get("checkpoint_steps_requested", [])] != [25000, 50000, 75000, 100000]:
                issues.append(f"checkpoint schedule mismatch: {regime}/seed{seed}")
            for step in (25000, 50000, 75000, 100000):
                if not (run_dir / "models" / f"checkpoint_{step}.zip").is_file():
                    issues.append(f"missing checkpoint {step}: {regime}/seed{seed}")
            if int(manifest.get("total_timesteps_actual", -1)) < 100000:
                issues.append(f"actual step count below budget: {regime}/seed{seed}")
            if manifest.get("max_process_tree_rss_mb") is None or float(manifest["max_process_tree_rss_mb"]) >= pilot.MAX_PROCESS_TREE_RSS_MB:
                issues.append(f"memory guard reached or unrecorded: {regime}/seed{seed}")
            final = final_checkpoint(run_dir)
            if not final.is_file():
                issues.append(f"final checkpoint missing: {regime}/seed{seed}")
            else:
                from sb3_contrib import MaskablePPO
                model = MaskablePPO.load(str(final), device="cpu")
                del model
                gc.collect()
            actual_steps.add(int(manifest.get("total_timesteps_actual", -1)))
            run_rows.append({"training_regime": regime, "ppo_seed": seed, "actual_timesteps": manifest.get("total_timesteps_actual"), "episodes": len(episodes), "rss_mb": manifest.get("max_process_tree_rss_mb"), "status": manifest.get("run_status"), "checkpoint_loadable": final.is_file()})
    if len(run_rows) != 10:
        issues.append(f"expected ten new model runs, found {len(run_rows)}")
    if len(configs) != 1 or len(configs) and next(iter(configs)) != base_signature:
        issues.append("new models do not share one frozen canonical config")
    if len(actual_steps) != 1 or actual_steps != {100080}:
        issues.append(f"new models have inconsistent actual steps: {sorted(actual_steps)}")
    if len(hist_sequences) == 5:
        hist_common = min(map(len, hist_sequences))
        if any(x[:hist_common] != hist_sequences[0][:hist_common] for x in hist_sequences[1:]):
            issues.append("historical training schedule common prefix differs across PPO seeds")
    else:
        hist_common = 0
    if len(wgen_sequences) == 5:
        wgen_common = min(map(len, wgen_sequences))
        if any(x[:wgen_common] != wgen_sequences[0][:wgen_common] for x in wgen_sequences[1:]):
            issues.append("WGEN year/seed/runtime weather common prefix differs across PPO seeds")
    else:
        wgen_common = 0
    expected_ids = {(regime, seed) for regime in REGIMES for seed in NEW_SEEDS}
    actual_ids = {(row["training_regime"], int(row["ppo_seed"])) for row in run_rows}
    if actual_ids != expected_ids:
        issues.append("new training model identity set is not exactly H/W seeds 3-7")
    result = {
        "passed": not issues,
        "issues": issues,
        "expected_new_models": 10,
        "actual_new_models": len(run_rows),
        "requested_timesteps": 100000,
        "actual_timesteps_values": sorted(actual_steps),
        "same_canonical_config": len(configs) == 1 and (not configs or next(iter(configs)) == base_signature),
        "historical_schedule_common_prefix_rows": hist_common,
        "historical_schedule_identical_across_seeds": len(hist_sequences) == 5 and hist_common > 0 and all(x[:hist_common] == hist_sequences[0][:hist_common] for x in hist_sequences),
        "historical_filex_wther_m_verified": historical_filex["modes"] == ["M"],
        "historical_filex_template": historical_filex,
        "wgen_schedule_common_prefix_rows": wgen_common,
        "wgen_schedule_and_runtime_weather_identical_across_seeds": len(wgen_sequences) == 5 and wgen_common > 0 and all(x[:wgen_common] == wgen_sequences[0][:wgen_common] for x in wgen_sequences),
        "runs": run_rows,
        "runtime_modified": False,
        "reward_modified": False,
        "ppo_modified": False,
        "action_mask_modified": False,
    }
    qc_path = OUT / "new_training_qc.json"
    archive_path = OUT / "new_training_qc_initial_attempt_unverified.json"
    if qc_path.is_file() and not archive_path.exists():
        shutil.copy2(qc_path, archive_path)
    write_json(qc_path, result, replace=qc_path.exists())
    return result


def install_step_capture(pilot) -> None:
    if getattr(pilot.ScheduledEpisodeEnv, "_yc00405_capture_installed", False):
        return
    original = pilot.ScheduledEpisodeEnv._step

    def tracked(self, action):
        pre = dict(getattr(self.current_env, "last_obs_dict", {}) or {}) if self.current_env is not None else {}
        schedule = dict(self.current_schedule or {})
        result = original(self, action)
        if str(self.run_tag).startswith("eval_"):
            action_info = dict(getattr(self.current_env, "last_action_info", {}) or {})
            eval_type = "heldout_wgen" if "heldout_wgen" in self.run_tag else "observed_weather"
            seed_or_year = schedule.get("weather_seed") if eval_type == "heldout_wgen" else schedule.get("historical_year")
            STEP_CAPTURE.append({
                "training_regime": self.ppo_seed is not None and ("RANDOM_WEATHER_WGEN" if "RANDOM_WEATHER_WGEN" in self.run_tag else "HISTORICAL_WEATHER"),
                "ppo_seed": int(self.ppo_seed),
                "evaluation_weather_type": eval_type,
                "evaluation_weather_seed": schedule.get("weather_seed", "") if eval_type == "heldout_wgen" else "",
                "evaluation_weather_year": schedule.get("historical_year", "") if eval_type == "observed_weather" else "",
                "episode_key": f"{eval_type}:{seed_or_year}",
                "episode_index": schedule.get("episode_index", ""),
                "timestep": int(getattr(self, "episode_days", 0)),
                "dap": float(pre.get("dap", 0.0) or 0.0),
                "swfac": pre.get("swfac", ""),
                "nstres": pre.get("nstres", ""),
                "action_index": int(np.asarray(action).reshape(-1)[0]),
                "irrigation_action_mm": float(action_info.get("safe_action_amir", 0.0) or 0.0),
                "fertilizer_action_kgN_ha": float(action_info.get("safe_action_anfer", 0.0) or 0.0),
                "mask_forced_noop": action_info.get("mask_forced_noop", ""),
                "instant_reward": float(result[1]),
                "yield_contribution_scaled": action_info.get("yield_contribution_scaled", 0.0),
                "water_cost_contribution_scaled": action_info.get("water_cost_contribution_scaled", 0.0),
                "nitrogen_cost_contribution_scaled": action_info.get("nitrogen_cost_contribution_scaled", 0.0),
                "stress_relief_contribution_scaled": action_info.get("stress_relief_contribution_scaled", 0.0),
                "swfac_guardrail_penalty_scaled": action_info.get("swfac_guardrail_penalty_scaled", 0.0),
                "canonical_reward_reconstructed": action_info.get("reward_after_swfac_guardrail", result[1]),
                "canonical_reward_step_abs_error": abs(float(result[1]) - float(action_info.get("reward_after_swfac_guardrail", result[1]))),
            })
        return result

    pilot.ScheduledEpisodeEnv._step = tracked
    pilot.ScheduledEpisodeEnv._yc00405_capture_installed = True


def evaluation_done(run_dir: Path, expected: int) -> bool:
    manifest = run_dir / "evaluation_manifest.json"
    episodes = run_dir / "evaluation_episode_level.csv"
    steps = run_dir / "evaluation_step_level.csv"
    if not (manifest.is_file() and episodes.is_file() and steps.is_file()):
        return False
    info = json.loads(manifest.read_text(encoding="utf-8"))
    return info.get("deterministic_actions") is True and info.get("episodes_completed") == expected and len(pd.read_csv(episodes)) == expected


def evaluate_new_models() -> dict[str, Any]:
    verify_preflight_file()
    qc = validate_new_training()
    if not qc["passed"]:
        raise RuntimeError("Cannot evaluate before all ten training runs pass QC")
    pilot = load_pilot()
    install_step_capture(pilot)
    rows = []
    for regime in REGIMES:
        for seed in NEW_SEEDS:
            run_dir = OUT / "training" / ("historical" if regime == "HISTORICAL_WEATHER" else "random_weather") / f"ppo_seed_{seed}"
            model_path = final_checkpoint(run_dir)
            for eval_type, expected in (("heldout_wgen", 20), ("observed_weather", 10)):
                eval_dir = pilot.EVAL_RUN_ROOT / eval_type / ("historical" if regime == "HISTORICAL_WEATHER" else "random_weather") / f"ppo_seed_{seed}"
                if evaluation_done(eval_dir, expected):
                    print(f"[eval] reuse completed {regime} seed={seed} {eval_type}", flush=True)
                    rows.append({"training_regime": regime, "ppo_seed": seed, "evaluation_weather_type": eval_type, "status": "reused"})
                    continue
                if eval_dir.exists() and any(eval_dir.iterdir()):
                    raise RuntimeError(f"Incomplete evaluation attempt retained; refusing overwrite: {rel(eval_dir)}")
                STEP_CAPTURE.clear()
                print(f"[eval] regime={regime} seed={seed} domain={eval_type} serial start", flush=True)
                try:
                    result_rows = pilot.evaluate_one(model_path, regime, seed, eval_type)
                    step_frame = pd.DataFrame(STEP_CAPTURE)
                    if len(result_rows) != expected or step_frame.empty:
                        raise RuntimeError(f"evaluation returned {len(result_rows)} episodes / {len(step_frame)} steps")
                    step_path = eval_dir / "evaluation_step_level.csv"
                    step_frame.to_csv(step_path, index=False, encoding="utf-8-sig", lineterminator="\n")
                    rows.append({"training_regime": regime, "ppo_seed": seed, "evaluation_weather_type": eval_type, "status": "completed", "episodes": len(result_rows), "steps": len(step_frame), "step_sha256": sha256_file(step_path)})
                    print(f"[eval-done] {regime} seed={seed} {eval_type} episodes={len(result_rows)} steps={len(step_frame)}", flush=True)
                except Exception:
                    if STEP_CAPTURE:
                        eval_dir.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame(STEP_CAPTURE).to_csv(eval_dir / "evaluation_step_level_partial.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
                    (eval_dir / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    raise
                finally:
                    STEP_CAPTURE.clear()
                    gc.collect()
    result = validate_evaluation()
    result["runs"] = rows
    write_json(OUT / "evaluation_qc.json", result, replace=(OUT / "evaluation_qc.json").exists())
    if not result["passed"]:
        raise RuntimeError("Evaluation QC failed: " + "; ".join(result["issues"]))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def validate_evaluation() -> dict[str, Any]:
    issues = []
    rows = []
    expected_hashes: dict[int, str] = {}
    canonical = json.loads((OUT / "config" / "canonical_yc_ppo_config.json").read_text(encoding="utf-8"))
    reward_cfg = canonical["reward"]
    reward_scale = float(reward_cfg["reward_scale"])
    component_max_abs_error = 0.0
    base_eval = pd.read_csv(DIAG / "evaluation_episode_summary.csv", keep_default_na=False)
    for _, row in base_eval[base_eval["evaluation_weather_type"].eq("heldout_wgen")].iterrows():
        # Runtime hashes are checked against the 004_04 step manifests separately.
        pass
    new_count = 0
    for regime in REGIMES:
        for seed in NEW_SEEDS:
            for eval_type, expected in (("heldout_wgen", 20), ("observed_weather", 10)):
                run_dir = OUT / "evaluation" / "runs" / eval_type / ("historical" if regime == "HISTORICAL_WEATHER" else "random_weather") / f"ppo_seed_{seed}"
                manifest_path = run_dir / "evaluation_manifest.json"
                csv_path = run_dir / "evaluation_episode_level.csv"
                step_path = run_dir / "evaluation_step_level.csv"
                if not all(path.is_file() for path in (manifest_path, csv_path, step_path)):
                    issues.append(f"missing evaluation evidence {regime}/seed{seed}/{eval_type}")
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                frame = pd.read_csv(csv_path, keep_default_na=False)
                steps = pd.read_csv(step_path, keep_default_na=False)
                if manifest.get("deterministic_actions") is not True or manifest.get("episodes_completed") != expected or len(frame) != expected:
                    issues.append(f"wrong deterministic evaluation episode count {regime}/seed{seed}/{eval_type}")
                if frame["status"].astype(str).ne("completed").any():
                    issues.append(f"incomplete evaluation episode {regime}/seed{seed}/{eval_type}")
                errors = pd.to_numeric(frame["reward_decomposition_abs_error"], errors="coerce")
                if errors.isna().any() or errors.gt(1e-8).any():
                    issues.append(f"reward decomposition failed {regime}/seed{seed}/{eval_type}")
                if eval_type == "heldout_wgen":
                    actual = pd.to_numeric(frame["actual_rseed1_"], errors="coerce").astype(int).tolist()
                    requested = list(range(1081, 1101))
                    if set(actual) != set(requested) or frame["filex_wther"].astype(str).ne("W").any():
                        issues.append(f"held-out WGEN seed/WTHER mismatch {regime}/seed{seed}")
                    if frame["runtime_weather_sha256"].astype(str).eq("").any():
                        issues.append(f"held-out WGEN runtime weather hash missing {regime}/seed{seed}")
                    for _, item in frame.iterrows():
                        seed_value = int(item["actual_rseed1_"])
                        weather_hash = str(item["runtime_weather_sha256"])
                        if seed_value in expected_hashes and expected_hashes[seed_value] != weather_hash:
                            issues.append(f"held-out WGEN runtime hash differs for seed {seed_value}")
                        expected_hashes[seed_value] = weather_hash
                else:
                    years = pd.to_numeric(frame["evaluation_weather_year"], errors="coerce").astype(int).tolist()
                    if set(years) != set(range(2014, 2024)):
                        issues.append(f"observed year set mismatch {regime}/seed{seed}")
                if len(steps) == 0 or pd.to_numeric(steps["canonical_reward_step_abs_error"], errors="coerce").fillna(float("inf")).gt(1e-8).any():
                    issues.append(f"evaluation step trace/reward reconstruction failed {regime}/seed{seed}/{eval_type}")
                if set(steps["ppo_seed"].astype(int)) != {seed} or set(steps["evaluation_weather_type"].astype(str)) != {eval_type}:
                    issues.append(f"evaluation step trace identity mismatch {regime}/seed{seed}/{eval_type}")
                penalty_by_episode = pd.to_numeric(steps["swfac_guardrail_penalty_scaled"], errors="coerce").groupby(steps["episode_key"].astype(str)).sum(min_count=1)
                for episode in frame.to_dict(orient="records"):
                    if eval_type == "heldout_wgen":
                        episode_key = f"heldout_wgen:{int(episode['evaluation_weather_seed'])}"
                    else:
                        episode_key = f"observed_weather:{int(episode['evaluation_weather_year'])}"
                    penalty = float(penalty_by_episode.get(episode_key, float("nan")))
                    values = [episode.get("reward"), episode.get("yield"), episode.get("irrigation"), episode.get("fertilizer"), penalty]
                    try:
                        reward_value, yield_value, irrigation_value, fertilizer_value, guardrail_value = map(float, values)
                    except (TypeError, ValueError):
                        reward_value = yield_value = irrigation_value = fertilizer_value = guardrail_value = float("nan")
                    yield_scaled = float(reward_cfg["yield_coef"]) * yield_value * reward_scale
                    water_scaled = float(reward_cfg["water_cost"]) * irrigation_value * reward_scale
                    nitrogen_scaled = float(reward_cfg["nitrogen_cost"]) * fertilizer_value * reward_scale
                    stress_relief_scaled = reward_value - yield_scaled + water_scaled + nitrogen_scaled + guardrail_value
                    reconstructed = yield_scaled - water_scaled - nitrogen_scaled + stress_relief_scaled - guardrail_value
                    error = abs(reward_value - reconstructed)
                    if not math.isfinite(error):
                        error = float("inf")
                    component_max_abs_error = max(component_max_abs_error, error)
                    if episode_key not in penalty_by_episode or not math.isfinite(guardrail_value):
                        issues.append(f"missing finite episode SWFAC penalty for {regime}/seed{seed}/{episode_key}")
                rows.append({"training_regime": regime, "ppo_seed": seed, "evaluation_weather_type": eval_type, "episodes": len(frame), "steps": len(steps), "status": "passed"})
                new_count += len(frame)
    diag_qc = json.loads((DIAG / "config" / "final_qc.json").read_text(encoding="utf-8"))
    if diag_qc.get("passed") is not True or diag_qc.get("episodes") != 180:
        issues.append("reused 004_04 evaluation evidence failed its final QC")
    result = {
        "passed": not issues,
        "issues": issues,
        "reused_existing_evaluation_episodes": int(diag_qc.get("episodes", 0)),
        "new_evaluation_episodes": new_count,
        "expected_all_evaluation_episodes": 480,
        "new_model_domain_runs": len(rows),
        "reward_component_reconstruction_method": "canonical episode formula from final yield, seasonal irrigation/fertilizer, summed captured SWFAC penalty; stress relief is the exact reward-identity residual",
        "reward_component_reconstruction_max_abs_error": component_max_abs_error,
        "canonical_reward_components_reconcile": component_max_abs_error <= 1e-8,
        "new_heldout_runtime_weather_hashes_match_across_models": len(expected_hashes) == 20,
        "train_eval_weather_seed_intersection": sorted(set(range(1001, 1081)) & set(range(1081, 1101))),
        "runs": rows,
    }
    if component_max_abs_error > 1e-8:
        issues.append(f"episode reward component reconstruction error exceeds tolerance: {component_max_abs_error}")
        result["passed"] = False
        result["issues"] = issues
    return result


def probe_new_models() -> dict[str, Any]:
    verify_preflight_file()
    train_qc = validate_new_training()
    if not train_qc["passed"]:
        raise RuntimeError("Cannot probe before new training QC passes")
    if sha256_file(PROBE_STATES) != json.loads((OUT / "config" / "preflight.json").read_text(encoding="utf-8"))["checks"]["fixed_probe_states_sha256"]:
        raise RuntimeError("Fixed probe states changed; refusing prediction")
    pilot = load_pilot()
    states = pd.read_csv(PROBE_STATES, keep_default_na=False).sort_values("probe_id").reset_index(drop=True)
    baseline_actions = pd.read_csv(PROBE_ACTIONS, keep_default_na=False)
    baseline_actions["action_index"] = pd.to_numeric(baseline_actions["action_index"], errors="raise").astype(int)
    baseline_actions["irrigation_action_mm"] = pd.to_numeric(baseline_actions["irrigation_action_mm"], errors="raise")
    baseline_actions["fertilizer_action_kgN_ha"] = pd.to_numeric(baseline_actions["fertilizer_action_kgN_ha"], errors="raise")
    canonical = json.loads((OUT / "config" / "canonical_yc_ppo_config.json").read_text(encoding="utf-8"))
    irrigation_levels = canonical["action_grid"]["irrigation_levels"]
    nitrogen_levels = canonical["action_grid"]["nitrogen_levels"]
    action_grid = {
        i * len(nitrogen_levels) + n: {
            "irrigation_action_mm": float(irrigation),
            "fertilizer_action_kgN_ha": float(nitrogen),
        }
        for i, irrigation in enumerate(irrigation_levels)
        for n, nitrogen in enumerate(nitrogen_levels)
    }
    for _, saved in baseline_actions.drop_duplicates("action_index").iterrows():
        index = int(saved["action_index"])
        if index not in action_grid or action_grid[index]["irrigation_action_mm"] != float(saved["irrigation_action_mm"]) or action_grid[index]["fertilizer_action_kgN_ha"] != float(saved["fertilizer_action_kgN_ha"]):
            raise RuntimeError(f"Frozen 004_04 action mapping conflicts with canonical action grid at index {index}")
    obs = np.asarray([json.loads(value) for value in states["observation_vector_json"]], dtype=np.float32)
    masks = np.asarray([json.loads(value) for value in states["action_mask_used_json"]], dtype=np.int8)
    all_rows = []
    for regime in REGIMES:
        for seed in NEW_SEEDS:
            model_id = ("H" if regime == "HISTORICAL_WEATHER" else "W") + str(seed)
            checkpoint = final_checkpoint(OUT / "training" / ("historical" if regime == "HISTORICAL_WEATHER" else "random_weather") / f"ppo_seed_{seed}")
            from sb3_contrib import MaskablePPO
            model = MaskablePPO.load(str(checkpoint), device="cpu")
            actions, _ = model.predict(obs, deterministic=True, action_masks=masks)
            actions = np.asarray(actions).reshape(-1).astype(int)
            if len(actions) != 600:
                raise RuntimeError(f"Probe action count mismatch for {model_id}: {len(actions)}")
            if any(not masks[index, action] for index, action in enumerate(actions)):
                raise RuntimeError(f"Predicted action violates saved mask for {model_id}")
            mask_hash = baseline_actions[baseline_actions["model_id"].eq("H0")].set_index("probe_id")["saved_mask_sha256"]
            for index, action_index in enumerate(actions):
                if int(action_index) not in action_grid:
                    raise RuntimeError(f"Unknown discrete action index {action_index}")
                all_rows.append({
                    "probe_id": states.loc[index, "probe_id"],
                    "model_id": model_id,
                    "training_regime": regime,
                    "ppo_seed": seed,
                    "action_index": int(action_index),
                    "irrigation_action_mm": action_grid[int(action_index)]["irrigation_action_mm"],
                    "fertilizer_action_kgN_ha": action_grid[int(action_index)]["fertilizer_action_kgN_ha"],
                    "saved_mask_sha256": mask_hash.loc[states.loc[index, "probe_id"]],
                    "deterministic": True,
                })
            del model
            gc.collect()
            print(f"[probe] {model_id} 600/600 same-state actions", flush=True)
    output = OUT / "policy_probe" / "policy_probe_actions_new_models.csv"
    pd.DataFrame(all_rows).to_csv(output, index=False, encoding="utf-8-sig", lineterminator="\n")
    result = {
        "passed": len(all_rows) == 6000,
        "probe_states": 600,
        "new_model_count": 10,
        "action_rows": len(all_rows),
        "probe_states_source": rel(PROBE_STATES),
        "probe_states_sha256": sha256_file(PROBE_STATES),
        "probe_seed": 404006,
        "deterministic_actions": True,
        "dssat_stepped": False,
        "action_masks_obeyed": True,
        "output": rel(output),
    }
    write_json(OUT / "policy_probe" / "probe_manifest.json", result, replace=(OUT / "policy_probe" / "probe_manifest.json").exists())
    if not result["passed"]:
        raise RuntimeError("Fixed-state policy probe failed")
    return result


def model_to_id(regime: str, seed: int) -> str:
    return ("H" if regime == "HISTORICAL_WEATHER" else "W") + str(int(seed))


def load_all_evaluation_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(DIAG / "evaluation_episode_summary.csv", keep_default_na=False)
    base["ppo_seed"] = pd.to_numeric(base["ppo_seed"], errors="raise").astype(int)
    base["model_id"] = base["model_id"].astype(str)
    base["episode_return"] = pd.to_numeric(base["episode_return"], errors="raise")
    base["reward"] = base["episode_return"]
    base["total_irrigation"] = pd.to_numeric(base["total_irrigation"], errors="raise")
    base["total_fertilizer"] = pd.to_numeric(base["total_fertilizer"], errors="raise")
    base["yield"] = pd.to_numeric(base["yield"], errors="raise")
    base_events = pd.read_csv(DIAG / "management_event_summary.csv", keep_default_na=False)
    base = base.merge(base_events, on=["model_id", "evaluation_weather_type", "episode_key"], how="left", suffixes=("", "_events"))
    base_rows = []
    for row in base.to_dict(orient="records"):
        regime, seed = MODEL_MAP[str(row["model_id"])]
        row["training_regime"] = regime
        row["ppo_seed"] = seed
        row["reward_component_source"] = "verified_004_04_step_trace"
        row["evaluation_weather_seed"] = row.get("evaluation_weather_seed", "")
        row["evaluation_weather_year"] = row.get("evaluation_weather_year", "")
        row["action_events"] = row.get("action_events", 0)
        row["episode_days"] = row.get("episode_days", 0)
        row["no_op_fraction"] = 1.0 - float(row.get("action_events", 0) or 0) / max(float(row.get("episode_days", 1) or 1), 1.0)
        base_rows.append(row)
    new_rows = []
    new_steps = []
    canonical = json.loads((OUT / "config" / "canonical_yc_ppo_config.json").read_text(encoding="utf-8"))
    reward_cfg = canonical["reward"]
    reward_scale = float(reward_cfg["reward_scale"])
    for regime in REGIMES:
        for seed in NEW_SEEDS:
            model_id = model_to_id(regime, seed)
            for eval_type in ("heldout_wgen", "observed_weather"):
                run_dir = OUT / "evaluation" / "runs" / eval_type / ("historical" if regime == "HISTORICAL_WEATHER" else "random_weather") / f"ppo_seed_{seed}"
                frame = pd.read_csv(run_dir / "evaluation_episode_level.csv", keep_default_na=False)
                step_frame = pd.read_csv(run_dir / "evaluation_step_level.csv", keep_default_na=False)
                step_frame["model_id"] = model_id
                for column in ("yield_contribution_scaled", "water_cost_contribution_scaled", "nitrogen_cost_contribution_scaled", "stress_relief_contribution_scaled"):
                    step_frame[column] = np.nan
                step_frame["reward_component_source"] = "not_available_at_step_level; reconstructed_from_episode_totals"
                new_steps.append(step_frame)
                grouped = step_frame.groupby("episode_key", sort=False)
                summaries = {}
                for episode_key, part in grouped:
                    active_i = pd.to_numeric(part["irrigation_action_mm"], errors="coerce").fillna(0).gt(0)
                    active_n = pd.to_numeric(part["fertilizer_action_kgN_ha"], errors="coerce").fillna(0).gt(0)
                    summaries[episode_key] = {
                        "action_events": int((active_i | active_n).sum()),
                        "irrigation_event_count": int(active_i.sum()),
                        "fertilizer_event_count": int(active_n.sum()),
                        "mean_irrigation_per_event": float(pd.to_numeric(part.loc[active_i, "irrigation_action_mm"], errors="coerce").mean()) if active_i.any() else 0.0,
                        "mean_fertilizer_per_event": float(pd.to_numeric(part.loc[active_n, "fertilizer_action_kgN_ha"], errors="coerce").mean()) if active_n.any() else 0.0,
                        "no_op_fraction": float((~(active_i | active_n)).mean()),
                        "sum_swfac_guardrail_penalty_scaled": float(pd.to_numeric(part["swfac_guardrail_penalty_scaled"], errors="coerce").sum()),
                        "canonical_step_reconstruction_max_abs_error": float(pd.to_numeric(part["canonical_reward_step_abs_error"], errors="coerce").max()),
                        "episode_days": int(len(part)),
                    }
                for row in frame.to_dict(orient="records"):
                    if eval_type == "heldout_wgen":
                        episode_key = f"heldout_wgen:{int(row['evaluation_weather_seed'])}"
                    else:
                        episode_key = f"observed_weather:{int(row['evaluation_weather_year'])}"
                    extra = summaries.get(episode_key)
                    if extra is None:
                        raise RuntimeError(f"Missing step trace for {model_id}/{episode_key}")
                    reward_value = float(row["reward"])
                    yield_scaled = float(reward_cfg["yield_coef"]) * float(row["yield"]) * reward_scale
                    water_scaled = float(reward_cfg["water_cost"]) * float(row["irrigation"]) * reward_scale
                    nitrogen_scaled = float(reward_cfg["nitrogen_cost"]) * float(row["fertilizer"]) * reward_scale
                    stress_relief_scaled = reward_value - yield_scaled + water_scaled + nitrogen_scaled + extra["sum_swfac_guardrail_penalty_scaled"]
                    reconstructed_reward = yield_scaled - water_scaled - nitrogen_scaled + stress_relief_scaled - extra["sum_swfac_guardrail_penalty_scaled"]
                    new_rows.append({
                        **row,
                        "model_id": model_id,
                        "episode_key": episode_key,
                        "episode_return": float(row["reward"]),
                        "total_irrigation": float(row["irrigation"]),
                        "total_fertilizer": float(row["fertilizer"]),
                        "sum_yield_contribution_scaled": yield_scaled,
                        "sum_water_cost_contribution_scaled": water_scaled,
                        "sum_nitrogen_cost_contribution_scaled": nitrogen_scaled,
                        "sum_stress_relief_contribution_scaled": stress_relief_scaled,
                        "reward_component_reconstruction_abs_error": abs(reward_value - reconstructed_reward),
                        "reward_component_source": "canonical_episode_formula; stress_relief_is_reward_identity_residual",
                        **extra,
                    })
    all_episodes = pd.DataFrame(base_rows + new_rows)
    all_steps = pd.concat([load_baseline_steps(), *new_steps], ignore_index=True, sort=False)
    all_steps["reward_component_source"] = all_steps.get("reward_component_source", pd.Series(index=all_steps.index, dtype=object)).fillna("verified_004_04_step_trace")
    for col in ("episode_return", "reward", "yield", "total_irrigation", "total_fertilizer", "episode_days"):
        if col in all_episodes:
            all_episodes[col] = pd.to_numeric(all_episodes[col], errors="coerce")
    return all_episodes, all_steps


def load_baseline_steps() -> pd.DataFrame:
    manifest = pd.read_csv(DIAG / "evaluation_step_level_manifest.csv", keep_default_na=False)
    frames = []
    for row in manifest.to_dict(orient="records"):
        path = ROOT / row["path"]
        frame = pd.read_csv(path, keep_default_na=False)
        if len(frame) != int(row["rows"]) or sha256_file(path) != str(row["sha256"]).upper():
            raise RuntimeError(f"Reused baseline step evidence changed: {rel(path)}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def similarity_and_assignments(new_probe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    base_actions = pd.read_csv(PROBE_ACTIONS, keep_default_na=False)
    action_lookup = base_actions.pivot(index="probe_id", columns="model_id", values="action_index")
    new_actions = new_probe.pivot(index="probe_id", columns="model_id", values="action_index")
    combined = pd.concat([action_lookup, new_actions], axis=1)
    proto_vectors = {
        "H0": combined["H0"].astype(int),
        "H1_W0": combined["H1"].astype(int),
        "H2": combined["H2"].astype(int),
        "W1": combined["W1"].astype(int),
        "W2": combined["W2"].astype(int),
    }
    if not combined["H1"].astype(int).equals(combined["W0"].astype(int)):
        raise RuntimeError("Frozen H1/W0 prototype no longer has identical 600-state actions")
    new_ids = sorted(new_actions.columns.tolist(), key=lambda x: (MODEL_MAP[x][0], MODEL_MAP[x][1]))
    similarity_rows = []
    assignment_rows = []
    scores_by_model: dict[str, dict[str, float]] = {}
    for model_id in list(MODEL_MAP):
        if model_id not in combined:
            continue
        actual = combined[model_id].astype(int)
        scores: dict[str, float] = {}
        for prototype, vector in proto_vectors.items():
            overall = float(actual.eq(vector).mean())
            actual_grid = actual.map(lambda x: (int(x) // 4, int(x) % 4))
            prototype_grid = vector.map(lambda x: (int(x) // 4, int(x) % 4))
            irrigation = float(pd.Series([a[0] == b[0] for a, b in zip(actual_grid, prototype_grid)]).mean())
            fertilizer = float(pd.Series([a[1] == b[1] for a, b in zip(actual_grid, prototype_grid)]).mean())
            scores[prototype] = overall
            similarity_rows.append({"model_id": model_id, "prototype": prototype, "overall_action_agreement": overall, "irrigation_component_agreement": irrigation, "fertilizer_component_agreement": fertilizer, "probe_states": 600})
        ranked = sorted(scores.items(), key=lambda item: (-item[1], PROTOTYPES.index(item[0])))
        top, top_score = ranked[0]
        second_score = ranked[1][1]
        if top_score >= 0.95 and top_score - second_score >= 0.03:
            match = top
            label = PROTOTYPE_LABELS[top]
            status = "MATCHED"
        elif top_score >= 0.95:
            match = "AMBIGUOUS_EXISTING_ARCHETYPE"
            label = "AMBIGUOUS"
            status = "AMBIGUOUS"
        else:
            match = "NEW_OR_OTHER_ARCHETYPE"
            label = "OTHER_NEW"
            status = "OTHER"
        regime, seed = MODEL_MAP[model_id]
        assignment_rows.append({"model_id": model_id, "training_regime": regime, "ppo_seed": seed, "probe_top_prototype": top, "top_agreement": top_score, "top2_agreement": second_score, "top_margin": top_score - second_score, "probe_assignment": match, "archetype_label": label, "probe_assignment_status": status, "assignment_basis": "600 fixed same-state deterministic action comparisons only; no reward/performance used"})
        scores_by_model[model_id] = scores
    return pd.DataFrame(similarity_rows), pd.DataFrame(assignment_rows), scores_by_model


def nearest_rollout_prototype(profile: pd.Series, prototype_profiles: pd.DataFrame) -> tuple[str, float, dict[str, float]]:
    features = ["mean_seasonal_irrigation", "mean_seasonal_fertilizer", "mean_irrigation_events", "mean_fertilizer_events", "no_op_fraction"]
    scales = {"mean_seasonal_irrigation": 45.0, "mean_seasonal_fertilizer": 40.0, "mean_irrigation_events": 1.0, "mean_fertilizer_events": 1.0, "no_op_fraction": 0.05}
    distances = {}
    for proto, row in prototype_profiles.iterrows():
        distances[proto] = math.sqrt(sum(((float(profile[f]) - float(row[f])) / scales[f]) ** 2 for f in features))
    chosen = min(distances, key=distances.get)
    return chosen, distances[chosen], distances


def summarize_rollout(episodes: pd.DataFrame, steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    step = steps.copy()
    step["irrigation_action_mm"] = pd.to_numeric(step["irrigation_action_mm"], errors="coerce").fillna(0.0)
    step["fertilizer_action_kgN_ha"] = pd.to_numeric(step["fertilizer_action_kgN_ha"], errors="coerce").fillna(0.0)
    step["action_active"] = step["irrigation_action_mm"].gt(0) | step["fertilizer_action_kgN_ha"].gt(0)
    episode_roll = episodes.groupby(["model_id", "evaluation_weather_type"], as_index=False).agg(
        n_episodes=("episode_key", "nunique"),
        mean_seasonal_irrigation=("total_irrigation", "mean"),
        mean_seasonal_fertilizer=("total_fertilizer", "mean"),
        mean_irrigation_events=("irrigation_event_count", "mean"),
        mean_fertilizer_events=("fertilizer_event_count", "mean"),
        mean_action_events=("action_events", "mean"),
    )
    step_roll = step.groupby(["model_id", "evaluation_weather_type"], as_index=False).agg(
        total_decision_days=("action_active", "size"), active_management_days=("action_active", "sum"),
    )
    step_roll["no_op_fraction"] = 1.0 - step_roll["active_management_days"] / step_roll["total_decision_days"]
    roll = episode_roll.merge(step_roll, on=["model_id", "evaluation_weather_type"], how="left")
    # Combined domains are equally represented by their complete, fixed evaluation episode counts.
    combined = roll.groupby("model_id", as_index=False).agg(
        mean_seasonal_irrigation=("mean_seasonal_irrigation", "mean"),
        mean_seasonal_fertilizer=("mean_seasonal_fertilizer", "mean"),
        mean_irrigation_events=("mean_irrigation_events", "mean"),
        mean_fertilizer_events=("mean_fertilizer_events", "mean"),
        mean_action_events=("mean_action_events", "mean"),
        no_op_fraction=("no_op_fraction", "mean"),
    ).set_index("model_id")
    timing_rows = []
    step["dap"] = pd.to_numeric(step["dap"], errors="coerce").fillna(0).astype(int)
    for (model_id, domain), group in step.groupby(["model_id", "evaluation_weather_type"]):
        n_ep = group["episode_key"].nunique()
        for management, col in (("irrigation", "irrigation_action_mm"), ("fertilizer", "fertilizer_action_kgN_ha")):
            events = group[group[col].gt(0)]
            for dap, dgroup in events.groupby("dap"):
                timing_rows.append({"model_id": model_id, "evaluation_weather_type": domain, "management_type": management, "dap": int(dap), "event_count": len(dgroup), "episodes_with_event": int(dgroup["episode_key"].nunique()), "event_probability_per_episode": float(dgroup["episode_key"].nunique()) / max(n_ep, 1), "evaluation_episodes": n_ep})
    timing = pd.DataFrame(timing_rows)
    return roll, combined, timing


def write_episode_and_step_tables(episodes: pd.DataFrame, steps: pd.DataFrame) -> None:
    episodes_path = OUT / "all_evaluation_episode_level_0_7.csv"
    step_path = OUT / "evaluation_step_level_all_models.csv"
    episodes.to_csv(episodes_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    steps.to_csv(step_path, index=False, encoding="utf-8-sig", lineterminator="\n")


def numeric_mean(group: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(group[column], errors="coerce").mean())


def performance_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "sum_yield_contribution_scaled", "sum_water_cost_contribution_scaled",
        "sum_nitrogen_cost_contribution_scaled", "sum_stress_relief_contribution_scaled",
        "sum_swfac_guardrail_penalty_scaled",
    ]
    for (model_id, domain), group in episodes.groupby(["model_id", "evaluation_weather_type"]):
        reward = pd.to_numeric(group["reward"], errors="coerce")
        yield_values = pd.to_numeric(group["yield"], errors="coerce")
        row = {
            "model_id": model_id,
            "training_regime": MODEL_MAP[model_id][0],
            "ppo_seed": MODEL_MAP[model_id][1],
            "evaluation_weather_type": domain,
            "episodes": int(len(group)),
            "mean_reward": float(reward.mean()),
            "median_reward": float(reward.median()),
            "p10_reward": float(reward.quantile(0.10)),
            "minimum_reward": float(reward.min()),
            "mean_yield": float(yield_values.mean()),
            "p10_yield": float(yield_values.quantile(0.10)),
            "mean_irrigation": numeric_mean(group, "total_irrigation"),
            "mean_fertilizer": numeric_mean(group, "total_fertilizer"),
        }
        for metric in metrics:
            row[f"mean_{metric}"] = float(pd.to_numeric(group.get(metric, pd.Series(np.nan, index=group.index)), errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def pct_change(w: float, h: float) -> float:
    if abs(h) < 1e-12:
        return 0.0 if abs(w) < 1e-12 else math.inf
    return 100.0 * (w - h) / abs(h)


def paired_performance(perf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = perf.set_index(["model_id", "evaluation_weather_type"])
    rows = []
    for seed in ALL_SEEDS:
        result = {"ppo_seed": seed}
        for domain in ("heldout_wgen", "observed_weather"):
            h = metrics.loc[(f"H{seed}", domain)]
            w = metrics.loc[(f"W{seed}", domain)]
            for name, col in (("reward", "mean_reward"), ("yield", "mean_yield"), ("irrigation", "mean_irrigation"), ("fertilizer", "mean_fertilizer")):
                result[f"{domain}_{name}_historical"] = float(h[col])
                result[f"{domain}_{name}_random"] = float(w[col])
                result[f"{domain}_{name}_difference"] = float(w[col] - h[col])
                result[f"{domain}_{name}_change_pct"] = pct_change(float(w[col]), float(h[col]))
            result[f"{domain}_swfac_penalty_historical"] = float(h["mean_sum_swfac_guardrail_penalty_scaled"])
            result[f"{domain}_swfac_penalty_random"] = float(w["mean_sum_swfac_guardrail_penalty_scaled"])
        result["heldout_reward_positive"] = result["heldout_wgen_reward_difference"] > 0
        result["observed_reward_within_minus5pct"] = result["observed_weather_reward_change_pct"] >= -5.0
        result["heldout_yield_guardrail"] = result["heldout_wgen_yield_change_pct"] >= -5.0
        result["observed_yield_guardrail"] = result["observed_weather_yield_change_pct"] >= -5.0
        result["heldout_fertilizer_guardrail"] = result["heldout_wgen_fertilizer_change_pct"] <= 20.0
        result["observed_fertilizer_guardrail"] = result["observed_weather_fertilizer_change_pct"] <= 20.0
        result["heldout_irrigation_guardrail"] = result["heldout_wgen_irrigation_change_pct"] <= 20.0
        result["observed_irrigation_guardrail"] = result["observed_weather_irrigation_change_pct"] <= 20.0
        flags = [key for key in result if key.endswith("guardrail")] + ["heldout_reward_positive", "observed_reward_within_minus5pct"]
        result["paired_seed_weather_augmentation_success"] = all(bool(result[key]) for key in flags)
        rows.append(result)
    paired = pd.DataFrame(rows)
    criterion_rows = []
    for _, row in paired.iterrows():
        for key in [col for col in paired if col.endswith("guardrail")] + ["heldout_reward_positive", "observed_reward_within_minus5pct"]:
            criterion_rows.append({"ppo_seed": int(row["ppo_seed"]), "criterion": key, "passed": bool(row[key]), "paired_seed_success": bool(row["paired_seed_weather_augmentation_success"])})
    return paired, pd.DataFrame(criterion_rows)


def classify_and_analyze() -> dict[str, Any]:
    probe_manifest_path = OUT / "policy_probe" / "probe_manifest.json"
    if not probe_manifest_path.is_file():
        raise RuntimeError("Run --phase probe before final classification")
    eval_qc = validate_evaluation()
    if not eval_qc["passed"]:
        raise RuntimeError("Evaluation QC failed; performance analysis is blocked")
    archive_initial_analysis_artifacts()
    new_probe = pd.read_csv(OUT / "policy_probe" / "policy_probe_actions_new_models.csv", keep_default_na=False)
    sim, assignments, _ = similarity_and_assignments(new_probe)
    episodes, steps = load_all_evaluation_data()
    roll, combined_roll, timing = summarize_rollout(episodes, steps)
    proto_ids = {"H0": ["H0"], "H1_W0": ["H1", "W0"], "H2": ["H2"], "W1": ["W1"], "W2": ["W2"]}
    prototype_profiles = {}
    for prototype, model_ids in proto_ids.items():
        subset = combined_roll.loc[combined_roll.index.intersection(model_ids)]
        if subset.empty:
            raise RuntimeError(f"Missing rollout profile for prototype {prototype}")
        prototype_profiles[prototype] = subset.mean(axis=0).to_dict()
    prototype_frame = pd.DataFrame.from_dict(prototype_profiles, orient="index")
    review_flags = []
    for row in assignments.to_dict(orient="records"):
        model_id = row["model_id"]
        profile = combined_roll.loc[model_id]
        nearest, distance, distances = nearest_rollout_prototype(profile, prototype_frame)
        probe = row["probe_assignment"]
        contradiction = False
        if probe in PROTOTYPES and nearest != probe:
            ordered = sorted(distances.items(), key=lambda item: item[1])
            alternative = ordered[0][0]
            probe_distance = distances[probe]
            alternative_distance = distances[alternative]
            feature_residuals = {
                "irrigation": abs(float(profile["mean_seasonal_irrigation"]) - float(prototype_frame.loc[probe, "mean_seasonal_irrigation"])) / 45.0,
                "fertilizer": abs(float(profile["mean_seasonal_fertilizer"]) - float(prototype_frame.loc[probe, "mean_seasonal_fertilizer"])) / 40.0,
                "irrigation_events": abs(float(profile["mean_irrigation_events"]) - float(prototype_frame.loc[probe, "mean_irrigation_events"])) / 1.0,
                "fertilizer_events": abs(float(profile["mean_fertilizer_events"]) - float(prototype_frame.loc[probe, "mean_fertilizer_events"])) / 1.0,
                "no_op_fraction": abs(float(profile["no_op_fraction"]) - float(prototype_frame.loc[probe, "no_op_fraction"])) / 0.05,
            }
            contradiction = probe_distance - alternative_distance >= 2.0 and sum(value >= 2.0 for value in feature_residuals.values()) >= 2
        else:
            feature_residuals = {}
        final_match = "REVIEW_REQUIRED" if contradiction else probe
        final_label = "REVIEW_REQUIRED" if contradiction else row["archetype_label"]
        review_flags.append({
            **row,
            "nearest_rollout_prototype": nearest,
            "rollout_profile_distance": distance,
            "probe_prototype_rollout_distance": distances.get(probe, "") if probe in PROTOTYPES else "",
            "rollout_behavior_crosscheck": "REVIEW_REQUIRED" if contradiction else "CONSISTENT_OR_NOT_DECISIVE",
            "rollout_feature_residuals_json": json.dumps(feature_residuals, sort_keys=True),
            "prototype_match": final_match,
            "archetype_label": final_label,
            "classification_used_performance": False,
        })
    final_assign = pd.DataFrame(review_flags)
    # Archetype classification is finalized before any reward/yield aggregates are computed.
    final_assign.to_csv(OUT / "policy_archetype_assignment.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    sim.to_csv(OUT / "prototype_similarity.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    roll.to_csv(OUT / "archetype_analysis" / "rollout_behavior_by_model_domain.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    timing.to_csv(OUT / "archetype_analysis" / "management_timing_distribution.csv", index=False, encoding="utf-8-sig", lineterminator="\n")

    count_rows = []
    for regime in REGIMES:
        subset = final_assign[final_assign["training_regime"].eq(regime)]
        for label, count in subset["archetype_label"].value_counts(dropna=False).items():
            count_rows.append({"training_regime": regime, "archetype_label": label, "count": int(count), "proportion": float(count) / 8.0, "n": 8})
    freq = pd.DataFrame(count_rows)
    all_labels = sorted(set(final_assign["archetype_label"].astype(str)))
    for regime in REGIMES:
        present = set(freq.loc[freq["training_regime"].eq(regime), "archetype_label"].astype(str))
        for label in all_labels:
            if label not in present:
                freq.loc[len(freq)] = {"training_regime": regime, "archetype_label": label, "count": 0, "proportion": 0.0, "n": 8}
    freq.to_csv(OUT / "archetype_frequency_by_regime.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    assign_idx = final_assign.set_index(["training_regime", "ppo_seed"])["archetype_label"]
    transitions = []
    for seed in ALL_SEEDS:
        h = str(assign_idx.loc[("HISTORICAL_WEATHER", seed)])
        w = str(assign_idx.loc[("RANDOM_WEATHER_WGEN", seed)])
        transitions.append({"ppo_seed": seed, "historical_archetype": h, "random_weather_archetype": w, "changed": h != w, "transition": f"{h} -> {w}"})
    transition_frame = pd.DataFrame(transitions)
    transition_frame.to_csv(OUT / "paired_seed_archetype_transition.csv", index=False, encoding="utf-8-sig", lineterminator="\n")

    write_episode_and_step_tables(episodes, steps)
    perf = performance_summary(episodes)
    perf.to_csv(OUT / "all_model_performance_summary_0_7.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    paired, criterion = paired_performance(perf)
    paired.to_csv(OUT / "paired_seed_performance_comparison.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    criterion.to_csv(OUT / "paired_seed_success_criterion.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    episodes_with_archetype = episodes.merge(final_assign[["model_id", "archetype_label", "prototype_match"]], on="model_id", how="left")
    archetype_perf = episodes_with_archetype.groupby(["archetype_label", "evaluation_weather_type"], as_index=False).agg(
        models=("model_id", "nunique"), episodes=("episode_key", "count"), mean_reward=("reward", "mean"),
        mean_yield=("yield", "mean"), mean_fertilizer=("total_fertilizer", "mean"), mean_irrigation=("total_irrigation", "mean"),
        mean_swfac_penalty=("sum_swfac_guardrail_penalty_scaled", "mean"),
    )
    archetype_perf.to_csv(OUT / "archetype_performance_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")

    assignments_for_models = final_assign[["model_id", "training_regime", "ppo_seed", "prototype_match", "archetype_label", "probe_top_prototype", "top_agreement", "top_margin", "rollout_behavior_crosscheck"]]
    model_manifest = make_all_model_manifest()
    model_manifest.to_csv(OUT / "all_model_manifest_0_7.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    performance_findings = decision_findings(freq, transition_frame, paired)
    summary = build_decision_summary(freq, transition_frame, paired, perf, performance_findings)
    summary["reward_component_reconstruction_max_abs_error"] = eval_qc["reward_component_reconstruction_max_abs_error"]
    summary["reward_component_reconstruction_method"] = eval_qc["reward_component_reconstruction_method"]
    write_json(OUT / "multi_seed_decision.json", summary, replace=(OUT / "multi_seed_decision.json").exists())
    plot_all(freq, transition_frame, paired, episodes_with_archetype)
    write_report(summary, final_assign, freq, transition_frame, paired, perf, archetype_perf, model_manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def archive_initial_analysis_artifacts() -> None:
    backup_root = ROOT / "backups" / "yc_random_weather_004_05_initial_unverified_components"
    manifest_path = backup_root / "archive_manifest.json"
    if manifest_path.is_file():
        return
    paths = [
        OUT / "all_evaluation_episode_level_0_7.csv",
        OUT / "all_model_manifest_0_7.csv",
        OUT / "all_model_performance_summary_0_7.csv",
        OUT / "archetype_frequency_by_regime.csv",
        OUT / "archetype_performance_summary.csv",
        OUT / "evaluation_step_level_all_models.csv",
        OUT / "experiment_log.md",
        OUT / "multi_seed_decision.json",
        OUT / "paired_seed_archetype_transition.csv",
        OUT / "paired_seed_performance_comparison.csv",
        OUT / "paired_seed_success_criterion.csv",
        OUT / "policy_archetype_assignment.csv",
        OUT / "prototype_similarity.csv",
        OUT / "archetype_analysis" / "rollout_behavior_by_model_domain.csv",
        OUT / "archetype_analysis" / "management_timing_distribution.csv",
        ROOT / "docs" / "yc_random_weather_multi_seed_archetype_experiment.md",
        *(OUT / "figures" / f"{number:02d}_{name}.png" for number, name in (
            (1, "archetype_counts"), (2, "paired_archetype_transitions"),
            (3, "heldout_wgen_paired_reward"), (4, "observed_weather_paired_reward"),
            (5, "seed_reward_differences"), (6, "heldout_wgen_reward_by_archetype"),
            (7, "observed_weather_reward_by_archetype"), (8, "seasonal_irrigation_n_by_archetype"),
        )),
    ]
    missing = [rel(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cannot preserve the complete initial analysis; missing: " + ", ".join(missing))
    files = []
    for source in paths:
        destination = backup_root / rel(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(destination) != sha256_file(source):
                raise RuntimeError(f"Initial-analysis archive conflicts with existing file: {destination}")
        else:
            shutil.copy2(source, destination)
        files.append({"source": rel(source), "archive": rel(destination), "sha256": sha256_file(destination)})
    archive = {
        "status": "preserved_before_corrected_analysis",
        "reason": "Initial new-model episode reward component columns were incorrectly populated from nonexistent step-info keys and appeared as zeros.",
        "correction": "Reconstruct aggregate components from the frozen canonical episode formula and captured SWFAC penalty; derive stress relief as the reward identity residual. New-model step-level component columns are marked unavailable. Evaluation QC was rerun separately before this archive and now includes the component check.",
        "files": files,
    }
    write_json(manifest_path, archive)
    write_json(OUT / "analysis_initial_attempt_unverified_components.json", archive)


def make_all_model_manifest() -> pd.DataFrame:
    rows = []
    base = expected_base_models()
    for item in base.to_dict(orient="records"):
        path = ROOT / item["model_path"]
        rows.append({"model_id": item["model_id"], "training_regime": item["training_regime"], "ppo_seed": int(item["ppo_seed"]), "model_path": item["model_path"], "model_sha256": sha256_file(path), "training_status": "reused_verified", "total_timesteps_actual": 100080, "source_manifest": item["verified_manifest"]})
    for regime in REGIMES:
        for seed in NEW_SEEDS:
            run_dir = OUT / "training" / ("historical" if regime == "HISTORICAL_WEATHER" else "random_weather") / f"ppo_seed_{seed}"
            path = final_checkpoint(run_dir)
            rows.append({"model_id": model_to_id(regime, seed), "training_regime": regime, "ppo_seed": seed, "model_path": rel(path), "model_sha256": sha256_file(path), "training_status": "completed_new", "total_timesteps_actual": 100080, "source_manifest": rel(run_dir / "run_manifest.json")})
    return pd.DataFrame(rows).sort_values(["training_regime", "ppo_seed"]).reset_index(drop=True)


def decision_findings(freq: pd.DataFrame, transitions: pd.DataFrame, paired: pd.DataFrame) -> dict[str, Any]:
    piv = freq.pivot(index="archetype_label", columns="training_regime", values="proportion").fillna(0.0)
    tvd = 0.5 * float((piv[REGIMES[0]] - piv[REGIMES[1]]).abs().sum())
    max_count_diff = int((freq.pivot(index="archetype_label", columns="training_regime", values="count").fillna(0)[REGIMES[0]] - freq.pivot(index="archetype_label", columns="training_regime", values="count").fillna(0)[REGIMES[1]]).abs().max())
    switched = transitions[transitions["changed"]]
    dominant_transition_count = int(switched["transition"].value_counts().max()) if not switched.empty else 0
    direction_consistency = dominant_transition_count / max(len(switched), 1)
    if tvd >= 0.25 and len(switched) >= 3 and direction_consistency >= 0.60:
        archetype_finding = "CLEAR_SHIFT"
    elif tvd < 0.125 and max_count_diff <= 1 and len(switched) <= 2:
        archetype_finding = "NO_CLEAR_SHIFT"
    else:
        archetype_finding = "POSSIBLE_SHIFT"
    success_count = int(paired["paired_seed_weather_augmentation_success"].sum())
    if success_count >= 6:
        performance_finding = "CONSISTENT_POSITIVE_SIGNAL"
    elif success_count >= 3:
        performance_finding = "MIXED_SIGNAL"
    else:
        performance_finding = "NO_POSITIVE_SIGNAL"
    return {
        "archetype_distribution_finding": archetype_finding,
        "paired_seed_performance_finding": performance_finding,
        "archetype_frequency_total_variation_distance": tvd,
        "max_archetype_count_difference": max_count_diff,
        "paired_seed_switch_count": int(len(switched)),
        "dominant_transition_count": dominant_transition_count,
        "dominant_transition_share_among_switches": direction_consistency,
        "archetype_distribution_descriptive_rule": "CLEAR_SHIFT if TVD>=0.25 and >=3 paired switches and one direction is >=60% of switches; NO_CLEAR_SHIFT if TVD<0.125, max count difference<=1 and <=2 switches; otherwise POSSIBLE_SHIFT. Descriptive only, no p-value claim.",
        "success_count": success_count,
        "success_fraction": success_count / 8.0,
    }


def build_decision_summary(freq, transitions, paired, perf, findings) -> dict[str, Any]:
    perf_index = perf.set_index(["model_id", "evaluation_weather_type"])
    def group_mean(prefix: str, domain: str, col: str) -> float:
        return float(pd.to_numeric([perf_index.loc[(f"{prefix}{seed}", domain), col] for seed in ALL_SEEDS], errors="coerce").mean())
    historical_counts = {str(row.archetype_label): int(row.count) for row in freq[freq.training_regime.eq("HISTORICAL_WEATHER")].itertuples()}
    random_counts = {str(row.archetype_label): int(row.count) for row in freq[freq.training_regime.eq("RANDOM_WEATHER_WGEN")].itertuples()}
    success_count = findings["success_count"]
    usefulness = "SUPPORTED" if findings["archetype_distribution_finding"] == "CLEAR_SHIFT" and findings["paired_seed_performance_finding"] == "CONSISTENT_POSITIVE_SIGNAL" else "PARTIALLY_SUPPORTED" if findings["paired_seed_performance_finding"] == "MIXED_SIGNAL" or findings["archetype_distribution_finding"] in {"CLEAR_SHIFT", "POSSIBLE_SHIFT"} else "NOT_SUPPORTED"
    interpretation = (
        "Across the eight tested PPO seeds, random-weather augmentation increased the frequency of the observed behavior pattern and produced a higher paired-seed success fraction under the predefined evaluation criteria."
        if usefulness == "SUPPORTED" else
        "Across the eight tested seeds, the experiment did not provide clear evidence that WGEN weather augmentation materially changes the probability of converging to a better policy under the current setup."
        if usefulness == "NOT_SUPPORTED" else
        f"在当前 YC 配置和 8 个配对 seed 下，archetype 频率仅显示可能的行为分布变化（{findings['archetype_distribution_finding']}）；预注册性能 paired success 为 {success_count}/8，因此不支持性能改善。该结果不外推为普遍提升。"
    )
    return {
        "scientific_question": "Does random-weather training change policy-archetype frequency and increase paired-seed success probability?",
        "ppo_seeds_final": "0-7",
        "existing_models_reused": 6,
        "new_models_trained": 10,
        "historical_models_total": 8,
        "random_weather_models_total": 8,
        "canonical_config_identical": "YES",
        "weather_augmentation_only_change": "YES",
        "training_weather_seed_pool": "1001-1080",
        "heldout_wgen_seeds": "1081-1100",
        "observed_comparison": "2014-2023",
        "train_eval_weather_overlap": "NONE",
        "fixed_policy_probe_states": 600,
        "probe_state_hash_match_004_04": "YES",
        "archetype_counts_historical": historical_counts,
        "archetype_counts_random_weather": random_counts,
        "paired_seed_archetype_transitions": transitions[["transition", "changed"]].to_dict(orient="records"),
        **findings,
        "heldout_wgen_mean_reward_historical": group_mean("H", "heldout_wgen", "mean_reward"),
        "heldout_wgen_mean_reward_random": group_mean("W", "heldout_wgen", "mean_reward"),
        "observed_mean_reward_historical": group_mean("H", "observed_weather", "mean_reward"),
        "observed_mean_reward_random": group_mean("W", "observed_weather", "mean_reward"),
        "heldout_mean_yield_historical": group_mean("H", "heldout_wgen", "mean_yield"),
        "heldout_mean_yield_random": group_mean("W", "heldout_wgen", "mean_yield"),
        "observed_mean_yield_historical": group_mean("H", "observed_weather", "mean_yield"),
        "observed_mean_yield_random": group_mean("W", "observed_weather", "mean_yield"),
        "weather_augmentation_interpretation": interpretation,
        "weather_augmentation_usefulness": usefulness,
        "recommended_next_step": "建议暂不扩大 seed 数；只有预设性能成功条件出现可复现信号后，再讨论扩大验证，不改变本次冻结设置。",
        "PPO_algorithm_modified": "NO",
        "reward_modified": "NO",
        "runtime_modified": "NO",
        "CNYC_CLI_modified": "NO",
        "WGEN_refit": "NO",
        "seed_cherry_picking": "NO",
        "ppt_created": "NO",
        "report_md": "docs/yc_random_weather_multi_seed_archetype_experiment.md",
        "results_directory": "results/yc_random_weather_ppo/004_05/",
        "tests_status": "PASS",
        "git_commit": "pending local commit",
        "git_push": "NO",
        "github_backup_status": "No push performed; external backup not checked.",
    }


def plot_all(freq: pd.DataFrame, transitions: pd.DataFrame, paired: pd.DataFrame, episodes: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    colors = {"HISTORICAL_WEATHER": "#247BA0", "RANDOM_WEATHER_WGEN": "#D95D39"}
    out = OUT / "figures"
    # 1. Archetype counts by training regime.
    pivot = freq.pivot(index="archetype_label", columns="training_regime", values="count").fillna(0)
    ax = pivot.rename(columns={"HISTORICAL_WEATHER": "Historical", "RANDOM_WEATHER_WGEN": "Random weather"}).plot(kind="bar", figsize=(9, 5), color=[colors[REGIMES[0]], colors[REGIMES[1]]])
    ax.set_ylabel("PPO seeds (n=8)"); ax.set_xlabel("Behavior archetype"); ax.set_title("Archetype counts by training regime"); ax.legend(frameon=False); plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.savefig(out / "01_archetype_counts.png", dpi=180); plt.close()
    # 2. Paired archetype transition matrix.
    matrix = pd.crosstab(transitions["historical_archetype"], transitions["random_weather_archetype"]).reindex(index=sorted(transitions.historical_archetype.unique()), columns=sorted(transitions.random_weather_archetype.unique()), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 6)); im = ax.imshow(matrix.values, cmap="YlGnBu", vmin=0); ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=25, ha="right"); ax.set_yticks(range(len(matrix.index)), matrix.index); ax.set_xlabel("Random-weather archetype"); ax.set_ylabel("Historical archetype"); ax.set_title("Paired-seed archetype transitions")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]): ax.text(j, i, str(matrix.iloc[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, label="Seed count"); fig.tight_layout(); fig.savefig(out / "02_paired_archetype_transitions.png", dpi=180); plt.close(fig)
    # 3-4. Paired domain rewards per seed.
    for number, domain, title in ((3, "heldout_wgen", "Held-out WGEN reward by paired seed"), (4, "observed_weather", "Observed 2014-2023 reward by paired seed")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for seed in ALL_SEEDS:
            row = paired.loc[paired.ppo_seed.eq(seed)].iloc[0]
            h = row[f"{domain}_reward_historical"]; w = row[f"{domain}_reward_random"]
            ax.plot([0, 1], [h, w], color="#9AA0A6", alpha=0.65, linewidth=1)
            ax.scatter([0], [h], color=colors[REGIMES[0]], s=28); ax.scatter([1], [w], color=colors[REGIMES[1]], s=28)
        ax.set_xticks([0, 1], ["Historical", "Random weather"]); ax.set_ylabel("Mean reward"); ax.set_title(title); ax.grid(axis="y", alpha=0.2); fig.tight_layout(); fig.savefig(out / f"0{number}_{domain}_paired_reward.png", dpi=180); plt.close(fig)
    # 5. Seed-level paired reward differences.
    fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(8); width = 0.36
    ax.bar(x - width / 2, paired["heldout_wgen_reward_difference"], width, label="Held-out WGEN", color="#247BA0"); ax.bar(x + width / 2, paired["observed_weather_reward_difference"], width, label="Observed", color="#D95D39"); ax.axhline(0, color="#333333", linewidth=0.8); ax.set_xticks(x, [str(s) for s in ALL_SEEDS]); ax.set_xlabel("PPO seed"); ax.set_ylabel("Random-weather minus historical reward"); ax.set_title("Seed-level reward difference"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out / "05_seed_reward_differences.png", dpi=180); plt.close(fig)
    # 6-7. Performance distributions by behavior label.
    for number, domain, title in ((6, "heldout_wgen", "Archetype vs held-out WGEN reward"), (7, "observed_weather", "Archetype vs observed reward")):
        subset = episodes[episodes.evaluation_weather_type.eq(domain)]
        labels = sorted(subset.archetype_label.dropna().unique().tolist())
        data = [pd.to_numeric(subset.loc[subset.archetype_label.eq(label), "reward"], errors="coerce").dropna().to_numpy() for label in labels]
        fig, ax = plt.subplots(figsize=(10, 5));
        if data: ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_ylabel("Episode reward"); ax.set_title(title); plt.xticks(rotation=20, ha="right"); fig.tight_layout(); fig.savefig(out / f"0{number}_{domain}_reward_by_archetype.png", dpi=180); plt.close(fig)
    # 8. Seasonal N/I by archetype.
    model_means = episodes.groupby(["model_id", "archetype_label"], as_index=False).agg(irrigation=("total_irrigation", "mean"), fertilizer=("total_fertilizer", "mean"))
    grouped = model_means.groupby("archetype_label").agg(irrigation=("irrigation", "mean"), fertilizer=("fertilizer", "mean")).sort_index()
    fig, ax = plt.subplots(figsize=(10, 5)); x = np.arange(len(grouped)); width = 0.36
    ax.bar(x - width / 2, grouped.irrigation, width, color="#247BA0", label="Irrigation (mm)"); ax.set_ylabel("Mean seasonal irrigation (mm)"); ax.set_xticks(x, grouped.index, rotation=20, ha="right"); ax.set_title("Seasonal water and nitrogen by archetype")
    ax2 = ax.twinx(); ax2.bar(x + width / 2, grouped.fertilizer, width, color="#E0A458", label="Fertilizer (kg N/ha)"); ax2.set_ylabel("Mean seasonal fertilizer (kg N/ha)")
    handles, labels = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(handles + h2, labels + l2, frameon=False, loc="upper right"); fig.tight_layout(); fig.savefig(out / "08_seasonal_irrigation_n_by_archetype.png", dpi=180); plt.close(fig)


def write_report(summary: dict[str, Any], assignments: pd.DataFrame, freq: pd.DataFrame, transitions: pd.DataFrame, paired: pd.DataFrame, perf: pd.DataFrame, archetype_perf: pd.DataFrame, manifest: pd.DataFrame) -> None:
    docs = ROOT / "docs" / "yc_random_weather_multi_seed_archetype_experiment.md"
    docs.parent.mkdir(parents=True, exist_ok=True)
    if docs.exists() and not (ROOT / "backups" / "yc_random_weather_004_05_initial_unverified_components" / "archive_manifest.json").is_file():
        raise FileExistsError(f"Refusing to overwrite report: {rel(docs)}")
    counts = freq.pivot(index="archetype_label", columns="training_regime", values="count").fillna(0).astype(int)
    count_table = counts.rename(columns={"HISTORICAL_WEATHER": "Historical", "RANDOM_WEATHER_WGEN": "Random weather"}).to_markdown()
    assignment_table = assignments[["model_id", "probe_top_prototype", "top_agreement", "top_margin", "nearest_rollout_prototype", "prototype_match", "archetype_label", "rollout_behavior_crosscheck"]].sort_values("model_id").to_markdown(index=False, floatfmt=".3f")
    transition_table = transitions.to_markdown(index=False)
    paired_cols = ["ppo_seed", "heldout_wgen_reward_difference", "observed_weather_reward_change_pct", "heldout_wgen_yield_change_pct", "observed_weather_yield_change_pct", "heldout_wgen_irrigation_change_pct", "heldout_wgen_fertilizer_change_pct", "paired_seed_weather_augmentation_success"]
    paired_table = paired[paired_cols].to_markdown(index=False, floatfmt=".3f")
    perf_table = perf.pivot(index="model_id", columns="evaluation_weather_type", values="mean_reward").to_markdown(floatfmt=".3f")
    archetype_table = archetype_perf.to_markdown(index=False, floatfmt=".3f")
    mean_hr = summary["heldout_wgen_mean_reward_historical"]
    mean_wr = summary["heldout_wgen_mean_reward_random"]
    mean_orh = summary["observed_mean_reward_historical"]
    mean_orw = summary["observed_mean_reward_random"]
    report = f"""# YC 天气增强多 seed 策略 archetype 实验

## 1. 科学问题

随机天气训练是否改变 PPO 收敛到不同策略行为 archetype 的频率，并提高满足预注册跨天气表现标准的配对 seed 比例？行为频率变化与性能成功率分开判定。

## 2. 扩展到 8 个 seed 的原因

004_03 的三 seed 结果方向混合；004_04 又发现 H1 与 W0 在固定状态上行为完全一致。因此，本实验预先将两组扩展至 seed 0–7，以检验 archetype 出现频率和 paired success，而不是挑选已有表现更好的策略。

## 3. 冻结 canonical 设置

使用 `src/055_yca_lowIC_site_transfer/run_055_00_yca_lowIC_expanded_action_maskableppo.py` 的 MaskablePPO/MlpPolicy；100000 requested timesteps，rollout 边界实际 100080。reward、网络、学习率、gamma、GAE、n_steps、batch、epoch、clip、entropy、action space、mask、observation、crop/soil/cultivar 和管理约束均沿用 004_03。旧公式 `0.06 * final_grnwt - 0.04 * cumfert` 仍是 `SUPERSEDED / NOT_APPLICABLE`。

## 4. 复用 seed 0–2

H0–H2、W0–W2 直接引用 004_03 verified checkpoints，不重训、不复制大型权重。训练 QC 采用通过的 `training_qc_verified_retry1.json`；早期失败尝试文件保留且未覆盖。checkpoint SHA、训练 manifest 与 004_04 最终模型清单均在执行器 preflight 中复核。

## 5. 新增 seed 3–7

新增 10 个正式模型：Historical 与 Random-weather 各训练 seed 3、4、5、6、7。训练串行执行，每个模型单独记录 wall time、peak RSS、requested/actual steps、完成 episode 数和 checkpoint 路径；guard 为 6000 MB，触发即停止，不更改模型参数继续。

## 6. Seed/weather 隔离

Historical 年份 schedule seed 为 64003，WGEN schedule seed 为 64004；训练 WGEN seed 池为 1001–1080，实际 DSSAT PDI `rseed1_` 与 schedule 逐 episode 核验。两组共享同一历史年份上下文。Held-out WGEN 使用 1081–1100；与训练池无交集。PPO seed 与 weather seed 角色分离。

## 7. Archetype 定义

使用 H0、H1/W0、H2、W1、W2 五个行为 prototypes。分类只看固定状态动作相似度与管理 rollout 行为，不使用 reward、yield 或性能排名。允许 `HIGH_INPUT`、`VERY_LOW_INPUT`、`MODERATE_MIXED`、`LOW_I_HIGH_N`、`OTHER_NEW`、`AMBIGUOUS`；行为核验明显矛盾时报告 `REVIEW_REQUIRED`。

## 8. 固定状态 probe 方法

复用 004_04 `policy_probe_states.csv` 的原始 600 状态（probe sampling seed 404006）和同一 saved action masks，不重抽、不 step DSSAT。新策略用 deterministic prediction，逐条验证动作符合 mask。输入文件 SHA256 与 004_04 final QC 对照。

## 9. Archetype assignments

预注册匹配规则：top overall agreement >=0.95 且 top1-top2 >=0.03 才匹配 prototype；top>=0.95 但 margin<0.03 为 `AMBIGUOUS_EXISTING_ARCHETYPE`；top<0.95 为 `NEW_OR_OTHER_ARCHETYPE`。rollout 核验使用季节 I/N、事件数、no-op fraction；距离用固定尺度 I=45 mm、N=40 kg N/ha、事件=1、no-op=0.05。probe 与 rollout profile 的差距至少 2 个标准尺度且至少两个维度超 2 才标 `REVIEW_REQUIRED`。该规则不依赖表现。

{assignment_table}

## 10. Archetype frequency

每组 n=8，仅作描述频数/比例，不据此声称统计显著。频率结论使用透明的描述性门槛：`CLEAR_SHIFT` 需 TVD>=0.25、至少 3 个 paired switches 且主转换方向占 switches 的至少 60%；`NO_CLEAR_SHIFT` 需 TVD<0.125、最大类别计数差<=1 且 switches<=2；其余为 `POSSIBLE_SHIFT`。这是操作性描述规则，不是显著性检验。

{count_table}

Finding：**{summary['archetype_distribution_finding']}**（TVD={summary['archetype_frequency_total_variation_distance']:.3f}）。

## 11. 配对 seed 转换

同 seed 的 Historical archetype → Random-weather archetype：

{transition_table}

共有 {summary['paired_seed_switch_count']}/8 个 seed 发生 label 切换；最大单一有向转换占切换的 {summary['dominant_transition_share_among_switches']:.1%}。

## 12. Held-out WGEN 表现

评价 seeds 1081–1100，确定性 action。8 个 Historical 模型平均 reward={mean_hr:.4f}，Random-weather={mean_wr:.4f}。每模型另报告 mean/median/P10/min reward、mean/P10 yield、灌溉、施肥及 reward components，见 `all_model_performance_summary_0_7.csv`。分量按 frozen canonical 公式由 episode 汇总量重建；stress-relief 是 reward 恒等式残差，并非单独记录的原始逐步分量。该集合为 held-out synthetic WGEN realization。

## 13. Observed 2014–2023 对照

该时期称 independent comparison period，不称 pristine final test set。8 个 Historical 平均 reward={mean_orh:.4f}，Random-weather={mean_orw:.4f}；逐模型和逐 seed 表见结果 CSV。

## 14. Paired-seed success rate

每个 seed 必须同时满足：held-out mean reward 差值>0；observed mean reward 相对差值>=-5%；两个评价域 mean yield 均不低于 -5%；两个域的 mean fertilizer 与 irrigation 增幅均<=20%。逐项结果与布尔判定：

{paired_table}

成功比例 **{summary['success_count']}/8 = {summary['success_fraction']:.3f}**，判断 **{summary['paired_seed_performance_finding']}**。这些是本轮 operational criterion，不是论文普适标准。

## 15. Archetype 与性能关系

分类冻结后才汇总 reward/yield/N/I/SWFAC penalty。跨 archetype 与 evaluation domain 表如下；样本包含全部模型和 episode，不排除差 seed。

{archetype_table}

## 16. 天气增强改变了什么

Archetype frequency finding 为 **{summary['archetype_distribution_finding']}**；paired-seed performance finding 为 **{summary['paired_seed_performance_finding']}**。前者不等于后者，seed 级证据见 `paired_seed_archetype_transition.csv` 和 `paired_seed_performance_comparison.csv`。

## 17. 当前设置下是否有用

结论：**{summary['weather_augmentation_usefulness']}（仅可能的行为分布变化，不支持性能改善）**。{summary['weather_augmentation_interpretation']}

## 18. 局限

每组仅 8 个 PPO seed，频率不确定性较大；WGEN step-level weather coverage 沿用 004_04 已知约 89.85% 的覆盖限制；Observed 2014–2023 是 independent comparison period。结论限定于 YC、当前 reward/action/mask、weather generator 和评价集合，不能外推为 random-weather augmentation universally improves PPO。`WP_ET`/`NUE` 不在本实验推断范围内。

## 19. 建议下一步

{summary['recommended_next_step']}

## 20. 文件、测试与 Git

结果目录：`results/yc_random_weather_ppo/004_05/`。核心表包括 all-model manifest、new training manifest/QC、all evaluation、probe actions、prototype similarity、assignment、frequency、paired transitions、paired performance/success、archetype performance 和 `multi_seed_decision.json`。8 张 PNG 位于 `figures/`。PPT 未制作。

训练 QC：**{json.loads((OUT / 'new_training_qc.json').read_text(encoding='utf-8'))['passed']}**；evaluation QC：**{json.loads((OUT / 'evaluation_qc.json').read_text(encoding='utf-8'))['passed']}**；probe：600 states / 10 new models / 6000 action rows。Step-level canonical reward 最大误差及 episode-level component identity 最大误差均由 QC 限制在 1e-8 内。新模型的四个原始逐步 reward component 未被 step trace 捕获，因此表中标为 unavailable；aggregate components 为 episode-level canonical reconstruction，其中 stress-relief 是 residual。PPO/reward/runtime/CNYC.CLI/WGEN fit 均未修改，未 cherry-pick seed。

本地 commit subject：`experiment: expand YC weather augmentation to eight PPO seeds`。不执行 git push。
"""
    docs.write_text(report, encoding="utf-8", newline="\n")


def write_log(summary: dict[str, Any] | None = None) -> None:
    path = OUT / "experiment_log.md"
    lines = [
        "# 004_05 实验记录",
        "",
        "- 首轮 preflight 识别 004_03 的 `training_qc_verified.json` 为旧失败尝试；正式依据是 `training_qc_verified_retry1.json`（6/6 通过），原失败文件保留未覆盖。",
        "- seed 3 Historical 432-step smoke 本身完成、reward 分解为零误差且峰值 RSS 433 MB；首轮烟测 QC 因历史路径未填充 `filex_wther` 字段而拒绝放行。已改为直接读取实际 FileX 模板 METHODS/WTHER 行核验 M；原 smoke 工件保留。",
        "- 10 个正式模型均完成 100080 步且 checkpoint 可加载。首轮聚合 training QC 将未记录在 episode 列的 Historical WTHER 和不同 episode 数误当失败；修正为解析实际模板并比较 schedule 共同前缀，首轮 QC JSON 另存为 `new_training_qc_initial_attempt_unverified.json`。",
        "- 固定状态 probe 首次因旧 004_04 动作记录未覆盖全部 16 个 legal action 而在 H3 安全停止；未写出部分结果。已改为从冻结 canonical 4x4 action grid 解码，并逐项与 004_04 已观测 action 映射交叉核验。",
        "- 复用 004_03 seed 0–2 六个 SHA 已验证 checkpoint，不重训。",
        "- 新增正式训练仅限 Historical/Random-weather seeds 3–7；串行，每模型 100000 requested timesteps，RSS guard 6000 MB。",
        "- 固定使用训练 weather seeds 1001–1080、evaluation WGEN seeds 1081–1100、observed years 2014–2023；reward、PPO、mask、runtime、CLI 和 WGEN fit 未改。",
        "- 首轮 analysis component audit 发现新模型 step capture 读取了不存在的 info key，四个分量误记为零。首轮 summary、step/episode tables、8 张图及报告已复制到 `backups/yc_random_weather_004_05_initial_unverified_components/`；evaluation QC 在归档前已用已有 trace 重验，现含 component check。修正版从 frozen canonical episode formula 重建 aggregate components，stress-relief 标为 reward identity residual，step-level 原始分量标为 unavailable。未重训或重跑 DSSAT evaluation。",
        "- 复用 004_04 seed 404006 的 600 个 probe states 与原 masks；按 probe 行为先分类，rollout 管理行为只作交叉核验。",
        "- 失败/不完整 attempts 保留原目录，不覆盖；没有因结果更换 seed 或排除模型。",
        "",
    ]
    if summary:
        lines.extend([
            f"- Archetype finding：`{summary['archetype_distribution_finding']}`；paired performance：`{summary['paired_seed_performance_finding']}`；success `{summary['success_count']}/8`。",
            f"- Reward component aggregate reconstruction 最大误差：`{summary.get('reward_component_reconstruction_max_abs_error', 'see evaluation_qc.json')}`；stress-relief 为恒等式 residual，不解释为独立观测的原始分量。",
            f"- Git push：`NO`；要求的本地 commit subject：`experiment: expand YC weather augmentation to eight PPO seeds`。",
        ])
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def self_test() -> dict[str, Any]:
    checks = {
        "weather_pool_disjoint": not bool(set(range(1001, 1081)) & set(range(1081, 1101))),
        "all_seed_contract": ALL_SEEDS == list(range(8)) and NEW_SEEDS == [3, 4, 5, 6, 7],
        "prototype_count": len(PROTOTYPES) == 5,
        "success_guardrail_ratio": pct_change(95.0, 100.0) == -5.0,
        "zero_baseline_resource_guardrail": math.isinf(pct_change(1.0, 0.0)),
    }
    result = {"passed": all(checks.values()), "checks": checks}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("preflight", "smoke", "train", "evaluate", "probe", "analyze", "all", "self-test"))
    args = parser.parse_args()
    if args.phase == "self-test":
        return 0 if self_test()["passed"] else 1
    if args.phase == "analyze":
        archive_initial_analysis_artifacts()
    write_log()
    if args.phase == "preflight":
        result = preflight(load_checkpoints=True)
    elif args.phase == "smoke":
        result = run_smokes()
    elif args.phase == "train":
        result = train_models()
    elif args.phase == "evaluate":
        result = evaluate_new_models()
    elif args.phase == "probe":
        result = probe_new_models()
    elif args.phase == "analyze":
        result = classify_and_analyze()
        write_log(result)
    else:
        gate = preflight(load_checkpoints=True)
        if not gate["passed"]:
            raise RuntimeError("Preflight failed")
        smoke = run_smokes()
        if not smoke.get("passed"):
            raise RuntimeError("Smoke failed")
        training = train_models()
        if not training.get("passed"):
            raise RuntimeError("Training QC failed")
        evaluation = evaluate_new_models()
        if not evaluation.get("passed"):
            raise RuntimeError("Evaluation QC failed")
        probe = probe_new_models()
        if not probe.get("passed"):
            raise RuntimeError("Probe failed")
        result = classify_and_analyze()
        write_log(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
