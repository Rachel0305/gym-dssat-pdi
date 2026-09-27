from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "yc_random_weather_ppo" / "004_17_yc_runtime_weather_recovery"
PILOT_PATH = ROOT / "results" / "yc_random_weather_ppo" / "004_03" / "run_controlled_pilot.py"
RUNNER_00405 = ROOT / "results" / "yc_random_weather_ppo" / "004_05" / "run_yc_weather_augmentation_multi_seed_archetype_004_05.py"
SCHEDULE = ROOT / "results" / "yc_random_weather_ppo" / "004_05" / "config" / "training_weather_schedule_random.csv"
TRAIN_LOG = ROOT / "results" / "yc_random_weather_ppo" / "004_05" / "training" / "random_weather" / "ppo_seed_5" / "training_episode_summary.csv"
EVAL_EPISODE_SUMMARY = ROOT / "results" / "yc_random_weather_ppo" / "004_05" / "all_evaluation_episode_level_0_7.csv"
STEP_TRACE = ROOT / "results" / "yc_random_weather_ppo" / "004_05" / "evaluation_step_level_all_models.csv"
CLI = ROOT / "results" / "yc_wgen_cli_pilot" / "003_06_05_02" / "final" / "CNYC.CLI"
FITTING_WEATHER = ROOT / "results" / "yc_weather_gapfill_finalize" / "yc_wgen_fitting_weather_2004_2013.csv"
EXPECTED_CLI_SHA256 = "65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929"
EXPECTED_FITTING_SHA256 = "4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34"
SMOKE_HISTORICAL_HASH = "13C121A6DFED924FAB7C38EC53DDCC6FB37A39B0B1D2578DDBE42D445D53FEF1"
WEATHER_SEEDS_TRAIN = range(1001, 1081)
WEATHER_SEEDS_HELDOUT = range(1081, 1101)
WEATHER_THRESHOLDS_C = (30.0, 32.0, 35.0)
STAGE_BINS = ((0, 30, "DAP0_30"), (31, 60, "DAP31_60"), (61, 90, "DAP61_90"), (91, 999, "DAP_gt90"))
MAX_EPISODE_STEPS = 500
RSS_LIMIT_MB = 2048.0


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n"
    if path.exists() and not replace:
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to overwrite existing 004_17 evidence: {path}")
        return
    path.write_text(content, encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing 004_17 CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    new_file = not path.exists()
    with path.open("a", encoding="utf-8-sig" if new_file else "utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")


def ensure_new_output_tree() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("provenance", "logs", "runtime_work", "recovered_weather", "canonical_weather", "coverage", "figures"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    (OUT / "runtime_work" / "tmp").mkdir(parents=True, exist_ok=True)


def source_chain_markdown() -> str:
    return """# 004_05 runtime weather generation chain (source audit)\n\n| Stage | Verified source | Location | Status / evidence |\n|---|---|---|---|\n| 004_05 runner | `results/yc_random_weather_ppo/004_05/run_yc_weather_augmentation_multi_seed_archetype_004_05.py` | imports `004_03/run_controlled_pilot.py`; training uses `ScheduledEpisodeEnv` | VERIFIED: no model training path is invoked by this recovery runner. |\n| Schedule / context | `results/yc_random_weather_ppo/004_05/config/training_weather_schedule_random.csv` | `episode_index`, `historical_year_context`, `weather_seed` | VERIFIED: seed1001 appears at episode 26 with context 2013; that exact pair is the smoke target. |\n| FileX WGEN mode | `results/yc_random_weather_ppo/004_03/run_controlled_pilot.py` | `set_wgen_filex_mode` (around line 395), `make_weather_env` (around line 434) | VERIFIED: task-local FileX method field is set to `W`; frozen `CNYC.CLI` is appended to runtime auxiliary files. |\n| RSEED1 assignment | same pilot file; runtime wrapper snapshot `results/yc_wgen_cli_pilot/003_06_07_01/runtime_source_snapshot/gym_dssat_pdi/envs/dssat_pdi.py` | `make_weather_env`; `_get_sockets_` launch hook; `DssatPdi.reset` | VERIFIED: `_rseed1` is set before DSSAT client launch; reset sends seed over PDI socket. Seed1001 is recorded in historical episode rows. |\n| Runtime invocation | historical `gym_dssat_pdi.envs.dssat_pdi.DssatPdi` source snapshot | `_write_pdi_yaml`, `_launch_client`, `_get_sockets_`, `_make_tmp_folder` | VERIFIED: runtime runs `/opt/dssat_pdi/run_dssat C fileX.MZX 1` from a temporary working directory. Runtime version is captured at smoke. |\n| WTH creation | `results/yc_wgen_cli_pilot/003_06_05_02/runtime/seed_105/wgen_status.json` and `runtime/runtime_snapshot/` | `_snapshot_runtime_files` in `scripts/run_yc_wgen_seed_pilot.py` scans `.WTH` and DSSAT files | VERIFIED LIMIT: historical pilot captured 118 daily weather states and snapshot-listed files, but no `.WTH`; its runtime directory only exposed runtime states, not a raw WGEN WTH. 004_05 rendered-input WTHs are copied observed source files and are not WGEN realizations. |\n| Daily weather capture | `results/yc_random_weather_ppo/004_03/run_controlled_pilot.py` | `make_weather_env` wraps `_get_state`; `_runtime_weather_hash` around lines 639–646 | VERIFIED: captures `RAIN/SRAD/TMAX/TMIN` from daily DSSAT PDI state. |\n| `runtime_weather_sha256` | `scripts/run_yc_wgen_seed_pilot.py` | `daily_weather_from_states` (around line 129), `canonical_weather_bytes` (around line 210); caller at 004_03 lines 639–646 and record at ~684 | VERIFIED: SHA256 of canonical state-derived CSV fields `DATE,DOY,RAIN,SRAD,TMAX,TMIN`, sorted by date/DOY, numeric `.12g`, UTF-8 with LF and trailing newline. It is not a raw WTH byte hash. The historical fallback date is 2008-06-01 plus state index when state date/DOY is absent. |\n| DSSAT consumption / cleanup | runtime snapshot `dssat_pdi.py` | `cwd=_tmp_folder`; `close()` calls `_close_tmp_folder()`; default `shutil.rmtree` | VERIFIED: runtime files are temporary and removed on close. The new runner relocates temp folders under 004_17 and archives any changed/new WTH before close. |\n\n## Historical anchor\n\nFrozen CLI SHA256: `65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929`. Frozen fitting-weather CSV SHA256: `4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34`. Historical seed1001/context2013 episode rows in PPO seeds 3–7 share canonical runtime hash `13C121A6DFED924FAB7C38EC53DDCC6FB37A39B0B1D2578DDBE42D445D53FEF1`; this is the smoke verification target.\n\n`crop-year context` identifies the 004_05 schedule input used to build the runtime FileX/planting context. It is not inferred from the weather-seed number. Runtime crop outputs are incidental side effects and are excluded from weather and performance claims.\n"""


def load_pilot():
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    pilot_dir = str(PILOT_PATH.parent)
    if pilot_dir not in sys.path:
        sys.path.insert(0, pilot_dir)
    import run_controlled_pilot as pilot

    pilot.TASK_ROOT = OUT
    pilot.CONFIG_ROOT = OUT / "runtime_config"
    pilot.CANONICAL_CONTEXT = None
    return pilot


def historical_hashes() -> dict[tuple[int, int], set[str]]:
    found: dict[tuple[int, int], set[str]] = {}
    training_dir = ROOT / "results" / "yc_random_weather_ppo" / "004_05" / "training" / "random_weather"
    for log_path in sorted(training_dir.glob("ppo_seed_*/training_episode_summary.csv")):
        for row in pd.read_csv(log_path, usecols=["training_weather_seed", "historical_year", "runtime_weather_sha256"], dtype=str).to_dict("records"):
            try:
                key = (int(row["training_weather_seed"]), int(float(row["historical_year"])))
            except (TypeError, ValueError):
                continue
            value = str(row.get("runtime_weather_sha256", "")).strip().upper()
            if value:
                found.setdefault(key, set()).add(value)
    if EVAL_EPISODE_SUMMARY.exists():
        required = ["ppo_seed", "evaluation_weather_type", "evaluation_weather_seed", "crop_year_context", "runtime_weather_sha256"]
        for row in pd.read_csv(EVAL_EPISODE_SUMMARY, usecols=required, dtype=str).to_dict("records"):
            if row.get("ppo_seed") != "5" or "heldout_wgen" not in str(row.get("evaluation_weather_type", "")).lower():
                continue
            try:
                key = (int(float(row["evaluation_weather_seed"])), int(float(row["crop_year_context"])))
            except (TypeError, ValueError):
                continue
            value = str(row.get("runtime_weather_sha256", "")).strip().upper()
            if value:
                found.setdefault(key, set()).add(value)
    if STEP_TRACE.exists():
        columns = ["ppo_seed", "evaluation_weather_type", "evaluation_weather_seed", "historical_year_context", "runtime_weather_sha256"]
        for chunk in pd.read_csv(STEP_TRACE, usecols=columns, chunksize=100_000, dtype=str):
            subset = chunk[(chunk["ppo_seed"] == "5") & chunk["evaluation_weather_type"].astype(str).str.contains("heldout_wgen", case=False, na=False)]
            for row in subset.drop_duplicates(["evaluation_weather_seed", "historical_year_context", "runtime_weather_sha256"]).to_dict("records"):
                try:
                    key = (int(float(row["evaluation_weather_seed"])), int(float(row["historical_year_context"])))
                except (TypeError, ValueError):
                    continue
                value = str(row.get("runtime_weather_sha256", "")).strip().upper()
                if value:
                    found.setdefault(key, set()).add(value)
    return found


def reconcile_existing_ledger() -> int:
    source = OUT / "hash_verification_context_replay_complete.csv"
    if not source.is_file():
        raise FileNotFoundError(f"No completed context-replay ledger to reconcile: {source}")
    target = OUT / "hash_verification_context_replay_reconciled.csv"
    records = pd.read_csv(source, dtype=str).fillna("").to_dict("records")
    historical = historical_hashes()
    for row in records:
        try:
            key = (int(row["weather_seed"]), int(float(row["crop_year"])))
        except (TypeError, ValueError):
            key = (-1, -1)
        candidates = historical.get(key, set())
        generated = str(row.get("generated_runtime_hash", "")).strip().upper()
        matched = bool(generated and generated in candidates)
        row["historical_hashes"] = ";".join(sorted(candidates))
        row["hash_match"] = str(matched)
        row["verification_status"] = "EXACT_MATCH" if matched else ("MISMATCH" if candidates and generated else "UNVERIFIABLE")
    if target.exists():
        saved = pd.read_csv(target, dtype=str).fillna("").to_dict("records")
        if saved != records:
            raise FileExistsError(f"Existing reconciled ledger differs; refusing to overwrite: {target}")
    else:
        write_csv(target, records)
    archive_manifest_path = OUT / "weather_archive_manifest.csv"
    archive_rows = pd.read_csv(archive_manifest_path, dtype=str).fillna("").to_dict("records") if archive_manifest_path.is_file() else []
    archive_by_episode = {row.get("episode_id", ""): row for row in archive_rows}
    verified_manifest_rows = []
    for row in records:
        archive = archive_by_episode.get(row.get("episode_id", ""), {})
        verified_manifest_rows.append({
            "episode_id": row.get("episode_id", ""),
            "archive_weather_id": archive.get("archive_weather_id", ""),
            "weather_seed": row.get("weather_seed", ""),
            "rseed1": row.get("rseed1", ""),
            "set": row.get("set", ""),
            "crop_year": row.get("crop_year", ""),
            "season_start": archive.get("season_start", ""),
            "station": archive.get("station", "YCA"),
            "raw_wth_path": archive.get("raw_wth_path", ""),
            "raw_wth_sha256": archive.get("raw_wth_sha256", ""),
            "raw_wth_status": row.get("raw_wth_status", archive.get("raw_wth_status", "")),
            "canonical_csv_path": row.get("canonical_csv_path", ""),
            "canonical_series_sha256": row.get("canonical_series_sha256", ""),
            "historical_runtime_hash": row.get("historical_hashes", ""),
            "generated_runtime_hash": row.get("generated_runtime_hash", ""),
            "hash_match_status": "MATCH" if row.get("verification_status") == "EXACT_MATCH" else row.get("verification_status", "UNVERIFIABLE"),
            "verification_level": row.get("verification_status", "UNVERIFIABLE"),
            "generator_version": archive.get("generator_version", ""),
            "runtime_version": archive.get("runtime_version", ""),
            "cli_sha256": archive.get("cli_sha256", ""),
            "source_runtime_path": archive.get("source_runtime_path", ""),
        })
    write_csv(OUT / "weather_archive_manifest_verified.csv", verified_manifest_rows)
    grouped: dict[str, dict[str, int]] = {}
    for row in records:
        group = row.get("set", "unknown")
        status = row.get("verification_status", "unknown")
        grouped.setdefault(group, {})[status] = grouped.setdefault(group, {}).get(status, 0) + 1
    unresolved = [
        {"weather_seed": row.get("weather_seed"), "crop_year": row.get("crop_year"), "verification_status": row.get("verification_status"),
         "generated_runtime_hash": row.get("generated_runtime_hash"), "historical_hashes": row.get("historical_hashes"), "daily_rows": row.get("daily_rows")}
        for row in records if row.get("verification_status") != "EXACT_MATCH"
    ]
    reconciliation = {
        "source_ledger": source.name,
        "reconciled_ledger": target.name,
        "verified_archive_manifest": "weather_archive_manifest_verified.csv",
        "historical_training_sources": "all available 004_05 random_weather PPO training_episode_summary.csv files",
        "historical_heldout_source": EVAL_EPISODE_SUMMARY.relative_to(ROOT).as_posix(),
        "heldout_model_seed_used_for_reference": 5,
        "counts_by_set_and_status": grouped,
        "unresolved_rows": unresolved,
        "formal_coverage_run": False,
        "coverage_reason": "Two training weather seed/context pairs did not match historical hashes; daily historical weather snapshots are unavailable for canonical row-level comparison.",
        "weather_recovery_status": "PARTIALLY_VERIFIED",
        "observed_training_coverage": "INSUFFICIENT",
        "weather_tail_gap": "INSUFFICIENT",
        "next_weather_action": "INSUFFICIENT",
    }
    write_json(OUT / "provenance" / "hash_verification_reconciliation.json", reconciliation, replace=True)
    decision_path = OUT / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.exists() else {}
    exact_count = sum(row.get("verification_status") == "EXACT_MATCH" for row in records)
    decision.update({
        "batch_count": len(records),
        "batch_exact_count": exact_count,
        "weather_recovery_status": "PARTIALLY_VERIFIED",
        "observed_training_coverage": "INSUFFICIENT",
        "weather_tail_gap": "INSUFFICIENT",
        "next_weather_action": "INSUFFICIENT",
        "coverage": {"coverage_status": "INSUFFICIENT", "reason": reconciliation["coverage_reason"], "unverified_rows": unresolved},
        "historical_hash_reconciliation": reconciliation,
        "authoritative_hash_ledger": target.name,
    })
    write_json(decision_path, decision, replace=True)
    smoke_records = [row for row in records if str(row.get("weather_seed")) == "1001" and str(row.get("crop_year")) == "2013"]
    write_report(decision, source_chain_markdown(), smoke_records, records, decision["coverage"], replace=True)
    return 0


def schedule_rows() -> list[dict[str, int]]:
    rows = []
    with SCHEDULE.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append({
                "episode_index": int(row["episode_index"]),
                "historical_year": int(row["historical_year_context"]),
                "weather_seed": int(row["weather_seed"]),
            })
    return rows


def process_rss_mb() -> float:
    try:
        import psutil

        process = psutil.Process()
        return sum(p.memory_info().rss for p in [process, *process.children(recursive=True)] if p.is_running()) / (1024 * 1024)
    except Exception:
        return float("nan")


def _scan_for_changed_wth(temp_folder: Path, baseline: dict[str, str], capture_dir: Path, episode_id: str) -> list[Path]:
    copied: list[Path] = []
    if not temp_folder.exists():
        return copied
    for path in sorted(temp_folder.rglob("*")):
        if not path.is_file() or path.suffix.upper() != ".WTH":
            continue
        digest = sha256_file(path)
        rel = path.relative_to(temp_folder).as_posix()
        if baseline.get(rel) == digest:
            continue
        target = capture_dir / episode_id / f"{digest[:12]}_{path.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)
        copied.append(target)
    return copied


def run_runtime_capture(
    pilot,
    weather_seed: int,
    crop_year: int,
    weather_set: str,
    episode_id: str,
    *,
    bootstrap_seed: int | None = None,
    predecessor_seeds: tuple[int, ...] = (),
) -> dict[str, Any]:
    import tempfile as tempfile_module

    tmp_base = OUT / "runtime_work" / "tmp"
    tmp_base.mkdir(parents=True, exist_ok=True)
    tempfile_module.tempdir = str(tmp_base)
    runner, engine, batch, rendering, config, split, selection, env_config = pilot._import_canonical()
    year_row = batch.base.direct_ppo.find_year(env_config, "YCA", int(crop_year))
    runtime_seed = int(pilot.RUNTIME_BOOTSTRAP_SEED)
    run_tag = f"00417_{weather_set}_{int(weather_seed)}_{int(crop_year)}"
    process_bootstrap_seed = int(bootstrap_seed if bootstrap_seed is not None else weather_seed)
    env = pilot.make_weather_env(config, env_config, int(crop_year), "RANDOM_WEATHER_WGEN", runtime_seed, run_tag, process_bootstrap_seed)
    dssat = pilot.find_dssat_instance(env)
    tmp_folder = Path(dssat._tmp_folder).resolve()
    baseline = {}
    for path in tmp_folder.rglob("*.WTH"):
        if path.is_file():
            baseline[path.relative_to(tmp_folder).as_posix()] = sha256_file(path)
    capture_dir = OUT / "runtime_wth_capture"
    changed_wth: list[Path] = []
    original_get_state = dssat._get_state

    def capture_state(*args, **kwargs):
        result = original_get_state(*args, **kwargs)
        if capturing_target[0]:
            changed_wth.extend(_scan_for_changed_wth(tmp_folder, baseline, capture_dir, episode_id))
        return result

    dssat._get_state = capture_state
    def run_noop_episode(seed: int, capture_target: bool) -> tuple[int, list[dict[str, Any]]]:
        if capture_target:
            capturing_target[0] = True
            dssat._pilot_weather_states = []
        else:
            capturing_target[0] = False
            dssat._pilot_weather_states = []
        dssat._random_generator = pilot.SequenceWeatherRng(int(seed))
        reset_result = env.reset(seed=None)
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            _obs = reset_result[0]
        else:
            _obs = reset_result
        actual = int(dssat._rseed1)
        if actual != int(seed):
            raise RuntimeError(f"RSEED1 mismatch: expected {seed}, got {actual}")
        count = 0
        ended_here = False
        while count < MAX_EPISODE_STEPS:
            step_result = env.step(0)
            count += 1
            if not isinstance(step_result, tuple):
                raise RuntimeError(f"Unexpected environment step response: {type(step_result)}")
            if len(step_result) == 5:
                _, _, terminated, truncated, _ = step_result
                ended_here = bool(terminated or truncated)
            elif len(step_result) == 4:
                _, _, done, _ = step_result
                ended_here = bool(done)
            else:
                raise RuntimeError(f"Unexpected environment step tuple length: {len(step_result)}")
            if capture_target:
                changed_wth.extend(_scan_for_changed_wth(tmp_folder, baseline, capture_dir, episode_id))
            if ended_here:
                break
        if not ended_here:
            raise RuntimeError(f"Weather capture exceeded safety limit of {MAX_EPISODE_STEPS} steps")
        return count, list(getattr(dssat, "_pilot_weather_states", []))

    capturing_target = [False]
    try:
        for prior_seed in predecessor_seeds:
            run_noop_episode(int(prior_seed), False)
        baseline.clear()
        for path in tmp_folder.rglob("*.WTH"):
            if path.is_file():
                baseline[path.relative_to(tmp_folder).as_posix()] = sha256_file(path)
        steps, states = run_noop_episode(int(weather_seed), True)
        actual_seed = int(dssat._rseed1)
        from scripts.run_yc_wgen_seed_pilot import canonical_weather_bytes, daily_weather_from_states, sha256_bytes as legacy_sha256

        legacy_rows = daily_weather_from_states(states)
        if len(legacy_rows) != len(states):
            raise RuntimeError(f"State-to-weather row mismatch: states={len(states)} rows={len(legacy_rows)}")
        enriched = []
        for state, row in zip(states, legacy_rows):
            dap = state.get("dap") if isinstance(state, dict) else None
            enriched.append({
                "DATE": row.get("DATE"),
                "DOY": row.get("DOY"),
                "DAP": dap,
                "RAIN": row.get("RAIN"),
                "SRAD": row.get("SRAD"),
                "TMAX": row.get("TMAX"),
                "TMIN": row.get("TMIN"),
            })
        runtime_hash = legacy_sha256(canonical_weather_bytes(legacy_rows)) if legacy_rows else ""
        raw_wth = changed_wth[-1] if changed_wth else None
        runtime_log = OUT / "logs" / "runtime" / f"{episode_id}.log"
        runtime_log.parent.mkdir(parents=True, exist_ok=True)
        canonical_set = "training" if weather_set == "training" else "heldout"
        from archive_runtime_weather import archive_runtime_weather

        archive = archive_runtime_weather(
            OUT,
            weather_seed=int(weather_seed),
            rseed1=int(getattr(dssat, "_rseed1", weather_seed)),
            weather_set=canonical_set,
            crop_year=int(crop_year),
            season_start=str(year_row["planting_date"]),
            station="YCA",
            episode_id=episode_id,
            rows=enriched,
            raw_wth_path=raw_wth,
            legacy_runtime_hash_sha256=runtime_hash,
            cli_sha256=sha256_file(CLI),
            generator_version="DSSAT runtime WTHER=W; version recorded below",
            runtime_version=importlib.metadata.version("gym-dssat-pdi"),
            source_runtime_path=str(getattr(dssat, "_run_dssat_location", "/opt/dssat_pdi/run_dssat")),
        )
        csv_path = OUT / archive["canonical_csv_path"]
        metadata = {
            "episode_id": episode_id,
            "weather_seed": int(weather_seed),
            "rseed1": actual_seed,
            "crop_year": int(crop_year),
            "season_start": str(year_row["planting_date"]),
            "station": "YCA",
            "cli_path": str(CLI),
            "cli_sha256": sha256_file(CLI),
            "fitting_weather_csv": str(FITTING_WEATHER),
            "fitting_weather_csv_sha256": sha256_file(FITTING_WEATHER),
            "generator_mode": "DSSAT runtime FileX WTHER=W through frozen CNYC.CLI",
            "generator_version": "runtime WGEN version not separately exposed by CLI",
            "gym_dssat_pdi_version": importlib.metadata.version("gym-dssat-pdi"),
            "runtime_binary_sha256": sha256_file(Path("/opt/dssat_pdi/run_dssat")) if Path("/opt/dssat_pdi/run_dssat").exists() else "UNAVAILABLE",
            "runtime_source_path": str(PILOT_PATH),
            "original_temporary_working_directory": str(tmp_folder),
            "raw_wth_capture_path": str(raw_wth) if raw_wth else "",
            "raw_wth_sha256": sha256_file(raw_wth) if raw_wth else "",
            "raw_wth_status": "COPIED_RUNTIME_WTH" if raw_wth else "NOT_EMITTED_OR_NOT_CHANGED_FROM_INPUT",
            "canonical_csv_path": str(csv_path),
            "canonical_series_sha256": archive["canonical_series_sha256"],
            "legacy_runtime_weather_sha256": runtime_hash,
            "captured_daily_rows": len(enriched),
            "step_count": steps,
            "runtime_arguments": {"run_tag": run_tag, "mode": "all", "evaluation": False, "linked_management": True, "action_index": 0, "action_semantics": "no irrigation/no nitrogen; weather-only capture, not policy evaluation", "process_bootstrap_rseed1": process_bootstrap_seed, "preceding_same_year_rseed1_resets": list(predecessor_seeds)},
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "side_effect_classification": "NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT",
            "crop_outcomes_used": False,
            "runtime_log": str(getattr(dssat, "log_saving_path", runtime_log)),
        }
        return {"metadata": metadata, "archive": archive, "rows": enriched, "legacy_rows": legacy_rows, "historical_runtime_hash": "", "rss_mb": process_rss_mb()}
    finally:
        try:
            env.close()
        except Exception:
            pass
        gc.collect()


def run_context_group(pilot, crop_year: int, scheduled_rows: list[dict[str, int]], target_seeds: set[int], weather_set: str) -> list[dict[str, Any]]:
    import tempfile as tempfile_module

    if not scheduled_rows:
        return []
    scheduled_rows = sorted(scheduled_rows, key=lambda row: row["episode_index"])
    tmp_base = OUT / "runtime_work" / "tmp"
    tmp_base.mkdir(parents=True, exist_ok=True)
    tempfile_module.tempdir = str(tmp_base)
    runner, engine, batch, rendering, config, split, selection, env_config = pilot._import_canonical()
    year_row = batch.base.direct_ppo.find_year(env_config, "YCA", int(crop_year))
    bootstrap_seed = int(scheduled_rows[0]["weather_seed"])
    run_tag = f"00417_{weather_set}_context_replay_{int(crop_year)}"
    env = pilot.make_weather_env(config, env_config, int(crop_year), "RANDOM_WEATHER_WGEN", int(pilot.RUNTIME_BOOTSTRAP_SEED), run_tag, bootstrap_seed)
    dssat = pilot.find_dssat_instance(env)
    tmp_folder = Path(dssat._tmp_folder).resolve()
    baseline: dict[str, str] = {}
    for path in tmp_folder.rglob("*.WTH"):
        if path.is_file():
            baseline[path.relative_to(tmp_folder).as_posix()] = sha256_file(path)
    active_episode = [""]
    active_capture = [False]
    copied_for_episode: list[Path] = []
    original_get_state = dssat._get_state

    def capture_state(*args, **kwargs):
        result = original_get_state(*args, **kwargs)
        if active_capture[0]:
            copied_for_episode.extend(_scan_for_changed_wth(tmp_folder, baseline, OUT / "runtime_wth_capture", active_episode[0]))
        return result

    dssat._get_state = capture_state
    captures: list[dict[str, Any]] = []
    from scripts.run_yc_wgen_seed_pilot import canonical_weather_bytes, daily_weather_from_states, sha256_bytes as legacy_sha256
    from archive_runtime_weather import archive_runtime_weather

    try:
        for row in scheduled_rows:
            seed = int(row["weather_seed"])
            episode_id = f"{weather_set}_context_replay_seed_{seed}_ctx_{crop_year}_ep_{row['episode_index']}"
            active_episode[0] = episode_id
            active_capture[0] = seed in target_seeds
            copied_for_episode = []
            baseline.clear()
            for path in tmp_folder.rglob("*.WTH"):
                if path.is_file():
                    baseline[path.relative_to(tmp_folder).as_posix()] = sha256_file(path)
            dssat._random_generator = pilot.SequenceWeatherRng(seed)
            dssat._pilot_weather_states = []
            env.reset(seed=None)
            actual_seed = int(dssat._rseed1)
            if actual_seed != seed:
                raise RuntimeError(f"Context replay RSEED1 mismatch: schedule={seed}, runtime={actual_seed}")
            steps = 0
            ended = False
            while steps < MAX_EPISODE_STEPS:
                step_result = env.step(0)
                steps += 1
                if len(step_result) == 5:
                    _, _, terminated, truncated, _ = step_result
                    ended = bool(terminated or truncated)
                elif len(step_result) == 4:
                    _, _, done, _ = step_result
                    ended = bool(done)
                else:
                    raise RuntimeError(f"Unexpected context replay step response length: {len(step_result)}")
                if ended:
                    break
            if not ended:
                raise RuntimeError(f"Context replay exceeded {MAX_EPISODE_STEPS} steps for {episode_id}")
            if seed not in target_seeds:
                append_jsonl(OUT / "logs" / "context_replay_progress.jsonl", {"weather_set": weather_set, "crop_year": crop_year, "episode_index": row["episode_index"], "seed": seed, "captured": False, "time_utc": datetime.now(timezone.utc).isoformat()})
                continue

            states = list(getattr(dssat, "_pilot_weather_states", []))
            legacy_rows = daily_weather_from_states(states)
            if len(legacy_rows) != len(states):
                raise RuntimeError(f"Context replay state/weather mismatch for {episode_id}: {len(states)} vs {len(legacy_rows)}")
            enriched = []
            for state, weather in zip(states, legacy_rows):
                enriched.append({
                    "DATE": weather.get("DATE"), "DOY": weather.get("DOY"),
                    "DAP": state.get("dap") if isinstance(state, dict) else None,
                    "RAIN": weather.get("RAIN"), "SRAD": weather.get("SRAD"),
                    "TMAX": weather.get("TMAX"), "TMIN": weather.get("TMIN"),
                })
            runtime_hash = legacy_sha256(canonical_weather_bytes(legacy_rows)) if legacy_rows else ""
            raw_wth = copied_for_episode[-1] if copied_for_episode else None
            archive = archive_runtime_weather(
                OUT,
                weather_seed=seed,
                rseed1=actual_seed,
                weather_set="training" if weather_set == "training" else "heldout",
                crop_year=int(crop_year),
                season_start=str(year_row["planting_date"]),
                station="YCA",
                episode_id=episode_id,
                rows=enriched,
                raw_wth_path=raw_wth,
                legacy_runtime_hash_sha256=runtime_hash,
                cli_sha256=sha256_file(CLI),
                generator_version="DSSAT runtime WTHER=W; exact generator build not separately exposed",
                runtime_version=importlib.metadata.version("gym-dssat-pdi"),
                source_runtime_path="/opt/dssat_pdi/run_dssat",
            )
            metadata = {
                "episode_id": episode_id,
                "weather_seed": seed,
                "rseed1": actual_seed,
                "crop_year": int(crop_year),
                "season_start": str(year_row["planting_date"]),
                "station": "YCA",
                "cli_path": str(CLI),
                "cli_sha256": sha256_file(CLI),
                "fitting_weather_csv": str(FITTING_WEATHER),
                "fitting_weather_csv_sha256": sha256_file(FITTING_WEATHER),
                "generator_mode": "DSSAT runtime FileX WTHER=W through frozen CNYC.CLI",
                "gym_dssat_pdi_version": importlib.metadata.version("gym-dssat-pdi"),
                "runtime_binary_sha256": sha256_file(Path("/opt/dssat_pdi/run_dssat")) if Path("/opt/dssat_pdi/run_dssat").exists() else "UNAVAILABLE",
                "runtime_source_path": str(PILOT_PATH),
                "original_temporary_working_directory": str(tmp_folder),
                "raw_wth_capture_path": str(raw_wth) if raw_wth else "",
                "raw_wth_sha256": sha256_file(raw_wth) if raw_wth else "",
                "raw_wth_status": "COPIED_RUNTIME_WTH" if raw_wth else "NOT_EMITTED_OR_NOT_CHANGED_FROM_INPUT",
                "canonical_csv_path": str(OUT / archive["canonical_csv_path"]),
                "canonical_series_sha256": archive["canonical_series_sha256"],
                "legacy_runtime_weather_sha256": runtime_hash,
                "captured_daily_rows": len(enriched),
                "step_count": steps,
                "runtime_arguments": {
                    "run_tag": run_tag,
                    "mode": "all",
                    "evaluation": False,
                    "action_index": 0,
                    "action_semantics": "no irrigation/no nitrogen; weather-only capture, not policy evaluation",
                    "process_bootstrap_rseed1": bootstrap_seed,
                    "preceding_same_year_rseed1_resets": [int(x["weather_seed"]) for x in scheduled_rows if x["episode_index"] < row["episode_index"]],
                    "schedule_episode_index": int(row["episode_index"]),
                },
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "side_effect_classification": "NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT",
                "crop_outcomes_used": False,
            }
            captures.append({"metadata": metadata, "archive": archive, "rows": enriched, "legacy_rows": legacy_rows, "rss_mb": process_rss_mb()})
            append_jsonl(OUT / "logs" / "context_replay_progress.jsonl", {"weather_set": weather_set, "crop_year": crop_year, "episode_index": row["episode_index"], "seed": seed, "captured": True, "runtime_hash": runtime_hash, "time_utc": datetime.now(timezone.utc).isoformat()})
            active_capture[0] = False
            if process_rss_mb() > RSS_LIMIT_MB:
                raise MemoryError(f"RSS exceeded {RSS_LIMIT_MB} MiB during context replay")
        return captures
    finally:
        active_capture[0] = False
        try:
            env.close()
        except Exception:
            pass
        gc.collect()


def save_capture(capture: dict[str, Any], historical: set[str]) -> dict[str, Any]:
    meta = capture["metadata"]
    generated = str(meta["legacy_runtime_weather_sha256"]).upper()
    matching = generated in historical if historical else False
    status = "EXACT_MATCH" if matching else ("MISMATCH" if historical and generated else "UNVERIFIABLE")
    meta["historical_runtime_weather_sha256_candidates"] = sorted(historical)
    meta["historical_hash_match"] = matching
    meta["verification_status"] = status
    meta["archive_manifest_row"] = capture["archive"]
    meta_path = OUT / "provenance" / f"{meta['episode_id']}_metadata.json"
    write_json(meta_path, meta)
    if int(meta["weather_seed"]) == 1001 and int(meta["crop_year"]) == 2013:
        replayed = bool(meta["runtime_arguments"].get("preceding_same_year_rseed1_resets"))
        write_json(OUT / "provenance" / "seed_1001_metadata.json", meta, replace=replayed)
        write_json(OUT / "provenance" / "seed_1001_verification.json", {
            "weather_seed": 1001,
            "rseed1": meta["rseed1"],
            "crop_year": 2013,
            "historical_runtime_weather_sha256_candidates": sorted(historical),
            "generated_runtime_weather_sha256": generated,
            "hash_match": matching,
            "verification_status": status,
            "canonical_series_sha256": meta["canonical_series_sha256"],
            "daily_rows": meta["captured_daily_rows"],
            "raw_wth_status": meta["raw_wth_status"],
        }, replace=replayed)
    capture["metadata_path"] = str(meta_path)
    return {
        "episode_id": meta["episode_id"],
        "weather_seed": meta["weather_seed"],
        "set": "heldout" if "heldout" in meta["episode_id"] else "training",
        "crop_year": meta["crop_year"],
        "rseed1": meta["rseed1"],
        "generated_runtime_hash": generated,
        "historical_hashes": ";".join(sorted(historical)),
        "hash_match": matching,
        "verification_status": status,
        "daily_rows": meta["captured_daily_rows"],
        "raw_wth_status": meta["raw_wth_status"],
        "canonical_series_sha256": meta["canonical_series_sha256"],
        "canonical_csv_path": meta["archive_manifest_row"]["canonical_csv_path"],
        "metadata_path": str(meta_path),
    }


def metric_summary(rows: list[dict[str, Any]], realization_id: str, group: str) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (int(row.get("DAP") or 0), str(row.get("DATE", ""))))
    def arr(name: str) -> np.ndarray:
        return np.asarray([float(r[name]) for r in ordered if r.get(name) not in (None, "") and np.isfinite(float(r[name]))], dtype=float)
    rain, tmax, tmin, srad = arr("RAIN"), arr("TMAX"), arr("TMIN"), arr("SRAD")
    daily = []
    for row in ordered:
        try:
            daily.append((float(row["RAIN"]), float(row["TMAX"]), float(row["TMIN"]), float(row["SRAD"])))
        except (KeyError, TypeError, ValueError):
            continue
    wet = rain[rain > 0.1]
    rain5 = [float(np.sum(rain[i:i + 5])) for i in range(max(0, len(rain) - 4))]
    def longest(mask: list[bool]) -> int:
        best = current = 0
        for yes in mask:
            current = current + 1 if yes else 0
            best = max(best, current)
        return best
    hot = [x[1] > 32.0 for x in daily]
    dry = [x[0] <= 1.0 for x in daily]
    metrics = {
        "realization_id": realization_id, "group": group, "n_days": len(daily),
        "rain_total": float(np.sum(rain)) if len(rain) else np.nan,
        "rain_days": int(np.sum(rain > 0.1)) if len(rain) else 0,
        "wet_day_intensity_mean": float(np.mean(wet)) if len(wet) else 0.0,
        "longest_dry_spell": longest([x <= 0.1 for x in rain]),
        "rx1day": float(np.max(rain)) if len(rain) else np.nan,
        "rx5day": float(max(rain5)) if rain5 else np.nan,
        "tmax_mean": float(np.mean(tmax)) if len(tmax) else np.nan,
        "tmax_max": float(np.max(tmax)) if len(tmax) else np.nan,
        "hot_days_gt30": int(np.sum(tmax > 30)) if len(tmax) else 0,
        "hot_days_gt32": int(np.sum(tmax > 32)) if len(tmax) else 0,
        "hot_days_gt35": int(np.sum(tmax > 35)) if len(tmax) else 0,
        "longest_hot_spell_gt32": longest(hot),
        "tmin_mean": float(np.mean(tmin)) if len(tmin) else np.nan,
        "tmin_min": float(np.min(tmin)) if len(tmin) else np.nan,
        "srad_mean": float(np.mean(srad)) if len(srad) else np.nan,
        "srad_min": float(np.min(srad)) if len(srad) else np.nan,
        "srad_max": float(np.max(srad)) if len(srad) else np.nan,
        "hot_dry_days_tmax_gt32_rain_le1": int(sum(h and d for h, d in zip(hot, dry))),
        "longest_hot_dry_spell": longest([h and d for h, d in zip(hot, dry)]),
    }
    return metrics


def stage_metrics(rows: list[dict[str, Any]], realization_id: str, group: str) -> list[dict[str, Any]]:
    result = []
    for lo, hi, label in STAGE_BINS:
        selected = [r for r in rows if r.get("DAP") not in (None, "") and lo <= float(r["DAP"]) <= hi]
        item = metric_summary(selected, realization_id, group)
        item["stage"] = label
        result.append(item)
    return result


def read_observed_trace() -> dict[str, list[dict[str, Any]]]:
    required = ["ppo_seed", "evaluation_weather_type", "evaluation_weather_year", "date", "dap", "RAIN", "SRAD", "TMAX", "TMIN"]
    observed: dict[str, list[dict[str, Any]]] = {}
    for chunk in pd.read_csv(STEP_TRACE, usecols=required, chunksize=100_000):
        selected = chunk[(chunk["ppo_seed"] == 5) & chunk["evaluation_weather_type"].astype(str).str.contains("observed", case=False, na=False)]
        for row in selected.to_dict("records"):
            year = str(int(float(row["evaluation_weather_year"])))
            observed.setdefault(year, []).append({
                "DATE": str(row["date"]), "DAP": int(float(row["dap"])),
                "RAIN": float(row["RAIN"]), "SRAD": float(row["SRAD"]),
                "TMAX": float(row["TMAX"]), "TMIN": float(row["TMIN"]),
            })
    for year in observed:
        unique = {}
        for row in observed[year]:
            unique[(row["DATE"], row["DAP"])] = row
        observed[year] = list(unique.values())
    return observed


def percentile_table(train_metrics: list[dict[str, Any]], observed_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ("rain_total", "rain_days", "longest_dry_spell", "rx1day", "rx5day", "tmax_mean", "tmax_max", "hot_days_gt30", "hot_days_gt32", "hot_days_gt35", "tmin_mean", "tmin_min", "srad_mean", "srad_min", "srad_max", "hot_dry_days_tmax_gt32_rain_le1")
    table = []
    for obs in observed_metrics:
        row = {"year": obs["realization_id"]}
        flags = []
        for metric in metrics:
            values = np.asarray([m[metric] for m in train_metrics if m.get("group") == "training" and np.isfinite(m.get(metric, np.nan))], dtype=float)
            value = float(obs.get(metric, np.nan))
            if len(values) == 0 or not np.isfinite(value):
                row[f"{metric}_percentile"] = np.nan
                continue
            percentile = 100.0 * float(np.mean(values <= value))
            lo, hi = float(np.min(values)), float(np.max(values))
            p5, p95 = [float(x) for x in np.percentile(values, [5, 95])]
            outside = value < lo or value > hi
            tail = value < p5 or value > p95
            row[f"{metric}_value"] = value
            row[f"{metric}_percentile"] = percentile
            row[f"{metric}_train_min"] = lo
            row[f"{metric}_train_max"] = hi
            row[f"{metric}_train_p5"] = p5
            row[f"{metric}_train_p95"] = p95
            row[f"{metric}_outside_support"] = outside
            row[f"{metric}_tail"] = tail
            flags.append((outside, tail))
        outside_n = sum(out for out, _ in flags)
        tail_n = sum(tail for _, tail in flags)
        if outside_n:
            row["coverage_flag"] = "OUTSIDE_TRAINING_SUPPORT"
        elif tail_n:
            row["coverage_flag"] = "TAIL_BUT_COVERED"
        else:
            row["coverage_flag"] = "WELL_COVERED"
        row["outside_metric_count"] = outside_n
        row["tail_metric_count"] = tail_n
        table.append(row)
    return table


def make_figures(train_metrics: list[dict[str, Any]], held_metrics: list[dict[str, Any]], observed_metrics: list[dict[str, Any]], stages: list[dict[str, Any]], percentiles: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Training WGEN": "#287D8E", "Held-out WGEN": "#E28E2C", "Observed": "#4C956C"}
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.grid": True, "grid.color": "#E4E8EB", "grid.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    metrics = [("rain_total", "Season rainfall (mm)"), ("longest_dry_spell", "Longest dry spell (days)"), ("tmax_mean", "Mean Tmax (°C)"), ("srad_mean", "Mean SRAD")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    groups = [(train_metrics, "Training WGEN"), (held_metrics, "Held-out WGEN"), (observed_metrics, "Observed")]
    for ax, (metric, title) in zip(axes.flat, metrics):
        for x, (data, label) in enumerate(groups, start=1):
            values = [float(row[metric]) for row in data if np.isfinite(row.get(metric, np.nan))]
            if values:
                bp = ax.boxplot(values, positions=[x], widths=0.52, patch_artist=True, showfliers=False)
                bp["boxes"][0].set_facecolor(colors[label]); bp["boxes"][0].set_alpha(0.75)
                ax.scatter(np.random.default_rng(17 + x).normal(x, 0.035, len(values)), values, s=10, color=colors[label], alpha=0.5, linewidths=0)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_xticks([1, 2, 3], ["Train", "Held-out", "Observed"])
    fig.savefig(OUT / "figures" / "figure1_seasonal_climate_distributions.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, metric, title in ((axes[0], "rain_total", "Rainfall by DAP stage"), (axes[1], "tmax_mean", "Mean Tmax by DAP stage")):
        stages_order = [x[2] for x in STAGE_BINS]
        for group_name, color in colors.items():
            values = []
            for stage in stages_order:
                vals = [float(r[metric]) for r in stages if r.get("group") == group_name and r.get("stage") == stage and np.isfinite(r.get(metric, np.nan))]
                values.append(vals)
            positions = np.arange(1, len(stages_order) + 1)
            means = [float(np.mean(v)) if v else np.nan for v in values]
            spread = [float(np.std(v)) if v else 0.0 for v in values]
            ax.errorbar(positions, means, yerr=spread, marker="o", linewidth=1.8, capsize=3, color=color, label=group_name)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_xticks(range(1, len(stages_order) + 1), ["0–30", "31–60", "61–90", ">90"])
        ax.legend(frameon=False)
    fig.savefig(OUT / "figures" / "figure2_dap_stage_climate.png", dpi=180)
    plt.close(fig)

    if percentiles:
        keys = [k for k in percentiles[0] if k.endswith("_percentile")]
        years = [str(r["year"]) for r in percentiles]
        matrix = np.asarray([[float(r.get(k, np.nan)) for k in keys] for r in percentiles], dtype=float)
        fig, ax = plt.subplots(figsize=(max(10, len(keys) * 0.48), 5.0), constrained_layout=True)
        im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=100, cmap="RdYlBu_r")
        ax.set_yticks(range(len(years)), years)
        ax.set_xticks(range(len(keys)), [k.replace("_percentile", "") for k in keys], rotation=60, ha="right")
        ax.set_title("Observed year positions within training WGEN (empirical percentile)", loc="left", fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Percentile")
        fig.savefig(OUT / "figures" / "figure3_observed_training_percentile_heatmap.png", dpi=180)
        plt.close(fig)

    selected = [r for r in observed_metrics if str(r["realization_id"]) in ("2014", "2019")]
    if selected and train_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        for ax, metric, label in ((axes[0], "rain_total", "Season rainfall (mm)"), (axes[1], "longest_dry_spell", "Longest dry spell (days)")):
            vals = [r[metric] for r in train_metrics if r.get("group") == "training" and np.isfinite(r.get(metric, np.nan))]
            ax.hist(vals, bins=12, color=colors["Training WGEN"], alpha=0.78, edgecolor="white")
            for obs in selected:
                ax.axvline(obs[metric], linewidth=2, label=str(obs["realization_id"]))
            ax.set_title(label, loc="left", fontsize=10, fontweight="bold")
            ax.legend(frameon=False)
        fig.suptitle("Seed5 weak-year weather profiles (associational)", x=0.02, ha="left", fontsize=11, fontweight="bold")
        fig.savefig(OUT / "figures" / "figure4_seed5_2014_2019_weather_profile.png", dpi=180)
        plt.close(fig)


def run_coverage(verification_path: Path | None = None) -> dict[str, Any]:
    import matplotlib
    train_rows, held_rows = [], []
    verification_path = verification_path or OUT / "hash_verification_context_replay.csv"
    if not verification_path.is_file():
        return {"coverage_status": "INSUFFICIENT", "reason": "No context-replay hash verification manifest exists."}
    verified = pd.read_csv(verification_path, dtype=str).to_dict("records")
    for row in verified:
        if row.get("verification_status") != "EXACT_MATCH":
            continue
        path = OUT / str(row.get("canonical_csv_path", ""))
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        records = frame.to_dict("records")
        group = "Training WGEN" if row.get("set") == "training" else "Held-out WGEN"
        metric = metric_summary(records, str(row.get("weather_seed")), group)
        (train_rows if row.get("set") == "training" else held_rows).append((str(row.get("weather_seed")), records, metric))
    if len(train_rows) != 80 or len(held_rows) != 20:
        return {"coverage_status": "INSUFFICIENT", "reason": f"Expected 80/20 canonical series; found {len(train_rows)}/{len(held_rows)}."}
    observed = read_observed_trace()
    train_metrics = [item[2] for item in train_rows]
    held_metrics = [item[2] for item in held_rows]
    observed_metrics = [metric_summary(rows, year, "Observed") for year, rows in sorted(observed.items())]
    all_stages = []
    for name, rows, _ in train_rows:
        all_stages.extend(stage_metrics(rows, name, "Training WGEN"))
    for name, rows, _ in held_rows:
        all_stages.extend(stage_metrics(rows, name, "Held-out WGEN"))
    for year, rows in observed.items():
        all_stages.extend(stage_metrics(rows, year, "Observed"))
    comp = []
    for data in (train_metrics, held_metrics, observed_metrics):
        comp.extend({k: row[k] for k in ("realization_id", "group", "hot_dry_days_tmax_gt32_rain_le1", "longest_hot_dry_spell", "n_days")} for row in data)
    percentiles = percentile_table(train_metrics, observed_metrics)
    write_csv(OUT / "coverage" / "weather_metrics_by_realization.csv", train_metrics + held_metrics)
    write_csv(OUT / "coverage" / "observed_weather_metrics.csv", observed_metrics)
    write_csv(OUT / "coverage" / "stage_weather_metrics.csv", all_stages)
    write_csv(OUT / "coverage" / "compound_extreme_metrics.csv", comp)
    write_csv(OUT / "coverage" / "observed_training_percentiles.csv", percentiles)
    held_summary = []
    for metric in ("rain_total", "longest_dry_spell", "tmax_mean", "tmin_mean", "srad_mean", "hot_dry_days_tmax_gt32_rain_le1"):
        a = np.asarray([r[metric] for r in train_metrics], dtype=float)
        b = np.asarray([r[metric] for r in held_metrics], dtype=float)
        pooled = float(np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)) if len(a) > 1 and len(b) > 1 else np.nan
        held_summary.append({"metric": metric, "training_mean": float(np.mean(a)), "heldout_mean": float(np.mean(b)), "mean_difference_heldout_minus_training": float(np.mean(b) - np.mean(a)), "standardized_difference": float((np.mean(b) - np.mean(a)) / pooled) if pooled > 0 else np.nan})
    write_csv(OUT / "coverage" / "heldout_vs_training_distribution.csv", held_summary)
    make_figures(train_metrics, held_metrics, observed_metrics, all_stages, percentiles)
    flags = [r["coverage_flag"] for r in percentiles]
    outside = sum(x == "OUTSIDE_TRAINING_SUPPORT" for x in flags)
    if not flags:
        observed_coverage = "INSUFFICIENT"
    elif outside == 0:
        observed_coverage = "GOOD" if all(x == "WELL_COVERED" for x in flags) else "PARTIAL"
    elif outside <= max(1, len(flags) // 3):
        observed_coverage = "PARTIAL"
    else:
        observed_coverage = "POOR"
    held_supported = set()
    for metric in ("rain_total", "longest_dry_spell", "tmax_mean", "srad_mean"):
        vals = [x[metric] for x in held_metrics]
        if vals:
            held_supported.add(metric)
    gaps = [r for r in percentiles if r["coverage_flag"] == "OUTSIDE_TRAINING_SUPPORT"]
    tail_gap = "NONE" if not gaps else "RANDOM_SAMPLING_GAP"
    recommendation = "KEEP_80" if not gaps else "TARGETED_TAIL_AUGMENTATION_CANDIDATE"
    return {"coverage_status": observed_coverage, "tail_gap": tail_gap, "recommendation": recommendation, "observed_year_count": len(observed), "training_count": len(train_metrics), "heldout_count": len(held_metrics), "observed_flag_counts": {flag: flags.count(flag) for flag in sorted(set(flags))}, "heldout_support_check_metrics": sorted(held_supported), "seed5_weak_years": [r for r in observed_metrics if str(r["realization_id"]) in ("2014", "2019")], "heldout_distribution_rows": held_summary}


def write_report(decision: dict[str, Any], chain: str, smoke_rows: list[dict[str, Any]], batch_rows: list[dict[str, Any]], coverage: dict[str, Any], *, replace: bool = False) -> None:
    OUT.joinpath("provenance", "runtime_weather_generation_chain.md").write_text(chain, encoding="utf-8", newline="\n")
    root_chain = OUT / "runtime_weather_generation_chain.md"
    if root_chain.exists() and root_chain.read_text(encoding="utf-8") != chain:
        raise FileExistsError(f"Refusing to overwrite generation-chain copy: {root_chain}")
    if not root_chain.exists():
        root_chain.write_text(chain, encoding="utf-8", newline="\n")
    context_evidence = """# Runtime context replay evidence\n\n004_05 uses one cached WGEN `DssatPdi` environment per crop-year (`ScheduledEpisodeEnv._env_for`; `RECREATE_WGEN_ENV_PER_EPISODE=False`). A weather seed alone is therefore not a complete launch context.\n\nFor the historical seed1001 smoke, the schedule gives: episode 7 / year 2013 / RSEED1 1026 (year-runtime bootstrap and first reset), episode 11 / year 2013 / RSEED1 1043, then episode 26 / year 2013 / RSEED1 1001. The context-aware smoke replayed that sequence in one runtime process. Its 103-row canonical runtime hash exactly matched the 004_05 historical hash.\n\nAn earlier independent-process diagnostic started year 2013 directly with seed1001 and produced 309 captured rows and a different hash. That attempt is retained as failed context evidence and excluded from the formal hash-verification manifest.\n\nFor batch recovery, the runner processes one crop-year environment at a time, serially, replaying same-year schedule rows through each weather seed's first-use episode. Held-out seeds use one 2008 context environment in scheduled order. Runtime crop outputs are side effects only and are not analyzed.\n"""
    context_path = OUT / "provenance" / "runtime_context_replay.md"
    if context_path.exists() and context_path.read_text(encoding="utf-8") != context_evidence:
        raise FileExistsError(f"Refusing to overwrite runtime-context evidence: {context_path}")
    if not context_path.exists():
        context_path.write_text(context_evidence, encoding="utf-8", newline="\n")
    exact_training = sum(row.get("set") == "training" and row.get("verification_status") == "EXACT_MATCH" for row in batch_rows)
    exact_heldout = sum(row.get("set") == "heldout" and row.get("verification_status") == "EXACT_MATCH" for row in batch_rows)
    lines = [
        "# YC 004_17 runtime WGEN 天气恢复与气候覆盖审计",
        "",
        "## 范围与边界",
        "本任务未训练 PPO、未加载 checkpoint、未运行策略评估。为捕获天气而执行的 DSSAT no-op episode 只作为 runtime 天气物化手段；其作物模拟输出属于 `NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT`，不进入任何性能统计。旧实验目录未修改。",
        "",
        "## 生成链与历史 hash",
        "完整逐层来源见 `results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery/provenance/runtime_weather_generation_chain.md`。历史 SHA256 是 PDI daily-state 序列化哈希，不是 WTH raw-byte hash。历史 pilot 的 runtime snapshot 枚举了可见输出文件但没有 `.WTH`；004_05 rendered input WTH 是 observed 源文件副本，不能视为 WGEN 实现。",
        "",
        "## Seed1001 smoke",
        f"- 状态：`{decision.get('smoke_status')}`；seed=1001、crop-year context=2013、schedule episode=26。",
        f"- 历史 hash：`{SMOKE_HISTORICAL_HASH}`；生成 hash：`{smoke_rows[0].get('generated_runtime_hash','') if smoke_rows else ''}`。",
        f"- raw WTH：`{smoke_rows[0].get('raw_wth_status','') if smoke_rows else '未运行'}`。仅变化/新生成于 runtime 的 WTH 才会复制；未将输入 WTH 冒充输出。",
        "- 初次新建进程、直接 bootstrap seed1001 的诊断尝试未复现缓存 runtime 上下文（309 captured states，hash 不匹配），已单独保留且不作为正式验证。正式 smoke 重放 2013 year-runtime 的 bootstrap seed1026 及前序 RSEED1=1026、1043 后，对 seed1001 逐日 canonical hash 精确匹配（103 captured rows）。",
        "",
        "## 批量恢复",
        f"- context replay 记录数：{len(batch_rows)}。最终逐 seed 对照见 `{('hash_verification_context_replay_reconciled.csv' if (OUT / 'hash_verification_context_replay_reconciled.csv').exists() else ('hash_verification_context_replay_complete.csv' if (OUT / 'hash_verification_context_replay_complete.csv').exists() else 'hash_verification_context_replay.csv'))}`；`hash_verification.csv` 保留运行尝试记录，不应与最终 reconciled ledger 混读。",
        f"- 已归档 training realization：{len({str(x.get('weather_seed')) for x in batch_rows if x.get('set') == 'training'})}/80；held-out：{len({str(x.get('weather_seed')) for x in batch_rows if x.get('set') == 'heldout'})}/20。",
        f"- 历史 hash reconciliation：training exact {exact_training}/80；held-out exact {exact_heldout}/20。held-out 依据 004_05 正式 evaluation episode summary 对照；带 RSEED1/历史 hash 的最终映射见 `weather_archive_manifest_verified.csv`。",
        *[f"- 未通过逐日历史 hash 的记录：seed={x.get('weather_seed')} / crop-year={x.get('crop_year')} / rows={x.get('daily_rows')} / generated={x.get('generated_runtime_hash')} / historical={x.get('historical_hashes')}。" for x in batch_rows if x.get("verification_status") != "EXACT_MATCH"],
        "- 若 smoke gate 未通过，不启动批量恢复或正式 coverage。批量 hash 不完整或不匹配时，不得把整体结果称为 exact historical reconstruction；未通过记录按未验证处理。",
        "",
        "## Coverage 与限制",
        f"- `observed_training_coverage = {decision.get('observed_training_coverage')}`",
        f"- `weather_tail_gap = {decision.get('weather_tail_gap')}`",
        f"- `next_weather_action = {decision.get('next_weather_action')}`",
        f"- Coverage状态：{coverage.get('coverage_status', 'NOT_RUN')}。Observed weather 来自 004_05 seed5 的官方 evaluation daily-state trace，按 year/DAP 去重；没有用产量解释天气。",
        *(["- 本轮正式 climate coverage 未执行：历史逐日 hash gate 未完整通过，且没有物化齐全的 80/20 exact historical series。"] if coverage.get("coverage_status") == "INSUFFICIENT" else ["- DAP 分段固定为 0–30、31–60、61–90、>90；热日阈值预先固定为 Tmax >30/32/35°C；hot-dry 日固定定义为 Tmax >32°C 且 rainfall ≤1 mm。", "- seed5 2014/2019 仅做天气分布位置和同类天气反例关联描述，不做因果结论。"]),
        "",
        "## 最终状态",
        f"`weather_recovery_status = {decision.get('weather_recovery_status')}`",
        f"`observed_training_coverage = {decision.get('observed_training_coverage')}`",
        f"`weather_tail_gap = {decision.get('weather_tail_gap')}`",
        f"`next_weather_action = {decision.get('next_weather_action')}`",
        "",
        "本报告未把运行时 side-effect 的 crop outcome 当科学结果。",
    ]
    path = ROOT / "docs" / "yc_random_weather_004_17_runtime_weather_recovery_and_coverage.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content and not replace:
        raise FileExistsError(f"Refusing to overwrite report: {path}")
    if not path.exists() or replace:
        path.write_text(content, encoding="utf-8", newline="\n")


def audit_archive_integrity() -> int:
    ledger_path = OUT / "hash_verification_context_replay_reconciled.csv"
    manifest_path = OUT / "weather_archive_manifest_verified.csv"
    if not ledger_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Reconciled hash ledger and verified archive manifest are required before integrity audit")
    with ledger_path.open("r", encoding="utf-8-sig", newline="") as stream:
        ledger = list(csv.DictReader(stream))
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    checks = []
    for row in manifest:
        path = OUT / row["canonical_csv_path"]
        exists = path.is_file()
        actual = sha256_file(path) if exists else ""
        expected = row.get("canonical_series_sha256", "")
        checks.append({
            "episode_id": row.get("episode_id", ""),
            "weather_seed": row.get("weather_seed", ""),
            "set": row.get("set", ""),
            "canonical_csv_path": row.get("canonical_csv_path", ""),
            "file_exists": exists,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "sha256_match": bool(exists and actual == expected),
            "historical_verification": row.get("verification_level", ""),
        })
    by_set: dict[str, dict[str, int]] = {}
    for row in ledger:
        group = row.get("set", "unknown")
        by_set.setdefault(group, {})[row.get("verification_status", "unknown")] = by_set.setdefault(group, {}).get(row.get("verification_status", "unknown"), 0) + 1
    raw_files = list((OUT / "recovered_weather").rglob("*.WTH"))
    canonical_files = list((OUT / "canonical_weather").rglob("*.csv"))
    base_manifest = OUT / "weather_archive_manifest.csv"
    with base_manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        base_rows = list(csv.DictReader(stream))
    check = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "check_type": "independent_process_read_only_archive_hash_check",
        "container_root": "/workspace",
        "host_root": r"C:\Users\DELL\gym_workspace\gym_dssat_pdi_bingo\gym-dssat-pdi",
        "ledger_rows": len(ledger),
        "verified_manifest_rows": len(manifest),
        "historical_verification_by_set": by_set,
        "canonical_files_referenced": len(checks),
        "canonical_files_hash_ok": sum(row["sha256_match"] for row in checks),
        "canonical_files_total_in_archive_including_retained_attempts": len(canonical_files),
        "archive_manifest_episode_rows_including_retained_attempts": len(base_rows),
        "raw_wth_files_emitted_and_archived": len(raw_files),
        "all_referenced_files_exist_and_hash_match": bool(checks) and all(row["sha256_match"] for row in checks),
        "formal_climate_coverage_run": False,
        "unresolved_historical_matches": [row for row in manifest if row.get("verification_level") != "EXACT_MATCH"],
        "non_scientific_runtime_crop_outputs_used": False,
    }
    csv_path = OUT / "provenance" / "final_archive_integrity.csv"
    json_path = OUT / "provenance" / "final_archive_integrity.json"
    write_csv(csv_path, checks)
    write_json(json_path, check)
    return 0 if check["all_referenced_files_exist_and_hash_match"] and len(ledger) == 100 and len(manifest) == 100 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "trace", "smoke"), default="all")
    parser.add_argument("--resume-batch", action="store_true", help="Continue only missing weather seeds from an existing context-replay ledger; preserve the partial ledger.")
    parser.add_argument("--reconcile-existing", action="store_true", help="Read-only hash reconciliation from existing 004_05 logs; does not launch DSSAT/WGEN.")
    parser.add_argument("--integrity-audit", action="store_true", help="Independently verify archived canonical CSV file hashes; does not launch DSSAT/WGEN.")
    args = parser.parse_args()
    ensure_new_output_tree()
    preserved = OUT / "provenance" / "initial_smoke_attempts"
    preserved.mkdir(parents=True, exist_ok=True)
    for source in (
        OUT / "decision.json",
        OUT / "provenance" / "seed_1001_metadata.json",
        OUT / "provenance" / "seed_1001_verification.json",
        ROOT / "docs" / "yc_random_weather_004_17_runtime_weather_recovery_and_coverage.md",
    ):
        if source.is_file():
            target = preserved / source.name
            if not target.exists():
                shutil.copy2(source, target)
    chain = source_chain_markdown()
    chain_path = OUT / "provenance" / "runtime_weather_generation_chain.md"
    if chain_path.exists() and chain_path.read_text(encoding="utf-8") != chain:
        raise FileExistsError(f"Refusing to overwrite existing provenance chain: {chain_path}")
    if not chain_path.exists():
        chain_path.write_text(chain, encoding="utf-8", newline="\n")

    if args.reconcile_existing:
        return reconcile_existing_ledger()
    if args.integrity_audit:
        return audit_archive_integrity()

    cli_hash = sha256_file(CLI)
    fitting_hash = sha256_file(FITTING_WEATHER)
    schedule = schedule_rows()
    smoke_matches = [r for r in schedule if r["weather_seed"] == 1001 and r["historical_year"] == 2013]
    if cli_hash != EXPECTED_CLI_SHA256 or fitting_hash != EXPECTED_FITTING_SHA256 or len(smoke_matches) == 0:
        decision = {"weather_recovery_status": "FAILED", "observed_training_coverage": "INSUFFICIENT", "weather_tail_gap": "INSUFFICIENT", "next_weather_action": "INSUFFICIENT", "smoke_status": "UNVERIFIABLE", "reason": "Frozen CLI hash or seed1001/context2013 schedule row did not validate."}
        write_json(OUT / "decision.json", decision, replace=True)
        write_report(decision, chain, [], [], {}, replace=True)
        return 2
    smoke_row = smoke_matches[0]
    expected_episode = 26
    if smoke_row["episode_index"] != expected_episode:
        raise RuntimeError(f"Unexpected seed1001/context2013 schedule episode: {smoke_row}")
    if args.phase == "trace":
        write_json(OUT / "decision.json", {"weather_recovery_status": "INSUFFICIENT", "observed_training_coverage": "INSUFFICIENT", "weather_tail_gap": "INSUFFICIENT", "next_weather_action": "INSUFFICIENT", "phase": "TRACE_ONLY", "reason": "Runtime capture was not requested."})
        return 0

    pilot = load_pilot()
    hist = historical_hashes()
    verify_path = OUT / "provenance" / "seed_1001_verification.json"
    meta_path = OUT / "provenance" / "seed_1001_metadata.json"
    if verify_path.is_file() and meta_path.is_file() and json.loads(verify_path.read_text(encoding="utf-8")).get("verification_status") == "EXACT_MATCH":
        smoke_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        smoke_capture = {"metadata": smoke_meta, "archive": smoke_meta["archive_manifest_row"]}
        smoke_record = save_capture(smoke_capture, hist.get((1001, 2013), {SMOKE_HISTORICAL_HASH}))
        smoke_csv_path = Path(smoke_meta["canonical_csv_path"])
        if not smoke_csv_path.is_absolute():
            smoke_csv_path = OUT / smoke_csv_path
    else:
        preceding_context_rows = [r for r in schedule if r["historical_year"] == 2013 and r["episode_index"] < expected_episode]
        if not preceding_context_rows:
            raise RuntimeError("Missing same-year predecessor schedule rows required to reproduce the cached 2013 runtime context")
        process_bootstrap_seed = preceding_context_rows[0]["weather_seed"]
        predecessor_seeds = tuple(r["weather_seed"] for r in preceding_context_rows)
        smoke_capture = run_runtime_capture(
            pilot,
            1001,
            2013,
            "training",
            "smoke_replay_seed_1001_ctx_2013_ep_26",
            bootstrap_seed=process_bootstrap_seed,
            predecessor_seeds=predecessor_seeds,
        )
        smoke_record = save_capture(smoke_capture, hist.get((1001, 2013), {SMOKE_HISTORICAL_HASH}))
        smoke_record["raw_wth_status"] = smoke_capture["metadata"]["raw_wth_status"]
        append_csv(OUT / "hash_verification.csv", [smoke_record])
        smoke_csv_path = OUT / smoke_record["canonical_csv_path"]
    with smoke_csv_path.open("rb") as stream:
        smoke_csv_sha = sha256_bytes(stream.read())
    if smoke_csv_sha != smoke_record["canonical_series_sha256"]:
        raise RuntimeError("Independent smoke CSV hash verification failed")
    smoke_status = smoke_record["verification_status"]
    batch_rows: list[dict[str, Any]] = []
    new_batch_rows: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {"coverage_status": "NOT_RUN"}
    if smoke_status == "EXACT_MATCH" and args.phase == "all":
        previous_rows: list[dict[str, Any]] = []
        partial_ledger = OUT / "hash_verification_context_replay.csv"
        if args.resume_batch:
            if not partial_ledger.is_file():
                raise FileNotFoundError(f"Cannot resume without existing partial ledger: {partial_ledger}")
            previous_rows = pd.read_csv(partial_ledger, dtype=str).fillna("").to_dict("records")
            for row in previous_rows:
                if row.get("verification_status") == "EXACT_MATCH":
                    expected = OUT / str(row.get("canonical_csv_path", ""))
                    if not expected.is_file():
                        raise FileNotFoundError(f"Existing exact ledger row has no archived canonical CSV: {expected}")
            batch_rows.extend(previous_rows)
        training_first = {}
        for row in schedule:
            training_first.setdefault(row["weather_seed"], row)
        if len(training_first) != 80:
            raise RuntimeError(f"Expected 80 unique training seeds in historical schedule, got {len(training_first)}")
        if not args.resume_batch:
            batch_rows.append(smoke_record)
        prior_training_seeds = {int(row["weather_seed"]) for row in previous_rows if row.get("set") == "training" and str(row.get("weather_seed", "")).isdigit()}
        prior_heldout_seeds = {int(row["weather_seed"]) for row in previous_rows if row.get("set") == "heldout" and str(row.get("weather_seed", "")).isdigit()}
        first_block_end = max(row["episode_index"] for row in training_first.values())
        first_block_rows = [row for row in schedule if row["episode_index"] <= first_block_end]
        if len({row["weather_seed"] for row in first_block_rows}) != 80:
            raise RuntimeError("The first-use schedule window does not contain all 80 unique weather seeds")
        batch_gate_failed = False
        for year in sorted({row["historical_year"] for row in training_first.values()}):
            year_targets = {seed for seed, row in training_first.items() if row["historical_year"] == year and seed != 1001 and seed not in prior_training_seeds}
            if not year_targets:
                continue
            last_target_episode = max(training_first[seed]["episode_index"] for seed in year_targets)
            year_sequence = [row for row in schedule if row["historical_year"] == year and row["episode_index"] <= last_target_episode]
            captures = run_context_group(pilot, int(year), year_sequence, year_targets, "training")
            for capture in captures:
                seed = int(capture["metadata"]["weather_seed"])
                target_row = training_first[seed]
                record = save_capture(capture, hist.get((seed, int(year)), set()))
                record["set"] = "training"
                if int(capture["metadata"]["runtime_arguments"]["schedule_episode_index"]) != int(target_row["episode_index"]):
                    raise RuntimeError(f"Captured episode does not match first-use schedule row for seed={seed}")
                batch_rows.append(record)
                new_batch_rows.append(record)
                append_jsonl(OUT / "logs" / "context_replay_progress.jsonl", {"seed": seed, "context": year, "status": record["verification_status"], "time_utc": datetime.now(timezone.utc).isoformat()})
            if process_rss_mb() > RSS_LIMIT_MB:
                raise MemoryError(f"RSS exceeded {RSS_LIMIT_MB} MiB; stopped serial recovery with completed files retained")
            if any(row["verification_status"] != "EXACT_MATCH" for row in batch_rows if row["set"] == "training" and row["weather_seed"] != 1001 and int(row["crop_year"]) == int(year)):
                batch_gate_failed = True
                if not args.resume_batch:
                    break

        if not batch_gate_failed or args.resume_batch:
            heldout_rows = [{"episode_index": i + 1, "historical_year": 2008, "weather_seed": seed} for i, seed in enumerate(WEATHER_SEEDS_HELDOUT)]
            missing_heldout = set(WEATHER_SEEDS_HELDOUT) - prior_heldout_seeds
            heldout_captures = run_context_group(pilot, 2008, heldout_rows, missing_heldout, "heldout") if missing_heldout else []
            for capture in heldout_captures:
                seed = int(capture["metadata"]["weather_seed"])
                record = save_capture(capture, hist.get((seed, 2008), set()))
                record["set"] = "heldout"
                batch_rows.append(record)
                new_batch_rows.append(record)
                append_jsonl(OUT / "logs" / "context_replay_progress.jsonl", {"seed": seed, "context": 2008, "status": record["verification_status"], "time_utc": datetime.now(timezone.utc).isoformat()})
        if args.resume_batch:
            complete_ledger = OUT / "hash_verification_context_replay_complete.csv"
            write_csv(complete_ledger, batch_rows)
            append_csv(OUT / "hash_verification.csv", new_batch_rows)
            coverage_ledger = complete_ledger
        else:
            write_csv(partial_ledger, batch_rows)
            append_csv(OUT / "hash_verification.csv", batch_rows[1:])
            coverage_ledger = partial_ledger
        statuses = [row["verification_status"] for row in batch_rows]
        all_exact = len(batch_rows) == 100 and all(status == "EXACT_MATCH" for status in statuses)
        if all_exact:
            coverage = run_coverage(coverage_ledger)
        else:
            mismatches = [f"seed={row.get('weather_seed')} year={row.get('crop_year')} status={row.get('verification_status')}" for row in batch_rows if row.get("verification_status") != "EXACT_MATCH"]
            coverage = {"coverage_status": "INSUFFICIENT", "reason": "Batch does not contain 100 exact historical realizations; formal historical-domain coverage is not claimed.", "unverified_rows": mismatches}
    elif smoke_status != "EXACT_MATCH":
        coverage = {"coverage_status": "INSUFFICIENT", "reason": f"Seed1001 smoke gate was {smoke_status}; batch and formal coverage were not started."}

    if smoke_status == "EXACT_MATCH" and args.phase == "all" and batch_rows and all(x["verification_status"] == "EXACT_MATCH" for x in batch_rows) and len(batch_rows) == 100:
        recovery_status = "EXACT"
        observed_coverage = coverage.get("coverage_status", "INSUFFICIENT")
        tail_gap = coverage.get("tail_gap", "INSUFFICIENT")
        next_action = coverage.get("recommendation", "INSUFFICIENT")
    else:
        recovery_status = "FAILED" if smoke_status in ("MISMATCH", "UNVERIFIABLE") else "PARTIALLY_VERIFIED"
        observed_coverage, tail_gap, next_action = "INSUFFICIENT", "INSUFFICIENT", "INSUFFICIENT"
    decision = {
        "weather_recovery_status": recovery_status,
        "observed_training_coverage": observed_coverage,
        "weather_tail_gap": tail_gap,
        "next_weather_action": next_action,
        "smoke_status": smoke_status,
        "smoke_context": {"weather_seed": 1001, "rseed1": smoke_capture["metadata"]["rseed1"], "crop_year": 2013, "schedule_episode": smoke_row["episode_index"]},
        "smoke_historical_hash": SMOKE_HISTORICAL_HASH,
        "smoke_generated_hash": smoke_record["generated_runtime_hash"],
        "raw_wth_emitted": smoke_record["raw_wth_status"],
        "batch_count": len(batch_rows),
        "batch_exact_count": sum(row["verification_status"] == "EXACT_MATCH" for row in batch_rows),
        "coverage": coverage,
        "crop_simulation_outputs_used": False,
        "side_effect_classification": "NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT",
        "historical_wth_note": "Historical runtime snapshot had no emitted WTH; actual daily state series is the weather evidence. Static rendered source WTH is excluded.",
    }
    write_json(OUT / "decision.json", decision, replace=True)
    write_report(decision, chain, [smoke_record], batch_rows, coverage, replace=True)
    return 0 if smoke_status == "EXACT_MATCH" else 3


if __name__ == "__main__":
    raise SystemExit(main())
