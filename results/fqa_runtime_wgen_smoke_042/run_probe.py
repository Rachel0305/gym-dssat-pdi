"""One FQA crop-season WGEN probe, isolated from YC and all PPO runners."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/fqa_runtime_wgen_smoke_042"
SOURCE = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/FQ"
SOURCE_MZX = SOURCE / "CNFQ0801.MZX"
CLI = ROOT / "results/fqa_weather_resume_041/CNFQ.CLI"
CULTIVAR = SOURCE / "MZCER048.CUL"
SOIL = SOURCE / "SOIL.SOL"
SEED = 101
MAX_STEPS = 240
MAX_SECONDS = 120
RSS_LIMIT_MB = 768


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare() -> dict:
    before = OUT / "inputs/fileX_before.MZX"
    after = OUT / "inputs/fileX_after_W.jinja2"
    if before.exists() or after.exists():
        raise FileExistsError("042 isolated input copies already exist; refusing overwrite")
    for path in (SOURCE_MZX, CLI, CULTIVAR, SOIL):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_hash = digest(SOURCE_MZX)
    cli_hash = digest(CLI)
    gate = json.loads((ROOT / "results/fqa_weather_resume_041/final_gate.json").read_text(encoding="utf-8"))
    if cli_hash.lower() != gate["output_sha256"]["CNFQ.CLI"].lower():
        raise RuntimeError("FQA CLI differs from 041 frozen output")
    raw = SOURCE_MZX.read_text(encoding="utf-8", errors="replace")
    if not re.search(r"(?m)^\s*1\s+1\s+1\s+0\s+Sim2007\b", raw):
        raise RuntimeError("FQA treatment 1 / 2007 not found")
    if not re.search(r"(?m)^\s*1\s+CNFQ2007\s+CNFQ0701\b", raw):
        raise RuntimeError("FQA WSTA CNFQ0701 not found")
    updated, count = re.subn(r"(?m)^(\s*1\s+ME\s+)M(\s+M\s+E\s+R\s+S\s+L\s+R\s+1\s+G\s+R\s+2\s*)$", r"\g<1>W\2", raw, count=1)
    if count != 1:
        raise RuntimeError("Exactly one treatment-1 WTHER=M row was not found")
    before.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_MZX, before)
    after.write_text(updated, encoding="utf-8", newline="\n")
    paths = [SOURCE_MZX, before, after, CLI, CULTIVAR, SOIL]
    record = {
        "task": "042", "site": "FQA", "scenario": "D222_BLANK_ZERO", "year": 2007,
        "treatment": 1, "weather_seed": SEED, "filex_wsta": "CNFQ0701",
        "source_wther": "M", "isolated_wther": "W", "expected_cli_basename": "CNFQ.CLI",
        "file_hashes": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths],
        "source_filex_preserved": digest(SOURCE_MZX) == source_hash,
        "coordinate_input_filex": "-99 placeholders in FIELD XCRD/YCRD/ELEV",
        "coordinate_input_cli": gate["station_coordinates_from_wth"],
        "max_steps": MAX_STEPS, "max_seconds": MAX_SECONDS, "rss_limit_mb": RSS_LIMIT_MB,
        "ppo_seed": "NOT_APPLICABLE", "formal_training": False,
    }
    save_json(OUT / "input_provenance.json", record)
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", required=True, choices=("attempt_01", "attempt_02_same_seed"))
    args = ap.parse_args()
    run_dir = OUT / "runtime" / args.attempt
    if run_dir.exists():
        raise FileExistsError(run_dir)
    provenance = json.loads((OUT / "input_provenance.json").read_text(encoding="utf-8"))
    expected = {item["path"]: item["sha256"] for item in provenance["file_hashes"]}
    for path_text, sha in expected.items():
        if digest(ROOT / path_text.replace("\\", "/")) != sha:
            raise RuntimeError(f"Frozen input hash changed: {path_text}")
    run_dir.mkdir(parents=True)
    tmp = OUT / "runtime/tmp"
    tmp.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(tmp)
    tempfile.tempdir = str(tmp)
    sys.path.insert(0, str(ROOT))
    from scripts import run_yc_wgen_seed_pilot as helper

    started = time.monotonic()
    env = None
    cls = None
    originals = None
    capture = []
    result = {
        "attempt": args.attempt, "site": "FQA", "scenario": "D222_BLANK_ZERO", "year": 2007,
        "treatment": 1, "rseed1": SEED, "formal_training": False,
        "ppo_executed": False, "status": "STARTED", "weather_capture": "NOT_OBSERVED",
    }
    try:
        import gym
        import gym_dssat_pdi.envs.dssat_pdi as dssat_module

        cls = helper._runtime_class(dssat_module)
        originals = (cls._get_state, cls._get_sockets_)

        def get_state(instance):
            value = originals[0](instance)
            state = getattr(instance, "_state", None)
            if isinstance(state, dict):
                capture.append(copy.deepcopy(state))
            return value

        def get_sockets(instance):
            instance._rseed1 = SEED
            value = originals[1](instance)
            if getattr(instance, "_server", None) is not None:
                import zmq
                instance._server.setsockopt(zmq.RCVTIMEO, 30000)
            return value

        cls._get_state = get_state
        cls._get_sockets_ = get_sockets
        result["dssat_launch_attempted"] = True
        env = gym.make(
            "gym_dssat_pdi:GymDssatPdi-v0",
            log_saving_path=str(run_dir / "runtime_log.txt"),
            mode="all",
            auxiliary_file_paths=[str(CULTIVAR), str(CLI), str(SOIL)],
            random_weather=True,
            seed=None,
            fileX_template_path=str(OUT / "inputs/fileX_after_W.jinja2"),
            experiment_number=1,
            evaluation=True,
            cultivar="maize",
            run_dssat_location="/opt/dssat_pdi/run_dssat",
        ).unwrapped
        cwd = Path(env._tmp_folder)
        result["runtime_cwd"] = str(cwd)
        actual_cli = cwd / "CNFQ.CLI"
        result["runtime_cli_found"] = actual_cli.is_file()
        if not actual_cli.is_file() or digest(actual_cli) != digest(CLI):
            raise RuntimeError("CNFQ.CLI missing or hash mismatch in DSSAT runtime directory")
        result["runtime_cli_sha256"] = digest(actual_cli)
        actual_filex = (cwd / "fileX.MZX").read_text(encoding="utf-8", errors="replace")
        result["runtime_filex_wther"] = helper.runtime_filex_mode(actual_filex)
        result["runtime_filex_wsta"] = re.search(r"(?m)^\s*1\s+CNFQ2007\s+(CNFQ\d+)\b", actual_filex).group(1) if re.search(r"(?m)^\s*1\s+CNFQ2007\s+(CNFQ\d+)\b", actual_filex) else None
        if result["runtime_filex_wther"] != "W" or result["runtime_filex_wsta"] != "CNFQ0701":
            raise RuntimeError("Runtime FileX WTHER/WSTA mismatch")
        yml = cwd / "dssat-pdi.yml"
        result["pdi_rseed1_configured"] = yml.is_file() and helper.pdi_seed_configured(yml.read_text(encoding="utf-8", errors="replace"), SEED)
        if not result["pdi_rseed1_configured"]:
            raise RuntimeError("Runtime PDI seed not confirmed")
        action = {key: 0.0 for key in env.action_variables}
        steps = 0
        peak_rss = 0.0
        while not bool(getattr(env, "done", False)) and steps < MAX_STEPS:
            if time.monotonic() - started > MAX_SECONDS:
                raise TimeoutError("FQA smoke wall-time limit exceeded")
            env.step(action)
            steps += 1
            peak_rss = max(peak_rss, helper._process_tree_rss_mb())
            if peak_rss > RSS_LIMIT_MB:
                raise MemoryError("FQA smoke process-tree RSS limit exceeded")
        result["steps"] = steps
        result["peak_process_tree_rss_mb"] = peak_rss
        result["episode_done"] = bool(getattr(env, "done", False))
        if not result["episode_done"]:
            raise TimeoutError("FQA smoke step limit exceeded")
        weather = helper.daily_weather_from_states(capture, start_date=date(2007, 6, 1))
        result["state_count"] = len(capture)
        result["weather_row_count"] = len(weather)
        result["weather_summary"] = helper.summarize_weather(weather)
        result["physical_sanity"] = helper.physical_sanity(weather)
        result["weather_fields_complete"] = bool(weather) and all(all(isinstance(row.get(key), (int, float)) for key in helper.WEATHER_FIELDS) for row in weather)
        if not result["weather_fields_complete"]:
            raise RuntimeError("No complete RAIN/SRAD/TMAX/TMIN runtime series captured")
        data = helper.canonical_weather_bytes(weather)
        weather_path = run_dir / "runtime_weather_daily.csv"
        weather_path.write_bytes(data)
        result["weather_sha256"] = hashlib.sha256(data).hexdigest().upper()
        result["weather_capture"] = "RUNTIME_DAILY_STATE"
        result["status"] = "RUNTIME_WEATHER_CAPTURED"
    except Exception as exc:
        result["status"] = "FAILED"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        (run_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if env is not None:
            cwd = Path(env._tmp_folder)
            snap = run_dir / "runtime_snapshot"
            snap.mkdir(exist_ok=True)
            names = {"CNFQ.CLI", "FILEX.MZX", "DSSAT-PDI.YML", "ERROR.OUT", "INFO.OUT", "WARNING.OUT", "RUNLIST.OUT", "DSSAT48.INP", "DSSAT48.INH"}
            for path in cwd.iterdir() if cwd.exists() else []:
                if path.is_file() and (path.name.upper() in names or path.suffix.upper() == ".WTH"):
                    shutil.copy2(path, snap / path.name)
            result["runtime_snapshot_files"] = [p.name for p in sorted(snap.iterdir())]
            result["new_or_changed_wth"] = []
            try:
                env.close()
            except Exception as exc:
                result["close_error"] = str(exc)
        if cls is not None and originals is not None:
            cls._get_state, cls._get_sockets_ = originals
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["source_mzx_sha256_after"] = digest(SOURCE_MZX)
        result["cli_sha256_after"] = digest(CLI)
        save_json(run_dir / "result.json", result)
    print(json.dumps({key: result.get(key) for key in ("attempt", "status", "steps", "weather_row_count", "weather_sha256", "error", "elapsed_seconds")}, ensure_ascii=False))
    return 0 if result["status"] == "RUNTIME_WEATHER_CAPTURED" else 2


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--prepare":
        print(json.dumps(prepare(), ensure_ascii=False))
    else:
        raise SystemExit(main())
