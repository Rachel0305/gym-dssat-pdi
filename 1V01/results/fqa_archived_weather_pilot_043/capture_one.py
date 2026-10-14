"""One isolated YC-control or FQA WGEN episode with full runtime weather archive."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MAX_STEPS = 240
MAX_SECONDS = 120
RSS_LIMIT_MB = 768
CONFIG = {
    "YC": {
        "template": ROOT / "results/yc_random_weather_ppo/004_03/runtime_templates/RANDOM_WEATHER_WGEN_ppo_seed_0_2007/YCA_2007_rseed1_1066.jinja2",
        "cli": ROOT / "results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI",
        "cultivar": ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/YC/MZCER048.CUL",
        "soil": ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/YC/SOIL.SOL",
        "station": "CNYC", "wsta": "CNYC0701", "start": date(2007, 6, 14), "allowed_seeds": {1066},
    },
    "FQA": {
        "template": ROOT / "results/fqa_runtime_wgen_smoke_042/inputs/fileX_after_W.jinja2",
        "cli": ROOT / "results/fqa_weather_resume_041/CNFQ.CLI",
        "cultivar": ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/FQ/MZCER048.CUL",
        "soil": ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/FQ/SOIL.SOL",
        "station": "CNFQ", "wsta": "CNFQ0701", "start": date(2007, 6, 1), "allowed_seeds": set(range(1001, 1006)),
    },
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


def save_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=CONFIG, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = CONFIG[args.site]
    if args.seed not in config["allowed_seeds"]:
        raise ValueError(f"seed {args.seed} outside frozen 043 pilot set")
    expected_ids = {"yc_2007_s1066"} | {f"fq_2007_s{s}" for s in range(1001, 1006)} | {"fq_2007_s1001_repeat"}
    if args.run_id not in expected_ids or (args.site == "YC") != args.run_id.startswith("yc_") or f"s{args.seed}" not in args.run_id:
        raise ValueError("run-id/site/seed mismatch")
    run_dir = OUT / "runtime" / args.run_id
    if run_dir.exists():
        raise FileExistsError(run_dir)
    for key in ("template", "cli", "cultivar", "soil"):
        if not config[key].is_file():
            raise FileNotFoundError(config[key])
    run_dir.mkdir(parents=True)
    temp_root = run_dir / "temp"
    temp_root.mkdir()
    os.environ["TMPDIR"] = str(temp_root)
    tempfile.tempdir = str(temp_root)
    source_manifest = {key: {"path": str(config[key].relative_to(ROOT)).replace("\\", "/"), "sha256": digest(config[key])} for key in ("template", "cli", "cultivar", "soil")}
    save_json(run_dir / "input_manifest.json", source_manifest)
    sys.path.insert(0, str(ROOT))
    from scripts import run_yc_wgen_seed_pilot as helper

    started = time.monotonic()
    env = cls = originals = None
    states = []
    result = {"run_id": args.run_id, "site": args.site, "year": 2007, "treatment": 1, "weather_seed": args.seed,
              "capture_context": "fresh process; all-zero controllable actions; separate project-local runtime directory", "status": "STARTED",
              "formal_training": False, "ppo_executed": False}
    try:
        import gym
        import gym_dssat_pdi.envs.dssat_pdi as dssat_module

        cls = helper._runtime_class(dssat_module)
        originals = (cls._get_state, cls._get_sockets_)

        def get_state(instance):
            value = originals[0](instance)
            state = getattr(instance, "_state", None)
            if isinstance(state, dict):
                states.append(copy.deepcopy(state))
            return value

        def get_sockets(instance):
            instance._rseed1 = args.seed
            value = originals[1](instance)
            if getattr(instance, "_server", None) is not None:
                import zmq
                instance._server.setsockopt(zmq.RCVTIMEO, 30000)
            return value

        cls._get_state = get_state
        cls._get_sockets_ = get_sockets
        env = gym.make(
            "gym_dssat_pdi:GymDssatPdi-v0", log_saving_path=str(run_dir / "runtime_log.txt"), mode="all",
            auxiliary_file_paths=[str(config["cultivar"]), str(config["cli"]), str(config["soil"])],
            random_weather=True, seed=None, fileX_template_path=str(config["template"]),
            experiment_number=1, evaluation=True, cultivar="maize", run_dssat_location="/opt/dssat_pdi/run_dssat",
        ).unwrapped
        cwd = Path(env._tmp_folder)
        result["runtime_cwd"] = str(cwd)
        runtime_cli = cwd / f"{config['station']}.CLI"
        result["runtime_cli_sha256"] = digest(runtime_cli) if runtime_cli.is_file() else None
        if result["runtime_cli_sha256"] != source_manifest["cli"]["sha256"]:
            raise RuntimeError("runtime CLI missing or differs from frozen source")
        rendered_filex = (cwd / "fileX.MZX").read_text(encoding="utf-8", errors="replace")
        result["runtime_filex_wther"] = helper.runtime_filex_mode(rendered_filex)
        result["runtime_filex_wsta_present"] = config["wsta"] in rendered_filex
        if result["runtime_filex_wther"] != "W" or not result["runtime_filex_wsta_present"]:
            raise RuntimeError("runtime FileX WTHER/WSTA mismatch")
        yml = cwd / "dssat-pdi.yml"
        result["runtime_seed_confirmed"] = yml.is_file() and helper.pdi_seed_configured(yml.read_text(encoding="utf-8", errors="replace"), args.seed)
        if not result["runtime_seed_confirmed"]:
            raise RuntimeError("PDI rseed1_ not confirmed")
        action = {key: 0.0 for key in env.action_variables}
        steps = 0
        peak_rss = 0.0
        while not bool(getattr(env, "done", False)) and steps < MAX_STEPS:
            if time.monotonic() - started > MAX_SECONDS:
                raise TimeoutError("wall-time limit exceeded")
            env.step(action)
            steps += 1
            peak_rss = max(peak_rss, helper._process_tree_rss_mb())
            if peak_rss > RSS_LIMIT_MB:
                raise MemoryError("process-tree RSS limit exceeded")
        result["steps"] = steps
        result["peak_process_tree_rss_mb"] = peak_rss
        result["episode_done"] = bool(getattr(env, "done", False))
        if not result["episode_done"]:
            raise TimeoutError("step limit exceeded")
        weather = helper.daily_weather_from_states(states, start_date=config["start"])
        result["state_count"] = len(states)
        result["weather_row_count"] = len(weather)
        result["weather_summary"] = helper.summarize_weather(weather)
        result["physical_sanity"] = helper.physical_sanity(weather)
        if len(weather) != steps or result["physical_sanity"]["status"] != "PASS":
            raise RuntimeError("runtime weather capture incomplete or physical screening failed")
        data = helper.canonical_weather_bytes(weather)
        weather_path = run_dir / "runtime_weather_daily.csv"
        weather_path.write_bytes(data)
        result["weather_archive"] = str(weather_path.relative_to(ROOT)).replace("\\", "/")
        result["weather_sha256"] = hashlib.sha256(data).hexdigest().upper()
        result["status"] = "ARCHIVED_RUNTIME_WEATHER"
    except Exception as exc:
        result["status"] = "FAILED"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        (run_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if env is not None:
            cwd = Path(env._tmp_folder)
            snapshot = run_dir / "runtime_snapshot"
            snapshot.mkdir(exist_ok=True)
            names = {f"{config['station']}.CLI", "FILEX.MZX", "DSSAT-PDI.YML", "ERROR.OUT", "INFO.OUT", "WARNING.OUT", "RUNLIST.OUT", "DSSAT48.INP", "DSSAT48.INH"}
            for path in cwd.iterdir() if cwd.exists() else []:
                if path.is_file() and (path.name.upper() in names or path.suffix.upper() == ".WTH"):
                    shutil.copy2(path, snapshot / path.name)
            result["runtime_snapshot_files"] = [p.name for p in sorted(snapshot.iterdir())]
            try:
                env.close()
            except Exception as exc:
                result["close_error"] = str(exc)
        if cls is not None and originals is not None:
            cls._get_state, cls._get_sockets_ = originals
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["input_hashes_unchanged"] = all(digest(config[key]) == source_manifest[key]["sha256"] for key in source_manifest)
        save_json(run_dir / "result.json", result)
    print(json.dumps({key: result.get(key) for key in ("run_id", "status", "steps", "weather_row_count", "weather_sha256", "error", "elapsed_seconds")}, ensure_ascii=False))
    return 0 if result["status"] == "ARCHIVED_RUNTIME_WEATHER" else 2


if __name__ == "__main__":
    raise SystemExit(main())
