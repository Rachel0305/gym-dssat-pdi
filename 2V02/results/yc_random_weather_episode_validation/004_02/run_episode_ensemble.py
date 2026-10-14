from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = PROJECT_ROOT / "results" / "yc_random_weather_episode_validation" / "004_02"
BASE_RUNNER = PROJECT_ROOT / "results" / "yc_wgen_cli_pilot" / "003_06_06" / "crop_smoke" / "run_dssat_crop_smoke.py"
SOURCE_FILEX = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "YC" / "CNYC0801.MZX"
CULTIVAR = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "YC" / "MZCER048.CUL"
SOIL = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "YC" / "SOIL.SOL"
CLI = PROJECT_ROOT / "results" / "yc_wgen_cli_pilot" / "003_06_05_02" / "final" / "CNYC.CLI"
TRAIN_WEATHER = PROJECT_ROOT / "results" / "yc_weather_gapfill_finalize" / "yc_wgen_fitting_weather_2004_2013.csv"
EXPECTED_CLI_SHA256 = "65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929"
EXPECTED_TRAIN_SHA256 = "4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34"
SEED_MIN, SEED_MAX = 1001, 1100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_runner():
    sys.path.insert(0, str(PROJECT_ROOT))
    spec = importlib.util.spec_from_file_location("yc_crop_smoke_runner", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load existing crop smoke runner: {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_seeds(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            start_text, end_text = token.split(":", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Descending seed range is not allowed: {token}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    if not values or len(values) != len(set(values)):
        raise ValueError("Provide a non-empty seed list without duplicates")
    if any(seed < SEED_MIN or seed > SEED_MAX for seed in values):
        raise ValueError(f"Seeds must be within {SEED_MIN}-{SEED_MAX}")
    return values


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"Refusing to overwrite different provenance: {path}")
        return
    path.write_text(encoded, encoding="utf-8", newline="\n")


def prepare(module) -> Path:
    if not TASK_ROOT.resolve().is_relative_to(PROJECT_ROOT.resolve()):
        raise RuntimeError("Task output path escaped the project root")
    for path in (BASE_RUNNER, SOURCE_FILEX, CULTIVAR, SOIL, CLI, TRAIN_WEATHER):
        if not path.is_file():
            raise FileNotFoundError(path)
    cli_hash = sha256_file(CLI)
    train_hash = sha256_file(TRAIN_WEATHER)
    if cli_hash != EXPECTED_CLI_SHA256 or train_hash != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("Frozen CLI or fitting weather hash preflight failed")

    configs = TASK_ROOT / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    template = configs / "CNYC0801_crop_template_v4.jinja2"
    template_text = module._coords_in_isolated_filex(
        module.prepare_wgen_filex(SOURCE_FILEX.read_text(encoding="utf-8", errors="replace"))
    )
    if template.exists() and template.read_text(encoding="utf-8") != template_text:
        raise FileExistsError(f"Refusing to overwrite a different isolated template: {template}")
    if not template.exists():
        template.write_text(template_text, encoding="utf-8", newline="\n")

    provenance = {
        "scope": "YC only; 2008 treatment 1; random-weather WGEN episodes; zero action as in existing WGEN pilot",
        "seed_plan": {"generation": "1001-1100 inclusive", "reproducibility_reruns": [1001, 1025, 1050, 1075, 1100]},
        "source_filex": str(SOURCE_FILEX.relative_to(PROJECT_ROOT)),
        "source_filex_sha256": sha256_file(SOURCE_FILEX),
        "isolated_template": str(template.relative_to(PROJECT_ROOT)),
        "isolated_template_sha256": sha256_file(template),
        "template_transform": "Existing validated crop-smoke helper; treatment 1 only, WTHER=W, station coordinates written to isolated fixed-width fields; source MZX is unchanged.",
        "cultivar_file": str(CULTIVAR.relative_to(PROJECT_ROOT)),
        "cultivar_sha256": sha256_file(CULTIVAR),
        "soil_file": str(SOIL.relative_to(PROJECT_ROOT)),
        "soil_sha256": sha256_file(SOIL),
        "cli_file": str(CLI.relative_to(PROJECT_ROOT)),
        "cli_sha256": cli_hash,
        "fitting_weather_file": str(TRAIN_WEATHER.relative_to(PROJECT_ROOT)),
        "fitting_weather_sha256": train_hash,
        "runtime": "Existing project Docker runtime via /opt/gym_dssat_pdi/bin/python; no runtime changes.",
        "validation_weather_used_for_generation": False,
        "wgen_refit": False,
        "ppo_training": False,
    }
    _write_json(TASK_ROOT / "generation_provenance.json", provenance)
    return template


def main() -> int:
    parser = argparse.ArgumentParser(description="Serial YC random-weather WGEN episode generator")
    parser.add_argument("--mode", choices=("generate", "replay"), required=True)
    parser.add_argument("--seeds", required=True, help="Comma-separated seeds and/or inclusive ranges, e.g. 1001 or 1001:1024")
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    module = load_runner()
    template = prepare(module)

    output_base = TASK_ROOT / "episodes" if args.mode == "generate" else TASK_ROOT / "reproducibility" / "runs"
    output_base.mkdir(parents=True, exist_ok=True)
    runtime_tmp = TASK_ROOT / "runtime_tmp"
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(runtime_tmp)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    tempfile.tempdir = str(runtime_tmp)
    module.OUT = output_base

    for seed in seeds:
        run_id = f"seed_{seed}" if args.mode == "generate" else f"seed_{seed}_repeat"
        print(f"[{args.mode}] seed={seed} start serial", flush=True)
        result = module.run_one(run_id, "W", seed, template)
        run_dir = output_base / run_id
        state_file = run_dir / "daily_state.csv"
        if state_file.is_file():
            compact_name = f"seed_{seed}.csv" if args.mode == "generate" else f"seed_{seed}_repeat.csv"
            compact_dir = TASK_ROOT / ("episodes" if args.mode == "generate" else "reproducibility")
            compact_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(state_file, compact_dir / compact_name)
        print(json.dumps({
            "mode": args.mode,
            "seed": seed,
            "status": result["runtime_status"],
            "steps": result["runtime_steps"],
            "date_end": result["simulation_last_weather_date"],
            "runtime_error": result["runtime_error"],
        }, ensure_ascii=False), flush=True)
        if result["runtime_status"] != "PASS":
            print("Stopping at first failed episode; its complete diagnostic output is retained.", flush=True)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
