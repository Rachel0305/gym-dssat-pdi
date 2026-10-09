"""Generate HLA GPCC_RAW 80/20 WGEN realizations with daily archives."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
import tempfile
import time
import traceback
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "weather_pool"
SOURCE = ROOT / "results/hla_wgen_8seed_053/run_weather_gate.py"
PLAN = ROOT / "results/hla_wgen_8seed_053/weather_pool_plan.json"
MODEL = ROOT / "results/hla_wgen_8seed_053/smoke_gpcc_raw_432_attempt02/models/hla_ppo_seed0_432.zip"
FITTING = ROOT / "results/hla_weather_enhancement_029/weather_fitting/gpcc_raw/fitting_weather.csv"
CLI = ROOT / "results/hla_weather_enhancement_029/weather_fitting/gpcc_raw/CNHL.CLI"
PROMPT = ROOT / "prompt_02/053_hla_wgen_8seed_resume.md"
RSS_LIMIT_MB = 1536
WALL_LIMIT_S = 900

spec = importlib.util.spec_from_file_location("hla_gate_053", SOURCE)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_json(path: Path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def preserve_partial(env, run_dir: Path):
    if env is None or env.current_schedule is None or env.episode_days <= 0:
        return None
    from scripts.run_yc_wgen_seed_pilot import canonical_weather_bytes, daily_weather_from_states
    row = env.current_schedule
    year = int(row["historical_year"])
    info = gate.smoke.fq.engine.base03222.direct_ppo.find_year(env.env_config, "HLA", year)
    dssat = gate.yc.find_dssat_instance(env.current_env)
    daily = daily_weather_from_states(dssat._pilot_weather_states, start_date=date.fromisoformat(info["planting_date"]))
    path = run_dir / f"failure_partial_episode_{int(row['episode_index']):04d}.csv"
    with path.open("xb") as f:
        f.write(canonical_weather_bytes(daily))
    return {"path": rel(path), "days": len(daily), "sha256": sha(path), "scheduled_seed": int(row["weather_seed"]), "year": year}


def main():
    import numpy as np
    import torch
    from sb3_contrib import MaskablePPO

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=("train", "heldout"))
    args = parser.parse_args()
    split = args.split
    run_dir = OUT / split
    if run_dir.exists():
        raise FileExistsError(f"Refusing existing split output: {run_dir}")
    for path in (SOURCE, PLAN, MODEL, FITTING, CLI, PROMPT):
        if not path.is_file():
            raise FileNotFoundError(path)
    frozen = json.loads(PLAN.read_text(encoding="utf-8"))
    if frozen != gate.build_plan():
        raise RuntimeError("045 full schedule drift")
    inventory = list(csv.DictReader((ROOT / "results/hla_weather_enhancement_029/weather_fitting/scenario_fitting_inventory.csv").open(encoding="utf-8")))
    selected = next(row for row in inventory if row["scenario"] == "GPCC_RAW")
    if sha(FITTING).lower() != selected["fitting_sha256"] or sha(CLI).lower() != selected["cli_sha256"]:
        raise RuntimeError("029 selected fitting weather or CLI hash drift")
    schedule = frozen[split]
    run_dir.mkdir(parents=True)
    temp = run_dir / "temp"
    temp.mkdir()
    os.environ["TMPDIR"] = str(temp)
    tempfile.tempdir = str(temp)
    write_json(run_dir / "schedule.json", schedule)
    write_json(run_dir / "source_manifest.json", {rel(path): sha(path) for path in (SOURCE, PLAN, MODEL, FITTING, CLI, PROMPT, Path(__file__))})
    gate.smoke.OUT = run_dir
    cfg, env_cfg, pf = gate.smoke.preflight()
    write_json(run_dir / "preflight.json", pf)

    def make_env(*a, **kw):
        env = gate.smoke.make_fq_env(*a, **kw)
        dssat = gate.yc.find_dssat_instance(env)
        dssat._pilot_bootstrap_seed = int(a[-1] if a else kw["bootstrap_weather_seed"])
        return env

    gate.yc.make_weather_env = make_env
    gate.yc.TASK_ROOT = run_dir
    random.seed(0)
    np.random.seed(0)
    torch.set_num_threads(1)
    outcome = {"split": split, "status": "FAILED", "episodes_expected": len(schedule), "pool_seed_range": [min(x["weather_seed"] for x in schedule), max(x["weather_seed"] for x in schedule)]}
    env = None
    started = time.monotonic()
    try:
        model = MaskablePPO.load(str(MODEL), device="cpu")
        env = gate.ArchivedEval(cfg, env_cfg, schedule, split, run_dir)
        with (run_dir / "resource_usage.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["episode_index", "year", "weather_seed", "days", "process_tree_rss_mb", "elapsed_seconds"])
            writer.writeheader()
            peak = 0.0
            for row in schedule:
                obs, _ = env._gym_env.reset(seed=None)
                days = 0
                while True:
                    action, _ = model.predict(obs, deterministic=True, action_masks=env._gym_env.action_masks())
                    obs, _, done, truncated, _ = env._gym_env.step(action)
                    days += 1
                    if days % 10 == 0:
                        rss = gate.yc._tree_rss_mb()
                        peak = max(peak, rss)
                        if rss >= RSS_LIMIT_MB:
                            raise MemoryError(f"RSS {rss:.1f} MB exceeds {RSS_LIMIT_MB}")
                    if days > 260 or time.monotonic() - started > WALL_LIMIT_S:
                        raise TimeoutError(f"episode {row['episode_index']} exceeded step or wall limit")
                    if done or truncated:
                        break
                rss = gate.yc._tree_rss_mb()
                peak = max(peak, rss)
                writer.writerow({"episode_index": row["episode_index"], "year": row["historical_year"], "weather_seed": row["weather_seed"], "days": days, "process_tree_rss_mb": round(rss, 3), "elapsed_seconds": round(time.monotonic() - started, 2)})
                stream.flush()
                print(f"[{split}] {row['episode_index']}/{len(schedule)} year={row['historical_year']} seed={row['weather_seed']} days={days} rss={rss:.0f}MB", flush=True)
                if rss >= RSS_LIMIT_MB:
                    raise MemoryError(f"RSS {rss:.1f} MB exceeds {RSS_LIMIT_MB}")
        outcome.update({"status": "GENERATED_ARCHIVED", "episodes_completed": len(env.archive_rows), "days_archived": sum(int(r["days"]) for r in env.archive_rows), "unique_weather_hashes": len({r["weather_sha256"] for r in env.archive_rows}), "peak_rss_mb": round(peak, 2)})
    except Exception:
        outcome["error"] = traceback.format_exc()
        (run_dir / "failure_traceback.txt").write_text(outcome["error"], encoding="utf-8")
        try:
            outcome["partial_weather"] = preserve_partial(env, run_dir)
        except Exception as exc:
            outcome["partial_archive_error"] = str(exc)
    finally:
        if env is not None:
            env.close()
        outcome["elapsed_seconds"] = round(time.monotonic() - started, 2)
        write_json(run_dir / "result.json", outcome)
    print(json.dumps({k: v for k, v in outcome.items() if k != "error"}, ensure_ascii=False), flush=True)
    return 0 if outcome["status"] == "GENERATED_ARCHIVED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
