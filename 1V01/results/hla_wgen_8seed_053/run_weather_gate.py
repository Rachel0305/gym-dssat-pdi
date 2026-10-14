"""FQA WGEN nine-year and heldout archive gate; evaluation only, no PPO learning."""
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
OUT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "src", ROOT / "src/054_hla_lowIC_site_transfer", ROOT / "results/yc_random_weather_ppo/004_03"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_controlled_pilot as yc

spec = importlib.util.spec_from_file_location("fqa_smoke_044", ROOT / "results/hla_wgen_8seed_053/run_ppo_smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)

MODEL = ROOT / "results/hla_wgen_8seed_053/smoke_gpcc_raw_432_attempt02/models/hla_ppo_seed0_432.zip"
PROMPT = ROOT / "prompt_02/053_hla_wgen_8seed_resume.md"
YEARS = list(range(2004, 2014))
TRAIN_SEEDS = list(range(1001, 1081))
HELDOUT_SEEDS = list(range(1081, 1101))
RSS_LIMIT_MB = 1536


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def build_plan():
    import numpy as np
    year_rng = np.random.default_rng(64003)
    train_seed_rng = np.random.default_rng(64004)
    years = []
    while len(years) < 80:
        years.extend(int(x) for x in year_rng.permutation(YEARS))
    seeds = [int(x) for x in train_seed_rng.permutation(TRAIN_SEEDS)]
    train = [{"episode_index": i + 1, "historical_year": years[i], "weather_seed": seeds[i], "pool": "train"} for i in range(80)]
    heldout = [{"episode_index": i + 1, "historical_year": 2007, "weather_seed": seed, "pool": "heldout"} for i, seed in enumerate(HELDOUT_SEEDS)]
    return {"training_year_schedule_seed": 64003, "training_weather_schedule_seed": 64004, "train": train, "heldout": heldout, "actual_train_episode_indices": list(range(1, 10)), "actual_heldout_seeds": [1081, 1100]}


class ArchivedEval(yc.ScheduledEpisodeEnv):
    def __init__(self, config, env_config, schedule, split, run_dir):
        self.split = split
        self.run_dir = run_dir
        self.archive_rows = []
        super().__init__(config, env_config, schedule, "RANDOM_WEATHER_WGEN", 0, f"hla_053_{split}", 66004, max_cached_years=1)

    def archive_current(self):
        from scripts.run_yc_wgen_seed_pilot import canonical_weather_bytes, daily_weather_from_states, physical_sanity, runtime_filex_mode, pdi_seed_configured
        row = self.current_schedule
        index = int(row["episode_index"])
        year = int(row["historical_year"])
        weather_seed = int(row["weather_seed"])
        dssat = yc.find_dssat_instance(self.current_env)
        year_info = smoke.fq.engine.base03222.direct_ppo.find_year(self.env_config, "HLA", year)
        daily = daily_weather_from_states(dssat._pilot_weather_states, start_date=date.fromisoformat(year_info["planting_date"]))
        if len(daily) != self.episode_days or self.episode_days == 0:
            raise RuntimeError(f"episode {index}: captured weather rows {len(daily)} != steps {self.episode_days}")
        screen = physical_sanity(daily)
        if screen["status"] != "PASS":
            raise RuntimeError(f"episode {index}: weather physical screen {screen}")
        weather_path = self.run_dir / "weather_daily" / f"episode_{index:04d}.csv"
        weather_path.parent.mkdir(parents=True, exist_ok=True)
        data = canonical_weather_bytes(daily)
        with weather_path.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        weather_hash = hashlib.sha256(data).hexdigest().upper()
        if sha(weather_path) != weather_hash:
            raise RuntimeError(f"episode {index}: disk weather hash mismatch")
        runtime_dir = Path(dssat._tmp_folder).resolve()
        if not runtime_dir.is_relative_to(ROOT):
            raise RuntimeError(f"episode {index}: runtime directory outside project: {runtime_dir}")
        files = {p.name.upper(): p for p in runtime_dir.iterdir() if p.is_file()}
        filex, cli, pdi = (files.get(name) for name in ("FILEX.MZX", "CNHL.CLI", "DSSAT-PDI.YML"))
        if any(p is None for p in (filex, cli, pdi)):
            raise RuntimeError(f"episode {index}: runtime files missing")
        filex_text = filex.read_text(encoding="utf-8", errors="replace")
        pdi_text = pdi.read_text(encoding="utf-8", errors="replace")
        bootstrap_seed = int(dssat._pilot_bootstrap_seed)
        proof = {"episode_index": index, "runtime_dir": rel(runtime_dir), "runtime_filex_sha256": sha(filex), "wther": runtime_filex_mode(filex_text), "wsta_confirmed": "CNHL" in filex_text, "runtime_cli_sha256": sha(cli), "source_cli_sha256": sha(smoke.CLI), "pdi_config_sha256": sha(pdi), "yaml_bootstrap_seed": bootstrap_seed, "yaml_bootstrap_confirmed": pdi_seed_configured(pdi_text, bootstrap_seed), "runtime_rseed1": int(dssat._rseed1), "scheduled_seed": weather_seed}
        if proof["wther"] != "W" or not proof["wsta_confirmed"] or proof["runtime_cli_sha256"] != proof["source_cli_sha256"] or not proof["yaml_bootstrap_confirmed"] or proof["runtime_rseed1"] != weather_seed:
            raise RuntimeError(f"episode {index}: runtime evidence mismatch: {proof}")
        proof_path = self.run_dir / "runtime_evidence" / f"episode_{index:04d}.json"
        proof_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(proof_path, proof)
        record = {"split": self.split, "episode_index": index, "year": year, "ppo_seed": 0, "weather_seed": weather_seed, "pdi_rseed1": int(dssat._rseed1), "days": self.episode_days, "weather_rows": len(daily), "weather_path": rel(weather_path), "weather_sha256": weather_hash, "runtime_evidence_path": rel(proof_path), "physical_status": screen["status"], "reward": round(self.episode_reward, 8), "status": "completed"}
        self.archive_rows.append(record)
        with (self.run_dir / "episode_manifest.csv").open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(record)

    def _step(self, action):
        result = super()._step(action)
        if result[2] or result[3]:
            self.archive_current()
        return result


def main():
    import numpy as np
    import torch
    from sb3_contrib import MaskablePPO

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "heldout"])
    args = parser.parse_args()
    split = args.split
    run_dir = OUT / split
    if run_dir.exists():
        raise FileExistsError(run_dir)
    if not MODEL.is_file() or not PROMPT.is_file():
        raise FileNotFoundError("044 model or 045 prompt missing")
    run_dir.mkdir(parents=True)
    temp = run_dir / "temp"
    temp.mkdir()
    os.environ["TMPDIR"] = str(temp)
    tempfile.tempdir = str(temp)
    plan = build_plan()
    if split == "train":
        write_json(OUT / "full_schedule_plan.json", plan)
    else:
        frozen = json.loads((OUT / "full_schedule_plan.json").read_text(encoding="utf-8"))
        if frozen != plan:
            raise RuntimeError("full schedule plan changed between split processes")
    schedule = plan["train"][:9] if split == "train" else [plan["heldout"][0], plan["heldout"][-1]]
    if split == "heldout":
        schedule = [{**row, "episode_index": i + 1} for i, row in enumerate(schedule)]
    write_json(run_dir / "actual_schedule.json", schedule)
    smoke.OUT = run_dir
    cfg, env_cfg, preflight = smoke.preflight()
    write_json(run_dir / "preflight.json", {**preflight, "model_path": rel(MODEL), "model_sha256": sha(MODEL), "prompt_sha256": sha(PROMPT)})
    def make_env(*a, **kw):
        env = smoke.make_fq_env(*a, **kw)
        dssat = yc.find_dssat_instance(env)
        dssat._pilot_bootstrap_seed = int(a[-1] if a else kw["bootstrap_weather_seed"])
        return env
    yc.make_weather_env = make_env
    yc.TASK_ROOT = run_dir
    random.seed(0)
    np.random.seed(0)
    torch.set_num_threads(1)
    env = None
    started = time.monotonic()
    outcome = {"split": split, "status": "FAILED", "episodes_expected": len(schedule)}
    try:
        model = MaskablePPO.load(str(MODEL), device="cpu")
        env = ArchivedEval(cfg, env_cfg, schedule, split, run_dir)
        resource_rows = []
        for row in schedule:
            obs, _ = env._gym_env.reset(seed=None)
            days = 0
            while True:
                action, _ = model.predict(obs, deterministic=True, action_masks=env._gym_env.action_masks())
                obs, reward, done, truncated, _ = env._gym_env.step(action)
                days += 1
                if days > 260:
                    raise RuntimeError(f"episode {row['episode_index']}: >260 days")
                if done or truncated:
                    break
            rss = yc._tree_rss_mb()
            resource_rows.append({"episode_index": row["episode_index"], "process_tree_rss_mb": rss})
            print(f"[{split}] episode={row['episode_index']} year={row['historical_year']} seed={row['weather_seed']} days={days} rss_mb={rss:.1f}", flush=True)
            if rss >= RSS_LIMIT_MB:
                raise MemoryError(f"process tree RSS {rss:.1f} MB exceeds {RSS_LIMIT_MB}")
        with (run_dir / "resource_usage.csv").open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode_index", "process_tree_rss_mb"])
            writer.writeheader()
            writer.writerows(resource_rows)
        outcome.update({"status": "PASS_ARCHIVE_GATE", "episodes_completed": len(env.archive_rows), "days_archived": sum(int(r["days"]) for r in env.archive_rows), "peak_rss_mb": max(r["process_tree_rss_mb"] for r in resource_rows)})
    except Exception:
        outcome["error"] = traceback.format_exc()
        (run_dir / "failure_traceback.txt").write_text(outcome["error"], encoding="utf-8")
    finally:
        if env is not None:
            env.close()
        outcome["elapsed_seconds"] = round(time.monotonic() - started, 2)
        write_json(run_dir / "result.json", outcome)
    print(json.dumps({k: v for k, v in outcome.items() if k != "error"}, ensure_ascii=False), flush=True)
    return 0 if outcome["status"] == "PASS_ARCHIVE_GATE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
