"""Isolated FQA multi-year WGEN PPO seed-0 10K training with exact weather archives."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import argparse
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
BASE = Path(__file__).resolve().parent
OUT = BASE / "attempt_01"
SMOKE_OUT = BASE / "smoke_attempt_01"
PROMPT = ROOT / "prompt_02/051_fqa_multiyear_wgen_ppo_100k_with_checkpoints.md"
SOURCE_045 = ROOT / "results/fqa_wgen_multiyear_heldout_gate_045/run_gate.py"
PLAN_045 = ROOT / "results/fqa_wgen_multiyear_heldout_gate_045/full_schedule_plan.json"
GATE_046 = ROOT / "results/fqa_wgen_full_pool_qc_046/final_gate.json"
CLI = ROOT / "results/fqa_weather_resume_041/CNFQ.CLI"
TOTAL_STEPS = 100_000
SMOKE_STEPS = 432
CHECKPOINT_STEPS = [25_000, 50_000, 75_000, 100_000]
RSS_LIMIT_MB = 1536
WALL_LIMIT_S = 7200

spec = importlib.util.spec_from_file_location("fqa_gate_045_for_047", SOURCE_045)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
yc = gate.yc
smoke = gate.smoke


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def schedule_rows(count: int):
    import numpy as np
    rng_year = np.random.default_rng(64003)
    rng_seed = np.random.default_rng(64004)
    years = []
    seeds = []
    while len(years) < count:
        years.extend(int(x) for x in rng_year.permutation(list(range(2005, 2014))))
    while len(seeds) < count:
        seeds.extend(int(x) for x in rng_seed.permutation(list(range(1001, 1081))))
    schedule = [{"episode_index": i + 1, "historical_year": years[i], "weather_seed": seeds[i], "pool": "train"} for i in range(count)]
    frozen = json.loads(PLAN_045.read_text(encoding="utf-8"))
    if schedule[:80] != frozen["train"]:
        raise RuntimeError("first 80 training episodes differ from frozen 045 plan")
    return schedule


class TrainingEnv(gate.ArchivedEval):
    def __init__(self, config, env_config, schedule):
        self.split = "train"
        self.run_dir = OUT
        self.archive_rows = []
        yc.ScheduledEpisodeEnv.__init__(self, config, env_config, schedule, "RANDOM_WEATHER_WGEN", 0, "fqa_051_multiyear_100k_train", 66004, max_cached_years=1)

    def archive_partial(self, status: str):
        from scripts.run_yc_wgen_seed_pilot import canonical_weather_bytes, daily_weather_from_states, physical_sanity, runtime_filex_mode, pdi_seed_configured
        row = self.current_schedule
        if row is None or self.episode_days <= 0:
            return
        index = int(row["episode_index"])
        if any(int(x["episode_index"]) == index for x in self.archive_rows):
            return
        year = int(row["historical_year"])
        weather_seed = int(row["weather_seed"])
        dssat = yc.find_dssat_instance(self.current_env)
        year_info = smoke.fq.engine.base03222.direct_ppo.find_year(self.env_config, "FQA", year)
        daily = daily_weather_from_states(dssat._pilot_weather_states, start_date=date.fromisoformat(year_info["planting_date"]))
        path = OUT / "weather_daily" / f"episode_{index:04d}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = canonical_weather_bytes(daily)
        with path.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        digest = hashlib.sha256(data).hexdigest().upper()
        if sha(path) != digest or len(daily) != self.episode_days:
            raise RuntimeError(f"partial episode {index}: weather byte/step mismatch")
        screen = physical_sanity(daily)
        runtime_dir = Path(dssat._tmp_folder).resolve()
        if not runtime_dir.is_relative_to(ROOT):
            raise RuntimeError("runtime directory outside project")
        files = {p.name.upper(): p for p in runtime_dir.iterdir() if p.is_file()}
        filex, cli, pdi = (files.get(name) for name in ("FILEX.MZX", "CNFQ.CLI", "DSSAT-PDI.YML"))
        if any(p is None for p in (filex, cli, pdi)):
            raise RuntimeError("partial episode runtime proof missing")
        proof = {"episode_index": index, "runtime_dir": rel(runtime_dir), "runtime_filex_sha256": sha(filex), "wther": runtime_filex_mode(filex.read_text(encoding="utf-8", errors="replace")), "wsta_confirmed": "CNFQ" in filex.read_text(encoding="utf-8", errors="replace"), "runtime_cli_sha256": sha(cli), "source_cli_sha256": sha(CLI), "pdi_config_sha256": sha(pdi), "yaml_bootstrap_seed": int(dssat._pilot_bootstrap_seed), "yaml_bootstrap_confirmed": pdi_seed_configured(pdi.read_text(encoding="utf-8", errors="replace"), int(dssat._pilot_bootstrap_seed)), "runtime_rseed1": int(dssat._rseed1), "scheduled_seed": weather_seed}
        if proof["wther"] != "W" or not proof["wsta_confirmed"] or proof["runtime_cli_sha256"] != proof["source_cli_sha256"] or not proof["yaml_bootstrap_confirmed"] or proof["runtime_rseed1"] != weather_seed:
            raise RuntimeError(f"partial episode {index}: runtime proof mismatch")
        proof_path = OUT / "runtime_evidence" / f"episode_{index:04d}.json"
        proof_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(proof_path, proof)
        record = {"split": "train", "episode_index": index, "year": year, "ppo_seed": 0, "weather_seed": weather_seed, "pdi_rseed1": int(dssat._rseed1), "days": self.episode_days, "weather_rows": len(daily), "weather_path": rel(path), "weather_sha256": digest, "runtime_evidence_path": rel(proof_path), "physical_status": screen["status"], "reward": round(self.episode_reward, 8), "status": status}
        self.archive_rows.append(record)
        with (OUT / "episode_manifest.csv").open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(record)
        if screen["status"] != "PASS":
            raise RuntimeError(f"partial episode {index}: physical screen {screen}")


def main():
    import numpy as np
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    global OUT

    parser = argparse.ArgumentParser(description="FQA seed-0 100K WGEN training with YC-style checkpoints and weather archives")
    parser.add_argument("--smoke", action="store_true", help="run the isolated 432-step archive smoke")
    args = parser.parse_args()
    is_smoke = bool(args.smoke)
    OUT = SMOKE_OUT if is_smoke else BASE / "attempt_01"

    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing attempt: {OUT}")
    if json.loads(GATE_046.read_text(encoding="utf-8"))["status"] != "PASS_INPUT_QC_ONLY":
        raise RuntimeError("046 input QC gate not passed")
    if json.loads((ROOT / "results/fqa_multiyear_wgen_ppo_047/attempt_01/final_gate.json").read_text(encoding="utf-8"))["status"] != "PASS_10K_ARCHIVE_ONLY":
        raise RuntimeError("047 FQA 10K archive gate not passed")
    if json.loads((ROOT / "results/fqa_047_checkpoint_full20_heldout_050/final_gate.json").read_text(encoding="utf-8"))["status"] != "PASS_CHECKPOINT_FULL20_PAIRED_ARCHIVE_ONLY":
        raise RuntimeError("050 full-20 heldout archive gate not passed")
    target_steps = SMOKE_STEPS if is_smoke else TOTAL_STEPS
    checkpoint_targets = [target_steps] if is_smoke else CHECKPOINT_STEPS
    schedule = schedule_rows(256 if is_smoke else TOTAL_STEPS)
    OUT.mkdir(parents=True)
    runtime_temp = OUT / "temp"
    runtime_temp.mkdir()
    os.environ["TMPDIR"] = str(runtime_temp)
    tempfile.tempdir = str(runtime_temp)
    write_json(OUT / "planned_episode_schedule.json", schedule)
    input_sources = (
        PROMPT,
        SOURCE_045,
        PLAN_045,
        GATE_046,
        CLI,
        Path(__file__),
        ROOT / "results/fqa_multiyear_wgen_ppo_047/attempt_01/effective_config.json",
        ROOT / "results/fqa_multiyear_wgen_ppo_047/attempt_01/final_gate.json",
        ROOT / "results/fqa_047_checkpoint_full20_heldout_050/final_gate.json",
    )
    write_json(OUT / "source_manifest.json", {rel(p): sha(p) for p in input_sources})
    smoke.OUT = OUT
    cfg, env_cfg, pf = smoke.preflight()
    frozen = json.loads((ROOT / "results/fqa_multiyear_wgen_ppo_047/attempt_01/effective_config.json").read_text(encoding="utf-8"))
    for key in ("ppo", "reward", "discrete_actions", "action_safety"):
        if cfg[key] != frozen[key]:
            raise RuntimeError(f"FQA 047 frozen contract drift: {key}")
    cfg["total_timesteps"] = TOTAL_STEPS
    write_json(OUT / "preflight.json", {"canonical": pf, "mode": "smoke" if is_smoke else "formal_100k", "prompt_sha256": sha(PROMPT), "schedule_count": len(schedule), "frozen_contract_keys": ["ppo", "reward", "discrete_actions", "action_safety"]})
    write_json(OUT / "effective_config.json", cfg)

    def make_env(*a, **kw):
        env = smoke.make_fq_env(*a, **kw)
        dssat = yc.find_dssat_instance(env)
        dssat._pilot_bootstrap_seed = int(a[-1] if a else kw["bootstrap_weather_seed"])
        return env

    yc.make_weather_env = make_env
    yc.TASK_ROOT = OUT
    yc.MAX_PROCESS_TREE_RSS_MB = RSS_LIMIT_MB
    random.seed(0)
    np.random.seed(0)
    torch.set_num_threads(1)
    started = time.monotonic()

    class WallLimit(BaseCallback):
        def _on_step(self) -> bool:
            return time.monotonic() - started < WALL_LIMIT_S

    outcome = {"status": "FAILED", "mode": "smoke" if is_smoke else "formal_100k", "requested_timesteps": target_steps, "ppo_seed": 0, "training_years": list(range(2005, 2014)), "training_seed_pool": [1001, 1080], "heldout_seed_pool_excluded": [1081, 1100], "checkpoint_steps_requested": checkpoint_targets}
    env = None
    callback = None
    model = None
    try:
        env = TrainingEnv(cfg, env_cfg, schedule)
        callback = yc.TrainingCallback(env, OUT, checkpoint_targets, smoke=is_smoke)
        model = MaskablePPO("MlpPolicy", env._gym_env, verbose=0, seed=0, **smoke.fq.engine.base03222.base.ppo_kwargs(cfg))
        model.learn(total_timesteps=target_steps, reset_num_timesteps=True, progress_bar=False, callback=CallbackList([callback.callback, WallLimit()]))
        callback.flush_episodes()
        if env.current_schedule is not None and env.episode_days > 0:
            env.archive_partial("partial_at_stop")
        actual = int(model.num_timesteps)
        archived = sum(int(row["days"]) for row in env.archive_rows)
        resource_path = OUT / "resource_usage.csv"
        with resource_path.open(encoding="utf-8", newline="") as f:
            resources = list(csv.DictReader(f))
        peak_rss = max(float(r["process_tree_rss_mb"]) for r in resources)
        if actual < target_steps or archived != actual or callback.stop_for_memory or peak_rss >= RSS_LIMIT_MB or time.monotonic() - started >= WALL_LIMIT_S:
            raise RuntimeError(f"step/archive/resource gate failed: actual={actual}, archived={archived}, peak_rss={peak_rss}, memory_stop={callback.stop_for_memory}, elapsed={time.monotonic() - started:.1f}")
        if not all((OUT / "models" / f"checkpoint_{x}.zip").is_file() for x in checkpoint_targets):
            raise RuntimeError("checkpoint missing")
        model_path = OUT / "models" / f"final_model_actual_{actual}.zip"
        model.save(str(model_path))
        MaskablePPO.load(str(model_path), device="cpu")
        outcome.update({"status": "PASS_432_STEP_SMOKE_ARCHIVE_ONLY" if is_smoke else "PASS_100K_ARCHIVE_ONLY", "actual_timesteps": actual, "archived_weather_days": archived, "completed_episodes": sum(r["status"] == "completed" for r in env.archive_rows), "partial_episodes": sum(r["status"] == "partial_at_stop" for r in env.archive_rows), "episode_archives": len(env.archive_rows), "unique_weather_hashes": len({r["weather_sha256"] for r in env.archive_rows}), "checkpoint_steps_actual": callback.actual_checkpoint_steps, "peak_process_tree_rss_mb": round(peak_rss, 2), "model_path": rel(model_path), "model_sha256": sha(model_path)})
    except BaseException:
        outcome["error"] = traceback.format_exc()
        (OUT / "failure_traceback.txt").write_text(outcome["error"], encoding="utf-8")
        if model is not None:
            try:
                emergency_path = OUT / "models" / f"checkpoint_emergency_step_{int(model.num_timesteps)}.zip"
                emergency_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(emergency_path))
                outcome["emergency_checkpoint"] = rel(emergency_path)
            except Exception as exc:
                outcome["emergency_checkpoint_error"] = str(exc)
        if env is not None:
            try:
                env.archive_partial("partial_after_failure")
            except Exception as exc:
                outcome["partial_archive_error"] = str(exc)
    finally:
        if callback is not None:
            callback.flush_episodes()
        if env is not None:
            env.close()
        outcome["elapsed_seconds"] = round(time.monotonic() - started, 2)
        outcome["episode_archives"] = len(env.archive_rows) if env is not None else 0
        outcome["checkpoint_steps_actual"] = callback.actual_checkpoint_steps if callback is not None else {}
        write_json(OUT / "run_result.json", outcome)
    print(json.dumps({k: v for k, v in outcome.items() if k != "error"}, ensure_ascii=False), flush=True)
    expected_status = "PASS_432_STEP_SMOKE_ARCHIVE_ONLY" if is_smoke else "PASS_100K_ARCHIVE_ONLY"
    return 0 if outcome["status"] == expected_status else 2


if __name__ == "__main__":
    raise SystemExit(main())
