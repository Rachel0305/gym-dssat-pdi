"""FQA 2K WGEN PPO smoke with an archive for every used weather day."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import time
import traceback
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "attempt_05"
for p in (ROOT, ROOT / "src", ROOT / "src/051_fqa_originIC_site_transfer", ROOT / "results/yc_random_weather_ppo/004_03"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_controlled_pilot as yc
import run_051_00_fqa_originIC_expanded_action_maskableppo as fq

CLI = ROOT / "results/fqa_weather_resume_041/CNFQ.CLI"
PROMPT = ROOT / "prompt_02/044_fqa_wgen_ppo_2k_episode_archive_smoke.md"
STEPS = 2000
PPO_SEED = 0
YEAR = 2007
RSS_LIMIT_MB = 1536
WALL_LIMIT_S = 1800


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, data):
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def preflight():
    import numpy as np
    import ppo_safe_rendering as rendering

    original_root = rendering.MULTISITE_INPUT_ROOT
    rendering.MULTISITE_INPUT_ROOT = fq.INPUT_PROFILES["originIC"]
    fq.engine.BINARY_IRRIGATION_LEVELS = list(fq.EXPECTED_IRRIGATION_LEVELS)
    fq.engine.BINARY_NITROGEN_LEVELS = list(fq.EXPECTED_NITROGEN_LEVELS)
    fq.engine.base03222.SITES = ["FQA"]
    fq.engine.base03222.base.StressAwareDiscreteWrapper = fq.engine.base04036.LateIrrigationReserveMaskWrapper
    try:
        canonical = fq.read_config(fq.DEFAULT_CONFIG)
        checks = fq.preflight(canonical)
        if checks["issues"]:
            raise RuntimeError("FQA canonical preflight: " + "; ".join(checks["issues"]))
        cfg = fq.engine.load_config()
        cfg["seed"] = PPO_SEED
        cfg["total_timesteps"] = STEPS
        cfg["paths"]["output_root"] = rel(OUT)
        cfg["runtime"]["smoke_station"] = "FQA"
        cfg["runtime"]["max_steps"] = 260
        split = fq.engine.base03222.load_split()
        split = split[split.station_code.eq("FQA")]
        selection = fq.engine.base03222.build_selection(split)
        env_cfg = fq.engine.base03222.direct_ppo.build_env_config(cfg, selection)
        env_cfg["paths"]["output_root"] = rel(OUT)
        info = fq.engine.base03222.direct_ppo.find_year(env_cfg, "FQA", YEAR)
        if list(map(float, cfg["discrete_actions"]["irrigation_levels"])) != fq.EXPECTED_IRRIGATION_LEVELS:
            raise RuntimeError("I grid drift")
        if list(map(float, cfg["discrete_actions"]["nitrogen_levels"])) != fq.EXPECTED_NITROGEN_LEVELS:
            raise RuntimeError("N grid drift")
        if not CLI.is_file():
            raise FileNotFoundError(CLI)
        inputs = [fq.DEFAULT_CONFIG, PROMPT, CLI, Path(__file__), ROOT / "src/run_sya_lowIC_binary_timing_maskableppo_042_10.py", ROOT / "results/yc_random_weather_ppo/004_03/run_controlled_pilot.py"]
        return cfg, env_cfg, {"canonical_preflight": checks, "planting_date": info["planting_date"], "input_sha256": {rel(p): sha(p) for p in inputs}, "ppo_seed": PPO_SEED, "year": YEAR, "weather_seed_pool": [1001, 1080], "heldout_pool_excluded": [1081, 1100], "rss_limit_mb": RSS_LIMIT_MB}
    finally:
        rendering.MULTISITE_INPUT_ROOT = original_root


class SeedRng:
    def __init__(self, seed):
        self.seed = int(seed)
    def randint(self, low, high, **kwargs):
        return self.seed


def make_fq_env(cfg, env_cfg, year, regime, runtime_seed, run_tag, bootstrap_weather_seed):
    import gym
    import gym_dssat_pdi.envs.dssat_pdi as dssat_module
    import ppo_safe_rendering as rendering
    from sb3_wrapper import GymDssatWrapper

    if regime != "RANDOM_WEATHER_WGEN":
        raise ValueError(regime)
    original_root = rendering.MULTISITE_INPUT_ROOT
    rendering.MULTISITE_INPUT_ROOT = fq.INPUT_PROFILES["originIC"]
    try:
        year_info = fq.engine.base03222.direct_ppo.find_year(env_cfg, "FQA", int(year))
        args = rendering.build_env_args("FQA", int(year), year_info["planting_date"], int(runtime_seed), env_cfg, run_tag, evaluation=False, mode="all", linked_management=True)
    finally:
        rendering.MULTISITE_INPUT_ROOT = original_root
    source = Path(args["fileX_template_path"])
    source_text = source.read_text(encoding="utf-8", errors="replace")
    text, n = re.subn(r"(?m)^([ \t]*\d+[ \t]+ME[ \t]+)\S+", r"\g<1>W", source_text)
    if n == 0 or any(x != "W" for x in re.findall(r"(?m)^\s*\d+\s+ME\s+(\S+)", text)):
        raise RuntimeError("FileX WTHER=W rendering failed")
    template_dir = OUT / "runtime_templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    template = template_dir / f"FQA_{year}_rseed1_{bootstrap_weather_seed}.jinja2"
    if template.exists():
        if template.read_text(encoding="utf-8") != text:
            raise FileExistsError(template)
    else:
        template.write_text(text, encoding="utf-8", newline="\n")
    args["fileX_template_path"] = str(template)
    args["random_weather"] = True
    args["auxiliary_file_paths"] = list(args["auxiliary_file_paths"]) + [str(CLI)]
    class_type = getattr(dssat_module, "DssatPdi", None)
    if class_type is None:
        from scripts.run_yc_wgen_seed_pilot import _runtime_class
        class_type = _runtime_class(dssat_module)
    original = class_type._get_sockets_
    def inject(instance):
        instance._rseed1 = int(bootstrap_weather_seed)
        return original(instance)
    class_type._get_sockets_ = inject
    try:
        raw = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **args).unwrapped
    finally:
        class_type._get_sockets_ = original
    raw._random_generator = SeedRng(bootstrap_weather_seed)
    raw._pilot_filex_wther = "W"
    raw._pilot_filex_template_path = str(template)
    raw._pilot_filex_template_sha256 = sha(template)
    raw._pilot_weather_states = []
    original_step = raw.step
    def capture(*a, **kw):
        result = original_step(*a, **kw)
        state = getattr(raw, "_state", None)
        if isinstance(state, dict):
            raw._pilot_weather_states.append(copy.deepcopy(state))
        return result
    raw.step = capture
    wrapped = GymDssatWrapper(raw)
    return fq.engine.base03222.base.StressAwareDiscreteWrapper(wrapped, cfg)


class ArchivedEnv(yc.ScheduledEpisodeEnv):
    def __init__(self, cfg, env_cfg, schedule):
        self.archive_rows = []
        super().__init__(cfg, env_cfg, schedule, "RANDOM_WEATHER_WGEN", PPO_SEED, "fqa_044_train", 66003, max_cached_years=1)

    def archive_current(self, status):
        from scripts.run_yc_wgen_seed_pilot import canonical_weather_bytes, daily_weather_from_states, physical_sanity, runtime_filex_mode, pdi_seed_configured
        row = self.current_schedule
        if row is None:
            return
        index = int(row["episode_index"])
        if any(int(r["episode_index"]) == index for r in self.archive_rows):
            return
        dssat = yc.find_dssat_instance(self.current_env)
        states = getattr(dssat, "_pilot_weather_states", [])
        year_info = fq.engine.base03222.direct_ppo.find_year(self.env_config, "FQA", YEAR)
        start_date = date.fromisoformat(year_info["planting_date"])
        weather = daily_weather_from_states(states, start_date=start_date)
        if len(weather) != self.episode_days or len(weather) == 0:
            raise RuntimeError(f"weather coverage mismatch episode={index} rows={len(weather)} steps={self.episode_days}")
        screen = physical_sanity(weather)
        if screen["status"] != "PASS":
            raise RuntimeError(f"physical screen failed episode={index}: {screen}")
        data = canonical_weather_bytes(weather)
        path = OUT / "weather_daily" / f"episode_{index:04d}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        digest = hashlib.sha256(data).hexdigest().upper()
        if sha(path) != digest:
            raise RuntimeError(f"weather archive hash reread mismatch episode={index}")
        if int(dssat._rseed1) != int(row["weather_seed"]):
            raise RuntimeError(f"PDI seed mismatch episode={index}")
        runtime_dir = Path(dssat._tmp_folder).resolve()
        if not runtime_dir.is_relative_to(ROOT):
            raise RuntimeError(f"runtime directory outside project: {runtime_dir}")
        files = {p.name.upper(): p for p in runtime_dir.iterdir() if p.is_file()}
        filex = files.get("FILEX.MZX")
        runtime_cli = files.get("CNFQ.CLI")
        pdi = files.get("DSSAT-PDI.YML")
        if any(p is None for p in (filex, runtime_cli, pdi)):
            raise RuntimeError(f"runtime evidence files missing episode={index}: {list(files)}")
        rendered = filex.read_text(encoding="utf-8", errors="replace")
        yml = pdi.read_text(encoding="utf-8", errors="replace")
        runtime_mode = runtime_filex_mode(rendered)
        bootstrap_seed = int(self.schedule[0]["weather_seed"])
        bootstrap_seed_ok = pdi_seed_configured(yml, bootstrap_seed)
        cli_ok = sha(runtime_cli) == sha(CLI)
        if runtime_mode != "W" or not bootstrap_seed_ok or not cli_ok or "CNFQ0701" not in rendered:
            raise RuntimeError(f"runtime evidence mismatch episode={index}: WTHER={runtime_mode}, bootstrap_seed={bootstrap_seed_ok}, CLI={cli_ok}")
        evidence = {"episode_index": index, "runtime_dir": rel(runtime_dir), "filex_sha256": sha(filex), "filex_wther": runtime_mode, "filex_wsta_confirmed": True, "pdi_config_sha256": sha(pdi), "pdi_yaml_bootstrap_seed": bootstrap_seed, "pdi_yaml_bootstrap_seed_confirmed": bootstrap_seed_ok, "pdi_runtime_rseed1": int(dssat._rseed1), "pdi_runtime_episode_seed_confirmed": int(dssat._rseed1) == int(row["weather_seed"]), "runtime_cli_sha256": sha(runtime_cli), "cli_source_match": cli_ok}
        evidence_path = OUT / "runtime_evidence" / f"episode_{index:04d}.json"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(evidence_path, evidence)
        record = {"episode_index": index, "status": status, "station": "FQA", "year": YEAR, "ppo_seed": PPO_SEED, "weather_seed": int(row["weather_seed"]), "pdi_rseed1": int(dssat._rseed1), "days_used": self.episode_days, "weather_rows": len(weather), "capture_calls": len(getattr(dssat, "_pilot_weather_states", [])), "weather_path": rel(path), "weather_sha256": digest, "filex_wther": runtime_mode, "filex_template_sha256": dssat._pilot_filex_template_sha256, "cli_sha256": sha(CLI), "runtime_evidence_path": rel(evidence_path), "physical_status": screen["status"]}
        self.archive_rows.append(record)
        manifest = OUT / "episode_manifest.csv"
        with manifest.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(record)

    def _step(self, action):
        value = super()._step(action)
        if value[2] or value[3]:
            self.archive_current("completed")
        return value


def main():
    import numpy as np
    import torch
    from sb3_contrib import MaskablePPO
    import ppo_safe_rendering as rendering

    if (OUT / "run_result.json").exists() or (OUT / "episode_manifest.csv").exists():
        raise FileExistsError("044 attempt already exists; preserve it")
    OUT.mkdir(parents=True, exist_ok=True)
    temp_root = OUT / "temp"
    temp_root.mkdir()
    os.environ["TMPDIR"] = str(temp_root)
    tempfile.tempdir = str(temp_root)
    cfg, env_cfg, pf = preflight()
    write_json(OUT / "preflight.json", pf)
    write_json(OUT / "effective_config.json", cfg)
    schedule = yc.schedule_rows("RANDOM_WEATHER_WGEN", 256)
    for row in schedule:
        row["historical_year"] = YEAR
    write_json(OUT / "episode_schedule.json", schedule)
    yc.make_weather_env = make_fq_env
    yc.TASK_ROOT = OUT
    yc.MAX_PROCESS_TREE_RSS_MB = RSS_LIMIT_MB
    start = time.monotonic()
    env = None
    result = {"status": "FAILED", "requested_steps": STEPS, "ppo_seed": PPO_SEED}
    original_root = rendering.MULTISITE_INPUT_ROOT
    rendering.MULTISITE_INPUT_ROOT = fq.INPUT_PROFILES["originIC"]
    try:
        random.seed(PPO_SEED)
        np.random.seed(PPO_SEED)
        torch.set_num_threads(1)
        env = ArchivedEnv(cfg, env_cfg, schedule)
        callback = yc.TrainingCallback(env, OUT, [STEPS], smoke=True)
        kwargs = fq.engine.base03222.base.ppo_kwargs(cfg)
        model = MaskablePPO("MlpPolicy", env._gym_env, verbose=0, seed=PPO_SEED, **kwargs)
        model.learn(total_timesteps=STEPS, reset_num_timesteps=True, progress_bar=False, callback=callback.callback)
        callback.flush_episodes()
        if env.current_schedule is not None and env.episode_days > 0:
            env.archive_current("partial_at_stop")
        actual = int(model.num_timesteps)
        archived = sum(int(r["days_used"]) for r in env.archive_rows)
        if actual != archived or actual < STEPS or callback.stop_for_memory:
            raise RuntimeError(f"step/archive/resource gate failed: actual={actual}, archived={archived}, memory_stop={callback.stop_for_memory}")
        if time.monotonic() - start > WALL_LIMIT_S:
            raise TimeoutError("wall limit exceeded")
        path = OUT / "models/fqa_ppo_seed0_2k.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
        MaskablePPO.load(str(path), device="cpu")
        result.update({"status": "PASS_SMOKE_ONLY", "actual_steps": actual, "archived_days": archived, "episode_archives": len(env.archive_rows), "completed_episodes": sum(r["status"] == "completed" for r in env.archive_rows), "model_path": rel(path), "model_sha256": sha(path)})
    except Exception:
        result["error"] = traceback.format_exc()
        (OUT / "failure_traceback.txt").write_text(result["error"], encoding="utf-8")
    finally:
        if env is not None:
            try:
                if env.current_schedule is not None and env.episode_days > 0:
                    env.archive_current("partial_after_failure")
            except Exception as exc:
                result["archive_after_failure_error"] = str(exc)
            env.close()
        rendering.MULTISITE_INPUT_ROOT = original_root
        result["elapsed_seconds"] = round(time.monotonic() - start, 2)
        write_json(OUT / "run_result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "error"}, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "PASS_SMOKE_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
