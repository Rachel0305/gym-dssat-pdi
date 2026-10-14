"""108_00 FQ current-weather observation 5K/10K plus corrected-2018 replay."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
import run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36 as reserve
import run_096_fqa_reward_ncost25_ckpt25k_five_scenario as audit_base
import run_107_fqa_current_weather_observation_smoke2k as weather_base

CONFIG = ROOT / "configs/108_00_fqa_current_weather_observation_5k10k_corrected2018.json"
PROMPT = ROOT / "prompts/108_00_fq_current_weather_observation_5k10k_corrected2018.md"
BASE_YAML = ROOT / "experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
INPUT_ROOT = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013"
LEVELS_I = [0.0, 15.0, 30.0, 45.0]
LEVELS_N = [0.0, 40.0, 80.0, 120.0]
STATION, SITE = "FQA", "FQ"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def make_effective_yaml(cfg: dict[str, Any]) -> Path:
    effective = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    effective["seed"] = int(cfg["seed"])
    ppo = dict(cfg["fixed_ppo"]); ppo.update(cfg.get("ppo_override", {}))
    effective["ppo"].update(ppo)
    effective["reward"].update(cfg["reward_override"])
    effective["discrete_actions"]["irrigation_levels"] = LEVELS_I
    effective["discrete_actions"]["nitrogen_levels"] = LEVELS_N
    path = ROOT / "configs/108_00_effective_fqa_current_weather_ppo_config.yaml"
    path.write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
    return path


def preflight(cfg: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    if not (INPUT_ROOT / "FQ/CNFQ0801.MZX").exists():
        issues.append("originIC FQ input/MZX missing")
    corrected = ROOT / cfg["input_roots"]["corrected_2018"] / "FQ/CNFQ1801.WTH"
    if not corrected.exists():
        issues.append("corrected 2018 WTH missing")
    for key in ["baseline_original", "baseline_corrected_2018"]:
        if not (ROOT / cfg["input_roots"][key]).exists():
            issues.append(f"missing baseline root: {key}")
    if cfg["training"] != {"total_timesteps": 10000, "checkpoint_steps": [5000, 10000]}:
        issues.append("training must be exactly 10K with 5K/10K checkpoints")
    if cfg["actions"]["irrigation_levels_mm"] != LEVELS_I or cfg["actions"]["nitrogen_levels_kg_ha"] != LEVELS_N:
        issues.append("action grid mismatch")
    if cfg["observation_contract"]["current_weather_features"] != weather_base.WEATHER_NAMES:
        issues.append("current-weather feature contract mismatch")
    if cfg["observation_contract"]["weather_forecast_enabled"] is not False:
        issues.append("weather forecast must remain disabled")
    return {"task_id": cfg["task_id"], "station_code": STATION, "site": SITE, "seed": int(cfg["seed"]), "issues": issues, "next_step_allowed": not issues, "training_steps": 10000, "checkpoints": [5000, 10000], "corrected_2018_exists": corrected.exists()}


def configure_training(cfg: dict[str, Any], out: Path, effective: Path) -> dict[str, Any]:
    old_engine = {key: getattr(engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_batch = {key: getattr(batch, key) for key in ["CONFIG", "OUT", "DOC", "SEED"]}
    old_names = dict(batch.SITE_NAMES)
    old_wrapper = batch.base.StressAwareDiscreteWrapper
    old_make_env = batch.base.make_env
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    doc = ROOT / "docs/108_00_fqa_current_weather_observation_5k10k_corrected2018_record.md"
    batch.base.StressAwareDiscreteWrapper = reserve.LateIrrigationReserveMaskWrapper
    base_make_env = old_make_env

    def make_env_current(config, env_config, station, year, seed, run_tag, evaluation=False):
        return weather_base.CurrentWeatherObservationWrapper(base_make_env(config, env_config, station, year, seed, run_tag, evaluation=evaluation), config, env_config, station, year)

    try:
        engine.TASK_ID = cfg["task_id"]; engine.TASK_NAME = cfg["task_name"]; engine.BASE_OUT = out; engine.BASE_DOC = doc; engine.PROMPT = PROMPT; engine.LOWIC_INPUT_ROOT = INPUT_ROOT; engine.STATION = STATION; engine.SITES = [STATION]; engine.BINARY_IRRIGATION_LEVELS = LEVELS_I; engine.BINARY_NITROGEN_LEVELS = LEVELS_N
        batch.SEED = int(cfg["seed"]); batch.CONFIG = effective; batch.OUT = out; batch.DOC = doc; batch.SITE_NAMES[STATION] = SITE
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = INPUT_ROOT
        batch.base.make_env = make_env_current
        engine.run_training(10000, [5000, 10000], suffix="")
    finally:
        batch.base.make_env = old_make_env; batch.base.StressAwareDiscreteWrapper = old_wrapper; ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
        for key, value in old_engine.items(): setattr(engine, key, value)
        for key, value in old_batch.items(): setattr(batch, key, value)
        batch.SITE_NAMES.clear(); batch.SITE_NAMES.update(old_names)
    return {"doc": doc, "make_env_current": make_env_current, "base_make_env": base_make_env}


def replay_and_tables(cfg: dict[str, Any], out: Path, effective: Path) -> list[dict[str, Any]]:
    summary_path = out / "evaluation/042_10_checkpoint_validation_summary.csv"
    inventory_path = out / "evaluation/042_10_training_checkpoint_inventory.csv"
    summary = pd.read_csv(summary_path, keep_default_na=False); inventory = pd.read_csv(inventory_path, keep_default_na=False)
    summary.to_csv(out / "evaluation/096_00_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(out / "evaluation/096_00_training_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
    gates: dict[int, dict[str, Any]] = {}
    for checkpoint in [5000, 10000]:
        audit, gate = audit_base.audit_25k(summary, out, checkpoint)
        audit.to_csv(out / "audits" / f"108_00_action_audit_ckpt{checkpoint}.csv", index=False, encoding="utf-8-sig")
        gates[checkpoint] = gate

    # Reuse the strict replay machinery, but wrap every evaluation env with the
    # same current-weather observation adapter used during training.
    old_wrapper = batch.base.StressAwareDiscreteWrapper
    old_make_env = batch.base.make_env
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    batch.base.StressAwareDiscreteWrapper = reserve.LateIrrigationReserveMaskWrapper
    base_make_env = old_make_env

    def make_env_current(config, env_config, station, year, seed, run_tag, evaluation=False):
        return weather_base.CurrentWeatherObservationWrapper(base_make_env(config, env_config, station, year, seed, run_tag, evaluation=evaluation), config, env_config, station, year)

    batch.base.make_env = make_env_current
    rows_all: list[dict[str, Any]] = []
    replay_results: dict[str, Any] = {}
    try:
        for checkpoint in [5000, 10000]:
            ckpt_out = out / f"replay_ckpt{checkpoint}"
            (ckpt_out / "evaluation").mkdir(parents=True, exist_ok=True)
            (ckpt_out / "audits").mkdir(parents=True, exist_ok=True)
            shutil.copy2(out / "evaluation/096_00_checkpoint_validation_summary.csv", ckpt_out / "evaluation/096_00_checkpoint_validation_summary.csv")
            shutil.copy2(out / "evaluation/096_00_training_checkpoint_inventory.csv", ckpt_out / "evaluation/096_00_training_checkpoint_inventory.csv")
            local_rows: list[dict[str, Any]] = []
            replay_cfg = dict(cfg)
            replay_cfg["training"] = {"total_timesteps": 10000}
            for year in cfg["scope"]["validation_years"]:
                local_rows.append(audit_base.replay_candidate(replay_cfg, ckpt_out, ckpt_out, effective, checkpoint, int(year), ROOT / cfg["input_roots"]["original"], "original", True))
            local_rows.append(audit_base.replay_candidate(replay_cfg, ckpt_out, ckpt_out, effective, checkpoint, 2018, ROOT / cfg["input_roots"]["corrected_2018"], "corrected2018", False))
            pd.DataFrame(local_rows).to_csv(ckpt_out / "tables_108_00_ppo_replay_provenance.csv", index=False, encoding="utf-8-sig")
            five = audit_base.build_five_scenario_tables(replay_cfg, ckpt_out, local_rows)
            replay_results[str(checkpoint)] = {"gate": gates[checkpoint], "replay_rows": len(local_rows), "five_scenario": five, "replay_root": rel(ckpt_out)}
            rows_all.extend([{**row, "checkpoint": checkpoint} for row in local_rows])
    finally:
        batch.base.make_env = old_make_env; batch.base.StressAwareDiscreteWrapper = old_wrapper; ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
    return [{"checkpoint": int(k), **v} for k, v in replay_results.items()]


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    out = ROOT / cfg["output_root"]
    pf = preflight(cfg)
    if "--dry-run" in sys.argv:
        print(json.dumps({**pf, "mode": "dry_run", "training_run": False, "five_scenario_replay": False, "output_root": rel(out), "prompt_exists": PROMPT.exists()}, ensure_ascii=False, indent=2))
        return 0
    if not pf["next_step_allowed"]:
        raise RuntimeError("preflight failed: " + "; ".join(pf["issues"]))
    finalize_existing = "--finalize-existing" in sys.argv
    if out.exists() and not finalize_existing:
        raise FileExistsError(f"refusing to overwrite {rel(out)}")
    (out / "configs").mkdir(parents=True, exist_ok=True); (out / "audits").mkdir(parents=True, exist_ok=True)
    if not finalize_existing:
        shutil.copy2(CONFIG, out / "configs" / CONFIG.name); shutil.copy2(PROMPT, out / "configs" / PROMPT.name)
    effective = make_effective_yaml(cfg)
    start = time.perf_counter(); rss0 = psutil.Process().memory_info().rss
    if not finalize_existing:
        configure_training(cfg, out, effective)
    replay = replay_and_tables(cfg, out, effective)
    result = {"task_id": cfg["task_id"], "mode": "completed_5k_10k_training_and_corrected2018_replay", "training_run": True, "external_n_included": False, "weather_forecast_included": False, "observation_contract": cfg["observation_contract"], "checkpoints": [5000, 10000], "checkpoint_results": replay, "effective_yaml": rel(effective), "output_root": rel(out), "elapsed_seconds": round(time.perf_counter() - start, 3), "rss_delta_bytes": psutil.Process().memory_info().rss - rss0, "formal_freeze_allowed": False, "next_step_allowed": False, "next_step_note": "Do not start 25K/100K automatically; inspect corrected-2018 five-scenario tables and WP_ET/PFP_N gaps first."}
    (out / "108_00_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "108_00_manifest.json").write_text(json.dumps({"config": cfg, "preflight": pf, "result": result}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# 108_00 FQ current-weather observation 5K/10K and corrected-2018 replay", "", "- 107 current-weather observation branch continued to 10K; checkpoints: 5K and 10K.", "- No external-N, no weather forecast, no 25K/100K.", "", "## Checkpoint/replay result", "", *[f"- `{item}`" for item in replay], "", "- Corrected 2018 is isolated provenance replay; raw 2018 remains endpoint oracle.", "- WP_ET/PFP_N and yearly wins must be read from the checkpoint-specific five-scenario tables; no automatic freeze.", ""]
    (ROOT / "docs/108_00_fqa_current_weather_observation_5k10k_corrected2018_record.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
