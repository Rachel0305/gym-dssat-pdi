"""085: SYA seed3 training-weather augmentation, isolated 2K smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
from run_078_sya_mask_relaxed_gap3_smoke2k import audit, loaded_kwargs

PLAN = ROOT / "configs" / "085_sya_training_weather_augmentation_smoke2k.json"
PROMPT = ROOT / "prompts" / "2026-08-13_sya_training_weather_augmentation_smoke2k.md"
BASE_YAML = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
DATA_RE = re.compile(r"^\s*(\d{7})\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s*$")


def rel(p: Path) -> str: return p.relative_to(ROOT).as_posix()
def load() -> dict[str, Any]: return json.loads(PLAN.read_text(encoding="utf-8"))
def out(c: dict[str, Any]) -> Path: return ROOT / c["output_root"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def perturb_weather(path: Path, seed: int, year: int, cfg: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed) * 1_000_003 + int(year) * 9_176)
    rain_cache: dict[int, float] = {}; temp_cache: dict[int, float] = {}; srad_cache: dict[int, float] = {}
    rain_clip = cfg["rain_clip"]; temp_clip = cfg["temperature_clip_c"]; srad_clip = cfg["srad_clip"]
    original_hash = sha256(path); lines = path.read_text(encoding="utf-8", errors="replace").splitlines(True); changed = 0; rain_vals = []; temp_vals = []; srad_vals = []
    out_lines: list[str] = []
    for line in lines:
        m = DATA_RE.match(line)
        if not m:
            out_lines.append(line); continue
        date, srad, tmax, tmin, rain = m.groups(); doy = int(date[-3:]); rb = (doy - 1) // int(cfg["rain_block_days"]); tb = (doy - 1) // int(cfg["temperature_block_days"]); sb = (doy - 1) // int(cfg["srad_block_days"])
        rain_factor = rain_cache.setdefault(rb, float(np.clip(rng.normal(1.0, cfg["rain_sigma"]), rain_clip[0], rain_clip[1])))
        temp_shift = temp_cache.setdefault(tb, float(np.clip(rng.normal(0.0, cfg["temperature_sigma_c"]), temp_clip[0], temp_clip[1])))
        srad_factor = srad_cache.setdefault(sb, float(np.clip(rng.normal(1.0, cfg["srad_sigma"]), srad_clip[0], srad_clip[1])))
        sr = max(0.0, float(srad) * srad_factor); tx = float(tmax) + temp_shift; tn = float(tmin) + temp_shift; rr = max(0.0, float(rain) * rain_factor)
        out_lines.append(f"{date:>7s}{sr:7.1f}{tx:7.1f}{tn:7.1f}{rr:7.1f}\n"); changed += 1; rain_vals.append(rain_factor); temp_vals.append(temp_shift); srad_vals.append(srad_factor)
    path.write_text("".join(out_lines), encoding="utf-8")
    return {"year": int(year), "rendered_weather": rel(path), "source_hash_before": original_hash, "rendered_hash_after": sha256(path), "data_rows_perturbed": changed, "rain_factor_min": float(min(rain_vals)) if rain_vals else None, "rain_factor_max": float(max(rain_vals)) if rain_vals else None, "temperature_shift_min_c": float(min(temp_vals)) if temp_vals else None, "temperature_shift_max_c": float(max(temp_vals)) if temp_vals else None, "srad_factor_min": float(min(srad_vals)) if srad_vals else None, "srad_factor_max": float(max(srad_vals)) if srad_vals else None}


def preflight(c: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []; input_root = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
    if not (input_root / "SY/CNSY1201.MZX").exists(): issues.append("originIC SY input missing")
    split = batch.load_split(); local = split[split.station_code.eq("SYA")].sort_values("year"); train = local[local.split.eq("train")].year.astype(int).tolist(); valid = local[local.split.eq("validation")].year.astype(int).tolist()
    if train != c["train_years"] or valid != c["validation_years"]: issues.append("split years mismatch")
    if c["seed"] != 3: issues.append("seed must be 3")
    if not PROMPT.exists() or not BASE_YAML.exists(): issues.append("prompt or base YAML missing")
    if out(c).exists(): issues.append("isolated output root already exists")
    return {"task": f"{c['task_id']}_{c['task_name']}", "seed": c["seed"], "input_root": rel(input_root), "train_years": train, "validation_years": valid, "issues": issues, "next_step_allowed": not issues}


def run() -> dict[str, Any]:
    c = load(); pf = preflight(c)
    if not pf["next_step_allowed"]: raise RuntimeError("; ".join(pf["issues"]))
    root = out(c); root.mkdir(parents=True); (root / "configs").mkdir(); shutil.copy2(PLAN, root / "configs" / PLAN.name); shutil.copy2(PROMPT, root / "configs" / PROMPT.name)
    effective = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8")); effective["seed"] = int(c["seed"]); effective["total_timesteps"] = 2000; effective["ppo"].update(c["ppo"]); effective["paths"]["output_root"] = rel(root)
    ep = ROOT / "configs" / "085_sya_training_weather_augmentation_effective.yaml"; ep.write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8"); shutil.copy2(ep, root / "configs" / ep.name)
    aug_records: list[dict[str, Any]] = []; original_build = direct_ppo.build_env_args
    aug_cfg = c["augmentation"]
    def augmented_build_env_args(*args, **kwargs):
        env_args = original_build(*args, **kwargs)
        evaluation = bool(kwargs.get("evaluation", False))
        if not evaluation:
            station = str(kwargs.get("station", args[0] if args else "")); year = int(kwargs.get("year", args[1] if len(args) > 1 else 0)); weather = Path(env_args["auxiliary_file_paths"][1]); rec = perturb_weather(weather, int(c["seed"]), year, aug_cfg); rec.update({"station": station, "scope": "training_only"}); aug_records.append(rec)
        return env_args
    direct_ppo.build_env_args = augmented_build_env_args
    old = (batch.CONFIG, batch.OUT, batch.DOC, batch.SEED, batch.TOTAL_TIMESTEPS, batch.CHECKPOINT_STEPS, list(batch.SITES), dict(batch.SITE_NAMES), ppo_safe_rendering.MULTISITE_INPUT_ROOT)
    start = time.perf_counter(); env = None
    try:
        batch.CONFIG, batch.OUT, batch.DOC, batch.SEED = ep, root, root / "085_record.md", int(c["seed"]); batch.TOTAL_TIMESTEPS = 2000; batch.CHECKPOINT_STEPS = [1000, 2000]; batch.SITES = ["SYA"]; batch.SITE_NAMES["SYA"] = "SY"; ppo_safe_rendering.MULTISITE_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
        config = batch.load_config(); split = batch.load_split(); selection = batch.build_selection(split); envcfg = batch.direct_ppo.build_env_config(config, selection); envcfg["paths"]["output_root"] = rel(root)
        train_years = split[split.station_code.eq("SYA") & split.split.eq("train")].year.astype(int).tolist(); env = batch.RandomYearEnv(config, envcfg, "SYA", train_years, int(c["seed"]))
        from sb3_contrib import MaskablePPO
        model = MaskablePPO("MlpPolicy", env, verbose=0, seed=int(c["seed"]), device="cpu", **batch.base.ppo_kwargs(config)); cb = batch.FixedStepCheckpointCallback("SYA", [1000, 2000]); model.learn(total_timesteps=2000, reset_num_timesteps=True, progress_bar=False, callback=cb.callback); env.close(); env = None
        inv = []
        for step in [1000, 2000]:
            p = batch.model_path("SYA", step); inv.append({"station_code": "SYA", "site": "SY", "seed": int(c["seed"]), "checkpoint_step": step, "run_status": "ok" if p.exists() else "missing", "model_path": rel(p) if p.exists() else "", "model_sha256": sha256(p) if p.exists() else ""})
        inventory = pd.DataFrame(inv); (root / "evaluation").mkdir(); inventory.to_csv(root / "evaluation" / "085_training_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
        rows = [batch.evaluate_checkpoint(config, envcfg, pd.Series(r), year) for r in inv for year in c["validation_years"]]; summary = pd.DataFrame(rows); summary.to_csv(root / "evaluation" / "085_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
        audits, gates = {}, {}
        for step in [1000, 2000]:
            a, g = audit(summary, step); p = root / "audits" / f"085_ckpt{step}_action_audit.csv"; p.parent.mkdir(exist_ok=True); a.to_csv(p, index=False, encoding="utf-8-sig"); audits[str(step)] = rel(p); gates[str(step)] = g
        (root / "evaluation" / "085_weather_augmentation_manifest.json").write_text(json.dumps({"augmentation": aug_cfg, "records": aug_records}, ensure_ascii=False, indent=2), encoding="utf-8")
        final_model = ROOT / inv[-1]["model_path"]; result = {**pf, "phase": "training_weather_augmentation_smoke2k", "elapsed_s": time.perf_counter() - start, "effective_yaml": rel(ep), "effective_kwargs_from_loaded_model": loaded_kwargs(final_model), "action_audits": audits, "action_gates": gates, "weather_augmentation_manifest": rel(root / "evaluation" / "085_weather_augmentation_manifest.json"), "WP_ET_status": "not available from smoke summaries; not inferred", "next_step_allowed": bool(gates["2000"]["mechanism_pass"]), "stop_reason": "smoke complete; no long training in this task"}
        (root / "085_smoke_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); (root / "085_run_manifest.json").write_text(json.dumps({"status": "completed_smoke", "config": c, "preflight": pf, "result": result}, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(result, ensure_ascii=False, indent=2)); return result
    finally:
        if env is not None: env.close()
        batch.CONFIG, batch.OUT, batch.DOC, batch.SEED, batch.TOTAL_TIMESTEPS, batch.CHECKPOINT_STEPS, batch.SITES, batch.SITE_NAMES, ppo_safe_rendering.MULTISITE_INPUT_ROOT = old
        direct_ppo.build_env_args = original_build


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=["dry-run", "smoke"], required=True); args = parser.parse_args(); print(json.dumps(preflight(load()), ensure_ascii=False, indent=2) if args.phase == "dry-run" else json.dumps(run(), ensure_ascii=False, indent=2))
