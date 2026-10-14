"""076: train/evaluate SY seed3 under canonical 032_22 contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
from run_074_sya_seed2_canonical_smoke2k import audit, loaded_kwargs

PLAN = ROOT / "configs" / "076_sya_seed3_canonical_03222_100k.json"
BASE_YAML = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def cfg() -> dict[str, Any]:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def out(c: dict[str, Any]) -> Path:
    return ROOT / c["output_root"]


def preflight(c: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    input_root = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
    if not (input_root / "SY/CNSY1201.MZX").exists(): issues.append("originIC SY MZX missing")
    split = batch.load_split(); local = split[split.station_code.eq("SYA")].sort_values("year")
    train = local[local.split.eq("train")].year.astype(int).tolist(); valid = local[local.split.eq("validation")].year.astype(int).tolist()
    if train != c["train_years"] or valid != c["validation_years"]: issues.append("split years mismatch")
    if c["seed"] != 3: issues.append("seed must be 3")
    if c["actions"] != {"irrigation_levels_mm": [0.0,15.0,30.0,45.0], "nitrogen_levels_kg_ha": [0.0,40.0,80.0,120.0]}: issues.append("action contract mismatch")
    if c["observation_contract"] != {"base":"032_22_raw_observation","normalization_enabled":False,"weather_forecast_enabled":False}: issues.append("observation contract mismatch")
    smoke = ROOT / c["smoke_result"]
    if not smoke.exists() or json.loads(smoke.read_text(encoding="utf-8")).get("next_step_allowed") is not True: issues.append("seed3 2K smoke gate did not pass")
    if not (ROOT / c["prompt"]).exists(): issues.append("registered prompt missing")
    if out(c).exists(): issues.append("isolated 076 output root already exists")
    return {"task": f"{c['task_id']}_{c['task_name']}", "seed": c["seed"], "input_root": rel(input_root), "train_years": train, "validation_years": valid, "issues": issues, "next_step_allowed": not issues}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--phase", choices=["dry-run", "train"], required=True); a = p.parse_args()
    c = cfg(); pf = preflight(c)
    if a.phase == "dry-run": print(json.dumps(pf, ensure_ascii=False, indent=2)); return
    if not pf["next_step_allowed"]: raise RuntimeError("; ".join(pf["issues"]))
    root = out(c); root.mkdir(parents=True); (root / "configs").mkdir()
    shutil.copy2(PLAN, root / "configs" / PLAN.name); shutil.copy2(ROOT / c["prompt"], root / "configs" / Path(c["prompt"]).name)
    effective = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8")); effective["seed"] = 3; effective["total_timesteps"] = 100000; effective["ppo"].update(c["ppo"]); effective["paths"]["output_root"] = rel(root)
    ep = ROOT / "configs" / "076_sya_seed3_canonical_effective.yaml"; ep.write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8"); shutil.copy2(ep, root / "configs" / ep.name)
    old = (batch.CONFIG, batch.OUT, batch.DOC, batch.SEED, batch.TOTAL_TIMESTEPS, batch.CHECKPOINT_STEPS, list(batch.SITES), dict(batch.SITE_NAMES), ppo_safe_rendering.MULTISITE_INPUT_ROOT)
    start = time.perf_counter(); env = None
    try:
        batch.CONFIG, batch.OUT, batch.DOC, batch.SEED = ep, root, root / "076_record.md", 3; batch.TOTAL_TIMESTEPS = 100000; batch.CHECKPOINT_STEPS = [25000,50000,75000,100000]; batch.SITES = ["SYA"]; batch.SITE_NAMES["SYA"] = "SY"; ppo_safe_rendering.MULTISITE_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
        config = batch.load_config(); split = batch.load_split(); selection = batch.build_selection(split); envcfg = batch.direct_ppo.build_env_config(config, selection); envcfg["paths"]["output_root"] = rel(root)
        train_years = split[split.station_code.eq("SYA") & split.split.eq("train")].year.astype(int).tolist();
        train_df, reset_df = batch.train_station(config, envcfg, "SYA", train_years)
        (root / "evaluation").mkdir(parents=True, exist_ok=True); train_df.to_csv(root / "evaluation" / "076_training_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
        (root / "logs").mkdir(parents=True, exist_ok=True); reset_df.to_csv(root / "logs" / "076_training_year_reset_counts.csv", index=False, encoding="utf-8-sig")
        if train_df.empty or not train_df.run_status.astype(str).str.startswith("ok").all(): raise RuntimeError("one or more checkpoint models missing")
        rows = []
        for row in train_df.itertuples(index=False):
            for year in c["validation_years"]: rows.append(batch.evaluate_checkpoint(config, envcfg, pd.Series(row._asdict()), int(year)))
        summary = pd.DataFrame(rows); summary.to_csv(root / "evaluation" / "076_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
        audits, gates = {}, {}
        for step in c["training"]["checkpoint_steps"]:
            audit_df, gate = audit(summary, int(step)); path = root / "audits" / f"076_ckpt{step}_action_audit.csv"; path.parent.mkdir(exist_ok=True); audit_df.to_csv(path, index=False, encoding="utf-8-sig"); audits[str(step)] = rel(path); gates[str(step)] = gate
        final_model = ROOT / str(train_df.iloc[-1]["model_path"])
        result = {**pf, "phase": "canonical_seed3_train100k", "elapsed_s": time.perf_counter() - start, "effective_yaml": rel(ep), "effective_kwargs_from_loaded_model": loaded_kwargs(final_model), "checkpoints": c["training"]["checkpoint_steps"], "action_audits": audits, "action_gates": gates, "WP_ET_status": "not available from direct training summary; corrected five-scenario replay required", "next_step_allowed": bool(any(g["mechanism_pass"] for g in gates.values())), "replay_policy": "only checkpoints with mechanism_pass=true may enter corrected 073 replay"}
        (root / "076_formal_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); (root / "076_run_manifest.json").write_text(json.dumps({"status":"completed_formal","config":c,"preflight":pf,"result":result}, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if env is not None: env.close()
        batch.CONFIG, batch.OUT, batch.DOC, batch.SEED, batch.TOTAL_TIMESTEPS, batch.CHECKPOINT_STEPS, batch.SITES, batch.SITE_NAMES, ppo_safe_rendering.MULTISITE_INPUT_ROOT = old


if __name__ == "__main__": main()
