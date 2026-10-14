"""084: SYA seed3 reserve-aware mask, isolated 25K checkpoint only."""
from __future__ import annotations

import argparse
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
import run_083_sya_reserve_aware_mask_smoke2k as impl

PLAN = ROOT / "configs" / "084_sya_reserve_aware_mask_25k.json"
PROMPT = ROOT / "prompts" / "2026-08-13_sya_reserve_aware_mask_25k.md"
BASE_YAML = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def load() -> dict[str, Any]:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def out(c: dict[str, Any]) -> Path:
    return ROOT / c["output_root"]


def preflight(c: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    input_root = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
    if not (input_root / "SY/CNSY1201.MZX").exists():
        issues.append("originIC SY input missing")
    split = batch.load_split(); local = split[split.station_code.eq("SYA")].sort_values("year")
    train = local[local.split.eq("train")].year.astype(int).tolist(); valid = local[local.split.eq("validation")].year.astype(int).tolist()
    if train != c["train_years"] or valid != c["validation_years"]: issues.append("split years mismatch")
    if c["seed"] != 3: issues.append("seed must be 3")
    smoke = ROOT / c["smoke_reference"]
    if not smoke.exists() or json.loads(smoke.read_text(encoding="utf-8")).get("next_step_allowed") is not True: issues.append("083 smoke gate did not pass")
    base_cfg = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8")); safety = base_cfg.get("action_safety", {})
    if safety.get("min_days_between_irrigation") != 7 or safety.get("min_days_between_fertilization") != 7: issues.append("canonical 7-day mask missing")
    if not PROMPT.exists() or not BASE_YAML.exists(): issues.append("prompt or base YAML missing")
    if out(c).exists(): issues.append("isolated output root already exists")
    return {"task": f"{c['task_id']}_{c['task_name']}", "seed": c["seed"], "input_root": rel(input_root), "train_years": train, "validation_years": valid, "issues": issues, "next_step_allowed": not issues}


def run() -> dict[str, Any]:
    c = load(); pf = preflight(c)
    if not pf["next_step_allowed"]: raise RuntimeError("; ".join(pf["issues"]))
    root = out(c); root.mkdir(parents=True); (root / "configs").mkdir()
    shutil.copy2(PLAN, root / "configs" / PLAN.name); shutil.copy2(PROMPT, root / "configs" / PROMPT.name)
    effective = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8")); effective["seed"] = int(c["seed"]); effective["total_timesteps"] = 25_000; effective["ppo"].update(c["ppo"])
    effective["action_safety"]["reserve_irrigation_mm"] = 15.0; effective["action_safety"]["reserve_n_kg_ha"] = 40.0; effective["paths"]["output_root"] = rel(root)
    ep = ROOT / "configs" / "084_sya_reserve_aware_effective.yaml"; ep.write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8"); shutil.copy2(ep, root / "configs" / ep.name)
    old_make = impl.base.make_env; impl.base.make_env = impl.reserve_make_env
    old = (batch.CONFIG, batch.OUT, batch.DOC, batch.SEED, batch.TOTAL_TIMESTEPS, batch.CHECKPOINT_STEPS, list(batch.SITES), dict(batch.SITE_NAMES), ppo_safe_rendering.MULTISITE_INPUT_ROOT)
    start = time.perf_counter(); env = None
    try:
        batch.CONFIG, batch.OUT, batch.DOC, batch.SEED = ep, root, root / "084_record.md", int(c["seed"]); batch.TOTAL_TIMESTEPS = 25_000; batch.CHECKPOINT_STEPS = [25_000]; batch.SITES = ["SYA"]; batch.SITE_NAMES["SYA"] = "SY"; ppo_safe_rendering.MULTISITE_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
        config = batch.load_config(); split = batch.load_split(); selection = batch.build_selection(split); envcfg = batch.direct_ppo.build_env_config(config, selection); envcfg["paths"]["output_root"] = rel(root)
        train_years = split[split.station_code.eq("SYA") & split.split.eq("train")].year.astype(int).tolist(); train_df, reset_df = batch.train_station(config, envcfg, "SYA", train_years)
        (root / "evaluation").mkdir(parents=True, exist_ok=True); train_df.to_csv(root / "evaluation" / "084_training_checkpoint_inventory.csv", index=False, encoding="utf-8-sig"); (root / "logs").mkdir(parents=True, exist_ok=True); reset_df.to_csv(root / "logs" / "084_training_year_reset_counts.csv", index=False, encoding="utf-8-sig")
        if train_df.empty or not train_df.run_status.astype(str).str.startswith("ok").all(): raise RuntimeError("25K checkpoint model missing")
        rows = [batch.evaluate_checkpoint(config, envcfg, pd.Series(row._asdict()), int(year)) for row in train_df.itertuples(index=False) for year in c["validation_years"]]; summary = pd.DataFrame(rows); summary.to_csv(root / "evaluation" / "084_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
        audit_df, gate = impl.audit(summary, 25_000); ap = root / "audits" / "084_ckpt25000_action_audit.csv"; ap.parent.mkdir(exist_ok=True); audit_df.to_csv(ap, index=False, encoding="utf-8-sig")
        model_path = ROOT / str(train_df.iloc[-1]["model_path"]); result = {**pf, "phase": "reserve_aware_mask_train25k", "elapsed_s": time.perf_counter() - start, "effective_yaml": rel(ep), "effective_kwargs_from_loaded_model": impl.loaded_kwargs(model_path), "checkpoint": 25_000, "action_audit": rel(ap), "action_gate": gate, "reserve_contract": c["intervention"], "WP_ET_status": "not available from direct checkpoint validation; corrected five-scenario replay required", "next_step_allowed": bool(gate["mechanism_pass"]), "stop_reason": "25K validation complete; no 50K/75K/100K or replay in this task"}
        (root / "084_formal_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); (root / "084_run_manifest.json").write_text(json.dumps({"status": "completed_25k_validation", "config": c, "preflight": pf, "result": result}, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(result, ensure_ascii=False, indent=2)); return result
    finally:
        if env is not None: env.close()
        batch.CONFIG, batch.OUT, batch.DOC, batch.SEED, batch.TOTAL_TIMESTEPS, batch.CHECKPOINT_STEPS, batch.SITES, batch.SITE_NAMES, ppo_safe_rendering.MULTISITE_INPUT_ROOT = old
        impl.base.make_env = old_make


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=["dry-run", "train"], required=True); args = parser.parse_args(); print(json.dumps(preflight(load()), ensure_ascii=False, indent=2) if args.phase == "dry-run" else json.dumps(run(), ensure_ascii=False, indent=2))
