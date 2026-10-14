"""HL seed1 lr=1e-4 continuation from passed 2K smoke to 25K.

This runner is isolated and has no 50K/75K/100K mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine

PLAN = ROOT / "configs/073_hl_seed1_lr1e4_rescue25k.json"
BASE_YAML = ROOT / "experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
INPUT_ROOT = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual"
OUT = ROOT / "benchmark_results/073_00_hla_lowIC_lr1e4_seed1_rescue25k"
PROMPT = ROOT / "prompts/2026-08-14_hl_seed1_lr1e4_rescue25k.md"
LEVELS_I, LEVELS_N = [0.0, 15.0, 30.0, 45.0], [0.0, 40.0, 80.0, 120.0]
SMOKE_ROOT = ROOT / "benchmark_results/072_00_hla_lowIC_lr1e4_seed1_maskableppo_smoke2k"
SMOKE_MODEL = SMOKE_ROOT / "models/HLA/HLA_half_split_stress_aware_maskableppo_seed1_ckpt2000.zip"
CHECKPOINTS = [5000, 10000, 25000]
TRAIN_YEARS = list(range(2004, 2014))
VALID_YEARS = list(range(2014, 2024))
REQUIRED = {"dap", "requested_discrete_action_index", "discrete_action_index", "raw_action_amir", "raw_action_anfer", "safe_action_amir", "safe_action_anfer", "previous_cumulative_irrigation", "previous_cumulative_n", "season_cumulative_irrigation", "season_cumulative_n"}


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def preflight() -> dict:
    split = batch.load_split()
    local = split[split.station_code.eq("HLA")].sort_values("year")
    train = local[local.split.eq("train")].year.astype(int).tolist()
    valid = local[local.split.eq("validation")].year.astype(int).tolist()
    issues = []
    mzx = INPUT_ROOT / "HL/CNHL0701_corrected_IC123.MZX"
    if not INPUT_ROOT.exists() or not mzx.exists(): issues.append("lowIC input or MZX missing")
    if train != TRAIN_YEARS or valid != VALID_YEARS: issues.append("split mismatch")
    if not SMOKE_MODEL.exists(): issues.append("passed 2K smoke model missing")
    if not (SMOKE_ROOT / "072_00_smoke_result.json").exists(): issues.append("passed 2K smoke result missing")
    else:
        result = json.loads((SMOKE_ROOT / "072_00_smoke_result.json").read_text(encoding="utf-8"))
        if result.get("next_step_allowed") is not True: issues.append("2K smoke gate is not passed")
    if OUT.exists() and any(OUT.iterdir()): issues.append("non-empty output root exists")
    return {"task": "073_00_hla_lowIC_lr1e4_seed1_rescue25k", "site": "HL", "seed": 1, "input_root": rel(INPUT_ROOT), "source_mzx": rel(mzx), "train_years": train, "validation_years": valid, "checkpoints": CHECKPOINTS, "output_root": rel(OUT), "issues": issues, "next_step_allowed": not issues}


def make_effective_yaml() -> Path:
    cfg = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    cfg["seed"] = 1
    cfg["total_timesteps"] = 25000
    cfg["paths"]["output_root"] = rel(OUT)
    cfg["ppo"].update({"learning_rate": 1e-4, "gamma": 1.0, "gae_lambda": 1.0, "n_steps": 144, "batch_size": 144, "n_epochs": 5, "ent_coef": 0.01, "clip_range": 0.2, "net_arch": [64, 64]})
    path = ROOT / "configs/073_hla_seed1_lr1e4_effective.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def audit_checkpoint(summary: pd.DataFrame, checkpoint: int):
    selected = summary[pd.to_numeric(summary.checkpoint_step, errors="coerce").eq(checkpoint)]
    rows = []
    for record in selected.itertuples(index=False):
        path = ROOT / str(record.daily_csv_path)
        rec = {"year": int(record.year), "daily_csv_path": str(record.daily_csv_path), "daily_exists": path.exists()}
        if not path.exists(): rows.append(rec); continue
        d = pd.read_csv(path); missing = sorted(REQUIRED - set(d.columns)); rec["missing_required_columns"] = ";".join(missing)
        if missing: rows.append(rec); continue
        req = pd.to_numeric(d.requested_discrete_action_index, errors="coerce"); actual = pd.to_numeric(d.discrete_action_index, errors="coerce")
        raw_i, raw_n = pd.to_numeric(d.raw_action_amir, errors="coerce"), pd.to_numeric(d.raw_action_anfer, errors="coerce")
        safe_i, safe_n = pd.to_numeric(d.safe_action_amir, errors="coerce"), pd.to_numeric(d.safe_action_anfer, errors="coerce")
        prev_i, prev_n = pd.to_numeric(d.previous_cumulative_irrigation, errors="coerce"), pd.to_numeric(d.previous_cumulative_n, errors="coerce")
        cum_i, cum_n = pd.to_numeric(d.season_cumulative_irrigation, errors="coerce"), pd.to_numeric(d.season_cumulative_n, errors="coerce")
        pos = (safe_i > 1e-9) | (safe_n > 1e-9); pairs = [f"I{i:g}/N{n:g}" for i, n in zip(safe_i[pos], safe_n[pos])]
        rec.update({"null_required_field_rows": int(d[list(REQUIRED)].isna().any(axis=1).sum()), "off_grid_rows": int((~safe_i.isin(LEVELS_I)).sum() + (~safe_n.isin(LEVELS_N)).sum()), "requested_to_safe_mismatch_rows": int((req != actual).sum()), "raw_to_safe_mismatch_rows": int((~np.isclose(raw_i, safe_i, atol=1e-6) | ~np.isclose(raw_n, safe_n, atol=1e-6)).sum()), "safe_to_dssat_mismatch_rows": int((pos & (~np.isclose(cum_i-prev_i, safe_i, atol=1e-6) | ~np.isclose(cum_n-prev_n, safe_n, atol=1e-6))).sum()), "positive_actions": int(pos.sum()), "after_dap1_actions": int((pos & (pd.to_numeric(d.dap, errors="coerce") > 1)).sum()), "action_signature": ";".join(sorted(set(pairs))), "unique_nonzero_action_pairs": len(set(pairs))})
        rows.append(rec)
    audit = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    def total(name): return int(pd.to_numeric(audit.get(name, pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    sig = audit.get("action_signature", pd.Series(dtype=str)).fillna("").astype(str)
    gate = {"checkpoint": checkpoint, "validation_rows": len(audit), "daily_files_complete": bool(len(audit) == 10 and audit.daily_exists.fillna(False).all()), "audit_fields_complete": bool(len(audit) == 10 and audit.get("missing_required_columns", pd.Series(dtype=str)).fillna("").eq("").all() and total("null_required_field_rows") == 0), "off_grid_rows": total("off_grid_rows"), "requested_to_safe_mismatch_rows": total("requested_to_safe_mismatch_rows"), "raw_to_safe_mismatch_rows": total("raw_to_safe_mismatch_rows"), "safe_to_dssat_mismatch_rows": total("safe_to_dssat_mismatch_rows"), "unique_nonzero_action_pairs": len({x for s in sig for x in s.split(";") if x}), "post_dap1_years": int((pd.to_numeric(audit.get("after_dap1_actions", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum()), "crossyear_action_signature_count": int(sig.nunique()), "all_action_collapse": bool(len(audit) and sig.nunique() == 1)}
    gate["mechanism_pass"] = bool(gate["validation_rows"] == 10 and gate["daily_files_complete"] and gate["audit_fields_complete"] and all(gate[x] == 0 for x in ["off_grid_rows", "requested_to_safe_mismatch_rows", "raw_to_safe_mismatch_rows", "safe_to_dssat_mismatch_rows"]) and gate["unique_nonzero_action_pairs"] >= 3 and gate["post_dap1_years"] >= 8 and gate["crossyear_action_signature_count"] >= 2 and not gate["all_action_collapse"])
    return audit, gate


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=["dry-run", "rescue25k"], required=True); args = parser.parse_args()
    pf = preflight(); print(json.dumps(pf, indent=2))
    if args.phase == "dry-run": return 0 if pf["next_step_allowed"] else 2
    if not pf["next_step_allowed"]: raise SystemExit("preflight failed")
    OUT.mkdir(parents=True); (OUT / "configs").mkdir(); (OUT / "audits").mkdir(); (OUT / "evaluation").mkdir()
    effective = make_effective_yaml(); shutil.copy2(effective, OUT / "configs" / effective.name); shutil.copy2(PLAN, OUT / "configs" / PLAN.name); shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)
    old_engine = {k: getattr(engine, k) for k in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_batch = (batch.CONFIG, batch.OUT, batch.DOC, batch.SEED, dict(batch.SITE_NAMES)); old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    start = time.perf_counter(); rss0 = psutil.Process().memory_info().rss
    try:
        batch.CONFIG = effective; batch.OUT = OUT; batch.DOC = ROOT / "docs/073_00_hla_lowIC_lr1e4_seed1_rescue25k_record.md"; batch.SEED = 1; batch.SITE_NAMES["HLA"] = "HL"
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = INPUT_ROOT
        engine.TASK_ID, engine.TASK_NAME, engine.BASE_OUT = "073_00", "hla_lowIC_lr1e4_seed1_rescue25k", OUT; engine.BASE_DOC = batch.DOC; engine.PROMPT = PROMPT; engine.LOWIC_INPUT_ROOT = INPUT_ROOT; engine.STATION = "HLA"; engine.SITES = ["HLA"]; engine.BINARY_IRRIGATION_LEVELS, engine.BINARY_NITROGEN_LEVELS = LEVELS_I, LEVELS_N
        config = batch.load_config(); selection = batch.build_selection(batch.load_split()); envcfg = batch.direct_ppo.build_env_config(config, selection); env = batch.RandomYearEnv(config, envcfg, "HLA", TRAIN_YEARS, 1)
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(str(SMOKE_MODEL), env=env, device="cpu")
        callback = batch.FixedStepCheckpointCallback("HLA", CHECKPOINTS)
        model.learn(total_timesteps=23000, reset_num_timesteps=False, progress_bar=False, callback=callback.callback)
        env.close()
        inventory = []
        for step in CHECKPOINTS:
            path = batch.model_path("HLA", step); inventory.append({"station_code": "HLA", "site": "HL", "seed": 1, "checkpoint_step": step, "run_status": "ok" if path.exists() else "missing", "model_path": rel(path) if path.exists() else "", "model_sha256": sha256(path) if path.exists() else ""})
        pd.DataFrame(inventory).to_csv(OUT / "evaluation/073_training_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
        rows = []
        for item in inventory:
            if item["run_status"] != "ok": continue
            row = pd.Series(item)
            for year in VALID_YEARS: rows.append(batch.evaluate_checkpoint(config, envcfg, row, year))
        summary = pd.DataFrame(rows); summary.to_csv(OUT / "evaluation/073_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
        audits, gates = {}, {}
        for step in CHECKPOINTS:
            audit, gate = audit_checkpoint(summary, step); path = OUT / "audits" / f"073_ckpt{step}_action_audit.csv"; audit.to_csv(path, index=False, encoding="utf-8-sig"); audits[str(step)] = rel(path); gates[str(step)] = gate
        result = {**pf, "phase": "rescue25k", "elapsed_s": time.perf_counter() - start, "rss_delta_bytes": psutil.Process().memory_info().rss - rss0, "effective_yaml": rel(effective), "action_audits": audits, "action_gates": gates, "WP_ET_status": "not available from training summaries; not inferred", "next_step_allowed": False, "stop_reason": "25K complete; no 50K/75K/100K authorized"}
        (OUT / "073_rescue_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); (OUT / "073_run_manifest.json").write_text(json.dumps({"status": "completed_rescue25k", "result": result}, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        for k, v in old_engine.items(): setattr(engine, k, v)
        batch.CONFIG, batch.OUT, batch.DOC, batch.SEED = old_batch[:4]; batch.SITE_NAMES.clear(); batch.SITE_NAMES.update(old_batch[4]); ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
    return 0


if __name__ == "__main__": raise SystemExit(main())
