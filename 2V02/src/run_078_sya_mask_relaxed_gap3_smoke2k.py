"""078: SYA seed3 single-factor action-mask relaxation, 2K smoke only."""
from __future__ import annotations

import argparse
import hashlib
import json
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
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch

PLAN = ROOT / "configs" / "078_sya_mask_relaxed_gap3_smoke2k.json"
PROMPT = ROOT / "prompts" / "2026-08-13_sya_mask_relaxed_gap3_smoke2k.md"
BASE_YAML = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
LEVELS_I = [0.0, 15.0, 30.0, 45.0]
LEVELS_N = [0.0, 40.0, 80.0, 120.0]
REQUIRED = {
    "dap", "requested_discrete_action_index", "discrete_action_index",
    "raw_action_amir", "raw_action_anfer", "safe_action_amir", "safe_action_anfer",
    "previous_cumulative_irrigation", "previous_cumulative_n",
    "season_cumulative_irrigation", "season_cumulative_n", "mask_valid_action_count",
}


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def load() -> dict[str, Any]:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def out(c: dict[str, Any]) -> Path:
    return ROOT / c["output_root"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def scalar(x: Any) -> float:
    if isinstance(x, (int, float, np.number)):
        return float(x)
    if hasattr(x, "value"):
        return float(x.value)
    if callable(x):
        return float(x(1.0))
    return float(x)


def loaded_kwargs(path: Path) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(str(path), device="cpu")
    ext = model.policy.mlp_extractor
    widths = lambda seq: [int(x.out_features) for x in seq if hasattr(x, "out_features")]
    return {
        "learning_rate": scalar(model.learning_rate), "gamma": float(model.gamma),
        "gae_lambda": float(model.gae_lambda), "n_steps": int(model.n_steps),
        "batch_size": int(model.batch_size), "n_epochs": int(model.n_epochs),
        "ent_coef": float(model.ent_coef), "clip_range": scalar(model.clip_range),
        "policy_net_widths": widths(ext.policy_net), "value_net_widths": widths(ext.value_net),
    }


def preflight(c: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    input_root = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
    if not (input_root / "SY/CNSY1201.MZX").exists():
        issues.append("originIC SY input missing")
    split = batch.load_split()
    local = split[split.station_code.eq("SYA")].sort_values("year")
    train = local[local.split.eq("train")].year.astype(int).tolist()
    valid = local[local.split.eq("validation")].year.astype(int).tolist()
    if train != c["train_years"] or valid != c["validation_years"]:
        issues.append("split years mismatch")
    if c["seed"] != 3:
        issues.append("seed must be 3")
    if c["actions"] != {"irrigation_levels_mm": LEVELS_I, "nitrogen_levels_kg_ha": LEVELS_N}:
        issues.append("action grid mismatch")
    if c["observation_contract"] != {"base": "032_22_raw_observation", "normalization_enabled": False, "weather_forecast_enabled": False}:
        issues.append("observation contract mismatch")
    if not PROMPT.exists() or not BASE_YAML.exists():
        issues.append("prompt or base YAML missing")
    if out(c).exists():
        issues.append("isolated output root already exists")
    return {
        "task": f"{c['task_id']}_{c['task_name']}", "seed": c["seed"],
        "input_root": rel(input_root), "train_years": train, "validation_years": valid,
        "baseline_reference": c["baseline_reference"], "issues": issues,
        "next_step_allowed": not issues,
    }


def _event_signature(df: pd.DataFrame) -> tuple[str, str, int, float]:
    i = pd.to_numeric(df.safe_action_amir, errors="coerce").fillna(0.0)
    n = pd.to_numeric(df.safe_action_anfer, errors="coerce").fillna(0.0)
    dap = pd.to_numeric(df.dap, errors="coerce")
    pos = (i.abs() > 1e-9) | (n.abs() > 1e-9)
    events = [f"DAP{float(d):g}:I{float(iv):g}/N{float(nv):g}" for d, iv, nv in zip(dap[pos], i[pos], n[pos])]
    pairs = sorted({x.split(":", 1)[1] for x in events})
    counts = pd.Series([x.split(":", 1)[1] for x in events]).value_counts().to_numpy(dtype=float) if events else np.array([])
    probs = counts / counts.sum() if counts.size else counts
    entropy = float(-(probs * np.log2(probs)).sum()) if probs.size else 0.0
    return ";".join(pairs) or "NONE", "|".join(events) or "NONE", int(pos.sum()), entropy


def audit(summary: pd.DataFrame, step: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    wanted = pd.to_numeric(summary.checkpoint_step, errors="coerce").eq(step)
    for r in summary[wanted].itertuples(index=False):
        p = ROOT / str(r.daily_csv_path)
        rec: dict[str, Any] = {"year": int(r.year), "daily_exists": p.exists(), "daily_csv_path": str(r.daily_csv_path)}
        if not p.exists():
            rec["missing_required_columns"] = "daily_file_missing"; rows.append(rec); continue
        d = pd.read_csv(p, low_memory=False)
        missing = sorted(REQUIRED - set(d.columns)); rec["missing_required_columns"] = ";".join(missing)
        if missing:
            rows.append(rec); continue
        z = d[list(REQUIRED)].apply(pd.to_numeric, errors="coerce")
        i, n = z.safe_action_amir, z.safe_action_anfer
        pos = (i.abs() > 1e-9) | (n.abs() > 1e-9)
        pair_sig, timed_sig, event_count, entropy = _event_signature(d)
        rec.update({
            "null_required_field_rows": int(z.isna().any(axis=1).sum()),
            "off_grid_rows": int((~i.isin(LEVELS_I)).sum() + (~n.isin(LEVELS_N)).sum()),
            "requested_to_safe_mismatch_rows": int((z.requested_discrete_action_index != z.discrete_action_index).sum()),
            "raw_to_safe_mismatch_rows": int((~np.isclose(z.raw_action_amir, i) | ~np.isclose(z.raw_action_anfer, n)).sum()),
            "safe_to_dssat_mismatch_rows": int((pos & (~np.isclose(z.season_cumulative_irrigation - z.previous_cumulative_irrigation, i) | ~np.isclose(z.season_cumulative_n - z.previous_cumulative_n, n))).sum()),
            "positive_actions": int(pos.sum()),
            "after_dap1_actions": int((pos & (z.dap > 1)).sum()),
            "pair_set_signature": pair_sig,
            "timed_event_signature": timed_sig,
            "event_count": event_count,
            "pair_entropy_bits": entropy,
            "mask_valid_action_count_min": int(z.mask_valid_action_count.min()),
            "mask_valid_action_count_mean": float(z.mask_valid_action_count.mean()),
            "mask_valid_action_count_one_rows": int((z.mask_valid_action_count == 1).sum()),
        })
        rows.append(rec)
    a = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    total = lambda k: int(pd.to_numeric(a.get(k, pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    timed = a.get("timed_event_signature", pd.Series(dtype=str)).fillna("").astype(str)
    pairs = {x for s in a.get("pair_set_signature", pd.Series(dtype=str)).fillna("") for x in s.split(";") if x}
    g = {
        "checkpoint": int(step), "validation_rows": len(a),
        "daily_files_complete": bool(len(a) == 10 and a.daily_exists.fillna(False).all()),
        "audit_fields_complete": bool(len(a) == 10 and (a.get("missing_required_columns", pd.Series(dtype=str)).fillna("") == "").all() and total("null_required_field_rows") == 0),
        "off_grid_rows": total("off_grid_rows"), "requested_to_safe_mismatch_rows": total("requested_to_safe_mismatch_rows"),
        "raw_to_safe_mismatch_rows": total("raw_to_safe_mismatch_rows"), "safe_to_dssat_mismatch_rows": total("safe_to_dssat_mismatch_rows"),
        "unique_nonzero_action_pairs": len(pairs), "post_dap1_years": int((pd.to_numeric(a.get("after_dap1_actions", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum()),
        "legacy_pair_set_signature_count": int(a.get("pair_set_signature", pd.Series(dtype=str)).nunique()),
        "timed_event_signature_count": int(timed.nunique()),
        "all_timed_event_collapse": bool(len(a) and timed.nunique() == 1),
        "mask_valid_action_count_one_fraction": float(total("mask_valid_action_count_one_rows") / max(1, int(a.get("event_count", pd.Series(dtype=float)).sum() + (len(a) * 0)))) if False else float(pd.to_numeric(a.get("mask_valid_action_count_one_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() / max(1, int(sum(len(pd.read_csv(ROOT / str(p), usecols=["mask_valid_action_count"])) for p in a.daily_csv_path))) if len(a) else 0.0),
    }
    g["mechanism_pass"] = bool(g["validation_rows"] == 10 and g["daily_files_complete"] and g["audit_fields_complete"] and all(g[k] == 0 for k in ["off_grid_rows", "requested_to_safe_mismatch_rows", "raw_to_safe_mismatch_rows", "safe_to_dssat_mismatch_rows"]) and g["unique_nonzero_action_pairs"] >= 3 and g["post_dap1_years"] >= 8 and g["timed_event_signature_count"] >= 2)
    return a, g


def run() -> dict[str, Any]:
    c = load(); pf = preflight(c)
    if not pf["next_step_allowed"]:
        raise RuntimeError("; ".join(pf["issues"]))
    root = out(c); root.mkdir(parents=True); (root / "configs").mkdir()
    shutil.copy2(PLAN, root / "configs" / PLAN.name); shutil.copy2(PROMPT, root / "configs" / PROMPT.name)
    effective = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    effective["seed"] = int(c["seed"]); effective["total_timesteps"] = 2000; effective["ppo"].update(c["ppo"])
    effective["action_safety"]["min_days_between_irrigation"] = 3; effective["action_safety"]["min_days_between_fertilization"] = 3
    effective["paths"]["output_root"] = rel(root)
    ep = ROOT / "configs" / "078_sya_mask_relaxed_gap3_effective.yaml"; ep.write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8"); shutil.copy2(ep, root / "configs" / ep.name)
    old = (batch.CONFIG, batch.OUT, batch.DOC, batch.SEED, batch.TOTAL_TIMESTEPS, batch.CHECKPOINT_STEPS, list(batch.SITES), dict(batch.SITE_NAMES), ppo_safe_rendering.MULTISITE_INPUT_ROOT)
    start = time.perf_counter(); env = None
    try:
        batch.CONFIG, batch.OUT, batch.DOC, batch.SEED = ep, root, root / "078_record.md", int(c["seed"]); batch.TOTAL_TIMESTEPS = 2000; batch.CHECKPOINT_STEPS = [1000, 2000]; batch.SITES = ["SYA"]; batch.SITE_NAMES["SYA"] = "SY"; ppo_safe_rendering.MULTISITE_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
        config = batch.load_config(); split = batch.load_split(); selection = batch.build_selection(split); envcfg = batch.direct_ppo.build_env_config(config, selection); envcfg["paths"]["output_root"] = rel(root)
        train_years = split[split.station_code.eq("SYA") & split.split.eq("train")].year.astype(int).tolist(); env = batch.RandomYearEnv(config, envcfg, "SYA", train_years, int(c["seed"]))
        from sb3_contrib import MaskablePPO
        model = MaskablePPO("MlpPolicy", env, verbose=0, seed=int(c["seed"]), device="cpu", **batch.base.ppo_kwargs(config)); cb = batch.FixedStepCheckpointCallback("SYA", [1000, 2000]); model.learn(total_timesteps=2000, reset_num_timesteps=True, progress_bar=False, callback=cb.callback); env.close(); env = None
        inv = []
        for step in [1000, 2000]:
            p = batch.model_path("SYA", step); inv.append({"station_code": "SYA", "site": "SY", "seed": int(c["seed"]), "checkpoint_step": step, "run_status": "ok" if p.exists() else "missing", "model_path": rel(p) if p.exists() else "", "model_sha256": sha(p) if p.exists() else ""})
        inventory = pd.DataFrame(inv); (root / "evaluation").mkdir(); inventory.to_csv(root / "evaluation" / "078_training_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
        rows = [batch.evaluate_checkpoint(config, envcfg, pd.Series(r), year) for r in inv for year in c["validation_years"]]; summary = pd.DataFrame(rows); summary.to_csv(root / "evaluation" / "078_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
        audits, gates = {}, {}
        for step in [1000, 2000]:
            a, g = audit(summary, step); p = root / "audits" / f"078_ckpt{step}_action_audit.csv"; p.parent.mkdir(exist_ok=True); a.to_csv(p, index=False, encoding="utf-8-sig"); audits[str(step)] = rel(p); gates[str(step)] = g
        final_model = ROOT / inv[-1]["model_path"]; result = {**pf, "phase": "mask_relaxed_gap3_smoke2k", "elapsed_s": time.perf_counter() - start, "effective_yaml": rel(ep), "effective_kwargs_from_loaded_model": loaded_kwargs(final_model), "action_audits": audits, "action_gates": gates, "WP_ET_status": "not available from smoke summaries; not inferred", "next_step_allowed": bool(gates["2000"]["mechanism_pass"]), "stop_reason": "smoke complete; no long training in this task"}
        (root / "078_smoke_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); (root / "078_run_manifest.json").write_text(json.dumps({"status": "completed_smoke", "config": c, "preflight": pf, "result": result}, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(result, ensure_ascii=False, indent=2)); return result
    finally:
        if env is not None: env.close()
        batch.CONFIG, batch.OUT, batch.DOC, batch.SEED, batch.TOTAL_TIMESTEPS, batch.CHECKPOINT_STEPS, batch.SITES, batch.SITE_NAMES, ppo_safe_rendering.MULTISITE_INPUT_ROOT = old


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=["dry-run", "smoke"], required=True); args = parser.parse_args(); c = load(); print(json.dumps(preflight(c), ensure_ascii=False, indent=2) if args.phase == "dry-run" else json.dumps(run(), ensure_ascii=False, indent=2))
