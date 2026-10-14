"""Isolated 064 HL PPO-kwargs smoke runner; never mutates 054_00 artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")) if str(ROOT / "src") not in sys.path else None

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine

CONFIG = ROOT / "configs" / "064_00_hla_lowIC_ppo_kwargs_smoke_sweep.json"
BASE_YAML = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
STATION, SITE = "HLA", "HLA"
LEVELS_I, LEVELS_N = [0.0, 15.0, 30.0, 45.0], [0.0, 40.0, 80.0, 120.0]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def effective_ppo(plan: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result = dict(plan["fixed_ppo"])
    result.update(candidate["override"])
    return result


def preflight(plan: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    split = batch.load_split()
    split = split[split.station_code.eq(STATION)].sort_values("year")
    train = split[split.split.eq("train")].year.astype(int).tolist()
    valid = split[split.split.eq("validation")].year.astype(int).tolist()
    ppo = effective_ppo(plan, candidate)
    issues: list[str] = []
    if not INPUT_ROOT.exists() or not (INPUT_ROOT / "HL" / "CNHL0701_corrected_IC123.MZX").exists():
        issues.append("HL lowIC input/MZX missing")
    if train != plan["scope"]["train_years"] or valid != plan["scope"]["validation_years"]:
        issues.append("train/validation split mismatch")
    if plan["actions"]["irrigation_levels_mm"] != LEVELS_I or plan["actions"]["nitrogen_levels_kg_ha"] != LEVELS_N:
        issues.append("action grid mismatch")
    if plan["observation_contract"] != {"base": "046_02_raw_observation", "normalization_enabled": False, "weather_forecast_enabled": False}:
        issues.append("observation/weather contract mismatch")
    expected = {"learning_rate": 3e-4, "gamma": 1.0, "gae_lambda": 1.0, "n_steps": 144, "batch_size": 144, "n_epochs": 5, "ent_coef": 0.01, "clip_range": 0.2, "net_arch": [64, 64]}
    changed = {key for key, value in ppo.items() if expected.get(key) != value}
    if changed != set(candidate["override"]):
        issues.append(f"non-single-factor PPO change: {sorted(changed)}")
    return {"candidate": candidate["id"], "effective_ppo": ppo, "train_years": train, "validation_years": valid,
            "input_root": rel(INPUT_ROOT), "source_mzx": rel(INPUT_ROOT / "HL" / "CNHL0701_corrected_IC123.MZX"),
            "issues": issues, "next_step_allowed": not issues}


def audit_rows(summary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in summary.itertuples(index=False):
        daily_path = ROOT / str(r.daily_csv_path)
        daily = pd.read_csv(daily_path)
        raw_i, raw_n = pd.to_numeric(daily.raw_action_amir), pd.to_numeric(daily.raw_action_anfer)
        safe_i, safe_n = pd.to_numeric(daily.safe_action_amir), pd.to_numeric(daily.safe_action_anfer)
        prev_i, prev_n = pd.to_numeric(daily.previous_cumulative_irrigation), pd.to_numeric(daily.previous_cumulative_n)
        cum_i, cum_n = pd.to_numeric(daily.season_cumulative_irrigation), pd.to_numeric(daily.season_cumulative_n)
        positive = (safe_i > 1e-9) | (safe_n > 1e-9)
        transmitted = (~positive) | (np.isclose(cum_i - prev_i, safe_i, atol=1e-6) & np.isclose(cum_n - prev_n, safe_n, atol=1e-6))
        pairs = sorted({f"I{float(i):g}/N{float(n):g}" for i, n in zip(safe_i[positive], safe_n[positive])})
        rows.append({"year": int(r.year), "daily_csv_path": str(r.daily_csv_path), "positive_actions": int(positive.sum()),
                     "after_dap1_actions": int((positive & (pd.to_numeric(daily.dap) > 1)).sum()),
                     "off_grid_rows": int((~safe_i.isin(LEVELS_I)).sum() + (~safe_n.isin(LEVELS_N)).sum()),
                     "requested_to_safe_changed_rows": int((pd.to_numeric(daily.requested_discrete_action_index) != pd.to_numeric(daily.discrete_action_index)).sum()),
                     "raw_to_safe_changed_rows": int((~np.isclose(raw_i, safe_i, atol=1e-6) | ~np.isclose(raw_n, safe_n, atol=1e-6)).sum()),
                     "safe_to_dssat_mismatch_rows": int((positive & ~transmitted).sum()), "action_pairs": ";".join(pairs)})
    audit = pd.DataFrame(rows).sort_values("year")
    all_pairs = sorted({x for text in audit.action_pairs for x in text.split(";") if x})
    gate = {"validation_rows": len(audit), "off_grid_rows": int(audit.off_grid_rows.sum()),
            "safe_to_dssat_mismatch_rows": int(audit.safe_to_dssat_mismatch_rows.sum()),
            "unique_nonzero_action_pairs": len(all_pairs), "mean_after_dap1_actions": float(audit.after_dap1_actions.mean()),
            "crossyear_action_signature_count": int(audit.action_pairs.nunique())}
    gate["mechanism_pass"] = bool(gate["validation_rows"] == 10 and gate["off_grid_rows"] == 0 and gate["safe_to_dssat_mismatch_rows"] == 0 and gate["unique_nonzero_action_pairs"] >= 3 and gate["mean_after_dap1_actions"] > 0 and gate["crossyear_action_signature_count"] != 1)
    return audit, gate


def observed_kwargs(model_path: Path) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(str(model_path), device="cpu")
    def widths(seq: Any) -> list[int]:
        return [int(layer.out_features) for layer in seq if hasattr(layer, "out_features")]
    extractor = model.policy.mlp_extractor
    return {"learning_rate": model.learning_rate, "gamma": model.gamma, "gae_lambda": model.gae_lambda, "n_steps": model.n_steps,
            "batch_size": model.batch_size, "n_epochs": model.n_epochs, "ent_coef": model.ent_coef, "clip_range": model.clip_range,
            "policy_kwargs": model.policy_kwargs, "policy_net_widths": widths(extractor.policy_net), "value_net_widths": widths(extractor.value_net)}


def run_candidate(plan: dict[str, Any], candidate: dict[str, Any], dry_run: bool, attempt: str = "") -> dict[str, Any]:
    pf = preflight(plan, candidate)
    suffix = f"_{attempt.strip()}" if attempt.strip() else ""
    name = f"{plan['task_name']}_{candidate['id']}_smoke2k{suffix}"
    out = ROOT / "benchmark_results" / f"{plan['task_id']}_{name}"
    if dry_run:
        return {**pf, "output_root": rel(out), "mode": "dry_run"}
    if not pf["next_step_allowed"]:
        raise RuntimeError("; ".join(pf["issues"]))
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {rel(out)}")
    (out / "configs").mkdir(parents=True)
    base_cfg = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    base_cfg["ppo"].update(pf["effective_ppo"])
    # The inherited engine copies CONFIG into OUT/configs; keep the generated
    # source outside OUT to avoid shutil.copy2(src, src) and preserve isolation.
    effective_yaml = ROOT / "configs" / f"064_00_effective_{candidate['id']}_ppo_config.yaml"
    effective_yaml.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")
    shutil.copy2(CONFIG, out / "configs" / CONFIG.name)
    old = {key: getattr(engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_config, old_out, old_doc = batch.CONFIG, batch.OUT, batch.DOC
    old_root, old_names = ppo_safe_rendering.MULTISITE_INPUT_ROOT, dict(batch.SITE_NAMES)
    try:
        engine.TASK_ID, engine.TASK_NAME, engine.BASE_OUT = plan["task_id"], name, out
        engine.BASE_DOC, engine.LOWIC_INPUT_ROOT = ROOT / "docs" / f"{plan['task_id']}_{name}_record.md", INPUT_ROOT
        engine.STATION, engine.SITES = STATION, [STATION]
        engine.BINARY_IRRIGATION_LEVELS, engine.BINARY_NITROGEN_LEVELS = LEVELS_I, LEVELS_N
        batch.CONFIG, batch.OUT, batch.DOC = effective_yaml, out, engine.BASE_DOC
        batch.SITE_NAMES[STATION] = SITE
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = INPUT_ROOT
        engine.run_training(2000, [1000, 2000], suffix="")
        prefix = plan["task_id"]
        src = out / "evaluation" / "042_10_checkpoint_validation_summary.csv"
        summary = pd.read_csv(src)
        summary.to_csv(out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
        final = summary[pd.to_numeric(summary.checkpoint_step).eq(2000)].copy()
        audit, gate = audit_rows(final)
        (out / "audits").mkdir(exist_ok=True)
        audit.to_csv(out / "audits" / f"{prefix}_validation_action_audit.csv", index=False, encoding="utf-8-sig")
        inv = pd.read_csv(out / "evaluation" / "042_10_training_checkpoint_inventory.csv")
        model_path = ROOT / str(inv[pd.to_numeric(inv.checkpoint_step).eq(2000)].iloc[0].model_path)
        observed = observed_kwargs(model_path)
        metrics = final[["final_grnwt", "total_irrigation", "total_n", "PFP_N"]].mean(numeric_only=True).to_dict()
        result = {**pf, "output_root": rel(out), "mode": "completed_smoke", "effective_yaml": rel(effective_yaml), "effective_kwargs_from_loaded_model": observed,
                  "validation_metrics_mean": metrics, "WP_ET_status": "not_available_in_2K_training_validation_summary; not inferred", "action_gate": gate,
                  "action_audit": rel(out / "audits" / f"{prefix}_validation_action_audit.csv"), "next_step_allowed": gate["mechanism_pass"]}
        (out / f"{prefix}_smoke_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (out / f"{prefix}_run_manifest.json").write_text(json.dumps({"status": "completed_smoke", "candidate": candidate, "preflight": pf, "effective_kwargs": pf["effective_ppo"], "observed": observed}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return result
    finally:
        for key, value in old.items(): setattr(engine, key, value)
        batch.CONFIG, batch.OUT, batch.DOC = old_config, old_out, old_doc
        batch.SITE_NAMES.clear(); batch.SITE_NAMES.update(old_names)
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=["A_net32", "B_lr1e4", "C_ent002"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--attempt", default="", help="isolated retry label; does not change candidate kwargs")
    args = parser.parse_args()
    plan = load_json(CONFIG)
    candidate = next(c for c in plan["candidates"] if c["id"] == args.candidate)
    print(json.dumps(run_candidate(plan, candidate, args.dry_run, args.attempt), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
