"""Seed=1 smoke validation for frozen SY 046_10 and LC 053_00 PPO policies.

This runner only supports an isolated 2K smoke. It is deliberately unable to
launch 25K/100K training, so a passing smoke still requires an explicit later
decision for resource-intensive seed validation.
"""
from __future__ import annotations

import argparse
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
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
import run_sya_ppo_configured_046_02 as base04602

PLAN_PATH = ROOT / "configs" / "069_sy_lc_seed1_stability_validation.json"
BASE_YAML = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
LEVELS_I = [0.0, 15.0, 30.0, 45.0]
LEVELS_N = [0.0, 40.0, 80.0, 120.0]
REQUIRED_DAILY_COLUMNS = {
    "dap", "requested_discrete_action_index", "discrete_action_index",
    "raw_action_amir", "raw_action_anfer", "safe_action_amir", "safe_action_anfer",
    "previous_cumulative_irrigation", "previous_cumulative_n",
    "season_cumulative_irrigation", "season_cumulative_n",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_plan() -> dict[str, Any]:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def cfg_for(site_key: str, smoke: bool = True) -> dict[str, Any]:
    plan = load_plan()
    spec = plan["experiments"][site_key]
    if not smoke:
        raise ValueError("This runner is smoke-only; 25K/100K require a separate authorized runner.")
    return {
        **spec,
        "runtime": plan["runtime"],
        "one_factor_contract": plan["one_factor_contract"],
        "seed": 1,
        "training": plan["smoke"],
        "actions": plan["actions"],
        "observation_contract": plan["observation_contract"],
        "scope": plan["scope"],
        "ppo": plan["ppo"],
        # The first empty SY root is preserved after a pre-training SameFileError.
        # This fresh attempt root prevents any overwrite of that failure evidence.
        "task_name": f"{spec['task_name']}_smoke2k_rerun1",
    }


def out_root(cfg: dict[str, Any]) -> Path:
    return ROOT / "benchmark_results" / f"{cfg['task_id']}_{cfg['task_name']}"


def input_root(cfg: dict[str, Any]) -> Path:
    return base04602.INPUT_PROFILES[str(cfg["input_profile"])]


def source_mzx(cfg: dict[str, Any]) -> Path:
    return input_root(cfg) / ("SY/CNSY1201.MZX" if cfg["station_code"] == "SYA" else "LC/CNLC0801.MZX")


def observed_kwargs(model_path: Path) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(str(model_path), device="cpu")
    def scalar(value: Any) -> float:
        """Stable-Baselines now wraps fixed schedules in FloatSchedule."""
        if isinstance(value, (int, float, np.number)):
            return float(value)
        if hasattr(value, "value"):
            return float(value.value)
        if callable(value):
            return float(value(1.0))
        return float(value)
    widths = lambda layers: [int(layer.out_features) for layer in layers if hasattr(layer, "out_features")]
    extractor = model.policy.mlp_extractor
    return {
        "learning_rate": scalar(model.learning_rate), "gamma": float(model.gamma),
        "gae_lambda": float(model.gae_lambda), "n_steps": int(model.n_steps),
        "batch_size": int(model.batch_size), "n_epochs": int(model.n_epochs),
        "ent_coef": float(model.ent_coef), "clip_range": scalar(model.clip_range),
        "policy_net_widths": widths(extractor.policy_net),
        "value_net_widths": widths(extractor.value_net),
    }


def preflight(cfg: dict[str, Any]) -> dict[str, Any]:
    split = batch.load_split()
    local = split[split["station_code"].astype(str).eq(str(cfg["station_code"]))].sort_values("year")
    train_years = local[local["split"].eq("train")]["year"].astype(int).tolist()
    validation_years = local[local["split"].eq("validation")]["year"].astype(int).tolist()
    original_model = ROOT / str(cfg["seed0_model"])
    root = input_root(cfg)
    issues: list[str] = []
    if int(cfg["seed"]) != 1:
        issues.append("seed must equal 1")
    if not root.exists() or not source_mzx(cfg).exists():
        issues.append("input root or station MZX is missing")
    if train_years != cfg["scope"]["train_years"] or validation_years != cfg["scope"]["validation_years"]:
        issues.append("configured train/validation years differ from engine split")
    if cfg["actions"]["irrigation_levels_mm"] != LEVELS_I or cfg["actions"]["nitrogen_levels_kg_ha"] != LEVELS_N:
        issues.append("16-action grid differs from frozen contract")
    if cfg["observation_contract"] != {"base": "046_02_raw_observation", "normalization_enabled": False, "weather_forecast_enabled": False}:
        issues.append("raw-observation/no-forecast contract differs")
    if not (ROOT / cfg["prompt"]).exists() or not original_model.exists():
        issues.append("prompt or frozen seed0 model is missing")
    if out_root(cfg).exists():
        issues.append("isolated output root already exists; refusing overwrite")
    frozen_kwargs: dict[str, Any] | None = None
    if original_model.exists():
        try:
            frozen_kwargs = observed_kwargs(original_model)
        except Exception as exc:  # dry run must reveal an unreadable model.
            issues.append(f"cannot load frozen seed0 model: {type(exc).__name__}: {exc}")
    return {
        "task": f"{cfg['task_id']}_{cfg['task_name']}", "reference_run": cfg["reference_run"],
        "station_code": cfg["station_code"], "site": cfg["site"], "seed": cfg["seed"],
        "input_profile": cfg["input_profile"], "input_root": rel(root), "source_mzx": rel(source_mzx(cfg)),
        "train_years": train_years, "validation_years": validation_years,
        "action_grid": cfg["actions"], "combined_action_count": 16,
        "observation_contract": cfg["observation_contract"], "configured_ppo": cfg["ppo"],
        "frozen_seed0_model": rel(original_model), "frozen_seed0_effective_ppo": frozen_kwargs,
        "output_root": rel(out_root(cfg)), "issues": issues, "next_step_allowed": not issues,
    }


def audit_checkpoint(summary: pd.DataFrame, checkpoint: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    selected = summary[pd.to_numeric(summary["checkpoint_step"], errors="coerce").eq(int(checkpoint))]
    for item in selected.itertuples(index=False):
        daily_path = ROOT / str(item.daily_csv_path)
        row: dict[str, Any] = {"year": int(item.year), "daily_csv_path": str(item.daily_csv_path), "daily_exists": daily_path.exists()}
        if not daily_path.exists():
            row["missing_required_columns"] = "daily_file_missing"
            records.append(row)
            continue
        daily = pd.read_csv(daily_path)
        missing = sorted(REQUIRED_DAILY_COLUMNS - set(daily.columns))
        row["missing_required_columns"] = ";".join(missing)
        if missing:
            records.append(row)
            continue
        required = daily[list(REQUIRED_DAILY_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        row["null_required_field_rows"] = int(required.isna().any(axis=1).sum())
        req, actual = required["requested_discrete_action_index"], required["discrete_action_index"]
        raw_i, raw_n = required["raw_action_amir"], required["raw_action_anfer"]
        safe_i, safe_n = required["safe_action_amir"], required["safe_action_anfer"]
        prev_i, prev_n = required["previous_cumulative_irrigation"], required["previous_cumulative_n"]
        cum_i, cum_n = required["season_cumulative_irrigation"], required["season_cumulative_n"]
        positive = (safe_i > 1e-9) | (safe_n > 1e-9)
        transmitted = (~positive) | (np.isclose(cum_i - prev_i, safe_i, atol=1e-6) & np.isclose(cum_n - prev_n, safe_n, atol=1e-6))
        pairs = sorted({f"I{float(i):g}/N{float(n):g}" for i, n in zip(safe_i[positive], safe_n[positive])})
        row.update({
            "off_grid_rows": int((~safe_i.isin(LEVELS_I)).sum() + (~safe_n.isin(LEVELS_N)).sum()),
            "requested_to_safe_mismatch_rows": int((req != actual).sum()),
            "raw_to_safe_mismatch_rows": int((~np.isclose(raw_i, safe_i, atol=1e-6) | ~np.isclose(raw_n, safe_n, atol=1e-6)).sum()),
            "safe_to_dssat_mismatch_rows": int((positive & ~transmitted).sum()),
            "positive_actions": int(positive.sum()),
            "after_dap1_actions": int((positive & (required["dap"] > 1)).sum()),
            "action_signature": ";".join(pairs),
        })
        records.append(row)
    audit = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    numeric_total = lambda col: int(pd.to_numeric(audit.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    signatures = audit.get("action_signature", pd.Series(dtype=str)).fillna("").astype(str)
    pairs = {pair for text in signatures for pair in text.split(";") if pair}
    missing = audit.get("missing_required_columns", pd.Series(dtype=str)).fillna("").astype(str)
    gate = {
        "checkpoint": int(checkpoint), "validation_years_expected": 10, "validation_rows": int(len(audit)),
        "daily_files_complete": bool(len(audit) == 10 and audit["daily_exists"].fillna(False).all()),
        "audit_fields_complete": bool(len(audit) == 10 and (missing == "").all() and numeric_total("null_required_field_rows") == 0),
        "off_grid_rows": numeric_total("off_grid_rows"),
        "requested_to_safe_mismatch_rows": numeric_total("requested_to_safe_mismatch_rows"),
        "raw_to_safe_mismatch_rows": numeric_total("raw_to_safe_mismatch_rows"),
        "safe_to_dssat_mismatch_rows": numeric_total("safe_to_dssat_mismatch_rows"),
        "unique_nonzero_action_pairs": len(pairs),
        "post_dap1_years": int((pd.to_numeric(audit.get("after_dap1_actions", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum()),
        "crossyear_action_signature_count": int(signatures.nunique()),
        "all_action_collapse": bool(len(audit) and signatures.nunique() == 1),
    }
    gate["mechanism_pass"] = bool(
        gate["validation_rows"] == 10 and gate["daily_files_complete"] and gate["audit_fields_complete"]
        and gate["off_grid_rows"] == 0 and gate["requested_to_safe_mismatch_rows"] == 0
        and gate["raw_to_safe_mismatch_rows"] == 0 and gate["safe_to_dssat_mismatch_rows"] == 0
        and gate["unique_nonzero_action_pairs"] >= 3 and gate["post_dap1_years"] >= 8
        and gate["crossyear_action_signature_count"] >= 2 and not gate["all_action_collapse"]
    )
    return audit, gate


def smoke_run(site_key: str) -> dict[str, Any]:
    cfg = cfg_for(site_key)
    pf = preflight(cfg)
    if not pf["next_step_allowed"]:
        raise RuntimeError("preflight failed: " + "; ".join(pf["issues"]))
    out = out_root(cfg)
    out.mkdir(parents=True)
    (out / "configs").mkdir()
    shutil.copy2(PLAN_PATH, out / "configs" / PLAN_PATH.name)
    shutil.copy2(ROOT / cfg["prompt"], out / "configs" / Path(cfg["prompt"]).name)
    effective = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    effective["seed"] = int(cfg["seed"])
    effective["ppo"].update(cfg["ppo"])
    effective["total_timesteps"] = int(cfg["training"]["total_timesteps"])
    effective["paths"]["output_root"] = rel(out)
    # base03222 copies CONFIG into OUT/configs.  Its source must therefore be
    # outside OUT/configs, otherwise shutil raises SameFileError before training.
    effective_path = ROOT / "configs" / f"{cfg['task_id']}_{cfg['station_code'].lower()}_effective_seed1_smoke_rerun1.yaml"
    effective_path.write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
    shutil.copy2(effective_path, out / "configs" / effective_path.name)
    old_engine = {key: getattr(engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_batch = batch.CONFIG, batch.OUT, batch.DOC, batch.SEED, dict(batch.SITE_NAMES)
    old_render_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    start = time.perf_counter()
    try:
        engine.TASK_ID, engine.TASK_NAME, engine.BASE_OUT = cfg["task_id"], cfg["task_name"], out
        engine.BASE_DOC = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
        engine.PROMPT, engine.LOWIC_INPUT_ROOT = ROOT / cfg["prompt"], input_root(cfg)
        engine.STATION, engine.SITES = cfg["station_code"], [cfg["station_code"]]
        engine.BINARY_IRRIGATION_LEVELS, engine.BINARY_NITROGEN_LEVELS = LEVELS_I, LEVELS_N
        batch.CONFIG, batch.OUT, batch.DOC, batch.SEED = effective_path, out, engine.BASE_DOC, int(cfg["seed"])
        batch.SITE_NAMES[cfg["station_code"]] = cfg["site"]
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = input_root(cfg)
        engine.run_training(int(cfg["training"]["total_timesteps"]), list(map(int, cfg["training"]["checkpoint_steps"])), suffix="")
    finally:
        for key, value in old_engine.items():
            setattr(engine, key, value)
        batch.CONFIG, batch.OUT, batch.DOC, batch.SEED = old_batch[:4]
        batch.SITE_NAMES.clear(); batch.SITE_NAMES.update(old_batch[4])
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_render_root
    summary = pd.read_csv(out / "evaluation" / "042_10_checkpoint_validation_summary.csv")
    inventory = pd.read_csv(out / "evaluation" / "042_10_training_checkpoint_inventory.csv")
    audits: dict[str, str] = {}
    gates: dict[str, Any] = {}
    for checkpoint in cfg["training"]["checkpoint_steps"]:
        audit, gate = audit_checkpoint(summary, int(checkpoint))
        audit_path = out / "audits" / f"{cfg['task_id']}_ckpt{checkpoint}_action_audit.csv"
        audit_path.parent.mkdir(exist_ok=True)
        audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
        audits[str(checkpoint)], gates[str(checkpoint)] = rel(audit_path), gate
    final_checkpoint = max(cfg["training"]["checkpoint_steps"])
    model_path = ROOT / str(inventory[pd.to_numeric(inventory["checkpoint_step"], errors="coerce").eq(final_checkpoint)].iloc[0]["model_path"])
    result = {
        **pf, "phase": "smoke2k", "elapsed_s": time.perf_counter() - start,
        "effective_config": rel(effective_path), "effective_kwargs_from_loaded_seed1_model": observed_kwargs(model_path),
        "action_audits": audits, "action_gates": gates,
        "WP_ET_status": "not_available_from_training summaries; not inferred",
        "next_step_allowed": gates[str(final_checkpoint)]["mechanism_pass"],
    }
    result["stop_reason"] = "smoke gate passed; long training is intentionally not authorized" if result["next_step_allowed"] else "2K smoke mechanism gate failed; long training forbidden"
    (out / f"{cfg['task_id']}_smoke_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"{cfg['task_id']}_run_manifest.json").write_text(json.dumps({"status": "completed_smoke", "config": cfg, "preflight": pf, "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=["SY", "LC"], required=True)
    parser.add_argument("--phase", choices=["dry-run", "smoke"], required=True)
    args = parser.parse_args()
    if args.phase == "dry-run":
        print(json.dumps(preflight(cfg_for(args.site)), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(smoke_run(args.site), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
