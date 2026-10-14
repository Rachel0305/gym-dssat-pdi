"""078/079 isolated HL/YC one-factor PPO stability smoke runner.

Only 2K smoke is exposed.  This runner never launches 25K/50K/75K/100K and
does not modify the frozen 054/055 outputs.
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
import psutil
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine


BASE_YAML = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
LEVELS_I = [0.0, 15.0, 30.0, 45.0]
LEVELS_N = [0.0, 40.0, 80.0, 120.0]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("station_code") not in {"HLA", "YCA"}:
        raise ValueError("This runner is registered only for HLA or YCA.")
    if plan.get("site") not in {"HL", "YC"}:
        raise ValueError("This runner is registered only for HL or YC.")
    return plan


def input_root(plan: dict[str, Any]) -> Path:
    return ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"


def source_mzx(plan: dict[str, Any]) -> Path:
    name = "CNHL0701_corrected_IC123.MZX" if plan["station_code"] == "HLA" else "CNYC0801.MZX"
    folder = "HL" if plan["station_code"] == "HLA" else "YC"
    return input_root(plan) / folder / name


def effective_ppo(plan: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result = dict(plan["fixed_ppo"])
    result.update(candidate["override"])
    return result


def preflight(plan: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    split = batch.load_split().copy()
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    split = split[split["station_code"].astype(str).eq(plan["station_code"])].sort_values("year")
    train = split.loc[split["split"].eq("train"), "year"].astype(int).tolist()
    valid = split.loc[split["split"].eq("validation"), "year"].astype(int).tolist()
    ppo = effective_ppo(plan, candidate)
    issues: list[str] = []
    if not input_root(plan).exists() or not source_mzx(plan).exists():
        issues.append(f"lowIC input or MZX missing: {rel(source_mzx(plan))}")
    if train != list(map(int, plan["scope"]["train_years"])):
        issues.append(f"train split mismatch: {train}")
    if valid != list(map(int, plan["scope"]["validation_years"])):
        issues.append(f"validation split mismatch: {valid}")
    if plan["actions"]["irrigation_levels_mm"] != LEVELS_I or plan["actions"]["nitrogen_levels_kg_ha"] != LEVELS_N:
        issues.append("action grid mismatch")
    expected_contract = {"base": "046_02_raw_observation", "normalization_enabled": False, "weather_forecast_enabled": False}
    if plan.get("observation_contract") != expected_contract:
        issues.append("observation/weather contract mismatch")
    allowed = {"clip_range", "n_epochs"}
    if set(candidate.get("override", {})) not in ({"clip_range"}, {"n_epochs"}):
        issues.append("candidate must change exactly one registered factor")
    expected = plan["fixed_ppo"]
    changed = {key for key, value in ppo.items() if expected.get(key) != value}
    if changed != set(candidate["override"]):
        issues.append(f"non-single-factor PPO change: {sorted(changed)}")
    if set(candidate["override"]) - allowed:
        issues.append("unapproved override")
    return {
        "task": f"{plan['task_id']}_{plan['task_name']}_{candidate['id']}",
        "candidate": candidate["id"],
        "station_code": plan["station_code"],
        "site": plan["site"],
        "input_root": rel(input_root(plan)),
        "source_mzx": rel(source_mzx(plan)),
        "train_years": train,
        "validation_years": valid,
        "effective_ppo": ppo,
        "issues": issues,
        "next_step_allowed": not issues,
    }


def _on_grid(values: pd.Series, allowed: list[float]) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return values.map(lambda x: bool(np.isclose(float(x), allowed, atol=1e-6).any()))


def audit_rows(summary: pd.DataFrame, require_crossyear_signature: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in summary.itertuples(index=False):
        daily_path = ROOT / str(record.daily_csv_path)
        if not daily_path.exists():
            rows.append({"year": int(record.year), "daily_exists": False})
            continue
        daily = pd.read_csv(daily_path)
        safe_i = pd.to_numeric(daily.get("safe_action_amir", pd.Series(0.0, index=daily.index)), errors="coerce").fillna(0.0)
        safe_n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(0.0, index=daily.index)), errors="coerce").fillna(0.0)
        raw_i = pd.to_numeric(daily.get("raw_action_amir", safe_i), errors="coerce").fillna(0.0)
        raw_n = pd.to_numeric(daily.get("raw_action_anfer", safe_n), errors="coerce").fillna(0.0)
        previous_i = pd.to_numeric(daily.get("previous_cumulative_irrigation", pd.Series(0.0, index=daily.index)), errors="coerce").fillna(0.0)
        previous_n = pd.to_numeric(daily.get("previous_cumulative_n", pd.Series(0.0, index=daily.index)), errors="coerce").fillna(0.0)
        cumulative_i = pd.to_numeric(daily.get("season_cumulative_irrigation", pd.Series(0.0, index=daily.index)), errors="coerce").fillna(0.0)
        cumulative_n = pd.to_numeric(daily.get("season_cumulative_n", pd.Series(0.0, index=daily.index)), errors="coerce").fillna(0.0)
        dap = pd.to_numeric(daily.get("dap", pd.Series(0.0, index=daily.index)), errors="coerce").fillna(0.0)
        positive = (safe_i > 1e-9) | (safe_n > 1e-9)
        transmitted = (~positive) | (np.isclose(cumulative_i - previous_i, safe_i, atol=1e-6) & np.isclose(cumulative_n - previous_n, safe_n, atol=1e-6))
        pairs = sorted({f"I{float(i):g}/N{float(n):g}" for i, n in zip(safe_i[positive], safe_n[positive])})
        requested = pd.to_numeric(daily.get("requested_discrete_action_index", daily.get("discrete_action_index", pd.Series(0, index=daily.index))), errors="coerce").fillna(0)
        actual = pd.to_numeric(daily.get("discrete_action_index", requested), errors="coerce").fillna(0)
        rows.append({
            "year": int(record.year),
            "daily_exists": True,
            "positive_actions": int(positive.sum()),
            "after_dap1_actions": int((positive & (dap > 1)).sum()),
            "off_grid_rows": int((~_on_grid(safe_i, LEVELS_I)).sum() + (~_on_grid(safe_n, LEVELS_N)).sum()),
            "requested_to_safe_changed_rows": int((requested != actual).sum()),
            "raw_to_safe_changed_rows": int((~np.isclose(raw_i, safe_i, atol=1e-6) | ~np.isclose(raw_n, safe_n, atol=1e-6)).sum()),
            "safe_to_dssat_mismatch_rows": int((positive & ~transmitted).sum()),
            "action_pairs": ";".join(pairs),
        })
    audit = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    all_pairs = sorted({piece for text in audit.get("action_pairs", pd.Series(dtype=str)).fillna("") for piece in str(text).split(";") if piece})
    signatures = int(audit.get("action_pairs", pd.Series(dtype=str)).nunique()) if not audit.empty else 0
    gate = {
        "validation_rows": int(len(audit)),
        "all_daily_files_exist": bool(len(audit) == 10 and audit["daily_exists"].all()),
        "off_grid_rows": int(audit.get("off_grid_rows", pd.Series(dtype=float)).sum()),
        "requested_to_safe_changed_rows": int(audit.get("requested_to_safe_changed_rows", pd.Series(dtype=float)).sum()),
        "raw_to_safe_changed_rows": int(audit.get("raw_to_safe_changed_rows", pd.Series(dtype=float)).sum()),
        "safe_to_dssat_mismatch_rows": int(audit.get("safe_to_dssat_mismatch_rows", pd.Series(dtype=float)).sum()),
        "unique_nonzero_action_pairs": len(all_pairs),
        "mean_after_dap1_actions": float(audit.get("after_dap1_actions", pd.Series(dtype=float)).mean()) if not audit.empty else 0.0,
        "crossyear_action_signature_count": signatures,
    }
    # This is a 2K technical smoke.  Cross-year signature count is recorded,
    # not used as a blocking condition, because the intervention targets late drift.
    gate["technical_pass"] = bool(
        gate["validation_rows"] == 10
        and gate["all_daily_files_exist"]
        and gate["off_grid_rows"] == 0
        and gate["requested_to_safe_changed_rows"] == 0
        and gate["raw_to_safe_changed_rows"] == 0
        and gate["safe_to_dssat_mismatch_rows"] == 0
        and gate["unique_nonzero_action_pairs"] >= 2
        and gate["mean_after_dap1_actions"] > 0
    )
    gate["crossyear_signature_gate"] = bool(gate["crossyear_action_signature_count"] >= 2)
    if require_crossyear_signature:
        gate["technical_pass"] = bool(gate["technical_pass"] and gate["crossyear_signature_gate"])
    return audit, gate


def observed_kwargs(model_path: Path) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(str(model_path), device="cpu")

    def widths(seq: Any) -> list[int]:
        return [int(layer.out_features) for layer in seq if hasattr(layer, "out_features")]

    extractor = model.policy.mlp_extractor
    return {
        "learning_rate": model.learning_rate,
        "gamma": model.gamma,
        "gae_lambda": model.gae_lambda,
        "n_steps": model.n_steps,
        "batch_size": model.batch_size,
        "n_epochs": model.n_epochs,
        "ent_coef": model.ent_coef,
        "clip_range": model.clip_range,
        "policy_kwargs": model.policy_kwargs,
        "policy_net_widths": widths(extractor.policy_net),
        "value_net_widths": widths(extractor.value_net),
    }


def run_candidate(plan: dict[str, Any], candidate: dict[str, Any], config_path: Path, dry_run: bool) -> dict[str, Any]:
    pf = preflight(plan, candidate)
    name = f"{plan['task_name']}_{candidate['id']}_smoke2k"
    out = ROOT / "benchmark_results" / f"{plan['task_id']}_{name}"
    if dry_run:
        return {**pf, "output_root": rel(out), "mode": "dry_run", "smoke_only": True}
    if not pf["next_step_allowed"]:
        raise RuntimeError("Preflight failed: " + "; ".join(pf["issues"]))
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {rel(out)}")
    out.mkdir(parents=True)
    (out / "configs").mkdir()
    shutil.copy2(config_path, out / "configs" / config_path.name)
    prompt = ROOT / plan.get("prompt", "prompts/078_hl_lowIC_stability_tuning.md" if plan["station_code"] == "HLA" else "prompts/079_yc_lowIC_stability_tuning.md")
    shutil.copy2(prompt, out / "configs" / prompt.name)
    effective_yaml = ROOT / "configs" / f"{plan['task_id']}_effective_{candidate['id']}_ppo_config.yaml"
    yaml_cfg = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    yaml_cfg["seed"] = int(plan["seed"])
    yaml_cfg["ppo"].update(pf["effective_ppo"])
    effective_yaml.write_text(yaml.safe_dump(yaml_cfg, sort_keys=False), encoding="utf-8")
    old = {key: getattr(engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"] if hasattr(engine, key)}
    old_config, old_out, old_doc, old_seed = batch.CONFIG, batch.OUT, batch.DOC, batch.SEED
    old_root, old_names = ppo_safe_rendering.MULTISITE_INPUT_ROOT, dict(batch.SITE_NAMES)
    start = time.perf_counter()
    rss0 = psutil.Process().memory_info().rss
    try:
        engine.TASK_ID, engine.TASK_NAME, engine.BASE_OUT = plan["task_id"], name, out
        engine.BASE_DOC = ROOT / "docs" / f"{plan['task_id']}_{name}_record.md"
        engine.PROMPT = prompt
        engine.LOWIC_INPUT_ROOT = input_root(plan)
        engine.STATION, engine.SITES = plan["station_code"], [plan["station_code"]]
        engine.BINARY_IRRIGATION_LEVELS, engine.BINARY_NITROGEN_LEVELS = LEVELS_I, LEVELS_N
        batch.CONFIG, batch.OUT, batch.DOC = effective_yaml, out, engine.BASE_DOC
        batch.SEED = int(plan["seed"])
        batch.SITE_NAMES[plan["station_code"]] = plan["site"]
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = input_root(plan)
        engine.run_training(2000, [1000, 2000], suffix="")
        summary_path = out / "evaluation" / "042_10_checkpoint_validation_summary.csv"
        summary = pd.read_csv(summary_path)
        final = summary[pd.to_numeric(summary["checkpoint_step"], errors="coerce").eq(2000)].copy()
        audit, gate = audit_rows(final, bool(plan.get("execution_policy", {}).get("require_crossyear_signature", False)))
        (out / "audits").mkdir(exist_ok=True)
        audit_path = out / "audits" / f"{plan['task_id']}_{candidate['id']}_validation_action_audit.csv"
        audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
        inventory = pd.read_csv(out / "evaluation" / "042_10_training_checkpoint_inventory.csv")
        model_row = inventory[pd.to_numeric(inventory["checkpoint_step"], errors="coerce").eq(2000)].iloc[0]
        model_path = ROOT / str(model_row["model_path"])
        observed = observed_kwargs(model_path)
        metrics = final[["final_grnwt", "total_irrigation", "total_n", "PFP_N"]].mean(numeric_only=True).to_dict()
        result = {
            **pf,
            "output_root": rel(out),
            "mode": "completed_smoke",
            "effective_yaml": rel(effective_yaml),
            "effective_kwargs_from_loaded_model": observed,
            "validation_metrics_mean": metrics,
            "WP_ET_status": "not_available_in_2K_training_validation_summary; not inferred",
            "action_gate": gate,
            "action_audit": rel(audit_path),
            "elapsed_s": round(time.perf_counter() - start, 3),
            "rss_delta_bytes": psutil.Process().memory_info().rss - rss0,
            "next_step_allowed": gate["technical_pass"],
        }
        (out / f"{plan['task_id']}_{candidate['id']}_smoke_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (out / f"{plan['task_id']}_{candidate['id']}_run_manifest.json").write_text(json.dumps({"status": "completed_smoke", "candidate": candidate, "preflight": pf, "effective_kwargs": pf["effective_ppo"], "observed": observed, "smoke_only": True}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return result
    finally:
        for key, value in old.items():
            setattr(engine, key, value)
        batch.CONFIG, batch.OUT, batch.DOC, batch.SEED = old_config, old_out, old_doc, old_seed
        batch.SITE_NAMES.clear()
        batch.SITE_NAMES.update(old_names)
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    plan = load_plan(config_path)
    candidate = next((item for item in plan["candidates"] if item["id"] == args.candidate), None)
    if candidate is None:
        raise ValueError(f"Unknown candidate {args.candidate}")
    print(json.dumps(run_candidate(plan, candidate, config_path, args.dry_run), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
