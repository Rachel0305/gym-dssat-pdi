"""Isolated seed=1 stability validation for HL baseline and YC lr=1e-4.

This runner intentionally has no 100K mode.  It creates fresh output roots and
uses the repaired lowIC renderer only for post-training full-metric replays.
"""
from __future__ import annotations

import argparse
import importlib.util
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

PLAN_PATH = ROOT / "configs" / "068_hl_yc_seed1_stability_validation.json"
BASE_YAML = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
LEVELS_I, LEVELS_N = [0.0, 15.0, 30.0, 45.0], [0.0, 40.0, 80.0, 120.0]
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


def experiment(plan: dict[str, Any], site_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if site_key not in plan["experiments"]:
        raise ValueError(f"unknown site key {site_key}")
    return plan["common"], plan["experiments"][site_key]


def cfg_for(common: dict[str, Any], spec: dict[str, Any], phase: str) -> dict[str, Any]:
    training = common["smoke"] if phase == "smoke" else common["validation_25k"]
    name = spec["task_name"] + ("_smoke2k" if phase == "smoke" else "")
    return {
        "task_id": spec["task_id"], "task_name": name, "reference_run": spec["reference_run"],
        "station_code": spec["station_code"], "site": spec["site"], "input_profile": common["input_profile"],
        "seed": common["seed"], "training": training, "actions": common["actions"],
        "observation_contract": common["observation_contract"], "scope": common["scope"],
        "ppo": spec["ppo"], "prompt": spec["prompt"],
    }


def out_root(cfg: dict[str, Any]) -> Path:
    return ROOT / "benchmark_results" / f"{cfg['task_id']}_{cfg['task_name']}"


def source_mzx(cfg: dict[str, Any]) -> Path:
    return INPUT_ROOT / ("HL/CNHL0701_corrected_IC123.MZX" if cfg["station_code"] == "HLA" else "YC/CNYC0801.MZX")


def preflight(cfg: dict[str, Any]) -> dict[str, Any]:
    split = batch.load_split()
    local = split[split.station_code.eq(cfg["station_code"])].sort_values("year")
    train = local[local.split.eq("train")].year.astype(int).tolist()
    valid = local[local.split.eq("validation")].year.astype(int).tolist()
    issues: list[str] = []
    if not INPUT_ROOT.exists() or not source_mzx(cfg).exists():
        issues.append("lowIC input root or station MZX missing")
    if train != cfg["scope"]["train_years"] or valid != cfg["scope"]["validation_years"]:
        issues.append("configured train/validation years mismatch engine split")
    if cfg["actions"]["irrigation_levels_mm"] != LEVELS_I or cfg["actions"]["nitrogen_levels_kg_ha"] != LEVELS_N:
        issues.append("declared action grid mismatch")
    if cfg["observation_contract"] != {"base": "046_02_raw_observation", "normalization_enabled": False, "weather_forecast_enabled": False}:
        issues.append("raw-observation/no-forecast contract mismatch")
    if int(cfg["seed"]) != 1:
        issues.append("this runner is registered only for seed=1")
    if not (ROOT / cfg["prompt"]).exists():
        issues.append("registered prompt missing")
    return {
        "task": f"{cfg['task_id']}_{cfg['task_name']}", "station_code": cfg["station_code"], "site": cfg["site"],
        "seed": cfg["seed"], "input_root": rel(INPUT_ROOT), "source_mzx": rel(source_mzx(cfg)),
        "train_years": train, "validation_years": valid, "action_grid": cfg["actions"],
        "observation_contract": cfg["observation_contract"], "effective_ppo": cfg["ppo"],
        "output_root": rel(out_root(cfg)), "issues": issues, "next_step_allowed": not issues,
    }


def observed_kwargs(model_path: Path) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(str(model_path), device="cpu")
    widths = lambda seq: [int(x.out_features) for x in seq if hasattr(x, "out_features")]
    e = model.policy.mlp_extractor
    return {"learning_rate": model.learning_rate, "gamma": model.gamma, "gae_lambda": model.gae_lambda,
            "n_steps": model.n_steps, "batch_size": model.batch_size, "n_epochs": model.n_epochs,
            "ent_coef": model.ent_coef, "clip_range": model.clip_range,
            "policy_net_widths": widths(e.policy_net), "value_net_widths": widths(e.value_net)}


def audit_checkpoint(summary: pd.DataFrame, checkpoint: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    final = summary[pd.to_numeric(summary["checkpoint_step"], errors="coerce").eq(int(checkpoint))].copy()
    rows: list[dict[str, Any]] = []
    for record in final.itertuples(index=False):
        path = ROOT / str(record.daily_csv_path)
        row: dict[str, Any] = {"year": int(record.year), "daily_csv_path": str(record.daily_csv_path), "daily_exists": path.exists()}
        if not path.exists():
            row["missing_required_columns"] = "daily_file_missing"
            rows.append(row); continue
        daily = pd.read_csv(path)
        missing = sorted(REQUIRED_DAILY_COLUMNS - set(daily.columns))
        row["missing_required_columns"] = ";".join(missing)
        if missing:
            rows.append(row); continue
        req = pd.to_numeric(daily.requested_discrete_action_index, errors="coerce")
        actual = pd.to_numeric(daily.discrete_action_index, errors="coerce")
        raw_i, raw_n = pd.to_numeric(daily.raw_action_amir, errors="coerce"), pd.to_numeric(daily.raw_action_anfer, errors="coerce")
        safe_i, safe_n = pd.to_numeric(daily.safe_action_amir, errors="coerce"), pd.to_numeric(daily.safe_action_anfer, errors="coerce")
        prev_i, prev_n = pd.to_numeric(daily.previous_cumulative_irrigation, errors="coerce"), pd.to_numeric(daily.previous_cumulative_n, errors="coerce")
        cum_i, cum_n = pd.to_numeric(daily.season_cumulative_irrigation, errors="coerce"), pd.to_numeric(daily.season_cumulative_n, errors="coerce")
        positive = (safe_i > 1e-9) | (safe_n > 1e-9)
        transmitted = (~positive) | (np.isclose(cum_i-prev_i, safe_i, atol=1e-6) & np.isclose(cum_n-prev_n, safe_n, atol=1e-6))
        pairs = sorted({f"I{float(i):g}/N{float(n):g}" for i,n in zip(safe_i[positive],safe_n[positive])})
        row.update({
            "off_grid_rows": int((~safe_i.isin(LEVELS_I)).sum() + (~safe_n.isin(LEVELS_N)).sum()),
            "requested_to_safe_mismatch_rows": int((req != actual).sum()),
            "raw_to_safe_mismatch_rows": int((~np.isclose(raw_i,safe_i,atol=1e-6) | ~np.isclose(raw_n,safe_n,atol=1e-6)).sum()),
            "safe_to_dssat_mismatch_rows": int((positive & ~transmitted).sum()),
            "positive_actions": int(positive.sum()),
            "after_dap1_actions": int((positive & (pd.to_numeric(daily.dap,errors="coerce") > 1)).sum()),
            "action_pairs": ";".join(pairs),
        })
        rows.append(row)
    audit = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    pairs = sorted({x for text in audit.get("action_pairs", pd.Series(dtype=str)).fillna("") for x in str(text).split(";") if x})
    def total(name: str) -> int:
        return int(pd.to_numeric(audit.get(name, pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    gaps = audit.get("missing_required_columns", pd.Series(dtype=str)).fillna("").astype(str)
    gate = {
        "checkpoint": int(checkpoint), "validation_years_expected": 10, "validation_rows": int(len(audit)),
        "daily_files_complete": bool(len(audit) == 10 and audit.daily_exists.fillna(False).all()),
        "audit_fields_complete": bool(len(audit) == 10 and (gaps == "").all()),
        "off_grid_rows": total("off_grid_rows"), "requested_to_safe_mismatch_rows": total("requested_to_safe_mismatch_rows"),
        "raw_to_safe_mismatch_rows": total("raw_to_safe_mismatch_rows"), "safe_to_dssat_mismatch_rows": total("safe_to_dssat_mismatch_rows"),
        "unique_nonzero_action_pairs": len(pairs), "post_dap1_years": int((pd.to_numeric(audit.get("after_dap1_actions", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum()),
        "crossyear_action_signature_count": int(audit.get("action_pairs", pd.Series(dtype=str)).fillna("").nunique()),
        "all_action_collapse": bool(len(audit) and audit.get("action_pairs", pd.Series(dtype=str)).fillna("").nunique() == 1),
    }
    gate["mechanism_pass"] = bool(gate["validation_rows"] == 10 and gate["daily_files_complete"] and gate["audit_fields_complete"] and gate["off_grid_rows"] == 0 and gate["requested_to_safe_mismatch_rows"] == 0 and gate["raw_to_safe_mismatch_rows"] == 0 and gate["safe_to_dssat_mismatch_rows"] == 0 and gate["unique_nonzero_action_pairs"] >= 3 and gate["post_dap1_years"] >= 8 and gate["crossyear_action_signature_count"] >= 2 and not gate["all_action_collapse"])
    return audit, gate


def phase_run(cfg: dict[str, Any], phase: str) -> dict[str, Any]:
    pf = preflight(cfg); out = out_root(cfg)
    if not pf["next_step_allowed"]: raise RuntimeError("; ".join(pf["issues"]))
    if phase == "validate25k":
        smoke_cfg = dict(cfg); smoke_cfg["task_name"] = cfg["task_name"] + "_smoke2k" if not cfg["task_name"].endswith("_smoke2k") else cfg["task_name"]
        smoke_result = out_root(smoke_cfg) / f"{cfg['task_id']}_smoke_result.json"
        if not smoke_result.exists() or json.loads(smoke_result.read_text(encoding="utf-8")).get("next_step_allowed") is not True:
            raise RuntimeError("registered 2K smoke gate did not pass; 25K is forbidden")
    if out.exists(): raise FileExistsError(f"refusing to overwrite {rel(out)}")
    out.mkdir(parents=True); (out / "configs").mkdir()
    shutil.copy2(PLAN_PATH, out / "configs" / PLAN_PATH.name)
    shutil.copy2(ROOT / cfg["prompt"], out / "configs" / Path(cfg["prompt"]).name)
    effective_yaml = ROOT / "configs" / f"068_effective_{cfg['station_code'].lower()}_seed1_{phase}.yaml"
    yaml_cfg = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8")); yaml_cfg["ppo"].update(cfg["ppo"]); yaml_cfg["seed"] = int(cfg["seed"])
    effective_yaml.write_text(yaml.safe_dump(yaml_cfg, sort_keys=False), encoding="utf-8")
    shutil.copy2(effective_yaml, out / "configs" / effective_yaml.name)
    old_engine = {key:getattr(engine,key) for key in ["TASK_ID","TASK_NAME","BASE_OUT","BASE_DOC","PROMPT","LOWIC_INPUT_ROOT","STATION","SITES","BINARY_IRRIGATION_LEVELS","BINARY_NITROGEN_LEVELS"]}
    old_batch = batch.CONFIG, batch.OUT, batch.DOC, batch.SEED, dict(batch.SITE_NAMES)
    old_renderer = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    start=time.perf_counter(); rss0=psutil.Process().memory_info().rss
    try:
        engine.TASK_ID,engine.TASK_NAME,engine.BASE_OUT=cfg["task_id"],cfg["task_name"],out
        engine.BASE_DOC=ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"; engine.PROMPT=ROOT / cfg["prompt"]
        engine.LOWIC_INPUT_ROOT=INPUT_ROOT; engine.STATION=cfg["station_code"]; engine.SITES=[cfg["station_code"]]
        engine.BINARY_IRRIGATION_LEVELS,engine.BINARY_NITROGEN_LEVELS=LEVELS_I,LEVELS_N
        batch.CONFIG,batch.OUT,batch.DOC,batch.SEED=effective_yaml,out,engine.BASE_DOC,int(cfg["seed"])
        batch.SITE_NAMES[cfg["station_code"]]=cfg["site"]; ppo_safe_rendering.MULTISITE_INPUT_ROOT=INPUT_ROOT
        engine.run_training(int(cfg["training"]["total_timesteps"]), list(map(int,cfg["training"]["checkpoint_steps"])), suffix="")
    finally:
        for key,value in old_engine.items(): setattr(engine,key,value)
        batch.CONFIG,batch.OUT,batch.DOC,batch.SEED=old_batch[:4]; batch.SITE_NAMES.clear(); batch.SITE_NAMES.update(old_batch[4]); ppo_safe_rendering.MULTISITE_INPUT_ROOT=old_renderer
    summary = pd.read_csv(out / "evaluation" / "042_10_checkpoint_validation_summary.csv")
    inventory = pd.read_csv(out / "evaluation" / "042_10_training_checkpoint_inventory.csv")
    audits, gates = {}, {}
    for checkpoint in cfg["training"]["checkpoint_steps"]:
        audit,gate=audit_checkpoint(summary,int(checkpoint)); audit_path=out / "audits" / f"{cfg['task_id']}_ckpt{checkpoint}_action_audit.csv"; audit_path.parent.mkdir(exist_ok=True); audit.to_csv(audit_path,index=False,encoding="utf-8-sig")
        audits[str(checkpoint)]=rel(audit_path); gates[str(checkpoint)]=gate
    final_ckpt=max(cfg["training"]["checkpoint_steps"]); model_rel=inventory[pd.to_numeric(inventory.checkpoint_step,errors="coerce").eq(final_ckpt)].iloc[0].model_path
    result={**pf,"phase":phase,"elapsed_s":time.perf_counter()-start,"rss_delta_bytes":psutil.Process().memory_info().rss-rss0,"effective_yaml":rel(effective_yaml),"effective_kwargs_from_loaded_model":observed_kwargs(ROOT / str(model_rel)),"action_audits":audits,"action_gates":gates,"WP_ET_status":"not_available_from_training summaries; not inferred"}
    result["next_step_allowed"] = gates[str(final_ckpt)]["mechanism_pass"] if phase == "smoke" else False
    result["stop_reason"] = "smoke gate failed" if phase == "smoke" and not result["next_step_allowed"] else ("25K completed; only replay/independent validation decision is allowed" if phase != "smoke" else "smoke gate passed")
    (out / f"{cfg['task_id']}_{phase}_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    (out / f"{cfg['task_id']}_run_manifest.json").write_text(json.dumps({"status":f"completed_{phase}","config":cfg,"preflight":pf,"result":result},ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--site",choices=["HL","YC"],required=True); parser.add_argument("--phase",choices=["dry-run","smoke","validate25k"],required=True); args=parser.parse_args()
    plan=load_plan(); common,spec=experiment(plan,args.site); phase="smoke" if args.phase in {"dry-run","smoke"} else "validate25k"; cfg=cfg_for(common,spec,phase)
    if args.phase=="dry-run": print(json.dumps(preflight(cfg),ensure_ascii=False,indent=2)); return
    print(json.dumps(phase_run(cfg,phase),ensure_ascii=False,indent=2,default=str))

if __name__=="__main__": main()
