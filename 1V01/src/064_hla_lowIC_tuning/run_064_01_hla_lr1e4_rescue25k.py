"""Isolated HL 25K rescue for the pre-registered lr=1e-4 candidate only."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")) if str(ROOT / "src") not in sys.path else None
import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine

HELPER_PATH = ROOT / "src" / "064_hla_lowIC_tuning" / "run_064_hla_ppo_kwargs_smoke.py"
spec = importlib.util.spec_from_file_location("r064", HELPER_PATH)
r064 = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(r064)
CONFIG = ROOT / "configs" / "064_01_hla_lowIC_lr1e4_rescue25k.json"
BASE_YAML = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"


def rel(p: Path) -> str: return p.relative_to(ROOT).as_posix()


def preflight(cfg: dict[str, Any]) -> dict[str, Any]:
    issues=[]
    expected={"learning_rate":1e-4,"gamma":1.0,"gae_lambda":1.0,"n_steps":144,"batch_size":144,"n_epochs":5,"ent_coef":0.01,"clip_range":0.2,"net_arch":[64,64]}
    if cfg.get("effective_ppo") != expected: issues.append("effective PPO kwargs differ from the pre-registered lr1e-4 rescue")
    if cfg.get("station_code") != "HLA" or cfg.get("site") != "HLA": issues.append("station/site mismatch")
    if cfg["actions"]["irrigation_levels_mm"] != r064.LEVELS_I or cfg["actions"]["nitrogen_levels_kg_ha"] != r064.LEVELS_N: issues.append("action grid mismatch")
    if not r064.INPUT_ROOT.exists() or not (r064.INPUT_ROOT/"HL"/"CNHL0701_corrected_IC123.MZX").exists(): issues.append("lowIC root/MZX missing")
    smoke = ROOT/"benchmark_results"/cfg["reference_smoke"] / "064_00_smoke_result.json"
    if not smoke.exists() or json.loads(smoke.read_text(encoding="utf-8")).get("next_step_allowed") is not True: issues.append("B_lr1e4 2K gate not found/passed")
    split=batch.load_split(); split=split[split.station_code.eq("HLA")]
    if split[split.split.eq("train")].year.astype(int).tolist()!=cfg["scope"]["train_years"]: issues.append("train years mismatch")
    if split[split.split.eq("validation")].year.astype(int).tolist()!=cfg["scope"]["validation_years"]: issues.append("validation years mismatch")
    return {"effective_ppo":cfg["effective_ppo"],"input_root":rel(r064.INPUT_ROOT),"source_mzx":rel(r064.INPUT_ROOT/"HL"/"CNHL0701_corrected_IC123.MZX"),"issues":issues,"next_step_allowed":not issues}


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--dry-run",action="store_true"); args=ap.parse_args()
    cfg=json.loads(CONFIG.read_text(encoding="utf-8")); pf=preflight(cfg)
    out=ROOT/"benchmark_results"/f"{cfg['task_id']}_{cfg['task_name']}"
    if args.dry_run:
        print(json.dumps({**pf,"output_root":rel(out),"checkpoints":cfg["training"]["checkpoint_steps"],"mode":"dry_run"},ensure_ascii=False,indent=2)); return
    if not pf["next_step_allowed"]: raise RuntimeError("; ".join(pf["issues"]))
    if out.exists(): raise FileExistsError(f"refusing to overwrite {rel(out)}")
    (out/"configs").mkdir(parents=True)
    effective=ROOT/"configs"/"064_01_effective_lr1e4_ppo_config.yaml"
    yaml_cfg=yaml.safe_load(BASE_YAML.read_text(encoding="utf-8")); yaml_cfg["ppo"].update(cfg["effective_ppo"])
    effective.write_text(yaml.safe_dump(yaml_cfg,sort_keys=False),encoding="utf-8")
    shutil.copy2(CONFIG,out/"configs"/CONFIG.name)
    old={k:getattr(engine,k) for k in ["TASK_ID","TASK_NAME","BASE_OUT","BASE_DOC","LOWIC_INPUT_ROOT","STATION","SITES","BINARY_IRRIGATION_LEVELS","BINARY_NITROGEN_LEVELS"]}
    old_config,old_out,old_doc=batch.CONFIG,batch.OUT,batch.DOC; old_root=ppo_safe_rendering.MULTISITE_INPUT_ROOT; old_names=dict(batch.SITE_NAMES)
    try:
        engine.TASK_ID,engine.TASK_NAME,engine.BASE_OUT=cfg["task_id"],cfg["task_name"],out
        engine.BASE_DOC,engine.LOWIC_INPUT_ROOT=ROOT/"docs"/f"{cfg['task_id']}_{cfg['task_name']}_record.md",r064.INPUT_ROOT
        engine.STATION,engine.SITES="HLA",["HLA"]; engine.BINARY_IRRIGATION_LEVELS,engine.BINARY_NITROGEN_LEVELS=r064.LEVELS_I,r064.LEVELS_N
        batch.CONFIG,batch.OUT,batch.DOC=effective,out,engine.BASE_DOC; batch.SITE_NAMES["HLA"]="HLA"; ppo_safe_rendering.MULTISITE_INPUT_ROOT=r064.INPUT_ROOT
        engine.run_training(int(cfg["training"]["total_timesteps"]),list(map(int,cfg["training"]["checkpoint_steps"])),suffix="")
        summary=pd.read_csv(out/"evaluation"/"042_10_checkpoint_validation_summary.csv")
        summary.to_csv(out/"evaluation"/"064_01_checkpoint_validation_summary.csv",index=False,encoding="utf-8-sig")
        inv=pd.read_csv(out/"evaluation"/"042_10_training_checkpoint_inventory.csv")
        checkpoint_results=[]; (out/"audits").mkdir(exist_ok=True)
        for step in cfg["training"]["checkpoint_steps"]:
            final=summary[pd.to_numeric(summary.checkpoint_step).eq(int(step))].copy(); audit,gate=r064.audit_rows(final)
            audit_path=out/"audits"/f"064_01_ckpt{step}_validation_action_audit.csv"; audit.to_csv(audit_path,index=False,encoding="utf-8-sig")
            model_path=ROOT/str(inv[pd.to_numeric(inv.checkpoint_step).eq(int(step))].iloc[0].model_path)
            checkpoint_results.append({"checkpoint_step":int(step),"validation_metrics_mean":final[["final_grnwt","total_irrigation","total_n","PFP_N"]].mean(numeric_only=True).to_dict(),"WP_ET_status":"not_available_in_25K_training_validation_summary; not inferred","action_gate":gate,"action_audit":rel(audit_path),"effective_kwargs_from_loaded_model":r064.observed_kwargs(model_path)})
        result={**pf,"mode":"completed_rescue25k","output_root":rel(out),"effective_yaml":rel(effective),"checkpoint_results":checkpoint_results,"training_year_action_diversity":"not_exported_by_inherited_engine; not estimated","rescue_mechanism_pass_at_25k":checkpoint_results[-1]["action_gate"]["mechanism_pass"],"next_step_allowed":False}
        (out/"064_01_rescue_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
        (out/"064_01_run_manifest.json").write_text(json.dumps({"status":"completed_rescue25k","config":cfg,"preflight":pf,"effective_kwargs":cfg["effective_ppo"]},ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    finally:
        for k,v in old.items(): setattr(engine,k,v)
        batch.CONFIG,batch.OUT,batch.DOC=old_config,old_out,old_doc; batch.SITE_NAMES.clear(); batch.SITE_NAMES.update(old_names); ppo_safe_rendering.MULTISITE_INPUT_ROOT=old_root

if __name__=="__main__": main()
