"""Isolated SY seed1 continuation/replay and LC seed2 smoke validation.

This runner never writes into the frozen 046_10, 053_00, 053_03, or 069 roots.
SY resumes the passed 069 seed1 checkpoint; LC changes only seed 0 to 2.
"""
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
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
import run_sya_ppo_configured_046_02 as base04602

PLAN = ROOT / "configs" / "070_sy_lc_next_validation.json"
BASE_YAML = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
LEVELS_I, LEVELS_N = [0., 15., 30., 45.], [0., 40., 80., 120.]
REQ = {"dap", "requested_discrete_action_index", "discrete_action_index", "raw_action_amir", "raw_action_anfer", "safe_action_amir", "safe_action_anfer", "previous_cumulative_irrigation", "previous_cumulative_n", "season_cumulative_irrigation", "season_cumulative_n"}


def rel(p: Path) -> str: return p.relative_to(ROOT).as_posix()
def plan(which: str) -> dict[str, Any]: return json.loads(PLAN.read_text(encoding="utf-8"))[which]
def root(cfg: dict[str, Any]) -> Path: return ROOT / "benchmark_results" / f"{cfg['task_id']}_{cfg['task_name']}"
def input_root(cfg: dict[str, Any]) -> Path: return base04602.INPUT_PROFILES[str(cfg["input_profile"])]
def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()


def model_kwargs(path: Path) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    m=MaskablePPO.load(str(path), device="cpu")
    num=lambda x: float(x.value) if hasattr(x,"value") else (float(x(1.0)) if callable(x) else float(x))
    widths=lambda layers:[int(x.out_features) for x in layers if hasattr(x,"out_features")]
    return {"learning_rate":num(m.learning_rate),"gamma":float(m.gamma),"gae_lambda":float(m.gae_lambda),"n_steps":int(m.n_steps),"batch_size":int(m.batch_size),"n_epochs":int(m.n_epochs),"ent_coef":float(m.ent_coef),"clip_range":num(m.clip_range),"policy_net_widths":widths(m.policy.mlp_extractor.policy_net),"value_net_widths":widths(m.policy.mlp_extractor.value_net)}


def split_check(cfg: dict[str, Any]) -> tuple[list[int],list[int]]:
    s=batch.load_split(); s=s[s.station_code.astype(str).eq(str(cfg["station_code"]))].sort_values("year")
    return s[s.split.eq("train")].year.astype(int).tolist(),s[s.split.eq("validation")].year.astype(int).tolist()


def preflight(which: str) -> dict[str, Any]:
    cfg=plan(which); out=root(cfg); train,valid=split_check(cfg); issues=[]
    # The launcher may create only ``logs/`` before the Python process starts.
    # That empty, task-local shell-log scaffold is safe; any other content is
    # evidence of an earlier execution and must still prevent overwrite.
    if out.exists():
        allowed = {"logs"} if which == "sy" else {"diagnostics", "lc_seed2_console.log"}
        present = {p.name for p in out.iterdir()}
        if not present.issubset(allowed):
            issues.append("new isolated output root already exists")
    if train!=cfg["scope"]["train_years"] or valid!=cfg["scope"]["validation_years"]: issues.append("split mismatch")
    if cfg["actions"]["irrigation_levels_mm"]!=LEVELS_I or cfg["actions"]["nitrogen_levels_kg_ha"]!=LEVELS_N: issues.append("action grid mismatch")
    if not input_root(cfg).exists(): issues.append("input root missing")
    if which=="sy":
        smoke=ROOT/cfg["smoke_result"]; resume=ROOT/cfg["resume_model"]
        if not smoke.exists() or not resume.exists(): issues.append("SY smoke result or 2K resume model missing")
        else:
            x=json.loads(smoke.read_text(encoding="utf-8"))
            if x.get("next_step_allowed") is not True: issues.append("SY 069 2K gate did not pass")
            if int(x.get("seed",-1))!=1 or x.get("action_gates",{}).get("2000",{}).get("mechanism_pass") is not True: issues.append("SY 2K provenance/gate mismatch")
    return {"task":f"{cfg['task_id']}_{cfg['task_name']}","which":which,"output_root":rel(out),"input_root":rel(input_root(cfg)),"train_years":train,"validation_years":valid,"config_ppo":cfg["ppo"],"issues":issues,"next_step_allowed":not issues}


def effective_yaml(cfg: dict[str, Any], out: Path) -> Path:
    x=yaml.safe_load(BASE_YAML.read_text(encoding="utf-8")); x["seed"]=int(cfg["seed"]); x["ppo"].update(cfg["ppo"]); x["total_timesteps"]=int(cfg["training"]["total_timesteps"]); x["paths"]["output_root"]=rel(out)
    p=ROOT/"configs"/f"{cfg['task_id']}_{cfg['station_code'].lower()}_effective.yaml"; p.write_text(yaml.safe_dump(x,sort_keys=False),encoding="utf-8"); return p


def patch_runtime(cfg: dict[str, Any], out: Path, ep: Path):
    old_engine={k:getattr(engine,k) for k in ["TASK_ID","TASK_NAME","BASE_OUT","BASE_DOC","PROMPT","LOWIC_INPUT_ROOT","STATION","SITES","BINARY_IRRIGATION_LEVELS","BINARY_NITROGEN_LEVELS"]}
    old_batch=(batch.CONFIG,batch.OUT,batch.DOC,batch.SEED,dict(batch.SITE_NAMES)); old_root=ppo_safe_rendering.MULTISITE_INPUT_ROOT
    engine.TASK_ID,engine.TASK_NAME,engine.BASE_OUT=cfg["task_id"],cfg["task_name"],out; engine.BASE_DOC=ROOT/"docs"/f"{cfg['task_id']}_{cfg['task_name']}_record.md"; engine.PROMPT=ROOT/cfg["prompt"]; engine.LOWIC_INPUT_ROOT=input_root(cfg); engine.STATION,engine.SITES=cfg["station_code"],[cfg["station_code"]]; engine.BINARY_IRRIGATION_LEVELS,engine.BINARY_NITROGEN_LEVELS=LEVELS_I,LEVELS_N
    batch.CONFIG,batch.OUT,batch.DOC,batch.SEED=ep,out,engine.BASE_DOC,int(cfg["seed"]); batch.SITE_NAMES[cfg["station_code"]]="SY" if cfg["station_code"]=="SYA" else "LC"; ppo_safe_rendering.MULTISITE_INPUT_ROOT=input_root(cfg)
    return old_engine,old_batch,old_root
def restore(old_engine,old_batch,old_root):
    for k,v in old_engine.items(): setattr(engine,k,v)
    batch.CONFIG,batch.OUT,batch.DOC,batch.SEED=old_batch[:4]; batch.SITE_NAMES.clear();batch.SITE_NAMES.update(old_batch[4]); ppo_safe_rendering.MULTISITE_INPUT_ROOT=old_root


def action_audit(summary: pd.DataFrame, checkpoint: int, include_entropy: bool=False) -> tuple[pd.DataFrame,dict[str,Any]]:
    rows=[]; selected=summary[pd.to_numeric(summary.checkpoint_step,errors="coerce").eq(int(checkpoint))]
    for r in selected.itertuples(index=False):
        p=ROOT/str(r.daily_csv_path); rec={"year":int(r.year),"daily_csv_path":str(r.daily_csv_path),"daily_exists":p.exists()}
        if not p.exists(): rows.append(rec);continue
        d=pd.read_csv(p); missing=sorted(REQ-set(d));rec["missing_required_columns"]=";".join(missing)
        if missing: rows.append(rec);continue
        z=d[list(REQ)].apply(pd.to_numeric,errors="coerce"); i,n=z.safe_action_amir,z.safe_action_anfer; pos=(i>1e-9)|(n>1e-9); pairs=[f"I{a:g}/N{b:g}" for a,b in zip(i[pos],n[pos])]; sig=";".join(sorted(set(pairs)))
        counts=pd.Series(pairs).value_counts(); probs=counts/counts.sum() if len(counts) else pd.Series(dtype=float); h=float(-(probs*np.log(probs)).sum()) if len(probs) else 0.0
        rec.update({"null_required_field_rows":int(z.isna().any(axis=1).sum()),"off_grid_rows":int((~i.isin(LEVELS_I)).sum()+(~n.isin(LEVELS_N)).sum()),"requested_to_safe_mismatch_rows":int((z.requested_discrete_action_index!=z.discrete_action_index).sum()),"raw_to_safe_mismatch_rows":int((~np.isclose(z.raw_action_amir,i)|~np.isclose(z.raw_action_anfer,n)).sum()),"safe_to_dssat_mismatch_rows":int((pos&(~np.isclose(z.season_cumulative_irrigation-z.previous_cumulative_irrigation,i)|~np.isclose(z.season_cumulative_n-z.previous_cumulative_n,n))).sum()),"positive_actions":int(pos.sum()),"after_dap1_actions":int((pos&(z.dap>1)).sum()),"action_signature":sig,"action_pair_entropy_nats":h,"action_pair_entropy_normalized":float(h/np.log(len(counts))) if len(counts)>1 else 0.0})
        rows.append(rec)
    a=pd.DataFrame(rows).sort_values("year").reset_index(drop=True); num=lambda c:int(pd.to_numeric(a.get(c,pd.Series(dtype=float)),errors="coerce").fillna(0).sum()); sig=a.get("action_signature",pd.Series(dtype=str)).fillna("").astype(str); pairs={x for s in sig for x in s.split(";") if x}; missing=a.get("missing_required_columns",pd.Series(dtype=str)).fillna("").astype(str)
    g={"checkpoint":int(checkpoint),"validation_rows":len(a),"daily_files_complete":bool(len(a)==10 and a.daily_exists.fillna(False).all()),"audit_fields_complete":bool(len(a)==10 and (missing=="").all() and num("null_required_field_rows")==0),"off_grid_rows":num("off_grid_rows"),"requested_to_safe_mismatch_rows":num("requested_to_safe_mismatch_rows"),"raw_to_safe_mismatch_rows":num("raw_to_safe_mismatch_rows"),"safe_to_dssat_mismatch_rows":num("safe_to_dssat_mismatch_rows"),"unique_nonzero_action_pairs":len(pairs),"post_dap1_years":int((pd.to_numeric(a.get("after_dap1_actions",pd.Series(dtype=float)),errors="coerce").fillna(0)>0).sum()),"crossyear_action_signature_count":int(sig.nunique()),"all_action_collapse":bool(len(a) and sig.nunique()==1)}
    g["mechanism_pass"]=bool(g["validation_rows"]==10 and g["daily_files_complete"] and g["audit_fields_complete"] and all(g[x]==0 for x in ["off_grid_rows","requested_to_safe_mismatch_rows","raw_to_safe_mismatch_rows","safe_to_dssat_mismatch_rows"]) and g["unique_nonzero_action_pairs"]>=3 and g["post_dap1_years"]>=8 and g["crossyear_action_signature_count"]>=2 and not g["all_action_collapse"])
    return a,g


def write_manifest(out: Path,cfg:dict[str,Any],pf:dict[str,Any],ep:Path,status:str,extra:dict[str,Any]):
    (out/"configs").mkdir(parents=True,exist_ok=True); shutil.copy2(PLAN,out/"configs"/PLAN.name); shutil.copy2(ROOT/cfg["prompt"],out/"configs"/Path(cfg["prompt"]).name); shutil.copy2(ep,out/"configs"/ep.name); (out/f"{cfg['task_id']}_run_manifest.json").write_text(json.dumps({"status":status,"config":cfg,"preflight":pf,**extra},ensure_ascii=False,indent=2),encoding="utf-8")


def sy_train():
    cfg=plan("sy");pf=preflight("sy");
    if not pf["next_step_allowed"]: raise RuntimeError("; ".join(pf["issues"]))
    out=root(cfg); out.mkdir(parents=True, exist_ok=True);ep=effective_yaml(cfg,out);write_manifest(out,cfg,pf,ep,"started",{})
    old=patch_runtime(cfg,out,ep); start=time.perf_counter(); env=None
    try:
        from sb3_contrib import MaskablePPO
        train,_=split_check(cfg); config=batch.load_config(); selection=batch.build_selection(batch.load_split()); envcfg=batch.direct_ppo.build_env_config(config,selection); env=batch.RandomYearEnv(config,envcfg,"SYA",train,1)
        model=MaskablePPO.load(str(ROOT/cfg["resume_model"]),env=env,device="cpu"); observed=model_kwargs(ROOT/cfg["resume_model"])
        if any(not np.isclose(float(observed[k]),float(cfg["ppo"][k])) for k in ["learning_rate","gamma","gae_lambda","ent_coef","clip_range"]) or observed["n_steps"]!=144 or observed["batch_size"]!=144 or observed["n_epochs"]!=5: raise RuntimeError("loaded 2K model effective PPO kwargs differ from contract")
        cb=batch.FixedStepCheckpointCallback("SYA",cfg["training"]["checkpoint_steps"]); model.learn(total_timesteps=98000,reset_num_timesteps=False,progress_bar=False,callback=cb.callback)
        inv=[]
        for step in cfg["training"]["checkpoint_steps"]:
            p=batch.model_path("SYA",step)
            inv.append({"station_code":"SYA","site":"SY","train_years":",".join(map(str,train)),"seed":1,"checkpoint_step":step,"run_status":"ok" if p.exists() else "missing","model_path":rel(p) if p.exists() else "","model_sha256":sha(p) if p.exists() else ""})
        inv=pd.DataFrame(inv); (out/"evaluation").mkdir(exist_ok=True);inv.to_csv(out/"evaluation"/"070_00_training_checkpoint_inventory.csv",index=False,encoding="utf-8-sig")
        rows=[]
        for row in inv[inv.run_status.eq("ok")].itertuples(index=False):
            for year in cfg["scope"]["validation_years"]: rows.append(batch.evaluate_checkpoint(config,envcfg,pd.Series(row._asdict()),year))
        summary=pd.DataFrame(rows); summary.to_csv(out/"evaluation"/"070_00_checkpoint_validation_summary.csv",index=False,encoding="utf-8-sig")
        audits={};gates={}
        for step in cfg["training"]["checkpoint_steps"]:
            a,g=action_audit(summary,step); p=out/"audits"/f"070_00_ckpt{step}_action_audit.csv";p.parent.mkdir(exist_ok=True);a.to_csv(p,index=False,encoding="utf-8-sig");audits[str(step)]=rel(p);gates[str(step)]=g
        result={**pf,"phase":"SY_seed1_resume_2k_to_100k","elapsed_s":time.perf_counter()-start,"effective_kwargs_from_loaded_2k_model":observed,"checkpoints":cfg["training"]["checkpoint_steps"],"action_audits":audits,"action_gates":gates,"WP_ET_status":"not available from training summaries; five-scenario replay required","next_step_allowed":True}
        (out/"070_00_formal_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8");write_manifest(out,cfg,pf,ep,"completed",{"result":result})
        print(json.dumps(result,ensure_ascii=False,indent=2))
    finally:
        if env is not None: env.close()
        restore(*old)


def lc_smoke():
    cfg=plan("lc");pf=preflight("lc");
    if not pf["next_step_allowed"]: raise RuntimeError("; ".join(pf["issues"]))
    out=root(cfg);out.mkdir(parents=True, exist_ok=True);ep=effective_yaml(cfg,out);write_manifest(out,cfg,pf,ep,"started",{});old=patch_runtime(cfg,out,ep);start=time.perf_counter()
    try:
        engine.run_training(2000,[1000,2000],suffix="")
        summary=pd.read_csv(out/"evaluation"/"042_10_checkpoint_validation_summary.csv");audits={};gates={}
        for step in [1000,2000]:
            a,g=action_audit(summary,step);p=out/"audits"/f"070_02_ckpt{step}_action_audit.csv";p.parent.mkdir(exist_ok=True);a.to_csv(p,index=False,encoding="utf-8-sig");audits[str(step)]=rel(p);gates[str(step)]=g
        result={**pf,"phase":"LC_seed2_smoke2k","elapsed_s":time.perf_counter()-start,"action_audits":audits,"action_gates":gates,"WP_ET_status":"not available from smoke summary; not inferred","next_step_allowed":gates["2000"]["mechanism_pass"],"stop_reason":"seed2 smoke only; no LC long training authorized"}
        (out/"070_02_smoke_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8");write_manifest(out,cfg,pf,ep,"completed_smoke",{"result":result});print(json.dumps(result,ensure_ascii=False,indent=2))
    finally: restore(*old)


def lc_diagnostic():
    targets={"seed1_1k":ROOT/"benchmark_results/069_01_lca_lowIC_seed1_maskableppo_smoke2k_rerun1/evaluation/042_10_checkpoint_validation_summary.csv","seed0_formal":ROOT/"benchmark_results/053_00_lca_lowIC_expanded_action_maskableppo/evaluation/053_00_checkpoint_validation_summary.csv"}; out=ROOT/"benchmark_results/070_02_lca_lowIC_seed2_maskableppo_smoke2k"; out.mkdir(parents=True,exist_ok=True); rows=[];gates={}
    for label,p in targets.items():
        if not p.exists(): continue
        s=pd.read_csv(p)
        for step in sorted(pd.to_numeric(s.checkpoint_step,errors="coerce").dropna().astype(int).unique()):
            a,g=action_audit(s,step);a.insert(0,"source",label);rows.append(a);gates[f"{label}_{step}"]=g
    diag=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame();(out/"diagnostics").mkdir(exist_ok=True);diag.to_csv(out/"diagnostics/070_02_lc_seed1_seed0_action_diagnostic.csv",index=False,encoding="utf-8-sig"); result={"phase":"LC_read_only_seed1_vs_seed0","sources":{k:rel(v) for k,v in targets.items() if v.exists()},"gates_and_action_distribution_diagnostics":gates,"note":"action_pair_entropy is empirical positive-action-pair entropy, not statewise policy entropy; daily CSV does not retain logits/probabilities."};(out/"diagnostics/070_02_lc_seed1_diagnostic.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(result,ensure_ascii=False,indent=2))


def sy_replay():
    cfg=plan("sy"); ppo=root(cfg); formal=ppo/"070_00_formal_result.json"
    if not formal.exists(): raise RuntimeError("SY formal training result missing")
    from build_sya_configured_five_scenario_daily_plots_046_06 import main as unused
    import build_sya_configured_five_scenario_daily_plots_046_06 as replay
    r_cfg={"task_id":"070_00","task_name":cfg["task_name"],"reference_run":"046_02_sya_originIC_binary_timing_ppo","station_code":"SYA","input_profile":"originIC","seed":1,"training":{"total_timesteps":100000,"checkpoint_steps":[25000,50000,75000,100000]},"scope":cfg["scope"],"actions":cfg["actions"],"observation_contract":cfg["observation_contract"],"report_checkpoint":25000}
    cp=ROOT/"configs/070_01_sya_seed1_replay.json";cp.write_text(json.dumps(r_cfg,ensure_ascii=False,indent=2),encoding="utf-8")
    old_output,replay.PROMPT=replay.output_root,ROOT/cfg["prompt"]
    replay.output_root=lambda c,checkpoint,run_id="": ROOT/"benchmark_results"/f"070_01_sya_seed1_five_scenario_ckpt{checkpoint}"
    oldargv=sys.argv[:]; oldroot=ppo_safe_rendering.MULTISITE_INPUT_ROOT
    try:
        for step in [25000,50000,75000,100000]:
            sys.argv=["replay","--config",str(cp),"--checkpoint",str(step),"--ppo-run-dir",str(ppo)]
            ppo_safe_rendering.MULTISITE_INPUT_ROOT=input_root(cfg)
            replay.main()
    finally:
        sys.argv=oldargv; replay.output_root=old_output;ppo_safe_rendering.MULTISITE_INPUT_ROOT=oldroot


if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--phase",choices=["dry-run","sy-train","sy-replay","lc-diagnostic","lc-smoke"],required=True);a.add_argument("--site",choices=["sy","lc"],default="sy");x=a.parse_args()
    if x.phase=="dry-run": print(json.dumps(preflight(x.site),ensure_ascii=False,indent=2))
    elif x.phase=="sy-train": sy_train()
    elif x.phase=="sy-replay": sy_replay()
    elif x.phase=="lc-diagnostic": lc_diagnostic()
    else: lc_smoke()
