"""Audit seed0 E2 10K for exact 5K reproduction and collapse."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sb3_contrib import MaskablePPO
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"src"):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
import forecast_engineered_observation_056_057 as forecast, ppo_safe_rendering
from audit_111_112_sy_lc_rolling7d_counterfactual import action_map,collect_states,counterfactual_decisions
from run_141E1_sya_actionable_weather_encoding_smoke2k import patch_contract
from run_144E2_sya_dual_branch_weather_reader_10k_seed0 import CONFIG,PROMPT,gate
from audit_143E2_5k_response_and_noforecast import strict_action_audit as _unused
RUN=ROOT/"benchmark_results/144E2_sya_originIC_dual_branch_weather_reader_10k_seed0"
REF=ROOT/"benchmark_results/143E2_sya_originIC_dual_branch_weather_reader_5k"
OUT=ROOT/"benchmark_results/144E2_sya_dual_branch_weather_reader_10k_seed0_validation"
DOC=ROOT/"docs/2026-08-16_sya_E2_dual_branch_weather_reader_10k_seed0_validation.md"

def rel(p): return p.relative_to(ROOT).as_posix()
def model(step,root=RUN): return MaskablePPO.load(str(root/f"models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt{step}.zip"),device="cpu")
def exact_policy(a,b):
    sa,sb=a.policy.state_dict(),b.policy.state_dict(); same=list(sa)==list(sb) and all(torch.equal(sa[k],sb[k]) for k in sa)
    return {"exact":same,"max_abs_delta":max((float(torch.max(torch.abs(sa[k]-sb[k])).item()) for k in sa),default=0.0)}
def metrics(frame,step):
    g=frame[frame.checkpoint_step.eq(step)]
    return {"years":int(g.year.nunique()),"mean_yield":float(g.final_grnwt.mean()),"mean_irrigation":float(g.total_irrigation.mean()),"mean_n":float(g.total_n.mean()),"mean_pfp_n":float(g.PFP_N.mean()),"unique_sequences":int(g.action_sequence.nunique())}
def strict(frame,step):
    g=frame[frame.checkpoint_step.eq(step)]; pairs=set(); mismatch=off=after=0
    for _,r in g.iterrows():
        d=pd.read_csv(ROOT/r.daily_csv_path)
        for c in ["safe_action_amir","safe_action_anfer","season_cumulative_irrigation","season_cumulative_n","dap"]: d[c]=pd.to_numeric(d[c],errors="coerce").fillna(0.0)
        di=d.season_cumulative_irrigation.diff().fillna(d.season_cumulative_irrigation); dn=d.season_cumulative_n.diff().fillna(d.season_cumulative_n)
        mismatch+=int((~np.isclose(di,d.safe_action_amir)).sum()+(~np.isclose(dn,d.safe_action_anfer)).sum()); off+=int((~d.safe_action_amir.isin([0,15,30,45])).sum()+(~d.safe_action_anfer.isin([0,40,80,120])).sum())
        pos=d.safe_action_amir.gt(0)|d.safe_action_anfer.gt(0); after+=int((pos&d.dap.gt(1)).sum()); pairs.update(zip(d.loc[pos,"safe_action_amir"],d.loc[pos,"safe_action_anfer"]))
    return {"transmission_mismatch":mismatch,"off_grid":off,"after_dap1":after,"unique_nonzero_pairs":len(pairs),"pairs":[f"I{i:g}/N{n:g}" for i,n in sorted(pairs)],"passed":bool(len(g)==10 and mismatch==0 and off==0 and after>0 and len(pairs)>=2)}
def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True); patch_contract(); cfg=forecast.read_json(CONFIG); frame=pd.read_csv(RUN/"evaluation/144E2_checkpoint_validation_summary.csv")
    m5,m10=model(5000),model(10000); ref5=MaskablePPO.load(str(REF/"models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt5000.zip"),device="cpu")
    reproduction=exact_policy(ref5,m5); met5,met10=metrics(frame,5000),metrics(frame,10000); strict10=strict(frame,10000)
    old={k:getattr(forecast.engine,k) for k in ["LOWIC_INPUT_ROOT","STATION","SITES","BINARY_IRRIGATION_LEVELS","BINARY_NITROGEN_LEVELS"]}; oldn=dict(forecast.engine.base03222.SITE_NAMES); oldr=ppo_safe_rendering.MULTISITE_INPUT_ROOT; oldmake=forecast.engine.base03222.base.make_env
    try:
        forecast.engine.LOWIC_INPUT_ROOT=forecast.INPUT_PROFILES["originIC"]; forecast.engine.STATION="SYA"; forecast.engine.SITES=["SYA"]; forecast.engine.BINARY_IRRIGATION_LEVELS=[0,15,30,45]; forecast.engine.BINARY_NITROGEN_LEVELS=[0,40,80,120]; forecast.engine.base03222.SITE_NAMES["SYA"]="SY"; ppo_safe_rendering.MULTISITE_INPUT_ROOT=forecast.INPUT_PROFILES["originIC"]
        rc=forecast.engine.load_config(); sel=forecast.engine.base03222.build_selection(forecast.engine.base03222.load_split()); ec=forecast.direct_ppo.build_env_config(rc,sel); make=forecast.make_forecast_env_factory(oldmake,cfg); states=collect_states(m10,make,rc,ec,"SYA",0); states.to_json(OUT/"144E2_10k_states.json",orient="records",force_ascii=False); probe=make(rc,ec,"SYA",2014,0,"144E2_probe",evaluation=True)
        try: grid=action_map(probe)
        finally: probe.close()
        dec=counterfactual_decisions(states,m10,grid,0)
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT=oldr; forecast.engine.base03222.base.make_env=oldmake; forecast.engine.base03222.SITE_NAMES.clear(); forecast.engine.base03222.SITE_NAMES.update(oldn)
        for k,v in old.items(): setattr(forecast.engine,k,v)
    dec.to_csv(OUT/"144E2_10k_counterfactual.csv",index=False,encoding="utf-8-sig"); swap=dec.swap_changed; shuf=dec.shuffled_changed; response={"states":len(dec),"swap_rows":int(swap.sum()),"swap_rate":float(swap.mean()),"shuffled_rows":int(shuf.sum()),"shuffled_rate":float(shuf.mean()),"irrigation_changed":int((~np.isclose(dec.real_i,dec.swap_i)).sum()),"nitrogen_changed":int((~np.isclose(dec.real_n,dec.swap_n)).sum()),"passed_gt_1pct":bool(max(swap.mean(),shuf.mean())>0.01)}
    drop=(met10["mean_yield"]-met5["mean_yield"])/met5["mean_yield"]; collapse={"weather_response_collapsed":not response["passed_gt_1pct"],"action_sequence_collapsed":met10["unique_sequences"]<=1 or strict10["unique_nonzero_pairs"]<2,"yield_collapsed_gt_5pct":drop < -0.05,"yield_change_5k_to_10k_fraction":drop}; collapse["any_collapse"]=any(collapse[k] for k in ["weather_response_collapsed","action_sequence_collapsed","yield_collapsed_gt_5pct"])
    summary={"status":"completed_seed0_10k_collapse_audit","authorization":gate(),"five_k_exact_reproduction":reproduction,"checkpoint_metrics":{"5000":met5,"10000":met10},"strict_10k":strict10,"weather_response_10k":response,"collapse":collapse,"wp_et":"unavailable_without_valid_ETCP","proceed_to_independent_seeds":True}
    (OUT/"144E2_seed0_10k_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); DOC.write_text("\n".join(["# 144E2 seed0 10K坍缩审计","",f"- 5K精确复现：`{reproduction['exact']}`。",f"- 5K/10K平均产量：`{met5['mean_yield']:.2f}` / `{met10['mean_yield']:.2f}` kg/ha。",f"- 10K天气响应：`{response['swap_rate']:.4%}`；动作序列：`{met10['unique_sequences']}`；任一坍缩：`{collapse['any_collapse']}`。","","本结果仅为seed0；下一步按用户授权执行seed1/seed2至10K。WP_ET unavailable。",""]),encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
