"""Run independent E2 seeds 1 and 2 to 10K, one process at a time."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"src"):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
import forecast_engineered_observation_056_057 as forecast
from run_141E1_sya_actionable_weather_encoding_smoke2k import patch_contract
from run_142E2_sya_dual_branch_weather_reader_smoke2k import patched_maskableppo
CONFIGS={1:ROOT/"configs/145E2S1_sya_originIC_dual_branch_weather_reader_10k_seed1.json",2:ROOT/"configs/146E2S2_sya_originIC_dual_branch_weather_reader_10k_seed2.json"}
IDS={1:"145E2S1",2:"146E2S2"}; PROMPT=ROOT/"prompts/2026-08-16_sya_E2_dual_branch_weather_reader_10k_independent_seeds.md"; SEED0=ROOT/"benchmark_results/144E2_sya_dual_branch_weather_reader_10k_seed0_validation/144E2_seed0_10k_summary.json"
def gate(seed):
    cfg=json.loads(CONFIGS[seed].read_text(encoding="utf-8")); base=json.loads((ROOT/"configs/144E2_sya_originIC_dual_branch_weather_reader_10k_seed0.json").read_text(encoding="utf-8")); audit=json.loads(SEED0.read_text(encoding="utf-8")) if SEED0.exists() else {}
    checks={"seed0_audit_exists":SEED0.exists(),"seed0_audit_completed":audit.get("status")=="completed_seed0_10k_collapse_audit","requested_seed_matches":cfg["seed"]==seed,"only_seed_differs":cfg["actions"]==base["actions"] and cfg["forecast_features"]==base["forecast_features"] and cfg["policy_architecture"]==base["policy_architecture"] and cfg["training"]==base["training"] and cfg["scope"]["train_years"]==base["scope"]["train_years"] and cfg["scope"]["validation_years"]==base["scope"]["validation_years"] and cfg["scope"]["reward_and_safety"]==base["scope"]["reward_and_safety"],"longer_runs_forbidden":cfg["execution_policy"]["no_25k_50k_75k_100k"] is True}; checks["next_step_allowed"]=all(checks.values()); return checks
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,choices=[1,2],required=True); ap.add_argument("--dry-run",action="store_true"); ap.add_argument("--train10k",action="store_true"); a=ap.parse_args()
    if a.dry_run==a.train10k: raise ValueError("choose one phase")
    cfgp=CONFIGS[a.seed]; auth=gate(a.seed)
    if not auth["next_step_allowed"]: raise RuntimeError(json.dumps(auth,ensure_ascii=False))
    patch_contract(); oldseed=forecast.engine.base03222.SEED; oldverify=forecast.verify_completed_smoke; oldrecord=forecast.write_record; forecast.engine.base03222.SEED=a.seed; forecast.verify_completed_smoke=lambda _cfg:auth
    def record(cfg,pf,copied,audit,action_gate,_phase,obs,calendar,smoke_verification=None): return oldrecord(cfg,pf,copied,audit,action_gate,f"independent seed{a.seed} 10K",obs,calendar,smoke_verification)
    forecast.write_record=record
    try:
        with patched_maskableppo(): result=forecast.run_forecast_experiment(cfgp,PROMPT,IDS[a.seed],a.dry_run,False,True)
    finally:
        forecast.engine.base03222.SEED=oldseed; forecast.verify_completed_smoke=oldverify; forecast.write_record=oldrecord
    result["independent_seed_gate"]=auth
    if a.train10k:
        out=ROOT/result["output_root"]; (out/f"{IDS[a.seed]}_10k_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
