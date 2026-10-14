#!/usr/bin/env python3
"""Build an advisor package for all 17 screened site-years."""

from __future__ import annotations

import hashlib, json, math, shutil, sys
from pathlib import Path
from typing import Any
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'src'):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from src import build_relaxed_success_five_scenario_daily_evidence_027_05 as old
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo

OUT=ROOT/'benchmark_results'/'028_12_screened_year_representative_advisor_package'
DOC=ROOT/'docs'/'2026-07-18_028_12_screened_year_representative_advisor_package.md'
STATION={'HLA':'Hailun','YC':'Yucheng','FQ':'Fengqiu','LC':'Luancheng','SY':'Shenyang'}
SELECTED={('HLA',2007):(0,180),('HLA',2010):(0,180),('HLA',2015):(1,60),('HLA',2016):(1,60),('HLA',2022):(1,60),('YC',2008):(2,120),('YC',2014):(2,120),('FQ',2013):(0,60),('FQ',2014):(1,60),('FQ',2016):(2,60),('FQ',2019):(2,60),('FQ',2020):(2,60),('FQ',2023):(1,60),('LC',2010):(0,240),('SY',2012):(1,60),('SY',2014):(0,120),('SY',2015):(2,240)}

def sha256(p:Path)->str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()

def model_path(site:str,seed:int,ckpt:int)->Path:
 if site=='HLA': return (ROOT/'benchmark_results'/'027_02_attempt2'/f'checkpoint_{ckpt:06d}.zip') if seed==0 else ROOT/'benchmark_results'/'027_03'/f'seed{seed}'/f'checkpoint_{ckpt:06d}.zip'
 if site=='YC': return ROOT/'benchmark_results'/'027_07_site_specific_stage_maskable_ppo_attempt2'/'YC'/f'seed{seed}'/f'checkpoint_{ckpt:06d}.zip'
 if site=='FQ': return (ROOT/'benchmark_results'/'027_07_site_specific_stage_maskable_ppo_attempt2'/'FQ'/'seed0'/f'checkpoint_{ckpt:06d}.zip') if seed==0 else ROOT/'benchmark_results'/'028_10_fq2016_missing_seed1_seed2'/'FQ'/f'seed{seed}'/f'checkpoint_{ckpt:06d}.zip'
 if site=='LC': return ROOT/'benchmark_results'/'027_07_site_specific_stage_maskable_ppo_attempt2'/'LC'/'seed0'/f'checkpoint_{ckpt:06d}.zip'
 if site=='SY': return {0:ROOT/'benchmark_results'/'026_03'/'checkpoint_000120.zip',1:ROOT/'benchmark_results'/'026_02'/'checkpoint_000060.zip',2:ROOT/'benchmark_results'/'026_04'/'checkpoint_000240.zip'}[seed]
 raise ValueError(site)

def baseline_snapshots(site:str,year:int)->dict[str,Path]:
 if site=='HLA':
  base=ROOT/'DSSAT_auto_validation'/'HLA_2004'/'hla_five_scenario_nstep_020_11'/'runs'/str(year); return {'null':base/'null'/'pdi_tmp_snapshot_eval','recorded_farmer':base/'recorded_farmer'/'pdi_tmp_snapshot_eval','dssat_auto':base/'dssat_auto'/'pdi_tmp_snapshot_eval','official_extension_expert':base/'extension_expert'/'pdi_tmp_snapshot_eval'}
 if site=='YC' and year==2014:
  base=ROOT/'benchmark_results'/'027_07_site_specific_stage_maskable_ppo_attempt2'/'YC'/'readiness'/'baseline_runs'; return {s:base/s/'pdi_tmp_snapshot_eval' for s in ('null','recorded_farmer','dssat_auto','official_extension_expert')}
 if site=='YC':
  base=ROOT/'DSSAT_auto_validation'/'multisite_new_cultivar_forward_screening_013_01'/'runs'/'YC'; return {'null':base/'2008_null'/'pdi_tmp_snapshot','recorded_farmer':base/'2008_recorded'/'pdi_tmp_snapshot','dssat_auto':base/'2008_dssat_auto'/'pdi_tmp_snapshot','official_extension_expert':ROOT/'benchmark_results'/'028_07_missing_official_expert_six_season_completion'/'runs_completed'/'YC2008'/'official_extension_expert'/'pdi_tmp_snapshot_eval'}
 if site=='FQ' and year==2016:
  base=ROOT/'benchmark_results'/'027_07_site_specific_stage_maskable_ppo_attempt2'/'FQ'/'readiness'/'baseline_runs'; return {s:base/s/'pdi_tmp_snapshot_eval' for s in ('null','recorded_farmer','dssat_auto','official_extension_expert')}
 if site=='FQ':
  base=ROOT/'DSSAT_auto_validation'/'fq_all_year_screen_and_dqn_transfer_014_01'/'runs'/str(year)/'seed0'; return {'null':base/'null'/'pdi_tmp_snapshot_eval','recorded_farmer':base/'recorded_shifted'/'pdi_tmp_snapshot_eval','dssat_auto':base/'dssat_auto'/'pdi_tmp_snapshot_eval','official_extension_expert':ROOT/'benchmark_results'/'028_07_missing_official_expert_six_season_completion'/'runs_completed'/f'FQ{year}'/'official_extension_expert'/'pdi_tmp_snapshot_eval'}
 if site=='LC':
  base=ROOT/'benchmark_results'/'027_07_site_specific_stage_maskable_ppo_attempt2'/'LC'/'readiness'/'baseline_runs'; return {s:base/s/'pdi_tmp_snapshot_eval' for s in ('null','recorded_farmer','dssat_auto','official_extension_expert')}
 if site=='SY':
  base=ROOT/'benchmark_results'/'026_07_attempt2'/str(year); return {'null':base/'baseline_source'/'runs'/str(year)/'seed0'/'null'/'pdi_tmp_snapshot_eval','recorded_farmer':base/'baseline_source'/'runs'/str(year)/'seed0'/'recorded'/'pdi_tmp_snapshot_eval','dssat_auto':base/'baseline_source'/'runs'/str(year)/'seed0'/'dssat_auto'/'pdi_tmp_snapshot_eval','official_extension_expert':base/'expert_source'/'runs'/str(year)/'seed0'/'transfer_official_extension_expert'/'pdi_tmp_snapshot_eval'}
 raise ValueError((site,year))

def rl_snapshot(site:str,year:int,seed:int)->Path:
 if site=='HLA' and year==2010: return ROOT/'benchmark_results'/'027_05_ppo_frozen_daily_completion'/'HLA2010'/'snapshot'
 if site=='HLA' and year==2007 and seed==0: return ROOT/'benchmark_results'/'028_08_smoke_hla2007_seed0'/'runs'/'2007'/'seed0'/'pdi_tmp_snapshot_eval'
 if site=='HLA': return ROOT/'benchmark_results'/'028_08_hla_frozen_maskableppo_screened_year_transfer'/'runs'/str(year)/f'seed{seed}'/'pdi_tmp_snapshot_eval'
 if site=='YC' and year==2014: return ROOT/'benchmark_results'/'028_04_existing_yc_fq_lc_ppo_frozen_daily'/'YC'/f'seed{seed}_checkpoint120'/'snapshot'
 if site=='YC': return ROOT/'benchmark_results'/'028_09_yc2014_frozen_maskableppo_yc2008_transfer'/'runs'/'2008'/f'seed{seed}'/'pdi_tmp_snapshot_eval'
 if site=='FQ' and year==2016: return OUT/'fq2016_seed2_frozen_reevaluation'/'snapshot'
 if site=='FQ' and year==2013 and seed==0: return ROOT/'benchmark_results'/'028_11_smoke_fq2013_seed0'/'runs'/'2013'/'seed0'/'pdi_tmp_snapshot_eval'
 if site=='FQ': return ROOT/'benchmark_results'/'028_11_fq_frozen_maskableppo_screened_year_transfer'/'runs'/str(year)/f'seed{seed}'/'pdi_tmp_snapshot_eval'
 if site=='LC': return ROOT/'benchmark_results'/'028_04_existing_yc_fq_lc_ppo_frozen_daily'/'LC'/'seed0_checkpoint240'/'snapshot'
 if site=='SY' and year==2014: return ROOT/'benchmark_results'/'027_05_ppo_frozen_daily_completion'/'SY2014'/'snapshot'
 return ROOT/'benchmark_results'/'028_05_sy_crossyear_frozen_ppo_daily'/str(year)/f'seed{seed}'/'snapshot'

def actions_for(site:str,year:int,seed:int)->pd.DataFrame:
 if site=='HLA' and year==2010:
  d=pd.read_csv(ROOT/'benchmark_results'/'027_02_attempt2'/'027_02_hla2010_seed0_checkpoint_stage_actions.csv'); return d[d.checkpoint==180].copy()
 if site=='HLA':
  d=pd.read_csv(ROOT/'benchmark_results'/'028_08_hla_frozen_maskableppo_screened_year_transfer'/'028_08_hla_frozen_transfer_stage_actions.csv'); return d[(d.year==year)&(d.seed==seed)].copy()
 if site=='YC' and year==2014: return pd.read_csv(ROOT/'benchmark_results'/'028_04_existing_yc_fq_lc_ppo_frozen_daily'/'YC'/f'seed{seed}_checkpoint120'/'stage_actions.csv')
 if site=='YC':
  d=pd.read_csv(ROOT/'benchmark_results'/'028_09_yc2014_frozen_maskableppo_yc2008_transfer'/'028_09_yc2008_frozen_transfer_stage_actions.csv'); return d[(d.year==year)&(d.seed==seed)].copy()
 if site=='FQ' and year==2016: return pd.read_csv(OUT/'fq2016_seed2_frozen_reevaluation'/'stage_actions.csv')
 if site=='FQ':
  d=pd.read_csv(ROOT/'benchmark_results'/'028_11_fq_frozen_maskableppo_screened_year_transfer'/'028_11_fq_frozen_transfer_actions.csv'); return d[(d.year==year)&(d.seed==seed)].copy()
 if site=='LC': return pd.read_csv(ROOT/'benchmark_results'/'028_04_existing_yc_fq_lc_ppo_frozen_daily'/'LC'/'seed0_checkpoint240'/'stage_actions.csv')
 if site=='SY' and year==2014:
  d=pd.read_csv(ROOT/'benchmark_results'/'026_03'/'026_03_seed0_checkpoint_stage_actions.csv'); return d[d.checkpoint==120].copy()
 return pd.read_csv(ROOT/'benchmark_results'/'028_05_sy_crossyear_frozen_ppo_daily'/str(year)/f'seed{seed}'/'stage_actions.csv')

def persist_fq2016_seed2():
 target=OUT/'fq2016_seed2_frozen_reevaluation'
 if target.exists(): return
 target.mkdir(parents=True); sp=siteppo.SPECS['FQ']; ready=ROOT/'benchmark_results'/'027_07_site_specific_stage_maskable_ppo_attempt2'/'FQ'/'readiness'; scaler=pd.read_csv(ready/'observation_scaler.csv'); reward=json.loads((ready/'readiness_result.json').read_text(encoding='utf-8'))['reward_config']; baselines=pd.read_csv(ready/'four_baseline_fresh_rerun.csv',keep_default_na=False)
 env=siteppo.make_stage_env(sp,target/'runtime',scaler,reward,1002,'028_12_frozen_eval_seed2'); mp=model_path('FQ',2,60); before=sha256(mp); acts=[]
 try:
  model=MaskablePPO.load(mp,device='cpu'); obs,info=env.reset(); done=False
  while not done:
   mask=get_action_masks(env); stage=int(env.stage_index); action,_=model.predict(obs,action_masks=mask,deterministic=True); obs,r,t,tr,info=env.step(int(action)); acts.append({'site':'FQ','year':2016,'seed':2,'stage_index':stage,**env.stage_rows[-1]}); done=bool(t or tr)
  shutil.copytree(siteppo.snapshot_from_env(env.raw_env),target/'snapshot')
 finally: env.close()
 if sha256(mp)!=before or len(acts)!=5: raise RuntimeError('FQ2016 frozen reevaluation invariant failed')
 pd.DataFrame(acts).to_csv(target/'stage_actions.csv',index=False,encoding='utf-8-sig'); (target/'reevaluation.json').write_text(json.dumps({'training_calls':0,'model_sha256':before,'action_sequence':','.join(str(x['action_index']) for x in acts)},ensure_ascii=False,indent=2),encoding='utf-8')

def main():
 if OUT.exists(): raise FileExistsError(OUT)
 OUT.mkdir(parents=True); persist_fq2016_seed2(); all_daily=[]; all_summary=[]; all_checks=[]; registry=[]
 for (site,year),(seed,ckpt) in SELECTED.items():
  case_dir=OUT/site/str(year); case_dir.mkdir(parents=True); mp=model_path(site,seed,ckpt); snaps=baseline_snapshots(site,year); snaps['rl_candidate']=rl_snapshot(site,year,seed)
  missing=[str(x) for x in snaps.values() if not x.exists()]
  if missing: raise FileNotFoundError(f'{site}{year}: {missing}')
  case=old.Case(site,STATION[site],year,seed,ckpt,mp,snaps,case_dir/'selection.json','Pre-registered representative seed; all seeds remain in source tables.')
  daily,summary,checks=old.build_case(case,algorithm='MaskablePPO'); daily.to_csv(case_dir/'five_scenario_daily.csv',index=False,encoding='utf-8-sig'); summary.to_csv(case_dir/'five_scenario_summary.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(checks).to_csv(case_dir/'daily_evidence_checks.csv',index=False,encoding='utf-8-sig')
  if not pd.DataFrame(checks).passed.all(): raise RuntimeError(f'{site}{year} daily checks failed')
  acts=actions_for(site,year,seed); acts.to_csv(case_dir/'stage_actions.csv',index=False,encoding='utf-8-sig'); renamed=acts.rename(columns={'executed_irrigation':'executed_irrigation_mm','executed_nitrogen':'executed_nitrogen_kg_ha','executed_executed_irrigation':'executed_irrigation_mm','executed_executed_nitrogen':'executed_nitrogen_kg_ha'})
  old.FIG=case_dir/'figures'; old.FIG.mkdir(); dummy=old.ExistingPPOCase(site,STATION[site],year,seed,ckpt,mp,case_dir/'five_scenario_summary.csv',case_dir/'five_scenario_summary.csv',case_dir/'stage_actions.csv',('checkpoint',ckpt)); figs=old.plot_ppo_endpoints(summary,dummy)+old.plot_daily(daily,case,algorithm='MaskablePPO')+old.plot_ppo_stage_actions(renamed,dummy)
  manifest={'site':site,'year':year,'seed':seed,'checkpoint':ckpt,'model_path':str(mp.relative_to(ROOT)).replace('\\','/'),'model_sha256':sha256(mp),'daily_sha256':sha256(case_dir/'five_scenario_daily.csv'),'summary_sha256':sha256(case_dir/'five_scenario_summary.csv'),'figures':[str(f.relative_to(ROOT)).replace('\\','/') for f in figs],'training_calls_for_package':0,'extra_dssat_evaluations_for_package':1 if site=='FQ' and year==2016 else 0}
  (case_dir/'evidence_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8'); registry.append(manifest); all_daily.append(daily); all_summary.append(summary); all_checks.append(pd.DataFrame(checks)); print(f'OK package {site}{year} seed{seed}')
 pd.DataFrame(registry).to_csv(OUT/'028_12_representative_registry.csv',index=False,encoding='utf-8-sig'); pd.concat(all_daily,ignore_index=True).to_csv(OUT/'028_12_all_representative_five_scenario_daily.csv',index=False,encoding='utf-8-sig'); pd.concat(all_summary,ignore_index=True).to_csv(OUT/'028_12_all_representative_five_scenario_summary.csv',index=False,encoding='utf-8-sig'); pd.concat(all_checks,ignore_index=True).to_csv(OUT/'028_12_all_daily_checks.csv',index=False,encoding='utf-8-sig')
 lines=['# 028_12 已筛选年份代表策略五情景汇报包','','共17个站点—年份；每年一个预注册代表seed，但全部seed稳定性仍以028_08/09/10/11及026/027源表为准。','','|site|year|seed|checkpoint|','|---|---:|---:|---:|']+[f'|{s}|{y}|{v[0]}|{v[1]}|' for (s,y),v in SELECTED.items()]+['','本任务新增训练0次；仅FQ2016 seed2因原训练期snapshot未持久化而做1季固定模型复评。每个case均有CSV、PNG、SVG和manifest。']
 DOC.write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(json.dumps({'status':'completed','cases':len(registry),'figures':sum(len(x['figures']) for x in registry),'training_calls':0,'new_dssat_seasons':1},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
