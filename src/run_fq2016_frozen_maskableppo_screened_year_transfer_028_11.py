#!/usr/bin/env python3
"""Frozen FQ2016 MaskablePPO transfer to five screened FQ years."""

from __future__ import annotations

import argparse, hashlib, json, math, shutil, sys
from pathlib import Path
from typing import Any
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'src'):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
import run_extension_expert_baseline_018_03 as extension
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo

OUT=ROOT/'benchmark_results'/'028_11_fq_frozen_maskableppo_screened_year_transfer'
SMOKE=ROOT/'benchmark_results'/'028_11_smoke_fq2013_seed0'
DOC=ROOT/'docs'/'2026-07-18_028_11_fq_frozen_maskableppo_screened_year_transfer.md'
YEARS=(2013,2014,2019,2020,2023)
SCALER=ROOT/'benchmark_results'/'027_07_site_specific_stage_maskable_ppo_attempt2'/'FQ'/'readiness'/'observation_scaler.csv'
MODELS={
0:ROOT/'benchmark_results'/'027_07_site_specific_stage_maskable_ppo_attempt2'/'FQ'/'seed0'/'checkpoint_000060.zip',
1:ROOT/'benchmark_results'/'028_10_fq2016_missing_seed1_seed2'/'FQ'/'seed1'/'checkpoint_000060.zip',
2:ROOT/'benchmark_results'/'028_10_fq2016_missing_seed1_seed2'/'FQ'/'seed2'/'checkpoint_000060.zip'}
SCREEN=ROOT/'DSSAT_auto_validation'/'fq_all_year_screen_and_dqn_transfer_014_01'
EXPERT=ROOT/'benchmark_results'/'028_07_missing_official_expert_six_season_completion'

def sha256(p:Path)->str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()

def spec(year:int)->siteppo.SiteSpec:
 yy=year%100
 return siteppo.SiteSpec('FQ','Fengqiu',year,2,siteppo.INPUT_PARENT/'FQ','CNFQ0801.MZX',f'CNFQ{yy:02d}01.WTH','FQ99001200','FQ0985',23,7,siteppo.OFFICIAL_HUANGHUAI_STAGES,siteppo.OFFICIAL_HUANGHUAI_STAGES[:-1],120,1,'07152',f'{yy:02d}153',f'{yy:02d}162',2,f'CNFQ{yy:02d}01')

def plant_end(snapshot:Path)->tuple[float,float]:
 t=extension.parse_dssat_table(snapshot/'PlantGro.OUT'); return float(t.GWAD.dropna().iloc[-1]),float(t.CWAD.dropna().iloc[-1])

def baseline_rows()->pd.DataFrame:
 old=pd.read_csv(SCREEN/'014_01_fq_all_year_screening_summary.csv',keep_default_na=False)
 expert_summary=pd.read_csv(EXPERT/'028_07_official_expert_summary.csv')
 rows=[]
 for year in YEARS:
  for scenario,dirname in [('null','null'),('recorded','recorded_shifted'),('dssat_auto','dssat_auto')]:
   snap=SCREEN/'runs'/str(year)/'seed0'/dirname/'pdi_tmp_snapshot_eval'
   src=old[(old.year.astype(int)==year)&(old.scenario.fillna('').replace('', 'null')==dirname)].iloc[0]
   y,b=plant_end(snap); i=float(src.event_irrigation_total or 0); n=float(src.event_fertilizer_total or 0)
   m=siteppo.strict_metrics_from_snapshot(snap,y,i,n)
   rows.append({'site':'FQ','year':year,'scenario':scenario,'final_gwad':y,'final_cwad':b,'irrigation_total':i,'fertilizer_total':n,'snapshot':str(snap.relative_to(ROOT)).replace('\\','/'),**m})
  er=expert_summary[(expert_summary.site=='FQ')&(expert_summary.year.astype(int)==year)].iloc[0]
  snap=EXPERT/'runs_completed'/f'FQ{year}'/'official_extension_expert'/'pdi_tmp_snapshot_eval'
  y,b=plant_end(snap); i=float(er.action_irrigation_total); n=float(er.action_fertilizer_total)
  m=siteppo.strict_metrics_from_snapshot(snap,y,i,n)
  rows.append({'site':'FQ','year':year,'scenario':'official_extension_expert','final_gwad':y,'final_cwad':b,'irrigation_total':i,'fertilizer_total':n,'snapshot':str(snap.relative_to(ROOT)).replace('\\','/'),**m})
 return pd.DataFrame(rows)

def evaluate(year:int,seed:int,root:Path,base:pd.DataFrame,scaler:pd.DataFrame)->tuple[dict[str,Any],list[dict[str,Any]]]:
 sp=spec(year); local=base[base.year==year]; run=root/'runs'/str(year)/f'seed{seed}'
 env=siteppo.make_stage_env(sp,run,scaler,{'local_null_yield':float(local[local.scenario=='null'].final_gwad.iloc[0]),'local_feasibility_yield':float(local[local.scenario=='official_extension_expert'].final_gwad.iloc[0])},1000+seed,f'frozen_transfer_{year}_seed{seed}')
 mp=MODELS[seed]; before=sha256(mp); acts=[]
 try:
  model=MaskablePPO.load(mp,device='cpu'); obs,info=env.reset(); done=False
  while not done:
   mask=get_action_masks(env); stage=int(env.stage_index); action,_=model.predict(obs,action_masks=mask,deterministic=True)
   obs,reward,term,trunc,info=env.step(int(action)); acts.append({'site':'FQ','year':year,'seed':seed,'stage_index':stage,**env.stage_rows[-1]}); done=bool(term or trunc)
  if env.last_result is None: raise RuntimeError('missing final result')
  snap=run/'pdi_tmp_snapshot_eval'; shutil.copytree(siteppo.snapshot_from_env(env.raw_env),snap)
  m=siteppo.strict_metrics_from_snapshot(snap,env.last_result['final_yield'],env.last_result['irrigation_total'],env.last_result['nitrogen_total'])
  row={'site':'FQ','year':year,'seed':seed,'source_model':str(mp.relative_to(ROOT)).replace('\\','/'),'model_sha256':before,'action_sequence':','.join(str(a['action_index']) for a in acts),'final_gwad':env.last_result['final_yield'],'final_cwad':env.last_result['final_biomass'],'irrigation_total':env.last_result['irrigation_total'],'fertilizer_total':env.last_result['nitrogen_total'],'invalid_action_attempts':env.invalid_attempts,'run_dir':str(run.relative_to(ROOT)).replace('\\','/'),**m}
 finally: env.close()
 if sha256(mp)!=before or len(acts)!=5: raise RuntimeError('frozen hash/stage invariant failed')
 return row,acts

def add_rule(df:pd.DataFrame,base:pd.DataFrame)->pd.DataFrame:
 out=[]
 for _,r in df.iterrows():
  b=base[base.year==int(r.year)]; ym=float(b.final_gwad.max()); wm=float(b.WP_ET_kg_m3.max()); pm=float(b.PFP_N_kg_kg.dropna().max()); d=r.to_dict(); p=float(r.PFP_N_kg_kg) if pd.notna(r.PFP_N_kg_kg) else math.nan
  d.update(four_baseline_max_yield=ym,four_baseline_max_WP_ET=wm,four_baseline_max_PFP_N=pm,yield_strict_win=float(r.final_gwad)>ym,wp_et_strict_win=float(r.WP_ET_kg_m3)>wm,pfp_n_strict_win=math.isfinite(p) and p>pm,yield_gap_pct=(float(r.final_gwad)/ym-1)*100,wp_et_gap_pct=(float(r.WP_ET_kg_m3)/wm-1)*100,pfp_n_gap_pct=(p/pm-1)*100 if math.isfinite(p) else math.nan)
  d['advisor_any_metric_win']=d['yield_strict_win'] or d['wp_et_strict_win'] or d['pfp_n_strict_win']; out.append(d)
 return pd.DataFrame(out)

def run_smoke():
 if SMOKE.exists(): raise FileExistsError(SMOKE)
 SMOKE.mkdir(parents=True); b=baseline_rows(); s=pd.read_csv(SCALER); r,a=evaluate(2013,0,SMOKE,b,s)
 pd.DataFrame(a).to_csv(SMOKE/'028_11_smoke_actions.csv',index=False,encoding='utf-8-sig'); (SMOKE/'028_11_smoke_result.json').write_text(json.dumps({'status':'passed','result':r,'training_steps':0},ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({'status':'passed','yield':r['final_gwad'],'actions':r['action_sequence']},ensure_ascii=False))

def reuse_smoke():
 p=json.loads((SMOKE/'028_11_smoke_result.json').read_text(encoding='utf-8')); return p['result'],pd.read_csv(SMOKE/'028_11_smoke_actions.csv').to_dict('records')

def main():
 if OUT.exists(): raise FileExistsError(OUT)
 OUT.mkdir(parents=True); b=baseline_rows(); s=pd.read_csv(SCALER); b.to_csv(OUT/'028_11_reused_four_baselines.csv',index=False,encoding='utf-8-sig'); rows=[]; acts=[]
 for y in YEARS:
  for seed in (0,1,2):
   r,a=reuse_smoke() if y==2013 and seed==0 else evaluate(y,seed,OUT,b,s); rows.append(r); acts.extend(a); print(f"OK FQ{y} seed{seed}: Y={r['final_gwad']:.3f} I={r['irrigation_total']} N={r['fertilizer_total']}")
 df=add_rule(pd.DataFrame(rows),b); df.to_csv(OUT/'028_11_fq_frozen_transfer_summary.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(acts).to_csv(OUT/'028_11_fq_frozen_transfer_actions.csv',index=False,encoding='utf-8-sig')
 matrix=df.groupby('year').advisor_any_metric_win.agg(['count','sum']).reset_index().rename(columns={'sum':'winner_count'}); matrix['at_least_two_of_three']=matrix.winner_count>=2; matrix.to_csv(OUT/'028_11_fq_year_seed_matrix.csv',index=False,encoding='utf-8-sig')
 payload={'status':'completed','year_matrix':matrix.to_dict('records'),'training_steps':0,'invalid_actions':int(df.invalid_action_attempts.sum())}; (OUT/'028_11_result.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
 write_doc(matrix); print(json.dumps(payload,ensure_ascii=False,indent=2))

def write_doc(matrix:pd.DataFrame):
 lines=['# 028_11 FQ2016 固定 MaskablePPO 跨年迁移','','状态：`completed`；训练步数 0。','','|year|seed count|winner count|2/3通过|','|---:|---:|---:|---|']
 for _,r in matrix.iterrows(): lines.append(f"|{int(r.year)}|{int(r['count'])}|{int(r.winner_count)}|{bool(r.at_least_two_of_three)}|")
 lines += ['', '执行记录：第一次结果报告尝试因容器缺少可选 `tabulate` 包而在 Markdown 渲染阶段失败；15 季科学结果和 CSV 均已先行写盘。本记录直接复用现有 CSV 完成，没有重跑 DSSAT。']
 DOC.write_text('\n'.join(lines)+'\n',encoding='utf-8')

def finalize_existing():
 matrix_path=OUT/'028_11_fq_year_seed_matrix.csv'; result_path=OUT/'028_11_result.json'
 if not matrix_path.is_file() or not result_path.is_file(): raise FileNotFoundError('Existing 028_11 outputs are incomplete')
 failure=OUT/'failed_attempt_1_missing_tabulate.json'
 if not failure.exists(): failure.write_text(json.dumps({'failure_stage':'markdown_report_only','scientific_seasons_completed':15,'dssat_reruns_for_fix':0,'error':'Missing optional dependency tabulate'},ensure_ascii=False,indent=2),encoding='utf-8')
 write_doc(pd.read_csv(matrix_path)); print('028_11 existing scientific outputs finalized; DSSAT reruns=0')

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--smoke-only',action='store_true'); p.add_argument('--finalize-existing',action='store_true'); a=p.parse_args(); finalize_existing() if a.finalize_existing else (run_smoke() if a.smoke_only else main())
