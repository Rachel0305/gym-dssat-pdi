#!/usr/bin/env python3
"""Consolidate screened-year MaskablePPO evidence for advisor reporting."""

from __future__ import annotations

import json, math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmark_results'/'028_13_screened_year_maskableppo_advisor_summary'
DOC=ROOT/'docs'/'2026-07-18_028_13_screened_year_maskableppo_advisor_summary.md'
PACKAGE=ROOT/'benchmark_results'/'028_12_screened_year_representative_advisor_package'

COUNTS={
('HLA',2007):(2,3),('HLA',2010):(2,3),('HLA',2015):(2,3),('HLA',2016):(2,3),('HLA',2022):(1,3),
('YC',2008):(1,3),('YC',2014):(2,3),
('FQ',2013):(1,3),('FQ',2014):(2,3),('FQ',2016):(2,3),('FQ',2019):(3,3),('FQ',2020):(1,3),('FQ',2023):(1,3),
('LC',2010):(1,1),
('SY',2012):(1,3),('SY',2014):(3,3),('SY',2015):(2,3)}
ANCHORS={('HLA',2010),('YC',2014),('FQ',2016),('LC',2010),('SY',2014)}

def exact_selected(site:int|str,year:int,seed:int)->dict:
 if site=='HLA' and year!=2010:
  d=pd.read_csv(ROOT/'benchmark_results'/'028_08_hla_frozen_maskableppo_screened_year_transfer'/'028_08_hla_frozen_transfer_summary.csv'); r=d[(d.year==year)&(d.seed==seed)].iloc[0]
  return {'y':r.final_gwad,'wp':r.WP_ET_kg_m3,'p':r.PFP_N_kg_kg,'i':r.irrigation_total,'n':r.fertilizer_total,'wy':r.yield_strict_win,'ww':r.wp_et_strict_win,'wpn':r.pfp_n_strict_win}
 if site=='YC' and year==2008:
  d=pd.read_csv(ROOT/'benchmark_results'/'028_09_yc2014_frozen_maskableppo_yc2008_transfer'/'028_09_yc2008_frozen_transfer_summary.csv'); r=d[(d.year==year)&(d.seed==seed)].iloc[0]
  return {'y':r.final_gwad,'wp':r.WP_ET_kg_m3,'p':r.PFP_N_kg_kg,'i':r.irrigation_total,'n':r.fertilizer_total,'wy':r.yield_strict_win,'ww':r.wp_et_strict_win,'wpn':r.pfp_n_strict_win}
 if site=='FQ' and year!=2016:
  d=pd.read_csv(ROOT/'benchmark_results'/'028_11_fq_frozen_maskableppo_screened_year_transfer'/'028_11_fq_frozen_transfer_summary.csv'); r=d[(d.year==year)&(d.seed==seed)].iloc[0]
  return {'y':r.final_gwad,'wp':r.WP_ET_kg_m3,'p':r.PFP_N_kg_kg,'i':r.irrigation_total,'n':r.fertilizer_total,'wy':r.yield_strict_win,'ww':r.wp_et_strict_win,'wpn':r.pfp_n_strict_win}
 if site=='FQ':
  d=pd.read_csv(ROOT/'benchmark_results'/'028_10_fq2016_missing_seed1_seed2'/'028_10_fq2016_three_seed_selected_summary.csv'); r=d[d.seed==seed].iloc[0]
  return {'y':r['yield'],'wp':r.WP_ET,'p':r.PFP_N,'i':r.irrigation,'n':r.nitrogen,'wy':r.winning_yield,'ww':r.winning_WP_ET,'wpn':r.winning_PFP_N}
 d=pd.read_csv(ROOT/'benchmark_results'/'028_02_four_baseline_envelope_readiness'/'028_02_current_selected_rl_strict_comparison.csv'); r=d[(d.site==site)&(d.year==year)&(d.seed==seed)].iloc[0]
 return {'y':r.grain_yield_kg_ha,'wp':r.WP_ET_kg_m3,'p':r.PFP_N_kg_kg,'i':r.actual_irrigation_mm,'n':r.actual_nitrogen_kg_ha,'wy':r.strict_yield_pass,'ww':r.strict_wp_pass,'wpn':r.strict_pfp_pass}

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 summary=pd.read_csv(PACKAGE/'028_12_all_representative_five_scenario_summary.csv',keep_default_na=False)
 daily=pd.read_csv(PACKAGE/'028_12_all_representative_five_scenario_daily.csv',keep_default_na=False)
 registry=pd.read_csv(PACKAGE/'028_12_representative_registry.csv')
 rows=[]
 for (site,year),(winners,seeds) in COUNTS.items():
  case=summary[(summary.site==site)&(summary.year.astype(int)==year)]; rl=case[case.scenario=='rl_candidate'].iloc[0]; base=case[case.scenario!='rl_candidate']; rd=daily[(daily.site==site)&(daily.requested_year.astype(int)==year)&(daily.scenario=='rl_candidate')]
  reg=registry[(registry.site==site)&(registry.year.astype(int)==year)].iloc[0]
  ex=exact_selected(site,year,int(reg.seed)); y=float(ex['y']); w=float(ex['wp']); p=float(ex['p']) if pd.notna(ex['p']) else math.nan; ymax=float(pd.to_numeric(base.final_grain_kg_ha,errors='coerce').max()); wpmax=float(pd.to_numeric(base.wp_et_kg_m3,errors='coerce').max()); pvals=pd.to_numeric(base.pfp_n_kg_kg,errors='coerce').dropna(); pmax=float(pvals.max()) if not pvals.empty else math.nan
  wins=[name for name,flag in [('yield',bool(ex['wy'])),('WP_ET',bool(ex['ww'])),('PFP_N',bool(ex['wpn']))] if flag]
  rows.append({'site':site,'year':year,'model_type':'stage_MaskablePPO','training_role':'local_anchor_training' if (site,year) in ANCHORS else 'frozen_same_site_crossyear_transfer','local_training_steps':240 if (site,year) in ANCHORS else 0,'source_training_steps':240,'representative_seed':int(reg.seed),'representative_checkpoint':int(reg.checkpoint),'yield_kg_ha':y,'WP_ET_kg_m3':w,'PFP_N_kg_kg':p,'irrigation_mm':float(ex['i']),'nitrogen_kg_ha':float(ex['n']),'rainfall_mm':float(rl.rain_total_mm),'max_water_stress_index':float(rl.max_water_stress_wspd),'max_nitrogen_stress_index':float(rl.max_nitrogen_stress_nstd),'mean_soil_water_mm':float(pd.to_numeric(rd.soil_water_mm,errors='coerce').mean()),'min_soil_water_mm':float(pd.to_numeric(rd.soil_water_mm,errors='coerce').min()),'winning_metrics':';'.join(wins),'representative_any_metric_win':bool(wins),'winner_seed_count':winners,'seed_count':seeds,'cross_seed_status':'not_assessed_one_seed' if seeds<3 else ('initially_stable_2of3_or_better' if winners>=2 else 'candidate_exists_not_cross_seed_stable'),'yield_gap_pct_vs_four_max':(y/ymax-1)*100,'WP_ET_gap_pct_vs_four_max':(w/wpmax-1)*100,'PFP_N_gap_pct_vs_four_max':(p/pmax-1)*100 if math.isfinite(p) else math.nan,'endpoint_precision_source':'full_precision_training_or_transfer_summary; daily figures retain integer PlantGro.OUT','figure_dir':f'benchmark_results/028_12_screened_year_representative_advisor_package/{site}/{year}/figures'})
 overview=pd.DataFrame(rows).sort_values(['site','year']); overview.to_csv(OUT/'028_13_site_year_overview.csv',index=False,encoding='utf-8-sig')
 station=overview.groupby('site').agg(screened_years=('year','count'),years_with_candidate=('representative_any_metric_win','sum'),years_cross_seed_stable=('cross_seed_status',lambda x:int((x=='initially_stable_2of3_or_better').sum())),years_not_stability_assessed=('cross_seed_status',lambda x:int((x=='not_assessed_one_seed').sum()))).reset_index(); station.to_csv(OUT/'028_13_station_summary.csv',index=False,encoding='utf-8-sig')
 fig,ax=plt.subplots(figsize=(13,6)); x=np.arange(len(overview)); colors=['#18864B' if s=='initially_stable_2of3_or_better' else '#D39B2A' if s=='candidate_exists_not_cross_seed_stable' else '#6A77B8' for s in overview.cross_seed_status]; ax.bar(x,overview.winner_seed_count,color=colors,edgecolor='#222',linewidth=.5); ax.axhline(2,color='#555',ls='--',lw=1,label='2/3 initial stability threshold'); ax.set_xticks(x,[f'{s}{y}' for s,y in zip(overview.site,overview.year)],rotation=55,ha='right'); ax.set_ylabel('Seeds satisfying advisor any-metric rule'); ax.set_ylim(0,3.35); ax.set_title('Stage MaskablePPO across screened site-years',loc='left',fontweight='bold'); ax.grid(axis='y',color='#e5e5e5'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT/'028_13_cross_seed_overview.png',dpi=220,bbox_inches='tight'); fig.savefig(OUT/'028_13_cross_seed_overview.svg',bbox_inches='tight'); plt.close(fig)
 stable=int((overview.cross_seed_status=='initially_stable_2of3_or_better').sum()); candidates=int(overview.representative_any_metric_win.sum())
 lines=['# 028_13 已筛选年份 MaskablePPO 导师总览','','## 结论先行','',f'- 17/17 个已筛选站点年都至少存在 1 个 seed，在产量、WP_ET、PFP_N 中至少一项严格超过四基线最大值。',f'- 其中 {stable}/16 个已完成三 seed 的站点年达到 ≥2/3 初步稳定；LC2010 仅 1 seed，不能判定跨 seed 稳定。','- 五个站点均有成功候选；但当前证据不支持“所有年份、所有 seed 都成功”。','- 可称为统一阶段型 MaskablePPO 框架：主算法和训练超参数一致；不可称为单一通用模型，因为 scaler、输入、可执行阶段数和模型权重仍是站点专属。','','## 五站点汇总','','|site|screened years|years with candidate|years >=2/3 stable|not assessed|','|---|---:|---:|---:|---:|']
 for _,r in station.iterrows(): lines.append(f"|{r.site}|{int(r.screened_years)}|{int(r.years_with_candidate)}|{int(r.years_cross_seed_stable)}|{int(r.years_not_stability_assessed)}|")
 lines += ['','## 年份级结果','','|site-year|representative seed|winning metrics|winner seeds|stability|yield|WP_ET|PFP_N|I|N|','|---|---:|---|---:|---|---:|---:|---:|---:|---:|']
 for _,r in overview.iterrows():
  p='NA' if pd.isna(r.PFP_N_kg_kg) else f'{r.PFP_N_kg_kg:.1f}'; lines.append(f"|{r.site}{int(r.year)}|{int(r.representative_seed)}|{r.winning_metrics}|{int(r.winner_seed_count)}/{int(r.seed_count)}|{r.cross_seed_status}|{r.yield_kg_ha:.0f}|{r.WP_ET_kg_m3:.2f}|{p}|{r.irrigation_mm:.0f}|{r.nitrogen_kg_ha:.0f}|")
 lines += ['','## 解释边界','','- “17/17存在候选”来自预注册代表 seed；没有隐藏其他失败 seed，全部 winner count 已列出。','- “接近”仍未设置人为容差；未领先指标的百分比差距保存在 CSV，交由导师判断。','- HLA2007 等 prepared/derived 输入变体及 FQ recorded_shifted 来源必须随结果一并说明。','- 每个年份的数据、图和模型哈希由 028_12 manifest 绑定。']
 DOC.write_text('\n'.join(lines)+'\n',encoding='utf-8'); payload={'status':'completed','screened_site_years':len(overview),'site_years_with_any_candidate':candidates,'three_seed_site_years':int((overview.seed_count==3).sum()),'three_seed_stable_site_years':stable,'sites_with_candidate':int((station.years_with_candidate>0).sum()),'training_calls':0}; (OUT/'028_13_result.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
