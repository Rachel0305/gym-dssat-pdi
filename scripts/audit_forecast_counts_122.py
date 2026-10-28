from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/forecast_publication_audit_122_20261027';OUT.mkdir(exist_ok=True)
PKG=ROOT/'results/forecast_five_scenario_120_corrected_HL_20261027'
d=pd.read_csv(PKG/'all_five_sites_seed_year_scenario.csv',keep_default_na=False)
metrics=['grain_yield_kg_ha','actual_irrigation_mm','actual_nitrogen_kg_ha','WP_ET_kg_m3','PFP_N_kg_kg']
d[metrics]=d[metrics].apply(pd.to_numeric,errors='coerce')
p=d[d.scenario.eq('rl_candidate')].copy();assert len(p)==400
assert p.WP_ET_kg_m3.notna().all()
assert p.PFP_N_kg_kg.isna().eq(p.actual_nitrogen_kg_ha.eq(0)).all()
p[p.PFP_N_kg_kg.isna()].to_csv(OUT/'undefined_PFP_N_26.csv',index=False)
p.groupby(['site','seed']).agg(years=('year','count'),WP_ET_valid=('WP_ET_kg_m3','count'),PFP_N_valid=('PFP_N_kg_kg','count'),zero_N_years=('actual_nitrogen_kg_ha',lambda s:int(s.eq(0).sum()))).to_csv(OUT/'efficiency_valid_counts_40.csv')
old=pd.read_csv(ROOT/'results/benefit_reliability_066_20261019/primary_policy_table_80.csv')
rows=[]
for (site,seed),q in p.groupby(['site','seed']):
    tab=PKG/site/f'seed_{seed:02d}'/'tables/five_scenario_daily.csv'
    days=pd.read_csv(tab,keep_default_na=False)
    ev=days[days.scenario.eq('rl_candidate')&((days.irrigation_executed_mm>0)|(days.nitrogen_executed_kg_ha>0))].copy()
    assert len(q)==10
    for y,g in ev.groupby('year'):
        endpoint=q[q.year.eq(y)].iloc[0]
        assert abs(g.irrigation_executed_mm.sum()-endpoint.actual_irrigation_mm)<1e-5
        assert abs(g.nitrogen_executed_kg_ha.sum()-endpoint.actual_nitrogen_kg_ha)<1e-5
    irrigation=ev[ev.irrigation_executed_mm.gt(0)]
    expert=days[days.scenario.eq('official_extension_expert')&days.irrigation_executed_mm.gt(0)]
    freq=len(irrigation)/len(expert)
    oldfreq=old[old.site.eq(site)].expert_irrigation_events_per_year.unique()
    sequences=[]
    for y in range(2014,2024):
        g=ev[ev.year.eq(y)].sort_values(['dap','irrigation_executed_mm','nitrogen_executed_kg_ha'])
        sequences.append(tuple(zip(g.dap.round(3),g.irrigation_executed_mm.round(3),g.nitrogen_executed_kg_ha.round(3))))
    bases=d[d.site.eq(site)&d.seed.eq(seed)&d.scenario.isin(['official_extension_expert','recorded_farmer_template'])]
    reference=bases.groupby('scenario')[metrics[:3]].mean().sort_values(metrics[0],ascending=False).iloc[0]
    y,I,N=q[metrics[:3]].mean();ry,rI,rN=reference.values
    diff=(y/ry-1)*100
    n1=diff>=-5-1e-8;n2=n1 and I<=rI+1e-8 and N<=rN+1e-8
    small=(irrigation.irrigation_executed_mm<=15).mean()
    gain=diff>=1-1e-8;saving=n2 and (I<=.9*rI+1e-8 or N<=.9*rN+1e-8)
    rows.append(dict(site=site,regime='forecast',seed=int(seed),grain_yield_kg_ha=y,total_irrigation_mm=I,total_n_kg_ha=N,reference_scenario=reference.name,reference_yield=ry,reference_I=rI,reference_N=rN,yield_delta_pct=diff,small_15mm_fraction=small,frequency_ratio_expert=freq,expert_irrigation_events_per_year=len(expert)/10,legacy_expert_events_per_year=oldfreq[0],fixed_full_sequence=len(set(sequences))==1,yield_gain=gain,yield_preserving_saving=saving,benefit_union=gain or saving,N1=n1,N2=n2,N3=n2 and small<=2/3 and freq<=1.5))
f=pd.DataFrame(rows);f.to_csv(OUT/'forecast_policy_table_40.csv',index=False)
old['N1']=old.yield_delta_pct.ge(-5-1e-8)
old['N2']=old.N1&old.total_irrigation_mm.le(old.reference_I)&old.total_n_kg_ha.le(old.reference_N)
old['N3']=old.N2&~old.small_15mm_fraction.gt(2/3)&~old.frequency_ratio_expert.gt(1.5)
combined=pd.concat([old,f],ignore_index=True)
combined.to_csv(OUT/'provisional_policy_table_120.csv',index=False)
def tally(q):
    return dict(n=len(q),gain=int(q.yield_gain.sum()),saving=int(q.yield_preserving_saving.sum()),overlap=int((q.yield_gain&q.yield_preserving_saving).sum()),union=int(q.benefit_union.sum()),N1=int(q.N1.sum()),N2=int(q.N2.sum()),N3=int(q.N3.sum()),N3_fixed=int((q.N3&q.fixed_full_sequence).sum()))
counts={name:tally(q) for name,q in [('legacy80',old),('forecast40',f),('provisional120',combined),('nonFQ_legacy64',old[old.site.ne('FQ')]),('nonFQ_forecast32',f[f.site.ne('FQ')]),('nonFQ96',combined[combined.site.ne('FQ')])]}
by_site=pd.DataFrame([dict(site=site,**tally(q)) for site,q in f.groupby('site')]);by_site.to_csv(OUT/'forecast_counts_by_site.csv',index=False)
(OUT/'counts.json').write_text(json.dumps(counts,indent=2))
(OUT/'scope_note.md').write_text('120为不同历史输入版本下的暂存合计，不是已统一输入的论文最终统计。FQ旧80策略中的16个采用修订2018输入及旧主参照8140.066；新增预报8个使用当前输入，主参照7311.9，专家灌溉次数也不同。先报告旧80与新增40分列；排除FQ的96也仅为条件一致的描述性筛选汇总，非预报因果试验。WP_ET 400/400齐全；26条PFP_N未定义均因N=0，不能填0或估算值。\n',encoding='utf-8')
print(json.dumps(counts,ensure_ascii=False));print(by_site.to_string(index=False));print('Missing PFP_N by site:',p.groupby('site').PFP_N_kg_kg.apply(lambda s:int(s.isna().sum())).to_dict());print(f[['site','expert_irrigation_events_per_year','legacy_expert_events_per_year']].drop_duplicates().to_string(index=False))
