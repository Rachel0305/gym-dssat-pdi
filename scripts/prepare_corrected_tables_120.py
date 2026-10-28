from pathlib import Path
import pandas as pd,json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/forecast_five_scenario_120_corrected_HL_20261027'
d=pd.read_csv(OUT/'all_five_sites_seed_year_scenario.csv',keep_default_na=False)
cols=['grain_yield_kg_ha','actual_irrigation_mm','actual_nitrogen_kg_ha','WP_ET_kg_m3','PFP_N_kg_kg','unified_reward']
d[cols]=d[cols].apply(pd.to_numeric,errors='coerce')
hl=d[d.site.eq('HL')&d.scenario.eq('dssat_auto')]
assert len(hl)==80 and hl.actual_nitrogen_kg_ha.eq(0).all() and hl.PFP_N_kg_kg.isna().all()
assert not (d.site.eq('HL')&d.scenario.eq('dssat_auto_external_n')).any()
gaps=[]
for (site,seed,year),g in d.groupby(['site','seed','year']):
    p=g[g.scenario.eq('rl_candidate')].iloc[0]
    for _,b in g[~g.scenario.eq('rl_candidate')].iterrows():gaps.append(dict(site=site,seed=int(seed),year=int(year),control=b.scenario,**{'delta_'+c:p[c]-b[c] for c in cols}))
pd.DataFrame(gaps).to_csv(OUT/'ppo_minus_each_control.csv',index=False)
tables={'逐年五情景':d[['site','seed','year','scenario']+cols+['snapshot_path']], '种子均值':d.groupby(['site','seed','scenario'])[cols].mean().reset_index(),'PPO对照差值':pd.DataFrame(gaps),'HL原生auto核验':pd.DataFrame(json.loads((OUT/'HL_native_auto_audit.json').read_text()))}
(OUT/'workbook_data.json').write_text(json.dumps({k:[list(v.columns)]+v.astype(object).where(v.notna(),None).values.tolist() for k,v in tables.items()},ensure_ascii=False),encoding='utf-8')
print('Corrected tables: HL native auto 80 records, all N=0, PFP_N blank; 2000 total rows')
