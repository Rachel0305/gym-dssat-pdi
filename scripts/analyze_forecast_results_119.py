from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/forecast_analysis_119_20261027'
OUT.mkdir(exist_ok=True)
SOURCE=ROOT/'results/four_site_five_scenario_118_20261026/all_five_sites_seed_year_scenario.csv'
d=pd.read_csv(SOURCE,keep_default_na=False)
M=['grain_yield_kg_ha','actual_irrigation_mm','actual_nitrogen_kg_ha','WP_ET_kg_m3','PFP_N_kg_kg']
d[M]=d[M].apply(pd.to_numeric,errors='coerce')
assert len(d)==2000 and not d.duplicated(['site','seed','year','scenario']).any()
assert set(d.year)==set(range(2014,2024))
means=d.groupby(['site','scenario'])[M].mean().reset_index()
means.to_csv(OUT/'five_scenario_site_means.csv',index=False)
seedmeans=d[d.scenario.eq('rl_candidate')].groupby(['site','seed'])[M].mean().reset_index()
seedmeans.to_csv(OUT/'forecast_seed_means.csv',index=False)
f=ROOT/'results/four_site_forecast_116_20261025/main100k_seed_year_rows.csv'
paired=pd.read_csv(f)
assert len(paired)==320 and not paired.duplicated(['site','seed','year']).any()
rows=[]
for site,g in paired.groupby('site'):
    row={'site':site}
    for a,b,n in [('forecast_yield_kg_ha','baseline_replay_yield_kg_ha','yield'),('forecast_I','baseline_replay_I','I'),('forecast_N','baseline_replay_N','N'),('forecast_PFP_N','baseline_replay_PFP_N','PFP_N')]:
        row[n+'_forecast']=g[a].mean();row[n+'_baseline']=g[b].mean();row[n+'_delta']=g[a].mean()-g[b].mean();row[n+'_pct']=(g[a].mean()/g[b].mean()-1)*100
    seeds=g.groupby('seed').mean(numeric_only=True)
    row['seed_yield_higher_count']=int((seeds.delta_yield_kg_ha>0).sum())
    row['yield_seed_sd']=seeds.forecast_yield_kg_ha.std();row['baseline_yield_seed_sd']=seeds.baseline_replay_yield_kg_ha.std()
    row['seed_mean_yield_min']=seeds.forecast_yield_kg_ha.min();row['seed_mean_yield_max']=seeds.forecast_yield_kg_ha.max()
    row['seed_pareto_count']=int(((seeds.delta_yield_kg_ha>=0)&(seeds.delta_I<=0)&(seeds.delta_N<=0)).sum())
    row['year_yield_higher_count']=int((g.groupby('year').delta_yield_kg_ha.mean()>0).sum())
    rows.append(row)
direct=pd.DataFrame(rows);direct.to_csv(OUT/'forecast_vs_no_forecast.csv',index=False)
paired.groupby(['site','year'])[['delta_yield_kg_ha','delta_I','delta_N']].mean().to_csv(OUT/'yearly_no_forecast_deltas.csv')
controls=[]
for site,g in d.groupby('site'):
    p=g[g.scenario.eq('rl_candidate')]
    for scenario,b in g[~g.scenario.eq('rl_candidate')].groupby('scenario'):
        row=dict(site=site,control=scenario)
        for m in M:row['delta_'+m]=p[m].mean()-b[m].mean()
        row['yield_pct']=(p.grain_yield_kg_ha.mean()/b.grain_yield_kg_ha.mean()-1)*100
        controls.append(row)
pd.DataFrame(controls).to_csv(OUT/'forecast_vs_four_controls.csv',index=False)
# Annual management variability within each trained seed; descriptive, no causal claim.
variability=d[d.scenario.eq('rl_candidate')].groupby(['site','seed']).agg(yield_mean=('grain_yield_kg_ha','mean'),I_mean=('actual_irrigation_mm','mean'),N_mean=('actual_nitrogen_kg_ha','mean'),I_unique=('actual_irrigation_mm','nunique'),N_unique=('actual_nitrogen_kg_ha','nunique'),I_min=('actual_irrigation_mm','min'),I_max=('actual_irrigation_mm','max'),N_min=('actual_nitrogen_kg_ha','min'),N_max=('actual_nitrogen_kg_ha','max'))
variability.to_csv(OUT/'management_seed_variability.csv')
(OUT/'provenance.json').write_text(json.dumps(dict(sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE,f]},years=list(range(2014,2024)),training_seeds_per_site=8,annual_observations_per_site=80,baseline_controls_shared=True,seed_numbers_not_strict_randomized_pairs=True,metrics='Equal weight across eight seeds and ten common years; PFP_N annual ratios averaged; undefined N=0 remains missing; descriptive only; SY direct comparison retained from audited 097 report',new_simulations=0),indent=2))
print('DIRECT\n'+direct.round(3).to_string(index=False))
print('FIVE SCENARIO\n'+means.round(3).to_string(index=False))
print('SEED VARIABILITY\n'+variability.round(2).to_string())
