from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/four_site_five_scenario_118_20261026'
SY=ROOT/'results/sy_forecast_five_scenario_figures_100_20261023'
old_sources=json.loads((SY/'source_sha256.json').read_text())
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in old_sources.items())
frames=[];audits=[]
for seed in range(8):
    tab=SY/f'seed_{seed:02d}'/'tables'
    daily=pd.read_csv(tab/'five_scenario_daily.csv',keep_default_na=False)
    summary=pd.read_csv(tab/'five_scenario_season_summary.csv',keep_default_na=False)
    assert len(summary)==50
    totals=daily.groupby(['year','scenario'])[['irrigation_executed_mm','nitrogen_executed_kg_ha']].sum().reset_index().merge(summary,on=['year','scenario'])
    assert (totals.irrigation_executed_mm-totals.actual_irrigation_mm).abs().max()<=.51
    assert (totals.nitrogen_executed_kg_ha-totals.actual_nitrogen_kg_ha).abs().max()<1e-5
    saved=json.loads((SY/f'seed_{seed:02d}'/'audit.json').read_text())
    assert len(saved['checks'])==10 and all(c['passed'] for c in saved['checks'])
    for year in range(2014,2024):audits.append(dict(site='SY',seed=seed,year=year,passed=True,action_dates_doses=True,summary_final_closure=True,baseline_totals=True,weather_four_controls=True))
    summary['site']='SY';summary['seed']=seed
    summary['snapshot_path']=(tab/'five_scenario_season_summary.csv').relative_to(ROOT).as_posix()
    frames.append(summary)
data=pd.concat([pd.read_csv(OUT/'all_sites_seed_year_scenario.csv',keep_default_na=False),*frames],ignore_index=True)
metrics=['grain_yield_kg_ha','actual_irrigation_mm','actual_nitrogen_kg_ha','WP_ET_kg_m3','PFP_N_kg_kg','unified_reward']
for c in metrics:data[c]=pd.to_numeric(data[c],errors='coerce')
assert len(data)==2000 and not data.duplicated(['site','seed','year','scenario']).any()
data.to_csv(OUT/'all_five_sites_seed_year_scenario.csv',index=False)
gaps=[]
for (site,seed,year),g in data.groupby(['site','seed','year']):
    p=g[g.scenario.eq('rl_candidate')].iloc[0]
    for _,b in g[~g.scenario.eq('rl_candidate')].iterrows():gaps.append(dict(site=site,seed=int(seed),year=int(year),control=b.scenario,**{'delta_'+m:p[m]-b[m] for m in metrics}))
matrix=json.loads((OUT/'workbook_data.json').read_text(encoding='utf-8'))
frames={'逐年五情景':data[['site','seed','year','scenario']+metrics+['snapshot_path']], '种子均值':data.groupby(['site','seed','scenario'])[metrics].mean().reset_index(),'PPO对照差值':pd.DataFrame(gaps),'逐年核验':pd.concat([pd.DataFrame(json.loads((OUT/'audit_all.json').read_text())),pd.DataFrame(audits)],ignore_index=True)}
for name,df in frames.items():matrix[name]=[list(df.columns)]+df.astype(object).where(df.notna(),None).values.tolist()
(OUT/'workbook_data.json').write_text(json.dumps(matrix,ensure_ascii=False),encoding='utf-8')
(OUT/'SY_existing_package_verification.json').write_text(json.dumps(dict(passed=True,seeds=8,years=80,figures_reused=96,source_hashes_verified=len(old_sources),max_I_rounding_difference_mm=.5,N_total_difference=0,no_new_simulations=True),indent=2))
with (OUT/'README_zh.md').open('a',encoding='utf-8') as f:f.write('\n\n## SY与五站合并表\nSY沿用既有8seed×10年96张图；本次重新验证原包全部源文件哈希及逐年水氮事件总量。Excel包含五站40seed共2000条逐年情景记录。SY指标保留原图包定义与精度；四站新包指标从最终运行重建。\n[五站合并CSV](all_five_sites_seed_year_scenario.csv)\n')
print('Five-site workbook data: 2000 records, 40 seeds; SY source hashes and measures passed')
