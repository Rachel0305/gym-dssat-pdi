"""Apply the frozen 233 exploratory screening rules to the new 100K cohort."""
from pathlib import Path
import sys,json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import screen_five_site_ppo_233 as old
OUT=ROOT/'results/sylc_augmented_strategy_screen_061'
OUT.mkdir(exist_ok=False)
prior=pd.read_csv(ROOT/'results/five_site_strategy_screen_233/all_48_strategy_screen.csv')
rows=[]
for site in ['SY','LC']:
    best_y,expert_y,expert_i,expert_n,expert_events=old.conventional(site)
    anchor=prior[(prior.site==site)&(prior.regime=='historical')&(prior.seed==0)].iloc[0]
    prefix='046_06_sya' if site=='SY' else '053_03_lca'
    for seed in range(8):
        root=ROOT/f'results/sylc_external_weather_060/five_scenario_archive/{site}/seed_{seed:02d}/tables'
        events=root/f'{prefix}_five_scenario_management_events.csv'
        m=old.actions(pd.read_csv(events),'irrigation_executed_mm','nitrogen_executed_kg_ha','rl_candidate')
        table=ROOT/f'results/sylc_external_weather_060/four_scenario_v3/{site}/seed_{seed:02d}/yearly_metrics.csv'
        f=pd.read_csv(table,keep_default_na=False);f=f[f.scenario.eq(f'ppo_seed_{seed}')]
        assert len(f)==10 and set(f.year)==set(range(2014,2024))
        y=f.grain_yield_kg_ha.mean();i=f.total_irrigation_mm.mean();n=f.total_n_kg_ha.mean()
        cap=i<=1.1*expert_i and n<=1.1*expert_n
        top=y>=best_y and cap and (i<expert_i or n<expert_n)
        conserve=y>=.95*best_y and cap and (i<=.9*expert_i or n<=.9*expert_n)
        route='top_yield' if top else 'conservation' if conserve else 'none'
        operations=m['irrigation_events_per_year']<=1.5*expert_events and m['small_15mm_fraction']<=2/3
        temporal=m['distinct_action_schedules']>=2
        material=y>=anchor.yield_kg_ha or i<=.9*anchor.irrigation_mm or n<=.9*anchor.nitrogen_kg_ha
        reasons=[name for name,failed in [('yield_resource',route=='none'),('operations',not operations),('fixed_10_year_schedule',not temporal),('no_material_gain_vs_anchor',not material)] if failed]
        rows.append(dict(site=site,regime='external_weather_059',seed=seed,checkpoint=100000,yield_kg_ha=y,irrigation_mm=i,nitrogen_kg_ha=n,WP_ET=f.WP_ET_kg_m3.mean(),PFP_N=pd.to_numeric(f.PFP_N_kg_grain_per_kg_N).mean(),**m,best_conventional_yield=best_y,expert_irrigation=expert_i,expert_nitrogen=expert_n,expert_irrigation_events_per_year=expert_events,anchor_seed=0,yield_resource_route=route,operation_gate=operations,temporal_gate=temporal,anchor_materiality_gate=material,provisional_pass=route!='none' and operations and temporal and material,failure_reasons=';'.join(reasons),metrics_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),events_sha256=hashlib.sha256(events.read_bytes()).hexdigest()))
f=pd.DataFrame(rows);f.to_csv(OUT/'all_16_seed_screen.csv',index=False);f[f.provisional_pass].to_csv(OUT/'provisional_candidates.csv',index=False)
rule=json.loads((ROOT/'results/five_site_strategy_screen_233/rule_manifest.json').read_text());rule['implementation_source']='src/screen_five_site_ppo_233.py';rule['implementation_sha256']=hashlib.sha256((ROOT/'src/screen_five_site_ppo_233.py').read_bytes()).hexdigest();(OUT/'rule_manifest.json').write_text(json.dumps(rule,indent=2))
print(f[['site','seed','yield_kg_ha','irrigation_mm','nitrogen_kg_ha','provisional_pass','failure_reasons']].to_string(index=False))
