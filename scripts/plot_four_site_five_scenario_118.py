"""Read-only evidence audit and serial figures from saved 100K evaluations."""
from pathlib import Path
import ast, json, re, hashlib, traceback, argparse, time, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/four_site_five_scenario_118_20261026'
SOURCES = {}
SCENARIOS = ['null','recorded_farmer_template','dssat_auto_external_n','official_extension_expert','rl_candidate']
LABELS = dict(zip(SCENARIOS,['Null','Recorded template','DSSAT auto + external N','Official expert','PPO + perfect 7-day summary']))
COLORS = dict(zip(SCENARIOS,['#555555','#C44E52','#D8A305','#7E63B6','#2A9D55']))
STYLES = dict(zip(SCENARIOS,['-','--','-.',':','-']))
YEARS = list(range(2014,2024))
BASES = {
 'HL':('054_03_hla_lowIC_054_00_hla_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000','054_03_hla'),
 'LC':('053_03_lca_lowIC_053_00_lca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000','053_03_lca'),
 'YC':('055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000','055_03_yca'),
 'FQ':('051_03_fqa_originIC_051_00_fqa_originIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000','051_03_fqa')}

def sha(p):
    p=Path(p); key=p.relative_to(ROOT).as_posix()
    if key not in SOURCES:
        SOURCES[key]=hashlib.sha256(p.read_bytes()).hexdigest()
    return SOURCES[key]

def load_functions(path,names,replace=None):
    """Load only pure reporting functions, excluding simulator/model imports."""
    p=ROOT/path;sha(p); tree=ast.parse(p.read_text(encoding='utf-8-sig'))
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            code=ast.unparse(node)
            if replace: code=replace(code)
            exec(compile('from __future__ import annotations\n'+code,str(p),'exec'),globals())

load_functions('src/build_relaxed_success_five_scenario_daily_evidence_027_05.py',{'parse_table','parse_management_events'})
load_functions('src/calculate_five_site_wue_nue_from_summary_019_10.py',{'clean_name','parse_value','parse_summary_out'})
load_functions('src/build_sya_configured_five_scenario_daily_plots_046_06.py',{'daily_from_snapshot'})
load_functions('src/053_lca_lowIC_site_transfer/run_053_03_lca_lowIC_five_scenario_figures.py',{'add_unified_reward','plot_year','draw_grouped'},
 lambda c:c.replace('LCA{year}', '{SITE} seed {SEED} | {year}').replace('053_03_lca{year}', '{SITE}_{year}'))
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
original_save=matplotlib.figure.Figure.savefig
def save_bundle(self,path,*a,**kw):
    original_save(self,path,*a,**kw)
    if Path(path).suffix=='.png':
        original_save(self,Path(path).with_suffix('.pdf'))
        original_save(self,Path(path).with_suffix('.svg'))
matplotlib.figure.Figure.savefig=save_bundle

def parse_table(path):
    lines=path.read_text(encoding='latin-1').splitlines()
    ix=max(i for i,l in enumerate(lines) if l.startswith('@'))
    cols=lines[ix].lstrip('@').split();rows=[]
    for line in lines[ix+1:]:
        parts=line.split()
        if parts and parts[0].lstrip('-').isdigit():rows.append((parts+['']*len(cols))[:len(cols)])
    return pd.DataFrame(rows,columns=cols).apply(pd.to_numeric,errors='coerce')

def parse_management_events(path):
    text=path.read_text(encoding='latin-1').rsplit('@RUN',1)[-1]
    rows=[]
    pattern=r'^\s*\d+\s+[A-Z]{3}\s+\d+,\s+\d{4}\s+(\d+)\s+\d+\s+(-?\d+)\s+MZ\s+.*?\b(Irrigation|Fertilizer)\s+([0-9.]+)'
    for line in text.splitlines():
        h=re.search(pattern,line,re.I)
        if h:rows.append(dict(doy=int(h[1]),dap=int(h[2]),operation=h[3].lower(),amount=float(h[4])))
    return pd.DataFrame(rows,columns=['doy','dap','operation','amount'])

def jsonread(p):
    sha(p);return json.loads(p.read_text(encoding='utf-8-sig'))

def make_tables():
    manifest=ROOT/'results/four_site_forecast_116_20261025/main100k_seed_year_rows.csv';sha(manifest)
    inventory=pd.read_csv(manifest,keep_default_na=False)
    assert len(inventory)==320 and not inventory.duplicated(['site','seed','year']).any()
    all_rows=[];audits=[]
    for site,(folder,prefix) in BASES.items():
        bp=ROOT/'benchmark_results'/folder/'tables'
        dp=bp/f'{prefix}_five_scenario_daily.csv';sp=bp/f'{prefix}_five_scenario_season_summary.csv';sha(dp);sha(sp)
        frozen=pd.read_csv(dp,keep_default_na=False);summary=pd.read_csv(sp,keep_default_na=False)
        frozen=frozen[~frozen.scenario.eq('rl_candidate')].copy();summary=summary[~summary.scenario.eq('rl_candidate')].copy()
        assert len(summary)==40
        # Rebuild each frozen control from its final appended run; never merge runs.
        control_days=[];control_rows=[]
        for old in summary.to_dict('records'):
            snap=ROOT/old['snapshot_path'];year=int(old['year']);scenario=old['scenario']
            for name in ['Weather.OUT','PlantGro.OUT','SoilWat.OUT','MgmtEvent.OUT','Summary.OUT']:sha(snap/name)
            d=daily_from_snapshot(snap,year,scenario);r=parse_summary_out(snap/'Summary.OUT')[-1]
            assert abs(r['HWAM']-float(old['grain_yield_kg_ha']))<=1,(site,year,scenario,'baseline yield')
            ev=parse_management_events(snap/'MgmtEvent.OUT')
            for col,key in [('irrigation_executed_mm','IRCM'),('nitrogen_executed_kg_ha','NICM')]:
                assert abs(d[col].sum()-r[key])<=.51+len(ev)*.051,(site,year,scenario,col,d[col].sum(),r[key])
            old.update(grain_yield_kg_ha=r['HWAM'],biomass_kg_ha=d.biomass_kg_ha.iloc[-1],actual_irrigation_mm=r['IRCM'],actual_nitrogen_kg_ha=r['NICM'],etcp_mm=r['ETCP'],WP_ET_kg_m3=r['HWAM']/r['ETCP']/10,PFP_N_kg_kg=r['HWAM']/r['NICM'] if r['NICM']>0 else np.nan,unified_reward=.158*r['HWAM']-1.1*r['IRCM']-1.58*r['NICM'],event_I_mm=d.irrigation_executed_mm.sum(),event_N_kg_ha=d.nitrogen_executed_kg_ha.sum(),event_I_minus_summary_mm=d.irrigation_executed_mm.sum()-r['IRCM'],event_N_minus_summary_kg_ha=d.nitrogen_executed_kg_ha.sum()-r['NICM'])
            control_days.append(d);control_rows.append(old)
        frozen=pd.concat(control_days,ignore_index=True);summary=pd.DataFrame(control_rows)
        for seed in range(8):
            dest=OUT/site/f'seed_{seed:02d}';tab=dest/'tables';tab.mkdir(parents=True,exist_ok=True)
            days=[];srows=[];checks=[]
            for year in YEARS:
                inv=inventory.query('site==@site and seed==@seed and year==@year').iloc[0]
                run=ROOT/inv.source; snap=run/f'eval_100000_{year}_snapshot/dssat'
                expected=jsonread(run/f'eval_100000_{year}_summary.json')
                for name in ['Weather.OUT','PlantGro.OUT','SoilWat.OUT','MgmtEvent.OUT','Summary.OUT']:sha(snap/name)
                day=daily_from_snapshot(snap,year,'rl_candidate');day['site']=site;day['station_code']=site+'A'
                assert abs(day.grain_yield_kg_ha.iloc[-1]-expected['yield_kg_ha'])<=1,(site,seed,year,'yield')
                final=parse_summary_out(snap/'Summary.OUT')[-1]
                assert abs(final['HWAM']-expected['yield_kg_ha'])<=1
                assert abs(final['IRCM']-expected['I'])<1e-5 and abs(final['NICM']-expected['N'])<1e-5
                assert abs(day.irrigation_executed_mm.sum()-expected['I'])<1e-5,(site,seed,year,'MgmtEvent I')
                assert abs(day.nitrogen_executed_kg_ha.sum()-expected['N'])<1e-5,(site,seed,year,'MgmtEvent N')
                trace=run/f'eval_100000_{year}_steps.jsonl';sha(trace)
                actions=pd.DataFrame([json.loads(line) for line in trace.read_text().splitlines()])
                actions=actions.groupby('date')[['I','N']].sum()
                observed=day.set_index(day.date.dt.strftime('%Y-%m-%d'))
                for date,act in actions.iterrows():
                    if act.I or act.N:
                        assert date in observed.index,(site,seed,year,date)
                        assert abs(observed.loc[date,'irrigation_executed_mm']-act.I)<1e-5
                        assert abs(observed.loc[date,'nitrogen_executed_kg_ha']-act.N)<1e-5
                assert abs(actions.I.sum()-expected['I'])<1e-5 and abs(actions.N.sum()-expected['N'])<1e-5
                base=frozen[frozen.year.eq(year)].copy();base['date']=pd.to_datetime(base.date)
                bs=summary[summary.year.eq(year)].copy()
                for scenario in SCENARIOS[:-1]:
                    b=base[base.scenario.eq(scenario)];m=bs[bs.scenario.eq(scenario)].iloc[0]
                    assert abs(b.irrigation_executed_mm.sum()-float(m.actual_irrigation_mm))<=.51+len(b)*.051
                    assert abs(b.nitrogen_executed_kg_ha.sum()-float(m.actual_nitrogen_kg_ha))<=.51+len(b)*.051
                    weather=b[['date','rainfall_mm','tmax_c','tmin_c']].merge(day[['date','rainfall_mm','tmax_c','tmin_c']],on='date',suffixes=('_b','_p'))
                    assert len(weather)>50 and not weather.date.duplicated().any()
                    for k in ['rainfall_mm','tmax_c','tmin_c']:
                        assert np.allclose(weather[k+'_b'],weather[k+'_p'],atol=.11,rtol=0),(site,seed,year,scenario,'weather',k)
                merged=add_unified_reward(pd.concat([base,day],ignore_index=True));merged['site']=site;merged['seed']=seed
                days.append(merged)
                etcp=final['ETCP'];assert etcp>0
                row=dict(site=site,station_code=site+'A',seed=seed,year=year,scenario='rl_candidate',grain_yield_kg_ha=expected['yield_kg_ha'],biomass_kg_ha=day.biomass_kg_ha.iloc[-1],actual_irrigation_mm=final['IRCM'],actual_nitrogen_kg_ha=final['NICM'],etcp_mm=etcp,WP_ET_kg_m3=expected['yield_kg_ha']/etcp/10,PFP_N_kg_kg=expected['yield_kg_ha']/final['NICM'] if final['NICM']>0 else np.nan,unified_reward=.158*expected['yield_kg_ha']-1.1*final['IRCM']-1.58*final['NICM'],snapshot_path=snap.relative_to(ROOT).as_posix())
                for br in bs.to_dict('records'):
                    br.update(site=site,seed=seed);srows.append(br)
                srows.append(row);checks.append(dict(site=site,seed=seed,year=year,passed=True,action_dates_doses=True,summary_final_closure=True,baseline_totals=True,weather_four_controls=True))
            daily=pd.concat(days,ignore_index=True);ss=pd.DataFrame(srows)
            for c in ['grain_yield_kg_ha','actual_irrigation_mm','actual_nitrogen_kg_ha','WP_ET_kg_m3','PFP_N_kg_kg','unified_reward']:ss[c]=pd.to_numeric(ss[c],errors='coerce')
            assert len(ss)==50
            daily.to_csv(tab/'five_scenario_daily.csv',index=False)
            ss.to_csv(tab/'five_scenario_season_summary.csv',index=False)
            daily[daily.irrigation_executed_mm.gt(0)|daily.nitrogen_executed_kg_ha.gt(0)].to_csv(tab/'five_scenario_management_events.csv',index=False)
            (dest/'audit.json').write_text(json.dumps(checks,indent=2));audits+=checks;all_rows+=srows
            print(f'AUDIT {site} seed{seed}: 10 years passed',flush=True)
    total=pd.DataFrame(all_rows)
    total.to_csv(OUT/'all_sites_seed_year_scenario.csv',index=False)
    metrics=['grain_yield_kg_ha','actual_irrigation_mm','actual_nitrogen_kg_ha','WP_ET_kg_m3','PFP_N_kg_kg','unified_reward']
    for c in metrics:total[c]=pd.to_numeric(total[c],errors='coerce')
    total.groupby(['site','seed','scenario'])[metrics].mean().to_csv(OUT/'seed_scenario_means.csv')
    total.groupby(['site','year','scenario'])[metrics].agg(['mean','std','count']).to_csv(OUT/'site_year_scenario_seed_statistics.csv')
    (OUT/'audit_all.json').write_text(json.dumps(audits,indent=2))
    (OUT/'source_sha256.json').write_text(json.dumps(SOURCES,indent=2))

def render():
    global SITE,SEED
    for SITE in BASES:
        for SEED in range(8):
            dest=OUT/SITE/f'seed_{SEED:02d}';fig=dest/'figures';fig.mkdir(exist_ok=True)
            daily=pd.read_csv(dest/'tables/five_scenario_daily.csv',keep_default_na=False)
            ss=pd.read_csv(dest/'tables/five_scenario_season_summary.csv',keep_default_na=False)
            for c in ['grain_yield_kg_ha','actual_irrigation_mm','actual_nitrogen_kg_ha','WP_ET_kg_m3','PFP_N_kg_kg','unified_reward']:ss[c]=pd.to_numeric(ss[c],errors='coerce')
            for year in YEARS:
                p=fig/f'{SITE}_{year}_five_scenario_daily.png'
                if not p.with_suffix('.svg').exists():plot_year(daily[daily.year.eq(year)],year,fig)
            for name,cols in [('metrics',['grain_yield_kg_ha','WP_ET_kg_m3','PFP_N_kg_kg']),('management',['actual_irrigation_mm','actual_nitrogen_kg_ha','unified_reward'])]:
                p=fig/f'five_scenario_{name}.png'
                if not p.with_suffix('.svg').exists():draw_grouped(ss,YEARS,cols,p)
            gc.collect();assert psutil.Process().memory_info().rss<1000*1024**2
            print(f'FIGURES {SITE} seed{SEED}: 12 PNG/PDF/SVG sets',flush=True)
    sources=json.loads((OUT/'source_sha256.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in sources.items())
    counts={ext:len(list(OUT.glob(f'*/seed_*/figures/*.{ext}'))) for ext in ['png','pdf','svg']}
    assert counts==dict(png=384,pdf=384,svg=384),counts
    (OUT/'result.json').write_text(json.dumps(dict(status='completed',sites=4,seeds=32,verified_seed_years=320,scenario_summary_rows=1600,figures=counts,new_simulations=0,sources_unchanged=True),indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--render',action='store_true');args=ap.parse_args()
    try:
        if not args.render:
            OUT.mkdir(exist_ok=True)
            (OUT/'figure_contract.md').write_text('Purpose: show annual outcome and management differences between four frozen station controls and each 100K forecast PPO seed. Quantitative grid matching the reference, ten validation years (2014-2023), all eight seeds, no seed/checkpoint selection. Daily panels: weather, soil water, water/N stress, executed I/N events, grain/biomass, common-score trajectory. Summary: yield, WP_ET, PFP_N and I/N/common score. PNG previews plus editable PDF/SVG, source CSV and action audit. WP_ET PPO = saved yield/(10*same-snapshot ETCP); frozen control metrics retained. Common score = 0.158*yield - 1.1*I - 1.58*N; not training reward/economic profit. N=0 PFP_N undefined, shown missing. Four controls shared across seeds, not independent replicates. Seed-year plots are raw runs without error bars; site-year tables mean/sample SD across 8 PPO seeds, no inference from repeated controls. Perfect 7-day forecast is idealized. Source hashes before/after, no simulation or model loading.\n',encoding='utf-8')
            make_tables()
        else:render()
    except BaseException:
        if OUT.exists():(OUT/'FAILED_audit_or_render.json').write_text(json.dumps(dict(traceback=traceback.format_exc()),indent=2))
        raise
