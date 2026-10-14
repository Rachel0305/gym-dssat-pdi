"""Ten independent full-year WGEN seeds, descriptive QC versus fitting weather."""
from pathlib import Path
import sys,json,os,tempfile
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_sylc_wgen_runtime_smoke_054 as runtime
OUT=ROOT/'results/sylc_external_weather_055'
def spells(flags):
    sizes=[];run=0
    for v in flags:
        if v:run+=1
        elif run:sizes.append(run);run=0
    if run:sizes.append(run)
    return sizes
def metrics(frame,group):
    output=[]
    for year,g in frame.groupby(frame.index.year):
        wet=g.RAIN>0
        dry_spells=spells(~wet);wet_spells=spells(wet)
        output.append({'group':group,'year':int(year),'rain_mm':float(g.RAIN.sum()),'wet_days':int(wet.sum()),'longest_dry_spell':max(dry_spells,default=0),'longest_wet_spell':max(wet_spells,default=0),'SRAD_TMAX_corr':float(g.SRAD.corr(g.TMAX)),'TMAX_TMIN_corr':float(g.TMAX.corr(g.TMIN)),**{v+'_'+stat:float(getattr(g[v],stat)()) for v in ['SRAD','TMAX','TMIN'] for stat in ['mean','std','min','max']}})
    return output
def main(site='SY'):
    prefix='' if site=='SY' else 'LC_'
    gate=json.loads((OUT/('path_gate.json' if site=='SY' else 'LC_path_gate_attempt003.json')).read_text());assert gate['full_year_path_pass']
    runtime.OUT=OUT;tmp=OUT/'tmp';os.environ['TMPDIR']=str(tmp);tempfile.tempdir=str(tmp)
    results=[gate['runs'][0],gate['runs'][2]]
    for seed in range(1003,1011):
        r=runtime.run_one(site,f'seed{seed}',seed,True);results.append(r)
        (OUT/(prefix+'ensemble_progress.json')).write_text(json.dumps(results,indent=2))
        if r['status']!='PASS':
            (OUT/(prefix+'ensemble_gate.json')).write_text(json.dumps({'status':'STOPPED_WEATHER_QC_FAILED','failed_seed':seed,'completed_seeds':len(results)-1,'PPO_smoke_allowed':False,'formal_training_allowed':False},indent=2))
            return
    fitpath=ROOT/('results/sylc_external_weather_20261012/measured_radiation_candidate_v2/SY_candidate_2005_2013.csv' if site=='SY' else 'results/sylc_external_weather_054/LC_v2/candidate_2000_2013.csv')
    fit=pd.read_csv(fitpath,parse_dates=['date']).set_index('date')
    annual=metrics(fit,'fit'); monthly=[];extremes=[]
    for item in results:
        tag=item['tag'];seed=item['weather_seed']
        f=pd.read_csv(OUT/f'{site}/runtime/{tag}/weather.csv')
        # Canonical helper CSV uses DATE/DOY plus weather columns.
        f.index=pd.to_datetime(f.DATE)
        annual.extend([r|{'weather_seed':seed} for r in metrics(f,'generated')])
        for month,g in f.groupby(f.index.month):
            reference=fit[fit.index.month==month]
            monthly.append({'weather_seed':seed,'month':int(month),'generated_rain_mm':float(g.RAIN.sum()),'fit_mean_month_rain_mm':float(reference.groupby(reference.index.year).RAIN.sum().mean()),'generated_wet_days':int((g.RAIN>0).sum()),'fit_mean_month_wet_days':float(reference.RAIN.gt(0).groupby(reference.index.year).sum().mean()),**{v+'_'+s:float(getattr(g[v],s)()) for v in ['SRAD','TMAX','TMIN'] for s in ['mean','std','min','max']}})
        for v in ['RAIN','SRAD','TMAX','TMIN']:
            for q in [0,.01,.05,.5,.95,.99,1]:
                extremes.append({'weather_seed':seed,'variable':v,'quantile':q,'generated':float(f[v].quantile(q)),'fitting':float(fit[v].quantile(q))})
    pd.DataFrame(annual).to_csv(OUT/(prefix+'annual_weather_comparison.csv'),index=False)
    pd.DataFrame(monthly).to_csv(OUT/(prefix+'monthly_weather_comparison.csv'),index=False)
    pd.DataFrame(extremes).to_csv(OUT/(prefix+'weather_quantile_comparison.csv'),index=False)
    summary={'status':'PASS_FULL_YEAR_ENGINEERING_AND_PHYSICAL_SCREEN','independent_weather_seeds':list(range(1001,1011)),'days_per_seed':365,'generated_days':3650,'fit_days':len(fit),'climate_representativeness':'DESCRIPTIVE_REVIEW_REQUIRED; ten samples do not establish distributional equivalence','PPO_smoke_allowed':False,'formal_training_allowed':False,'crop_management_output_use':'DIAGNOSTIC_ONLY'}
    (OUT/(prefix+'ensemble_gate.json')).write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--site',choices=['SY','LC'],default='SY');main(parser.parse_args().site)
