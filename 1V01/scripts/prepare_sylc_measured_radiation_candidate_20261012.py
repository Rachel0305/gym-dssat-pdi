"""Train-only, provenance-labelled weather candidate; never touches frozen inputs."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_dssat_cli import WeatherDay, fit_monthly_statistics, crosscheck_statistics
OUT=ROOT/'results/sylc_external_weather_20261012/measured_radiation_candidate_v2'
def alpha(v):
    v=np.asarray(v); v=v[v>0]
    if len(v)<3:return np.nan
    y=np.log(v.mean())-np.log(v).mean()
    return (8.898919+9.05995*y+.9775373*y*y)/(y*(17.79728+11.968477*y+y*y)) if y>0 else np.nan
def main():
    OUT.mkdir(exist_ok=False)
    gaps=pd.read_csv(ROOT/'results/sy_lc_random_weather_015/nonrain_provenance_resolution_484.csv',parse_dates=['date'])
    decisions={}; provenance=[]
    for site,station in [('SY','SYA'),('LC','LCA')]:
        original=pd.read_csv(ROOT/f'results/sylc_external_weather_20261012/radiation_choice/{site}_frozen_training_weather.csv',parse_dates=['date']).set_index('date')
        ext=pd.read_csv(ROOT/f'results/sylc_external_weather_20261012/{site}_daily.csv',parse_dates=['date']).set_index('date').reindex(original.index)
        candidate=original.copy()
        sources=pd.DataFrame('frozen_agricultural_station',index=original.index,columns=original.columns)
        # External rain becomes the quantitative donor; missing days use recorded
        # agricultural rain, not invented zero or monthly means.
        valid=ext.RAIN.notna()
        candidate.loc[valid,'RAIN']=ext.loc[valid,'RAIN']
        sources.loc[valid,'RAIN']='neighbour_observed_precipitation'
        sources.loc[~valid,'RAIN']='agricultural_rain_fallback_for_neighbour_missing'
        for var in ['SRAD','TMAX','TMIN']:
            dates=pd.DatetimeIndex(gaps[(gaps.site==station)&(gaps.variable==var)].date)
            if var=='SRAD' and site=='LC':
                donor=original.loc[~original.index.isin(dates),'SRAD'].groupby(lambda d:d.month).mean()
                candidate.loc[dates,var]=[donor[d.month] for d in dates]
                sources.loc[dates,var]='agricultural_training_only_month_mean_flagged'
            else:
                donor='SRAD_observed' if var=='SRAD' else var
                assert ext.loc[dates,donor].notna().all()
                candidate.loc[dates,var]=ext.loc[dates,donor]
                sources.loc[dates,var]='neighbour_measured_radiation' if var=='SRAD' else 'neighbour_observed_temperature'
        assert candidate.notna().all().all()
        assert (candidate.SRAD>=0).all() and (candidate.RAIN>=0).all() and (candidate.TMAX>=candidate.TMIN).all()
        candidate.to_csv(OUT/f'{site}_candidate_2005_2013.csv',float_format='%.6f')
        sources.to_csv(OUT/f'{site}_daily_sources.csv')
        records=[WeatherDay(d.date(),float(r.SRAD),float(r.TMAX),float(r.TMIN),float(r.RAIN)) for d,r in candidate.iterrows()]
        stats=fit_monthly_statistics(records)
        pd.DataFrame(stats).to_csv(OUT/f'{site}_cli_parameters.csv',index=False)
        crosscheck=crosscheck_statistics(records,stats)
        loo=[]
        for month,g in candidate.groupby(candidate.index.month):
            for omitted in [None]+list(range(2005,2014)):
                h=g if omitted is None else g[g.index.year!=omitted]
                raw=alpha(h.RAIN)
                loo.append(dict(month=int(month),omitted_year=omitted,wet_days=int((h.RAIN>0).sum()),wet_years=len(set(h[h.RAIN>0].index.year)),raw_alpha=raw,clamped=bool(raw>=1)))
        pd.DataFrame(loo).to_csv(OUT/f'{site}_gamma_leave_one_year_out.csv',index=False)
        january=[r for r in loo if r['month']==1]
        january_stable=all(np.isfinite(r['raw_alpha']) and r['raw_alpha']<1 for r in january)
        decisions[site]={'quantitative_qc':True,'crosscheck':crosscheck,'rain_fallback_days':int((~valid).sum()),'estimated_rsds_used':False,'january_no_clamp_full_and_leave_one_out':january_stable,'stage':'CANDIDATE_ONLY_NOT_INJECTED','remaining':['mixed_source_rain_sensitivity','radiation_gap_effect','weather_generation_and_DSSAT_smoke']}
    (OUT/'gate.json').write_text(json.dumps(decisions,indent=2,default=str))
    print(json.dumps({s:{k:v for k,v in d.items() if k!='crosscheck'}|{'crosscheck_passed':d['crosscheck']['passed']} for s,d in decisions.items()},indent=2))
if __name__=='__main__':main()
