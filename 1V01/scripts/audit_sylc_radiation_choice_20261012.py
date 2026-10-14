"""Compare frozen radiation to measured neighbour data, without replacing inputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/sylc_external_weather_20261012/radiation_choice'
def main():
    OUT.mkdir(exist_ok=False)
    gaps=pd.read_csv(ROOT/'results/sy_lc_random_weather_015/nonrain_provenance_resolution_484.csv',parse_dates=['date'])
    rows=[]; comparison=[]
    for site,station in [('SY','SYA'),('LC','LCA')]:
        records=[]
        sub=gaps[gaps.site==station]
        for year in range(2005,2014):
            base='multisite_new_cultivar_inputs_013' if site=='SY' else 'multisite_new_cultivar_inputs_013_lowIC_manual'
            file=ROOT/f'DSSAT_auto_validation/{base}/{site}/CN{site}{year%100:02d}01.WTH'
            for line in file.read_text().splitlines():
                t=line.split()
                if len(t)>=5 and t[0].isdigit() and len(t[0])==7:
                    records.append([pd.to_datetime(t[0],format='%Y%j')]+[float(v) for v in t[1:5]])
        frame=pd.DataFrame(records,columns=['date','SRAD','TMAX','TMIN','RAIN']).set_index('date').sort_index()
        assert len(frame)==3287 and not frame.index.duplicated().any()
        ext=pd.read_csv(ROOT/f'results/sylc_external_weather_20261012/{site}_daily.csv',parse_dates=['date']).set_index('date')
        mask=frame.index.isin(sub[sub.variable=='SRAD'].date)
        for year,g in frame.groupby(frame.index.year):
            for month,gg in g.groupby(g.index.month):
                rows.append(dict(site=site,year=int(year),month=int(month),days=len(gg),SRAD_std=gg.SRAD.std(),flagged_srad_days=int(gg.index.isin(sub[sub.variable=='SRAD'].date).sum())))
        if site=='SY':
            pairs=pd.DataFrame({'original':frame.SRAD,'neighbour_observed':ext.SRAD_observed.reindex(frame.index),'flagged':mask})
            pairs.to_csv(OUT/'SY_daily_radiation_pairs.csv')
            for month,g in pairs[~pairs.flagged].groupby(pairs[~pairs.flagged].index.month):
                g=g.dropna()
                comparison.append(dict(month=int(month),n=len(g),bias_neighbour_minus_original=(g.neighbour_observed-g.original).mean(),rmse=np.sqrt(((g.neighbour_observed-g.original)**2).mean()),correlation=g.original.corr(g.neighbour_observed)))
        frame.to_csv(OUT/f'{site}_frozen_training_weather.csv')
    pd.DataFrame(rows).to_csv(OUT/'original_monthly_radiation_qc.csv',index=False)
    pd.DataFrame(comparison).to_csv(OUT/'SY_measured_neighbour_overlap.csv',index=False)
    status={'user_confirmed_above_30000':'missing','rsds_required':False,'SY':'Original SRAD not adequate unchanged for WGEN; neighbour measured rsds0 covers all 160 flagged dates. Check bias before gap-only replacement.','LC':'13 provenance flags; original SRAD usable as candidate with train-only gap treatment and sensitivity review. No need to replace whole radiation series with estimated rsds.','candidate_injection':'NOT_DONE','FQ_seed7':'fresh isolated smoke running, then technical-gated formal; behavior outcome retained'}
    (OUT/'decision.json').write_text(json.dumps(status,indent=2))
    print(pd.DataFrame(comparison).round(3).to_string(index=False))
if __name__=='__main__': main()
