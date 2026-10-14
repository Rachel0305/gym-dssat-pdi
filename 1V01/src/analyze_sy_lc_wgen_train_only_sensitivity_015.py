"""Static monthly-moment sensitivity to train-only mean replacement.

Counterfactual values exist only in memory; no weather or CLI files are made.
"""
from pathlib import Path
import csv
import json
import math
import statistics
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'results/sy_lc_random_weather_015'
OUT=AUDIT/'wgen_readiness_review'
TARGET=OUT/'train_only_replacement_sensitivity.csv'
sys.path.insert(0,str(ROOT/'src'))
from audit_sy_lc_random_weather_source_015 import parse_wth


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def moments(values):
    return statistics.mean(values),statistics.stdev(values)


def main():
    if TARGET.exists():raise SystemExit('Refusing to overwrite sensitivity audit.')
    gaps=rows(AUDIT/'nonrain_provenance_resolution_484.csv')
    raw=rows(AUDIT/'weather_gap_details.csv')
    results=[]
    for site,short,prefix,subdir in (
        ('SYA','SY','CNSY','multisite_new_cultivar_inputs_013'),
        ('LCA','LC','CNLC','multisite_new_cultivar_inputs_013_lowIC_manual')):
        cleaned=pd.read_csv(ROOT/f'weather_clean/{site}_weather_cleaned.csv',dtype={'date':str})
        fit={}
        for var in ('SRAD','TMAX','TMIN'):
            missing={r['date'] for r in raw if r['site']==site and r['variable']==var and r['kind']=='RAW_MISSING_VALUE'}
            for month in range(1,13):
                observed=cleaned[(cleaned.year.between(2005,2013))&(cleaned.month==month)&~cleaned.date.isin(missing)][var]
                assert len(observed)>1
                fit[(var,month)]=float(observed.mean())
        selected={(r['date'],r['variable']) for r in gaps if r['site']==site}
        daily=[]
        for year in range(2005,2014):
            p=ROOT/'DSSAT_auto_validation'/subdir/short/f'{prefix}{year%100:02d}01.WTH'
            records,err=parse_wth(p,year)
            assert not err
            for day,vals in records:daily.append((day.isoformat(),day.month,vals))
        for month in range(1,13):
            month_days=[x for x in daily if x[1]==month]
            for var in ('SRAD','TMAX','TMIN'):
                for group in ('all','wet','dry'):
                    if var=='TMIN' and group!='all':continue
                    group_days=[x for x in month_days if group=='all' or (x[2]['RAIN']>0)==(group=='wet')]
                    before=[x[2][var] for x in group_days]
                    after=[fit[(var,month)] if (x[0],var) in selected else x[2][var] for x in group_days]
                    assert len(before)>1
                    m0,s0=moments(before);m1,s1=moments(after)
                    n=sum((x[0],var) in selected for x in group_days)
                    results.append(dict(site=site,month=month,variable=var,group=group,days=len(group_days),replaced_cells=n,
                        mean_current=m0,mean_train_only_counterfactual=m1,absolute_mean_change=abs(m1-m0),
                        sample_sd_current=s0,sample_sd_train_only_counterfactual=s1,
                        relative_sd_change=abs(s1-s0)/s0 if s0 else None))
    with TARGET.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    summary={}
    for site in ('SYA','LCA'):
        hits=[r for r in results if r['site']==site and r['replaced_cells']]
        by_mean=max(hits,key=lambda r:r['absolute_mean_change'])
        by_sd=max(hits,key=lambda r:r['relative_sd_change'])
        summary[site]={'max_abs_monthly_group_mean_change':{k:by_mean[k] for k in ('month','variable','group','replaced_cells','absolute_mean_change')},
                       'max_rel_monthly_group_sd_change':{k:by_sd[k] for k in ('month','variable','group','replaced_cells','relative_sd_change')}}
    (OUT/'train_only_replacement_sensitivity_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
