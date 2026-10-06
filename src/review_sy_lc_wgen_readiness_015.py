"""Read-only WGEN input-method audit; computes diagnostics, never CLI/WGEN.

Uses frozen 2005-2013 WTH and existing 484-cell provenance table. Writes only
new review artifacts, refusing to overwrite an earlier review.
"""
from collections import Counter
from datetime import timedelta
from pathlib import Path
import csv
import hashlib
import json
import statistics
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'results/sy_lc_random_weather_015'
OUT = AUDIT / 'wgen_readiness_review'
sys.path.insert(0, str(ROOT/'src'))
from audit_sy_lc_random_weather_source_015 import parse_wth


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def std(values):
    return statistics.stdev(values) if len(values)>1 else None


def longest_run(dates):
    ordered=sorted(dates)
    longest=0;current=0;previous=None
    for day in ordered:
        current=current+1 if previous is not None and day-previous==timedelta(days=1) else 1
        longest=max(longest,current);previous=day
    return longest


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit('Refusing to overwrite previous readiness review.')
    hashes=json.loads((AUDIT/'input_sha256.json').read_text(encoding='utf-8'))
    for name,h in hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h, name
    source=read_rows(AUDIT/'nonrain_provenance_resolution_484.csv')
    assert len(source)==484
    flags={(r['site'],r['date'],r['variable']) for r in source}
    monthly=[]; annual=[]; conditional=[]; leakage=[]; summary={}
    for site,short,prefix,subdir in (
        ('SYA','SY','CNSY','multisite_new_cultivar_inputs_013'),
        ('LCA','LC','CNLC','multisite_new_cultivar_inputs_013_lowIC_manual'),
    ):
        daily=[]
        for year in range(2005,2014):
            p=ROOT/'DSSAT_auto_validation'/subdir/short/f'{prefix}{year%100:02d}01.WTH'
            parsed,errors=parse_wth(p,year)
            assert not errors
            for d, vals in parsed:
                daily.append({'date':d,'day':d.isoformat(),'year':year,'month':d.month,'wet':vals['RAIN']>0,**vals})
        assert len(daily)==3287
        sr=[r for r in source if r['site']==site]
        affected={r['date'] for r in sr}
        affected_dates={r['date'] for r in sr}
        by_year=Counter(int(r['date'][:4]) for r in sr)
        for year in range(2005,2014):
            year_days=[x for x in daily if x['year']==year]
            flagged_dates={x['date'] for x in year_days if x['day'] in affected_dates}
            annual.append(dict(site=site,year=year,days=len(year_days),flagged_cells=by_year[year],flagged_days=len(flagged_dates),
                               longest_flagged_day_run=longest_run(flagged_dates),
                               srad_flagged=sum((site,x['day'],'SRAD') in flags for x in year_days),
                               tmax_flagged=sum((site,x['day'],'TMAX') in flags for x in year_days),
                               tmin_flagged=sum((site,x['day'],'TMIN') in flags for x in year_days)))
        for month in range(1,13):
            days=[x for x in daily if x['month']==month]
            wet=[x for x in days if x['wet']];dry=[x for x in days if not x['wet']]
            row=dict(site=site,month=month,days=len(days),wet_days=len(wet),dry_days=len(dry),
                     flagged_days=sum(x['day'] in affected_dates for x in days))
            for var in ('SRAD','TMAX','TMIN'):
                row[var.lower()+'_filled_cells']=sum((site,x['day'],var) in flags for x in days)
            monthly.append(row)
            for var in ('SRAD','TMAX','TMIN'):
                for group,label in ((days,'all'),(wet,'wet'),(dry,'dry')):
                    if var=='TMIN' and label!='all':continue
                    unflagged=[x for x in group if (site,x['day'],var) not in flags]
                    allv=[x[var] for x in group];unv=[x[var] for x in unflagged]
                    full_sd=std(allv);un_sd=std(unv)
                    conditional.append(dict(site=site,month=month,variable=var,wet_dry_group=label,
                        n_all=len(allv),n_without_flagged=len(unv),flagged=len(allv)-len(unv),
                        mean_all=statistics.mean(allv) if allv else None,
                        mean_without_flagged=statistics.mean(unv) if unv else None,
                        sample_sd_all=full_sd,sample_sd_without_flagged=un_sd,
                        sd_relative_change_if_excluded=abs(full_sd-un_sd)/un_sd if full_sd is not None and un_sd else None))
        cleaned=pd.read_csv(ROOT/f'weather_clean/{site}_weather_cleaned.csv',dtype={'date':str})
        raw_gaps=read_rows(AUDIT/'weather_gap_details.csv')
        for var in ('SRAD','TMAX','TMIN'):
            raw_missing={r['date'] for r in raw_gaps if r['site']==site and r['variable']==var and r['kind']=='RAW_MISSING_VALUE'}
            for month in range(1,13):
                selected=cleaned[(cleaned.month==month)&~cleaned.date.isin(raw_missing)]
                train=selected[selected.year.between(2005,2013)]
                full_mean=float(selected[var].mean())
                train_mean=float(train[var].mean())
                target=sum(x['site']==site and x['variable']==var and int(x['date'][5:7])==month for x in source)
                leakage.append(dict(site=site,month=month,variable=var,flagged_cells=target,
                    full_period_month_mean=full_mean,train_only_month_mean=train_mean,
                    difference_full_minus_train=full_mean-train_mean,
                    train_observed_count=len(train),full_period_observed_count=len(selected)))
        relevant=[r for r in conditional if r['site']==site and r['wet_dry_group'] in ('wet','dry') and r['flagged']>0]
        changed=[r['sd_relative_change_if_excluded'] for r in relevant if r['sd_relative_change_if_excluded'] is not None]
        hasflags=[r for r in leakage if r['site']==site and r['flagged_cells']>0]
        summary[site]={
            'train_days':len(daily), 'rain_positive_days':sum(x['wet'] for x in daily),
            'flagged_nonrain_cells':len(sr), 'flagged_days':len(affected),
            'flagged_fraction_of_three_weather_channels':len(sr)/(len(daily)*3),
            'maximum_one_year_flagged_cells':max(by_year.values()),
            'maximum_one_year_flagged_cells_year':max(by_year,key=by_year.get),
            'longest_flagged_day_run':max(x['longest_flagged_day_run'] for x in annual if x['site']==site),
            'minimum_wet_days_in_any_calendar_month':min(x['wet_days'] for x in monthly if x['site']==site),
            'minimum_dry_days_in_any_calendar_month':min(x['dry_days'] for x in monthly if x['site']==site),
            'maximum_conditional_sd_relative_change_if_flagged_excluded':max(changed) if changed else 0,
            'maximum_abs_full_vs_train_month_mean_difference_at_flagged_months':max(abs(x['difference_full_minus_train']) for x in hasflags),
        }
    OUT.mkdir(parents=True,exist_ok=False)
    save_csv(OUT/'annual_fill_distribution.csv',annual)
    save_csv(OUT/'monthly_coverage.csv',monthly)
    save_csv(OUT/'conditional_moment_sensitivity.csv',conditional)
    save_csv(OUT/'full_period_vs_train_only_fill_mean.csv',leakage)
    (OUT/'diagnostic_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
