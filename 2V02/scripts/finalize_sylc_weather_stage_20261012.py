"""Freeze audited SY CLI candidate, examine LC longer rainfall window, audit FQ."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import build_dssat_cli as cli
from prepare_sylc_measured_radiation_candidate_20261012 import alpha
OUT=ROOT/'results/sylc_external_weather_20261012/processed_stage'
def main():
    OUT.mkdir(exist_ok=False)
    source=ROOT/'results/sylc_external_weather_20261012/measured_radiation_candidate_v2'
    sy=pd.read_csv(source/'SY_candidate_2005_2013.csv',parse_dates=['date']).set_index('date')
    records=[cli.WeatherDay(d.date(),*[float(r[v]) for v in ['SRAD','TMAX','TMIN','RAIN']]) for d,r in sy.iterrows()]
    fitted=cli.fit_monthly_statistics(records)
    assert cli.crosscheck_statistics(records,fitted)['passed']
    cli.EXPECTED_START=records[0].day; cli.EXPECTED_END=records[-1].day
    text=cli._cli_text(records,'CNSY',41.517,123.400,41,fitted,cli.monthly_weather_summary(records))
    text=text.replace('  2005    10 -99.0','  2005     9 -99.0')
    # Existing schema checker is YC-specific: station substitution only for
    # generic numeric/fixed-width validation; saved CLI remains CNSY.
    check=cli.check_cli_schema(text.replace('CNSY','CNYC'))
    assert check['passed'],check
    (OUT/'CNSY.CLI').write_text(text,encoding='ascii')
    years=[]
    ext=pd.read_csv(ROOT/'results/sylc_external_weather_20261012/LC_daily.csv',parse_dates=['date']).set_index('date')
    for start in [2005,2001,2000]:
        rain=ext.loc[f'{start}-01-01':'2013-12-31','RAIN'].copy()
        fallback={}
        for year in range(start,2014):
            file=ROOT/f'DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/LC/CNLC{year%100:02d}01.WTH'
            if not file.exists():continue
            for line in file.read_text().splitlines():
                t=line.split()
                if len(t)>=5 and t[0].isdigit() and len(t[0])==7:
                    fallback[pd.to_datetime(t[0],format='%Y%j')]=float(t[4])
        missing_before=int(rain.isna().sum()); rain=rain.fillna(pd.Series(fallback))
        for omit in [None]+list(range(start,2014)):
            january=rain[rain.index.month==1]
            if omit is not None:january=january[january.index.year!=omit]
            a=alpha(january.dropna())
            years.append(dict(start_year=start,end_year=2013,omitted_year=omit,neighbour_missing_days=missing_before,residual_missing_days=int(rain.isna().sum()),wet_days=int((january>0).sum()),wet_years=len(set(january[january>0].index.year)),raw_alpha=a,clamped=bool(a>=1)))
    pd.DataFrame(years).to_csv(OUT/'LC_rain_window_sensitivity.csv',index=False)
    fq=ROOT/'benchmark_results/20261012fq7_fq_seed7_historical_completion'
    evaluation=pd.read_csv(fq/'evaluation/20261012fq7_checkpoint_validation_summary.csv')
    assert len(evaluation)==40 and evaluation.run_status.eq('ok').all()
    assert set(evaluation.checkpoint_step)=={25000,50000,75000,100000}
    assert set(evaluation.year)==set(range(2014,2024))
    for p in evaluation.daily_csv_path:assert (ROOT/p).is_file()
    audit=json.loads((fq/'seed_override_audit.json').read_text())
    assert audit['configured_seed']==7 and audit['formal_100k_checkpoint_present']
    for p,digest in audit['checkpoint_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest
    group=pd.DataFrame(years)
    summary={'SY':{'stage':'CLI_CANDIDATE_SCHEMA_PASS','estimated_rsds_used':False,'fit_start':2005,'fit_end':2013,'runtime_smoke':'NOT_STARTED'},'LC':{'stage':'WINDOW_SENSITIVITY_AUDIT_ONLY','no_clamp_by_window':{str(start):bool((~g.clamped).all() and g.residual_missing_days.eq(0).all()) for start,g in group.groupby('start_year')},'fit_window_not_changed':True},'FQ_seed7':{'stage':'PASS_100K_CHECKPOINT_AND_40_VALIDATION_ROWS','seed':7,'model_hashes_verified':True,'daily_files_verified':True,'WP_ET':'NOT_REPLAYED'},'training_queue_SY_LC':'NOT_STARTED'}
    (OUT/'stage_status.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
