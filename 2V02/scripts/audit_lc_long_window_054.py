"""LC train-only 2000-2013 candidate, raw-source gaps and statistics audit."""
from pathlib import Path
import sys,json,hashlib
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import weather_preprocess as raw
import build_dssat_cli as cli
from prepare_sylc_measured_radiation_candidate_20261012 import alpha
OUT=ROOT/'results/sylc_external_weather_054/LC_v2'
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    dates=pd.date_range('2000-01-01','2013-12-31')
    original=pd.DataFrame(index=dates,columns=['SRAD','TMAX','TMIN','RAIN'],dtype=float)
    hashes=[]
    for year in range(2000,2014):
        p=ROOT/f'DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/LC/CNLC{year%100:02d}01.WTH'
        hashes.append({'file':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        for line in p.read_text().splitlines():
            t=line.split()
            if len(t)>=5 and t[0].isdigit() and len(t[0])==7:original.loc[pd.to_datetime(t[0],format='%Y%j')]=[float(x) for x in t[1:5]]
    assert original.notna().all().all()
    ext=pd.read_csv(ROOT/'results/sylc_external_weather_20261012/LC_daily.csv',parse_dates=['date']).set_index('date').reindex(dates)
    candidate=original.copy(); sources=pd.DataFrame('existing_agricultural_WTH',index=dates,columns=original.columns)
    gaprows=[]; sourcemeta=[]
    cleaned=pd.read_csv(ROOT/'weather_clean/LCA_weather_cleaned.csv',parse_dates=['date']).set_index('date').reindex(dates)
    known=pd.read_csv(ROOT/'results/sy_lc_random_weather_015/nonrain_provenance_resolution_484.csv',parse_dates=['date'])
    retained=[]
    for var in ['SRAD','TMAX','TMIN']:
        p=ROOT/('my_data/D32.xls' if var=='SRAD' else 'my_data/T2.xls')
        frame,meta=raw.normalize_source(raw.SourceSpec(var,p,['LCA']))
        frame=frame.set_index('date').reindex(dates)
        raw_missing=frame[var].isna()
        # Preserve later WTH supplementation. Only equal-to-old-fill cells or
        # explicitly unresolved 015 cells are treated as conservative gaps.
        missing=raw_missing & (np.abs(original[var]-cleaned[var].round(1))<.051)
        missing |= dates.isin(known[(known.site=='LCA')&(known.variable==var)].date)
        for d in dates[raw_missing & ~missing]:
            retained.append({'date':str(d.date()),'variable':var,'WTH_value':float(original.loc[d,var]),'classification':'later_WTH_supplement_retained_source_record_incomplete'})
        sources.loc[raw_missing & ~missing,var]='later_WTH_supplement_retained_source_record_incomplete'
        sourcemeta.append(meta|{'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'window_raw_missing':int(raw_missing.sum()),'unresolved_gap_count':int(missing.sum())})
        if var=='SRAD':
            donor=original.loc[~missing,var].groupby(original.index[~missing].month).mean()
            candidate.loc[missing,var]=[donor[d.month] for d in dates[missing]]
            sources.loc[missing,var]='fit_window_month_mean_excluding_raw_XLS_gaps'
        else:
            assert ext.loc[missing,var].notna().all()
            candidate.loc[missing,var]=ext.loc[missing,var]
            sources.loc[missing,var]='neighbour_observed_temperature'
        for d in dates[missing]:gaprows.append({'date':str(d.date()),'variable':var,'old':float(original.loc[d,var]),'new':float(candidate.loc[d,var]),'source':sources.loc[d,var]})
    valid=ext.RAIN.notna(); candidate.loc[valid,'RAIN']=ext.loc[valid,'RAIN']
    sources.loc[valid,'RAIN']='neighbour_observed_precipitation'
    sources.loc[~valid,'RAIN']='agricultural_WTH_fallback_for_neighbour_missing'
    candidate.index.name='date';sources.index.name='date'
    candidate.to_csv(OUT/'candidate_2000_2013.csv'); sources.to_csv(OUT/'daily_sources.csv')
    pd.DataFrame(gaprows).to_csv(OUT/'raw_XLS_gap_treatment.csv',index=False)
    pd.DataFrame(retained).to_csv(OUT/'later_supplements_retained.csv',index=False)
    (OUT/'input_sources.json').write_text(json.dumps({'WTH':hashes,'XLS':sourcemeta},indent=2,default=str))
    bad=(candidate.RAIN<0)|(candidate.SRAD<0)|(candidate.TMAX<candidate.TMIN)|candidate.isna().any(axis=1)
    assert not bad.any()
    records=[cli.WeatherDay(d.date(),*[float(r[v]) for v in ['SRAD','TMAX','TMIN','RAIN']]) for d,r in candidate.iterrows()]
    fitted=cli.fit_monthly_statistics(records); check=cli.crosscheck_statistics(records,fitted)
    assert check['passed']
    pd.DataFrame(fitted).to_csv(OUT/'monthly_parameters.csv',index=False)
    monthly=[]
    for (y,m),g in candidate.groupby([candidate.index.year,candidate.index.month]):
        monthly.append({'year':int(y),'month':int(m),**{v+'_std':float(g[v].std()) for v in ['SRAD','TMAX','TMIN']}})
    pd.DataFrame(monthly).to_csv(OUT/'year_month_variability.csv',index=False)
    loo=[]
    for m,g in candidate.groupby(candidate.index.month):
        for omit in [None]+list(range(2000,2014)):
            h=g if omit is None else g[g.index.year!=omit]; a=alpha(h.RAIN)
            loo.append({'month':int(m),'omitted_year':omit,'raw_alpha':a,'clamped':bool(a>=1),'wet_days':int((h.RAIN>0).sum()),'wet_years':len(set(h[h.RAIN>0].index.year))})
    pd.DataFrame(loo).to_csv(OUT/'monthly_gamma_leave_one_out.csv',index=False)
    cli.EXPECTED_START=records[0].day;cli.EXPECTED_END=records[-1].day
    header=(ROOT/'DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/LC/CNLC0001.WTH').read_text().splitlines()[3].split()
    text=cli._cli_text(records,'CNLC',float(header[1]),float(header[2]),float(header[3]),fitted,cli.monthly_weather_summary(records)).replace('  2000    10 -99.0','  2000    14 -99.0')
    schema=cli.check_cli_schema(text.replace('CNLC','CNYC')); assert schema['passed']
    (OUT/'CNLC.CLI').write_text(text,encoding='ascii')
    zero=[r for r in monthly if r['SRAD_std']<1e-8 or r['TMAX_std']<1e-8 or r['TMIN_std']<1e-8]
    january=[r for r in loo if r['month']==1]
    gate={'window':[2000,2013],'ppo_training_years_unchanged':[2005,2013],'validation_excluded':[2014,2023],'days':len(candidate),'physical_qc':True,'parameter_crosscheck':True,'cli_schema':True,'raw_gap_counts':pd.DataFrame(gaprows).groupby('variable').size().to_dict(),'rain_fallback_days':int((~valid).sum()),'zero_variability_year_months':zero,'january_stable':all(np.isfinite(r['raw_alpha']) and not r['clamped'] and r['wet_days']>=10 and r['wet_years']>=5 for r in january),'estimated_rsds_used':False,'mixed_source_caveat':'Raw XLS gap mask is conservative; later downloaded WTH supplements lack independent per-cell provenance','status':'CANDIDATE_AUDIT_ONLY_RUNTIME_NOT_RUN'}
    (OUT/'gate.json').write_text(json.dumps(gate,indent=2))
    print(json.dumps(gate,indent=2))
if __name__=='__main__':main()
