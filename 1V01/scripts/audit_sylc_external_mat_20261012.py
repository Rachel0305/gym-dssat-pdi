"""Read-only source audit; new derived outputs only, one MAT at a time."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/external/SYLC_气象站数据'
OUT = ROOT / 'results/sylc_external_weather_20261012'

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    inventory, yearly, metadata, flags = [], [], [], []
    for site, station in [('SY', 54342), ('LC', 53698)]:
        series = []
        for prefix, names, scale in [
            ('prec', ['RAIN'], .1), ('tems', ['TAVG','TMAX','TMIN'], .1),
            ('rsds', ['SRAD'], 1.), ('rhus', ['RHUM'], 1.),
            ('wins', ['WIND'], .1), ('ssds', ['SUNH'], .1)]:
            path = SOURCE / f'{prefix}_{station}.mat'
            raw = loadmat(path)
            keys = [k for k in raw if not k.startswith('_')]
            assert len(keys) == 1
            a = raw[keys[0]].astype(float)
            assert a.shape[1] == 7 + len(names)
            dates = pd.to_datetime(pd.DataFrame(a[:,4:7].astype(int), columns=['year','month','day']))
            assert not dates.duplicated().any(), path
            assert np.all(a[:,0] == station)
            values = a[:,7:].copy()
            invalid = (~np.isfinite(values)) | (np.abs(values) >= 30000)
            for r,c in np.argwhere(invalid):
                flags.append(dict(site=site,date=str(dates.iloc[r].date()),variable=names[c],raw_value=values[r,c],source=path.name))
            values[invalid] = np.nan
            series.append(pd.DataFrame(values*scale, index=pd.DatetimeIndex(dates), columns=names))
            inventory.append(dict(site=site,file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(a),start=str(dates.min().date()),end=str(dates.max().date()),sentinel_cells=int(invalid.sum()),scale=scale))
            m = pd.DataFrame(a[:,:4], columns=['station','latitude_degmin','longitude_degmin','elevation_decimetres'])
            m['date'] = dates
            for _,g in m.groupby(list(m.columns[:4]), sort=False):
                metadata.append(dict(site=site,source=prefix,**g.iloc[0,:4].to_dict(),start=str(g.date.min().date()),end=str(g.date.max().date()),days=len(g)))
        frame = pd.concat(series, axis=1).sort_index().reindex(pd.date_range('2000-01-01','2018-12-31'))
        frame.index.name='date'
        frame['SRAD_source']='estimated_rsds'
        if site == 'SY':
            path=SOURCE/'rsds0_54342.mat'
            a=loadmat(path)['rsds0'].astype(float)
            dates=pd.to_datetime(pd.DataFrame(a[:,4:7].astype(int),columns=['year','month','day']))
            assert not dates.duplicated().any()
            v=a[:,7].copy(); invalid=(~np.isfinite(v)) | (np.abs(v)>=30000); v[invalid]=np.nan
            frame['SRAD_observed']=pd.Series(v*.01,index=pd.DatetimeIndex(dates))
            inventory.append(dict(site=site,file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(a),start=str(dates.min().date()),end=str(dates.max().date()),sentinel_cells=int(invalid.sum()),scale=.01))
        frame.to_csv(OUT/f'{site}_daily.csv',float_format='%.6f')
        for year,g in frame.groupby(frame.index.year):
            core=g[['RAIN','SRAD','TMAX','TMIN']]
            bad=(g.RAIN<0)|(g.SRAD<0)|(g.TMAX<g.TMIN)|(g.TMAX>65)|(g.TMIN < -65)
            yearly.append(dict(site=site,year=int(year),days=len(g),complete_core_days=int(core.notna().all(axis=1).sum()),physical_error_days=int(bad.sum()),**{f'{c}_missing':int(g[c].isna().sum()) for c in ['RAIN','SRAD','TMAX','TMIN','WIND']},split='fit_candidate' if year<=2013 else 'excluded_validation_period'))
        fit=frame.loc['2005':'2013',['RAIN','SRAD','TMAX','TMIN']]
        fit.to_csv(OUT/f'{site}_fit_2005_2013.csv',float_format='%.6f')
        months=[]
        for month,g in fit.groupby(fit.index.month):
            wet=g.RAIN[g.RAIN>0]
            y=float(np.log(wet.mean())-np.log(wet).mean()) if len(wet)>1 else np.nan
            alpha=(8.898919+9.05995*y+.9775373*y*y)/(y*(17.79728+11.968477*y+y*y)) if y>0 else np.nan
            months.append(dict(month=int(month),wet_days=len(wet),wet_years=len(set(wet.index.year)),raw_alpha=alpha,alpha_clamped=bool(alpha>=1),**{f'{c}_std':float(g[c].std()) for c in ['SRAD','TMAX','TMIN']}))
        pd.DataFrame(months).to_csv(OUT/f'{site}_monthly_fit_audit.csv',index=False)
    pd.DataFrame(inventory).to_csv(OUT/'source_inventory.csv',index=False)
    pd.DataFrame(yearly).to_csv(OUT/'yearly_qc.csv',index=False)
    pd.DataFrame(metadata).to_csv(OUT/'station_metadata_segments.csv',index=False)
    pd.DataFrame(flags,columns=['site','date','variable','raw_value','source']).to_csv(OUT/'sentinel_values.csv',index=False)
    status={'data_conversion':'COMPLETE','runtime':'DOCKER_ENGINE_UNAVAILABLE','training':'NOT_STARTED','seeds':list(range(8)),'fit_years':list(range(2005,2014)),'validation_years':list(range(2014,2024)),'source_type':'neighbour_station; estimated radiation retained separately from measured radiation','next_gates':['radiation and precipitation code audit','station relocation audit','CLI parameter and weather generation audit','DSSAT smoke','PPO smoke'],'FQ_seed7':'old smoke blocked by behavior gate, technical checks passed; new isolated cohort-completion attempt required'}
    (OUT/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(pd.DataFrame(yearly).to_string(index=False))

if __name__=='__main__':
    main()
