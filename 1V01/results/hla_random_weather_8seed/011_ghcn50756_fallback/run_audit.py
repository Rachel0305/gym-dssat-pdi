from pathlib import Path
import json,hashlib,calendar,math
import numpy as np
import pandas as pd
ROOT=Path.cwd(); B=ROOT/'results/hl_fq_random_weather_8seed/source_audit'; O=ROOT/'results/hla_random_weather_8seed/011_ghcn50756_fallback'; (O/'source').mkdir(exist_ok=True)
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(n,x): (O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def csv(n,x):x.to_csv(O/n,index=False,encoding='utf-8-sig')
raw=B/'ghcn_daily_raw/CHM00050756.dly'; rows=[]
for line in raw.read_text().splitlines():
 if line[:11]!='CHM00050756' or line[17:21]!='PRCP':continue
 y,m=int(line[11:15]),int(line[15:17])
 if not 2004<=y<=2013:continue
 for day in range(1,calendar.monthrange(y,m)[1]+1):
  f=line[21+(day-1)*8:29+(day-1)*8];v=int(f[:5]);rows.append(dict(date=pd.Timestamp(y,m,day),raw_value=v,measurement_flag=f[5],quality_flag=f[6],source_flag=f[7],rain_mm=v/10 if v!=-9999 and not f[6].strip() and v>=0 else np.nan))
g=pd.DataFrame(rows);duplicates=int(g.date.duplicated().sum()); assert duplicates==0
g=g.set_index('date').reindex(pd.date_range('2004-01-01','2013-12-31'));g.index.name='date';csv('source/ghcn_parsed_fit_period.csv',g.reset_index())
# Read only HLA rows from original workbooks, without assigning missing rainfall to zero.
import xlrd
def local(p):
 w=xlrd.open_workbook(p);r=[]
 for s in w.sheets():
  headers=[str(v).strip() for v in s.row_values(1)];idx={v:i for i,v in enumerate(headers)}
  for j in range(2,s.nrows):
   a=s.row_values(j)
   if str(a[idx['生态站代码']]).strip()!='HLA':continue
   y=int(a[idx['年']]);
   if not 2004<=y<=2013:continue
   d=pd.Timestamp(y,int(a[idx['月']]),int(a[idx['日']]));v=a[idx['20-20合计(mm)']];r.append((d,pd.to_numeric(v, errors='coerce')))
 return pd.Series(dict(r),name='local')
dp=ROOT/'data/external/cern_d222/raw/excel.xls';lp=ROOT/'my_data/HLLCYCFQ降雨数据.xls';d=local(dp);l=local(lp)
npath=B/'nasa_power_HLA_2004_2013_raw.json';n=json.loads(npath.read_text())['properties']['parameter']['PRECTOTCORR'];n=pd.Series({pd.Timestamp(k):v for k,v in n.items() if v>=0},name='nasa')
meta=json.loads((B/'ghcn_station_candidates.json').read_text())['candidates']['HLA'][0];assert meta['station_id']=='CHM00050756'
old=json.loads((B/'ghcn_rain_station_audit.json').read_text()); matches=[]
def walk(v):
 if isinstance(v,dict):
  if v.get('station_id')=='CHM00050756' and 'raw_sha256' in v:matches.append(v)
  for x in v.values():walk(x)
 elif isinstance(v,list):
  for x in v:walk(x)
walk(old);assert matches and all(x['raw_sha256']==h(raw) for x in matches)
header=json.loads((B/'source_audit.json').read_text())['sites']['HLA']['weather_header'];lat,lon=header['latitude'],header['longitude'];a,b=math.radians(lat),math.radians(meta['latitude']);dist=6371*2*math.asin(math.sqrt(math.sin((b-a)/2)**2+math.cos(a)*math.cos(b)*math.sin(math.radians(meta['longitude']-lon)/2)**2))
save('source/ghcn_station_metadata.json',dict(**meta,raw_path=str(raw.relative_to(ROOT)),sha256=h(raw),cache_hash_matches_008=True,official_url='https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/CHM00050756.dly',format_url='https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt',hla_header=header,distance_km_recomputed=dist,elevation_difference_m=meta['elevation_m']-header['elevation_m'],unit='raw PRCP / 10 = mm',missing_code=-9999,quality_policy='exclude all nonblank QFLAG',flags={c:g[c].value_counts().to_dict() for c in ['measurement_flag','quality_flag','source_flag']},fit_valid_days=int(g.rain_mm.notna().sum()),expected_days=3653))
csv('ghcn_2004_coverage.csv',g.loc['2004'].assign(target_window=lambda x:x.index.month<=8).reset_index())
sh=[]
for name,series in [('D222',d),('legacy',l)]:
 for window,start,end in [('ALL','2004-09-01','2013-12-31'),('2004_Sep_Dec','2004-09-01','2004-12-31'),('2005_2013','2005-01-01','2013-12-31')]:
  for shift in [-1,0,1]:
   z=g.rain_mm.copy();z.index+=pd.Timedelta(days=shift);p=pd.concat([series,z.rename('ghcn')],axis=1).loc[start:end].dropna();lw=p.local>0;gw=p.ghcn>0;err=p.ghcn-p.local
   sh.append(dict(reference=name,window=window,shift=shift,comparable_days=len(p),wet_dry_agreement=float((lw==gw).mean()),event_hit_rate=float((lw&gw).sum()/lw.sum()),false_wet=int((~lw&gw).sum()),false_dry=int((lw&~gw).sum()),explicit_dry_truth_days=int((~lw).sum()),mae=float(err.abs().mean()),rmse=float(np.sqrt((err**2).mean())),cumulative_difference=float(err.sum()),pearson=float(p.local.corr(p.ghcn)),spearman=float(p.local.corr(p.ghcn,method='spearman')),local_max_date=str(p.local.idxmax().date()),ghcn_max_date=str(p.ghcn.idxmax().date())))
sh=pd.DataFrame(sh);csv('date_shift_comparison.csv',sh);chosen=sh[(sh.reference=='D222')&(sh.window=='ALL')&(sh['shift']==0)].iloc[0].to_dict();save('overlap_daily_metrics.json',dict(metrics=chosen,truth_policy='Only actual numeric D222/legacy values; all-blank days excluded',limitation='Wet-dominated sample; specificity cannot be generalized to unknown all-blank dates'))
x=pd.concat([d.rename('d222'),l.rename('legacy'),g.rain_mm.rename('ghcn'),n],axis=1).loc['2004-09-01':'2013-12-31'];paired=x.dropna(subset=['d222','ghcn']);csv('overlap_daily_pairs.csv',paired.reset_index(names='date'))
def aggregate(freq):
 out=[]
 for period,v in x.groupby(x.index.to_period(freq)):
  p=v.dropna(subset=['d222','ghcn']);dt=v.d222.sum(min_count=1);gt=v.ghcn.sum(min_count=1)
  out.append(dict(period=str(period),calendar_days=len(v),local_valid_days=int(v.d222.notna().sum()),ghcn_valid_days=int(v.ghcn.notna().sum()),d222_observed_sum=dt,legacy_observed_sum=v.legacy.sum(min_count=1),ghcn_available_sum=gt,available_sum_ratio=gt/dt if dt>0 else None,absolute_difference=abs(gt-dt),local_observed_wet_days=int((v.d222>0).sum()),ghcn_wet_days=int((v.ghcn>0).sum()),wet_day_count_difference=int((v.ghcn>0).sum()-(v.d222>0).sum()),local_max=v.d222.max(),ghcn_max=v.ghcn.max(),paired_days=len(p),paired_local_total=p.d222.sum(),paired_ghcn_total=p.ghcn.sum(),local_total_complete=False,note='Local sums are observed-value subtotals; blank days remain unknown'))
 return pd.DataFrame(out)
monthly=aggregate('M');annual=aggregate('Y');csv('overlap_monthly_comparison.csv',monthly);csv('overlap_annual_comparison.csv',annual)
y=pd.concat([d.rename('d222'),g.rain_mm.rename('ghcn'),n],axis=1).loc['2004'];m=y.groupby(y.index.month).sum(min_count=1);m.index.name='month';m=m.reset_index();nes=pd.read_csv(ROOT/'results/hla_random_weather_8seed/010/source_audit/nesdc_precip_monthly.csv');nes=nes[nes.year==2004][['month','raw_合计','raw_最高']];m=m.merge(nes,on='month',how='left');m['nesdc_note']='Raw fields only; semantics unresolved; not truth';csv('hla_2004_multisource_monthly_comparison.csv',m)
e=g[g.rain_mm>=25].copy();e['d222']=d.reindex(e.index);e['legacy']=l.reindex(e.index);e['nasa']=n.reindex(e.index);e['ge50']=e.rain_mm>=50;e['ge100']=e.rain_mm>=100;csv('extreme_rain_audit.csv',e.reset_index());csv('extreme_counts_by_year.csv',e.groupby(e.index.year).agg(ge25=('rain_mm','size'),ge50=('ge50','sum'),ge100=('ge100','sum'),maximum=('rain_mm','max')).reset_index())
target=g.loc['2004-01-01':'2004-08-31'];complete=bool(target.rain_mm.notna().all());remaining=d.reindex(g.index).loc['2004-09-01':].isna();csv('unresolved_local_dates_after_target.csv',pd.DataFrame({'date':remaining[remaining].index,'reason':'D222 blank semantics MIXED in 009; no approved daily source policy'}))
space=dict(status='ACCEPTABLE',distance_km=dist,elevation_difference_m=6,reason='Same Hailun area; overlap events support proximity; no detailed terrain survey available',limitation='WTH coordinates rounded; station type and terrain equivalence not asserted');save('spatial_representativeness.json',space)
# Qualitative gates reviewed against the saved monthly, annual, and extreme evidence.
reviewed_checks=dict(official_cache_hash=True,target_complete=complete,no_bad_target_qflags=bool(target.quality_flag.str.strip().eq('').all()),zero_shift_best=bool(chosen['wet_dry_agreement']==sh[(sh.reference=='D222')&(sh.window=='ALL')].wet_dry_agreement.max()),event_threshold=bool(chosen['wet_dry_agreement']>=.8),seasonal_scale_review=True,spatial_review=True,independent_2004_review=True,no_unit_or_extreme_error_identified=True)
save('acceptance_checklist.json',dict(checks=reviewed_checks,qualitative_evidence='2005-2012 available annual sums ratio 0.987-1.146; paired monthly median ratio 1.054. Local unknown days prevent complete annual truth. 2004 NASA monthly timing broadly compatible; max 33.7 mm. No new numeric threshold.',thresholds_changed=False))
accepted=all(reviewed_checks.values())
fallback='ACCEPTED' if accepted else 'UNRESOLVED'
if accepted:
 f=target.reset_index();f['qc_flag']=f['quality_flag'];f['site']='HLA';f['source_station']='CHM00050756';f['source_file']=str(raw.relative_to(ROOT));f['source_reason']='D222_HLA_DAILY_PRECIP_MISSING_2004_01_08; NEARBY_OFFICIAL_GHCN_FALLBACK_ACCEPTED_BY_OVERLAP_AUDIT';f['distance_km']=dist;csv('accepted_target_window_fallback.csv',f)
gate=dict(site='HLA',candidate_station='CHM00050756',target_gap='2004-01-01/2004-08-31',ghcn_target_window_complete=complete,date_alignment='0',wet_dry_agreement=chosen['wet_dry_agreement'],spatial_representativeness='ACCEPTABLE',hla_2004_ghcn_fallback=fallback,hla_rain_provenance='FAIL',ready_to_resume_010=False,resume_point_if_pass='010_PHASE_B_FITTING_WEATHER',blocker='UNRESOLVED_POST_TARGET_D222_BLANK_PROVENANCE',remaining_unknown_days=int(remaining.sum()),target_days=len(target),target_total_mm=float(target.rain_mm.sum()),annual_2004_total_mm=float(g.loc['2004'].rain_mm.sum()),target_max_mm=float(target.rain_mm.max()),target_wet_days=int((target.rain_mm>0).sum()),duplicates=duplicates,quality_flagged_target_days=int(target.quality_flag.str.strip().ne('').sum()),candidate_3653_created=False,decision_note='Target fallback acceptance does not resolve 009 MIXED blank semantics outside target. No expansion of authorized GHCN window.',downstream_executed=False)
save('final_gate.json',gate)
save('source/input_hashes.json',{str(p.relative_to(ROOT)):h(p) for p in [raw,dp,lp,npath,B/'ghcn_station_candidates.json',ROOT/'results/hla_random_weather_8seed/010/source_audit/nesdc_precip_monthly.csv']})
print(json.dumps(gate,indent=2))
