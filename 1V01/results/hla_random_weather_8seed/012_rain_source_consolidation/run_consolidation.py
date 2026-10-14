from pathlib import Path
import json,hashlib,calendar
from datetime import datetime
import pandas as pd
import numpy as np
import xlrd
R=Path(__file__).resolve().parents[3];O=Path(__file__).resolve().parent;B=R/'results/hl_fq_random_weather_8seed/source_audit';P=R/'results/hla_random_weather_8seed/011_ghcn50756_fallback'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def js(name,obj): (O/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def cs(name,df):df.to_csv(O/name,index=False,encoding='utf-8-sig')
a=json.loads((B/'ghcn_rain_station_audit.json').read_text(encoding='utf-8'))['sites']['HLA'];tested=sorted(a['tested_candidates'],key=lambda x:(x['distance_km'],x['station_id']));approved=[x for x in tested if x['gate']=='PASS'];secondary=[x for x in approved if x['station_id']!='CHM00050756'];days=pd.date_range('2004-01-01','2013-12-31')
inputs=[B/'ghcn_rain_station_audit.json',B/'ghcn_station_candidates.json',B/'ghcn_multi_station_coverage.json',B/'ghcn_multi_station_candidate_rain_gapfill.csv',B/'ghcn_HLA_candidate_rain_gapfill.csv',B/'weather_source_audit_final.json',B/'cern_d222_reaudit/final_gate.json',B/'cern_d222_reaudit/blank_semantics_audit.csv',B/'cern_d222_reaudit/overlap_same_day.csv',R/'results/hla_random_weather_8seed/010/source_audit/hla_2004_monthly_crosscheck.csv',R/'data/external/cern_d222/raw/excel.xls',R/'my_data/HLLCYCFQ降雨数据.xls',B/'nasa_power_HLA_2004_2013_raw.json']+[P/n for n in ['accepted_target_window_fallback.csv','ghcn_2004_coverage.csv','date_shift_comparison.csv','final_gate.json']]
frames={};checks=[]
for m in tested:
 p=R/m['raw_path'];inputs.append(p);assert digest(p)==m['raw_sha256'],str(p)
 rows=[]
 for line in p.read_text().splitlines():
  if line[:11]!=m['station_id'] or line[17:21]!='PRCP':continue
  y,mo=int(line[11:15]),int(line[15:17])
  if not 2004<=y<=2013:continue
  for day in range(1,calendar.monthrange(y,mo)[1]+1):
   f=line[21+8*(day-1):29+8*(day-1)];v=int(f[:5]);q=f[6];status='MISSING_-9999' if v==-9999 else 'QC_REJECTED' if q.strip() else 'NEGATIVE_REJECTED' if v<0 else 'VALID'
   rows.append(dict(date=pd.Timestamp(y,mo,day),rain=v/10 if status=='VALID' else np.nan,mflag=f[5],qflag=q,sflag=f[7],status=status))
 df=pd.DataFrame(rows);assert not df.date.duplicated().any();frames[m['station_id']]=df.set_index('date').reindex(days);frames[m['station_id']]['status']=frames[m['station_id']].status.fillna('NO_PRCP_RECORD')
for rel,expected in json.loads((P/'source/input_hashes.json').read_text(encoding='utf-8')).items():
 p=R/rel;assert digest(p)==expected,rel
assert json.loads((P/'final_gate.json').read_text())['hla_2004_ghcn_fallback']=='ACCEPTED'
def loc(path):
 w=xlrd.open_workbook(str(path));rows=[]
 for sh in w.sheets():
  hd={str(v).strip():i for i,v in enumerate(sh.row_values(1))}
  for i in range(2,sh.nrows):
   v=sh.row_values(i)
   if str(v[hd['生态站代码']]).strip()!='HLA':continue
   y=int(v[hd['年']])
   if not 2004<=y<=2013:continue
   rows.append((pd.Timestamp(y,int(v[hd['月']]),int(v[hd['日']])),pd.to_numeric(v[hd['20-20合计(mm)']],errors='coerce')))
 s=pd.Series([v for _,v in rows],index=[d for d,_ in rows]);assert not s.index.duplicated().any();return s.reindex(days)
d=loc(R/'data/external/cern_d222/raw/excel.xls');l=loc(R/'my_data/HLLCYCFQ降雨数据.xls');assert ((d-l).dropna()==0).all();local=d.combine_first(l)
nj=json.loads((B/'nasa_power_HLA_2004_2013_raw.json').read_text());n=pd.Series({pd.Timestamp(k):v for k,v in nj['properties']['parameter']['PRECTOTCORR'].items() if v>=0})
records=[];unresolved=[];ranking=[]
for day in days:
 g=frames['CHM00050756'].loc[day];chosen=None
 for rank,m in enumerate(secondary,1):
  f=frames[m['station_id']].loc[day]
  if pd.isna(local[day]) and pd.isna(g.rain):ranking.append(dict(date=day.date(),rank=rank,station=m['station_id'],distance_km=m['distance_km'],status=f.status,rain_mm=f.rain))
  if chosen is None and pd.notna(f.rain):chosen=(m,f)
 row=dict(site='HLA',date=str(day.date()),year=day.year,month=day.month,day=day.day,local_d222_rain_mm=d[day],local_legacy_rain_mm=l[day],ghcn_50756_rain_mm=g.rain,ghcn_50756_qflag=g.qflag,ghcn_50756_mflag=g.mflag,ghcn_50756_sflag=g.sflag,secondary_ghcn_station=chosen[0]['station_id'] if chosen else None,secondary_ghcn_rain_mm=chosen[1].rain if chosen else None,nasa_reference_rain_mm=n.get(day,np.nan),selected_rain_mm=np.nan,selected_source='NONE',selected_station_id=None,source_priority=4,source_reason='NO_APPROVED_SOURCE_HAS_VALID_PRCP',qc_status='UNRESOLVED',rain_status='UNRESOLVED',selected_mflag=None,selected_qflag=None,selected_sflag=None,selected_source_file=None)
 if pd.notna(local[day]): row.update(selected_rain_mm=local[day],selected_source='HLA_LOCAL_D222_CHAIN',selected_station_id='HLA',source_priority=1,source_reason='EXPLICIT_LOCAL_NUMERIC_VALUE;SHARED_D222_LEGACY_CHAIN',selected_source_file='data/external/cern_d222/raw/excel.xls;my_data/HLLCYCFQ降雨数据.xls')
 else:
  use=(next(m for m in approved if m['station_id']=='CHM00050756'),g) if pd.notna(g.rain) else chosen
  if use:
   m,f=use;priority=2 if m['station_id']=='CHM00050756' else 3;row.update(selected_rain_mm=f.rain,selected_source='NOAA_GHCN_DAILY' if priority==2 else 'NOAA_GHCN_DAILY_SECONDARY',selected_station_id=m['station_id'],source_priority=priority,source_reason='LOCAL_BLANK;012_AUTHORIZED_50756_GENERAL_FALLBACK' if priority==2 else 'LOCAL_AND_50756_UNAVAILABLE;NEAREST_008_APPROVED_VALID_STATION',selected_mflag=f.mflag,selected_qflag=f.qflag,selected_sflag=f.sflag,selected_source_file=m['raw_path'])
 if row['source_priority']<4:row.update(qc_status='PASS',rain_status='RESOLVED')
 else:
  u=dict(date=str(day.date()),d222_status='NO_EXPLICIT_NUMERIC_VALUE',ghcn_50756_status=g.status,ghcn_50756_qflag=g.qflag,nasa_reference=n.get(day,np.nan),unresolved_reason='ALL_APPROVED_GHCN_UNAVAILABLE_AND_LOCAL_BLANK')
  for m in secondary:u[m['station_id']+'_status']=frames[m['station_id']].loc[day,'status'];u[m['station_id']+'_qflag']=frames[m['station_id']].loc[day,'qflag']
  non=next((m for m in tested if m['gate']!='PASS' and pd.notna(frames[m['station_id']].loc[day,'rain'])),None)
  u.update(nearest_available_nonapproved_station=non['station_id'] if non else None,nonapproved_distance_km=non['distance_km'] if non else None,nonapproved_rain_mm=frames[non['station_id']].loc[day,'rain'] if non else None,nonapproved_gate='NOT_PROMOTED');unresolved.append(u)
 records.append(row)
df=pd.DataFrame(records);ud=pd.DataFrame(unresolved);cs('daily_source_selection_2004_2013.csv',df);cs('unresolved_dates_final.csv',ud);cs('2013_unresolved_focus.csv',ud[ud.date.str.startswith('2013')]);cs('secondary_candidate_ranking.csv',pd.DataFrame(ranking))
def summary(cols):
 rows=[]
 for key,x in df.groupby(cols):
  key=key if isinstance(key,tuple) else (key,);complete=bool(x.selected_rain_mm.notna().all());r=dict(zip(cols,key));r.update(total_days=len(x),local_days=int((x.source_priority==1).sum()),ghcn50756_days=int((x.source_priority==2).sum()),secondary_days=int((x.source_priority==3).sum()),unresolved_days=int((x.source_priority==4).sum()),selected_precipitation_mm=float(x.selected_rain_mm.sum()),precipitation_label='annual total' if complete and cols==['year'] else 'complete month total' if complete else 'observed subtotal',wet_day_count=int((x.selected_rain_mm>0).sum()) if complete else None,max_daily_rain_mm=float(x.selected_rain_mm.max()) if complete else None);rows.append(r)
 return pd.DataFrame(rows)
cs('coverage_summary_by_year.csv',summary(['year']));cs('coverage_summary_by_month.csv',summary(['year','month']))
runs=[]
for date in pd.to_datetime(ud.date):
 if runs and date==runs[-1][1]+pd.Timedelta(days=1):runs[-1][1]=date;runs[-1][2]+=1
 else:runs.append([date,date,1])
cs('unresolved_runs.csv',pd.DataFrame([dict(start=str(a.date()),end=str(b.date()),days=c) for a,b,c in runs]))
js('source_hierarchy.json',dict(primary='HLA_LOCAL_D222_CHAIN',fallback='CHM00050756',secondary_order=secondary,original_008_thresholds_preserved=True,qualification_caveat='008 secondary event metrics used blank-as-zero; this task explicitly reuses frozen approvals, not a new independent 011-style validation',nasa='REFERENCE_ONLY',blank_to_zero=False,nonapproved='INFORMATION_ONLY'))
pro=[dict(path=str(p.relative_to(R)),sha256=digest(p),size=p.stat().st_size,mtime=datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat()) for p in dict.fromkeys(inputs)];js('input_provenance.json',dict(files=pro,raw_hashes_match_008_011=True))
gate=dict(site='HLA',period='2004-01-01/2013-12-31',total_days=len(df),local_d222_days=int((df.source_priority==1).sum()),ghcn_50756_days=int((df.source_priority==2).sum()),secondary_ghcn_days=int((df.source_priority==3).sum()),unresolved_days=len(ud),unresolved_percentage=100*len(ud)/len(df),unresolved_years=sorted(set(pd.to_datetime(ud.date).dt.year.tolist())),ghcn50756_general_fallback='ACCEPTED',hla_rain_provenance='PARTIAL' if len(ud) else 'PASS',ready_to_resume_010=not bool(len(ud)),resume_point_if_pass='010_PHASE_B_FITTING_WEATHER',earliest_unresolved=ud.date.min(),latest_unresolved=ud.date.max(),longest_consecutive_unresolved_run=max(r[2] for r in runs),nasa_zero_days=int((ud.nasa_reference==0).sum()),nasa_positive_days=int((ud.nasa_reference>0).sum()),secondary_counts=df[df.source_priority==3].selected_station_id.value_counts().to_dict(),no_downstream_execution=True)
js('final_gate.json',gate)
if not len(ud):cs('hla_rain_provenance_candidate_2004_2013.csv',df)
assert len(df)==3653 and df.date.nunique()==3653 and (df.selected_rain_mm.dropna()>=0).all();assert all(digest(R/p['path'])==p['sha256'] for p in pro)
# Frozen 008 gapfill is an independent implementation crosscheck, restricted to HLA.
old=pd.read_csv(B/'ghcn_multi_station_candidate_rain_gapfill.csv');old=old[old.site=='HLA'];cur=df[df.source_priority.isin([2,3])];cmp=cur.merge(old,on='date');assert len(cmp)==len(cur)==len(old);assert np.allclose(cmp.selected_rain_mm,cmp.candidate_value_mm);assert (cmp.selected_station_id==cmp.candidate_station_id).all()
js('verification.json',dict(raw_hashes_unchanged=True,calendar_3653_unique=True,nonnegative_values=True,local_values_preserved=True,independent_008_selection_parity=True,blank_inferred_zero=False))
print(json.dumps(gate,indent=2))
