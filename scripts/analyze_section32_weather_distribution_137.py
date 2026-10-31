"""Read archived weather only. No DSSAT, training, weather generation or policy imports."""
from pathlib import Path
import csv,json,hashlib,re,shutil,collections
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/section32_weather_distribution_20261031';OUT=BASE;i=1
while OUT.exists():i+=1;OUT=Path(str(BASE)+f'_attempt{i:02d}')
OUT.mkdir();HASH={};S=['SY','HL','LC','YC','FQ'];FIELDS=['RAIN','SRAD','TMAX','TMIN'];records=[];exposure=[];issues=[];audit=[];cache={}
def path(p):
 p=Path(p);p=p if p.is_absolute() else ROOT/p;assert p.resolve().is_relative_to(ROOT);return p
def hashfile(p):
 p=path(p);h=hashlib.sha256(p.read_bytes()).hexdigest();HASH[p.relative_to(ROOT).as_posix()]=h;return h
def read(p):
 p=path(p);hashfile(p);return pd.read_csv(p)
def js(p):
 p=path(p);hashfile(p);return json.loads(p.read_text(encoding='utf-8-sig'))
def txt(p):
 p=path(p);hashfile(p);return p.read_text(encoding='utf-8-sig')
def save(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
# Frozen before descriptive calculation. DAP0–90 is the union of three existing stage definitions.
contract={'window':'DAP0_to_DAP90_inclusive','days':91,'basis':'existing DAP0_30, DAP31_60, DAP61_90 stages; fixed management window, NOT full growing season','planting_anchor':'same representative planting DOY used by build_env_config; YC recovered archive final DAP0','dry_day_rule':'RAIN <= 0 mm (existing wet=RAIN>0 convention)','metrics':['seasonal_precipitation','maximum_dry_spell','seasonal_temperature','srad_mean'],'dedup':'SHA256 of numeric RAIN/SRAD/TMAX/TMIN within window, dates omitted; exact unique windows, not independent climate replicates','incomplete':'not extrapolated; recorded separately','no_new_simulation':True}
(OUT/'analysis_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
ph=read('data/observed_phenology_dates_standardized.csv');ph.planting_date=pd.to_datetime(ph.planting_date);DOY=ph.groupby('station_code').planting_date.apply(lambda x:int(round(x.dt.dayofyear.median()))).to_dict()
def wth(p):
 p=path(p);sha=hashfile(p);rows=[];started=False
 for line in p.read_text(encoding='utf-8-sig',errors='replace').splitlines():
  if re.match(r'^\s*@\s*DATE\s+SRAD',line,re.I):started=True;continue
  q=line.split()
  if started and len(q)>=5 and re.fullmatch(r'\d{7}',q[0]):rows.append(dict(DATE=pd.to_datetime(q[0],format='%Y%j'),SRAD=float(q[1]),TMAX=float(q[2]),TMIN=float(q[3]),RAIN=float(q[4])))
 d=pd.DataFrame(rows);assert len(d)>0 and not d.DATE.duplicated().any();return d,sha
def window(d,site,year,mode='date'):
 if mode=='DAP':
  zero=np.flatnonzero(pd.to_numeric(d.DAP).to_numpy()==0);assert len(zero);d=d.iloc[zero[-1]:].copy();d=d[(d.DAP>=0)&(d.DAP<=90)].copy();offset=d.DAP.astype(int).tolist()
 else:
  d=d.copy();d.DATE=pd.to_datetime(d.DATE);plant=pd.Timestamp(year=year,month=1,day=1)+pd.Timedelta(days=DOY[site+'A']-1);offsets=(d.DATE-plant).dt.days;d=d[(offsets>=0)&(offsets<=90)].copy();offset=((d.DATE-plant).dt.days).astype(int).tolist()
 if offset!=list(range(91)):return None
 assert np.isfinite(d[FIELDS].to_numpy(dtype=float)).all();return d
def add(d,site,group,year,sample,source,weight=1,ppo_seed=None,status='verified',mode='date'):
 q=window(d,site,int(year),mode)
 if q is None:
  issues.append(dict(site=site,group=group,sample=sample,source=str(source),reason='fixed_window_incomplete'));return None
 values=q[FIELDS].to_numpy(dtype='<f8');sha=hashlib.sha256(values.tobytes()).hexdigest()
 rain=q.RAIN.to_numpy();longest=run=0
 for x in rain:run=run+1 if x<=0 else 0;longest=max(longest,run)
 row=dict(site=site,group=group,weather_id=sha,year=year,sample=sample,source=str(source),verification=status,n_days=91,seasonal_precipitation=float(rain.sum()),maximum_dry_spell=longest,seasonal_temperature=float(q.TMAX.mean()),srad_mean=float(q.SRAD.mean()),invalid_physical_days=int(((q.RAIN<0)|(q.SRAD<0)|(q.TMAX<q.TMIN)|(q.TMAX>60)|(q.TMIN< -60)).sum()))
 records.append(row);exposure.append(dict(site=site,group=group,weather_id=sha,ppo_seed=ppo_seed,year=year,sample=sample,count=weight,verification=status));return sha
render='DSSAT_auto_validation/multisite_new_cultivar_inputs_013'
windows=[]
for site in S:
 train=list(range(2005 if site in ['SY','LC','FQ'] else 2004,2014))
 for group,years in [('H',train),('T',list(range(2014,2024)))]:
  for year in years:
   file=ROOT/render/site/f'CN{site}{year%100:02d}01.WTH'
   if site=='FQ' and year==2018:file=ROOT/'results/fq2018_forecast_reassessment_125_20261027/seed0/eval_100000_2018_snapshot/dssat/CNFQ1801.WTH'
   if not file.exists():
    candidates=list((ROOT/render/site).glob(f'CN{site}{year%100:02d}01*.WTH'));assert len(candidates)==1;file=candidates[0]
   d,_=wth(file);add(d,site,group,year,str(year),file.relative_to(ROOT).as_posix(),status='frozen_input_file; historical_exposure_frequency_not_reconstructed' if group=='H' else 'evaluation_input; FQ2018_125_QC' if site=='FQ' and year==2018 else 'evaluation_input')
   start=pd.Timestamp(year=year,month=1,day=1)+pd.Timedelta(days=DOY[site+'A']-1);windows.append(dict(site=site,year=year,start=start.date(),end=(start+pd.Timedelta(days=90)).date(),days=91,cross_year=False,basis='representative planting DOY in frozen runtime construction'))
  audit.append(dict(site=site,weather_group=group,source=render+'/'+site if not(site=='FQ' and group=='T') else 'multisite inputs; FQ2018=125 snapshot',years_samples=','.join(map(str,years)),actual_training_exposure_verified='input identity chain; frequencies not reconstructed' if group=='H' else 'not training; evaluation input',qc_status='existing research inputs; physical flags retained',limitation='source QC limitations apply; T is evaluation input, not certified raw observations'))
# SY/LC: verify all 8 manifests and real bank hashes, do not count unused initialization.
for site in ['SY','LC']:
 for seed in range(8):
  manifest=js(f'results/sylc_external_weather_059/{site}/seed{seed}/weather_manifest.json')
  completion=js(f'results/sylc_external_weather_059/{site}/seed{seed}/completion_audit.json')
  entries=[x for x in manifest if not x.get('unused_initialization',False)];assert len(entries)==completion['sampling_episodes']
  for x in entries:
   file=x['bank_WTH'];h=hashfile(file);assert h.lower()==x['source_sha256'].lower()==x['WTH_sha256'].lower()
   k=(site,file)
   if k not in cache:
    d,_=wth(file);cache[k]=d
   add(cache[k],site,'A',x['year'],f"seed{seed}:episode{x['sampling_episode']}",file,ppo_seed=seed,status='selected_frozen_WTH_verified; full91_day_exposure_not_proven_by_manifest')
 audit.append(dict(site=site,weather_group='A',source='059 eight seed selection manifests -> 057 bank',years_samples='2005–2013; 90 mapped WTH, 20 source calendars, 10 weather seeds',actual_training_exposure_verified='selection & hashes 8/8 verified; no per-step trace here',qc_status='calendar/runtime gates passed; 10-seed climate representativeness limited',limitation='future days in selected full WTH not proven encountered; repeated calendars not independent climates'))
# HL/FQ: exact archived daily exposure, read each file serially. Cache metrics by original bytes hash.
archive_cache={};episode_counts=[]
for site in ['HL','FQ']:
 for seed in range(8):
  folder=f'results/hla_wgen_8seed_053/seed_{seed:02d}/attempt_01' if site=='HL' else 'results/fqa_multiyear_wgen_ppo_051/attempt_01' if seed==0 else f'results/fqa_multiyear_wgen_ppo_052_8seed/seed_{seed:02d}/attempt_01'
  manifest=read(folder+'/episode_manifest.csv');gate=js(folder+'/audit_gate.json');assert gate['status']=='PASS_100K_ARCHIVE_ONLY'
  valid=0
  for x in manifest.to_dict('records'):
   file=x['weather_path'];h=hashfile(file);assert h.lower()==x['weather_sha256'].lower()
   k=(site,h,int(x['year']))
   if k not in archive_cache:archive_cache[k]=read(file)
   d=archive_cache[k];assert len(d)==int(x['weather_rows'])
   sha=add(d,site,'A',int(x['year']),f"seed{seed}:episode{x['episode_index']}",file,ppo_seed=seed,status='actual_daily_exposure; '+x['status']);valid+=sha is not None
  episode_counts.append(dict(site=site,ppo_seed=seed,manifest_episodes=len(manifest),full_fixed_window_episodes=valid,incomplete_window_episodes=len(manifest)-valid))
 audit.append(dict(site=site,weather_group='A',source='per-seed episode_manifest + SHA-verified weather_daily',years_samples='2004–2013' if site=='HL' else '2005–2013',actual_training_exposure_verified='all 8 seeds daily file hash verification',qc_status='archive gate PASS; existing input QC conditional',limitation='fixed-window-incomplete episodes excluded and counted; shared schedules not independent weather pools; HL GPCC/mixed-source caveat' if site=='HL' else 'D222 blanks=0 assumption; source/candidate T/SRAD overlays'))
# YC: use ONLY exact78 from already recovered archives; no replay. Do not promote provisional 1003/1036.
archive='results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery';ym=read(archive+'/weather_archive_manifest_verified.csv');ym=ym[(ym['set']=='training')&ym.verification_level.eq('EXACT_MATCH')&~ym.weather_seed.isin([1003,1036])];assert len(ym)==78
for x in ym.to_dict('records'):
 file=archive+'/'+x['canonical_csv_path'];h=hashfile(file);assert h.upper()==x['canonical_series_sha256'].upper();d=read(file)
 add(d,'YC','A',int(x['crop_year']),str(x['weather_seed']),file,ppo_seed=None,status='EXACT78_verified_subset; one recovered context per weather seed',mode='DAP')
yc_models=read('results/yc_random_weather_ppo/004_05/all_model_manifest_0_7.csv');schedule_counts=[]
for x in yc_models[yc_models.training_regime.eq('RANDOM_WEATHER_WGEN')].to_dict('records'):
 file=str(Path(x['source_manifest']).parent/'training_episode_summary.csv').replace('\\','/')
 if not path(file).exists():issues.append(dict(site='YC',group='A',sample=x['model_id'],source=file,reason='training_episode_summary_missing'));continue
 q=read(file)
 for (year,seed),g in q.groupby(['historical_year','training_weather_seed']):schedule_counts.append(dict(site='YC',ppo_seed=x['ppo_seed'],year=int(year),weather_seed=int(seed),episodes=len(g),recovered_exact_seed=seed not in [1003,1036],context_match='recovered subset context may not cover all selected contexts'))
audit.append(dict(site='YC',weather_group='A',source=archive+'; 004_18B/004_19 provenance',years_samples='78 exact recovered seed/context series; 2 unresolved excluded',actual_training_exposure_verified='partial: exact78 subset, not entire 8seed runtime history',qc_status='004_19 partial coverage; low-SRAD tails noted',limitation='other crop-year contexts and two historical realizations unresolved; legacy dates NOT real calendar; use final DAP0'))
raw=pd.DataFrame(records);save('weather_window_records_all_exposures.csv',records);save('weather_exposure_frequencies.csv',exposure);save('training_episode_window_coverage.csv',episode_counts);save('yc_selected_weather_context_counts.csv',schedule_counts);save('weather_source_audit.csv',audit);save('weather_window_calendar.csv',windows);save('weather_window_exclusions.csv',issues)
unique=raw.drop_duplicates(['site','group','weather_id']).copy();unique.to_csv(OUT/'weather_distribution_plot_data.csv',index=False)
M=['seasonal_precipitation','maximum_dry_spell','seasonal_temperature','srad_mean'];summ=[];coverage=[]
for (site,group),q in unique.groupby(['site','group'],sort=False):
 for m in M:
  v=q[m];summ.append(dict(site=site,group=group,metric=m,n_unique_windows=len(q),mean=v.mean(),sd=v.std(ddof=1),minimum=v.min(),q05=v.quantile(.05),q25=v.quantile(.25),median=v.median(),q75=v.quantile(.75),q95=v.quantile(.95),maximum=v.max(),physical_flag_windows=int(q.invalid_physical_days.gt(0).sum())))
for site in S:
 for m in M:
  t=unique[unique.site.eq(site)&unique.group.eq('T')][m]
  for group in ['H','A']:
   v=unique[unique.site.eq(site)&unique.group.eq(group)][m];coverage.append(dict(site=site,metric=m,training_group=group,n_training_unique=len(v),n_evaluation=len(t),covered_minmax=int(t.between(v.min(),v.max()).sum()),covered_q05_q95=int(t.between(v.quantile(.05),v.quantile(.95)).sum()),range_width=v.max()-v.min()))
save('weather_distribution_summary.csv',summ);save('evaluation_weather_coverage.csv',coverage)
# Pool versus selected-exposure distributions are separate. Repetitions are weights, not extra climates.
weighted=[]
for (site,group),q in raw.groupby(['site','group'],sort=False):
 for m in M:weighted.append(dict(site=site,group=group,metric=m,n_selected_records=len(q),weighted_mean=q[m].mean(),weighted_sd=q[m].std(),scope='episode-selected bank window' if site in ['SY','LC'] and group=='A' else 'actual archived complete-window episodes' if site in ['HL','FQ'] and group=='A' else 'unweighted input years or recovered YC subset; not actual exposure weighting'))
save('selected_exposure_weighted_summary.csv',weighted)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.titlesize':9,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,5,figsize=(7.2,4.5));fig.subplots_adjust(left=.085,right=.99,bottom=.22,top=.87,wspace=.65,hspace=.36);colors=['#666666','#26758A','#B36C30'];rng=np.random.default_rng(137)
for col,site in enumerate(S):
 for row,m in enumerate(M[:2]):
  ax=axes[row,col];counts=[]
  for j,group in enumerate(['H','A','T']):
   v=unique[unique.site.eq(site)&unique.group.eq(group)][m].to_numpy();counts.append(len(v));ax.boxplot([v],positions=[j],widths=.5,showfliers=False,patch_artist=True,boxprops={'facecolor':'white','edgecolor':colors[j],'linewidth':.8},medianprops={'color':colors[j],'linewidth':1},whiskerprops={'color':colors[j]},capprops={'color':colors[j]});ax.scatter(j+rng.uniform(-.14,.14,len(v)),v,s=3 if len(v)>100 else 5,alpha=.28 if len(v)>100 else .65,color=colors[j],edgecolors='none',rasterized=False)
  ax.set_xticks([0,1,2],labels=[f'{g}\nn={n}' for g,n in zip(['H','A','T'],counts)]);ax.tick_params(axis='x',length=0)
  if row==0:ax.set_title(site,fontweight='bold')
  if col==0:ax.set_ylabel('91-day precipitation (mm)' if row==0 else 'Longest dry spell (days)')
  ax.set_ylim(bottom=0)
fig.text(.085,.07,'H: historical input years   A: unique augmented windows   T: evaluation input years',fontsize=7)
fig.text(.085,.025,'Fixed DAP 0–90; RAIN ≤ 0 mm defines dry days. Independent y-scales. YC A: exact78 subset.',fontsize=6.7)
for ext in ['pdf','svg','png']:fig.savefig(OUT/f'weather_distribution_comparison.{ext}',dpi=400)
plt.close(fig)
yielddata=read('results/section32_bootstrap_robustness_20261031/bootstrap_robustness_summary.csv').set_index('site');su=pd.DataFrame(summ);co=pd.DataFrame(coverage);matrix=[]
for site in S:
 parts=[];div=[]
 for m in M:
  h=su[(su.site==site)&(su.group=='H')&(su.metric==m)].iloc[0];a=su[(su.site==site)&(su.group=='A')&(su.metric==m)].iloc[0];c=co[(co.site==site)&(co.metric==m)].set_index('training_group')
  div.append(f"{m}:范围{h.maximum-h.minimum:.1f}→{a.maximum-a.minimum:.1f}, SD{h.sd:.1f}→{a.sd:.1f}");parts.append(f"{m}:H{int(c.loc['H','covered_minmax'])}/10→A{int(c.loc['A','covered_minmax'])}/10")
 matrix.append(dict(site=site,diversity='; '.join(div),coverage='; '.join(parts),yield_change=float(yielddata.loc[site,'mean_yield_delta']),interpretation='descriptive coverage only; no causal yield explanation; provenance restrictions retained'))
save('discussion_evidence_matrix.csv',matrix)
lines=['# 天气分布与Discussion证据','', '状态：DISCUSSION_EVIDENCE_PARTIAL','', 'Q1. 五站是否扩大训练天气分布？按站点和指标回答见下表；范围与SD并不总同向。SY/LC是选择库窗口，HL/FQ是已归档完整窗口，YC只代表EXACT78子集，不能统一宣称完整实际训练分布已恢复。','Q2. 是否改善评价天气覆盖？下表给四指标min/max覆盖；另存P5–P95。覆盖只是固定10年的单变量经验支持，不等于联合分布或独立气候总体覆盖。','Q3. 能否解释缺乏一致收益？可检验新增窗口与评价输入的错位，但不能从覆盖推断收益因果；完整训练暴露及来源限制使解释仍为部分证据。','Q4. 能否解释站点表现差异？五站增强方法、样本池及来源不同，不能据五点归纳普遍规律或判定天气分布决定产量。','Q5. 可以开始写Discussion吗？可以写有边界的事实与限制；状态为DISCUSSION_EVIDENCE_PARTIAL，不阻止讨论，也不声称机制闭合。','', '## 预先固定的分析窗口','本轮用既有DAP0–30、31–60、61–90阶段并集，固定播种日至DAP90共91天。它是固定管理窗口，不是全生育季。所有H/A/T在同站点同一相对窗口，不随策略成熟或收获长度改变；不将全年指标替代生育期。播种DOY来自原runtime build_env_config的物候中位数。未观察天气分布后调阈值；干日沿用既有RAIN<=0定义。温度用平均TMAX；四变量均参与增强且YC已有SRAD问题，所以保留SRAD均值，未新增尾部挖掘。','', '## 五站证据']
for r in matrix:lines+=['',f"### {r['site']}（产量A−H {r['yield_change']:+.1f} kg/ha）",r['diversity'],r['coverage']]
lines+=['','## 来源与冲突处理','本轮发现004_17、18B、19晚于004_16恢复gate失败。旧“无法恢复”不能覆盖后来EXACT78证据；004_19整季指标因长度99–110天不等，不直接混入本轮固定91天图。1003/1036仍按18B unresolved处理；旧manifest的MISMATCH不自行升级为确定逐日不符。FULL80候选不纳入正式A。','SY/LC旧WGEN停止记录后有054–059新的日历/注入链。057每站20个源日历映射90份WTH，不把90当90独立气候；本轮验证所有8seed选择manifest与源SHA，选中窗口不等于已证明每个窗口的91天全被PPO访问。055旧状态CSV晚标一天，不使用，改读057最终WTH。','HL/FQ读取所有8seed逐episode CSV及hash，完整91天才进入分布；截断窗口单列exclusions，不能声称覆盖所有训练日。相同窗口去重；共享训练天气不计8倍独立样本。','H来自实际渲染使用的multisite冻存WTH，频次未完整重建，因此H按年份等权；A同时提供已选回合频次/窗口摘要，不将其与H称为严格匹配暴露权重比较。','FQ2018 T使用125已修订评价快照；该启发式TMIN修订是评价条件，不是独立真值。H/其他T保留既有填补或混合来源；异常天数见数据，不以“物理无异常”替代源数据QC。','', '## 事实、关联与假设','事实：图表和CSV中的单变量范围、SD和覆盖计数。关联：同站增强窗口覆盖变化与既有产量变化并列，仅描述。未验证机制：PPO是否利用信息、训练优化难度、奖励限制、管理时序是否造成减产；本轮未读取critic/网络或开展新模拟。','', '## 是否继续','本轮到来源、描述分布及解释边界为止。可开始Discussion事实段；先核对未闭合的实际暴露/晚季与来源再作更强主张。若后续授权管理日志分析，可单独检查灌溉施氮与胁迫时序，本轮不启动。']
report='\n'.join(lines);(OUT/'weather_distribution_analysis.md').write_text(report,encoding='utf-8')
(OUT/'discussion_evidence_matrix.md').write_text('\n'.join(['# Discussion证据矩阵']+[f"\n## {r['site']}\n{r['diversity']}\n\n{r['coverage']}\n\n{r['interpretation']}" for r in matrix]),encoding='utf-8')
(OUT/'discussion_outline_zh.md').write_text('# Discussion构思\n\n1. 区分天气序列数量、91天窗口的经验范围及实际暴露，不能从样本增多直接推断有效多样性。\n2. 按站点说明降水/干旱/温度/辐射的覆盖变化；覆盖扩大与覆盖错位可以同时存在。\n3. 将覆盖变化与产量及资源权衡并列，但不作因果解释。\n4. 讨论固定91天窗口、历史输入来源、共享调度、未闭合暴露以及YC78+2限制。\n5. 结论落在“丰富天气输入不是已证明的充分条件”，不落在“天气增强无效”或“PPO没有利用天气”。\n',encoding='utf-8')
caption='''**Figure D1 | Historical, augmented and evaluation weather over a fixed crop-management window.** Columns represent sites and rows show total precipitation and the longest dry spell over DAP 0–90 (91 days). H denotes historical-training input years, A distinct archived augmented windows, and T evaluation-input years (2014–2023). DAP is aligned to the frozen representative planting date; YC recovered series use the final DAP=0 anchor because their legacy dates are not calendar dates. Dry days have RAIN≤0 mm. Boxes show the median and interquartile range, whiskers extend to 1.5×IQR, and all distinct-window points are displayed. n indicates distinct numeric weather windows, not independent climate replicates. The y-axis scales differ between sites. SY/LC A represents verified selected-bank windows, not confirmed full-window daily encounters; HL/FQ A uses archived daily episodes complete over this window; YC A is the 78-realization verified subset, excluding two unresolved candidates. Duplicate windows are displayed once. Coverage results are descriptive, sample-size-dependent and conditional on source/QC restrictions; they do not establish PPO generalization or mechanisms. FQ2018 uses the archived quality-control version. Late-season weather beyond DAP90 is outside scope.**'''
(OUT/'figure_caption.md').write_text(caption,encoding='utf-8')
source_docs=['docs/sylc_external_weather_054_execution.md','docs/sylc_external_weather_055_execution.md','docs/sylc_external_weather_057_execution.md','docs/yc_random_weather_004_18B_seed1003_1036_closure.md','docs/yc_random_weather_004_19_formal_climate_coverage_audit.md','docs/hla_wgen_training_053_record.md','src/ppo_safe_rendering.py','src/run_all_year_direct_action_safe_ppo.py','src/run_004_18A_yc_weather_summary_for_reporting.py','scripts/freeze_sylc_weather_bank_057.py']
for p in source_docs:txt(p)
(OUT/'analysis_reproducibility.md').write_text('# 复现说明\n\n运行scripts/analyze_section32_weather_distribution_137.py，仅需Python/pandas/numpy/matplotlib，串行读文件，无模拟导入。原始档案路径与SHA见source_sha256.json，不上传大批原始逐日文件。合并去重窗口值、所有暴露计数、排除记录和图数据均保存。计算中逐文件读取，缓存只保留小型日表，不并行、不加载模型。\n\n窗口与阈值先写入analysis_contract.json。H等权年份，A图是去重窗口，抽样频次另表，不把重复序列当独立样本。\n',encoding='utf-8')
doc=ROOT/'docs/2026-10-31_section32_weather_distribution_analysis.md';assert not doc.exists();doc.write_text(report+'\n\n'+caption,encoding='utf-8');shutil.copy2(Path(__file__),OUT/Path(__file__).name);txt('prompts/137_section32_weather_distribution_analysis.md');hashfile(Path(__file__));(OUT/'source_sha256.json').write_text(json.dumps(HASH,indent=2),encoding='utf-8');(OUT/'verification.json').write_text(json.dumps({'status':'DISCUSSION_EVIDENCE_PARTIAL','raw_records':len(raw),'unique_windows':len(unique),'exclusions':len(issues),'new_training':0,'new_simulation':0,'new_weather':0},indent=2),encoding='utf-8');print(json.dumps({'output':str(OUT),'raw_records':len(raw),'unique_windows':len(unique),'exclusions':len(issues)},ensure_ascii=True))
