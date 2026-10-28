"""Saved raw-output and fitting-window audit; no simulation or model loading."""
from pathlib import Path
import ast, re, json, hashlib, io, time
import pandas as pd
import psutil
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/fq2018_generator_audit_124_20261027';OUT.mkdir(exist_ok=False)
sources={};start=time.time()
def read(p):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    assert p.resolve().is_relative_to(ROOT.resolve())
    b=p.read_bytes();sources[p.relative_to(ROOT).as_posix()]=hashlib.sha256(b).hexdigest();return b
def csv(p):return pd.read_csv(io.BytesIO(read(p)),keep_default_na=False)
def text(p):return read(p).decode('utf-8',errors='replace')
def sha(p):read(p);return sources[Path(p).relative_to(ROOT).as_posix()]
# Extract only pure parsers, not top-level application code.
code=ast.parse(text('src/calculate_five_site_wue_nue_from_summary_019_10.py'))
for node in code.body:
    if isinstance(node,ast.FunctionDef) and node.name in ['clean_name','parse_value','parse_summary_out']:exec(compile(ast.Module(body=[node],type_ignores=[]),'saved_parser','exec'))
q=csv('results/forecast_five_scenario_120_corrected_HL_20261027/all_five_sites_seed_year_scenario.csv')
fq=q[q.site.eq('FQ')&q.year.eq(2018)]
selected=fq[(fq.scenario.ne('rl_candidate')&fq.seed.eq(0))|fq.scenario.eq('rl_candidate')]
rows=[];inputs=[]
for r in selected.itertuples():
    snap=ROOT/r.snapshot_path;read(snap/'Summary.OUT');records=parse_summary_out(snap/'Summary.OUT');last=records[-1]
    warn=text(snap/'WARNING.OUT') if (snap/'WARNING.OUT').exists() else ''
    rows.append(dict(scenario=r.scenario,seed=r.seed,snapshot=r.snapshot_path,table_yield=r.grain_yield_kg_ha,HWAM=last['HWAM'],IRCM=last['IRCM'],NICM=last['NICM'],MDAT=last.get('MDAT'),HDAT=last.get('HDAT'),summary_rows=len(records),yield_matches=abs(last['HWAM']-r.grain_yield_kg_ha)<=1,slowed_grain_filling='slowed grain filling' in warn.lower(),precipitation_error='precipitation data error' in warn.lower(),normal_termination_not_certified_by_summary_alone=True))
    for name in ['CNFQ1801.WTH','SOIL.SOL','MZCER048.CUL','fileX.MZX','dssat-pdi.yml','WARNING.OUT','Summary.OUT']:
        p=snap/name;inputs.append(dict(scenario=r.scenario,seed=r.seed,name=name,path=str(p.relative_to(ROOT)),exists=p.exists(),sha256=sha(p) if p.exists() else ''))
pd.DataFrame(rows).to_csv(OUT/'FQ2018_raw_endpoints_12.csv',index=False)
pd.DataFrame(inputs).to_csv(OUT/'FQ2018_input_hashes.csv',index=False)
corrected=ROOT/'results/fqa_wgen_052_corrected2018_231/seed_00/snapshot'
original=ROOT/selected[selected.scenario.eq('official_extension_expert')].iloc[0].snapshot_path
def wth(p):
    lines=text(p).splitlines();i=next(i for i,l in enumerate(lines) if l.lstrip().startswith('@') and 'DATE' in l)
    cols=lines[i].replace('@','').split();data=[l.split() for l in lines[i+1:] if l.strip() and l.strip()[0].isdigit()]
    return pd.DataFrame(data,columns=cols).apply(pd.to_numeric)
a=wth(original/'CNFQ1801.WTH');b=wth(corrected/'CNFQ1801.WTH');m=a.merge(b,on='DATE',suffixes=('_original','_corrected'),validate='one_to_one');diff=[]
for field in ['SRAD','TMAX','TMIN','RAIN']:
    for row in m[m[field+'_original'].ne(m[field+'_corrected'])].to_dict('records'):
        diff.append(dict(DATE=row['DATE'],variable=field,original=row[field+'_original'],corrected=row[field+'_corrected'],delta=row[field+'_corrected']-row[field+'_original']))
pd.DataFrame(diff).to_csv(OUT/'FQ2018_WTH_changed_cells.csv',index=False)
identity=[]
for name in ['SOIL.SOL','MZCER048.CUL','fileX.MZX','CNFQ1801.WTH']:
    pa=original/name;pb=corrected/name
    identity.append(dict(file=name,original_sha256=sha(pa),corrected_sha256=sha(pb),identical=sha(pa)==sha(pb),note='fileX management differs: expert versus augmented PPO; byte inequality does not isolate physical settings' if name=='fileX.MZX' else ''))
pd.DataFrame(identity).to_csv(OUT/'original_vs_corrected_snapshot_files.csv',index=False)
read(corrected/'Summary.OUT');cor=parse_summary_out(corrected/'Summary.OUT')[-1]
old=csv('2V02/deliverables/FQ/historical/seed_00/tables/yearly_metrics.csv');old[old.year.eq(2018)].to_csv(OUT/'old_corrected_reference_endpoints.csv',index=False)
read('results/hl_fq_8seed_cross_site/055_03_native_auto_corrected2018/DIAGNOSTIC.md')
fits=[('SY','results/sylc_external_weather_20261012/measured_radiation_candidate_v2/SY_candidate_2005_2013.csv','results/sylc_external_weather_20261012/processed_stage/CNSY.CLI','scripts/run_sylc_wgen_runtime_smoke_054.py'),('LC','results/sylc_external_weather_054/LC_v2/candidate_2000_2013.csv','results/sylc_external_weather_054/LC_v2/CNLC.CLI','scripts/run_sylc_wgen_runtime_smoke_054.py'),('HL','results/hla_weather_enhancement_029/weather_fitting/gpcc_raw/fitting_weather.csv','results/hla_weather_enhancement_029/weather_fitting/gpcc_raw/CNHL.CLI','results/hla_wgen_8seed_053/seed_00/attempt_01/source_manifest.json'),('YC','results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv','results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI','results/yc_random_weather_ppo/004_03/run_controlled_pilot.py'),('FQ','results/fqa_weather_resume_041/fitting_weather.csv','results/fqa_weather_resume_041/CNFQ.CLI','results/fqa_multiyear_wgen_ppo_051/attempt_01/source_manifest.json')]
frows=[]
for site,fit,cli,link in fits:
    f=csv(fit);col=next(c for c in f if c.lower()=='date');dates=pd.to_datetime(f[col]);years=sorted(dates.dt.year.unique().tolist());clisha=sha(ROOT/cli);linktext=text(link)
    frows.append(dict(site=site,fitting_path=fit,rows=len(f),start=str(dates.min().date()),end=str(dates.max().date()),years=','.join(map(str,years)),overlap_test_years=','.join(map(str,sorted(set(years)&set(range(2014,2024))))),cli_path=cli,cli_sha256=clisha,link_source=link,cli_hash_present_in_source=clisha.lower() in linktext.lower(),full_upstream_no_leakage_certified=False))
pd.DataFrame(frows).to_csv(OUT/'generator_fitting_windows_5.csv',index=False)
# Read saved gates and explicit build scripts as provenance; do not execute them.
for p in ['scripts/audit_lc_long_window_054.py','scripts/screen_sy_full_year_ensemble_055.py','results/fqa_weather_resume_041/build_static_fitting.py','results/sylc_external_weather_055/path_gate.json','results/sylc_external_weather_055/LC_path_gate_attempt003.json','results/sylc_external_weather_057/SY/source_manifest.csv','results/sylc_external_weather_057/LC/source_manifest.csv']:
    if (ROOT/p).exists():read(p)
replay=pd.DataFrame([dict(site='FQ',method='forecast',seed=s,year=2018,existing_model_only=True,stage='PLANNED_NOT_RUN',input_version='pending provenance-based freeze',smoke_first=s==0) for s in range(8)])
replay.to_csv(OUT/'minimum_forecast_reassessment_plan_8.csv',index=False)
stats={'raw_zero_yield_rows':sum(r['HWAM']==0 for r in rows),'raw_endpoint_rows':len(rows),'all_table_yields_match':all(r['yield_matches'] for r in rows),'slowed_grain_filling_warnings':sum(r['slowed_grain_filling'] for r in rows),'changed_weather_cells':len(diff),'changed_variables':sorted(set(r['variable'] for r in diff)),'corrected_existing_augmented_seed0_yield':cor['HWAM'],'fitting_files_with_test_year_overlap':sum(bool(r['overlap_test_years']) for r in frows),'simulations':0,'models_loaded':0}
report=f'''# 124：FQ2018与天气拟合窗口审计

## FQ2018零产量已定位到实际归档模拟输出

四个基线及8个预报策略共{len(rows)}个快照，最终Summary.OUT的HWAM全部为0，与120表一致，不是本轮绘图漏读或缺失值补0；{stats['slowed_grain_filling_warnings']}个WARNING含slowed grain filling。正常终止状态不能只由Summary存在或该警告判定，尚未逐步复核全部终止握手。

旧2V02的FQ2018采用历史任务已有的隔离Tmin修正版本，当前预报队列使用原版。对归档原版专家WTH与修正版增强seed0 WTH逐日比较：共{len(diff)}个单元不同，涉及{stats['changed_variables']}。土壤、品种和fileX字节比较见original_vs_corrected_snapshot_files.csv；fileX属于不同管理策略，不应将其差异直接当物理合同差异。旧修正版增强seed0快照产量为{cor['HWAM']} kg/ha，但它不是预报seed0，不能代填预报结果。

结论：当前旧增强/原PPO与新增预报的FQ三路线混用了不同2018测试天气版本。归档材料支持温度输入版本与零产量相关；不能仅凭这一组非同策略快照比较建立温度改动的严格因果归因，也不能自动认定修正天气更接近真实观测。保留所有零产量和修正结果，分版本报告。

## 五站拟合窗口

generator_fitting_windows_5.csv记录实际拟合CSV的日期、CLI哈希及链接来源。SY 2005—2013；LC 2000—2013；HL/YC 2004—2013；FQ 2005—2013（以CSV实查为准）。这些文件的测试年份交集为0，比仅核对训练回合年份更进一步；仍不代表全部插补、再分析、空间产品或气候统计上游链已排除信息泄漏。HL的gpcc_raw文件标有GRIDDED_REFERENCE_SENSITIVITY_ONLY，不得隐去其数据性质，不能称五站统一纯实测增强。代码路径引用是拟合候选关联，不等价于每个训练运行的完整物理合同认证。

## 下一阶段具体工作

首先追踪2018 Tmin异常与修正规则的原始观测来源，冻结可辩护的评价版本。若采用已有修正版：至少需要现有8个预报模型在FQ2018同一修正版下重新评价，先seed0单年smoke，再其余7个串行；沿用旧修正版原/增强16模型前，还须核验同一物理合同与基线口径。若选原版：需对旧16模型回收或重评原版2018端点。不可混合两条路线，只改表不回放也不可行。

minimum_forecast_reassessment_plan_8.csv仅为待执行清单，当前未运行模型或DSSAT。smoke应核验输入日期/逐日天气、原始Summary及事件水氮闭合、动作合法性、终止状态、保护文件和资源规则；不能以必须获得正产量作为验收条件。无需重新训练。
'''
(OUT/'report_zh.md').write_text(report,encoding='utf-8')
(OUT/'audit_summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'source_sha256.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'resources.json').write_text(json.dumps({'elapsed_seconds':time.time()-start,'rss_MiB':psutil.Process().memory_info().rss/1024**2,'simulations':0,'models_loaded':0},indent=2))
print(json.dumps(stats,ensure_ascii=False,indent=2));print(pd.DataFrame(frows)[['site','start','end','rows','cli_hash_present_in_source']].to_string(index=False))
