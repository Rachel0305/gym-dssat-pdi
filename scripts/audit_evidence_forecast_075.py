"""Read-only evidence audit. No simulator, ML runtime, pickle or model loading."""
from pathlib import Path
from collections import Counter
import hashlib,json,zipfile,re
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/evidence_forecast_audit_075_20261022/attempt02';OUT.mkdir(exist_ok=False)
sources={};issues=[]
def path(v):
    p=Path(v);p=p if p.is_absolute() else ROOT/p;p=p.resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Path outside project: '+str(p))
    return p
def digest(p):
    p=path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(262144),b''):h.update(block)
    sources[str(p.relative_to(ROOT))]=h.hexdigest();return h.hexdigest()
def read(v):
    p=path(v);digest(p)
    if p.suffix.lower()=='.json':return json.loads(p.read_text(encoding='utf-8-sig'))
    if p.suffix.lower()=='.csv':return pd.read_csv(p)
    return p.read_text(encoding='utf-8-sig',errors='replace')
def save(d,n):d.to_csv(OUT/(n+'.csv'),index=False,encoding='utf-8-sig')
def md(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in rows])

registry=read('2V02/MODEL_REGISTRY.csv');policies=read('results/benefit_reliability_066_20261019/primary_policy_table_80.csv');annual=read('2V02/deliverables/all_80_policy_800_year_metrics.csv');oldconfig=read('results/advisor_evidence_065_20261018_attempt02/config_file_audit_80.csv')
keys=['site','regime','seed'];assert len(registry)==80 and len(policies)==80 and len(annual)==800
assert not registry.duplicated(keys).any() and not annual.duplicated(keys+['year']).any()
assert annual.groupby(keys).year.apply(lambda x:set(x)==set(range(2014,2024))).all()
models=[];contracts=[]
for r in registry.itertuples():
    f=path(r.model_path)
    if not f.exists():f=path(Path('2V02')/r.model_path)
    sha=digest(f)
    with zipfile.ZipFile(f) as z:meta=json.loads(z.read('data'))
    models.append(dict(site=r.site,regime=r.regime,seed=r.seed,model_path=str(f.relative_to(ROOT)),sha256=sha,registry_match=sha==str(r.model_sha256).lower(),seed_json=meta.get('seed'),timesteps_json=meta.get('num_timesteps'),seed_match=meta.get('seed')==r.seed,steps_100k=meta.get('num_timesteps')==100000))
    parts=f.parts;idx=parts.index('models');runroot=Path(*parts[:idx]);mans=list(runroot.glob('*run_manifest.json'));mf=None;pf={};cfg={}
    if len(mans)==1:
        mf=mans[0];v=read(mf);pf=v.get('preflight',{});cfg=v.get('config',{})
    elif (runroot/'preflight.json').exists():
        mf=runroot/'preflight.json';v=read(mf);pf=v.get('canonical',{}).get('canonical_preflight',v.get('canonical_preflight',{}))
    oc=oldconfig[oldconfig.site.eq(r.site)&oldconfig.regime.eq(r.regime)&oldconfig.seed.eq(r.seed)].iloc[0]
    configfile=path(oc.config_source);digest(configfile)
    if configfile.suffix=='.json':archived=read(configfile)
    else:archived={}
    if not cfg:cfg=archived
    inputfile=pf.get('source_mzx');expected=pf.get('source_mzx_sha256');inputsha=None
    if inputfile and path(inputfile).exists():inputsha=digest(path(inputfile))
    contracts.append(dict(site=r.site,regime=r.regime,seed=r.seed,run_root=str(runroot.relative_to(ROOT)),manifest=str(mf.relative_to(ROOT)) if mf else '',config_source=str(configfile.relative_to(ROOT)),config_role=oc.file_role,input_root=pf.get('resolved_input_root',''),source_mzx=inputfile or '',current_mzx_sha256=inputsha or '',recorded_mzx_sha256=expected or '',recorded_current_mzx_match=(inputsha==str(expected).lower()) if expected else None,declared_input_profile=pf.get('input_profile',cfg.get('input_profile','')),train_years=json.dumps(pf.get('engine_train_years',cfg.get('scope',{}).get('train_years',[]))),validation_years=json.dumps(pf.get('engine_validation_years',cfg.get('scope',{}).get('validation_years',[]))),reward_safety_declaration=cfg.get('scope',{}).get('reward_and_safety',''),ppo_fields_in_saved_json=bool(archived.get('effective_ppo',archived.get('ppo'))),safety_fields_in_saved_json=bool(archived.get('action_safety')),forecast_enabled=cfg.get('observation_contract',{}).get('weather_forecast_enabled'),full_runtime_contract_certified=False))
save(pd.DataFrame(models),'model_identity_metadata_80');save(pd.DataFrame(contracts),'saved_contract_and_input_lineage_80')
print('Model identities and saved lineage inspected.',flush=True)

means=annual.groupby(keys)[['grain_yield_kg_ha','total_irrigation_mm','total_n_kg_ha']].mean().reset_index();check=means.merge(policies,on=keys,suffixes=('_raw','_table'),validate='one_to_one')
mean_errors={m:float(np.abs(check[m+'_raw']-check[m+'_table']).max()) for m in ['grain_yield_kg_ha','total_irrigation_mm','total_n_kg_ha']}
closure=[]
for r in policies.itertuples():
    e=read(r.event_source)
    if r.site=='YC' and r.regime=='historical':e=e.rename(columns={'evaluation_weather_year':'year','irrigation_action_mm':'irrigation_executed_mm','fertilizer_action_kgN_ha':'nitrogen_executed_kg_ha'})
    elif r.site=='YC':e=e.rename(columns={'irrigation':'irrigation_executed_mm','nitrogen':'nitrogen_executed_kg_ha'})
    else:e=e[e.scenario.eq('rl_candidate' if r.site in ['SY','LC'] else f'ppo_seed_{r.seed}')].copy()
    correction=''
    if r.site=='FQ':
        correction=(f'results/fqa_wgen_052_corrected2018_231/seed_{r.seed:02d}/corrected2018_daily.csv' if r.regime=='augmented' else (f'results/hl_fq_8seed_cross_site/055_03_native_auto_corrected2018/replays/FQ/seed{r.seed}/corrected2018_ppo_daily.csv' if r.seed<7 else 'results/version2_fq_seed7_corrected2018_063/seed7/corrected2018_ppo_daily.csv'))
        corrected=read(correction);assert set(corrected.year)=={2018};e=pd.concat([e[e.year.ne(2018)],corrected],ignore_index=True)
    q=annual[annual.site.eq(r.site)&annual.regime.eq(r.regime)&annual.seed.eq(r.seed)]
    for row in q.itertuples():
        y=e[e.year.eq(row.year)];closure.append(dict(site=r.site,regime=r.regime,seed=r.seed,year=row.year,irrigation_error=float(y.irrigation_executed_mm.sum()-row.total_irrigation_mm),nitrogen_error=float(y.nitrogen_executed_kg_ha.sum()-row.total_n_kg_ha),event_source=r.event_source,corrected_source=correction if row.year==2018 else ''))
save(pd.DataFrame(closure),'event_endpoint_closure_800')
baselineparts=[]
for r in registry.itertuples():
    panel=read(f'2V02/deliverables/{r.site}/{r.regime}/seed_{r.seed:02d}/tables/yearly_metrics.csv');panel.scenario=panel.scenario.fillna('null');q=panel[~panel.scenario.str.startswith('ppo')].copy();q['site']=r.site;baselineparts.append(q)
base=pd.concat(baselineparts,ignore_index=True);spread=base.groupby(['site','year','scenario'])[['grain_yield_kg_ha','total_irrigation_mm','total_n_kg_ha']].agg(lambda v:float(v.max()-v.min()));save(spread.reset_index(),'baseline_panel_spread_150')
print('800 endpoint/event totals and baseline panels reconciled.',flush=True)

# Existing observations are evidence leads, not a substitute for model validation.
phen=read('data/observed_phenology_dates_standardized.csv');ph=phen.groupby('station_code').agg(n_records=('year','size'),years=('year',lambda v:','.join(map(str,sorted(v.tolist()))))).reset_index();save(ph,'phenology_observation_inventory')
obs=[]
for profile in ['multisite_new_cultivar_inputs_013','multisite_new_cultivar_inputs_013_lowIC_manual']:
    for site in ['SY','LC','HL','FQ','YC']:
        droot=ROOT/'DSSAT_auto_validation'/profile/site
        for f in droot.glob('*.MZA'):
            columns=None
            for line in read(f).splitlines():
                if line.startswith('@TRNO'):columns=line[1:].split()
                elif columns and line.strip() and not line.lstrip().startswith(('!','*','@')):
                    values=line.split()
                    if len(values)==len(columns):obs.append(dict(profile=profile,site=site,file=str(f.relative_to(ROOT)),**dict(zip(columns,values))))
save(pd.DataFrame(obs),'mza_observation_rows_in_audited_input_roots')
calpath='DSSAT_auto_validation/HLA_2004/cultivar_calibration_HLA2004_480/figures/hla_2007_cal_2009_val_observed_vs_simulated.csv';cal=read(calpath);cal['yield_error_pct']=(cal.sim_hwam/cal.obs_hwam-1)*100;save(cal,'HLA_candidate_observed_simulated')
coeff=[]
for rel in ['DSSAT_auto_validation/HLA_2004/cultivar_calibration_HLA2004_480/input_corrected_package/MZCER048.CUL','DSSAT_auto_validation/multisite_new_cultivar_inputs_013/HL/MZCER048.CUL','DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/HL/MZCER048.CUL','my_data/MZCER048.CUL']:
    lines=[l for l in read(rel).splitlines() if l.startswith('HY0006')];assert len(lines)==1
    vals=list(map(float,lines[0].split()[-6:]));coeff.append(dict(file=rel,sha256=digest(rel),cultivar='HY0006',**dict(zip(['P1','P2','P5','G2','G3','PHINT'],vals)),matches_candidate=np.allclose(vals,cal.iloc[0][['P1','P2','P5','G2','G3','PHINT']].astype(float))))
save(pd.DataFrame(coeff),'HLA_cultivar_coefficients_partial_link')
calroot=ROOT/'DSSAT_auto_validation/HLA_2004/cultivar_calibration_HLA2004_480/input_corrected_package';actroot=ROOT/'DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/HL';calfiles=[]
def section(txt,start):
    lines=txt.splitlines();on=False;out=[]
    for line in lines:
        if line.startswith(start):on=True
        elif on and line.startswith('*'):break
        if on and line.strip() and not line.lstrip().startswith('!'):out.append(' '.join(line.split()))
    return '\n'.join(out)
for name in ['MZCER048.CUL','SOIL.SOL','CNHL0701_corrected_IC123.MZX','CNHL0701.MZA']:
    x=read(calroot/name);y=read(actroot/name);comp=dict(file=name,calibration_sha256=digest(calroot/name),current_lowIC_sha256=digest(actroot/name),whole_file_equal=(calroot/name).read_bytes()==(actroot/name).read_bytes())
    if name.endswith('.MZX'):comp.update(initial_condition_section_equal=section(x,'*INITIAL CONDITIONS')==section(y,'*INITIAL CONDITIONS'),field_section_equal=section(x,'*FIELDS')==section(y,'*FIELDS'))
    if name=='SOIL.SOL':comp['selected_soil_HL99001200_equal']=section(x,'*HL99001200')==section(y,'*HL99001200')
    calfiles.append(comp)
save(pd.DataFrame(calfiles),'HLA_calibration_current_input_comparison')
weatherreport=read('docs/2026-06-06_all_year_weather_calibration_validation_report.md')
assert 'Actual cultivar calibration still requires observed yield' in weatherreport
print('Observation inventories and partial HLA input link inspected.',flush=True)

# Existing same-architecture oracle signal / zero-signal comparison, seed0.
runpaths={'forecast':'benchmark_results/143E2_sya_originIC_dual_branch_weather_reader_5k','zero_signal':'benchmark_results/152E3_sya_originIC_dual_branch_noforecast_control_5k_seed0'}
configs={};pref={};forecastrows=[];fannual=[];faudits=[]
for branch,rel in runpaths.items():
    root=ROOT/rel;prefix='143E2' if branch=='forecast' else '152E3';v=read(root/(prefix+'_run_manifest.json'));configs[branch]=v['config'];pref[branch]=v['preflight']
    e=read(root/'evaluation'/(prefix+'_checkpoint_validation_summary.csv'))
    assert not e.duplicated(['year','checkpoint_step']).any()
    for ck,q in e.groupby('checkpoint_step'):
        assert len(q)==10;fannual.append(q.assign(branch=branch))
        for f in q.model_path.unique():digest(path(f))
        forecastrows.append(dict(branch=branch,checkpoint=int(ck),seed=0,n_years=len(q),yield_mean_kg_ha=float(q.final_grnwt.mean()),irrigation_mean_mm=float(q.total_irrigation.mean()),nitrogen_mean_kg_ha=float(q.total_n.mean()),mean_PFP_N=float(q.PFP_N.mean()),distinct_action_sequences=int(q.action_sequence.nunique())))
    audit=read(root/'audits'/(prefix+'_formal_action_forecast_audit.csv'));faudits.append(dict(branch=branch,rows=len(audit),off_grid_I=int(audit.off_grid_irrigation_rows.sum()),off_grid_N=int(audit.off_grid_nitrogen_rows.sum()),positive_after_day1=int(audit.positive_action_rows_after_dap1.sum())))
    calendar=read(root/'audits'/('143E2_forecast_calendar_audit.csv' if branch=='forecast' else '152E3_zero_weather_calendar_audit.csv'))
    if branch=='forecast':
        assert calendar.future_window_excludes_today.all()
        assert (pd.to_datetime(calendar.future_window_starts)-pd.to_datetime(calendar.current_date)).dt.days.eq(1).all()
save(pd.DataFrame(forecastrows),'SY_forecast_zero_signal_existing_results');save(pd.concat(fannual,ignore_index=True),'SY_forecast_zero_signal_existing_years');save(pd.DataFrame(faudits),'SY_archived_action_audit_checks')
comparison=[]
for component in ['policy_architecture','actions','training','seed']:
    comparison.append(dict(component=component,equal=configs['forecast'].get(component)==configs['zero_signal'].get(component),evidence='archived run manifest configs'))
for component in ['train_years','validation_years','reward_and_safety','external_n','native_dssat_automatic_irrigation']:
    comparison.append(dict(component=component,equal=configs['forecast']['scope'].get(component)==configs['zero_signal']['scope'].get(component),evidence='archived scope declaration; reward implementation still requires runtime lineage'))
for component in ['resolved_input_root','source_mzx','engine_train_years','engine_validation_years']:
    comparison.append(dict(component=component,equal=pref['forecast'].get(component)==pref['zero_signal'].get(component),evidence='saved preflight'))
save(pd.DataFrame(comparison),'SY_same_architecture_comparison')
response=read('benchmark_results/142E2_sya_dual_branch_weather_reader_validation/142E2_validation_summary.json');lc=read('benchmark_results/057_00_lca_lowIC_forecast_engineered_maskableppo_smoke2k_counterfactual_audit/057_00_counterfactual_summary.json');oldpair=read('benchmark_results/140_00_sya_paired_forecast_noforecast_response_gate/140_00_paired_metric_detail.json')
pd.DataFrame([dict(branch='SY142E2_2k',n_states=response['counterfactual']['decision_states'],swap_change_rate=response['counterfactual']['swap_action_change_rate'],shuffle_change_rate=response['counterfactual']['shuffled_action_change_rate'],executed_in_simulator=False,interpretation='fixed-state action response at 2K; not 5K performance effect'),dict(branch='LC057_2k',n_states=lc['n_decision_states'],swap_change_rate=lc['swap_action_change_rate'],shuffle_change_rate=lc['shuffled_action_change_rate'],executed_in_simulator=False,interpretation='zero deterministic action change on visited states; historical replay endpoint discrepancy remains')]).to_csv(OUT/'existing_counterfactual_evidence.csv',index=False,encoding='utf-8-sig')
# Trace current forecast code without importing it.
for f in ['src/run_143E2_sya_dual_branch_weather_reader_5k.py','src/run_152E3_sya_dual_branch_noforecast_control.py','src/forecast_engineered_observation_056_057.py','src/run_sya_originIC_expanded_action_maskableppo_046_10.py','src/run_sya_ppo_configured_046_02.py','src/run_sya_lowIC_binary_timing_maskableppo_042_10.py','docs/2026-08-11_lc_forecast_counterfactual_audit.md','docs/soil_water_and_nitrogen_initialization_summary.md']:
    if path(f).exists():digest(path(f))

M=pd.DataFrame(models);C=pd.DataFrame(contracts);EC=pd.DataFrame(closure);F=pd.DataFrame(forecastrows);F5=F[F.checkpoint.eq(5000)].set_index('branch');gain=float(F5.loc['forecast','yield_mean_kg_ha']-F5.loc['zero_signal','yield_mean_kg_ha']);pct=gain/F5.loc['zero_signal','yield_mean_kg_ha']*100
summary=dict(models=80,model_hash_matches=int(M.registry_match.sum()),unique_model_hashes=M.sha256.nunique(),seed_metadata_matches=int(M.seed_match.sum()),checkpoint_100k=int(M.steps_100k.sum()),annual_rows=800,mean_max_errors=mean_errors,event_max_I_error=float(EC.irrigation_error.abs().max()),event_max_N_error=float(EC.nitrogen_error.abs().max()),baseline_max_spread=float(spread.max().max()),manifest_or_preflight_found=int(C.manifest.ne('').sum()),recorded_MZX_hashes=int(C.recorded_mzx_sha256.ne('').sum()),recorded_MZX_current_mismatches=int(C.recorded_current_mzx_match.eq(False).sum()),HLA_lowIC_coefficients_match_candidate=bool(coeff[2]['matches_candidate']),SY_same_structure_5k_yield_gain=gain,SY_same_structure_5k_yield_gain_percent=pct,forecast_training_authorized_by_this_audit=False,forecast_inference_or_simulation_executed=False,full_nonYC_runtime_certification=False)
(OUT/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
report=f'''# 现有结果可信依据与天气预报可行性审计

日期：2026-10-22。范围为项目内归档文件和当前输入的只读核对；未运行模型、DSSAT、环境构造或训练，未确认当前容器可运行性。不能用静态审计替代实际重放。

## 1. 先给判断

**现有四条结果具备可追溯的计算数据依据，可继续写模拟评估初稿；不等于已完成五站独立农学验证。天气预报具备已有实现与单站继续诊断的条件，尚不支持直接启动五站正式训练。优先复核SY现有同结构有/无天气信号模型，不从零重建。**

## 2. 已核实的数值与模型身份

- 80/80模型文件找到且SHA256与2V02注册表一致；{M.sha256.nunique()}个不同文件哈希。仅读取ZIP中纯JSON，未解序列化权重。
- seed字段一致{int(M.seed_match.sum())}/80；num_timesteps=100000为{int(M.steps_100k.sum())}/80。
- 800年度行唯一，80策略均覆盖2014—2023；十年均值与正文策略表的最大误差：{json.dumps(mean_errors,ensure_ascii=False)}。
- 重新读取80条事件来源，并为16个FQ策略替换修订2018轨迹后，800年度累计灌溉、施氮与端点最大误差分别为{EC.irrigation_error.abs().max():.6g} mm、{EC.nitrogen_error.abs().max():.6g} kg N/ha。
- 各策略面板的三种相同静态基线在站点—年份内一致，最大产量/水氮差异为{spread.max().max():.6g}。

这些检查支持“目前的图、表由同一批已归档模型及一致版本结果产生”，不独立证明DSSAT对实际大田的预测准确。

## 3. 配置与输入来源：比‘完全没有证据’更完整，但仍非全链路认证

{md(['项目','本轮证据','能支持什么','仍不能说什么'],[['模型输入来源',f'{int(C.manifest.ne("").sum())}/80找到运行manifest或preflight','来源与声明可追溯，详见80行表','未保存的完整运行时状态已恢复'],['输入哈希',f'{int(C.recorded_mzx_sha256.ne("").sum())}条有显式MZX哈希，当前不一致{int(C.recorded_current_mzx_match.eq(False).sum())}条','可辨别有签名的模板是否发生变化','没有哈希的历史输入一律与当前相同'],['YC', '之前56项有效配置成对一致记录保留','七类保存设置可比','证明实际天气日序列完整'],['SY/LC外部增强','057/060另有来源合同与摘要哈希审计','不是仅靠文件夹名称比较','直接认证奖励/安全层/二进制只改天气'],['HL/FQ增强','有效配置、canonical preflight、逐episode运行证据存在','可继续追踪实际WGEN/输入覆盖','只读base YAML即可确定最终运行环境']])}

当前80行来源表增加了可追溯信息，但未把静态源码或历史门禁PASS自动升级成完整运行时认证。尤其有效配置中的通用my_data品种文件声明，可能被站点渲染覆盖；没有追到实际临时目录前，不判定为品种用错，也不判定已经无差异。旧072回放问题发生于另一个25K分支，不能直接据此宣告当前80模型失效。

## 4. 校验材料：确认了一个有利的部分关联

1. 五站实测物候标准化表有{len(phen)}条记录，年份分布见phenology_observation_inventory.csv。当前审计的两个站点输入根中，SY、LC、HL存在带产量/物候字段的MZA文件；FQ、YC在这两个目录下未找到MZA，不等于全项目没有原始观测。
2. HLA 2007/2009候选对照CSV实际存在：观测7420/8835 kg/ha，模拟7986/8695 kg/ha，误差约+7.63%/−1.58%。2007为校准、2009为该候选验证，不能把两年都称独立验证。
3. **当前HL站点输入包的HY0006六个系数与该候选完全一致**（P1=235.2、P2=0.494、P5=625.3、G2=278、G3=15.5、PHINT=40）。这是本轮新增核实的关联，比之前笼统说“参数可能不同”更准确。
4. 但HL当前lowIC与候选校验包的整个品种库、土壤文件和MZX并非全部字节相同。分段结果见HLA_calibration_current_input_comparison.csv。my_data通用品种库HY0006参数与站点库也不同。**系数相同只完成品种系数这一环，不能自动证明当前训练/评价使用了相同土壤、初始条件及二进制。**
5. 早期“全年天气校准验证报告”明确写了actual cultivar calibration still requires observed yield/phenology/fixed management records；该报告不构成五站实际校验完成的证据。

因此，论文应写“存在部分观测及HLA候选校验，已建立部分品种关联”，不能写“完全没有观测”，也不能写“全部80策略所用模型已经过五站独立校验”。本轮未复跑这些校验。

## 5. 天气预报不是空白工作：已有可用原始产物

### 5.1 SY同结构预报信号与零信号对照

143E2和152E3保存的manifest中，网络结构、动作网格、seed、训练步数、训练/验证年份、输入根与MZX均一致；scope内奖励/安全继承声明也一致，逐项见SY_same_architecture_comparison.csv。152E3保留天气分支宽度但将输入置零，因此避免直接把网络容量变化当天气信息收益。奖励/安全继承实现和运行时来源仍应复核，不能仅凭声明断言因果闭环。

{md(['分支','检查点','十年均产 kg/ha','灌溉 mm','施氮 kg N/ha','不同年度操作序列数'],[[r.branch,r.checkpoint,f'{r.yield_mean_kg_ha:.2f}',f'{r.irrigation_mean_mm:.1f}',f'{r.nitrogen_mean_kg_ha:.1f}',r.distinct_action_sequences] for r in F.itertuples()])}

5K、seed0时有预报较同结构零信号高{gain:.2f} kg/ha（{pct:.2f}%），两边均灌溉240 mm、施氮240 kg N/ha。**这是一个可继续诊断的单seed线索，不是多seed可靠性证据，也不是节水减氮收益。** 不能与本文100K队列直接混合排名。Calendar原表确认预报从下一天开始，动作审计无离散网格越界且有DAP1后操作；这些证明链路存在，不证明预报决策有效。

### 5.2 信息响应证据存在，但必须分清检查点和含义

- SY142E2的2K归档固定状态审计：{response['counterfactual']['decision_states']}个状态，swap确定性动作变化率{100*response['counterfactual']['swap_action_change_rate']:.2f}%，shuffle为{100*response['counterfactual']['shuffled_action_change_rate']:.2f}%。反事实动作未进入DSSAT，因此没有反事实产量收益。该审计属于2K模型，不能替代143E2的5K模型审计。
- shuffle也能改变动作是输入敏感性的可能表现，**不是仅凭动作变化率就能判断的负对照收益**。当前没有由这些动作改变率证明真实预报优于打乱预报；必须另查农学方向和在同一评价条件下的收益。
- 早期140单分支记录的5K预报均产8807.93，对照10009.38 kg/ha，且水氮投入不同；提示路线与训练过程会影响结果。不可只挑143E2的正向结果。
- LC057旧2K审计：{lc['n_decision_states']}状态swap/shuffle确定性动作变化率均0；旧审计回放均产{lc['real_rollout_yield_mean']:.2f}与旧smoke记录8335.04不一致。该LC分支本轮不作为可靠增益证据，也不直接延长训练；0变化不证明天气信息一般无效。

## 6. 可行性结论与最小下一步

**判断：已有实现和归档支持SY单站预报信息审计；本轮不建议直接新开100K或五站多seed。**

{md(['顺序','最小任务','完成标准','计算安排'],[['1','复核SY143E2/152E3实际输入与运行设置','同一奖励、约束、渲染输入、品种土壤及执行文件能对应；不能只核对config声明','先静态证据，无训练'],['2','对这两个5K冻结模型做预报响应诊断','固定同一已访问状态和动作mask；只在有多个合法动作时解释响应；同时看概率与确定性动作，区分“敏感”和“有益”','需后续最小推理方案；本次未执行'],['3','必要时核对真实/置换预报的冻结策略回放','对同一模型与相同年份分别评价；各条件独立rollout，全部输入与端点可追溯，报告产量水氮及操作时机','需单独计划的少量模拟；先smoke'],['4','再决定是否新的单站受控训练','先固定问题、预算、评价指标、全部检查点报告与停止条件；不要求必须得到正结果','只有前三项明确后再决定，不默认扩8seed或5站']])}

若SY5K没有可解释响应，或回放输入不一致，应先定位实现/证据问题，不自动增加步数。若有响应但没有收益，报告“当前实现下的信息响应未转化为收益”，不推导天气信息不重要。完美未来天气是oracle诊断，不代表现实预报验证；未来真实预报误差研究不纳入当前最小任务。

## 7. 对当前论文的直接影响

- 四条结论可保留为固定历史窗口下的模拟评估；80模型身份、800年结果和事件累计量有可追溯依据。
- 校验范围按上述部分关联写清；未完成的运行时和校验绑定，不再笼统重复成“所有结果不可信”。
- 自适应规则基线的代码和旧输出存在，但本轮未把它们统一成修订FQ2018后的可比五站表，不能说已经补齐。该任务与预报是否可实现是不同问题。
- 当前最有价值的预报增量是复核既有SY信息价值线索，而不是为了加创新点直接开另一批大规模模型。

## 8. 证据文件和执行边界

所有检查均可由scripts/audit_evidence_forecast_075.py复算；读取的输入及流式模型SHA256见source_sha256.json。详细80配置、80模型、800事件闭合、150基线一致性、校验关联、SY原始年度对照与反事实摘要均在本目录。历史文档中提及的门禁状态不是本轮重新执行结果；未验证容器和依赖运行状态，也未进行新的冻结推理或回放。
'''
(OUT/'可信依据与预报可行性审计.md').write_text(report,encoding='utf-8')
digest(Path(__file__));digest(ROOT/'prompts/2026-10-22_075_evidence_and_forecast_feasibility.md')
(OUT/'source_sha256.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False),flush=True)
