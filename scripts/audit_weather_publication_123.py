"""Serial saved-evidence audit. No model imports or simulator execution."""
from pathlib import Path
import json,hashlib,time
import pandas as pd
import psutil
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/weather_publication_audit_123_20261027/attempt02'
OUT.mkdir(exist_ok=False)
start=time.time();sources={}
def read(p):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    assert p.resolve().is_relative_to(ROOT.resolve())
    b=p.read_bytes();sources[p.relative_to(ROOT).as_posix()]=hashlib.sha256(b).hexdigest()
    return b
def js(p):return json.loads(read(p))
def csv(p):
    import io
    return pd.read_csv(io.BytesIO(read(p)),keep_default_na=False)
lineage=csv('results/evidence_forecast_audit_075_20261022/saved_contract_and_input_lineage_80.csv')
old=csv('results/benefit_reliability_066_20261019/primary_policy_table_80.csv')
forecast=csv('results/forecast_publication_audit_122_20261027/forecast_policy_table_40.csv')
rows=[];checks=[]
for r in lineage[lineage.regime.eq('augmented')].itertuples():
    root=ROOT/r.run_root;site=r.site;seed=int(r.seed)
    rec={'site':site,'seed':seed,'run_root':r.run_root,'full_runtime_contract_previously_certified':r.full_runtime_contract_certified,'generator_fitting_leakage_checked':False}
    if site in ['SY','LC']:
        base=Path(f'results/sylc_external_weather_059/{site}/seed{seed}')
        gate=js(base/'gate.json');q=pd.DataFrame(js(base/'weather_manifest.json'))
        q=q[~q.unused_initialization]
        rec.update(episodes=len(q),unique_weather_seeds=q.weather_seed.nunique(),unique_weather_hashes=q.WTH_sha256.nunique(),train_years=','.join(map(str,sorted(q.year.unique()))),injection_pass=gate.get('injection_pass'),runtime_matched_states=gate.get('training_weather_states_matched'),recorded_hash_pairs_match=bool(q.source_sha256.eq(q.WTH_sha256).all()),evidence=str(base))
        samples=q.iloc[sorted(set([0,len(q)//2,len(q)-1]))]
        for z in samples.itertuples():
            p=ROOT/z.WTH
            checks.append(dict(site=site,seed=seed,path=z.WTH,exists=p.exists(),hash_match=hashlib.sha256(read(p)).hexdigest().lower()==z.WTH_sha256.lower() if p.exists() else None))
    else:
        name='episode_manifest.csv' if site=='FQ' else 'training_episode_summary.csv'
        q=csv(root/name)
        if 'split' in q:q=q[q.split.eq('train')]
        if 'status' in q:q=q[q.status.eq('completed')]
        year='year' if site=='FQ' else 'historical_year';ws='weather_seed' if site=='FQ' else 'training_weather_seed';wh='weather_sha256' if site=='FQ' else 'runtime_weather_sha256'
        rec.update(episodes=len(q),unique_weather_seeds=q[ws].nunique(),unique_weather_hashes=q[wh].nunique(),train_years=','.join(map(str,sorted(q[year].unique()))),recorded_hash_pairs_match=None,injection_pass=None,runtime_matched_states=None,evidence=str((root/name).relative_to(ROOT)))
        if site!='FQ':rec.update(weather_source=','.join(map(str,q.training_weather_source.unique())),wther=','.join(map(str,q.filex_wther.unique())),seed_transmission_matches=bool(q.scheduled_rseed1_.eq(q.actual_rseed1_sent_to_pdi).all()))
        else:
            for z in q.iloc[sorted(set([0,len(q)//2,len(q)-1]))].itertuples():
                p=ROOT/z.weather_path
                checks.append(dict(site=site,seed=seed,path=z.weather_path,exists=p.exists(),hash_match=hashlib.sha256(read(p)).hexdigest().lower()==z.weather_sha256.lower() if p.exists() else None))
                if hasattr(z,'runtime_evidence_path') and (ROOT/z.runtime_evidence_path).exists():
                    rt=js(z.runtime_evidence_path);checks.append(dict(site=site,seed=seed,path=z.runtime_evidence_path,exists=True,hash_match=None,wther=rt.get('wther'),seed_transmission_matches=rt.get('runtime_rseed1')==rt.get('scheduled_seed')))
    years=set(map(int,rec['train_years'].split(',')));rec['overlap_validation_years']=','.join(map(str,sorted(years&set(range(2014,2024)))))
    rows.append(rec)
aug=pd.DataFrame(rows);assert len(aug)==40
aug.to_csv(OUT/'augmentation_training_evidence_40.csv',index=False)
pd.DataFrame(checks).to_csv(OUT/'runtime_sample_checks.csv',index=False)
# Existing frozen references versus corrected current references. Numerical equality is not contract identity.
baseparts=[]
for site in ['SY','HL','LC','YC','FQ']:
    panel=csv(f'2V02/deliverables/{site}/historical/seed_00/tables/yearly_metrics.csv')
    panel=panel[~panel.scenario.str.startswith('ppo')].copy();panel['site']=site;baseparts.append(panel)
base=pd.concat(baseparts,ignore_index=True)
current=csv('results/forecast_five_scenario_120_corrected_HL_20261027/all_five_sites_seed_year_scenario.csv')
cols=['grain_yield_kg_ha','actual_irrigation_mm','actual_nitrogen_kg_ha']
new=current[current.scenario.ne('rl_candidate')].groupby(['site','year','scenario'],as_index=False)[cols].mean().rename(columns={'actual_irrigation_mm':'total_irrigation_mm','actual_nitrogen_kg_ha':'total_n_kg_ha'})
comparison=base.merge(new,on=['site','year','scenario'],how='outer',suffixes=('_old','_current'),indicator=True)
for m in ['grain_yield_kg_ha','total_irrigation_mm','total_n_kg_ha']:comparison[m+'_delta']=comparison[m+'_current']-comparison[m+'_old']
comparison.to_csv(OUT/'reference_endpoint_comparison.csv',index=False)
allpol=pd.concat([old,forecast],ignore_index=True)
summary=allpol.groupby(['site','regime']).agg(n=('seed','count'),yield_mean=('grain_yield_kg_ha','mean'),yield_sd=('grain_yield_kg_ha','std'),yield_min=('grain_yield_kg_ha','min'),yield_max=('grain_yield_kg_ha','max'),I_mean=('total_irrigation_mm','mean'),N_mean=('total_n_kg_ha','mean'),fixed_schedules=('fixed_full_sequence','sum')).reset_index()
assert len(summary)==15;summary.to_csv(OUT/'three_route_descriptive_15.csv',index=False)
annual=csv('results/paper_results_tables_074_20261022/S3_主参照年度差值_800.csv')
annual['loss_over_5pct']=annual.yield_delta_pct.lt(-5)
annual.groupby(['site','regime','seed']).agg(years=('year','count'),loss_years=('loss_over_5pct','sum'),worst_delta_pct=('yield_delta_pct','min')).to_csv(OUT/'legacy_annual_risk_per_seed_80.csv')
diag=js('results/sy_forecast_diagnostic_092_20261022/result.json')
ds=pd.DataFrame(diag['summaries']);ds.to_csv(OUT/'existing_forecast_permutation_12.csv',index=False)
sh=ds[ds.kind.eq('shuffle')]
if sh.empty:sh=ds[ds.kind.ne('replay')]
t4=csv('results/paper_results_tables_074_20261022/T4_天气增强对照_15项.csv')
differences=comparison[comparison.filter(like='_delta').abs().gt(1.0).any(axis=1)]
stats={'augmentation_models':len(aug),'episodes_recorded':int(aug.episodes.sum()),'training_validation_year_overlap_models':int(aug.overlap_validation_years.ne('').sum()),'sample_weather_hash_failures':sum(z.get('hash_match') is False for z in checks),'reference_changed_rows':len(differences),'reference_changed_sites':sorted(differences.site.unique().tolist()),'permutations':len(sh),'same_state_changed':int(sh.same_state_changed.sum()),'same_state_eligible':int(sh.same_state_eligible.sum()),'permutations_with_action_change':int(sh.same_state_changed.gt(0).sum()),'max_abs_yield_delta':float(sh.delta_yield.abs().max()),'augmented_intervals_contain_zero':int(t4.interval_includes_zero.sum())}
report=f'''# 123 天气实验论文证据审计

本轮仅读取已有记录，没有模型加载、模拟或训练。结果为第一阶段审计，不代表已完成统一重评或40模型机制实验。

## 天气增强是否进入训练

全部40个增强策略均找到逐回合证据，共{stats['episodes_recorded']}条完成/注入记录；各模型天气种子、天气指纹及年份见augmentation_training_evidence_40.csv。SY/LC注入记录与源天气指纹逐条比较；FQ抽查运行证据；HL/YC核对WTHER与实际传递种子。当前文件抽查见runtime_sample_checks.csv，不把抽查称全量物理复验。训练年份与2014—2023交集模型数为{stats['training_validation_year_overlap_models']}，但这不能排除天气生成器拟合所用数据泄漏：拟合窗口尚未逐站闭合。

## 评价可比性

旧与当前参照端点发生变化的站点为{stats['reference_changed_sites']}，明细见reference_endpoint_comparison.csv。HL原生auto替换属已授权的对照修正；FQ差异仍须绑定输入版本后统一重评。相同端点不能证明WTH/SOL/品种/初始条件或奖励合同完全相同。075的full_runtime_contract_certified均为False，代表未认证，不能误读为已证实不同。保留120暂存统计，不作统一因果比较。

## 增强改变了什么、是否有用

three_route_descriptive_15.csv保存三路线的均值、种子标准差、极值、投入与固定日程数。legacy_annual_risk_per_seed_80.csv保存旧80策略年度损失次数；不能将800个年份当800个独立模型，也不能按编号直接配对。旧增强对照15项区间中{stats['augmented_intervals_contain_zero']}项包含0，只支持未发现一致收益，不证明无效或等效。行为变化与风险变化是描述性结果，不能直接归因于增强；需控制运行合同与生成器拟合窗口。不同站点的外部天气银行与WGEN路线不能当同一个统一处理。

## 已有预报机制诊断与叙述纠正

SY seed0已有3次复现、9次预报置换。重算：同状态多合法动作时点改变{stats['same_state_changed']}/{stats['same_state_eligible']}，{stats['permutations_with_action_change']}/9次置换出现同状态动作变化，最大绝对产量差{stats['max_abs_yield_delta']:.6f} kg/ha。原092报告结尾“未发现响应、完全相同”措辞过强，与其表内2023年的微小变化不一致；本轮保留原文件，纠正为“指定样本中响应极弱，不能将收益归因于充分利用预报”。这不是40模型普遍性结论。打乱预报不代表业务预报误差。

## 下一阶段冻结门槛

1. 先逐站绑定三路线测试WTH/SOL/品种/初始条件、对照、动作/奖励及模型身份；FQ列为必须解决的版本差异，不先增训。
2. 生成器拟合年份及实际训练天气分布须与测试数据隔离；再定义干旱/热年，不能按模型失败年份挑选异常天气。
3. 预报诊断预先固定模型全集、年份及变换种子，先单模型单年smoke；固定状态下比较动作/概率，闭环比较收益，二者分别报告。
4. 原PPO/增强/预报保持分列；新增增强+预报须明确互补性问题后再设计四组实验。

本轮数据未改写。严禁将本审计表视为已统一输入的正式120模型论文终稿。
'''
(OUT/'report_zh.md').write_text(report,encoding='utf-8')
(OUT/'audit_summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'source_sha256.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'resources.json').write_text(json.dumps({'elapsed_seconds':time.time()-start,'rss_MiB':psutil.Process().memory_info().rss/1024**2,'models_loaded':0,'simulations':0,'training_runs':0},indent=2))
print(json.dumps(stats,ensure_ascii=False,indent=2))
