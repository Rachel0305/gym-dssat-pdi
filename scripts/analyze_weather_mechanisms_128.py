"""Evidence tiers and descriptive nested response analysis; no simulations."""
from pathlib import Path
import json,hashlib,io,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/weather_mechanism_evidence_128_20261027';OUT.mkdir(exist_ok=False)
sources={};start=time.time()
def read(p):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    assert p.resolve().is_relative_to(ROOT.resolve())
    b=p.read_bytes();sources[p.relative_to(ROOT).as_posix()]=hashlib.sha256(b).hexdigest();return b
def csv(p):return pd.read_csv(io.BytesIO(read(p)))
a=csv('results/three_route_results_126_20261027/annual_reference_risk_1200.csv');p=csv('results/three_route_results_126_20261027/policy_120.csv')
keys=['site','regime','seed'];risks=[]
for key,g in a.groupby(keys):
    assert len(g)==10
    risks.append(dict(zip(keys,key),loss_years=int(g.yield_delta_pct.lt(-5).sum()),worst_delta_pct=g.yield_delta_pct.min(),worst_two_mean_pct=g.yield_delta_pct.nsmallest(2).mean(),annual_delta_sd=g.yield_delta_pct.std(ddof=1)))
r=pd.DataFrame(risks).merge(p,on=keys,validate='one_to_one');r.to_csv(OUT/'risk_behavior_policy_120.csv',index=False)
g=r.groupby(['site','regime']).agg(n=('seed','count'),mean_loss_years=('loss_years','mean'),median_loss_years=('loss_years','median'),mean_worst_delta_pct=('worst_delta_pct','mean'),mean_worst_two_pct=('worst_two_mean_pct','mean'),mean_annual_delta_sd=('annual_delta_sd','mean'),between_seed_yield_sd=('grain_yield_kg_ha','std'),fixed_sequence_seeds=('fixed_full_sequence','sum'),I_mean=('total_irrigation_mm','mean'),N_mean=('total_n_kg_ha','mean')).reset_index();g.to_csv(OUT/'risk_behavior_group_15.csv',index=False)
effects=[]
for site in ['SY','HL','LC','YC','FQ']:
    z=g[g.site.eq(site)].set_index('regime');d={'site':site}
    for col in ['mean_loss_years','mean_worst_two_pct','between_seed_yield_sd','fixed_sequence_seeds','I_mean','N_mean']:d[col+'_aug_minus_original']=z.loc['augmented',col]-z.loc['historical',col]
    effects.append(d)
e=pd.DataFrame(effects);e.to_csv(OUT/'augmentation_risk_behavior_changes_5.csv',index=False)
plan=json.loads(read('results/forecast_policy_response_127_20261027/frozen_plan.json'));lookup={};source_records={}
for model in plan['models']:
    for entry in model['years']:
        trace=[json.loads(x) for x in read(entry['trace']).decode().splitlines()];assert hashlib.sha256((ROOT/entry['trace']).read_bytes()).hexdigest()==entry['trace_sha256']
        for i,row in enumerate(trace):
            k=(model['site'],model['seed'],entry['year'],i);lookup[k]=('early' if i/len(trace)<1/3 else 'middle' if i/len(trace)<2/3 else 'late');source_records[k]=dict(date=row['date'],I=row['I'],N=row['N'],original_forecast=row['supplied_forecast'])
summary_parts=[];confidence_parts=[];response=[];negative_count=0;negative_min=0.;rows_count=0
path=ROOT/'results/forecast_policy_response_127_20261027/state_response.csv'
# Stream hash and CSV so the full 185k-row diagnostic need not reside in memory.
h=hashlib.sha256()
with path.open('rb') as f:
    for b in iter(lambda:f.read(262144),b''):h.update(b)
sources[path.relative_to(ROOT).as_posix()]=h.hexdigest()
for c in pd.read_csv(path,chunksize=20000):
    rows_count+=len(c);negative_count+=int(c.JS.lt(0).sum());negative_min=min(negative_min,float(c.JS.min()))
    k=list(zip(c.site,c.seed,c.year,c.step));c['phase']=[lookup[x] for x in k]
    c['legal_band']=np.select([c.legal_count.eq(1),c.legal_count.le(4)],['one','two_to_four'],default='five_to_sixteen')
    c['condition']=np.where(c.variant.str.startswith('shuffle'),'shuffle','zero_block')
    c['eligible']=c.legal_count.gt(1);c['eligible_change']=c.action_changed&c.eligible
    c['eligible_input_changed']=c.input_changed&c.eligible
    # Use one shuffle replicate for baseline confidence: original probability repeats identically.
    base=c[c.variant.eq('shuffle_0')&c.eligible].copy();base['confidence_band']=pd.cut(base.original_action_probability,[-.001,.5,.9,.99,1.001],labels=['<=0.5','0.5-0.9','0.9-0.99','>0.99'])
    confidence_parts.append(base.groupby(['site','confidence_band'],observed=True).agg(states=('step','count')).reset_index())
    summary_parts.append(c.groupby(['site','condition','phase','legal_band']).agg(trials=('step','count'),eligible_trials=('eligible','sum'),input_changed=('input_changed','sum'),eligible_input_changed=('eligible_input_changed','sum'),action_changed=('action_changed','sum'),TV_sum=('TV','sum'),TV_max=('TV','max')).reset_index())
    changed=c[c.condition.eq('shuffle')&c.eligible_change]
    for row in changed.itertuples():
        source=source_records[(row.site,row.seed,row.year,row.step)]
        response.append(dict(site=row.site,seed=row.seed,year=row.year,step=row.step,phase=row.phase,date=source['date'],variant=row.variant,donor_step=row.donor_step,legal_count=row.legal_count,original_action=row.original_action,changed_action=row.changed_action,original_executed_I=source['I'],original_executed_N=source['N'],original_action_probability=row.original_action_probability,TV=row.TV,original_forecast=json.dumps(source['original_forecast'])))
s=pd.concat(summary_parts).groupby(['site','condition','phase','legal_band'],as_index=False).agg(trials=('trials','sum'),eligible_trials=('eligible_trials','sum'),input_changed=('input_changed','sum'),eligible_input_changed=('eligible_input_changed','sum'),action_changed=('action_changed','sum'),TV_sum=('TV_sum','sum'),TV_max=('TV_max','max'));s['action_change_pct']=100*s.action_changed/s.trials;s['TV_mean']=s.TV_sum/s.trials;s.to_csv(OUT/'phase_legal_response.csv',index=False)
conf=pd.concat(confidence_parts).groupby(['site','confidence_band'],observed=True,as_index=False).states.sum();conf.to_csv(OUT/'original_action_confidence.csv',index=False)
resp=pd.DataFrame(response);resp.to_csv(OUT/'response_combinations_52.csv',index=False);unique=resp.drop_duplicates(['site','seed','year','step']);assert len(unique)==30;unique.to_csv(OUT/'unique_response_states_30.csv',index=False)
resp.groupby(['site','seed']).agg(changed_combinations=('step','count'),changed_years=('year','nunique'),max_TV=('TV','max')).to_csv(OUT/'responding_seed_concentration.csv')
ph=resp.groupby('phase').size().to_dict();phase_unique=unique.groupby('phase').size().to_dict()
phased=s[s.condition.eq('shuffle')&s.legal_band.ne('one')].groupby('phase').agg(trials=('trials','sum'),changed=('action_changed','sum'));phased['pct']=100*phased.changed/phased.trials;phased.to_csv(OUT/'phase_readout.csv')
single=s[s.condition.eq('shuffle')&s.legal_band.eq('one')].trials.sum()/3;allstates=s[s.condition.eq('shuffle')].trials.sum()/3
matrix=[
dict(explanation='增强实际进入训练',tier='已证实',evidence='123逐回合记录40/40；不同天气种子/指纹',limit='不等于泛化获益，记录不等于独立天气样本'),
dict(explanation='追加预报块动作依赖弱',tier='已证实于归档状态与指定变换',evidence='127全部40模型；52/77658组合改变动作',limit='不推出闭环产量不变，也不推出没有使用任何天气'),
dict(explanation='单合法动作限制响应',tier='已证实的背景约束',evidence=f'{int(single)}/{int(allstates)}原状态仅1合法动作',limit='多合法动作中仍响应弱，不能解释全部现象'),
dict(explanation='近确定性概率或较大动作间隔使argmax稳定',tier='支持关联',evidence='原动作概率分布及TV/argmax对照',limit='不证明由某个训练超参数导致'),
dict(explanation='增强可能改变风险和资源偏好',tier='描述性关联',evidence='15组年度损失/下尾/固定措施/水氮',limit='不配对seed；无完整单因素认证，不能确认因果'),
dict(explanation='基础观测使预报信息冗余',tier='待验证假说',evidence='基础观测包含既往天气、作物/土壤状态',limit='未做基础信息消融或信息增益试验'),
dict(explanation='奖励或约束削弱提前决策的价值',tier='待验证假说',evidence='固定措施和合法动作背景',limit='未改变奖励/约束作受控比较，不可当已证实原因'),
dict(explanation='增强覆盖不足或固定预算使任务更难',tier='待验证假说',evidence='已有训练与测试及路线差异',limit='没有本轮受控学习曲线或覆盖距离实验'),
]
pd.DataFrame(matrix).to_csv(OUT/'mechanism_evidence_matrix.csv',index=False)
lines=['# 天气扩展作用与可能原因：讨论稿','', '本轮仅分析已有126/127数据，没有模型加载、模拟或训练，不扩大天气审查。统一FQ2018采用既有质量控制版本，非独立验证天气真值。','', '## 天气增强：进入训练，但没有一致的风险或资源收益','', '|站点|损失>5%年份数变化|最差两年平均差值变化（百分点）|跨seed产量SD变化（kg/ha）|固定序列策略数变化|','|---|---:|---:|---:|---:|']
for row in e.to_dict('records'):lines.append(f"|{row['site']}|{row['mean_loss_years_aug_minus_original']:+.2f}|{row['mean_worst_two_pct_aug_minus_original']:+.2f}|{row['between_seed_yield_sd_aug_minus_original']:+.1f}|{row['fixed_sequence_seeds_aug_minus_original']:+.0f}|")
lines+=['', '变化为增强减原PPO；最差两年指标基于每策略10年相对各年真实主参照的差值，正值表示下尾差值改善，不是新增独立验证集或CVaR估计。损失年数为每策略十年中损失超过5%的年份数，再对8策略求均值。减少均值损失或SD只属描述性改善，不证明稳健性显著提高；不同站点处理异质，不能合成统一因果效果。水氮变化须同时阅读risk_behavior_group_15.csv，固定序列并不自动等于策略无效。','', '增强改变了训练经历与模型输出分布，但未呈现一致的平均收益、下尾收益和资源改善。现有证据不支持简单归因为“增强没进入训练”。覆盖不足、分布偏移、预算不足或优化困难属于可讨论的替代解释；当前数据不足以确定其中任何一种为主因。','', '## 天气预报：弱响应不只来自没有可选动作','',f'46480个原状态中，{int(single)}个（{100*single/allstates:.2f}%）仅一个合法动作。127主要分析已排除这些状态：在25886个多合法动作状态的3次置换中，52/77658个组合改变动作，涉及30个唯一原状态。因此，约束压缩动作空间是背景因素，但不能单独解释弱响应。','', '|轨迹三等份|置换变化组合/多合法组合|比例|','|---|---:|---:|']
for phase,row in phased.iterrows():lines.append(f'|{phase}|{int(row.changed)}/{int(row.trials)}|{row.pct:.4f}%|')
lines+=['', '轨迹前中后三等份按决策步序定义，不是实测生育阶段。阶段间可选动作与剩余水氮预算不同，不能由比例差异推断关键生理窗口；响应状态清单同时保存日期、原实际投入和置换类别。不同变换为嵌套重复，不作为独立样本检验。','', '概率分布轻微变化而argmax大多不变，说明“预报完全没有进入网络”表述不成立；更合适的是“在已访问状态和指定扰动下，对追加预报的决策依赖较弱”。基础状态仍包含天气与作物信息，网络也可能利用预报改变未测量的价值输出；不能由本诊断证明信息冗余或网络失效。即使动作改变次数很少，少数关键动作仍可能影响闭环产量，本轮未测量此作用。','', '## 论文如何组织这些解释','', '将机制讨论限定为三层：实际训练/响应证据；风险、固定措施、动作空间和置信度的描述性关联；奖励、信息冗余、训练覆盖与预算的待验证假说。论文可以写“增加天气经历或输入不自动转化为稳定的管理收益，需联合验证策略依赖、跨种子重复性与资源代价”，不能写“天气增强/预报没有价值”或已经找到了单一失败原因。','', '下一步优先整合方法、结果与讨论，而非再次训练更大模型或增强+预报组合。若需要闭环补充，30个响应状态可指导有限机制回放，但按响应选择的样本不能用于总体收益估计；无变化轨迹在固定置换方案下也不能直接推断所有其他扰动均无影响。','', '## 数值质量说明','', f'127逐状态JS有{negative_count}条微小负值，最小{negative_min:.3g}，属于float32概率与对数运算的数值非负性误差。原文件保持不变，本轮不以JS负值做解释，主结论基于动作及非负TV。该误差不改变动作复现或52次响应计数。']
(OUT/'discussion_zh.md').write_text('\n'.join(lines),encoding='utf-8')
stats=dict(policy_rows=len(r),group_rows=len(g),response_combinations=len(resp),unique_response_states=len(unique),single_legal_states=int(single),original_states=int(allstates),phase_changes=ph,phase_unique=phase_unique,negative_JS_rows=negative_count,minimum_JS=negative_min,new_simulations=0,models_loaded=0,elapsed_seconds=time.time()-start)
(OUT/'QA.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8');(OUT/'source_sha256.json').write_text(json.dumps(sources,indent=2))
print(json.dumps(stats,ensure_ascii=False,indent=2));print(e.to_string(index=False));print(phased.to_string())
