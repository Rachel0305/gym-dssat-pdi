"""Build evidence-linked manuscript results and tables without new experiments."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import t
from statistics import NormalDist

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/paper_results_tables_074_20261022';OUT.mkdir(exist_ok=False)
paths={
 'policy':'results/benefit_reliability_066_20261019/primary_policy_table_80.csv',
 'refs':'results/benefit_reliability_066_20261019/reference_real_scenarios.csv',
 'table1':'results/paper_core_069_20261020/main_table_10_cohorts.csv',
 'table2':'results/paper_repeatability_070_20261020/repeatability_table_10.csv',
 'table3':'results/paper_operations_071_20261020/final_v2/site_counts_5.csv',
 'annual':'results/paper_operations_071_20261020/final_v2/annual_source_800.csv',
 'table4':'results/paper_weather_072_20261020/final_v2/weather_contrasts_15.csv',
 'prompt':'prompts/2026-10-22_074_results_tables_and_text.md'}
data={k:pd.read_csv(ROOT/v) for k,v in paths.items() if k!='prompt'}
p=data['policy'];keys=['site','regime','seed'];sites=['SY','LC','YC','HL','FQ'];routes=['historical','augmented']
assert len(p)==80 and not p.duplicated(keys).any();assert p.groupby(['site','regime']).size().eq(8).all()
a=data['annual'];assert len(a)==800 and not a.duplicated(keys+['year']).any();assert a.groupby(keys).size().eq(10).all()
assert a.year.between(2014,2023).all()
for site in sites:
    ref=data['refs'][data['refs'].site.eq(site)].sort_values('grain_yield_kg_ha',ascending=False).iloc[0]
    q=p[p.site.eq(site)];assert q.reference_scenario.eq(ref.reference).all()
    assert a.loc[a.site.eq(site),'reference_scenario'].eq(ref.reference).all()
    for raw,base in [('grain_yield_kg_ha','reference_yield'),('total_irrigation_mm','reference_I'),('total_n_kg_ha','reference_N')]:assert np.allclose(q[base],ref[raw])
re_y=(p.grain_yield_kg_ha/p.reference_yield-1)*100
gain=re_y.ge(1);saving=re_y.ge(-5)&p.total_irrigation_mm.le(p.reference_I)&p.total_n_kg_ha.le(p.reference_N)&(p.total_irrigation_mm.le(.9*p.reference_I)|p.total_n_kg_ha.le(.9*p.reference_N))
assert gain.eq(p.yield_gain).all() and saving.eq(p.yield_preserving_saving).all()
assert [int(gain.sum()),int(saving.sum()),int((gain&saving).sum()),int((gain|saving).sum())]==[13,31,1,43]
T1=data['table1'].copy();T2=data['table2'].copy();T3=data['table3'].copy();T4=data['table4'].copy()
for site in sites:
    for route in routes:
        q=p[p.site.eq(site)&p.regime.eq(route)];r1=T1[T1.site.eq(site)&T1.route.eq(route)].iloc[0];r2=T2[T2.site.eq(site)&T2.route.eq(route)].iloc[0]
        for col in ['grain_yield_kg_ha','total_irrigation_mm','total_n_kg_ha','yield_delta_kg_ha','I_delta_mm','N_delta_kg_ha']:
            assert np.isclose(q[col].mean(),r1[col+'_mean'])
        g=int(q.yield_gain.sum());s=int(q.yield_preserving_saving.sum());overlap=int((q.yield_gain&q.yield_preserving_saving).sum());k=g+s-overlap
        assert [g,s,overlap,k]==[r2.gain_count,r2.saving_count,r2.overlap_count,r2.benefit_count]
        z=NormalDist().inv_cdf(.975);ph=k/8;den=1+z*z/8;center=(ph+z*z/16)/den;half=z*np.sqrt(ph*(1-ph)/8+z*z/(4*64))/den
        assert np.allclose([max(0,center-half),min(1,center+half)],[r2.wilson95_low,r2.wilson95_high])
        assert np.allclose([q.grain_yield_kg_ha.max(),q.grain_yield_kg_ha.median()],[r2.best_yield_kg_ha,r2.median_yield_kg_ha])
T1['yield_delta_pct_mean']=(T1.grain_yield_kg_ha_mean/T1.reference_yield-1)*100
p['N1']=p.yield_delta_pct.ge(-5-1e-8);p['N2']=p.N1&p.total_irrigation_mm.le(p.reference_I)&p.total_n_kg_ha.le(p.reference_N)
p['N3']=p.N2&~p.small_15mm_fraction.gt(2/3)&~p.frequency_ratio_expert.gt(1.5)
annualstats=a.groupby(keys).agg(annual_min=('yield_delta_pct','min'),annual_loss_gt5_count=('yield_delta_pct',lambda v:int(v.lt(-5).sum())))
p=p.merge(annualstats,on=keys,validate='one_to_one');assert np.allclose(p.annual_min,p.worst_year_yield_delta_pct)
for site in sites:
    q=p[p.site.eq(site)];old=T3[T3.site.eq(site)].iloc[0]
    assert [int(q.N1.sum()),int(q.N2.sum()),int(q.N3.sum())]==[old.N1,old.N2,old.N3]
    assert int((q.N3&q.fixed_full_sequence).sum())==old.N3_fixed
    for n in ['N2','N3']:assert int((q[n]&q.annual_loss_gt5_count.gt(0)).sum())==old[n+'_with_annual_loss_gt5']
for row in T4.itertuples():
    h=p.loc[p.site.eq(row.site)&p.regime.eq('historical'),row.metric];v=p.loc[p.site.eq(row.site)&p.regime.eq('augmented'),row.metric]
    vh=h.var(ddof=1)/8;va=v.var(ddof=1)/8;df=(vh+va)**2/(vh*vh/7+va*va/7);delta=v.mean()-h.mean();half=t.ppf(.975,df)*np.sqrt(vh+va)
    assert np.allclose([delta,df,delta-half,delta+half],[row.delta_augmented_minus_historical,row.welch_df,row.welch95_low,row.welch95_high],rtol=0,atol=1e-8)

def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def route(r):return '未增强' if r=='historical' else '增强'
def save_table(name,frame,text):
    frame.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig');(OUT/(name+'.md')).write_text(text,encoding='utf-8')
S0=p.groupby('site',sort=False)[['reference_scenario','reference_yield','reference_I','reference_N']].first().reindex(sites).reset_index()
s0text='# 表S0 五站主参照方案\n\n'+mdtable(['站点','实际参照','产量 kg/ha','灌溉 mm','施氮 kg N/ha'],[[r.site,'记录模板' if r.site=='LC' else '专家',f'{r.reference_yield:.2f}',f'{r.reference_I:.1f}',f'{r.reference_N:.1f}'] for r in S0.itertuples()])+'\n\n依据两种实际方案的十年均产选择较高者；不是逐年最大值拼接。投入取同一方案自身值。操作频率参考另以专家为准。\n'
save_table('S0_五站主参照',S0,s0text)
t1text='# 表T1 各站点与训练路线的产量和投入\n\n'+mdtable(['站点','路线','产量 kg/ha','产量差 %','灌溉 mm','灌溉差 mm','施氮 kg N/ha','施氮差 kg N/ha'],[[r.site,route(r.route),f'{r.grain_yield_kg_ha_mean:.1f}',f'{r.yield_delta_pct_mean:+.2f}',f'{r.total_irrigation_mm_mean:.1f}',f'{r.I_delta_mm_mean:+.1f}',f'{r.total_n_kg_ha_mean:.1f}',f'{r.N_delta_kg_ha_mean:+.1f}'] for r in T1.itertuples()])+'\n\n每行n=8，所列均为组均值，差值相对表S0同一实际方案；产量差率按组均产相对参照计算。完整CSV另含绝对产量差、中位数和范围。R1散点图中的黑色横线是中位数，与本表组均值不同。\n'
save_table('T1_产量水氮投入_10队列',T1,t1text)
t2text='# 表T2 收益重复数与最高产量的代表性\n\n'+mdtable(['站点','路线','增产型','节资型','重叠','并集/8','Wilson 95%区间','最高产量差 %','中位产量差 %','最高减中位 kg/ha'],[[r.site,route(r.route),r.gain_count,r.saving_count,r.overlap_count,f'{r.benefit_count}/8',f'{100*r.wilson95_low:.1f}–{100*r.wilson95_high:.1f}%',f'{r.best_yield_delta_pct:+.3f}',f'{r.median_yield_delta_pct:+.3f}',f'{r.best_minus_median_kg_ha:.2f}'] for r in T2.itertuples()])+'\n\n增产型为十年均产≥主参照101%；节资型为均产≥参照95%、两投入均不增加且至少一项减少≥10%。并集=两类之和−重叠。阈值事后探索性固定；Wilson为独立训练二项近似，不是未来天气或田间成功概率。最高产量不是综合收益最优。\n'
save_table('T2_收益重复性_10队列',T2,t2text)
t3text='# 表T3 产量、投入和操作信息对应的嵌套集合\n\n'+mdtable(['站点','总策略','产量条件N1','再加投入N2','再加操作N3','投入条件减少','操作参考减少','N3内固定日程'],[[r.site,r.n_policies,r.N1,r.N2,r.N3,r.removed_by_input_condition,r.removed_by_operation_reference,r.N3_fixed] for r in T3.itertuples()]+[['总计',80,51,31,19,20,12,7]])+'\n\nN1为十年均产≥主参照95%；N2再要求水氮均不增加；N3再要求≤15mm小灌事件占比≤2/3、频率≤专家1.5倍。固定日程保留。该集合起点不是收益并集43，不能与旧43→31→12混用。探索性参考不代表农学不可行判决，19不是部署名单。\n'
save_table('T3_投入操作信息_5站',T3,t3text)
metricnames={'grain_yield_kg_ha':'产量 kg/ha','total_irrigation_mm':'灌溉 mm','total_n_kg_ha':'施氮 kg N/ha'}
t4text='# 表T4 天气增强路线减未增强路线的逐站差值\n\n'+mdtable(['站点','指标','未增强均值','增强均值','均值差','Welch 95%区间'],[[r.site,metricnames[r.metric],f'{r.historical_mean:.2f}',f'{r.augmented_mean:.2f}',f'{r.delta_augmented_minus_historical:+.2f}',f'[{r.welch95_low:.2f}, {r.welch95_high:.2f}]'] for r in T4.itertuples()])+'\n\n每路线8个模型的十年均值；独立两样本Welch区间，不配对seed。15个逐项区间未校正多重比较。零点是增强与未增强相同，不是管理基线。五站增强处理不同，不合并总体因果效应。全部区间包含零，不证明等效或无效。\n'
save_table('T4_天气增强对照_15项',T4,t4text)
S1=T3[['site','N2','N2_with_annual_loss_gt5','N3','N3_with_annual_loss_gt5']].copy();S1['N2_without_annual_loss_gt5']=S1.N2-S1.N2_with_annual_loss_gt5;S1['N3_without_annual_loss_gt5']=S1.N3-S1.N3_with_annual_loss_gt5
s1text='# 表S1 满足十年产量/投入条件后的年度表现\n\n'+mdtable(['站点','N2策略数','其中至少一年减产>5%','N3策略数','其中至少一年减产>5%'],[[r.site,r.N2,r.N2_with_annual_loss_gt5,r.N3,r.N3_with_annual_loss_gt5] for r in S1.itertuples()]+[['合计',31,15,19,14]])+'\n\n每策略2014—2023各年对同一实际主参照当年产量计算百分比。每策略最多计一次，年份不作为独立训练样本。两组为嵌套集合，15/31与14/19差异不表示操作筛选造成年度风险增加。FQ的10个N2策略中9个最差年为修订输入后的2018。年度5%仅作补充描述，不新增筛除条件。\n'
save_table('S1_年度风险_5站',S1,s1text)
p.to_csv(OUT/'S2_完整策略及核对标记_80.csv',index=False,encoding='utf-8-sig')
a.to_csv(OUT/'S3_主参照年度差值_800.csv',index=False,encoding='utf-8-sig')

lc=p[p.site.eq('LC')&p.yield_gain];ss=p[p.yield_preserving_saving]
assert len(lc)==12 and int(ss.yield_delta_pct.ge(-1).sum())==20 and int(ss.yield_delta_pct.ge(0).sum())==15
lcmeans=T1[T1.site.eq('LC')].set_index('route')
results=f'''# 3. Results（中文初稿）

本节基于五个站点、两条训练路线各8次独立训练所得的80个PPO策略。所有策略均使用固定100K检查点，并在2014—2023历史天气下评价，共800个策略年度结果。除天气路线间直接对照外，产量与投入均相对本站主参照报告：LC采用记录管理模板，SY、YC、HL和FQ采用专家方案（表S0）。正文图暂以R1—R4、表以T1—T4作工作编号，全文排版时统一编号。

## 3.1 不同站点的收益类型与水氮投入权衡

所测试PPO策略的产量和投入表现因站点而异（图R1；表T1）。相对主参照，13/80个策略达到十年均产增加至少1%的增产条件，其中LC占12个、SY占1个。LC增产策略的产量增加{lc.yield_delta_kg_ha.min():.1f}—{lc.yield_delta_kg_ha.max():.1f} kg/ha（中位数{lc.yield_delta_kg_ha.median():.2f} kg/ha），灌溉增加{lc.I_delta_mm.min():.1f}—{lc.I_delta_mm.max():.1f} mm（中位数{lc.I_delta_mm.median():.1f} mm），施氮减少10—170 kg N/ha。该增产子集不代表整条训练路线的平均收益：LC未增强与增强路线的组均产相对模板分别变化{lcmeans.loc['historical','yield_delta_kg_ha_mean']:+.1f}和{lcmeans.loc['augmented','yield_delta_kg_ha_mean']:+.1f} kg/ha。

另有31/80个策略满足产量损失不超过5%、灌溉和施氮均不增加且至少一项节省10%的节资条件，分布于SY 10个、HL 10个、FQ 10个和YC 1个，LC为0个。其中20个策略的产量不低于主参照99%，15个不低于主参照产量。增产型与节资型重叠1个，并集为43个（表T2）。因此，当前队列同时包含追加灌溉下的增产和不同产量变化下的节资，不能将两类收益解释为同一种管理优势。

## 3.2 收益重复比例与最高产量的代表性

满足上述任一收益条件的训练重复数在站点与路线之间不同（图R2a；表T2）。未增强与增强路线分别为SY 6/8与4/8、LC 6/8与6/8、YC 0/8与1/8、HL 6/8与4/8、FQ 5/8与5/8。LC的收益主要来自增产，其他站点主要来自节资。n=8条件下，6/8对应的Wilson 95%区间为40.9%—92.9%，0/8和1/8的区间分别为0—32.4%和2.2%—47.1%；这些区间描述固定验证窗口下的训练重复不确定性，不作为未来应用成功率。

最高产量与组内中位产量的差距也因站点和路线而异（图R2b）。YC两路线的最高产量均接近主参照，而中位产量分别低约20.11%和19.97%，最高产量比中位数高约1650.9和1640.0 kg/ha。LC未增强路线最高与中位产量分别较模板增加10.42%和4.98%，增强路线分别增加5.63%和2.63%。这些结果区分了最高产模型与典型训练结果；最高产量本身不表示综合收益最优。

## 3.3 加入投入与操作信息后的评价集合变化

从仅依据十年产量的条件开始，80个策略中51个均产达到主参照95%及以上；增加灌溉与施氮均不增加的条件后，剩31个；进一步要求小灌占比和灌溉频率不超过探索性参考线后，剩19个（图R3a；表T3）。逐站变化分别为SY 10→10→6、LC 13→0→0、YC 3→1→1、HL 10→10→3和FQ 15→10→9。LC的变化反映其满足产量条件的策略均未同时满足两项投入不增加条件，不表示LC增产没有收益。

在上述31个策略中，12个超过小灌≤15 mm事件占比2/3或灌溉频率为专家1.5倍的至少一条参考线。19个未超线策略中，7个十年操作日程完全固定，并作为标签保留而未排除。产量、投入和操作信息因而对应不同的评价维度；探索性参考线不构成田间可行性判定，19个策略也不作为部署推荐名单。

作为补充年度分析，31个满足十年产量与投入条件的策略中，15个至少有一年产量低于当年同一主参照超过5%；19个未超操作参考线的策略中仍有14个（表S1；年度风险补充图）。FQ的10个此类节资策略中9个最差年为2018，相关结果以修订后的年度输入为条件。两集合相互嵌套，比例变化不表示操作筛选导致年度风险增加；该年度参考也未加入原有筛选条件。

## 3.4 天气增强路线的逐站产量与投入对照

天气增强与未增强路线的对照未呈现一致的均值变化方向（图R4；表T4）。增强减未增强的产量差分别为SY −481.65、LC −328.96、YC −247.19、HL −324.00和FQ +60.69 kg/ha。五站产量差的Welch 95%区间均包含零，其中SY约为[−1345.6, +382.3] kg/ha，YC约为[−1627.6, +1133.3] kg/ha。各路线内部的原始策略结果与组均值一并展示，差值区间以训练策略为单位计算，而未将验证年份视为独立重复。

投入变化方向同样不一致。SY和HL的灌溉与施氮点估计均降低；LC和YC灌溉降低而施氮增加；FQ两项投入点估计均增加。五站灌溉差及施氮差的95%区间也均包含零（表T4）。这些逐站结果在当前训练路线、检查点和历史验证窗口下未显示一致的产量与资源投入收益，但不能据区间跨零推断增强无效或两路线等效。五站增强处理不同，且非YC运行时合同仍待完整核实，故不估计统一的跨站增强因果效应。
'''
(OUT/'Results_中文初稿.md').write_text(results,encoding='utf-8')
mapping=[
 ('1','收益类型和投入权衡','R1','results/paper_core_069_20261020/core_80_policies.pdf','T1、S0、T2','3.1','R1横线中位数；T1列组均值；LC增产子集不等于路线均值'),
 ('2','收益重复性和冠军代表性','R2','results/paper_repeatability_070_20261020/benefit_repeatability.pdf','T2','3.2','a为Wilson比例；b为最高和中位，不是综合最优'),
 ('3','产量投入操作联合评价','R3a','results/paper_operations_071_20261020/intuitive_v3/conclusion3_zh.pdf','T3、S1','3.3','当前文件仍含年度风险b；正文仅用a的拆分导出尚未执行'),
 ('4','天气增强逐站对照','R4','results/paper_weather_estimation_073_20261021/final_v2/yield_estimation_zh.pdf','T4','3.4','本图横线为组均值；下层Welch区间；水氮图配套'),
]
pd.DataFrame(mapping,columns=['结论编号','主题','工作图号','当前图件路径','配套表','结果小节','注意']).to_csv(OUT/'图表段落对应.csv',index=False,encoding='utf-8-sig')
terminology='''# 术语与口径

|统一用语|定义或注意|
|---|---|
|未增强路线|historical-weather training；不称空白对照|
|增强路线|augmented-weather training；各站增强不同|
|主参照|LC模板，其余专家；同一方案自身水氮量|
|训练重复|每路线8个seed模型；不称随机生成方案|
|增产型|十年均产≥参照101%|
|产量损失≤5%的节资|两投入不增加、至少一项减少10%；不直接简称保产|
|收益重复比例|两类收益并集在8次训练中的比例；不是田间成功率|
|操作参考|小灌占比与灌溉频率的探索性条件；不作失败判定|
|固定日程|十年完整动作序列相同，仅作标签|
|均值/中位数|069黑线中位数，070最高与中位数，073黑线均值|
|Wilson区间|收益计数的二项近似区间|
|Welch区间|增强减未增强均值差的独立两样本区间|
|统计样本|策略十年均值；年度为策略内重复测量|
'''
(OUT/'术语与统计口径.md').write_text(terminology,encoding='utf-8')
readme='''# 四条结论的表格与结果文字包

日期2026-10-22。无新增训练、模拟或模型加载。中文Results按已确认主线编写，不是全文定稿。

## 优先阅读

1. [Results中文初稿](Results_中文初稿.md)：四节可直接修改的结果文字。
2. [配套表格总览](配套表格总览.md)：T1—T4正文候选表、S0和S1补充表及全部表注；各表另有独立CSV和Markdown。
3. 图表段落对应.csv：每条结论的图、表、段落和应注意的统计定义。

工作图号R1—R4、工作表号T1—T4/S0—S3尚未映射全文最终编号；五站地图、方法表等不在本次重排范围。R3当前仍为上下组合文件，将上半幅单独导出尚未执行；不能说本次已拆图。R4以产量估计图作为候选，灌溉和施氮配套图保留在073目录。

## 四条编号结论

1. 收益类型因站点而异。13/80增产≥1%，LC占12；LC增产伴随追加灌溉、减少施氮。另31个满足损失≤5%的节资条件，不能混为同一收益。
2. 收益在部分站点的训练重复中出现，但比例不同；最高产量未必代表典型结果。YC最高接近基线，中位低约20%。
3. 投入与操作信息补充产量评价：51→31→19。固定日程保留；年度表现另作补充，不把19视为推荐名单。
4. 当前各站天气增强路线未呈现一致产量与投入收益。全部15项Welch区间跨零，不等于证明无效或等效。

## 本轮证据核对

T1从80策略均值复核；T2重算收益定义、交集和并集、Wilson区间、最高/中位；T3从策略级重算条件数；年度风险从800年度行重算；T4重算Welch并核对072。完整CSV保留精度，阅读表只作四舍五入。正文结果数字由上述验证表和原始策略表支撑。

## 尚未闭合的依据与措辞边界

- 非YC的实际运行时合同未全部核实；不能声称两路线绝对只改天气。
- 历史校准材料未全面绑定当前模型的品种、土壤、初始条件；不能声称五站当前模型已完成独立校验。
- 自适应规则基线仍有缺口；本稿结果只针对已核验参照，不回答对所有简单反馈规则的优劣。
- 所有收益是DSSAT模拟结果；没有田间部署、自主最优选策或未来天气可靠性证据。
- 门槛是事后探索性定义，固定日程不排除。正文不能将策略集合缩小表述成训练本身变好。

## 数据和脚本

S2完整策略分析80行及S3主参照年度差值800行在本目录，供结果核查。更多年度资源端点仍以原2V02/066归档为准，S3仅为年度产量相对参照差值，不冒充含水氮的完整年度端点表。source_sha256.json保存输入和脚本哈希。执行脚本scripts/assemble_results_tables_074.py为轻量表格整理。
'''
(OUT/'README_四条结论配套材料.md').write_text(readme,encoding='utf-8')
(OUT/'配套表格总览.md').write_text('\n\n---\n\n'.join([s0text,t1text,t2text,t3text,t4text,s1text]),encoding='utf-8')
evidence=[
 {'claim':'LC增产伴随追加灌溉和减氮','evidence':'LC满足增产条件12策略连续指标、T1/T2及S2','status':'supported within simulation','boundary':'不推广为全部LC模型平均收益'},
 {'claim':'收益重复性与最好结果代表性因站点不同','evidence':'T2十队列计数、Wilson区间、最高与中位','status':'descriptive support','boundary':'不声称显著站点差异或未来成功概率'},
 {'claim':'投入操作信息改变评价集合','evidence':'T3及S2的N1/N2/N3标记','status':'supported under exploratory criteria','boundary':'不是阈值农学验证，19不推荐部署'},
 {'claim':'平均收益未覆盖年度减产','evidence':'S1、S3、S2年度最差值','status':'descriptive support','boundary':'年度>5%不新增筛选；FQ2018输入依赖'},
 {'claim':'天气增强未呈现一致收益','evidence':'T4的15项差值与Welch区间','status':'conditional route comparison','boundary':'五站异质处理，非YC运行时合同未完全核实，无无效/等效证明'},
]
(OUT/'claim_evidence_map.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
qa=dict(policies=80,annual_rows=800,cohorts=10,T1_means_verified=True,T2_counts_wilson_max_median_verified=True,T3_counts_verified=True,T4_welch_verified=15,gain=13,saving=31,overlap=1,benefit_union=43,N1_N2_N3=[51,31,19],fixed_among_N3=7,annual_risk_N2=15,annual_risk_N3=14,no_new_training=True,no_new_simulation=True,working_figure_numbers=True)
(OUT/'quality_checks.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
hashes={v:hashlib.sha256((ROOT/v).read_bytes()).hexdigest() for v in paths.values()};hashes['scripts/assemble_results_tables_074.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(OUT/'source_sha256.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(qa,ensure_ascii=False))
