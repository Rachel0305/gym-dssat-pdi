"""Bound hindsight values to the tested actions and report resource costs."""
from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/single_decision_value_130_20261027'
d=pd.read_csv(OUT/'branch_results.csv');points=pd.read_json(OUT/'frozen_points.json');assert len(d)==12
assert json.loads((OUT/'protected_files_check.json').read_text())['unchanged']
rows=[];all_deltas=[]
for point in points.to_dict('records'):
    year=point['year'];q=d[d.year.eq(year)];original=q[q.dose_mm.eq(point['original_I'])].iloc[0];best=q.sort_values(['reward_sum','I'],ascending=[False,True]).iloc[0]
    for row in q.itertuples():all_deltas.append(dict(year=year,dose_mm=row.dose_mm,reward_delta=row.reward_sum-original.reward_sum,yield_delta=row.yield_kg_ha-original.yield_kg_ha,I_delta=row.I-original.I,N_delta=row.N-original.N))
    rows.append(dict(year=year,date=point['date'],step=point['step'],rain_next3=point['rain_next3'],original_dose=point['original_I'],best_tested_dose=best.dose_mm,reward_gain=best.reward_sum-original.reward_sum,yield_gain=best.yield_kg_ha-original.yield_kg_ha,I_delta=best.I-original.I,N_delta=best.N-original.N,original_is_best=original.reward_sum>=best.reward_sum-1e-7))
t=pd.DataFrame(rows);t.to_csv(OUT/'best_tested_action_value_3.csv',index=False);pd.DataFrame(all_deltas).to_csv(OUT/'all_branch_deltas_12.csv',index=False)
lines=['# 单次决策闭环价值：三个固定湿天气点','', 'SY seed0，固定2014/2018/2023，按首个符合条件的原轨迹决策点选取，未看干预收益。仅该点选择0/15/30/45mm之一，氮不变，随后恢复原策略；合计12次单年回放。原15mm分支逐状态完整复现，前缀、模型/土壤/天气/品种、最终Summary水氮闭合均通过。','', '|年份/日期|未来3日雨量|原剂量|四个测试动作中最高回报剂量|回报增加|产量增加kg/ha|全年I变化mm|全年N变化|','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in t.itertuples():lines.append(f'|{r.year}/{r.date}|{r.rain_next3:.1f}|{r.original_dose:.0f}|{r.best_tested_dose:.0f}|{r.reward_gain:+.6f}|{r.yield_gain:+.2f}|{r.I_delta:+.0f}|{r.N_delta:+.0f}|')
lines+=['', '最高回报指原训练环境实际累计reward_sum，不是临时重定义目标。它只是这四种动作、原策略后续行为和已知年份下的事后局部比较，不是全局最优、学习后的新策略或外部泛化结果。','', '## 可确认的改进空间与代价','', '2014所选点原15mm已有最高回报；2018改为0mm后产量约增加26.9kg/ha、回报增加约0.00425，但全年灌溉不减；2023改为45mm后产量约增加294.0kg/ha、回报增加约0.01345，全年额外灌溉30mm，氮不变。不能将后者称保产节水或全面优势。','', '三点均在未来三日有较大降雨时，但最优测试剂量分别15、0、45mm，说明本样本中单一雨量阈值不足以选择更优动作。不同年份基础状态与生长阶段也不同，不能将剂量差异单独归因于降雨。','', '这轮通过只改变一次动作的控制，确认了两个已选状态内存在相对当前确定性动作更高的累计环境回报；但尚未验证模型能学会识别这些状态，更没有证明加入天气预报能改善该学习。','', '## 与129合并判定','', '129的反复雨前停灌规则未通过预设平均节水15mm门槛，也未优于固定错位gate，故停止该规则，不改阈值找正结果。130进一步显示：想使模型学得更好，应首先研究状态条件下的动作价值识别与后续补灌/资源配置，而非仅让预报输入更敏感。','', '可检验的下一学习假说是：在固定原PPO/奖励/动作的条件下，基于训练年份中合法动作的反事实闭环比较，提供更明确的动作价值学习信号，能否减少当前策略的局部动作遗憾。需要独立训练年份构造样本、同预算原训练对照与未参与选动作的年份评价；本轮3个测试年不得拿来拟合后再称验证。','', '## 当前目标状态','', '已找到局部改进机会，并排除一个简单阈值规则；尚未找到并验证可推广的学习改进机制。不进行仅用这3个测试点训练/蒸馏的过拟合尝试，不宣称目标已达成。本轮完成21次技术有效的有限评价（129九次、130十二次），新训练0次，原数据与模型均未覆盖。下一阶段应冻结基于训练年反事实动作价值的学习试验后，再进行小规模学习smoke。']
(OUT/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')
(OUT/'scientific_decision.json').write_text(json.dumps(dict(status='LOCAL_ACTION_VALUE_OPPORTUNITIES_IDENTIFIED_NOT_LEARNED',tested_points=3,points_with_reward_improvement=int((t.reward_gain>1e-7).sum()),retraining_runs=0,generalizable_learning_improvement_verified=False,total_simulations_129_130=21),indent=2))
(OUT/'execution_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'scripts/test_single_decision_value_130.py',ROOT/'scripts/summarize_decision_value_130.py',ROOT/'prompts/2026-10-27_130_single_decision_value.md',OUT/'frozen_points.json',OUT/'branch_results.csv']},indent=2));print(t.to_string(index=False))
