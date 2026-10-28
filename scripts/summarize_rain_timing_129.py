"""Evaluate the preregistered pilot gate; preserve negative results."""
from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/rain_timing_mechanism_129_20261027'
d=pd.read_csv(OUT/'results_9.csv');assert len(d)==9
assert json.loads((OUT/'protected_files_check.json').read_text())['unchanged']
b=d[d.arm.eq('baseline')].set_index('year');rows=[];persistence=[]
for arm in ['correct_gate','shifted_gate']:
    q=d[d.arm.eq(arm)].set_index('year')
    for year,r in q.iterrows():
        rows.append(dict(year=year,arm=arm,yield_delta=r.yield_kg_ha-b.loc[year,'yield_kg_ha'],yield_delta_pct=(r.yield_kg_ha/b.loc[year,'yield_kg_ha']-1)*100,I_delta=r.I-b.loc[year,'I'],N_delta=r.N-b.loc[year,'N'],gate_triggers=r.triggered_actions))
        folder=OUT/f'{year}_{arm}';dec=[json.loads(x) for x in (folder/'gate_decisions.jsonl').read_text().splitlines()];trace=[json.loads(x) for x in (folder/f'eval_100000_{year}_steps.jsonl').read_text().splitlines()]
        triggers=[x['step'] for x in dec if x['triggered']]
        for step in triggers:
            later=[z for z in trace if z['step']>step and z['I']>0]
            persistence.append(dict(year=year,arm=arm,gate_step=step,gate_date=trace[step]['date'],proposed_action=dec[step]['original_action'],selected_action=dec[step]['selected_action'],next_actual_irrigation_step=later[0]['step'] if later else None,days_to_next_actual_irrigation=later[0]['step']-step if later else None,note='Repeated gated proposals are not independent irrigation events or causal water savings'))
delta=pd.DataFrame(rows);delta.to_csv(OUT/'paired_year_deltas_6.csv',index=False);pd.DataFrame(persistence).to_csv(OUT/'gate_and_subsequent_irrigation.csv',index=False)
means=d.groupby('arm')[['yield_kg_ha','I','N','triggered_actions']].mean();means.to_csv(OUT/'three_year_means_3.csv')
base=means.loc['baseline'];correct=means.loc['correct_gate'];shift=means.loc['shifted_gate'];cq=delta[delta.arm.eq('correct_gate')]
checks=dict(mean_yield_ge_99pct=bool(correct.yield_kg_ha>=.99*base.yield_kg_ha),mean_irrigation_saving_ge_15mm=bool(base.I-correct.I>=15),mean_N_no_increase=bool(correct.N<=base.N),no_year_loss_over_5pct=bool(cq.yield_delta_pct.ge(-5).all()),correct_yield_ge_shifted_at_no_greater_inputs=bool(correct.yield_kg_ha>=shift.yield_kg_ha and correct.I<=shift.I and correct.N<=shift.N))
decision=dict(status='PILOT_FAILED_EFFICACY_GATE_STOP_RULE_DIRECTION',technical_evaluations_passed=9,checks=checks,mean_yield_delta=float(correct.yield_kg_ha-base.yield_kg_ha),mean_yield_delta_pct=float((correct.yield_kg_ha/base.yield_kg_ha-1)*100),mean_irrigation_saving_mm=float(base.I-correct.I),mean_N_delta=float(correct.N-base.N),correct_minus_shifted_yield=float(correct.yield_kg_ha-shift.yield_kg_ha),new_training_started=False,threshold_retuned=False,claim_learned_improvement_allowed=False)
(OUT/'scientific_decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 降雨时序机制pilot：结果与门槛判定','', '目标是验证一个可能的改进机制，非承诺产生正结果。固定SY seed0与2014/2018/2023，已有100000步模型；所有9次技术评价通过，原动作/物理输入/模型复现与最终Summary水氮闭合，保护文件未改变。','', '|年份|分支|产量差kg/ha|产量差%|灌溉差mm|氮差kg/ha|触发次数|','|---|---|---:|---:|---:|---:|---:|']
for r in delta.itertuples():lines.append(f'|{r.year}|{r.arm}|{r.yield_delta:+.2f}|{r.yield_delta_pct:+.3f}|{r.I_delta:+.1f}|{r.N_delta:+.1f}|{r.gate_triggers:.0f}|')
lines+=['',f"正确gate三年平均产量变化{decision['mean_yield_delta']:+.2f} kg/ha（{decision['mean_yield_delta_pct']:+.3f}%），平均节水{decision['mean_irrigation_saving_mm']:.1f}mm，氮不变；相对错位gate平均产量{decision['correct_minus_shifted_yield']:+.2f}kg/ha，三年平均水氮相同。",'', '## 预先门槛未通过，不调整阈值重试','', '正确gate通过99%平均产量和不增加N门槛、无单年超过5%减产；但平均节水仅10mm，未达到预设15mm，且相对错位gate未表现更优。不能声称正确利用降雨时序产生了可推广收益。样本只有一个模型三个年份，三年不是独立训练重复；错位方案也只有一个固定28日控制，不泛化为所有错误预报。','', '## 这轮排除了什么、发现了什么','', '已确认简单的“未来3日雨量>=15mm就延迟本次灌溉”在本pilot中不足以通过收益门槛。2014/2018各净少灌15mm，2023虽触发2次却最终灌溉总量不变，说明跳过当日动作不保证净节水。触发记录包含同一策略在连续日反复提议灌溉，不能将每次触发剂量相加后称为已节约的水；gate_and_subsequent_irrigation.csv保存后续实际灌溉。后续补灌/状态反馈抵消延迟收益是合理解释，但本试验未单独操纵该机制，不能断言已证明其为唯一原因。','', '## 后续研究边界','', '停止这个阈值规则方向，不把它蒸馏成新模型、不扩站或大批增训，也不改变门槛以获得正结果。尚未找到通过门槛的学习改进机制。若继续追求学习改进，下一候选应针对动态决策价值而非只提高预报敏感度：先固定少量合法决策点比较不同动作的闭环回报，判断原策略是否确有可改进动作，再考虑独立训练年份上的受控学习试验。该候选尚未执行，不声称存在oracle收益。','', '本轮是机制pilot的负结果，保留全部成功/失败收益和技术日志，无训练、无参数调整、无源数据修改。']
(OUT/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'prompts/2026-10-27_129_rain_timing_mechanism_pilot.md',ROOT/'scripts/test_rain_timing_mechanism_129.py',ROOT/'scripts/summarize_rain_timing_129.py',OUT/'frozen_protocol.json',OUT/'results_9.csv']};(OUT/'execution_sha256.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(decision,ensure_ascii=False,indent=2))
