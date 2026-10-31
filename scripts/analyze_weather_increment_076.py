"""Low-memory, endpoint-only exploratory seed bootstrap; no simulator imports."""
from pathlib import Path
import csv,json,hashlib,statistics as st,random,math
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/weather_increment_076_20261030'
OUT.mkdir(exist_ok=False)
hashes={}
def read(rel):
 p=ROOT/rel;hashes[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(name,rows):
 with (OUT/name).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
p=read('results/three_route_results_126_20261027/policy_120.csv')
a=read('results/three_route_results_126_20261027/annual_reference_risk_1200.csv')
assert len(p)==120 and len(a)==1200
key=lambda r:(r['site'],r['regime'],r['seed'])
assert len({key(r) for r in p})==120 and len({key(r)+(r['year'],) for r in a})==1200
metrics=['grain_yield_kg_ha','total_irrigation_mm','total_n_kg_ha','PFP_N']
groups=[];effects=[];rates=[];rng=random.Random(20261030)
def wilson(k,n):
 z=1.95996398454;d=1+z*z/n;c=(k/n+z*z/(2*n))/d;h=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/d
 return c-h,c+h
for site in ['SY','HL','LC','YC','FQ']:
 g={}
 for route in ['historical','augmented','forecast']:
  q=[r for r in p if r['site']==site and r['regime']==route];assert {int(r['seed']) for r in q}==set(range(8));g[route]=q
  for r in q:
   ar=[x for x in a if key(x)==key(r)];assert {int(x['year']) for x in ar}==set(range(2014,2024))
   for m in metrics:
    vals=[float(x[m]) for x in ar if x[m]]
    if vals:assert abs(st.mean(vals)-float(r[m]))<1e-6
  for m in metrics:
   v=[float(r[m]) for r in q if r[m]]
   groups.append(dict(site=site,regime=route,metric=m,n_defined=len(v),mean=st.mean(v),sd=st.stdev(v),minimum=min(v),maximum=max(v)))
  for flag in ['N1','saving','N3']:
   k=sum(r[flag]=='True' for r in q);lo,hi=wilson(k,8)
   rates.append(dict(site=site,regime=route,existing_posthoc_rule=flag,count=k,n=8,rate=k/8,wilson_low=lo,wilson_high=hi))
 for target,base in [('augmented','historical'),('forecast','augmented')]:
  for m in metrics:
   x=[float(r[m]) for r in g[target] if r[m]];y=[float(r[m]) for r in g[base] if r[m]]
   boots=sorted(st.mean(rng.choices(x,k=len(x)))-st.mean(rng.choices(y,k=len(y))) for _ in range(4000))
   effects.append(dict(site=site,contrast=target+' minus '+base,metric=m,difference=st.mean(x)-st.mean(y),bootstrap_low=boots[100],bootstrap_high=boots[3899],target_n=len(x),base_n=len(y)))
save('policy_120.csv',p);save('group_metrics_60.csv',groups);save('increment_effects_40.csv',effects);save('existing_rule_rates_45.csv',rates)
lines=['# 天气增量收益全seed分析','', '覆盖120策略、1200年度端点。固定2014—2023测试集；独立seed重采样4000次，区间只反映该测试集上的训练重复不确定性，不代表未来年份总体。每组仅8seed，区间为探索性、未校正多重比较。','', '## 产量增量（kg/ha）','|站点|比较|均值差|95% bootstrap区间|','|---|---|---:|---|']
for r in effects:
 if r['metric']==metrics[0]:lines.append(f"|{r['site']}|{r['contrast']}|{r['difference']:.1f}|[{r['bootstrap_low']:.1f}, {r['bootstrap_high']:.1f}]|")
lines+=['','## 解释边界','综合奖励未完成：126统一表及其组装脚本未提供三组同口径评价奖励；072的critic/局部行为回报不能替代120策略十年综合回报。本轮不据此给奖励排名。','PFP_N沿用年度比值的十年均值，不是十年平均产量除以平均施氮量；零施氮年度为未定义，不补零。','可靠性表沿用126事后规则，N1为均产不低于参照95%，saving为既有保产节资条件，N3含资源及操作约束，不称唯一成功率。Wilson区间条件于当前测试年份与seed样本。','预报减增强仅为观察组差异；全部合同一致性未在本轮重新审计，不能直接称预报因果效应。区间跨零不证明无效，无实际意义阈值不能作等效结论。','本轮没有训练、模拟或模型推理；未开展机制归因。奖励缺口需下一步追溯120策略实际评价episode回报及奖励版本，无法闭合则继续保留缺失。']
(OUT/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')
for rel in ['scripts/analyze_weather_increment_076.py','prompt_02/076_weather_increment_all_seed_20261030.md','scripts/assemble_three_route_results_126.py']:
 hashes[rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
(OUT/'source_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
(OUT/'verification.json').write_text(json.dumps(dict(policies=120,annual_rows=1200,groups=15,effects=len(effects),bootstrap_replicates=4000,new_training=0,new_simulations=0,reward_status='missing_not_substituted'),indent=2),encoding='utf-8')
print('\n'.join(lines));print(OUT)
