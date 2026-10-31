"""Reproduce legacy random streams; only requested LOSO and Welch sensitivity."""
from pathlib import Path
import csv,json,hashlib,random,statistics as st,math,shutil,re
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/section32_bootstrap_robustness_20261031'
OUT=BASE;i=1
while OUT.exists():i+=1;OUT=Path(str(BASE)+f'_attempt{i:02d}')
OUT.mkdir();HASH={}
def read(rel):
 p=ROOT/rel;HASH[str(rel)]=hashlib.sha256(p.read_bytes()).hexdigest()
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def textread(rel):
 p=ROOT/rel;HASH[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return p.read_text(encoding='utf-8-sig')
def save(name,rows):
 with (OUT/name).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
a=read('results/three_route_results_126_20261027/annual_1200.csv');p=read('results/three_route_results_126_20261027/policy_120.csv')
oldmean=read('results/weather_increment_076_20261030/increment_effects_40.csv');oldsd=read('results/section32_stability_080_20261031/site_stability_contrast_5.csv')
reg=read('2V02/MODEL_REGISTRY.csv');ident=read('results/all_seed_diagnosis_075_20261029_attempt02/historical_augmented_model_identity_80.csv')
S=['SY','HL','LC','YC','FQ'];R=['historical','augmented'];key=lambda r:(r['site'],r['regime'],int(r['seed']))
aa=[r for r in a if r['regime'] in R];assert len(aa)==800 and len({key(r)+(int(r['year']),) for r in aa})==800
pol=[];identity=[]
for site in S:
 for route in R:
  for seed in range(8):
   q=[r for r in aa if key(r)==(site,route,seed)];assert len(q)==10 and {int(r['year']) for r in q}==set(range(2014,2024))
   v=[float(r['grain_yield_kg_ha']) for r in q];assert all(math.isfinite(x) and x>=0 for x in v)
   mu=st.mean(v);sd=st.stdev(v);assert abs(sd-math.sqrt(sum((x-mu)**2 for x in v)/9))<1e-8
   pp=next(r for r in p if key(r)==(site,route,seed));assert abs(mu-float(pp['grain_yield_kg_ha']))<1e-8
   pol.append(dict(site=site,regime=route,seed=seed,n_years=10,mean_yield=mu,year_sd=sd))
   ir=next(r for r in ident if key(r)==(site,route,seed));path=ROOT/ir['matched_path']
   if not path.exists():path=ROOT/'2V02'/ir['matched_path']
   assert path.resolve().is_relative_to(ROOT)
   sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==ir['sha256']
   identity.append(dict(site=site,regime=route,seed=seed,sha256=sha,model_path=str(path.relative_to(ROOT))))
assert len({r['sha256'] for r in identity})==80
save('model_identity_80.csv',identity);save('per_seed_statistics_80.csv',pol)
# Preserve original order of p and RNG consumption, including forecast/other metrics.
rng=random.Random(20261030);meanrep={};metrics=['grain_yield_kg_ha','total_irrigation_mm','total_n_kg_ha','PFP_N']
for site in S:
 for target,base in [('augmented','historical'),('forecast','augmented')]:
  for m in metrics:
   x=[float(r[m]) for r in p if r['site']==site and r['regime']==target and r[m]];y=[float(r[m]) for r in p if r['site']==site and r['regime']==base and r[m]]
   b=sorted(st.mean(rng.choices(x,k=len(x)))-st.mean(rng.choices(y,k=len(y))) for _ in range(4000))
   if target=='augmented' and m==metrics[0]:meanrep[site]=(st.mean(x)-st.mean(y),b[100],b[3899])
rng=random.Random(20261031);summary=[];loso=[]
for site in S:
 h=[r['year_sd'] for r in pol if r['site']==site and r['regime']=='historical'];g=[r['year_sd'] for r in pol if r['site']==site and r['regime']=='augmented']
 delta=st.mean(g)-st.mean(h);b=sorted(st.mean(rng.choices(g,k=8))-st.mean(rng.choices(h,k=8)) for _ in range(10000));lo,hi=b[250],b[9749]
 old=next(r for r in oldsd if r['site']==site)
 for value,k in [(delta,'difference'),(lo,'bootstrap_low'),(hi,'bootstrap_high')]:assert abs(value-float(old[k]))<1e-8
 om=next(r for r in oldmean if r['site']==site and r['contrast']=='augmented minus historical' and r['metric']==metrics[0])
 for value,k in zip(meanrep[site],['difference','bootstrap_low','bootstrap_high']):assert abs(value-float(om[k]))<1e-8
 u,v=st.variance(g)/8,st.variance(h)/8;se=math.sqrt(u+v);df=(u+v)**2/(u*u/7+v*v/7);crit=float(t.ppf(.975,df));wl,wh=delta-crit*se,delta+crit*se
 local=[]
 for route in R:
  for seed in range(8):
   gg=g[:seed]+g[seed+1:] if route=='augmented' else g;hh=h[:seed]+h[seed+1:] if route=='historical' else h
   d=st.mean(gg)-st.mean(hh);row=dict(site=site,deleted_group=route,deleted_seed=seed,delta_sd=d,difference_from_full=d-delta);loso.append(row);local.append(row)
 low=min(local,key=lambda r:r['delta_sd']);high=max(local,key=lambda r:r['delta_sd'])
 summary.append(dict(site=site,mean_yield_delta=meanrep[site][0],mean_yield_boot_low=meanrep[site][1],mean_yield_boot_high=meanrep[site][2],historical_sd_mean=st.mean(h),augmented_sd_mean=st.mean(g),delta_sd=delta,bootstrap_low=lo,bootstrap_high=hi,welch_low=wl,welch_high=wh,welch_df=df,loso_min=low['delta_sd'],loso_max=high['delta_sd'],loso_min_deleted=f"{low['deleted_group']}:{low['deleted_seed']}",loso_max_deleted=f"{high['deleted_group']}:{high['deleted_seed']}",loso_direction_reversed=any(d['delta_sd']*delta<0 for d in local),bootstrap_excludes_zero=lo*hi>0,welch_excludes_zero=wl*wh>0,legacy_reproduced=True))
save('bootstrap_robustness_summary.csv',summary);save('leave_one_seed_out_80.csv',loso)
# Source audits: record actual limits, do not replace weather-QC with statistical reproduction.
evidence=['scripts/analyze_weather_increment_076.py','scripts/section32_year_stability_080.py','scripts/assemble_three_route_results_126.py','2V02/docs/yc_random_weather_004_16_wgen_weather_reconstruction_and_coverage.md','2V02/docs/hla_wgen_training_053_record.md','2V02/docs/sylc_external_weather_060_execution.md','2V02/results/yc_random_weather_ppo/004_05/config/experiment_design.json','2V02/results/sylc_external_weather_057/independent_contract_audit.json','2V02/results/hla_wgen_8seed_053/final_gate.json','2V02/results/fqa_multiyear_wgen_ppo_052_8seed/final_gate.json','results/section32_stability_080_20261031/section_3_2_updated_zh.md','results/section32_contract_followup_078_20261031/report_zh.md']
for rel in evidence:textread(rel)
hl=next(r for r in summary if r['site']=='HL');sensitive=hl['bootstrap_excludes_zero']!=hl['welch_excludes_zero'];classification='B' if sensitive else 'A'
methods='''# Statistical analysis for Results 3.2

For each site and training regime, eight seed-labelled training runs were evaluated deterministically over the same fixed ten years (2014–2023). We first calculated each policy's ten-year mean yield and sample standard deviation across years (ddof=1). The contrasts were the differences between the augmented and historical groups in the mean of these eight policy-level summaries. Annual records were not treated as independent training replicates.

We resampled eight policy-level summaries with replacement independently within each group and computed the augmented-minus-historical difference. The archived mean-yield analysis used 4,000 replicates with Python random.Random(20261030); the interannual-SD analysis used 10,000 replicates with random.Random(20261031). Random streams were shared across sites in the order SY, HL, LC, YC, FQ. Exact reproduction of the mean-yield intervals also retains the intervening metric and forecast-route draws from the original script. These intervening draws are implementation details, not additional sensitivity analyses.

The percentile intervals used discrete empirical order statistics: for sorted zero-based bootstrap draws, indices 100 and 3899 at B=4000, and 250 and 9749 at B=10000. These are non-interpolated empirical percentile intervals, not BCa intervals. Index conventions can produce small Monte Carlo/order-statistic differences from interpolated quantiles; the archived convention was retained without selecting more favourable bounds.

Matching numerical seed labels did not establish a randomized matched-pair design. Weather schedules differ between regimes, and matched RNG trajectories, common-random-number coupling and prespecified matched blocks were not established across all sites. We therefore retained independent-group resampling. Distinct checkpoint hashes and seed labels support separate training runs but do not prove mathematical stochastic independence. Inference is conditional on shared input data and the finite fixed weather schedules, not variability across independent training-weather datasets. Coincident policy behaviour does not constitute duplicate training runs.

Sensitivity checks were limited to deleting one seed at a time from either group (16 contrasts per site, with group sizes 7 versus 8), and a Welch–Satterthwaite interval for the difference between group means of policy-level interannual SDs. For the latter, SE=sqrt(s_A²/8+s_H²/8), df=(s_A²/8+s_H²/8)²/[(s_A²/8)²/7+(s_H²/8)²/7], and the interval is delta ± t_(0.975,df)×SE. Welch is an approximate small-sample sensitivity reference, not a gold standard.

All 95% intervals are exploratory and unadjusted for comparisons across sites and endpoints. With eight runs per group, tail uncertainty is weakly resolved. These intervals primarily reflect training-seed variability conditional on the ten evaluation years and fixed weather schedules; they do not quantify uncertainty for arbitrary future years, certify pristine final-test status, establish equivalence, or verify causal learning mechanisms. A zero-crossing interval is not evidence of no effect. Lower interannual SD is interpreted jointly with mean yield and resource inputs.
'''
(OUT/'methods_statistical_analysis.md').write_text(methods,encoding='utf-8')
table=['|站点|ΔSD|Bootstrap 95%区间|Welch 95%区间|删一种子后范围|','|---|---:|---|---|---|']
for r in summary:table.append(f"|{r['site']}|{r['delta_sd']:+.1f}|[{r['bootstrap_low']:.1f}, {r['bootstrap_high']:.1f}]|[{r['welch_low']:.1f}, {r['welch_high']:.1f}]|[{r['loso_min']:.1f}, {r['loso_max']:.1f}]|")
report=f'''# Results 3.2 Bootstrap审计

1. 现有Bootstrap方法正确吗？是。076平均产量与080年际SD的点估计、原随机流区间均独立复现至1e-8；未把年度当训练重复。离散percentile区间实现合理，非BCa。
2. 8seed支持什么推断？固定十年和既有天气调度下训练重复的探索性差异；小样本、未校正多重比较，不能推广任意未来年份或宣称因果机制。
3. HL是否稳健？点估计的正方向在16次单seed删除中均保持，范围[{hl['loso_min']:.1f}, {hl['loso_max']:.1f}]；但Bootstrap区间[{hl['bootstrap_low']:.1f}, {hl['bootstrap_high']:.1f}]与Welch区间[{hl['welch_low']:.1f}, {hl['welch_high']:.1f}]的零覆盖判断{'不同' if sensitive else '相同'}。不能把正方向稳健等同区间排零稳健。
4. 其他四站结论改变吗？没有发现新增确定性改善证据，继续描述变化；逐站删种子方向与区间见表，不将方向变化隐去。
5. 冻结判断：READY_WITH_CAVEATS。分类{classification}：计算正确，{'HL区间排零对方法敏感' if sensitive else '主要结论保持'}。统计收尾完成，有限样本与来源限制随文保留，不扩展诊断。

## 数据与训练单位审计

五站各H/A 8seed，800条唯一年度结果；每策略完整2014—2023，产量有限且非负，均值与126策略表一致。80个模型SHA256重新读取核对，通过且互不重复。未发现本次输入中重复seed、缺失年度或重复模型文件。文件独立不证明随机性完全独立；历史训练的续训只能算同一seed一次，不额外计重复。

不同组同数字seed可能共享初始化，不能仅凭编号作配对；没有全站预设匹配区组/同步随机轨迹证据，保持独立抽样。天气池和调度在组内可共享，因此估计不包含天气池抽样不确定性。

126使用既有FQ2018质量控制修订，历史、增强及对应评价按修订版本汇总；该修订是输入条件，不是随机替换差seed。旧零产量版本不能混入。126记录SY/LC等身份源与076/080一致，未发生本次替换、重新训练或新模拟。独立模型哈希不能单独证明所有runtime条件完全相同。

## 有效性与天气质量边界

SY/LC：057已有16项独立合同PASS，060是后续外部天气增强结果；旧WGEN停止记录不直接否定后续路线。HL/FQ：各8seed训练/评价归档门槛PASS；训练天气调度共享。HL存在native FIELD坐标限制及GPCC补雨情景，不能删掉该条件。YC：004_05设计记录仅天气增强变化、拟合2004—2013、2014—2023独立比较期但非认证的pristine final test；004_16重建gate未通过，训练实际天气覆盖和尾部QC证据不足，它不判定旧训练作物结果无效。因而所有站点可用于既有条件下描述性结果，不能声称五站统一通过同等级完整天气分布QC。统计冻结不补齐这些来源缺口，也不扩大天气审查。

## 原方法复核

076：独立组seed重采样，4000次，随机种子20261030；每seed先十年均产，原序列跨站并消费其他指标/预报比较的抽样。080：先每seed十年样本SD(ddof=1)，后两组SD均值差，10000次，随机种子20261031。两者均非配对、非BCa。4000次零基索引[100,3899]；10000次[250,9749]。本次保留相同次序复现而未执行旧脚本的写文件入口，未覆盖旧结果。

## 稳健性检查

'''+'\n'.join(table)+f'''

单位kg/ha。LOSO每次仅从H或A删除一个seed，16次/站；它检查点估计影响，不重新计算删seed后的置信区间。
HL最小值删去{hl['loso_min_deleted']}，最大值删去{hl['loso_max_deleted']}；方向不反转，但区间排零依赖方法，应保留描述性增加及方法敏感说明。其余站点方向是否反转见CSV。

五站及多指标95%区间均未作多重比较校正，不能仅凭HL原区间排零作正式确认性显著声明。Welch同样有8seed小样本限制。未新增检验、未调整方法获得预期结果、未把跨零解释为无效。

## 收尾

冻结是READY_WITH_CAVEATS：允许将审计后3.2作为当前证据版本收尾，不能将其标为无条件独立最终测试或算法机制验证。原结果不覆盖，修改仅降低HL区间结论确定性并补统计方法。无须新增训练、模拟、天气分组或其他指标。
'''
(OUT/'bootstrap_audit_report.md').write_text(report,encoding='utf-8')
original=textread('results/section32_stability_080_20261031/section_3_2_updated_zh.md')
start=original.index('为评价同一策略的跨年份波动');end=original.index('\n\n综合来看',start)
paragraph='为评价同一策略的跨年份波动，先计算各seed在固定十年评价期内的产量样本标准差，再比较两组各8个标准差的均值。SY、HL、LC、YC和FQ的差值分别为−22.8、+145.2、+354.1、−191.0和+57.4 kg ha⁻¹。'
paragraph+=f"HL的平均年际标准差由884.0增加至1029.2 kg ha⁻¹，Bootstrap 95%区间为［7.6，304.0］kg ha⁻¹；Welch区间为［{hl['welch_low']:.1f}，{hl['welch_high']:.1f}］kg ha⁻¹。逐次删除H或A的一个seed后，HL差值保持正向，但区间是否排零对方法敏感，因此该增加仅作探索性描述。其他四站的Bootstrap区间均跨零。较低的年际标准差不能单独认定管理改善。"
final=original[:start]+paragraph+original[end:]+'\n本文区间均为未经多重比较校正的探索性95%区间，反映固定十年评价条件下的训练seed不确定性，不代表任意未来年份；区间跨零不证明无效。\n'
(OUT/'section_3_2_final_zh.md').write_text(final,encoding='utf-8')
doc=ROOT/'docs/2026-10-31_section32_bootstrap_audit.md';assert not doc.exists();doc.write_text(report+'\n\nMethods统计表述\n\n'+methods,encoding='utf-8')
shutil.copy2(Path(__file__),OUT/Path(__file__).name)
textread('prompts/135_section32_bootstrap_robustness_audit.md');HASH[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(OUT/'source_sha256.json').write_text(json.dumps(HASH,indent=2),encoding='utf-8')
(OUT/'verification.json').write_text(json.dumps(dict(status='READY_WITH_CAVEATS',classification=classification,annual=800,policies=80,unique_model_hashes=80,legacy_mean_intervals_reproduced=5,legacy_sd_intervals_reproduced=5,loso=80,new_training=0,new_simulations=0),indent=2),encoding='utf-8')
print(json.dumps({'output':str(OUT),'classification':classification,'HL':hl},ensure_ascii=True))
