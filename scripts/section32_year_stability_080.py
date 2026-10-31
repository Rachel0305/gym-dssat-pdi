"""Fixed evaluation-year yield SD, independent-seed bootstrap; no simulation."""
from pathlib import Path
import csv,json,hashlib,random,statistics as st
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/section32_stability_080_20261031';OUT.mkdir(exist_ok=False)
source=ROOT/'results/three_route_results_126_20261027/annual_1200.csv'
with source.open(encoding='utf-8-sig',newline='') as f:a=[r for r in csv.DictReader(f) if r['regime'] in ['historical','augmented']]
assert len(a)==800 and len({(r['site'],r['regime'],r['seed'],r['year']) for r in a})==800
def save(name,rows):
 with (OUT/name).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
pol=[];summ=[];rng=random.Random(20261031)
for site in ['SY','HL','LC','YC','FQ']:
 for route in ['historical','augmented']:
  for seed in range(8):
   q=[r for r in a if r['site']==site and r['regime']==route and int(r['seed'])==seed];assert len(q)==10 and {int(r['year']) for r in q}==set(range(2014,2024))
   v=[float(r['grain_yield_kg_ha']) for r in q]
   pol.append(dict(site=site,regime=route,seed=seed,n_years=10,mean_yield=st.mean(v),yearly_yield_sample_sd=st.stdev(v)))
 h=[r['yearly_yield_sample_sd'] for r in pol if r['site']==site and r['regime']=='historical'];g=[r['yearly_yield_sample_sd'] for r in pol if r['site']==site and r['regime']=='augmented']
 b=sorted(st.mean(rng.choices(g,k=8))-st.mean(rng.choices(h,k=8)) for _ in range(10000))
 summ.append(dict(site=site,historical_mean_year_sd=st.mean(h),augmented_mean_year_sd=st.mean(g),difference=st.mean(g)-st.mean(h),change_pct=100*(st.mean(g)/st.mean(h)-1),bootstrap_low=b[250],bootstrap_high=b[9749],n_seeds_each=8))
save('per_seed_yearly_yield_sd_80.csv',pol);save('site_stability_contrast_5.csv',summ)
table=['|站点|历史组年际SD均值|增强组年际SD均值|差值|95%区间|','|---|---:|---:|---:|---|']
for r in summ:table.append(f"|{r['site']}|{r['historical_mean_year_sd']:.1f}|{r['augmented_mean_year_sd']:.1f}|{r['difference']:+.1f}|[{r['bootstrap_low']:.1f}, {r['bootstrap_high']:.1f}]|")
paragraph='为评价同一策略的跨年份波动，对每个随机种子分别计算2014—2023年产量的样本标准差，再在各组8个种子间取平均。'
for r in summ:paragraph+=f"{r['site']}的年际产量标准差由历史组的{r['historical_mean_year_sd']:.1f}变为增强组的{r['augmented_mean_year_sd']:.1f} kg ha⁻¹，差值为{r['difference']:+.1f} kg ha⁻¹（探索性95% bootstrap区间：{r['bootstrap_low']:.1f}至{r['bootstrap_high']:.1f}）。"
paragraph+='该指标描述同一策略在固定十年评价集上的产量波动，与不同seed十年均产的离散程度不同。标准差降低不能单独认定管理改善，也不能证明策略主动适应天气，需与平均产量、水氮投入和生产率共同解释。'
draft='''# 3.2 天气增强对跨年水氮管理表现的影响

在2014—2023年共同评价年份内，天气增强相对历史天气训练的产量变化呈现站点差异。基于每组8个随机种子的十年平均结果，SY、HL、LC和YC的平均产量分别降低约482、324、329和247 kg ha⁻¹，FQ增加约61 kg ha⁻¹。五个站点产量差异的探索性95% bootstrap区间均跨越零，当前样本未提供天气增强带来一致增产收益的明确证据。这不能用于认定天气增强无效。

产量变化伴随着不同的水氮投入权衡。SY和HL的平均灌溉量分别减少45.2和28.9 mm，施氮量分别减少9.5和11.5 kg N ha⁻¹，PFP_N分别提高15.4和11.2 kg grain kg⁻¹ N；但WP_ET分别下降0.086和0.048 kg m⁻³。其中HL的WP_ET差异区间为［−0.085，−0.013］kg m⁻³，为未经多重比较校正的探索性下降信号。LC和YC虽然分别减少灌溉9.2和11.4 mm，但施氮量分别增加46.5和11.0 kg N ha⁻¹，且产量、WP_ET与PFP_N的均值均下降。FQ的小幅增产伴随施氮量增加8.0 kg N ha⁻¹，WP_ET增加0.031 kg m⁻³，而PFP_N降低11.5 kg grain kg⁻¹ N。

年度比较显示，SY、HL、LC、YC和FQ分别有1、5、3、2和7个年份的组平均产量高于历史训练组。HL虽然有一半年份呈正差，但年度增益最高仅约21 kg ha⁻¹，最大损失约1209 kg ha⁻¹，十年平均产量仍下降；正差年份数量不能替代变化幅度的评价。

'''+paragraph+'\n\n综合来看，天气增强的评价应同时考虑平均产量、年际波动与资源权衡，现有证据不支持其在所有站点带来一致的综合改善。跨年份波动的变化并不等同于训练可靠性的变化；不同seed之间的表现差异另在3.4分析。\n'
(OUT/'section_3_2_updated_zh.md').write_text(draft,encoding='utf-8')
note=['# 年际波动补充分析','','\n'.join(table),'','单位均为kg/ha；SD为每seed十年样本标准差（ddof=1），组统计是8个SD的均值，不是先平均seed再计算年度SD。','独立seed bootstrap10000次，随机种子20261031；固定十年，不重采样年份，不将80年度当独立训练重复。区间未作多重比较校正；不推断未来年份总体或天气适应机制。','本轮事后探索性补充；未追加训练/模拟。旧3.2文件不覆盖，新正文单独交付。',paragraph]
(OUT/'report_zh.md').write_text('\n'.join(note),encoding='utf-8')
inputs=[source,Path(__file__).resolve(),ROOT/'results/section32_077_20261031/annual_summary_20.csv',ROOT/'results/wp_et_079_20261031_attempt02/wp_et_contrasts_15.csv']
(OUT/'source_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2),encoding='utf-8')
(OUT/'verification.json').write_text(json.dumps(dict(annual_rows=800,policies=80,sites=5,years=list(range(2014,2024)),sample_sd_ddof=1,bootstrap=10000,new_training=0,new_simulations=0),indent=2),encoding='utf-8')
print(json.dumps(summ))
