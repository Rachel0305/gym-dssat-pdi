"""Render frozen Results 3.2 estimates; no bootstrap, model or simulator execution."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/section32_figures_20261031';OUT=BASE;i=1
while OUT.exists():i+=1;OUT=Path(str(BASE)+f'_attempt{i:02d}')
OUT.mkdir();HASH={}
def read(rel):
 p=ROOT/rel;HASH[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p)
def text(rel):
 p=ROOT/rel;HASH[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return p.read_text(encoding='utf-8-sig')
audit=read('results/section32_bootstrap_robustness_20261031/bootstrap_robustness_summary.csv')
policy=read('results/three_route_results_126_20261027/policy_120.csv');annual=read('results/three_route_results_126_20261027/annual_1200.csv')
effect=read('results/weather_increment_076_20261030/increment_effects_40.csv');wp=read('results/wp_et_079_20261031_attempt02/wp_et_contrasts_15.csv');yr=read('results/section32_077_20261031/annual_increment_200.csv')
sd=read('results/section32_bootstrap_robustness_20261031/per_seed_statistics_80.csv')
draft=text('results/section32_bootstrap_robustness_20261031/section_3_2_final_zh.md');text('docs/2026-10-31_section32_bootstrap_audit.md')
S=['SY','HL','LC','YC','FQ'];audit=audit.set_index('site').loc[S];metrics=['grain_yield_kg_ha','total_irrigation_mm','total_n_kg_ha','PFP_N','WP_ET_kg_m3'];names=['Yield','Irrigation','N application','PFP_N','WP_ET'];units=['kg ha-1','mm','kg N ha-1','kg grain kg-1 N','kg m-3'];table=[]
for site in S:
 for m,name,unit in zip(metrics,names,units):
  q=policy[policy.site.eq(site)];h=q[q.regime.eq('historical')][m].mean();a=q[q.regime.eq('augmented')][m].mean();delta=a-h
  if m=='WP_ET_kg_m3':e=wp[wp.site.eq(site)&wp.contrast.eq('augmented minus historical')].iloc[0];lo,hi=e.low,e.high
  else:e=effect[effect.site.eq(site)&effect.contrast.eq('augmented minus historical')&effect.metric.eq(m)].iloc[0];lo,hi=e.bootstrap_low,e.bootstrap_high
  assert np.isclose(delta,e.difference,atol=1e-8)
  if m==metrics[0]:assert np.isclose(delta,audit.loc[site,'mean_yield_delta'],atol=1e-8) and np.isclose(lo,audit.loc[site,'mean_yield_boot_low']) and np.isclose(hi,audit.loc[site,'mean_yield_boot_high'])
  table.append(dict(site=site,metric=name,unit=unit,historical_mean=h,augmented_mean=a,difference=delta,relative_change_pct=100*delta/h if h!=0 else np.nan,bootstrap_low=lo,bootstrap_high=hi,welch_low=np.nan,welch_high=np.nan))
 r=audit.loc[site];table.append(dict(site=site,metric='Interannual yield SD',unit='kg ha-1',historical_mean=r.historical_sd_mean,augmented_mean=r.augmented_sd_mean,difference=r.delta_sd,relative_change_pct=100*r.delta_sd/r.historical_sd_mean,bootstrap_low=r.bootstrap_low,bootstrap_high=r.bootstrap_high,welch_low=r.welch_low,welch_high=r.welch_high))
tbl=pd.DataFrame(table);tbl.to_csv(OUT/'table_section32_supplementary.csv',index=False)
plot=tbl.copy();plot['panel']=np.where(plot.metric.eq('Yield'),'A',np.where(plot.metric.eq('Interannual yield SD'),'B','C'));plot.to_csv(OUT/'plotting_data.csv',index=False)
rows=[]
for site in S:
 for year in range(2014,2024):
  q=annual[annual.site.eq(site)&annual.year.eq(year)];h=q[q.regime.eq('historical')];a=q[q.regime.eq('augmented')];assert len(h)==len(a)==8
  value=a.grain_yield_kg_ha.mean()-h.grain_yield_kg_ha.mean();e=yr[yr.site.eq(site)&yr.year.eq(year)&yr.metric.eq('grain_yield_kg_ha')].iloc[0];assert abs(value-e.difference)<1e-7
  rows.append(dict(site=site,year=year,difference=value))
years=pd.DataFrame(rows);counts=years.groupby('site').difference.apply(lambda x:int((x>0).sum()));assert counts.reindex(S).tolist()==[1,5,3,2,7]
years.to_csv(OUT/'plotting_data_annual.csv',index=False)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':8,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
blue='#246B8E';orange='#B35D22';grey='#6A737B'
fig=plt.figure(figsize=(7.2,6.3));gs=fig.add_gridspec(2,2,height_ratios=[1.1,1],left=.115,right=.96,bottom=.1,top=.95,wspace=.36,hspace=.58)
ax=fig.add_subplot(gs[0,0]);bx=fig.add_subplot(gs[0,1]);y=np.arange(5)
ax.axvline(0,color='#777777',ls=':',lw=.8,zorder=0)
ax.errorbar(audit.mean_yield_delta,y,xerr=np.vstack([audit.mean_yield_delta-audit.mean_yield_boot_low,audit.mean_yield_boot_high-audit.mean_yield_delta]),fmt='o',color=blue,ms=4,capsize=2,lw=1.2)
ax.set(yticks=y,yticklabels=S,ylim=(4.6,-.6),xlabel='Yield difference (kg ha$^{-1}$)',title='Mean yield');ax.text(-.19,1.08,'A',transform=ax.transAxes,fontweight='bold',fontsize=11)
bx.axvline(0,color='#777777',ls=':',lw=.8,zorder=0)
for j,site in enumerate(S):
 r=audit.loc[site];bx.plot([r.welch_low,r.welch_high],[j+.10]*2,color=orange,ls='--',lw=1.3);bx.plot([r.bootstrap_low,r.bootstrap_high],[j-.10]*2,color=blue,lw=1.3);bx.plot(r.delta_sd,j,'o',color='#283640',ms=4)
bx.set(yticks=y,yticklabels=S,ylim=(4.6,-.6),xlabel='Interannual yield SD difference (kg ha$^{-1}$)',title='Interannual yield variability');bx.text(-.19,1.08,'B',transform=bx.transAxes,fontweight='bold',fontsize=11)
bx.legend(handles=[Line2D([0],[0],color=blue,label='Bootstrap 95% CI'),Line2D([0],[0],color=orange,ls='--',label='Welch 95% CI')],loc='upper center',bbox_to_anchor=(.45,-.25),ncol=2,frameon=False,handlelength=1.7,columnspacing=.9)
cx=fig.add_subplot(gs[1,:]);heat=tbl[tbl.metric.isin(names[1:])].pivot(index='site',columns='metric',values='relative_change_pct').reindex(index=S,columns=names[1:]);v=heat.to_numpy();bound=np.nanmax(abs(v));im=cx.imshow(v,cmap='PuOr_r',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),aspect='auto',interpolation='nearest')
cx.set(yticks=y,yticklabels=S,xticks=np.arange(4),xticklabels=['Irrigation','N application','PFP$_N$','WP$_{ET}$']);cx.tick_params(length=0);cx.set_title('Resource inputs and productivity',pad=10);cx.text(-.09,1.10,'C',transform=cx.transAxes,fontweight='bold',fontsize=11)
for j in range(5):
 for k in range(4):cx.text(k,j,f'{v[j,k]:+.1f}%' if np.isfinite(v[j,k]) else 'NA',ha='center',va='center',fontsize=8,color='white' if abs(v[j,k])>.65*bound else '#202020')
cb=fig.colorbar(im,ax=cx,fraction=.025,pad=.025);cb.ax.set_title('% change',fontsize=7,pad=8);cb.ax.tick_params(labelsize=7)
fig.text(.115,.015,'All contrasts: A − H. Colours indicate direction, not management benefit.',fontsize=7,color='#555555')
def export(fig,name):
 for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=400)
 fig.canvas.draw()
export(fig,'figure_section32_main');plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(7.2,4.6),sharey=True);fig.subplots_adjust(left=.10,right=.98,bottom=.14,top=.91,wspace=.18,hspace=.48)
for i,(ax,site) in enumerate(zip(axes.flat,S)):
 q=years[years.site.eq(site)];ax.axhline(0,color='#777777',lw=.7,ls=':');ax.plot(q.year,q.difference,'o-',lw=1,color=blue,ms=3);ax.set_title(site,loc='left',fontweight='bold');ax.set_ylim(-1650,500);ax.set_xticks([2014,2017,2020,2023]);ax.tick_params(axis='x',rotation=35);ax.set_xlabel('Year')
axes[1,2].axis('off');axes[1,2].text(.05,.80,'A − H\n\n8 training seeds per group\nAll 10 evaluation years\nNo annual significance tests',transform=axes[1,2].transAxes,fontsize=8,va='top',linespacing=1.5)
fig.supylabel('Yield difference (kg ha$^{-1}$)',x=.015,fontsize=8);export(fig,'figure_section32_annual');plt.close(fig)
captions='''# Figure captions

**Figure X | Effects of weather augmentation on management performance and interannual yield variability across five sites.** H denotes historical-weather training and A denotes weather-augmented training. All differences are A minus H. **A**, Difference in group-average ten-year mean yield, with exploratory 95% percentile bootstrap confidence intervals. **B**, Difference in the group mean of policy-specific interannual yield sample standard deviations (SD; ddof=1), with bootstrap intervals (solid blue) and Welch intervals (dashed orange) vertically offset for readability. Each SD is calculated across the same ten evaluation years, not across training seeds. The shared point marks the estimated difference. For HL, the positive point-estimate direction survives deletion of any one seed, but zero exclusion depends on interval method. **C**, Descriptive percentage changes, calculated as 100×(A−H)/H from group means. PFP_N is grain yield per unit N applied; WP_ET is grain yield per unit evapotranspiration. Cell colours encode numerical direction rather than management benefit. Each regime includes eight training seeds evaluated over 2014–2023. Bootstrap resampling units are policies, independently within regimes. All 95% intervals are unadjusted for multiple comparisons and reflect training-seed uncertainty conditional on the fixed evaluation years and weather schedules; they do not establish equivalence, causal mechanisms or performance in arbitrary future years. No significance symbols are used.

**Figure Sx | Year-specific yield differences between weather-augmented and historical training.** Points show the difference between the annual group means of eight policies per regime (A−H), for all evaluation years from 2014 to 2023. Lines connect the observed annual estimates without smoothing. The five sites share the same vertical scale; dotted lines indicate zero difference. Years are repeated evaluations of the policies, not independent training replicates. No annual significance tests were performed.

**Table Sx | Management performance and interannual variability by site and training regime.** Values report H and A group means, absolute differences (A−H), percentage differences relative to H, and archived exploratory 95% bootstrap intervals for absolute differences. Welch intervals are supplied only for interannual SD, as audited. Yield, irrigation, N application, PFP_N and WP_ET are averaged over valid years within each policy, then across the eight policies. PFP_N is the mean of annual yield/N ratios, not the ratio of mean yield to mean N; zero-N annual values remain undefined rather than zero. WP_ET is the mean of annual yield/(10×ETCP) ratios (kg m−3). Interannual SD is computed separately for each policy over ten years (ddof=1) and then averaged across policies. Missing interval fields mean not supplied for that statistic, not zero uncertainty. Intervals are unadjusted for multiple comparisons.
'''
(OUT/'figure_captions.md').write_text(captions,encoding='utf-8')
tablemd=['# Table Sx','', '|Site|Metric|Unit|H|A|A−H|Change (%)|Bootstrap 95%|Welch 95%|','|---|---|---|---:|---:|---:|---:|---|---|']
for r in table:
 tablemd.append(f"|{r['site']}|{r['metric']}|{r['unit']}|{r['historical_mean']:.3f}|{r['augmented_mean']:.3f}|{r['difference']:.3f}|{r['relative_change_pct']:.2f}|[{r['bootstrap_low']:.3f}, {r['bootstrap_high']:.3f}]|"+(f"[{r['welch_low']:.3f}, {r['welch_high']:.3f}]" if np.isfinite(r['welch_low']) else 'NA')+'|')
(OUT/'table_section32_supplementary.md').write_text('\n'.join(tablemd)+'\n\n'+captions.split('**Table Sx')[1],encoding='utf-8')
paras=draft.split('\n\n');targets=[('在2014','（Figure X，Panel A；Table Sx）'),('产量变化','（Figure X，Panel C；Table Sx）'),('年度比较','（Figure Sx；年度精确值见绘图数据）'),('为评价','（Figure X，Panel B；Table Sx）')]
for i,pa in enumerate(paras):
 for prefix,ref in targets:
  if pa.startswith(prefix):paras[i]=pa.rstrip()[:-1]+ref+'。'
(OUT/'section_3_2_with_figures_zh.md').write_text('\n\n'.join(paras),encoding='utf-8')
qc='''# 3.2图表质量与一致性记录

状态：FIGURES_READY_WITH_CAVEATS。本轮只读取既有冻结统计结果，未重算bootstrap或执行新检验。
图形合同：天气增强未显示统一改善，平均产量、年际波动与资源权衡分别由A、B、C提供互补证据。quantitative grid；Python matplotlib；183mm双栏宽，主图160mm高，英文8pt主体，PNG400dpi，PDF TrueType及SVG文本保留。未指定目标期刊，尚需按目标期刊最终版式检查。
数据：135审计五站均产/SD区间，076其他三指标区间，079 WP_ET区间，126组均值与年度端点。逐项差值对齐检查通过；H/A不混入forecast。30条表格记录，50条年度差值。SY/HL/LC/YC/FQ正差年数1/5/3/2/7；HL最大年度增益21.3408，最大年度损失−1208.8058 kg/ha。正差年数非独立二项重复。
Panel C使用组均值百分比，不同单位指标不混成综合分；以零为中心、对称全范围色标，未截断极值，无星号。历史分母均非零；表格缺失Welch区间标NA。年际SD仅组内policy跨年样本SD均值，不误写seed间SD。
正文只增加图表引用，保留原结论与数字。Table Sx包含正文重点提及的HL WP_ET区间及所有五站SD、Welch值。所有年度值以CSV保存，Figure Sx绘制全部十年及相同纵轴，无平滑、无逐年检验。
限制：READY_WITH_CAVEATS，未校正多重比较、8seed小样本与固定评价年份；保留审计中的天气数据来源和非认证pristine final test限制。HL的bootstrap排零不写稳健显著性。图形完整不补齐天气QC来源缺口。不以跨零证明无效，不以SD降低推断适应能力。
'''
(OUT/'figure_quality_check.md').write_text(qc,encoding='utf-8')
doc=ROOT/'docs/2026-10-31_section32_figure_design.md'
if doc.exists():
 backup=ROOT/'backups/section32_figures136';backup.mkdir(exist_ok=True);shutil.copy2(doc,backup/(OUT.name+'_previous_design.md'))
doc.write_text(qc+'\n\n## 图表与正文对应\n第一段A；第二段C与Table Sx；第三段Figure Sx；第四段B；第五段总体总结。\n\n'+captions,encoding='utf-8')
shutil.copy2(Path(__file__),OUT/Path(__file__).name);text('prompts/136_section32_figures_and_tables.md');HASH[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(OUT/'source_sha256.json').write_text(json.dumps(HASH,indent=2),encoding='utf-8')
(OUT/'verification.json').write_text(json.dumps(dict(status='FIGURES_READY_WITH_CAVEATS',table_rows=30,annual_points=50,positive_years=counts.reindex(S).tolist(),new_bootstrap=0,new_training=0,new_simulation=0,figure_width_mm=182.88,png_dpi=400,forecast_included=False),indent=2),encoding='utf-8')
print(json.dumps({'output':str(OUT),'status':'generated'},ensure_ascii=True))
