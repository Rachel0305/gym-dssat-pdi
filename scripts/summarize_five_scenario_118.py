from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_four_site_five_scenario_118 as plot
OUT=plot.OUT
data=pd.read_csv(OUT/'all_sites_seed_year_scenario.csv',keep_default_na=False)
metrics=['grain_yield_kg_ha','actual_irrigation_mm','actual_nitrogen_kg_ha','WP_ET_kg_m3','PFP_N_kg_kg','unified_reward']
for c in metrics:data[c]=pd.to_numeric(data[c],errors='coerce')
gaps=[]
for (site,seed,year),g in data.groupby(['site','seed','year']):
    p=g[g.scenario.eq('rl_candidate')].iloc[0]
    for _,b in g[~g.scenario.eq('rl_candidate')].iterrows():
        gaps.append(dict(site=site,seed=int(seed),year=int(year),control=b.scenario,**{'delta_'+m:p[m]-b[m] for m in metrics}))
pd.DataFrame(gaps).to_csv(OUT/'ppo_minus_each_control.csv',index=False)
audit=pd.DataFrame(json.loads((OUT/'audit_all.json').read_text()))
controls=data[~data.scenario.eq('rl_candidate')].drop_duplicates(['site','year','scenario'])
controls[['site','year','scenario','actual_irrigation_mm','event_I_mm','event_I_minus_summary_mm','actual_nitrogen_kg_ha','event_N_kg_ha','event_N_minus_summary_kg_ha']].to_csv(OUT/'control_management_rounding_audit.csv',index=False)
sheets={
 '逐年五情景':data[['site','seed','year','scenario']+metrics+['snapshot_path']],
 '种子均值':data.groupby(['site','seed','scenario'])[metrics].mean().reset_index(),
 'PPO对照差值':pd.DataFrame(gaps),
 '措施舍入核验':pd.read_csv(OUT/'control_management_rounding_audit.csv',keep_default_na=False),
 '逐年核验':audit,
}
matrix={name:[list(df.columns)]+df.astype(object).where(df.notna(),None).values.tolist() for name,df in sheets.items()}
(OUT/'workbook_data.json').write_text(json.dumps(matrix,ensure_ascii=False),encoding='utf-8')
for site,g in data.groupby('site'):
    dest=OUT/site/'station_summary';dest.mkdir(exist_ok=True)
    mean=g.groupby(['year','scenario'])[metrics].mean().reset_index()
    mean.to_csv(dest/'yearly_five_scenario_means.csv',index=False)
    plot.LABELS['rl_candidate']='Forecast PPO mean (8 seeds)'
    plot.draw_grouped(mean,plot.YEARS,['grain_yield_kg_ha','WP_ET_kg_m3','PFP_N_kg_kg'],dest/'yearly_metrics.png')
    plot.draw_grouped(mean,plot.YEARS,['actual_irrigation_mm','actual_nitrogen_kg_ha','unified_reward'],dest/'yearly_management.png')
    fig,axes=plt.subplots(3,1,figsize=(14,10),sharex=True)
    for seed,s in g[g.scenario.eq('rl_candidate')].groupby('seed'):
        for ax,m in zip(axes,metrics[:3]):ax.plot(s.year,s[m],marker='o',lw=1.1,label=f'seed {seed}')
    for ax,m in zip(axes,metrics[:3]):ax.set_ylabel(m);ax.grid(alpha=.2)
    axes[0].legend(ncol=8);axes[-1].set_xticks(plot.YEARS);axes[-1].set_xlabel('Validation year')
    fig.suptitle(f'{site} | Forecast PPO: all 8 seeds');fig.tight_layout()
    fig.savefig(dest/'all_seeds_yearly_comparison.png',dpi=200);plt.close(fig)
    print('STATION SUMMARY',site,flush=True)
lines=['# 四站逐年五情景图表','', 'HL、LC、YC、FQ各8个seed；2014–2023固定100K验证，无新增模拟。','',
 '五情景：Null、Recorded template、DSSAT auto + external N、Official expert、PPO + perfect 7-day summary。',
 '每seed：10张逐年八面板过程图、指标汇总和措施汇总各1张；每图PNG/PDF/SVG。',
 '核验：PPO动作日期/剂量对照MgmtEvent，最终Summary.OUT水氮闭合；四组对照从快照最后运行重新提取，天气与PPO匹配。旧表HL2014自动灌溉混合追加运行导致167mm，已按最后运行提取。',
 'MgmtEvent剂量显示到0.1、Summary水氮常为整数，两者显示精度差保留在措施舍入核验表；不人为缩放事件。',
 '统一展示分数=0.158×产量−1.1×灌溉−1.58×施氮，不是训练奖励或经济收益。PFP_N在N=0时未定义。WP_ET=同次Summary产量/(10×ETCP)。',
 '8seed共用四组固定对照，对照重复不构成独立样本。站点图为均值，逐seed图为原始记录，无显著性检验。','',
 '[Excel结果表](../../outputs/five_scenario_118/five_scenario_results.xlsx)',
 '[1600行逐年情景表](all_sites_seed_year_scenario.csv)', '[PPO减各对照差值](ppo_minus_each_control.csv)',
 '[SY已完成同类图包](../sy_forecast_five_scenario_figures_100_20261023/README_zh.md)','']
for site in plot.BASES:
    lines += [f'## {site}',f'[站点汇总](./{site}/station_summary/yearly_metrics.png)',f'[8种子逐年比较](./{site}/station_summary/all_seeds_yearly_comparison.png)']
    for seed in range(8):lines.append(f'- [seed {seed:02d} 指标](./{site}/seed_{seed:02d}/figures/five_scenario_metrics.png) · [2014过程图](./{site}/seed_{seed:02d}/figures/{site}_2014_five_scenario_daily.png) · [逐年表](./{site}/seed_{seed:02d}/tables/five_scenario_season_summary.csv)')
(OUT/'README_zh.md').write_text('\n'.join(lines),encoding='utf-8')
