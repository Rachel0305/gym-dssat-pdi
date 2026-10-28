"""Serial tables and manuscript plots; no model or simulator imports."""
from pathlib import Path
import io,json,hashlib,re,time
import pandas as pd
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/three_route_results_126_20261027';OUT.mkdir(exist_ok=False)
sources={};started=time.time();SITES=['SY','HL','LC','YC','FQ'];ROUTES=['historical','augmented','forecast'];MET=['grain_yield_kg_ha','total_irrigation_mm','total_n_kg_ha']
def read(p):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    assert p.resolve().is_relative_to(ROOT.resolve())
    b=p.read_bytes();sources[p.relative_to(ROOT).as_posix()]=hashlib.sha256(b).hexdigest();return b
def csv(p):return pd.read_csv(io.BytesIO(read(p)),keep_default_na=False)
def numeric(q,cols):
    for col in cols:q[col]=pd.to_numeric(q[col],errors='coerce')
    return q
old=csv('results/benefit_reliability_066_20261019/primary_policy_table_80.csv')
f=csv('results/forecast_publication_audit_122_20261027/forecast_policy_table_40.csv')
d=csv('results/forecast_five_scenario_120_corrected_HL_20261027/all_five_sites_seed_year_scenario.csv')
annual=[];references=[]
for r in old.itertuples():
    q=csv(f'2V02/deliverables/{r.site}/{r.regime}/seed_{int(r.seed):02d}/tables/yearly_metrics.csv')
    a=q[q.scenario.eq(f'ppo_seed_{r.seed}')].copy();assert len(a)==10
    a['site']=r.site;a['regime']=r.regime;a['seed']=r.seed
    a=numeric(a,MET+['WP_ET_kg_m3','PFP_N']);annual.append(a[['site','regime','seed','year']+MET+['WP_ET_kg_m3','PFP_N']])
    for metric in MET:assert abs(a[metric].mean()-getattr(r,metric))<1e-6,(r.site,r.seed,metric)
    if r.regime=='historical' and r.seed==0:
        ref=q[q.scenario.eq(r.reference_scenario)].copy();ref['site']=r.site;references.append(ref[['site','year','scenario']+MET])
references=pd.concat(references,ignore_index=True);assert len(references)==50
fp=numeric(d[d.scenario.eq('rl_candidate')].copy(),['grain_yield_kg_ha','actual_irrigation_mm','actual_nitrogen_kg_ha','WP_ET_kg_m3','PFP_N_kg_kg'])
fp=fp.rename(columns={'actual_irrigation_mm':'total_irrigation_mm','actual_nitrogen_kg_ha':'total_n_kg_ha','PFP_N_kg_kg':'PFP_N'});fp['regime']='forecast'
def finalval(p,key):
    lines=read(p).decode(errors='replace').splitlines();h=next(x for x in lines if x.startswith('@'));last=next(x for x in reversed(lines) if x.strip() and x.lstrip()[0].isdigit());end=h.index(key)+len(key);return float(last[end-5:end])
for seed in range(8):
    folder=Path(f'results/fq2018_forecast_reassessment_125_20261027/seed{seed}');s=json.loads(read(folder/'eval_100000_2018_summary.json'));assert json.loads(read(folder/'technical_gate.json'))['status']=='passed'
    mask=fp.site.eq('FQ')&fp.seed.eq(seed)&fp.year.eq(2018);assert mask.sum()==1
    et=finalval(folder/'eval_100000_2018_snapshot/dssat/Summary.OUT','ETCP');assert et>0
    fp.loc[mask,MET+['WP_ET_kg_m3','PFP_N']]=[s['yield_kg_ha'],s['I'],s['N'],s['yield_kg_ha']/et/10,s['yield_kg_ha']/s['N'] if s['N'] else np.nan]
annual.append(fp[['site','regime','seed','year']+MET+['WP_ET_kg_m3','PFP_N']]);a=pd.concat(annual,ignore_index=True)
keys=['site','regime','seed'];assert len(a)==1200 and not a.duplicated(keys+['year']).any();assert a.groupby(keys).size().eq(10).all();assert a.groupby(['site','regime']).size().eq(80).all()
assert a.WP_ET_kg_m3.notna().all();assert a.PFP_N.isna().eq(a.total_n_kg_ha.eq(0)).all()
p=a.groupby(keys,as_index=False)[MET+['WP_ET_kg_m3','PFP_N']].mean()
refs=references.groupby('site')[MET].mean().rename(columns=dict(zip(MET,['reference_yield','reference_I','reference_N'])))
p=p.merge(refs,on='site',validate='many_to_one')
# Preserve old behavior evidence; reconstruct forecast with corrected FQ2018 events.
behavior=old[keys+['small_15mm_fraction','frequency_ratio_expert','fixed_full_sequence']].copy();beh=[]
for site in SITES:
    for seed in range(8):
        days=csv(f'results/forecast_five_scenario_120_corrected_HL_20261027/{site}/seed_{seed:02d}/tables/five_scenario_daily.csv');days=days[days.scenario.eq('rl_candidate')]
        if site=='FQ':days=days[days.year.ne(2018)]
        ev=days[(days.irrigation_executed_mm>0)|(days.nitrogen_executed_kg_ha>0)][['year','dap','irrigation_executed_mm','nitrogen_executed_kg_ha']].copy()
        if site=='FQ':
            path=f'results/fq2018_forecast_reassessment_125_20261027/seed{seed}/eval_100000_2018_snapshot/dssat/MgmtEvent.OUT'
            txt=read(path).decode('latin-1').rsplit('@RUN',1)[-1];new=[]
            pattern=r'^\s*\d+\s+[A-Z]{3}\s+\d+,\s+\d{4}\s+(\d+)\s+\d+\s+(-?\d+)\s+MZ\s+.*?\b(Irrigation|Fertilizer)\s+([0-9.]+)'
            for line in txt.splitlines():
                h=re.search(pattern,line,re.I)
                if h:new.append(dict(year=2018,dap=int(h[2]),irrigation_executed_mm=float(h[4]) if h[3].lower()=='irrigation' else 0,nitrogen_executed_kg_ha=float(h[4]) if h[3].lower()=='fertilizer' else 0))
            ev=pd.concat([ev,pd.DataFrame(new)],ignore_index=True)
        ev=ev.groupby(['year','dap'],as_index=False)[['irrigation_executed_mm','nitrogen_executed_kg_ha']].sum()
        for year,g in ev.groupby('year'):
            end=a[a.site.eq(site)&a.regime.eq('forecast')&a.seed.eq(seed)&a.year.eq(year)].iloc[0]
            assert abs(g.irrigation_executed_mm.sum()-end.total_irrigation_mm)<1e-5
            assert abs(g.nitrogen_executed_kg_ha.sum()-end.total_n_kg_ha)<1e-5
        irr=ev[ev.irrigation_executed_mm.gt(0)];expertfreq=old[old.site.eq(site)].expert_irrigation_events_per_year.unique();assert len(expertfreq)==1
        seq=[tuple(map(tuple,ev[ev.year.eq(y)][['dap','irrigation_executed_mm','nitrogen_executed_kg_ha']].round(3).to_numpy())) for y in range(2014,2024)]
        beh.append(dict(site=site,regime='forecast',seed=seed,small_15mm_fraction=irr.irrigation_executed_mm.le(15).mean(),frequency_ratio_expert=len(irr)/10/expertfreq[0],fixed_full_sequence=len(set(seq))==1))
behavior=pd.concat([behavior,pd.DataFrame(beh)],ignore_index=True);p=p.merge(behavior,on=keys,validate='one_to_one')
p['yield_delta_pct']=(p.grain_yield_kg_ha/p.reference_yield-1)*100
p['gain']=p.yield_delta_pct.ge(1-1e-8);p['N1']=p.yield_delta_pct.ge(-5-1e-8)
p['N2']=p.N1&p.total_irrigation_mm.le(p.reference_I+1e-8)&p.total_n_kg_ha.le(p.reference_N+1e-8)
p['saving']=p.N2&(p.total_irrigation_mm.le(.9*p.reference_I+1e-8)|p.total_n_kg_ha.le(.9*p.reference_N+1e-8));p['union']=p.gain|p.saving
p['N3']=p.N2&p.small_15mm_fraction.le(2/3)&p.frequency_ratio_expert.le(1.5)
def tally(q):return dict(n=len(q),gain=int(q.gain.sum()),saving=int(q.saving.sum()),overlap=int((q.gain&q.saving).sum()),union=int(q.union.sum()),N1=int(q.N1.sum()),N2=int(q.N2.sum()),N3=int(q.N3.sum()),N3_fixed=int((q.N3&q.fixed_full_sequence).sum()))
counts={'legacy80':tally(p[p.regime.ne('forecast')]),'forecast40':tally(p[p.regime.eq('forecast')]),'combined120':tally(p)}
assert [counts['legacy80'][k] for k in ['gain','saving','N1','N2','N3']]==[13,31,51,31,19]
summary=p.groupby(['site','regime']).agg(n=('seed','count'),yield_mean=('grain_yield_kg_ha','mean'),yield_sd=('grain_yield_kg_ha','std'),I_mean=('total_irrigation_mm','mean'),N_mean=('total_n_kg_ha','mean'),WP_ET_mean=('WP_ET_kg_m3','mean'),PFP_N_mean=('PFP_N','mean'),PFP_N_valid_seeds=('PFP_N','count')).reset_index()
contrasts=[]
for site in SITES:
    for route in ['augmented','forecast']:
        for metric in MET:
            x=p[p.site.eq(site)&p.regime.eq(route)][metric];y=p[p.site.eq(site)&p.regime.eq('historical')][metric]
            vx=x.var(ddof=1)/8;vy=y.var(ddof=1)/8;se=np.sqrt(vx+vy);df=(vx+vy)**2/(vx*vx/7+vy*vy/7);delta=x.mean()-y.mean();margin=t.ppf(.975,df)*se
            contrasts.append(dict(site=site,route=route,metric=metric,delta=delta,welch95_low=delta-margin,welch95_high=delta+margin,includes_zero=delta-margin<=0<=delta+margin))
c=pd.DataFrame(contrasts)
ar=a.merge(references.rename(columns={'grain_yield_kg_ha':'reference_yield','total_irrigation_mm':'reference_I','total_n_kg_ha':'reference_N'}).drop(columns='scenario'),on=['site','year'],validate='many_to_one');ar['yield_delta_pct']=(ar.grain_yield_kg_ha/ar.reference_yield-1)*100
ar['loss_over_5pct']=ar.yield_delta_pct.lt(-5)
for name,q in [('annual_1200',a),('policy_120',p),('site_route_summary_15',summary),('contrasts_30',c),('annual_reference_risk_1200',ar),('reference_annual_50',references)]:q.to_csv(OUT/(name+'.csv'),index=False)
pd.DataFrame([dict(site=s,route=r,**tally(g)) for (s,r),g in p.groupby(['site','regime'])]).to_csv(OUT/'counts_by_site_route_15.csv',index=False)
(OUT/'counts.json').write_text(json.dumps(counts,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
colors=['#555555','#D18B27','#168A71'];names=['Original PPO','Weather augmentation','Perfect-weather forecast']
fig,axes=plt.subplots(5,3,figsize=(13,14));rng=np.random.default_rng(126)
for i,site in enumerate(SITES):
    for j,metric in enumerate(MET):
        ax=axes[i,j]
        for k,route in enumerate(ROUTES):
            v=p[p.site.eq(site)&p.regime.eq(route)][metric].to_numpy();ax.scatter(np.full(8,k)+rng.uniform(-.10,.10,8),v,s=20,color=colors[k],alpha=.75);ax.plot([k-.2,k+.2],[v.mean()]*2,color='black',lw=2)
        ax.set_xticks(range(3),['Original','Augmented','Forecast']);ax.set_title(site+' | '+['Yield (kg/ha)','Irrigation (mm)','Nitrogen (kg/ha)'][j]);ax.grid(axis='y',alpha=.2)
fig.suptitle('Three routes | 8 seeds per route | 2014-2023 means',y=.995);fig.tight_layout(rect=[0,.025,1,.98]);fig.text(.03,.008,'FQ 2018: common heuristic TMIN quality-control version. Dots = seeds; black line = mean. Descriptive route comparisons.',fontsize=9)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'three_routes_all_seeds.{ext}',dpi=160)
plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,5))
for ax,metric,title in zip(axes,MET,['Yield difference (kg/ha)','Irrigation difference (mm)','Nitrogen difference (kg/ha)']):
    for k,route in enumerate(['augmented','forecast']):
        z=c[c.route.eq(route)&c.metric.eq(metric)].set_index('site').loc[SITES];y=np.arange(5)+(k-.5)*.18
        ax.errorbar(z.delta,y,xerr=[z.delta-z.welch95_low,z.welch95_high-z.delta],fmt='o',color=colors[k+1],label=names[k+1],capsize=3)
    ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(5),SITES);ax.set_title(title);ax.invert_yaxis()
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.92),ncol=2,frameon=False,fontsize=9);fig.suptitle('Route minus original PPO | independent-seed Welch 95% intervals');fig.tight_layout(rect=[0,.06,1,.83]);fig.text(.03,.015,'n=8 per route; intervals unadjusted for multiple comparisons. Zero-crossing is not evidence of equivalence.',fontsize=9)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'route_contrasts.{ext}',dpi=160)
plt.close(fig)
report=f'''# 三路线结果：共同FQ2018质量控制版本

本轮仅整理数据和绘图，没有新增模拟或训练。旧80策略从2V02读取，保持原计数；新增预报40策略替换125实际补评的8条FQ2018记录。三路线共享FQ2018的088 TMIN质量控制版本。其他运行合同尚未全部认证，故这是条件明确的描述性比较，不能声称严格单因素因果效果。

## 更新后的计数

{json.dumps(counts,ensure_ascii=False,indent=2)}

增产为十年均产相对主参照>=1%；节资为产量>=95%、水氮均不增加且至少一项<=90%；并集去重。N1为产量门槛，N2增加水氮不增加，N3增加小额灌溉占比<=2/3及事件频率<=1.5倍专家。固定日程保留。门槛为事后探索性，不将通过名单当部署推荐。

## 四条结论如何修订

1. 站点收益及资源权衡：在三路线共120策略中报告新增产/节资计数；同时保持每路线每站8策略的均值与全部点，不能只呈现最佳种子。
2. 重复性：用15组独立种子离散度和年度风险描述典型表现与下行风险；不把1200年度当独立模型，不按编号配对种子。
3. 投入和操作：采用更新后的N1→N2→N3及实际2018事件，不能只替换产量仍沿用原版操作标签。零氮PFP_N保持未定义，平均比值须报告有效数量；PFP_N不是NUE。
4. 天气扩展：增强与预报均与原PPO分站比较，产量/灌溉/氮30项Welch区间见contrasts_30.csv。天气增强改变训练经历，预报改变观测信息；响应和收益必须分开。理想预报并非实务预报，当前尚无增强+预报第四组，不推断互补效应。

## 图表和口径

three_routes_all_seeds展示全部8种子；route_contrasts展示独立种子Welch区间，未校正多重比较。annual_1200与policy_120可逐行复算，counts_by_site_route_15保留站点异质性。PFP_N组均值来自每策略有效年度比值均值再跨策略平均，零N年度不填0，不得隐去有效数差异。

只把共同2018质量控制版本称为质量控制/敏感性输入，不声称15处符号修正为独立观测真值。原版零产量和原120表图不覆盖；旧122暂存计数由本版有明确输入条件的计数替代引用。停止扩大天气审查，后续围绕预报响应机制与论文表达开展。
'''
(OUT/'report_zh.md').write_text(report,encoding='utf-8');(OUT/'source_sha256.json').write_text(json.dumps(sources,indent=2))
(OUT/'QA.json').write_text(json.dumps(dict(annual_rows=1200,policies=120,groups=15,contrasts=30,legacy_counts_reproduced=True,forecast_event_closure=True,original_results_overwritten=False,new_simulations=0,elapsed_seconds=time.time()-started),indent=2))
print(json.dumps(counts,indent=2));print(summary[summary.site.eq('FQ')].to_string(index=False));print(c[c.site.eq('FQ')].to_string(index=False))
