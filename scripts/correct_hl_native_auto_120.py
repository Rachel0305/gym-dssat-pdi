"""Correct HL comparator using archived native DSSAT auto; organize local deliverables."""
from pathlib import Path
import json,hashlib,shutil,re,gc
import pandas as pd
import numpy as np
import plot_four_site_five_scenario_118 as m
ROOT=m.ROOT
OLD=m.OUT
OUT=ROOT/'results/forecast_five_scenario_120_corrected_HL_20261027'
OUT.mkdir(exist_ok=False)
SOURCES={}
def track(p):SOURCES[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
def folder(site,seed):return OUT/site/f'seed_{seed:02d}'
def archive_figures(source,dest):
    for f in source.iterdir():
        if f.suffix.lower() not in ['.png','.pdf','.svg']:continue
        year=re.search(r'(201[4-9]|202[0-3])',f.stem)
        target=dest/'yearly'/year[1] if year else dest/'summary'
        target.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target/f.name)

# Preserve all existing artifacts. Unaffected stations are copied into a self-contained local folder.
for site in ['SY','LC','YC','FQ']:
    for seed in range(8):
        source=(ROOT/'results/sy_forecast_five_scenario_figures_100_20261023'/f'seed_{seed:02d}') if site=='SY' else OLD/site/f'seed_{seed:02d}'
        dest=folder(site,seed);dest.mkdir(parents=True)
        archive_figures(source/'figures',dest)
        shutil.copytree(source/'tables',dest/'tables')
    shutil.copytree(OLD/site/'station_summary',OUT/site/'station_summary')

native=[];checks=[];days={}
for year in m.YEARS:
    snap=ROOT/'benchmark_results/054_02_hla_lowIC_four_baselines_static_level1/snapshots/HLA'/str(year)/'dssat_auto'
    for name in ['Weather.OUT','PlantGro.OUT','SoilWat.OUT','MgmtEvent.OUT','Summary.OUT','fileX.MZX']:track(snap/name)
    row=m.parse_summary_out(snap/'Summary.OUT')[-1]
    events=m.parse_management_events(snap/'MgmtEvent.OUT')
    day=m.daily_from_snapshot(snap,year,'dssat_auto');day['site']='HL';day['station_code']='HLA'
    assert row['NICM']==0 and not events.operation.eq('fertilizer').any()
    assert day.nitrogen_executed_kg_ha.sum()==0
    assert abs(day.irrigation_executed_mm.sum()-row['IRCM'])<=.51
    text=(snap/'fileX.MZX').read_text(encoding='latin-1')
    management=next(l for l in text.splitlines() if re.match(r'\s*1\s+MA\s',l)).split()
    assert management[3]=='A',management
    days[year]=day
    native.append(dict(site='HL',station_code='HLA',year=year,scenario='dssat_auto',grain_yield_kg_ha=row['HWAM'],biomass_kg_ha=day.biomass_kg_ha.iloc[-1],actual_irrigation_mm=row['IRCM'],actual_nitrogen_kg_ha=0,etcp_mm=row['ETCP'],WP_ET_kg_m3=row['HWAM']/row['ETCP']/10,PFP_N_kg_kg=np.nan,unified_reward=.158*row['HWAM']-1.1*row['IRCM'],snapshot_path=snap.relative_to(ROOT).as_posix(),event_I_mm=day.irrigation_executed_mm.sum(),event_N_kg_ha=0))
    checks.append(dict(year=year,passed=True,native_irrigation_mode=management[3],archived_fertilizer_mode=management[4],external_n_controller=False,Summary_NICM=0,final_run_fertilizer_events=0,event_irrigation_mm=day.irrigation_executed_mm.sum(),summary_irrigation_mm=row['IRCM'],snapshot=snap.relative_to(ROOT).as_posix()))
pd.DataFrame(native).to_csv(OUT/'HL_native_auto_10years.csv',index=False)
(OUT/'HL_native_auto_audit.json').write_text(json.dumps(checks,indent=2))
m.SCENARIOS=[('dssat_auto' if s=='dssat_auto_external_n' else s) for s in m.SCENARIOS]
m.LABELS['dssat_auto']='DSSAT auto (native; N=0)'
m.COLORS['dssat_auto']=m.COLORS['dssat_auto_external_n'];m.STYLES['dssat_auto']='-.'
# Retain publication vectors; reduce PNG compression cost, not image resolution.
def save_fast(self,path,*a,**kw):
    if Path(path).suffix=='.png':kw['pil_kwargs']={'compress_level':1}
    m.original_save(self,path,*a,**kw)
    if Path(path).suffix=='.png':
        m.original_save(self,Path(path).with_suffix('.pdf'))
        m.original_save(self,Path(path).with_suffix('.svg'))
m.matplotlib.figure.Figure.savefig=save_fast
summaries=[]
for seed in range(8):
    source=OLD/'HL'/f'seed_{seed:02d}'/'tables';dest=folder('HL',seed);tab=dest/'tables';fig=dest/'figures'
    tab.mkdir(parents=True);fig.mkdir()
    track(source/'five_scenario_daily.csv');track(source/'five_scenario_season_summary.csv')
    oldday=pd.read_csv(source/'five_scenario_daily.csv',keep_default_na=False)
    oldsum=pd.read_csv(source/'five_scenario_season_summary.csv',keep_default_na=False)
    oldday=oldday[~oldday.scenario.eq('dssat_auto_external_n')].copy();oldsum=oldsum[~oldsum.scenario.eq('dssat_auto_external_n')].copy()
    native_day=pd.concat(days.values(),ignore_index=True);native_day['seed']=seed
    for year in m.YEARS:
        base=oldday[oldday.year.eq(year)&oldday.scenario.eq('null')].copy();base['date']=pd.to_datetime(base.date)
        comp=base.merge(days[year],on='date',suffixes=('_b','_a'))
        assert len(comp)>50
        for c in ['rainfall_mm','tmax_c','tmin_c']:assert np.allclose(comp[c+'_b'],comp[c+'_a'],atol=.11,rtol=0)
    daily=m.add_unified_reward(pd.concat([oldday,native_day],ignore_index=True))
    ss=pd.concat([oldsum,pd.DataFrame(native).assign(seed=seed)],ignore_index=True)
    cols=['grain_yield_kg_ha','WP_ET_kg_m3','PFP_N_kg_kg','actual_irrigation_mm','actual_nitrogen_kg_ha','unified_reward']
    ss[cols]=ss[cols].apply(pd.to_numeric,errors='coerce')
    assert len(ss)==50 and set(ss.scenario)==set(m.SCENARIOS)
    daily.to_csv(tab/'five_scenario_daily.csv',index=False);ss.to_csv(tab/'five_scenario_season_summary.csv',index=False)
    daily[daily.irrigation_executed_mm.gt(0)|daily.nitrogen_executed_kg_ha.gt(0)].to_csv(tab/'five_scenario_management_events.csv',index=False)
    m.SITE='HL';m.SEED=seed
    for year in m.YEARS:m.plot_year(daily[daily.year.eq(year)],year,fig)
    m.draw_grouped(ss,m.YEARS,cols[:3],fig/'five_scenario_metrics.png')
    m.draw_grouped(ss,m.YEARS,cols[3:],fig/'five_scenario_management.png')
    # Place generated files directly into the final local year/summary hierarchy.
    for f in list(fig.iterdir()):
        year=re.search(r'(201[4-9]|202[0-3])',f.stem);target=dest/'yearly'/year[1] if year else dest/'summary'
        target.mkdir(parents=True,exist_ok=True);f.rename(target/f.name)
    summaries.append(ss);gc.collect();print(f'HL seed{seed}: corrected 10 yearly + 2 summary figures',flush=True)

allp=OLD/'all_five_sites_seed_year_scenario.csv';track(allp)
allrows=pd.read_csv(allp,keep_default_na=False);allrows=allrows[~allrows.site.eq('HL')]
allrows=pd.concat([allrows,*summaries],ignore_index=True);assert len(allrows)==2000
allrows.to_csv(OUT/'all_five_sites_seed_year_scenario.csv',index=False)
metrics=['grain_yield_kg_ha','actual_irrigation_mm','actual_nitrogen_kg_ha','WP_ET_kg_m3','PFP_N_kg_kg','unified_reward']
allrows[metrics]=allrows[metrics].apply(pd.to_numeric,errors='coerce')
allrows.groupby(['site','seed','scenario'])[metrics].mean().to_csv(OUT/'seed_scenario_means.csv')
hl=pd.concat(summaries);mean=hl.groupby(['year','scenario'])[metrics].mean().reset_index();station=OUT/'HL/station_summary';station.mkdir()
m.LABELS['rl_candidate']='Forecast PPO mean (8 seeds)'
mean.to_csv(station/'yearly_five_scenario_means.csv',index=False)
m.draw_grouped(mean,m.YEARS,[metrics[0],metrics[3],metrics[4]],station/'yearly_metrics.png')
m.draw_grouped(mean,m.YEARS,[metrics[1],metrics[2],metrics[5]],station/'yearly_management.png')
# PPO-only seed comparison did not change.
for f in (OLD/'HL/station_summary').glob('all_seeds_yearly_comparison.*'):shutil.copy2(f,station/f.name)
links=['<!doctype html><html lang=zh><meta charset=utf-8><title>五站本地图表 · HL原生auto修正</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}section{padding:20px;border:1px solid #ddd;margin:20px 0}a{margin-right:12px}summary{padding:10px;cursor:pointer}</style><h1>五站本地图表 · HL原生DSSAT auto修正</h1><p>HL对照为原生DSSAT auto，无外部施氮；十年实际N=0。其他站点保留既有口径。所有图片与表格均已复制到本目录，按站点/seed/年份归档。</p><p><a href=all_five_sites_seed_year_scenario.csv>五站结果表</a><a href=HL_native_auto_10years.csv>HL原生auto十年表</a><a href=HL_native_auto_audit.json>HL措施核验</a></p>']
for site in ['SY','HL','LC','YC','FQ']:
    links.append(f'<section><h2>{site}</h2><a href={site}/station_summary/yearly_metrics.png>站点逐年指标</a><a href={site}/station_summary/yearly_management.png>站点逐年措施</a>')
    for seed in range(8):
        base=folder(site,seed);links.append(f'<details><summary>seed {seed:02d}</summary>')
        for group in ['summary','yearly','tables']:
            for f in sorted((base/group).rglob('*')):
                if f.is_file():links.append(f'<p><a href={f.relative_to(OUT).as_posix()}>{group} / {f.parent.name} / {f.name}</a></p>')
        links.append('</details>')
    links.append('</section>')
(OUT/'index.html').write_text('\n'.join(links),encoding='utf-8')
(OUT/'source_sha256.json').write_text(json.dumps(SOURCES,indent=2))
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in SOURCES.items())
(OUT/'README_zh.md').write_text('五站40seed本地图表归档。每站/seed_00至07/yearly/2014至2023存放逐年图，summary存指标和措施汇总，tables存CSV。HL改用054_02原生DSSAT auto十年快照，Summary.OUT NICM=0且MgmtEvent最终运行无施肥；删除的是新表内外部N对照行，原始118结果和训练数据未改。原生归档fileX的FERTI字段为A，但十年自动施肥实际执行量均为0；不将其描述成FERTI开关已禁用。无新增模拟或训练。HL对照PFP_N因N=0未定义。其他站点图表原样复制。\n',encoding='utf-8')
(OUT/'result.json').write_text(json.dumps(dict(status='completed',HL_corrected_seed_figures=96,HL_native_auto_NICM_all_zero=True,original_sources_unchanged=True,new_simulations=0,local_pngs=len(list(OUT.rglob('*.png')))),indent=2))
print('CORRECTION AND LOCAL ARCHIVE COMPLETED',flush=True)
