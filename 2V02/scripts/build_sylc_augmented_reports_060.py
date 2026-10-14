"""Frozen-style reports, exact ETCP replay, paired cohorts and verified backup."""
import os
os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
from pathlib import Path
import sys,json,hashlib,inspect,subprocess,time,csv,zipfile,signal,argparse
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/sylc_external_weather_060'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/053_lca_lowIC_site_transfer')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def runroot(site,seed):return ROOT/f'benchmark_results/059{site}s{seed}_frozen_weather_augmented_100k'
def control(site,seed):
    if seed==0:return ROOT/('benchmark_results/046_10_sya_originIC_expanded_action_maskableppo' if site=='SY' else 'benchmark_results/053_00_lca_lowIC_expanded_action_maskableppo')
    task='sya_originIC_expanded_action_maskableppo' if site=='SY' else 'lca_lowIC_expanded_action_maskableppo'
    return ROOT/f'benchmark_results/231_{site.lower()}_seed{seed}_{task}_seed{seed}_extension'
def frame(p):
    import pandas as pd
    return pd.read_csv(p,keep_default_na=False)
def audit_contracts():
    assert json.loads((ROOT/'results/sylc_external_weather_059/queue_status.json').read_text())['status']=='COMPLETE'
    records=[]
    import plot_sy_lc_seed_extension_231 as template
    for site in ['SY','LC']:
        for seed in range(8):
            a=ROOT/f'results/sylc_external_weather_059/{site}/seed{seed}/config.json'
            b=ROOT/('configs/046_10_sya_originIC_expanded_action_maskableppo.json' if site=='SY' else 'configs/053_00_lca_lowIC_expanded_action_maskableppo.json') if seed==0 else ROOT/f'results/sy_lc_8seed_extension_231/configs/{site.lower()}_seed{seed}.json'
            x=json.loads(a.read_text());y=json.loads(b.read_text())
            for obj in [x,y]:
                for key in ['task_id','task_name','seed']:obj.pop(key,None)
            assert x==y,(site,seed,'contract difference')
            for label,root in [('augmented',runroot(site,seed)),('historical',control(site,seed))]:
                p=root/'evaluation/032_22_checkpoint_validation_summary.csv';f=frame(p)
                assert len(f)==40 and set(zip(f.checkpoint_step,f.year))=={(s,y) for s in [25000,50000,75000,100000] for y in range(2014,2024)}
                assert f.run_status.str.startswith('ok').all() and set(f.seed)=={seed}
                inv=frame(root/'evaluation/032_22_training_checkpoint_inventory.csv')
                for r in inv.itertuples():assert sha(ROOT/r.model_path)==r.model_sha256
                records.append({'site':site,'seed':seed,'domain':label,'config_sha256':sha(a if label=='augmented' else b),'summary':p.relative_to(ROOT).as_posix(),'summary_sha256':sha(p),'rows':len(f)})
            complete=json.loads((ROOT/f'results/sylc_external_weather_059/{site}/seed{seed}/completion_audit.json').read_text());assert complete['all_archive_sha256_verified']
        for p in template.source_tables(site):records.append({'site':site,'frozen_baseline':p.relative_to(ROOT).as_posix(),'sha256':sha(p)})
    audit=OUT/'source_contract_audit.json'
    if audit.exists():assert json.loads(audit.read_text())==records
    else:audit.write_text(json.dumps(records,indent=2))
def worker(site,seed):
    import plot_sy_lc_seed_extension_231 as template
    template.REPORT=OUT/'five_scenario_archive'
    source=inspect.getsource(template.run)
    source=source.replace('config_path = BASE / "configs" / f"{site.lower()}_seed{seed}.json"','config_path = ROOT / f"results/sylc_external_weather_059/{site}/seed{seed}/config.json"')
    source=source.replace('ppo_root = ROOT / "benchmark_results" / f"231_{site.lower()}_seed{seed}_{TASK[site]}_seed{seed}_extension"','ppo_root = ROOT / f"benchmark_results/059{site}s{seed}_frozen_weather_augmented_100k"')
    source=source.replace('f"231_{site.lower()}_seed{seed}_formal_result.json"','f"059{site}s{seed}_formal_result.json"')
    exec(compile(source,__file__,'exec'),template.__dict__)
    existing=template.REPORT/site/f'seed_{seed:02d}'/'audit.json'
    if not existing.exists():template.run(site,seed)
    else:
        a=json.loads(existing.read_text());assert a['summary_rows']==50 and a['figure_count']==12
    p=template.REPORT/site/f'seed_{seed:02d}'
    table=p/'tables'/f'{template.PREFIX[site]}_five_scenario_season_summary.csv'
    data=template.normalize_summary(site,frame(table)) if site=='SY' else frame(table)
    expected=frame(runroot(site,seed)/'evaluation/032_22_checkpoint_validation_summary.csv');expected=expected[expected.checkpoint_step.eq(100000)].set_index('year')
    replay=data[data.scenario.eq('rl_candidate')]
    for r in replay.itertuples():
        e=expected.loc[r.year]
        assert abs(r.grain_yield_kg_ha-e.final_grnwt)<=1
        assert abs(r.actual_irrigation_mm-e.total_irrigation)<=.51
        assert abs(r.actual_nitrogen_kg_ha-e.total_n)<=.51
        # Existing reports round yield to 1 kg, ETCP to 0.1 mm and WP to 0.01.
        assert r.etcp_mm>0
        lower=(r.grain_yield_kg_ha-.5)/((r.etcp_mm+.05)*10)-.005
        upper=(r.grain_yield_kg_ha+.5)/((r.etcp_mm-.05)*10)+.005
        assert lower<=r.WP_ET_kg_m3<=upper
    mapping={'actual_irrigation_mm':'total_irrigation_mm','actual_nitrogen_kg_ha':'total_n_kg_ha','PFP_N_kg_kg':'PFP_N_kg_grain_per_kg_N'}
    keep=['null','recorded_farmer_template','official_extension_expert','rl_candidate']
    data=data[data.scenario.isin(keep)].rename(columns=mapping).copy();data['scenario']=data.scenario.replace({'rl_candidate':f'ppo_seed_{seed}'})
    import pandas as pd
    for column in ['grain_yield_kg_ha','total_irrigation_mm','total_n_kg_ha','WP_ET_kg_m3','PFP_N_kg_grain_per_kg_N']:
        data[column]=pd.to_numeric(data[column],errors='coerce')
    dest=OUT/'four_scenario_v3'/site/f'seed_{seed:02d}';dest.mkdir(parents=True,exist_ok=False)
    data.to_csv(dest/'yearly_metrics.csv',index=False)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    metrics=[('grain_yield_kg_ha','Yield (kg/ha)'),('total_irrigation_mm','Irrigation (mm)'),('total_n_kg_ha','N (kg/ha)'),('WP_ET_kg_m3','WP_ET (kg/m3)'),('PFP_N_kg_grain_per_kg_N','PFP_N (kg/kg)')]
    keep=keep[:-1]+[f'ppo_seed_{seed}'];names=['Null','Recorded','Expert',f'PPO seed {seed}'];years=list(range(2014,2024));colors=['#666666','#c44e52','#7e63b6','#2a9d55']
    cols=[m for m,_ in metrics]
    data.groupby('scenario')[cols].mean().to_csv(dest/'ten_year_means.csv')
    for metric,label in metrics:
        fig,ax=plt.subplots(figsize=(12,4.7));x=np.arange(10)
        for i,scenario in enumerate(keep):
            values=data[data.scenario.eq(scenario)].set_index('year').reindex(years)[metric].to_numpy(dtype=float)
            if metric=='PFP_N_kg_grain_per_kg_N':
                nitrogen=data[data.scenario.eq(scenario)].set_index('year').reindex(years).total_n_kg_ha.to_numpy()
                assert np.isfinite(values[nitrogen>0]).all()
                ax.text(.01,.01,'N=0: PFP_N undefined',transform=ax.transAxes,fontsize=8)
            else:assert np.isfinite(values).all(),(site,seed,metric)
            ax.bar(x+(i-1.5)*.2,values,.19,color=colors[i],label=names[i])
        ax.set_xticks(x,years);ax.set_ylabel(label);ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.13));ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(dest/f'yearly_{metric}.png',dpi=180);plt.close(fig)
    assert len(list((p/'figures').glob('*.png')))==12 and len(data)==40
    (dest/'audit.json').write_text(json.dumps({'site':site,'seed':seed,'rows':40,'figures':5,'replay_endpoint_and_ETCP_pass':True,'source':table.relative_to(ROOT).as_posix()},indent=2))
    print('PACKAGE_COMPLETE',site,seed,flush=True)
def summarize():
    import pandas as pd
    import numpy as np
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plot_sy_lc_seed_extension_231 as template
    all_pairs=[];report=[]
    for site in ['SY','LC']:
        paired=[];main=[];expert_rows=[]
        for seed in range(8):
            a=frame(runroot(site,seed)/'evaluation/032_22_checkpoint_validation_summary.csv')
            b=frame(control(site,seed)/'evaluation/032_22_checkpoint_validation_summary.csv')
            cols=['year','checkpoint_step','final_grnwt','total_irrigation','total_n','PFP_N']
            pair=a[cols].merge(b[cols],on=['year','checkpoint_step'],suffixes=('_augmented','_historical'),validate='one_to_one');pair['site']=site;pair['seed']=seed;paired.append(pair)
            aug=frame(OUT/'four_scenario_v3'/site/f'seed_{seed:02d}/yearly_metrics.csv');aug=aug[aug.scenario.eq(f'ppo_seed_{seed}')].copy();aug['seed']=seed
            _,baseline=template.source_tables(site)
            old_path=baseline if seed==0 else ROOT/f'results/sy_lc_8seed_extension_231/five_scenario_figures_v2/{site}/seed_{seed:02d}/tables/{template.PREFIX[site]}_five_scenario_season_summary.csv'
            old=template.normalize_summary(site,frame(old_path)) if site=='SY' else frame(old_path)
            old=old[old.scenario.eq('rl_candidate')][['year','WP_ET_kg_m3']]
            aug=aug.merge(old,on='year',suffixes=('_augmented','_historical'),validate='one_to_one');main.append(aug)
            four=frame(OUT/'four_scenario_v3'/site/f'seed_{seed:02d}/yearly_metrics.csv');expert=four[four.scenario.eq('official_extension_expert')]
            record={'site':site,'seed':seed,'validation_years':10}
            for metric in ['grain_yield_kg_ha','total_irrigation_mm','total_n_kg_ha','WP_ET_kg_m3','PFP_N_kg_grain_per_kg_N']:
                av=pd.to_numeric(aug['WP_ET_kg_m3_augmented' if metric=='WP_ET_kg_m3' else metric],errors='coerce').mean();ev=pd.to_numeric(expert[metric],errors='coerce').mean()
                record.update({f'ppo_mean_{metric}':av,f'expert_mean_{metric}':ev,f'delta_ppo_minus_expert_{metric}':av-ev})
            expert_rows.append(record)
        dest=OUT/'cohort_summary'/site;dest.mkdir(parents=True,exist_ok=False)
        pd.DataFrame(expert_rows).to_csv(dest/'eight_seed_comparison.csv',index=False)
        full=pd.concat(paired);assert len(full)==320;all_pairs.append(full)
        for metric in ['final_grnwt','total_irrigation','total_n','PFP_N']:
            for domain in ['augmented','historical']:full[metric+'_'+domain]=pd.to_numeric(full[metric+'_'+domain],errors='coerce')
        wp=pd.concat(main)[['year','seed','WP_ET_kg_m3_augmented','WP_ET_kg_m3_historical']]
        mainpairs=full[full.checkpoint_step.eq(100000)].merge(wp,on=['year','seed'],validate='one_to_one');assert len(mainpairs)==80
        metrics=['final_grnwt','total_irrigation','total_n','PFP_N','WP_ET_kg_m3']
        for metric in metrics:
            for domain in ['augmented','historical']:mainpairs[metric+'_'+domain]=pd.to_numeric(mainpairs[metric+'_'+domain],errors='coerce')
        for metric in metrics:mainpairs['delta_'+metric]=mainpairs[metric+'_augmented']-mainpairs[metric+'_historical']
        mainpairs.to_csv(dest/'augmented_vs_historical_100k_yearly.csv',index=False)
        columns=[m+'_'+d for m in metrics for d in ['augmented','historical']]+['delta_'+m for m in metrics]
        means=mainpairs.groupby('seed')[columns].mean();means.to_csv(dest/'augmented_vs_historical_100k_seed_means.csv');means.agg(['mean','std']).to_csv(dest/'augmented_vs_historical_100k_cohort_mean_sd.csv')
        for metric in metrics:
            fig,ax=plt.subplots(figsize=(9,4.5));x=np.arange(8)
            for i,domain in enumerate(['historical','augmented']):ax.bar(x+(i-.5)*.36,means[metric+'_'+domain],.35,label=domain,color=['#777777','#2a9d55'][i])
            ax.set_xticks(x,[f'seed{s}' for s in range(8)]);ax.set_ylabel(metric);ax.set_title(f'{site}: 100K, mean over 2014–2023');ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(dest/f'paired_seed_means_{metric}.png',dpi=180);plt.close(fig)
        full.groupby('checkpoint_step')[[m+'_'+d for m in ['final_grnwt','total_irrigation','total_n','PFP_N'] for d in ['historical','augmented']]].mean().to_csv(dest/'all_checkpoint_cohort_means.csv')
        record={'site':site,'seed_count':8,'years':10,'paired_checkpoint_rows':320,'100k_paired_rows':80,'delta_100k_seed_mean':{m:float(means['delta_'+m].mean()) for m in metrics}}
        report.append(record)
    pd.concat(all_pairs).to_csv(OUT/'all_checkpoint_paired_yearly.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2))
    text='# 060 SY/LC天气增强八种子结果\n\n沿用既有站点四情景主报告与SY/LC五情景逐日存档，全部seed0–7、2014–2023、四checkpoint保留，统一100K主比较。未增强控制逐字段合同一致；WP_ET仅100K由精确ETCP回放计算。\n\n'
    for r in report:text+=f"- {r['site']} 增强减未增强（八种子十年均值）："+json.dumps(r['delta_100k_seed_mean'],ensure_ascii=False)+'\n'
    text+='\n产量、灌溉、施氮、PFP_N与WP_ET必须共同解释；不以产量单项上升判定综合改善。跨站不作统一排名，seed为变异统计单位，std为样本SD。四情景不展示自动基线，完整五情景来源仍保留。未回放25/50/75K的WP_ET，不给近似值。\n'
    (OUT/'README.md').write_text(text,encoding='utf-8')
    doc=ROOT/'docs/sylc_external_weather_060_execution.md';doc.write_text(text,encoding='utf-8')
    images=sorted(list((OUT/'five_scenario_archive').rglob('*.png'))+list((OUT/'four_scenario_v3').rglob('yearly_*.png'))+list((OUT/'cohort_summary').rglob('*.png')));assert len(images)==282,len(images)
    html='<html><meta charset="utf-8"><title>SY LC 060</title><body><h1>SY LC 八种子天气增强结果</h1>'
    for p in images:html+=f'<h3>{p.relative_to(OUT)}</h3><img loading="lazy" style="max-width:100%" src="{p.relative_to(OUT).as_posix()}">'
    (OUT/'gallery.html').write_text(html+'</body></html>',encoding='utf-8')
    files=sorted(p for p in OUT.rglob('*') if p.is_file())+[doc,Path(__file__).resolve(),ROOT/'prompt_02/060_sylc_augmented_eight_seed_figures_and_comparison.md']
    pd.DataFrame([{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in files]).to_csv(OUT/'artifact_manifest.csv',index=False)
    backup=ROOT/'backups/sylc_060_reports';backup.mkdir(exist_ok=False)
    with zipfile.ZipFile(backup/'reports_060.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[OUT/'artifact_manifest.csv']:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(backup/'reports_060.zip') as z:assert all(hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).hexdigest()==sha(p) for p in files)
    (backup/'verification.json').write_text(json.dumps({'files':len(files)+1,'figures':len(images),'all_sha256_verified':True,'zip_sha256':sha(backup/'reports_060.zip')},indent=2))
    (OUT/'status.json').write_text(json.dumps({'status':'COMPLETE','seeds':16,'figures':len(images),'paired_rows':640,'main_100k_rows':160,'backup_verified':True},indent=2))
def main():
    import psutil
    OUT.mkdir(exist_ok=True);audit_contracts();completed=[]
    for site in ['SY','LC']:
        for seed in range(8):
            if (OUT/'four_scenario_v3'/site/f'seed_{seed:02d}'/'audit.json').exists():
                completed.append({'site':site,'seed':seed});continue
            logpath=OUT/f'{site}_seed{seed}.log'
            attempt=1
            while logpath.exists():
                attempt+=1;logpath=OUT/f'{site}_seed{seed}_attempt{attempt:03d}.log'
            with logpath.open('x') as log:
                p=subprocess.Popen([sys.executable,__file__,'--worker',site,'--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                start=time.monotonic();peak=0
                while p.poll() is None:
                    try:
                        parent=psutil.Process(p.pid);rss=sum(v.memory_info().rss for v in [parent]+parent.children(recursive=True) if v.is_running())/1024**2;peak=max(peak,rss)
                        if rss>1536 or time.monotonic()-start>900:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);break
                    except psutil.NoSuchProcess:pass
                    (OUT/'status.json').write_text(json.dumps({'status':'RUNNING','site':site,'seed':seed,'completed':completed,'peak_rss_mb':peak},indent=2));time.sleep(2)
            resource={'exit_code':p.returncode,'seconds':time.monotonic()-start,'peak_rss_mb':peak};(OUT/f'{site}_seed{seed}_resource.json').write_text(json.dumps(resource,indent=2))
            if p.returncode:
                (OUT/'status.json').write_text(json.dumps({'status':'STOPPED_FAILURE','site':site,'seed':seed,'resource':resource},indent=2));raise RuntimeError(f'{site} seed{seed} report failed')
            completed.append({'site':site,'seed':seed})
    summarize()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['SY','LC']);p.add_argument('--seed',type=int);a=p.parse_args()
    if a.worker:worker(a.worker,a.seed)
    else:main()
