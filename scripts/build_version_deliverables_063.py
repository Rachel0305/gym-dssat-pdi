"""Create consistent per-seed PNG/table packages and a verified version index."""
import os
os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']})
from pathlib import Path
import sys,json,hashlib,csv,argparse
import pandas as pd,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/053_lca_lowIC_site_transfer')]
import plot_sy_lc_seed_extension_231 as sylc
BASES=['null','recorded_farmer_template','official_extension_expert']
METRICS=['grain_yield_kg_ha','total_irrigation_mm','total_n_kg_ha','PFP_N','WP_ET_kg_m3']
LABELS=['Yield (kg/ha)','Irrigation (mm)','N (kg/ha)','PFP_N (kg/kg)','WP_ET (kg/m3)']
def read(p):return pd.read_csv(p,keep_default_na=False)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def normalize(f):
    f=f.rename(columns={'final_grain_kg_ha':'grain_yield_kg_ha','irrigation_mm':'total_irrigation_mm','nitrogen_kg_ha':'total_n_kg_ha','actual_irrigation_mm':'total_irrigation_mm','actual_nitrogen_kg_ha':'total_n_kg_ha','PFP_N_kg_kg':'PFP_N','PFP_N_kg_grain_per_kg_N':'PFP_N'})
    for c in METRICS+['etcp_mm']:
        if c not in f:f[c]=np.nan
        f[c]=pd.to_numeric(f[c],errors='coerce')
    return f
def source_table(site,regime,seed):
    if site in ['SY','LC']:
        if regime=='augmented':return ROOT/f'results/sylc_external_weather_060/four_scenario_v3/{site}/seed_{seed:02d}/yearly_metrics.csv'
        return sylc.source_tables(site)[1] if seed==0 else ROOT/f'results/sy_lc_8seed_extension_231/five_scenario_figures_v2/{site}/seed_{seed:02d}/tables/{sylc.PREFIX[site]}_five_scenario_season_summary.csv'
    if regime=='historical':return ROOT/f'results/hl_fq_8seed_cross_site/055_03_five_scenario/{site}/best_seed_seed{seed}/tables/{site.lower()}_seed{seed}_yearly_five_scenario_metrics.csv'
    root='hla_wgen_8seed_053/055_03_five_scenario/HL' if site=='HL' else 'fqa_multiyear_wgen_ppo_052_8seed/055_03_five_scenario/FQ'
    return ROOT/f'results/{root}/best_seed_seed{seed}/tables/{site.lower()}_seed{seed}_yearly_five_scenario_metrics.csv'
def main(version):
    target=Path('\\\\?\\'+str(ROOT/version)) if os.name=='nt' else ROOT/version
    assert (target/'COPY_VERIFICATION.json').exists()
    dest=target/'deliverables';dest.mkdir(exist_ok=False)
    yc_eval=read(ROOT/'results/yc_random_weather_ppo/004_05/all_evaluation_episode_level_0_7.csv')
    yc_models=read(ROOT/'results/yc_random_weather_ppo/004_05/all_model_manifest_0_7.csv')
    yc_metrics=read(ROOT/('results/version2_yc_wp_et_063_attempt002/yc_16_model_160_year_exact_metrics.csv' if version=='2V02' else 'results/yc_random_weather_ppo/004_07_02_seed345_etcp_replay/replay_episode_summary.csv'))
    registry=[];tables=[];coverage=[]
    for site in ['SY','LC','HL','FQ','YC']:
        for regime in ['historical','augmented']:
            for seed in range(8):
                if site=='YC':
                    g='HISTORICAL_WEATHER' if regime=='historical' else 'RANDOM_WEATHER_WGEN'
                    e=yc_eval[(yc_eval.training_regime==g)&yc_eval.ppo_seed.eq(seed)&yc_eval.evaluation_weather_type.eq('observed_weather')].copy()
                    assert len(e)==10
                    p=pd.DataFrame({'year':pd.to_numeric(e.evaluation_weather_year).astype(int),'grain_yield_kg_ha':pd.to_numeric(e['yield']),'total_irrigation_mm':pd.to_numeric(e.total_irrigation),'total_n_kg_ha':pd.to_numeric(e.total_fertilizer),'scenario':f'ppo_seed_{seed}'})
                    p['PFP_N']=p.grain_yield_kg_ha/p.total_n_kg_ha.replace(0,np.nan);p['WP_ET_kg_m3']=np.nan;p['etcp_mm']=np.nan
                    if version=='2V02':ex=yc_metrics[(yc_metrics.training_regime==g)&yc_metrics.seed.eq(seed)]
                    else:ex=yc_metrics[yc_metrics.seed.eq(seed)] if regime=='augmented' and seed in [3,4,5] else yc_metrics.iloc[:0]
                    for row in ex.itertuples():
                        p.loc[p.year.eq(row.year),['WP_ET_kg_m3','etcp_mm']]=[row.wp_et_kg_m3,row.etcp_mm]
                    baseline=normalize(read(ROOT/'benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/055_03_yca_five_scenario_season_summary.csv'))
                    f=pd.concat([baseline[baseline.scenario.isin(BASES)],p],ignore_index=True)
                    mr=yc_models[(yc_models.training_regime==g)&yc_models.ppo_seed.eq(seed)].iloc[0];model_path=mr.model_path
                    source='results/yc_random_weather_ppo/004_05/all_evaluation_episode_level_0_7.csv'
                else:
                    path=source_table(site,regime,seed);source=path.relative_to(ROOT).as_posix();f=normalize(read(path));f=f[f.scenario.isin(BASES+['rl_candidate',f'ppo_seed_{seed}'])].copy();f.loc[f.scenario.eq('rl_candidate'),'scenario']=f'ppo_seed_{seed}'
                    if site=='FQ' and version=='2V02':
                        baseline=normalize(read(ROOT/'benchmark_results/088_00_fq2018_input_quality_correction_run_corrected2018/evaluation/051_02_baseline_summary.csv'))
                        for scenario in BASES:
                            row=baseline[(baseline.year==2018)&baseline.scenario.eq(scenario)].iloc[0]
                            for col in METRICS+['etcp_mm']:f.loc[f.year.eq(2018)&f.scenario.eq(scenario),col]=row[col]
                        if regime=='historical':
                            base=ROOT/'results/hl_fq_8seed_cross_site/055_03_native_auto_corrected2018/replays/FQ' if seed<7 else ROOT/'results/version2_fq_seed7_corrected2018_063'
                            d=json.loads((base/f'seed{seed}/corrected2018_ppo_metrics.json').read_text())
                            vals=[d[k] for k in ['corrected_yield_kg_ha','corrected_irrigation_mm','corrected_n_kg_ha','corrected_PFP_N_kg_kg','corrected_WP_ET_kg_m3','corrected_etcp_mm']]
                        else:
                            row=normalize(read(ROOT/f'results/fqa_wgen_052_corrected2018_231/seed_{seed:02d}/corrected2018_metrics.csv')).iloc[0];vals=[row[k] for k in METRICS+['etcp_mm']]
                        f.loc[f.year.eq(2018)&f.scenario.eq(f'ppo_seed_{seed}'),METRICS+['etcp_mm']]=vals
                    # Registry resolves actual trained checkpoints, preserving model bytes.
                    if site in ['SY','LC']:
                        if regime=='augmented':run=ROOT/f'benchmark_results/059{site}s{seed}_frozen_weather_augmented_100k'
                        elif seed==0:run=ROOT/('benchmark_results/046_10_sya_originIC_expanded_action_maskableppo' if site=='SY' else 'benchmark_results/053_00_lca_lowIC_expanded_action_maskableppo')
                        else:
                            c=json.loads((ROOT/f'results/sy_lc_8seed_extension_231/configs/{site.lower()}_seed{seed}.json').read_text());run=ROOT/'benchmark_results'/f"{c['task_id']}_{c['task_name']}"
                        inv=read(run/'evaluation/032_22_training_checkpoint_inventory.csv');model_path=inv[inv.checkpoint_step.eq(100000)].iloc[0].model_path
                    elif regime=='historical':
                        if site=='FQ' and seed==7:run=ROOT/'benchmark_results/20261012fq7_fq_seed7_historical_completion'
                        elif seed==0:run=ROOT/('benchmark_results/054_00_hla_lowIC_expanded_action_maskableppo' if site=='HL' else 'benchmark_results/051_00_fqa_originIC_expanded_action_maskableppo')
                        else:
                            c=json.loads((ROOT/f'configs/hl_fq_8seed_cross_site/{site.lower()}_seed{seed}.json').read_text());run=ROOT/'benchmark_results'/f"{c['task_id']}_{c['task_name']}"
                        inv=read(run/'evaluation/032_22_training_checkpoint_inventory.csv');model_path=inv[inv.checkpoint_step.eq(100000)].iloc[0].model_path
                    else:
                        root=ROOT/('results/hla_wgen_8seed_053' if site=='HL' else 'results/fqa_multiyear_wgen_ppo_052_8seed')
                        if site=='FQ' and seed==0:root=ROOT/'results/fqa_multiyear_wgen_ppo_051'
                        matches=list(root.glob(('attempt_01' if site=='FQ' and seed==0 else f'seed_{seed:02d}/attempt_01')+'/**/checkpoint_100000.zip'))
                        if not matches:matches=list(root.glob(('attempt_01' if site=='FQ' and seed==0 else f'seed_{seed:02d}/attempt_01')+'/**/*100000*.zip'))
                        assert len(matches)==1,(site,seed,matches);model_path=matches[0].relative_to(ROOT).as_posix()
                f=normalize(f);f=f[['year','scenario']+METRICS+['etcp_mm']].sort_values(['year','scenario']);assert len(f)==40
                assert set(zip(f.year,f.scenario))=={(y,s) for y in range(2014,2024) for s in BASES+[f'ppo_seed_{seed}']}
                folder=dest/site/regime/f'seed_{seed:02d}';figures=folder/'figures';tab=folder/'tables';figures.mkdir(parents=True);tab.mkdir()
                f.to_csv(tab/'yearly_metrics.csv',index=False);f.groupby('scenario')[METRICS].mean().to_csv(tab/'ten_year_means.csv')
                p=f[f.scenario.eq(f'ppo_seed_{seed}')].copy();p.insert(0,'seed',seed);p.insert(0,'regime',regime);p.insert(0,'site',site);tables.append(p)
                coverage.append({'site':site,'regime':regime,'seed':seed,'WP_ET_years':int(p.WP_ET_kg_m3.notna().sum())})
                for metric,label in zip(METRICS,LABELS):
                    fig,ax=plt.subplots(figsize=(12,4.7));x=np.arange(10)
                    for k,scenario in enumerate(BASES+[f'ppo_seed_{seed}']):
                        vals=f[f.scenario.eq(scenario)].set_index('year').reindex(range(2014,2024))[metric].to_numpy(dtype=float)
                        ax.bar(x+(k-1.5)*.2,vals,.19,color=['#666666','#c44e52','#7e63b6','#2a9d55'][k],label=['Null','Recorded','Expert',f'PPO seed {seed}'][k])
                    ax.set_xticks(x,range(2014,2024));ax.set_ylabel(label);ax.set_title(f'{version} {site} {regime} | 100K');ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.18));ax.grid(axis='y',alpha=.2)
                    if metric=='PFP_N':ax.text(.01,.01,'N=0: PFP_N undefined',transform=ax.transAxes,fontsize=8)
                    if metric=='WP_ET_kg_m3' and p.WP_ET_kg_m3.isna().any():ax.text(.01,.01,'PPO exact WP_ET unavailable in original release',transform=ax.transAxes,fontsize=8)
                    fig.tight_layout();fig.savefig(figures/f'yearly_{metric}.png',dpi=180);plt.close(fig)
                assert (target/model_path).exists(),model_path
                registry.append({'site':site,'regime':regime,'seed':seed,'checkpoint':100000,'model_path':model_path,'model_sha256':sha(target/model_path),'original_metrics_source':source,'primary_table':(tab/'yearly_metrics.csv').relative_to(target).as_posix(),'WP_ET_years':int(p.WP_ET_kg_m3.notna().sum())})
                print(version,site,regime,seed,'COMPLETE',flush=True)
    allmetrics=pd.concat(tables,ignore_index=True);assert len(allmetrics)==800
    if version=='2V02':assert allmetrics.WP_ET_kg_m3.notna().all()
    allmetrics.to_csv(dest/'all_80_policy_800_year_metrics.csv',index=False);pd.DataFrame(registry).to_csv(target/'MODEL_REGISTRY.csv',index=False);pd.DataFrame(coverage).to_csv(target/'METRIC_COVERAGE.csv',index=False)
    seedmeans=allmetrics.groupby(['site','regime','seed'])[METRICS].mean();seedmeans.to_csv(dest/'seed_means.csv');seedmeans.groupby(['site','regime']).agg(['mean','std']).to_csv(dest/'cohort_mean_sd.csv')
    # No automatic-baseline conflation: original five-scenario archives remain provenance.
    text=f'# {version} 五站双域八种子项目\n\n本版本固定80个100K策略与2014–2023共800个年度端点。代码在src/scripts，配置在configs，源输入在DSSAT_auto_validation/data，模型与验证在原相对路径benchmark_results/results。主报告统一四情景：Null、Recorded、Expert、PPO。\n\n'
    text+=('原版：FQ保留原2018天气及零产量结果；YC保留当时精确WP_ET的缺项，缺失不填0。' if version=='1V01' else '修订版：FQ所有16个策略的2018端点改为修正TMIN天气下的确定性回放；YC全部16个策略十年精确ETCP/WP_ET已核验。模型和训练历史没有改变。')
    text+='\n\n## 主结果入口\n\n'
    for site in ['SY','LC','HL','FQ','YC']:text+=f'- [{site} 未增强](deliverables/{site}/historical/) / [{site} 增强](deliverables/{site}/augmented/)；seed_00–seed_07各有figures与tables。\n'
    text+='\n每策略5张独立PNG，共400张统一主图；此前逐日过程图及事件表保留在results中的原目录。MODEL_REGISTRY.csv索引80个真实模型；METRIC_COVERAGE.csv记录指标可用性；deliverables/all_80_policy_800_year_metrics.csv为主端点表。五情景旧图属于历史存档，2V02比较时优先用deliverables，避免误读旧FQ2018图。\n\n## 复现\n\n在本文件夹作为项目根目录运行，先用Python执行restore_large_files.py恢复字节分块的大表；文件恢复后hash自动核验。项目代码需要已有gym-DSSAT/PPO环境与DSSAT4.8运行时（例如原容器/opt/gym_dssat_pdi/bin/python和/opt/dssat_pdi/run_dssat）；本归档不复制系统级运行时或大缓存。代码保留原路径，模型保持原sha256。SOURCE_MANIFEST.csv记录复制来源，RELEASE_MANIFEST.csv为最终归档逐文件hash。\n\n主对照固定100K，不做逐年择优。原25/50/75K训练验证保留；V02的FQ输入修订与YC效率补齐仅针对100K主端点，不把其他checkpoint描述成已做相同修订。各站增强算法和天气库不同，不做混站排名，探索性筛选不替代八seed总体证据。\n'
    (target/'README.md').write_text(text,encoding='utf-8')
    (target/'.gitattributes').write_text('* -text\n')
    chunkmeta=json.loads((target/'large_file_chunks.json').read_text());(target/'.gitignore').write_text('__pycache__/\n*.pyc\n'+''.join('/'+x['path']+'\n' for x in chunkmeta))
    # Include the release tools created after the initial source snapshot.
    for p in ROOT.glob('scripts/*063.py'):
        dst=target/'scripts'/p.name
        if dst.exists() and sha(dst)!=sha(p):
            backup=target/'source_build_history'/p.name;backup.parent.mkdir(exist_ok=True);dst.rename(backup)
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():dst.write_bytes(p.read_bytes())
    records=[]
    for p in sorted(target.rglob('*')):
        if p.is_file() and p.name not in ['RELEASE_MANIFEST.csv','RELEASE_QA.json']:records.append({'path':p.relative_to(target).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    pd.DataFrame(records).to_csv(target/'RELEASE_MANIFEST.csv',index=False)
    (target/'RELEASE_QA.json').write_text(json.dumps({'status':'PASS','policies':80,'endpoint_rows':800,'primary_pngs':400,'models_sha256_verified':80,'files':len(records),'bytes':sum(x['bytes'] for x in records),'YC_complete_WP_ET':version=='2V02','FQ2018_corrected':version=='2V02'},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('version',choices=['1V01','2V02']);a=p.parse_args();main(a.version)
