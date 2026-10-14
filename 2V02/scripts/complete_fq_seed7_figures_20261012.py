"""Append seed7 figures using frozen baseline and plotting code; exact ETCP replay."""
import os
os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
from pathlib import Path
import sys,json,hashlib,shutil,zipfile
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hl_fq_8seed_cross_site import plot_hl_fq_five_scenario_05503_style as plot
from hl_fq_8seed_cross_site import replay_wp_et_05503 as replay
from hl_fq_8seed_cross_site import finalize_05503_yearly_figures as final
CFG=ROOT/'results/fq_seed7_completion_20261012/fq_seed7.json'
PPO=ROOT/'benchmark_results/20261012fq7_fq_seed7_historical_completion'
DEST=plot.OUT_ROOT/'FQ/best_seed_seed7'
def main():
    if DEST.exists():raise FileExistsError(DEST)
    run=replay.load_runner('FQ'); cfg=json.loads(CFG.read_text())
    expected=replay.selected_summary(PPO,'FQA',7)
    work=plot.OUT_ROOT/'wp_et_replay/FQ/seed7'
    work.mkdir(parents=True,exist_ok=True)
    for year in plot.YEARS:
        metric=work/f'{year}_wp_et.json'
        if metric.exists():continue
        snapshot=run.replay_ppo_snapshot(cfg,100000,year,work,PPO)
        daily=run.daily_from_snapshot(snapshot,year,'rl_candidate')
        y=float(daily.grain_yield_kg_ha.iloc[-1]); measured=run.baseline.metrics_from_snapshot(snapshot,y)
        saved=expected[year]
        assert abs(y-float(saved['final_grnwt']))<=1
        assert abs(measured['actual_irrigation_mm']-float(saved['total_irrigation']))<=.51
        assert abs(measured['actual_nitrogen_kg_ha']-float(saved['total_n']))<=.51
        record={'site':'FQ','seed':7,'year':year,'saved_yield_kg_ha':float(saved['final_grnwt']),'replay_yield_kg_ha':y,'saved_irrigation_mm':float(saved['total_irrigation']),'replay_irrigation_mm':measured['actual_irrigation_mm'],'saved_n_kg_ha':float(saved['total_n']),'replay_n_kg_ha':measured['actual_nitrogen_kg_ha'],'etcp_mm':measured['etcp_mm'],'WP_ET_kg_m3':measured['WP_ET_kg_m3'],'snapshot_path':snapshot.relative_to(ROOT).as_posix(),'metric_source':'Summary.OUT ETCP from deterministic 100K checkpoint replay'}
        metric.write_text(json.dumps(record,indent=2))
        print('REPLAY',year,flush=True)
    plot.config_and_output=lambda site,seed:(CFG,PPO)
    summary,daily_base=plot.load_baseline('FQ')
    metrics,daily,actions,audit=plot.load_ppo_seed('FQ',7,summary,daily_base)
    assert len(metrics)==50
    figs=DEST/'figures'; tables=DEST/'tables'; figs.mkdir(parents=True); tables.mkdir()
    plot.write_table_files('FQ',7,metrics,tables,audit)
    plot.write_event_and_daily_tables('FQ',7,daily,actions,tables)
    plot.plot_metric_bars('FQ',7,metrics,figs)
    plot.plot_management_bars('FQ',7,metrics,figs)
    plot.plot_reward('FQ',7,metrics,figs)
    for year in plot.YEARS:
        plot.plot_daily_year('FQ',7,year,daily,figs,bool(audit['weather_input_match_by_calendar_date'][str(year)]['weather_values_match_by_calendar_date']))
    plot.write_seed_readme('FQ',7,metrics,audit,DEST)
    final.SEEDS_BY_SITE={'FQ':[7]}
    final.main()
    assert len(list(figs.glob('*.png')))==23
    combined=[]
    for seed in range(8):
        path=plot.OUT_ROOT/f'FQ/best_seed_seed{seed}/tables/fq_seed{seed}_yearly_five_scenario_metrics.csv'
        f=pd.read_csv(path,keep_default_na=False)
        f=f[f.scenario.eq(f'ppo_seed_{seed}')].copy(); f['seed']=seed; combined.append(f)
    all_seeds=pd.concat(combined,ignore_index=True)
    assert len(all_seeds)==80
    all_seeds.to_csv(DEST/'fq_8seed_100k_yearly_metrics.csv',index=False)
    cols=['grain_yield_kg_ha','total_irrigation_mm','total_n_kg_ha','WP_ET_kg_m3','PFP_N_kg_grain_per_kg_N']
    for c in cols:all_seeds[c]=pd.to_numeric(all_seeds[c],errors='coerce')
    means=all_seeds.groupby('seed')[cols].mean()
    means.to_csv(DEST/'fq_8seed_100k_seed_means.csv')
    means.agg(['mean','std']).to_csv(DEST/'fq_8seed_100k_cohort_summary.csv')
    backup=ROOT/'backups/fq_seed7_figures_20261012'; backup.mkdir(exist_ok=False)
    paths=list(DEST.rglob('*'))+list(work.rglob('*'))+list((ROOT/'results/fq_seed7_completion_20261012').rglob('*'))+list(PPO.rglob('*'))+[Path(__file__).resolve()]
    files=sorted({p for p in paths if p.is_file() and '__pycache__' not in p.parts})
    manifest=[]
    with zipfile.ZipFile(backup/'fq_seed7_complete.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in files:
            rel=p.relative_to(ROOT).as_posix(); digest=hashlib.sha256(p.read_bytes()).hexdigest()
            z.write(p,rel); manifest.append({'path':rel,'bytes':p.stat().st_size,'sha256':digest})
    pd.DataFrame(manifest).to_csv(backup/'manifest.csv',index=False)
    with zipfile.ZipFile(backup/'fq_seed7_complete.zip') as z:
        assert z.testzip() is None
        for r in manifest:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
    (backup/'verification.json').write_text(json.dumps({'files':len(files),'figures':23,'five_scenario_metric_rows':50,'eight_seed_rows':80,'all_archive_hashes_match':True},indent=2))
    print('COMPLETE',str(backup),flush=True)
if __name__=='__main__':main()
