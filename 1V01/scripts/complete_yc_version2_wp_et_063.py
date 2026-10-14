"""Exact YC endpoint replay for both frozen 8-seed regimes; no training."""
import os
os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']})
from pathlib import Path
import sys,json,csv,inspect,time,subprocess,signal,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'results/version2_yc_wp_et_063_attempt002'
def worker(group,seed):
    import pandas as pd
    import run_004_07_02_yc_seed345_etcp_replay as replay
    regime='HISTORICAL_WEATHER' if group=='historical' else 'RANDOM_WEATHER_WGEN'
    replay.REGIME=regime;replay.RSS_LIMIT_MB=1536;replay.TASK_ROOT=OUT/group/f'seed{seed}';replay.TASK_ROOT.mkdir(parents=True)
    manifest=pd.read_csv(ROOT/'results/yc_random_weather_ppo/004_05/all_model_manifest_0_7.csv')
    modelrow=manifest[(manifest.training_regime==regime)&(manifest.ppo_seed==seed)].iloc[0]
    assert hashlib.sha256((ROOT/modelrow.model_path).read_bytes()).hexdigest().lower()==modelrow.model_sha256.lower()
    ev=(ROOT/'results/yc_random_weather_ppo/004_03/evaluation/verified_weather_refresh/observed_weather' if seed<=2 else ROOT/'results/yc_random_weather_ppo/004_05/evaluation/runs/observed_weather')/group/f'ppo_seed_{seed}'
    replay.EVAL_ROOT=ev.parent
    steps=pd.read_csv(ROOT/'results/yc_random_weather_ppo/004_05/evaluation_step_level_all_models.csv')
    steps=steps[(steps.training_regime==regime)&(steps.ppo_seed==seed)&steps.evaluation_weather_type.eq('observed_weather')]
    assert len(steps.evaluation_weather_year.unique())==10
    stepfile=replay.TASK_ROOT/'frozen_formal_step_rows.csv';steps.to_csv(stepfile,index=False)
    def paths(s,y):
        tag=f'eval_observed_weather_{regime}_{s}_{y}'
        return ROOT/modelrow.model_path,ev/'evaluation_episode_level.csv',stepfile,ev/'rendered_inputs/YCA'/str(y)/tag
    replay.frozen_paths=paths
    for name in ['load_expected_episode','check_rendered_inputs','run_one']:
        code=inspect.getsource(getattr(replay,name)).replace('RANDOM_WEATHER_WGEN','{REGIME}')
        exec(compile(code,__file__,'exec'),replay.__dict__)
    helper=replay.load_summary_tools()
    # Existing exact random-weather seed3/4/5 results are model-hash compatible.
    if group=='random_weather' and seed in [3,4,5]:
        f=pd.read_csv(ROOT/'results/yc_random_weather_ppo/004_07_02_seed345_etcp_replay/replay_episode_summary.csv')
        f=f[f.seed.eq(seed)].copy();assert len(f)==10
        assert f.model_sha256.str.lower().eq(str(modelrow.model_sha256).lower()).all()
        f['training_regime']=regime;f['replay_source']='reused_exact_004_07_02'
        f.to_csv(replay.TASK_ROOT/'exact_metrics.csv',index=False);return
    pilot=replay.load_runtime();rows=[]
    for year in [2019]+[y for y in range(2014,2024) if y!=2019]:
        row=replay.run_one(pilot,helper,seed,year,'smoke' if year==2019 else 'formal_replay')
        row.update(training_regime=regime,replay_source='exact_replay_063')
        rows.append(row);pd.DataFrame(rows).to_csv(replay.TASK_ROOT/'exact_metrics.csv',index=False)
    assert len(rows)==10
def main():
    import psutil,pandas as pd
    OUT.mkdir(exist_ok=False)
    for group in ['historical','random_weather']:
        for seed in range(8):
            log=OUT/f'{group}_seed{seed}.log'
            with log.open('x') as f:
                p=subprocess.Popen([sys.executable,__file__,'--group',group,'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
                start=time.monotonic();peak=0
                while p.poll() is None:
                    parent=psutil.Process(p.pid)
                    rss=sum(x.memory_info().rss for x in [parent]+parent.children(recursive=True) if x.is_running())/1024**2;peak=max(peak,rss)
                    if rss>1536 or time.monotonic()-start>600:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);break
                    time.sleep(2)
            (OUT/f'{group}_seed{seed}_resource.json').write_text(json.dumps({'exit_code':p.returncode,'seconds':time.monotonic()-start,'peak_rss_mb':peak},indent=2))
            if p.returncode:raise RuntimeError(f'{group} seed{seed} failed; inspect retained log')
    frames=[pd.read_csv(p) for p in OUT.glob('*/seed*/exact_metrics.csv')]
    f=pd.concat(frames,ignore_index=True);assert len(f)==160
    assert not f.duplicated(['training_regime','seed','year']).any()
    assert f.etcp_mm.gt(0).all() and f.wp_et_kg_m3.notna().all()
    f.to_csv(OUT/'yc_16_model_160_year_exact_metrics.csv',index=False)
    (OUT/'gate.json').write_text(json.dumps({'status':'PASS','models':16,'rows':160,'new_training':False,'exact_ETCP_available':True},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group');p.add_argument('--seed',type=int);a=p.parse_args()
    if a.group:worker(a.group,a.seed)
    else:main()
