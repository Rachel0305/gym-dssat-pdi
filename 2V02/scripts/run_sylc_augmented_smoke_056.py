"""One-environment weather injection smoke, frozen station PPO contracts."""
import os
os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
from pathlib import Path
import sys,json,csv,hashlib,calendar,shutil,argparse,time,subprocess
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/sylc_external_weather_056/attempt005'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/053_lca_lowIC_site_transfer')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def worker(site):
    import pandas as pd
    import numpy as np
    import ppo_safe_rendering as rendering
    import gym_dssat_pdi.envs.dssat_pdi as module
    import run_sya_originIC_expanded_action_maskableppo_046_10 as sy
    import run_053_00_lca_lowIC_expanded_action_maskableppo as lc
    runner=sy if site=='SY' else lc; engine=runner.engine.base03222
    directory=OUT/site;directory.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(runner.DEFAULT_CONFIG.read_text());cfg['task_id']='056e'+site;cfg['task_name']='external_weather_seed0_smoke';cfg['seed']=0
    path=directory/'config.json';path.write_text(json.dumps(cfg,indent=2))
    expected={};manifest=[];checked=0;training_checked=0
    original_copy=rendering.copy_weather_to_rendered_dir
    counter=0
    def inject(station,year,template,config):
        nonlocal counter
        if 'train' not in str(template):return original_copy(station,year,template,config)
        counter+=1
        leap=calendar.isleap(year)
        seed=1001 if leap else 1001+(counter%2)
        tag='leap2008_seed1001' if leap else ('seed1001_a' if site=='SY' and seed==1001 else 'seed1002' if site=='SY' else 'attempt003_seed1001_a' if seed==1001 else 'attempt003_seed1002')
        source=ROOT/f'results/sylc_external_weather_055/{site}/runtime/{tag}/weather.csv'
        weather=pd.read_csv(source);assert len(weather)==(366 if leap else 365)
        weather['DATE']=pd.date_range(f'{year}-01-01',f'{year}-12-31')
        prior=rendering.source_weather_path(station,year)
        header=prior.read_text().splitlines()[:5]
        dest=template.parent/f'{rendering.target_weather_stem(station,year)}.WTH'
        lines=header+[f"{d.strftime('%Y%j')} {r.SRAD:5.1f} {r.TMAX:5.1f} {r.TMIN:5.1f} {r.RAIN:5.1f}" for d,r in zip(weather.DATE,weather.itertuples())]
        assert not dest.exists()
        dest.write_text('\n'.join(lines)+'\n')
        record={'episode_environment':counter,'year':year,'weather_seed':seed,'source_csv':str(source.relative_to(ROOT)),'source_sha256':sha(source),'WTH':str(dest.relative_to(ROOT)),'WTH_sha256':sha(dest),'days':len(weather)}
        manifest.append(record)
        expected[sha(dest)]={int(d.strftime('%Y%j')):{v:round(float(getattr(r,v)),1) for v in ['RAIN','SRAD','TMAX','TMIN']} for d,r in zip(weather.DATE,weather.itertuples())}
        (directory/'weather_manifest.json').write_text(json.dumps(manifest,indent=2))
        return dest
    rendering.copy_weather_to_rendered_dir=inject
    cls=next(v for v in vars(module).values() if isinstance(v,type) and v.__module__==module.__name__ and hasattr(v,'_get_state'))
    old_state=cls._get_state
    import zmq
    old_sockets=cls._get_sockets_
    old_close_client=cls._close_client
    def sockets(instance):
        result=old_sockets(instance)
        instance._server.setsockopt(zmq.RCVTIMEO,30000)
        instance._server.setsockopt(zmq.SNDTIMEO,30000)
        instance._server.setsockopt(zmq.LINGER,0)
        return result
    def close_client(instance):
        try:return old_close_client(instance)
        except zmq.Again:
            import psutil
            pid=instance._client_process_pid
            if pid and psutil.pid_exists(pid):raise
            print('Closed disconnected completed DSSAT client',flush=True)
    cls._get_sockets_=sockets;cls._close_client=close_client
    cache={};date_offsets={}
    def state(instance):
        nonlocal checked,training_checked
        value=old_state(instance)
        folder=instance._tmp_folder
        if folder not in cache:
            files=list(Path(folder).glob('CN'+site+'*.WTH'))
            cache[folder]=expected.get(sha(files[0])) if files else None
        data=cache[folder]
        if data and isinstance(value[1],dict) and 'yrdoy' in value[1]:
            import datetime
            s=value[1];token=int(s['yrdoy'])
            day=datetime.datetime.strptime(str(token),'%Y%j')
            candidates=date_offsets.get(folder,{0,-1})
            matched=set()
            for offset in candidates:
                target=int((day+datetime.timedelta(days=offset)).strftime('%Y%j'))
                if target in data and all(abs(float(s[v.lower()])-data[target][v])<=1e-4 for v in ['RAIN','SRAD','TMAX','TMIN']):matched.add(offset)
            if not matched:raise RuntimeError(f'Weather injection mismatch {token}; consistent date offsets {candidates}')
            date_offsets[folder]=matched
            checked+=1;training_checked+=1
        return value
    cls._get_state=state
    old_make=engine.base.make_env
    factory_counter=0
    def make_env(config,env_config,station,year,seed,run_tag,evaluation=False):
        nonlocal factory_counter
        if not evaluation:
            factory_counter+=1
            run_tag+=f'_aug_episode_{factory_counter:04d}'
        return old_make(config,env_config,station,year,seed,run_tag,evaluation=evaluation)
    engine.base.make_env=make_env
    old_reset=engine.RandomYearEnv.reset
    def reset(instance,*args,**kwargs):
        for env in instance.envs.values():env.close()
        instance.envs.clear()
        return old_reset(instance,*args,**kwargs)
    engine.RandomYearEnv.reset=reset
    engine.SEED=0
    try:
        result=runner.run(path,dry_run=False,smoke=True,formal=False)
        seeds=set(r['weather_seed'] for r in manifest)
        gate={'site':site,'PPO_seed':0,'requested_steps':2000,'training_weather_states_matched':checked,'weather_seeds':sorted(seeds),'weather_environment_count':len(manifest),'runtime_weather_date_offsets':{k:sorted(v) for k,v in date_offsets.items()},'injection_pass':checked>2000 and len(seeds)>=2,'result':result,'formal_training_allowed':False}
        (directory/'gate.json').write_text(json.dumps(gate,indent=2))
        assert gate['injection_pass']
        print(json.dumps({'site':site,'injection_pass':True,'states':checked,'weather_seeds':sorted(seeds)},indent=2),flush=True)
    finally:
        cls._get_state=old_state;rendering.copy_weather_to_rendered_dir=original_copy;engine.RandomYearEnv.reset=old_reset;engine.base.make_env=old_make;cls._get_sockets_=old_sockets;cls._close_client=old_close_client
def main():
    import psutil
    OUT.mkdir(exist_ok=False)
    for site in ['SY','LC']:
        with (OUT/f'{site}.log').open('x') as log:
            p=subprocess.Popen([sys.executable,__file__,'--worker',site],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            start=time.monotonic();peak=0;failure=None
            while p.poll() is None:
                try:
                    parent=psutil.Process(p.pid);rss=sum(x.memory_info().rss for x in [parent]+parent.children(recursive=True) if x.is_running())/1024**2
                    peak=max(peak,rss)
                    if rss>1536 or time.monotonic()-start>600:
                        failure='RESOURCE_LIMIT';import signal;os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);break
                except psutil.NoSuchProcess:pass
                time.sleep(2)
        status={'exit_code':p.returncode,'peak_tree_rss_mb':peak,'seconds':time.monotonic()-start,'failure':failure}
        (OUT/f'{site}_process.json').write_text(json.dumps(status,indent=2))
        if p.returncode or failure:raise RuntimeError(f'{site} stopped; inspect log')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=['SY','LC']);args=parser.parse_args()
    if args.worker:worker(args.worker)
    else:main()
