"""One-environment weather injection smoke, frozen station PPO contracts."""
import os
os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
from pathlib import Path
import sys,json,csv,hashlib,calendar,shutil,argparse,time,subprocess
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/sylc_external_weather_059'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/053_lca_lowIC_site_transfer')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def worker(site,ppo_seed):
    import pandas as pd
    import numpy as np
    import ppo_safe_rendering as rendering
    import gym_dssat_pdi.envs.dssat_pdi as module
    import run_sya_originIC_expanded_action_maskableppo_046_10 as sy
    import run_053_00_lca_lowIC_expanded_action_maskableppo as lc
    runner=sy if site=='SY' else lc; engine=runner.engine.base03222
    directory=OUT/site/f'seed{ppo_seed}';directory.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/f'results/sylc_external_weather_057/{site}/configs/seed{ppo_seed}.json').read_text());cfg['task_id']=f'059{site}s{ppo_seed}';cfg['task_name']='frozen_weather_augmented_100k';cfg['seed']=ppo_seed
    path=directory/'config.json';path.write_text(json.dumps(cfg,indent=2))
    expected={};manifest=[];checked=0;training_checked=0
    original_copy=rendering.copy_weather_to_rendered_dir
    counter=0
    active_episode=-1
    plan=list(csv.DictReader((ROOT/f'results/sylc_external_weather_057/{site}/sampling/seed{ppo_seed}.csv').open()))
    bank_manifest=list(csv.DictReader((ROOT/'results/sylc_external_weather_057/weather_manifest.csv').open()))
    bank_hashes={r['WTH']:r['sha256'] for r in bank_manifest}
    def inject(station,year,template,config):
        nonlocal counter
        if 'train' not in str(template):return original_copy(station,year,template,config)
        counter+=1
        if active_episode<0:
            seed=1001
        else:
            row=plan[active_episode]
            assert year==int(row['year']), (active_episode,year,row)
            seed=int(row['weather_seed'])
        source=ROOT/f'results/sylc_external_weather_057/{site}/bank/{year}/seed{seed}/{rendering.target_weather_stem(station,year)}.WTH'
        source_hash=sha(source);assert source_hash==bank_hashes[source.relative_to(ROOT).as_posix()]
        dest=template.parent/source.name;assert not dest.exists()
        shutil.copyfile(source,dest);assert sha(dest)==source_hash
        values={}
        for line in source.read_text().splitlines()[5:]:
            fields=line.split();values[int(fields[0])]=dict(zip(['SRAD','TMAX','TMIN','RAIN'],map(float,fields[1:5])))
        expected[sha(dest)]=values
        assert len(values)==(366 if calendar.isleap(year) else 365)
        record={'episode_environment':counter,'sampling_episode':active_episode,'unused_initialization':active_episode<0,'year':year,'weather_seed':seed,'bank_WTH':source.relative_to(ROOT).as_posix(),'source_sha256':source_hash,'WTH':dest.relative_to(ROOT).as_posix(),'WTH_sha256':sha(dest),'days':len(values)}
        manifest.append(record)
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
            candidates={-1}
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
        nonlocal active_episode
        active_episode+=1
        assert active_episode<len(plan)
        for env in instance.envs.values():env.close()
        instance.envs.clear()
        return old_reset(instance,*args,**kwargs)
    engine.RandomYearEnv.reset=reset
    engine.SEED=ppo_seed
    original_verify=runner.verify_completed_smoke
    def verified_smoke(base_cfg):
        smoke_cfg=json.loads((ROOT/f'results/sylc_external_weather_058/{site}/config.json').read_text())
        frozen=json.loads((ROOT/f'results/sylc_external_weather_057/{site}/configs/seed{ppo_seed}.json').read_text())
        for obj in [smoke_cfg,frozen]:
            for key in ['task_id','task_name','seed']:obj.pop(key)
        assert smoke_cfg==frozen
        proof=original_verify(json.loads((ROOT/f'results/sylc_external_weather_058/{site}/config.json').read_text()))
        assert proof['next_step_allowed']
        proof['shared_seed0_smoke_for_frozen_cohort']=True
        return proof
    runner.verify_completed_smoke=verified_smoke
    try:
        result=runner.run(path,dry_run=False,smoke=False,formal=True)
        seeds=set(r['weather_seed'] for r in manifest if not r['unused_initialization'])
        gate={'site':site,'PPO_seed':ppo_seed,'requested_steps':100000,'training_weather_states_matched':checked,'weather_seeds':sorted(seeds),'weather_environment_count':len(manifest),'runtime_weather_date_offsets':{k:sorted(v) for k,v in date_offsets.items()},'injection_pass':checked>100000 and len(seeds)>=2,'result':result,'formal_training_allowed':False}
        (directory/'gate.json').write_text(json.dumps(gate,indent=2))
        assert gate['injection_pass']
        assert result['smoke_verification']['next_step_allowed']
        verify_and_backup(site,ppo_seed,directory,result,manifest)
        assert all(v=={-1} for v in date_offsets.values())
        print(json.dumps({'site':site,'injection_pass':True,'states':checked,'weather_seeds':sorted(seeds)},indent=2),flush=True)
    finally:
        cls._get_state=old_state;rendering.copy_weather_to_rendered_dir=original_copy;engine.RandomYearEnv.reset=old_reset;engine.base.make_env=old_make;cls._get_sockets_=old_sockets;cls._close_client=old_close_client;runner.verify_completed_smoke=original_verify
def verify_and_backup(site,seed,directory,result,manifest):
    import zipfile
    run=ROOT/result['output_root']
    def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
    inventory=rows(run/'evaluation/032_22_training_checkpoint_inventory.csv')
    assert len(inventory)==4 and {int(r['checkpoint_step']) for r in inventory}=={25000,50000,75000,100000}
    assert all(r['run_status']=='ok' and int(r['seed'])==seed and sha(ROOT/r['model_path'])==r['model_sha256'] for r in inventory)
    evaluation=rows(run/'evaluation/032_22_checkpoint_validation_summary.csv')
    assert len(evaluation)==40 and all(r['run_status'].startswith('ok') for r in evaluation)
    assert {(int(r['checkpoint_step']),int(r['year'])) for r in evaluation}=={(s,y) for s in [25000,50000,75000,100000] for y in range(2014,2024)}
    train=[r for r in manifest if not r['unused_initialization']]
    switches=rows(run/'logs'/f'032_22_{site}A_training_year_switch_log.csv')
    assert [int(r['year']) for r in switches]==[r['year'] for r in train]
    import ppo_safe_rendering as rendering
    augmented={r['WTH'] for r in manifest};historical=0
    for p in (run/'rendered_inputs').rglob('*.WTH'):
        if p.relative_to(ROOT).as_posix() in augmented:continue
        year=2000+int(p.stem[4:6]);assert 2014<=year<=2023
        assert sha(p)==sha(rendering.source_weather_path(site+'A',year));historical+=1
    assert historical==40
    backup=ROOT/'backups/sylc_059_formal'/site/f'seed{seed}';backup.mkdir(parents=True,exist_ok=False)
    files=[p for p in run.rglob('*') if p.is_file() and 'rendered_inputs' not in p.parts]+list(directory.glob('*.json'))
    archive=backup/'formal_result.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:assert all(hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).hexdigest()==sha(p) for p in files)
    evidence={'site':site,'seed':seed,'models':4,'validation_rows':40,'historical_WTH_hashes':historical,'sampling_episodes':len(train),'files_backed_up':len(files),'all_archive_sha256_verified':True,'archive_sha256':sha(archive)}
    (directory/'completion_audit.json').write_text(json.dumps(evidence,indent=2))
    (backup/'verification.json').write_text(json.dumps(evidence,indent=2))

def preflight():
    assert json.loads((ROOT/'results/sylc_external_weather_058/stage_decision.json').read_text())['status']=='PASS_FROZEN_BANK_SMOKE_AND_CONTRACT'
    assert json.loads((ROOT/'results/sylc_external_weather_058/leap_injection_attempt002/gate.json').read_text())['status']=='PASS_FROZEN_366_DAY_RUNTIME_INJECTION'
    for r in csv.DictReader((ROOT/'results/sylc_external_weather_057/artifact_manifest.csv').open()):assert sha(ROOT/r['path'])==r['sha256'],r['path']

def main():
    import psutil,signal
    preflight();OUT.mkdir(exist_ok=False)
    completed=[]
    for site in ['SY','LC']:
        for seed in range(8):
            preflight()
            queue={'status':'RUNNING','active_site':site,'active_seed':seed,'completed':completed,'controller_pid':os.getpid(),'RSS_limit_mb':1536,'seed_timeout_seconds':7200}
            (OUT/'queue_status.json').write_text(json.dumps(queue,indent=2))
            with (OUT/f'{site}_seed{seed}.log').open('x') as log:
                p=subprocess.Popen([sys.executable,__file__,'--worker',site,'--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                start=time.monotonic();peak=0;failure=None
                while p.poll() is None:
                    try:
                        parent=psutil.Process(p.pid);rss=sum(x.memory_info().rss for x in [parent]+parent.children(recursive=True) if x.is_running())/1024**2;peak=max(peak,rss)
                        if rss>1536 or time.monotonic()-start>7200:
                            failure='RESOURCE_LIMIT';os.killpg(p.pid,signal.SIGTERM)
                            try:p.wait(timeout=10)
                            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                            break
                    except psutil.NoSuchProcess:pass
                    queue.update({'worker_pid':p.pid,'peak_tree_rss_mb':peak,'elapsed_seconds':time.monotonic()-start})
                    (OUT/'queue_status.json').write_text(json.dumps(queue,indent=2));time.sleep(5)
            status={'exit_code':p.returncode,'peak_tree_rss_mb':peak,'seconds':time.monotonic()-start,'failure':failure}
            (OUT/f'{site}_seed{seed}_process.json').write_text(json.dumps(status,indent=2))
            if p.returncode or failure:
                queue.update({'status':'STOPPED_FAILURE','failure':status});(OUT/'queue_status.json').write_text(json.dumps(queue,indent=2));raise RuntimeError(f'{site} seed{seed} failed; queue stopped')
            completed.append({'site':site,'seed':seed,'process':status})
    (OUT/'queue_status.json').write_text(json.dumps({'status':'COMPLETE','completed':completed},indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=['SY','LC']);parser.add_argument('--seed',type=int,choices=range(8));args=parser.parse_args()
    if args.worker:
        preflight();worker(args.worker,args.seed)
    else:main()
