"""097 persistent queue: wait for SY, then HL/LC/YC/FQ, one worker at a time."""
from pathlib import Path
import csv,fcntl,hashlib,json,os,signal,statistics,subprocess,sys,time,traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sy_forecast_disk_guard_095 import directory_bytes
from forecast_active_clock_101 import ActiveClock,total_paused
from forecast_thread_guard_115 import ThreadGuard
PLAN_PATH=ROOT/'configs/four_site_forecast_116.json'
PLAN=json.loads(PLAN_PATH.read_text())
OUT=ROOT/PLAN['output_root']
RUNNER=ROOT/'scripts/run_four_site_forecast_116.py'
START=time.monotonic()

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(262144),b''):h.update(b)
    return h.hexdigest()
def verify_frozen():
    for p,h in PLAN['frozen_sha256'].items():assert sha(ROOT/p)==h,('Protected source drift',p)
def active_simulations():
    import psutil
    found=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            name=p.info['name'] or '';args=p.info['cmdline'] or []
            if name.startswith('dscsm') or ('python' in name and any('run_sy_forecast_seed_extension_095.py' in a or 'run_four_site_forecast_104.py' in a for a in args)):
                found.append(dict(pid=p.pid,name=name,args=args))
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return found
def terminate_owned(pid):
    import psutil
    try:procs=[*psutil.Process(pid).children(recursive=True),psutil.Process(pid)]
    except psutil.NoSuchProcess:return
    for p in procs:
        try:p.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(procs,timeout=5)
    for p in alive:
        try:p.kill()
        except psutil.NoSuchProcess:pass
def host_memory():
    try:
        value=read(OUT/'host_memory.json')
        return value['available_bytes'] if 0<=time.time()-value['updated_unix']<=120 else 0
    except (OSError,ValueError,KeyError):
        return 0

def classify_dssat_threads(names):
    return sum(not n.startswith('ZMQbg/') for n in names),sum(n.startswith('ZMQbg/') for n in names)
def dssat_thread_counts(pid):
    result=subprocess.run(['ps','-T','-p',str(pid),'-o','comm='],capture_output=True,text=True,timeout=5)
    assert result.returncode in [0,1],result.stderr
    return classify_dssat_threads([n.strip() for n in result.stdout.splitlines() if n.strip()])
def resources():
    import psutil,shutil
    # Running workers use owned-RSS limits; guest free RAM is not a kill condition.
    assert shutil.disk_usage(OUT).free>=12*1024**3,'Free project disk below12GiB'
    assert time.monotonic()-START-total_paused(OUT,queue=True)<72*3600,'Queue72h deadline reached'
    # Host pressure is handled cooperatively by the worker, never by killing it.
def aggregate():
    rows=[];means=[]
    for site in PLAN['site_order']:
        for seed in PLAN['seeds']:
            folder=((ROOT/'results/hl_recovery_106_20261023/formal') if seed==0 else ROOT/'results/four_site_forecast_110_20261023/HL'/f'seed{seed}') if site=='HL' else (ROOT/'results/lc_evaluation_114_20261024' if site=='LC' and seed==0 else OUT/site/f'seed{seed}')
            if site=='LC' and seed in [1,2,3]:folder=ROOT/'results/four_site_forecast_114_20261024/LC'/f'seed{seed}'
            if site=='LC' and seed==4:folder=ROOT/'results/lc_recovery_115_20261024/formal'
            if site=='YC' or (site=='LC' and seed>=5):folder=ROOT/'results/four_site_forecast_115_20261024'/site/f'seed{seed}'
            if not (folder/'result.json').exists():continue
            if read(folder/'result.json')['status']!='completed_100k_and_40_evaluations':continue
            assert read(folder/('evaluation107_protected_check.json' if (site=='HL' and seed==0) or (site=='LC' and seed==0) else 'protected_files_check.json'))['unchanged']
            vals=read(folder/'evaluation_summaries.json');assert len(vals)==40
            assert {(r['checkpoint_step'],r['year']) for r in vals}=={(s,y) for s in [25000,50000,75000,100000] for y in range(2014,2024)}
            base=read(folder/'baseline_replays.json');assert sorted(r['year'] for r in base)==list(range(2014,2024))
            for r in vals:
                if r['checkpoint_step']!=100000:continue
                b=next(b for b in base if b['year']==r['year'])
                row=dict(site=site,seed=seed,year=r['year'],checkpoint_step=100000)
                for k in ['yield_kg_ha','I','N','PFP_N']:
                    row['forecast_'+k]=r[k];row['baseline_replay_'+k]=b[k]
                    row['delta_'+k]=None if r[k] is None or b[k] is None else r[k]-b[k]
                row['source']=str(folder.relative_to(ROOT));rows.append(row)
            q=[r for r in rows if r['site']==site and r['seed']==seed]
            assert len(q)==10
            means.append(dict(site=site,seed=seed,**{k:statistics.mean(r[k] for r in q if r[k] is not None) if any(r[k] is not None for r in q) else None for k in q[0] if k.startswith(('forecast_','baseline_replay_','delta_'))}))
    for name,data in [('main100k_seed_year_rows.csv',rows),('main100k_seed_means.csv',means)]:
        if not data:continue
        p=OUT/name;temp=p.with_suffix('.csv.tmp')
        with temp.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        os.replace(temp,p)
    stats={}
    for site in PLAN['site_order']:
        q=[r for r in means if r['site']==site]
        stats[site]=dict(completed_seeds=[r['seed'] for r in q],complete=len(q)==8,metrics={k:dict(n=len(v),mean=statistics.mean(v) if v else None,sample_sd=statistics.stdev(v) if len(v)>1 else None) for k in ['forecast_yield_kg_ha','baseline_replay_yield_kg_ha','delta_yield_kg_ha','forecast_I','baseline_replay_I','forecast_N','baseline_replay_N','forecast_PFP_N','baseline_replay_PFP_N'] for v in [[r[k] for r in q if r[k] is not None]]})
    dump('summary_partial.json',dict(completed_seed_runs=len(means),seed_year_rows=len(rows),sites=stats,notes=['Incomplete sites are descriptive progress only.','Main model fixed100K; all seeds and years retained.','Baseline replay uses same current source inputs; historical archived values may differ.','Forecast37dim MLP compared with archived25dim MLP; same seed is not paired initialization or a causal information-value test.','No WP_ET or NUE claim without separate exact ETCP extraction.']))
    lines=['# 四站天气预报串行队列','',f'已完成 {len(means)}/32 个正式seed；保留全部seed、年份与固定100K结果。','', '|站点|完成seed|状态|','|---|---|---|']
    lines.extend(f"|{s}|{','.join(map(str,stats[s]['completed_seeds'])) or '无'}|{'完整' if stats[s]['complete'] else '未完成'}|" for s in PLAN['site_order'])
    lines+=['','顺序：SY队列完成后，HL→LC→YC→FQ；每站技术门槛和2K smoke通过后再seed0–7。','比较使用各站原始25维模型在本轮相同输入上的确定性回放；不同输入维数及历史训练抽样限制因果解释。','产量须同时结合灌溉、施氮、PFP_N；不作未经ETCP核验的WP_ET/NUE结论。','实时状态见queue_status.json，错误见FAILED.json，各站和seed日志原样保留。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
def launch(site,seed,smoke):
    import psutil
    # User explicitly removed host-free-RAM startup threshold. Runtime pause remains.
    while True:
        try:
            telemetry=read(OUT/'host_memory.json')
            if 0<=time.time()-telemetry['updated_unix']<=120:break
        except (OSError,ValueError,KeyError):pass
        dump('queue_status.json',dict(status='waiting_for_host_telemetry',site=site,seed=seed,completed_runs=[dict(site='HL',seed=0)],updated_unix=time.time()))
        time.sleep(5)
    if (OUT/'queue_status.json').exists():
        state=read(OUT/'queue_status.json')
        if state['status'] in ['waiting_for_host_memory','waiting_for_host_telemetry']:
            state.update(status='site_smoke' if smoke else 'running',updated_unix=time.time());dump('queue_status.json',state)
    resources();verify_frozen();assert not active_simulations(),('Other simulator/worker active',active_simulations())
    env=dict(os.environ,FORECAST097_SITE=site,FORECAST097_SEED=str(seed),FORECAST097_SMOKE='1' if smoke else '0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    folder=OUT/site/('smoke_seed0' if smoke else f'seed{seed}');assert not folder.exists(),'No overwrite/restart'
    folder.parent.mkdir(exist_ok=True)
    start=time.monotonic();samples=[];clock=ActiveClock(folder)
    with (OUT/site/('smoke_seed0_manager.log' if smoke else f'seed{seed}_manager.log')).open('w') as log:
        child=subprocess.Popen([sys.executable,str(RUNNER)],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        dump('active_child.json',dict(site=site,seed=seed,smoke=smoke,pid=child.pid))
        try:
            thread_guard=ThreadGuard(folder)
            with (OUT/site/('smoke_resources.csv' if smoke else f'seed{seed}_resources.csv')).open('w',newline='') as resource_log:
                w=csv.writer(resource_log);w.writerow(['seconds','tree_rss_mib','live_dssat_count'])
                ticks=0
                while child.poll() is None:
                    resources();ticks+=1
                    try:tree=[psutil.Process(child.pid),*psutil.Process(child.pid).children(recursive=True)]
                    except psutil.NoSuchProcess:tree=[]
                    rss=0;dssat=0;max_compute_threads=0;max_ipc_threads=0
                    for p in tree:
                        try:
                            rss+=p.memory_info().rss
                            if p.name().startswith('dscsm'):
                                dssat+=1;compute,ipc=dssat_thread_counts(p.pid)
                                max_compute_threads=max(max_compute_threads,compute);max_ipc_threads=max(max_ipc_threads,ipc)
                        except psutil.NoSuchProcess:pass
                    elapsed=clock.elapsed()
                    w.writerow([elapsed,rss/1024**2,dssat]);resource_log.flush()
                    assert rss<=1152*1024**2,'Owned process tree exceeds1152MiB'
                    try:thread_guard.observe(tree)
                    except psutil.Error:pass
                    # Owned concurrency is handled by cooperative guard; external simulations are checked before launch.
                    assert elapsed<(1800 if smoke else 8*3600),'Run wall-clock deadline'
                    if ticks%60==0:assert directory_bytes(folder)<=2*1024**3,'Seed output exceeds2GiB'
                    dump('live_resource.json',dict(site=site,seed=seed,smoke=smoke,seconds=elapsed,tree_rss_mib=rss/1024**2,live_dssat_count=dssat,dssat_max_compute_threads=max_compute_threads,dssat_max_ipc_threads=max_ipc_threads,host_available_mib=host_memory()/1024**2,pid=child.pid))
                    time.sleep(5)
            assert child.returncode==0,('Worker failed',site,seed,smoke)
        finally:
            if child.poll() is None:terminate_owned(child.pid)
    assert read(folder/'result.json')['status']==('passed_technical_smoke' if smoke else 'completed_100k_and_40_evaluations')
    assert read(folder/('evaluation107_protected_check.json' if (site=='HL' and seed==0) or (site=='LC' and seed==0) else 'protected_files_check.json'))['unchanged']
    assert read(folder/'optimization_log_gate.json')['status']=='passed'
    if smoke:
        assert read(folder/'date_gate.json')['status']=='passed'
        assert read(folder/'memory_gate.json')['status']=='passed'
    assert not active_simulations(),'Worker leaked process'
    verify_frozen()
def main():
    import psutil
    assert PLAN.get('production_gate')=='OPEN_VERIFIED','Production gate closed'
    for gate in PLAN['gate_evidence']:assert read(ROOT/gate)['status']=='passed',gate
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'queue.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'queue_status.json').exists(),'No automatic restart'
    verify_frozen();dump('manager_identity.json',dict(pid=os.getpid(),started_unix=time.time(),plan_sha256=sha(PLAN_PATH),site_order=PLAN['site_order'],model_count=32,threads=1))
    done=[dict(site=s,seed=i) for s in ['HL','LC','YC'] for i in range(8)];aggregate()
    try:
        while True:
            resources();sy=read(ROOT/PLAN['sy_dependency'])
            if sy['status']=='failed':raise RuntimeError('SY dependency failed; downstream queue stopped')
            if sy['status']=='completed':break
            dump('queue_status.json',dict(status='waiting_for_SY',sy=sy,completed_runs=done,pid=os.getpid(),updated_unix=time.time()))
            time.sleep(30)
        # SY writes its final status before exiting; permit a brief orderly exit.
        for _ in range(12):
            if not active_simulations():break
            time.sleep(5)
        assert not active_simulations(),'SY/external simulation still active'
        assert read(ROOT/'results/sy_forecast_seeds1_7_095_20261022/result.json')['status']=='completed'
        assert read(ROOT/'results/sy_forecast_seeds1_7_095_20261022/old_protected_files_check.json')['unchanged']
        for site in PLAN['site_order']:
            if site in ['HL','LC','YC']:continue
            dump('queue_status.json',dict(status='site_smoke',site=site,seed=0,completed_runs=done,pid=os.getpid(),updated_unix=time.time()))
            if not (OUT/site/'smoke_seed0/result.json').exists():launch(site,0,True)
            assert read(OUT/site/'smoke_seed0/result.json')['status']=='passed_technical_smoke'
            assert read(OUT/site/'smoke_seed0/protected_files_check.json')['unchanged']
            for seed in PLAN['seeds']:
                if site=='LC' and seed<=4:continue
                dump('queue_status.json',dict(status='running',site=site,seed=seed,completed_runs=done,pid=os.getpid(),updated_unix=time.time()))
                launch(site,seed,False);done.append(dict(site=site,seed=seed));aggregate()
            dump(f'{site}/site_complete.json',dict(status='completed',seeds=8,forecast_evaluation_rows=320,baseline_replay_rows=80))
        aggregate();verify_frozen()
        dump('result.json',dict(status='completed',formal_seed_runs=32,forecast_evaluation_rows=1280,baseline_replay_rows=320,source_hashes_unchanged=True))
        dump('queue_status.json',dict(status='completed',completed_runs=done,pid=os.getpid(),updated_unix=time.time()))
    except BaseException:
        dump('FAILED.json',dict(traceback=traceback.format_exc(),completed_runs=done))
        aggregate();dump('queue_status.json',dict(status='failed',completed_runs=done,pid=os.getpid(),updated_unix=time.time()));raise

if __name__=='__main__':
    dependency=ROOT/'results/lc_recovery_115_20261024/formal'
    while not (dependency/'result.json').exists():
        assert not (dependency.parent/'FAILED_manager.json').exists(),'Recovery failed; no downstream training started'
        time.sleep(10)
    assert read(dependency/'result.json')['status']=='completed_100k_and_40_evaluations'
    assert read(dependency/'protected_files_check.json')['unchanged']
    main()
