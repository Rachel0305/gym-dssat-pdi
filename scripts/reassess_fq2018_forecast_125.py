"""Eight existing models, one corrected year, serial; no training entrypoint."""
from pathlib import Path
import os,sys,json,hashlib,time,threading,subprocess,traceback
ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'results/fq2018_forecast_reassessment_125_20261027'
SOURCE=ROOT/'results/four_site_forecast_116_20261025/FQ'
CORRECTED=ROOT/'benchmark_results/088_00_fq2018_input_quality_correction/corrected_input'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(262144),b''):h.update(b)
    return h.hexdigest()
def dump(p,v):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(tmp,p)
def worker(seed):
    assert 0<=seed<8
    os.environ.update(FORECAST097_SITE='FQ',FORECAST097_SEED=str(seed));os.environ.pop('FORECAST097_SMOKE',None)
    src=ROOT/'scripts/run_four_site_forecast_116.py';code=src.read_text()
    old="path=OUT/f'forecast_seed{SEED}_ckpt{step}.zip'"
    assert code.count(old)==1
    code=code.replace(old,"path=ROOT/'results/four_site_forecast_116_20261025/FQ'/f'seed{SEED}'/f'forecast_seed{SEED}_ckpt{step}.zip'")
    ns={'__file__':str(src),'__name__':'isolated_125_evaluation'};exec(compile(code,str(src),'exec'),ns)
    ns['BATCH']=TASK;ns['OUT']=TASK/f'seed{seed}';ns['SPEC']=dict(ns['SPEC']);ns['SPEC']['input_root']=CORRECTED.relative_to(ROOT).as_posix()
    sys.path.insert(0,str(ROOT/'src'))
    import psutil
    from forecast_thread_guard_115 import ThreadGuard
    guard=ThreadGuard(ns['OUT']);stop=threading.Event()
    def observe():
        while not stop.wait(2):
            try:
                parent=psutil.Process();guard.observe([parent]+parent.children(recursive=True))
                dump(ns['OUT']/'live_resource.json',dict(unix=time.time(),rss_bytes=sum(p.memory_info().rss for p in [parent]+parent.children(recursive=True) if p.is_running())))
            except (psutil.NoSuchProcess,ProcessLookupError):pass
    t=threading.Thread(target=observe,daemon=True);t.start()
    try:ns['worker']('eval_100000_2018')
    finally:stop.set();t.join(timeout=3)
    folder=ns['OUT'];snap=folder/'eval_100000_2018_snapshot/dssat';s=json.loads((folder/'eval_100000_2018_summary.json').read_text())
    assert s['year']==2018 and s['normal_termination'] and s['model_sha256']==sha(SOURCE/f'seed{seed}/forecast_seed{seed}_ckpt100000.zip')
    assert sha(snap/'CNFQ1801.WTH')==sha(CORRECTED/'FQ/CNFQ1801.WTH')
    oldsnap=SOURCE/f'seed{seed}/eval_100000_2018_snapshot/dssat'
    for name in ['SOIL.SOL','MZCER048.CUL']:assert sha(snap/name)==sha(oldsnap/name),name
    # Final fixed-width Summary fields are authoritative and independently checked.
    lines=(snap/'Summary.OUT').read_text().splitlines();head=next(x for x in lines if x.startswith('@'));last=next(x for x in reversed(lines) if x.strip() and x.lstrip()[0].isdigit())
    def val(k):end=head.index(k)+len(k);return float(last[end-5:end])
    assert abs(val('HWAM')-s['yield_kg_ha'])<=1 and val('IRCM')==s['I'] and val('NICM')==s['N']
    assert json.loads((folder/'eval_100000_2018_resource_closure.json').read_text())['status']=='passed'
    dump(folder/'technical_gate.json',dict(status='passed',seed=seed,year=2018,corrected_weather_hash=sha(snap/'CNFQ1801.WTH'),soil_cultivar_unchanged=True,model_identity=True,normal_termination=True,summary_resource_closure=True,training=False))
def manager():
    import psutil,csv
    TASK.mkdir(exist_ok=False)
    protected=[ROOT/'scripts/run_four_site_forecast_116.py',ROOT/'configs/four_site_forecast_116.json',CORRECTED/'FQ/CNFQ1801.WTH',ROOT/'DSSAT_auto_validation/multisite_new_cultivar_inputs_013/FQ/CNFQ1801.WTH']
    for seed in range(8):protected.extend([SOURCE/f'seed{seed}/forecast_seed{seed}_ckpt100000.zip',SOURCE/f'seed{seed}/eval_100000_2018_summary.json'])
    before={p.relative_to(ROOT).as_posix():sha(p) for p in protected};dump(TASK/'protected_before.json',before)
    correction=json.loads((ROOT/'benchmark_results/088_00_fq2018_input_quality_correction/088_00_correction_manifest.json').read_text())
    assert sha(CORRECTED/'FQ/CNFQ1801.WTH')==correction['corrected_weather_sha256']
    stop=threading.Event()
    def telemetry():
        while not stop.is_set():
            dump(TASK/'host_memory.json',dict(updated_unix=time.time(),available_bytes=psutil.virtual_memory().available,monitor_pid=os.getpid()))
            stop.wait(2)
    t=threading.Thread(target=telemetry,daemon=True);t.start();results=[]
    try:
        for seed in range(8):
            if seed:assert json.loads((TASK/'seed0/technical_gate.json').read_text())['status']=='passed'
            folder=TASK/f'seed{seed}';folder.mkdir()
            dump(TASK/'manager_status.json',dict(status='smoke' if seed==0 else 'evaluating',seed=seed,completed=len(results),pid=os.getpid()))
            command=['docker','exec','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-w','/workspace','nifty_taussig','/opt/gym_dssat_pdi/bin/python','scripts/reassess_fq2018_forecast_125.py','--worker',str(seed)]
            with (folder/'worker.log').open('x') as log:
                p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            assert p.returncode==0,('worker_failed',seed,p.returncode)
            assert json.loads((folder/'technical_gate.json').read_text())['status']=='passed'
            s=json.loads((folder/'eval_100000_2018_summary.json').read_text());old=json.loads((SOURCE/f'seed{seed}/eval_100000_2018_summary.json').read_text())
            results.append(dict(seed=seed,year=2018,original_yield=old['yield_kg_ha'],corrected_yield=s['yield_kg_ha'],original_I=old['I'],corrected_I=s['I'],original_N=old['N'],corrected_N=s['N']))
            dump(TASK/'completed_partial.json',results)
        with (TASK/'original_vs_corrected_8.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
        dump(TASK/'result.json',dict(status='completed',evaluations=8,training_runs=0,version='088 heuristic TMIN correction; not independently verified observations'))
        dump(TASK/'manager_status.json',dict(status='completed',completed=8))
    except BaseException:
        dump(TASK/'FAILED.json',dict(traceback=traceback.format_exc(),completed=len(results)));raise
    finally:
        stop.set();t.join(timeout=3);after={p.relative_to(ROOT).as_posix():sha(p) for p in protected};dump(TASK/'protected_files_check.json',dict(unchanged=before==after));assert before==after
if __name__=='__main__':
    if '--worker' in sys.argv:worker(int(sys.argv[-1]))
    else:manager()
