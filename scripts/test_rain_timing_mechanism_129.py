"""Frozen SY seed0 three-arm pilot; no training entrypoint."""
from pathlib import Path
import os,sys,json,time,hashlib,threading,subprocess,traceback
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/rain_timing_mechanism_129_20261027';SOURCE=ROOT/'results/sy_forecast_seed0_100k_091_20261022';YEARS=[2014,2018,2023];ARMS=['baseline','correct_gate','shifted_gate']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
    for i in range(50):
        try:os.replace(tmp,p);return
        except PermissionError:
            if i==49:raise
            time.sleep(.05)
def worker(year,arm):
    assert year in YEARS and arm in ARMS
    folder=OUT/f'{year}_{arm}';os.environ['SY_FORECAST_SEED']='0';os.environ.pop('SY_FORECAST_SMOKE',None)
    source=ROOT/'scripts/run_sy_forecast_seed_extension_095.py';code=source.read_text();code=code.replace('assert SEED in range(1,8)','assert SEED in range(8)')
    key="path=OUT/f'forecast_seed{SEED}_ckpt{step}.zip'";assert code.count(key)==1;code=code.replace(key,"path=ROOT/'results/sy_forecast_seed0_100k_091_20261022'/f'forecast_seed{SEED}_ckpt{step}.zip'")
    key='                    assert mask[action]';assert code.count(key)==1;code=code.replace(key,'                    action=gate_action(action,mask,env,step,year)\n'+key)
    code=code.replace("                    date=check_observation(env,obs);mask=", "                    pauser.checkpoint()\n                    date=check_observation(env,obs);mask=")
    ns={'__file__':str(source),'__name__':'mechanism129_isolated'};exec(compile(code,str(source),'exec'),ns);ns['OUT']=folder;ns['BATCH']=OUT
    sys.path.insert(0,str(ROOT/'src'))
    from forecast_memory_pause_108 import MemoryPause
    pauser=MemoryPause(OUT/'host_memory.json',folder/'pause.json');ns['pauser']=pauser
    prior=[json.loads(x) for x in (SOURCE/f'eval_100000_{year}_steps.jsonl').read_text().splitlines()]
    gate_log=(folder/'gate_decisions.jsonl').open('x');counter=[0]
    def gate(action,mask,env,step,actualyear):
        assert actualyear==year
        raw=env.last_raw_forecast_features;rain=float(raw['rain_next3_mm']);donor=(step+28)%len(prior);supplied=float(prior[donor]['pre_action_raw_forecast']['rain_next3_mm']) if arm=='shifted_gate' else rain
        candidate=action%4;trigger=arm!='baseline' and action//4>0 and supplied>=15 and bool(mask[candidate]);selected=candidate if trigger else action;counter[0]+=int(trigger)
        gate_log.write(json.dumps(dict(step=step,original_action=action,selected_action=selected,correct_rain_next3=rain,supplied_gate_rain=supplied,donor_step=donor if arm=='shifted_gate' else step,triggered=trigger,substitute_legal=bool(mask[candidate])))+'\n');gate_log.flush();return selected
    ns['gate_action']=gate
    try:ns['worker'](f'eval_100000_{year}')
    finally:gate_log.close()
    s=json.loads((folder/f'eval_100000_{year}_summary.json').read_text());snap=folder/f'eval_100000_{year}_snapshot/dssat'
    lines=(snap/'Summary.OUT').read_text().splitlines();head=next(x for x in lines if x.startswith('@'));last=next(x for x in reversed(lines) if x.strip() and x.lstrip()[0].isdigit())
    def val(k):end=head.index(k)+len(k);return float(last[end-5:end])
    assert s['normal_termination'] and abs(s['yield_kg_ha']-val('HWAM'))<=1 and s['I']==val('IRCM') and s['N']==val('NICM')
    oldsnap=SOURCE/f'eval_100000_{year}_snapshot/dssat'
    for name in ['SOIL.SOL','MZCER048.CUL',f'CNSY{year%100:02d}01.WTH']:assert sha(snap/name)==sha(oldsnap/name),name
    if arm=='baseline':
        rows=[json.loads(x) for x in (folder/f'eval_100000_{year}_steps.jsonl').read_text().splitlines()];assert len(rows)==len(prior)
        import numpy as np
        assert all(a['action']==b['action'] and a['mask']==b['mask'] and np.allclose(a['obs'],b['obs'],atol=1e-6,rtol=0) for a,b in zip(rows,prior))
        expected=json.loads((SOURCE/f'eval_100000_{year}_summary.json').read_text());assert abs(s['yield_kg_ha']-expected['yield_kg_ha'])<=1 and s['I']==expected['I'] and s['N']==expected['N']
    dump(folder/'technical_gate.json',dict(status='passed',year=year,arm=arm,model_sha256=s['model_sha256'],triggered_actions=counter[0],summary_resource_closure=True,physical_inputs_equal=True,baseline_trace_reproduced=arm=='baseline'))
def manager():
    import psutil,csv
    OUT.mkdir(exist_ok=False);dump(OUT/'frozen_protocol.json',dict(site='SY',seed=0,years=YEARS,arms=ARMS,rain_next3_threshold_mm=15,shift_days=28,maximum_simulations=9,training_runs=0,yield_floor_mean=.99,irrigation_saving_mean_mm=15,N_no_increase=True))
    protected=[SOURCE/'forecast_seed0_ckpt100000.zip',ROOT/'scripts/run_sy_forecast_seed_extension_095.py']+[SOURCE/f'eval_100000_{y}_summary.json' for y in YEARS]+[SOURCE/f'eval_100000_{y}_steps.jsonl' for y in YEARS]
    before={p.relative_to(ROOT).as_posix():sha(p) for p in protected};dump(OUT/'protected_before.json',before);stop=threading.Event()
    def telemetry():
        while not stop.is_set():
            try:dump(OUT/'host_memory.json',dict(updated_unix=time.time(),available_bytes=psutil.virtual_memory().available,pid=os.getpid()))
            except PermissionError:pass
            stop.wait(2)
    t=threading.Thread(target=telemetry,daemon=True);t.start();results=[]
    try:
        for year in YEARS:
            for arm in ARMS:
                if results:assert json.loads((OUT/'2014_baseline/technical_gate.json').read_text())['status']=='passed'
                folder=OUT/f'{year}_{arm}';folder.mkdir();dump(OUT/'manager_status.json',dict(year=year,arm=arm,completed=len(results)))
                cmd=['docker','exec','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-w','/workspace','nifty_taussig','/opt/gym_dssat_pdi/bin/python','scripts/test_rain_timing_mechanism_129.py','--worker',str(year),arm]
                with (folder/'worker.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
                assert r.returncode==0,('technical_failure',year,arm,r.returncode)
                s=json.loads((folder/f'eval_100000_{year}_summary.json').read_text());gate=json.loads((folder/'technical_gate.json').read_text());assert gate['status']=='passed';s.update(arm=arm,triggered_actions=gate['triggered_actions']);results.append(s);dump(OUT/'results_partial.json',results)
        with (OUT/'results_9.csv').open('x',newline='') as f:
            fields=['year','arm','yield_kg_ha','I','N','steps','triggered_actions'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)
        dump(OUT/'result.json',dict(status='completed',simulations=9,training_runs=0))
    except BaseException:dump(OUT/'FAILED.json',dict(traceback=traceback.format_exc(),completed=len(results)));raise
    finally:
        stop.set();t.join(timeout=3);after={p.relative_to(ROOT).as_posix():sha(p) for p in protected};dump(OUT/'protected_files_check.json',dict(unchanged=before==after));assert before==after
if __name__=='__main__':
    if '--worker' in sys.argv:worker(int(sys.argv[-2]),sys.argv[-1])
    else:manager()
