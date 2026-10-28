"""At most 12 one-step dose branches; restore the original policy thereafter."""
from pathlib import Path
import os,sys,json,time,threading,subprocess,inspect,traceback
import test_rain_timing_mechanism_129 as base
ROOT=base.ROOT;OUT=ROOT/'results/single_decision_value_130_20261027';DOSES=[0,15,30,45]
def setup_points():
    points=[]
    for year in base.YEARS:
        rows=[json.loads(x) for x in (base.SOURCE/f'eval_100000_{year}_steps.jsonl').read_text().splitlines()]
        eligible=[r for r in rows if r['step']>0 and r['I']>0 and r['pre_action_raw_forecast']['rain_next3_mm']>=15 and all(r['mask'][i*4+r['action']%4] for i in range(4))]
        if eligible:
            r=eligible[0];assert DOSES[r['action']//4]==r['I'];points.append(dict(year=year,step=r['step'],date=r['date'],original_action=r['action'],original_I=r['I'],original_N=r['N'],rain_next3=r['pre_action_raw_forecast']['rain_next3_mm']))
    return points
def child(year,dose):
    point=next(p for p in json.loads((OUT/'frozen_points.json').read_text()) if p['year']==year)
    base.OUT=OUT;base.ARMS=[f'branch_{x}' for x in DOSES]
    code=inspect.getsource(base.worker)
    begin=code.index('    def gate(action,mask,env,step,actualyear):');end=code.index("    ns['gate_action']=gate",begin)
    gate='''    def gate(action,mask,env,step,actualyear):
        assert actualyear==year
        selected=(DOSES.index(dose)*4+action%4) if step==point['step'] else action
        assert mask[selected]
        counter[0]+=int(selected!=action)
        gate_log.write(json.dumps(dict(step=step,original_action=action,selected_action=selected,target_step=point['step'],dose_mm=dose,intervened=step==point['step']))+'\\n');gate_log.flush();return selected
'''
    code=code[:begin]+gate+code[end:];ns=dict(vars(base));ns.update(DOSES=DOSES,dose=dose,point=point);exec(compile(code,__file__,'exec'),ns)
    (OUT/f'{year}_branch_{dose}'/'executed_worker_source.py').write_text(code,encoding='utf-8')
    ns['worker'](year,f'branch_{dose}')
    folder=OUT/f'{year}_branch_{dose}';new=[json.loads(x) for x in (folder/f'eval_100000_{year}_steps.jsonl').read_text().splitlines()];old=[json.loads(x) for x in (base.SOURCE/f'eval_100000_{year}_steps.jsonl').read_text().splitlines()]
    import numpy as np
    assert all(a['action']==b['action'] and a['mask']==b['mask'] and np.allclose(a['obs'],b['obs'],atol=1e-6,rtol=0) for a,b in zip(new[:point['step']],old[:point['step']]))
    assert new[point['step']]['I']==dose and new[point['step']]['N']==point['original_N']
    if dose==point['original_I']:
        assert len(new)==len(old) and all(a['action']==b['action'] and np.allclose(a['obs'],b['obs'],atol=1e-6,rtol=0) for a,b in zip(new,old))
    base.dump(folder/'single_decision_gate.json',dict(status='passed',prefix_reproduced=True,target_step=point['step'],dose_mm=dose,N_at_target_unchanged=True,original_branch_full_reproduction=dose==point['original_I']))
def manager():
    import psutil,csv
    OUT.mkdir(exist_ok=False);points=setup_points();base.dump(OUT/'frozen_points.json',points);stop=threading.Event()
    protected=[base.SOURCE/'forecast_seed0_ckpt100000.zip',ROOT/'scripts/test_rain_timing_mechanism_129.py'];before={str(p.relative_to(ROOT)):base.sha(p) for p in protected}
    base.dump(OUT/'protected_before.json',before)
    def telemetry():
        while not stop.is_set():
            try:base.dump(OUT/'host_memory.json',dict(updated_unix=time.time(),available_bytes=psutil.virtual_memory().available,pid=os.getpid()))
            except PermissionError:pass
            stop.wait(2)
    t=threading.Thread(target=telemetry,daemon=True);t.start();results=[]
    try:
        for point in points:
            year=point['year'];order=[point['original_I']]+[x for x in DOSES if x!=point['original_I']]
            for dose in order:
                dose=int(dose);folder=OUT/f'{year}_branch_{dose}';folder.mkdir();base.dump(OUT/'manager_status.json',dict(year=year,dose=dose,completed=len(results)))
                cmd=['docker','exec','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-w','/workspace','nifty_taussig','/opt/gym_dssat_pdi/bin/python','scripts/test_single_decision_value_130.py','--worker',str(year),str(dose)]
                with (folder/'worker.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
                assert r.returncode==0,('branch_failed',year,dose)
                assert json.loads((folder/'single_decision_gate.json').read_text())['status']=='passed'
                s=json.loads((folder/f'eval_100000_{year}_summary.json').read_text());s.update(dose_mm=dose,point_step=point['step']);results.append(s);base.dump(OUT/'results_partial.json',results)
        if results:
            with (OUT/'branch_results.csv').open('x',newline='') as f:
                cols=['year','dose_mm','point_step','yield_kg_ha','I','N','reward_sum','steps'];w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(results)
        base.dump(OUT/'result.json',dict(status='completed' if points else 'no_eligible_points',simulations=len(results),training_runs=0))
    except BaseException:base.dump(OUT/'FAILED.json',dict(traceback=traceback.format_exc(),completed=len(results)));raise
    finally:
        stop.set();t.join(timeout=3);after={str(p.relative_to(ROOT)):base.sha(p) for p in protected};base.dump(OUT/'protected_files_check.json',dict(unchanged=before==after));assert before==after
if __name__=='__main__':
    if '--worker' in sys.argv:child(int(sys.argv[-2]),int(sys.argv[-1]))
    else:manager()
