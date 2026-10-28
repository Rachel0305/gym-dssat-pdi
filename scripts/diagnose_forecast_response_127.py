"""Frozen-state policy dependence; no environment or simulator construction."""
from pathlib import Path
import os,sys,json,hashlib,time,threading,subprocess,csv,gc
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/forecast_policy_response_127_20261027'
SITES=['SY','HL','LC','YC','FQ'];YEARS=list(range(2014,2024))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(262144),b''):h.update(b)
    return h.hexdigest()
def dump(p,v):
    p=Path(p);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
    for attempt in range(50):
        try:os.replace(tmp,p);return
        except PermissionError:
            if attempt==49:raise
            time.sleep(.05)
def prepare():
    import pandas as pd
    OUT.mkdir(exist_ok=False);inv=pd.read_csv(ROOT/'results/four_site_forecast_116_20261025/main100k_seed_year_rows.csv');plan=[]
    for si,site in enumerate(SITES):
        for seed in range(8):
            base=(ROOT/'results/sy_forecast_seed0_100k_091_20261022' if seed==0 else ROOT/f'results/sy_forecast_seeds1_7_095_20261022/seed{seed}') if site=='SY' else ROOT/inv[inv.site.eq(site)&inv.seed.eq(seed)&inv.year.eq(2014)].iloc[0].source
            modelbase=ROOT/'results/four_site_forecast_110_20261023/LC/seed0' if site=='LC' and seed==0 else base
            model=modelbase/f'forecast_seed{seed}_ckpt100000.zip';assert model.exists()
            entries=[]
            for year in YEARS:
                folder=ROOT/f'results/fq2018_forecast_reassessment_125_20261027/seed{seed}' if site=='FQ' and year==2018 else base
                trace=folder/f'eval_100000_{year}_steps.jsonl';summary=folder/f'eval_100000_{year}_summary.json';assert trace.exists() and summary.exists()
                expected=json.loads(summary.read_text());assert sha(model)==expected['model_sha256']
                entries.append(dict(year=year,trace=trace.relative_to(ROOT).as_posix(),trace_sha256=sha(trace),rng_seed=12700000+si*100000+seed*1000+year))
            plan.append(dict(site=site,seed=seed,model=model.relative_to(ROOT).as_posix(),model_sha256=sha(model),years=entries))
    dump(OUT/'frozen_plan.json',dict(models=plan,variants=['shuffle_0','shuffle_1','shuffle_2','zero_block'],base_seed=127,simulations=0,training=0))
def worker():
    sys.path.insert(0,str(ROOT/'src'))
    import numpy as np,torch,psutil
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from ppo_optimization_logging_104 import LoggedMaskablePPO
    from forecast_memory_pause_108 import MemoryPause
    pause=MemoryPause(OUT/'host_memory.json',OUT/'pause_analysis.json')
    guest=MemoryPause(OUT/'guest_memory.json',OUT/'pause_guest.json')
    plan=json.loads((OUT/'frozen_plan.json').read_text());summaries=[];protected={};start=time.time();peak=0;smoke=False
    fields=['site','seed','year','variant','step','donor_step','legal_count','input_changed','original_action','changed_action','action_changed','TV','JS','original_action_probability','changed_original_action_probability','max_probability_delta']
    def probs(model,obs,masks):
        parts=[]
        with torch.no_grad():
            for i in range(0,len(obs),128):
                dump(OUT/'guest_memory.json',dict(updated_unix=time.time(),available_bytes=psutil.virtual_memory().available));guest.checkpoint();pause.checkpoint()
                dist=model.policy.get_distribution(torch.as_tensor(obs[i:i+128],dtype=torch.float32),action_masks=masks[i:i+128]);parts.append(dist.distribution.probs.cpu().numpy())
        return np.concatenate(parts)
    try:
        with (OUT/'state_response.csv').open('x',newline='') as output:
            writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader()
            for item in plan['models']:
                path=ROOT/item['model'];assert sha(path)==item['model_sha256'];protected[item['model']]=item['model_sha256']
                pause.checkpoint();model=LoggedMaskablePPO.load(str(path),device='cpu');assert model.num_timesteps==100000;model.policy.set_training_mode(False)
                for entry in item['years']:
                    trace=ROOT/entry['trace'];assert sha(trace)==entry['trace_sha256'];protected[entry['trace']]=entry['trace_sha256']
                    rows=[json.loads(x) for x in trace.read_text().splitlines()];obs=np.asarray([r['obs'] for r in rows],dtype=np.float32);masks=np.asarray([r['mask'] for r in rows],dtype=bool);actions=np.asarray([r['action'] for r in rows]);n=len(rows)
                    assert obs.shape[1]==model.observation_space.shape[0] and masks.shape==(n,16) and np.isfinite(obs).all()
                    base=probs(model,obs,masks);pred=base.argmax(axis=1);assert np.array_equal(pred,actions),('reproduction_failed',item['site'],item['seed'],entry['year'])
                    assert np.isfinite(base).all() and np.allclose(base.sum(1),1,atol=1e-6) and np.all(base[~masks]<1e-8)
                    if not smoke:
                        assert item['site']=='SY' and item['seed']==0 and entry['year']==2014
                        assert np.array_equal(base,probs(model,obs,masks));dump(OUT/'smoke.json',dict(status='passed',site='SY',seed=0,year=2014,states=n,action_reproduction=True,repeat_inference_exact=True,mask_probabilities_valid=True));smoke=True
                    rng=np.random.default_rng(entry['rng_seed']);donors=[]
                    for repeat in range(3):
                        index=rng.permutation(n)
                        if np.array_equal(index,np.arange(n)):index=np.roll(index,1)
                        donors.append((f'shuffle_{repeat}',index))
                    dump(OUT/f"permutation_{item['site']}_{item['seed']}_{entry['year']}.json",dict(seed=entry['rng_seed'],indices={v:i.tolist() for v,i in donors}))
                    donors.append(('zero_block',np.full(n,-1)));eligible=masks.sum(1)>1
                    for variant,index in donors:
                        new=obs.copy();new[:,-12:]=0 if variant=='zero_block' else obs[index,-12:]
                        assert np.array_equal(new[:,:-12],obs[:,:-12]);changed=np.any(new[:,-12:]!=obs[:,-12:],axis=1)
                        pr=probs(model,new,masks);assert np.isfinite(pr).all() and np.allclose(pr.sum(1),1,atol=1e-6) and np.all(pr[~masks]<1e-8)
                        action=pr.argmax(1);tv=.5*np.abs(pr-base).sum(1);mid=(pr+base)/2
                        def kl(x,y):return (x*np.log(np.maximum(x,1e-30)/np.maximum(y,1e-30))).sum(1)
                        js=.5*(kl(base,mid)+kl(pr,mid));delta=np.abs(pr-base).max(1);change=action!=actions
                        for i in range(n):writer.writerow(dict(site=item['site'],seed=item['seed'],year=entry['year'],variant=variant,step=i,donor_step=int(index[i]),legal_count=int(masks[i].sum()),input_changed=bool(changed[i]),original_action=int(actions[i]),changed_action=int(action[i]),action_changed=bool(change[i]),TV=float(tv[i]),JS=float(js[i]),original_action_probability=float(base[i,actions[i]]),changed_original_action_probability=float(pr[i,actions[i]]),max_probability_delta=float(delta[i])))
                        e=eligible&changed
                        summaries.append(dict(site=item['site'],seed=item['seed'],year=entry['year'],variant=variant,states=n,eligible=int(eligible.sum()),input_changed=int(changed.sum()),eligible_changed_input=int(e.sum()),action_changed=int(change[eligible].sum()),action_changed_given_changed_input=int(change[e].sum()),TV_sum_eligible=float(tv[eligible].sum()),TV_max_eligible=float(tv[eligible].max()) if eligible.any() else 0,JS_sum_eligible=float(js[eligible].sum()),max_probability_delta=float(delta[eligible].max()) if eligible.any() else 0))
                    output.flush();dump(OUT/'progress.json',dict(site=item['site'],seed=item['seed'],year=entry['year'],model_years_completed=len(summaries)//4,models_completed=(len(summaries)//4)//10))
                del model;gc.collect();peak=max(peak,psutil.Process().memory_info().rss)
        import pandas as pd
        s=pd.DataFrame(summaries);s.to_csv(OUT/'model_year_variant_1600.csv',index=False)
        for name,keys in [('policy_variant_160',['site','seed','variant']),('site_variant_20',['site','variant'])]:
            g=s.groupby(keys).agg(model_years=('year','count'),states=('states','sum'),eligible=('eligible','sum'),input_changed=('input_changed','sum'),eligible_changed_input=('eligible_changed_input','sum'),action_changed=('action_changed','sum'),action_changed_given_changed_input=('action_changed_given_changed_input','sum'),TV_sum_eligible=('TV_sum_eligible','sum'),TV_max_eligible=('TV_max_eligible','max'),JS_sum_eligible=('JS_sum_eligible','sum'))
            g['action_change_rate']=g.action_changed/g.eligible;g['action_change_rate_given_changed_input']=g.action_changed_given_changed_input/g.eligible_changed_input;g['TV_mean_eligible']=g.TV_sum_eligible/g.eligible;g['JS_mean_eligible']=g.JS_sum_eligible/g.eligible;g.to_csv(OUT/(name+'.csv'))
        dump(OUT/'result.json',dict(status='completed',models=40,model_years=400,variant_groups=len(s),elapsed_seconds=time.time()-start,post_model_sample_peak_rss_MiB=peak/1024**2,simulations=0,training_runs=0))
    finally:
        changes=[p for p,h in protected.items() if sha(ROOT/p)!=h];dump(OUT/'protected_files_check.json',dict(unchanged=not changes,checked=len(protected),changes=changes));assert not changes
def manager():
    import psutil
    prepare();stop=threading.Event()
    def telemetry():
        while not stop.is_set():
            dump(OUT/'host_memory.json',dict(updated_unix=time.time(),available_bytes=psutil.virtual_memory().available,pid=os.getpid()));stop.wait(2)
    t=threading.Thread(target=telemetry,daemon=True);t.start()
    try:
        cmd=['docker','exec','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-w','/workspace','nifty_taussig','/opt/gym_dssat_pdi/bin/python','scripts/diagnose_forecast_response_127.py','--worker']
        with (OUT/'worker.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        assert r.returncode==0,('offline_worker_failed',r.returncode)
        assert json.loads((OUT/'result.json').read_text())['status']=='completed'
    except BaseException:
        import traceback;dump(OUT/'FAILED.json',dict(traceback=traceback.format_exc()));raise
    finally:stop.set();t.join(timeout=3)
if __name__=='__main__':
    if '--worker' in sys.argv:worker()
    else:manager()
