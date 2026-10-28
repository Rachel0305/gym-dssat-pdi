"""Serial equal-environment-budget PPO/DDQN pilot, after archived backup."""
from pathlib import Path
import os,sys,json,time,threading,subprocess,traceback,inspect
import test_rain_timing_mechanism_129 as utils
ROOT=utils.ROOT;OUT=ROOT/'results/double_dqn_pilot_132_20261028/attempt02'
def train_impl(make,rollout,engine,runtime,ec,seed,algo,budget,folder,check_observation):
    import numpy as np,pandas as pd,torch,signal,gc,psutil
    from forecast_memory_pause_108 import MemoryPause
    from strict_double_dqn_132 import StrictDoubleDQN,technical_test
    from ppo_optimization_logging_104 import LoggedMaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from sb3_contrib.common.maskable.utils import get_action_masks
    assert technical_test()['status']=='passed';utils.dump(folder/'double_dqn_unit_gate.json',technical_test())
    host=MemoryPause(OUT/'host_memory.json',folder/'pause_host.json');guest=MemoryPause(folder/'guest_memory.json',folder/'pause_guest.json');start=time.time();peak=[0];counter=[0]
    def resource():
        utils.dump(folder/'guest_memory.json',dict(updated_unix=time.time(),available_bytes=psutil.virtual_memory().available));host.checkpoint();guest.checkpoint();peak[0]=max(peak[0],psutil.Process().memory_info().rss)
    runtime['seed']=seed
    class Env(engine.base03222.RandomYearEnv):
        def _env_for(self,year):
            for y in list(self.envs):
                if y!=int(year):self.envs.pop(y).close();gc.collect()
            if year not in self.envs:self.envs[year]=make(year,f'train_{year}',False)
            return self.envs[year]
        def reset(self,*args,**kwargs):
            signal.alarm(180);resource();return super().reset(*args,**kwargs)
        def step(self,action):
            resource();mask=np.asarray(self.action_masks(),bool);assert mask[int(action)]
            check_observation(self.current_env,self.current_env.last_raw_enhanced_obs)
            result=super().step(action);counter[0]+=1
            if counter[0]%144==0:utils.dump(folder/'training_progress.json',dict(steps=counter[0],budget=budget,algo=algo,seed=seed,rss_MiB=peak[0]/1024**2))
            if result[2] or result[3]:signal.alarm(0)
            return result
    env=Env(runtime,ec,'SYA',list(range(2005,2014)),seed);updates=[]
    try:
        if algo=='ddqn':
            model=StrictDoubleDQN(37,16,learning_rate=1e-4,gamma=float(runtime['ppo']['gamma']),buffer_size=10000,batch_size=64,net_arch=[64,64],total_timesteps=budget,exploration_final_eps=.05,exploration_fraction=.8,seed=seed)
            obs,_=env.reset()
            for step in range(budget):
                mask=np.asarray(get_action_masks(env),bool);action=model.select_action(obs,mask,step);nxt,reward,term,trunc,_=env.step(action);done=term or trunc
                nxtmask=np.zeros(16,dtype=bool) if done else np.asarray(get_action_masks(env),bool)
                model.add_transition(observation=obs,action=action,reward=reward,next_observation=nxt,done=done,mask=mask,next_mask=nxtmask)
                if step+1>=288 and (step+1)%4==0:
                    u=model.train_step();u['step']=step+1;updates.append(u)
                if (step+1)%288==0:model.sync_target()
                obs=env.reset()[0] if done else nxt
            model.save(folder/'final_model.pt',budget);pd.DataFrame(updates).to_csv(folder/'optimization.csv',index=False)
            reloaded,_=StrictDoubleDQN.load(folder/'final_model.pt');assert all(torch.equal(v,reloaded.online.state_dict()[k]) for k,v in model.online.state_dict().items());del reloaded
        else:
            model=LoggedMaskablePPO('MlpPolicy',env,verbose=0,seed=seed,device='cpu',**engine.base03222.base.ppo_kwargs(runtime));model.set_optimization_log(folder/'optimization.csv');model.learn(total_timesteps=budget);assert model.num_timesteps==budget;model.save(folder/'final_model.zip')
        assert counter[0]==budget
        pd.DataFrame(env.switch_log).to_csv(folder/'training_year_sequence.csv',index=False)
    finally:
        signal.alarm(0);env.close();del env;gc.collect()
    results=[]
    for year in ([2014] if budget==2016 else [2014,2018,2023]):
        resource();_,r,_=rollout(year,f'eval_{year}',model=model);r.update(algo=algo,seed=seed,budget=budget);results.append(r);utils.dump(folder/f'eval_{year}_summary.json',r)
        snap=folder/f'eval_{year}_snapshot/dssat';lines=(snap/'Summary.OUT').read_text().splitlines();h=next(x for x in lines if x.startswith('@'));last=next(x for x in reversed(lines) if x.strip() and x.lstrip()[0].isdigit())
        def val(k):end=h.index(k)+len(k);return float(last[end-5:end])
        assert abs(val('HWAM')-r['yield_kg_ha'])<=1 and val('IRCM')==r['I'] and val('NICM')==r['N'] and r['normal_termination']
    utils.dump(folder/'result.json',dict(status='passed',algo=algo,seed=seed,budget=budget,evaluations=results,seconds=time.time()-start,peak_sampled_rss_MiB=peak[0]/1024**2,optimizer_updates=len(updates) if algo=='ddqn' else model._n_updates))
def worker(algo,seed,budget,tag):
    os.environ['SY_FORECAST_SEED']=str(seed);os.environ.pop('SY_FORECAST_SMOKE',None);sys.path.insert(0,str(ROOT/'src'))
    source=ROOT/'scripts/run_sy_forecast_seed_extension_095.py';code=source.read_text().replace('assert SEED in range(1,8)','assert SEED in range(8)')
    code=code[code.index('def worker(stage):'):code.index("    if stage=='memory_probe':")]
    code+='    train_impl(make,rollout,engine,runtime,ec,SEED,ALGO,BUDGET,OUT,check_observation)\n'
    # Load original module only to obtain reporting globals, never invoke its main.
    ns={'__file__':str(source),'__name__':'isolated132'};full=source.read_text().replace('assert SEED in range(1,8)','assert SEED in range(8)');exec(compile(full,str(source),'exec'),ns)
    folder=OUT/tag;ns.update(OUT=folder,BATCH=OUT,train_impl=train_impl,ALGO=algo,BUDGET=budget);exec(compile(code,str(source),'exec'),ns);ns['worker']('pilot')
def manager():
    import psutil
    OUT.mkdir(exist_ok=False);stop=threading.Event()
    protected=[ROOT/'scripts/run_sy_forecast_seed_extension_095.py',ROOT/'src/strict_maskable_dqn_040.py',ROOT/'results/sy_forecast_seed0_100k_091_20261022/forecast_seed0_ckpt100000.zip'];before={str(p.relative_to(ROOT)):utils.sha(p) for p in protected};utils.dump(OUT/'protected_before.json',before)
    def telemetry():
        while not stop.is_set():
            try:utils.dump(OUT/'host_memory.json',dict(updated_unix=time.time(),available_bytes=psutil.virtual_memory().available,pid=os.getpid()))
            except PermissionError:pass
            stop.wait(2)
    t=threading.Thread(target=telemetry,daemon=True);t.start();completed=[]
    jobs=[('ddqn',0,2016,'smoke_ddqn'),('ppo',0,2016,'smoke_ppo')]+[(algo,seed,10080,f'{algo}_seed{seed}') for seed in range(3) for algo in ['ddqn','ppo']]
    utils.dump(OUT/'frozen_jobs.json',jobs)
    try:
        for algo,seed,budget,tag in jobs:
            if budget>2016:
                for smoke in ['smoke_ddqn','smoke_ppo']:assert json.loads((OUT/smoke/'result.json').read_text())['status']=='passed'
            folder=OUT/tag;folder.mkdir();utils.dump(OUT/'manager_status.json',dict(tag=tag,completed=completed))
            cmd=['docker','exec','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-w','/workspace','nifty_taussig','/opt/gym_dssat_pdi/bin/python','scripts/run_double_dqn_pilot_132.py','--worker',algo,str(seed),str(budget),tag]
            with (folder/'worker.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            assert r.returncode==0,('technical_failure_no_retry',tag)
            completed.append(tag);utils.dump(OUT/'completed.json',completed)
        utils.dump(OUT/'result.json',dict(status='completed',completed=completed))
    except BaseException:utils.dump(OUT/'FAILED.json',dict(traceback=traceback.format_exc(),completed=completed));raise
    finally:
        stop.set();t.join(timeout=3);after={str(p.relative_to(ROOT)):utils.sha(p) for p in protected};utils.dump(OUT/'protected_files_check.json',dict(unchanged=before==after));assert before==after
if __name__=='__main__':
    if '--worker' in sys.argv:worker(sys.argv[-4],int(sys.argv[-3]),int(sys.argv[-2]),sys.argv[-1])
    else:manager()
