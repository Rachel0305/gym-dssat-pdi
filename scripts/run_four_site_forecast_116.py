"""097 isolated four-site forecast transfer, derived from095; serial CPU only."""
from pathlib import Path
import os,sys,json,csv,hashlib,time,signal,tempfile,shutil,subprocess,traceback

ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/'configs/four_site_forecast_116.json').read_text())
SITE=os.environ['FORECAST097_SITE']
SPEC=PLAN['sites'][SITE]
STATION=SPEC['station']
BASE_DIM=SPEC['base_observation_dim']
TRAIN_YEARS=SPEC['train_years']
SEED=int(os.environ.get('FORECAST097_SEED','0'))
SMOKE=os.environ.get('FORECAST097_SMOKE')=='1'
assert SEED in range(8)
BATCH=ROOT/PLAN['output_root']
OUT=BATCH/SITE/('smoke_seed0' if SMOKE else f'seed{SEED}')
REQUESTED=2000 if SMOKE else 100000
ACTUAL=((REQUESTED+143)//144)*144
CHECKPOINTS=[2000] if SMOKE else [25000,50000,75000,100000]
BASELINE=ROOT/SPEC['baseline_runs'][str(SEED)]
STARTED=time.monotonic()
sys.path.insert(0,str(ROOT/'src'))
from sy_forecast_disk_guard_095 import directory_bytes
from forecast_active_clock_101 import ActiveClock
RUN_CLOCK=ActiveClock(OUT)

def dump(name,obj):
    p=OUT/name;p.parent.mkdir(exist_ok=True,parents=True)
    temporary=p.with_name(p.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)),encoding='utf-8')
    os.replace(temporary,p)

def sha(p):
    p=Path(p).resolve();assert p.is_relative_to(ROOT)
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(262144),b''):h.update(b)
    return h.hexdigest()

def watch(stage,limit,seconds):
    import psutil
    assert psutil.virtual_memory().available>=2*1024**3,'Prelaunch guest RAM below2GiB'
    samples=[];start=time.monotonic();clock=ActiveClock(OUT);whole_clock=ActiveClock(OUT)
    prior_active=time.monotonic()-STARTED
    with (OUT/f'{stage}.log').open('w') as log:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',stage],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while p.poll() is None:
                try:tree=[psutil.Process(p.pid),*psutil.Process(p.pid).children(recursive=True)]
                except psutil.Error:tree=[]
                rss=0
                for q in tree:
                    try:rss+=q.memory_info().rss
                    except psutil.Error:pass
                elapsed=clock.elapsed();samples.append(dict(seconds=elapsed,rss_tree_mib=rss/1024**2))
                if rss>limit*1024**2 or elapsed>seconds:raise RuntimeError(f'{stage}: resource limit')
                if RUN_CLOCK.elapsed()>8*3600:raise RuntimeError('Whole batch8-hour limit')
                if len(samples)%10==0:dump('live_resource.json',dict(stage=stage,seconds=elapsed,rss_tree_mib=rss/1024**2,pid=p.pid))
                if len(samples)%60==0 and directory_bytes(OUT)>2*1024**3:raise RuntimeError('Output limit2GiB')
                time.sleep(1)
            assert p.returncode==0,f'{stage} failed; inspect log'
        finally:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            dump(f'{stage}_resources.json',samples)

def worker(stage):
    for p in [ROOT,ROOT/'src']:sys.path.insert(0,str(p))
    tmp=OUT/f'temp_{stage}';tmp.mkdir();tempfile.tempdir=str(tmp)
    from forecast_thread_guard_115 import MemoryPause
    pauser=MemoryPause(BATCH/'host_memory.json',OUT/f'pause_{stage}.json')
    pauser.write(False)
    pauser.checkpoint()
    def flush_training():
        try:training_log.flush()
        except (NameError,UnboundLocalError):pass
    import torch,numpy as np,pandas as pd
    import atexit,gc,weakref
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from ppo_optimization_logging_104 import LoggedMaskablePPO as MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.callbacks import BaseCallback
    import stable_baselines3,zipfile
    with zipfile.ZipFile(ROOT/'results/sy_forecast_action_date_smoke_087_20261022/forecast_seed0_ckpt2000.zip') as z:
        assert stable_baselines3.__version__==z.read('_stable_baselines3_version').decode().strip(),'SB3 version drift since smoke'
    dump(f'{stage}_dependencies.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,stable_baselines3=stable_baselines3.__version__))
    import forecast_engineered_observation_087_action_phase_copy as forecast
    if SITE=="FQ":
        from fq_weather_information_date_116 import last_observed_weather_date,expected_observed_weather
        forecast.last_observed_weather_date=last_observed_weather_date
    import run_141E1_sya_actionable_weather_encoding_smoke2k as e1
    import ppo_safe_rendering
    from sy_forecast_calendar_driver_085 import calendar_date_from_state,next_calendar_window
    # Bind only the COPY; do not patch the original forecast module.
    forecast.FORECAST_FEATURE_NAMES=list(e1.NORM_FEATURES)
    forecast.RAW_FORECAST_FEATURE_NAMES=list(e1.RAW_FEATURES)
    forecast.raw_forecast_features_for_date=e1.raw_actionable_encoding
    forecast.normalize_forecast_features=e1.normalize_actionable_encoding
    engine=forecast.engine
    engine.LOWIC_INPUT_ROOT=ROOT/SPEC['input_root'];engine.STATION=STATION;engine.SITES=[STATION]
    engine.BINARY_IRRIGATION_LEVELS=[0.,15.,30.,45.];engine.BINARY_NITROGEN_LEVELS=[0.,40.,80.,120.]
    engine.base03222.SITE_NAMES[STATION]=SITE;ppo_safe_rendering.MULTISITE_INPUT_ROOT=engine.LOWIC_INPUT_ROOT
    engine.patch_base_module(OUT,OUT/'runtime_record.md',REQUESTED,CHECKPOINTS)
    runtime=engine.load_config();runtime['paths']['output_root']=OUT.relative_to(ROOT).as_posix();runtime['seed']=SEED;runtime['runtime']['smoke_station']=STATION
    split=engine.base03222.load_split();assert sorted(split.loc[split['split'].eq('train'),'year'].astype(int))==TRAIN_YEARS;assert sorted(split.loc[split['split'].eq('validation'),'year'].astype(int))==list(range(2014,2024));selection=engine.base03222.build_selection(split)
    ec=forecast.direct_ppo.build_env_config(runtime,selection)
    cfg=json.loads((ROOT/'configs/143E2_sya_originIC_dual_branch_weather_reader_5k.json').read_text())
    cfg['station_code']=STATION;cfg['site']=SITE;cfg['input_profile']=SPEC['input_profile']
    original_make=engine.base03222.base.make_env
    dump(f'{stage}_runtime.json',runtime);dump(f'{stage}_env_config.json',ec)
    prior=json.loads((ROOT/'results/sy_forecast_action_date_smoke_087_20261022/smoke_runtime.json').read_text())
    for section in ['ppo','reward','action_safety','discrete_actions']:
        assert runtime[section]==prior[section],f'Contract drift:{section}'
    kwargs=engine.base03222.base.ppo_kwargs(runtime)
    metadata=next(m for m in SPEC['baseline_metadata'] if m['seed']==SEED)['ppo']
    assert all(kwargs[k]==v for k,v in metadata.items()),'PPO drift from site baseline archive'
    assert sha(ROOT/SPEC['source_mzx'])==SPEC['source_mzx_sha256']
    for rel,h in PLAN['frozen_sha256'].items():assert sha(ROOT/rel)==h,rel
    dump(f'{stage}_site_contract.json',dict(site=SITE,station=STATION,input_root=str(engine.LOWIC_INPUT_ROOT.relative_to(ROOT)),train_years=TRAIN_YEARS,validation_years=list(range(2014,2024)),effective_ppo=kwargs,appended_weather_features=e1.NORM_FEATURES))
    if stage=='contract':return
    captured=[];oldargs=forecast.direct_ppo.build_env_args
    def capture(*a,**kw):
        result=oldargs(*a,**kw);captured.append(result);dump(f'{stage}_env_args.json',captured);return result
    forecast.direct_ppo.build_env_args=capture
    class ReleasingForecastWrapper(forecast.EngineeredForecastObservationWrapper):
        def close(self):
            raw=self.unwrapped
            result=super().close()
            if raw.closed:
                atexit.unregister(raw._cleanup_process)
            return result
    def make(year,tag,evaluation):
        base=original_make(runtime,ec,STATION,year,SEED,tag,evaluation=evaluation)
        try:
            enhanced=ReleasingForecastWrapper(base,runtime,ec,cfg,STATION,year)
            assert enhanced.base_observation_dim==BASE_DIM
            return enhanced
        except Exception:base.close();raise
    def chain(env):
        result=[]
        while True:
            result.append(type(env).__module__+'.'+type(env).__name__)
            if 'env' not in vars(env):return result
            env=vars(env)['env']
    def check_observation(env,obs):
        pauser.checkpoint(flush_training)
        assert np.asarray(obs).shape==(BASE_DIM+12,) and np.isfinite(obs).all()
        state=env.unwrapped._state;date=calendar_date_from_state(state)
        observed_date=date-pd.Timedelta(days=1)
        assert pd.Timestamp(observed_date)==env.last_observation_date
        window=next_calendar_window({'yrdoy':int(observed_date.strftime('%Y%j'))},env.actual_driver_records)
        assert window[0][0]==date and window[-1][0]==date+pd.Timedelta(days=6)
        actual=env.actual_driver_records[observed_date]
        if SITE=="FQ":actual=expected_observed_weather(actual)
        # Raw weather is previous-day information; yrDOY is the pending action day.
        for k in ['rain','tmin','tmax','srad']:
            assert abs(float(state[k])-actual[k])<=1e-4,(date,k,state[k],actual[k])
        r=np.array([v['rain'] for d,v in window]);tx=np.array([v['tmax'] for d,v in window]);tn=np.array([v['tmin'] for d,v in window]);sr=np.array([v['srad'] for d,v in window]);wet=np.flatnonzero(r>=1)
        raw=dict(rain_next1_mm=float(r[0]),rain_next3_mm=float(sum(r[:3])),rain_next7_mm=float(sum(r)),rain_max_next7_mm=float(max(r)),rain_days_next7=float(sum(r>=1)),first_rain_lead_days=float(wet[0]+1 if len(wet) else 7),tmax_mean_next7_c=float(tx.mean()),tmax_max_next7_c=float(tx.max()),tmin_mean_next7_c=float(tn.mean()),srad_mean_next7_mj_m2=float(sr.mean()),valid_days_next7=7.,dry_spell_next7_flag=float(sum(r)<=5))
        assert all(abs(raw[k]-env.last_raw_forecast_features[k])<=1e-6 for k in raw)
        norm=e1.normalize_actionable_encoding(raw,cfg['forecast_features']['scales'])
        assert np.allclose(obs[BASE_DIM:],[norm[k] for k in e1.NORM_FEATURES],atol=1e-5,rtol=0)
        return date
    def rollout(year,tag,model=None,pulse=False):
        signal.alarm(180);env=None
        try:
            env=make(year,tag,True);dump(f'{tag}_wrapper_chain.json',chain(env))
            assert 'LateIrrigationReserveMaskWrapper' in chain(env)[1]
            obs,info=env.reset();rows=[];saved_obs=[];total_i=total_n=0.
            with (OUT/f'{tag}_steps.jsonl').open('w') as log:
                for step in range(260):
                    date=check_observation(env,obs);mask=np.asarray(get_action_masks(env),dtype=bool)
                    assert mask.shape==(16,) and mask.any()
                    state=dict(env.unwrapped._state)
                    action=5 if pulse and step==0 else 0 if pulse else int(model.predict(obs[:BASE_DIM] if tag.startswith('baseline_') else obs,action_masks=mask,deterministic=True)[0])
                    assert mask[action]
                    current_obs=obs.copy();pre_forecast=dict(env.last_raw_forecast_features)
                    nxt,reward,term,trunc,info=env.step(action)
                    ai=dict(env.last_action_info);post=dict(env.unwrapped._state)
                    postdate=calendar_date_from_state(post)
                    if not term:assert (postdate-date).days==1
                    i=float(ai['safe_action_amir']);n=float(ai['safe_action_anfer']);total_i+=i;total_n+=n
                    terminal_stale_state=bool(term and not trunc and postdate==date)
                    if not terminal_stale_state:
                        assert abs(float(post['totir'])-total_i)<1e-4
                        assert abs(float(post['cumsumfert'])-total_n)<1e-4
                    row=dict(step=step,yrdoy=state['yrdoy'],date=str(date),last_observed_weather_date=str(date-pd.Timedelta(days=1)),post_yrdoy=post['yrdoy'],window_first=str(date),window_last=str(date+pd.Timedelta(days=6)),action=action,I=i,N=n,total_I=total_i,total_N=total_n,pre_totir=state['totir'],post_totir=post['totir'],pre_raw_weather={k:state[k] for k in ['rain','tmax','tmin','srad']},reward=float(reward),terminated=bool(term),truncated=bool(trunc),obs=current_obs.tolist(),mask=mask.tolist(),pre_action_raw_forecast=pre_forecast,post_action_raw_forecast=dict(env.last_raw_forecast_features),post_mask_count=int(ai['mask_valid_action_count']))
                    # Forecast wrapper has now advanced; keep pre-action features explicitly.
                    row.update(requested_I=i,requested_N=n,terminal_stale_state=terminal_stale_state,closure_source='final Summary.OUT required' if terminal_stale_state else 'live state')
                    row['supplied_forecast']=current_obs[BASE_DIM:].tolist()
                    log.write(json.dumps(row,default=lambda v:v.item() if hasattr(v,'item') else str(v))+'\n');log.flush();rows.append(row)
                    if len(saved_obs)<4:saved_obs.append((current_obs,mask.copy(),action))
                    obs=nxt
                    if term or trunc:break
                assert term and not trunc
            snapshot=OUT/f'{tag}_snapshot';snapshot.mkdir()
            rawtmp=Path(env.unwrapped._tmp_folder).resolve();assert rawtmp.is_relative_to(OUT)
            shutil.copytree(rawtmp,snapshot/'dssat')
            summary_lines=(snapshot/'dssat/Summary.OUT').read_text(errors='replace').splitlines()
            summary_header=next(line for line in summary_lines if line.startswith('@'))
            final_record=next(line for line in reversed(summary_lines) if line.strip() and line.lstrip()[0].isdigit())
            def final_value(key):
                end=summary_header.index(key)+len(key)
                return float(final_record[end-5:end])
            final_i=final_value('IRCM');final_n=final_value('NICM')
            assert abs(final_i-total_i)<1e-4,('Final DSSAT irrigation mismatch',final_i,total_i)
            assert abs(final_n-total_n)<1e-4,('Final DSSAT nitrogen mismatch',final_n,total_n)
            dump(f'{tag}_resource_closure.json',dict(status='passed',requested_I=total_i,requested_N=total_n,actual_I=final_i,actual_N=final_n,source='snapshot/dssat/Summary.OUT final record',terminal_live_state_stale=bool(rows[-1]['terminal_stale_state']),reward_unchanged=True))
            latest=forecast.direct_ppo.latest_observation_dict(env.env,obs[:BASE_DIM],info)
            result=dict(year=year,steps=len(rows),yield_kg_ha=forecast.direct_ppo.scalar(latest.get('grnwt',float('nan'))),I=total_i,N=total_n,PFP_N=None if total_n==0 else forecast.direct_ppo.scalar(latest.get('grnwt',float('nan')))/total_n,normal_termination=True)
            dump(f'{tag}_result.json',result)
            return rows,result,saved_obs
        finally:
            signal.alarm(0)
            if env is not None:env.close()
    if stage=='memory_probe':
        import psutil
        refs=[]; evidence=[]
        for cycle in range(18):
            signal.alarm(180)
            env=make(TRAIN_YEARS[cycle%len(TRAIN_YEARS)],f'probe_{cycle}',False)
            try:
                obs,info=env.reset()
                for j in range(8):
                    check_observation(env,obs)
                    obs,rew,term,trunc,info=env.step(0)
                    assert not term and not trunc
            except BaseException:
                env.close();raise
            refs.append(weakref.ref(env.unwrapped))
            env.close();del env;gc.collect()
            alive=sum(r() is not None for r in refs)
            evidence.append(dict(cycle=cycle,alive_closed_instances=alive,rss_mib=psutil.Process().memory_info().rss/1024**2))
            dump('memory_probe_progress.json',evidence)
            assert alive==0,'Closed DSSAT instance retained'
        signal.alarm(0)
        growth=evidence[-1]['rss_mib']-evidence[8]['rss_mib']
        assert growth<32,('RSS growth after warmup',growth)
        dump('memory_gate.json',dict(status='passed',cycles=18,steps=144,all_closed_instances_released=True,growth_after_warmup_mib=growth))
        return
    if stage=='date_probe':
        rows,result,_=rollout(2014,'date_probe2014',pulse=True)
        path=OUT/'date_probe2014_snapshot/dssat/SoilWat.OUT'
        daily={};header=None
        for line in path.read_text().splitlines():
            if line.startswith('@YEAR'):header=line.split();header[0]='YEAR'
            elif header and line.strip() and line.split()[0]=='2014':
                vals=line.split()
                if len(vals)==len(header):daily[int(vals[0])*1000+int(vals[1])]=dict(zip(header,vals))
        assert rows and all(int(r['yrdoy']) in daily for r in rows),'Missing independent action-date rows'
        first=int(rows[0]['yrdoy'])
        assert float(daily[first]['IRRC'])==15.,'SoilWat does not locate irrigation on action-state date'
        assert all(abs(float(daily[int(r['yrdoy'])]['IRRC'])-r['total_I'])<=1e-4 for r in rows if int(r['yrdoy']) in daily)
        evidence=[dict(date=r['date'],action=r['action'],I=r['I'],driver_date=r['yrdoy'],DSSAT_daily_IRRC=float(daily[int(r['yrdoy'])]['IRRC']),post_totir=r['post_totir'],window_first=r['window_first'],window_last=r['window_last']) for r in rows if int(r['yrdoy']) in daily]
        pd.DataFrame(evidence).to_csv(OUT/'action_date_evidence.csv',index=False)
        dump('date_gate.json',dict(status='passed',steps_checked=len(rows),action_applied_on='pre-action raw yrdoy calendar date D',current_weather='raw weather matches D-1; action day weather is not in the base observation',forecast_window='D..D+6, immediately following last observed weather date',pulse_I=15,pulse_N=40,independent_physical_output=str(path.relative_to(ROOT)),snapshot_weather_sha256=sha(next((OUT/'date_probe2014_snapshot/dssat').glob('*.WTH')))))
        return
    if stage=='compat_2000_2014':
        source=ROOT/'results/sy_forecast_action_date_smoke_087_20261022'
        model=MaskablePPO.load(str(source/'forecast_seed0_ckpt2000.zip'),device='cpu')
        rows,result,_=rollout(2014,stage,model=model)
        old=[json.loads(line) for line in (source/'smoke_eval2014_steps.jsonl').open()]
        assert len(rows)==len(old)==144
        for r,b in zip(rows,old):
            assert r['action']==b['action'] and r['I']==b['I'] and r['N']==b['N']
            assert r['mask']==b['mask'] and np.allclose(r['obs'],b['obs'],atol=1e-6,rtol=0)
        expected=json.loads((source/'smoke_eval2014_result.json').read_text())
        assert abs(result['yield_kg_ha']-expected['yield_kg_ha'])<=.5
        assert result['I']==expected['I'] and result['N']==expected['N']
        dump('upgrade_compatibility_gate.json',dict(status='passed',steps=144,actions_inputs_masks_and_doses_equal=True,yield_matches=True,training_started=False))
        return
    if stage.startswith('baseline_'):
        year=int(stage.split('_')[1]);assert year in range(2014,2024)
        path=ROOT/SPEC['baseline_models'][str(SEED)]
        model=MaskablePPO.load(str(path),device='cpu')
        assert model.observation_space.shape==(BASE_DIM,) and model.num_timesteps==100000
        _,result,_=rollout(year,stage,model=model)
        result.update(seed=SEED,model_sha256=sha(path),model_source=str(path.relative_to(ROOT)),steps_file=f'{stage}_steps.jsonl')
        dump(f'{stage}_summary.json',result);return
    if stage.startswith('eval_'):
        _,step,year=stage.split('_');step=int(step);year=int(year)
        assert step in CHECKPOINTS and year in range(2014,2024)
        path=OUT/f'forecast_seed{SEED}_ckpt{step}.zip'
        model=MaskablePPO.load(str(path),device='cpu')
        assert model.num_timesteps==step and model._n_updates==((step-1)//144)*5
        reloaded=MaskablePPO.load(str(path),device='cpu')
        assert all(torch.equal(v,reloaded.policy.state_dict()[k]) for k,v in model.policy.state_dict().items())
        rows,result,examples=rollout(year,stage,model=model)
        for obs,mask,action in examples:assert int(reloaded.predict(obs,action_masks=mask,deterministic=True)[0])==action
        result.update(checkpoint_step=step,model_sha256=sha(path),irrigation_event_count=sum(r['I']>0 for r in rows),nitrogen_event_count=sum(r['N']>0 for r in rows),reward_sum=sum(r['reward'] for r in rows),steps_file=f'{stage}_steps.jsonl')
        dump(f'{stage}_summary.json',result)
        return
    assert stage=='train'
    training_log=(OUT/'training_steps.csv').open('w',newline='')
    training_writer=csv.writer(training_log)
    training_writer.writerow(['step','year','yrdoy','action','legal_count','I','N','reward','terminated','truncated']+[f'obs_{k}' for k in range(BASE_DIM+12)])
    from forecast_recovery_101 import save_bundle
    counter=[0]
    class SingleActiveRandomYear(engine.base03222.RandomYearEnv):
        def _env_for(self,year):
            year=int(year)
            for oldyear in list(self.envs):
                if oldyear!=year:self.envs.pop(oldyear).close();gc.collect()
            if year not in self.envs:self.envs[year]=make(year,f'093_seed{SEED}_train_{year}',False)
            return self.envs[year]
        def reset(self,*args,**kwargs):
            signal.alarm(180)
            self.replay_actions=[]
            return super().reset(*args,**kwargs)
        def step(self,action):
            date=check_observation(self.current_env,self.current_env.last_raw_enhanced_obs)
            vector=self.current_env.last_raw_enhanced_obs.copy()
            mask=np.asarray(self.current_env.action_masks(),dtype=bool)
            requested=int(np.asarray(action).item());assert mask[requested]
            self.replay_actions.append(requested)
            obs,reward,term,trunc,info=super().step(action)
            assert np.isfinite(obs).all() and np.isfinite(reward)
            ai=self.current_env.last_action_info;counter[0]+=1
            training_writer.writerow([counter[0],self.current_year,int(date.strftime('%Y%j')),requested,int(mask.sum()),ai['safe_action_amir'],ai['safe_action_anfer'],float(reward),bool(term),bool(trunc),*vector.tolist()])
            if counter[0]%24==0:training_log.flush()
            if term or trunc:signal.alarm(0)
            return obs,reward,term,trunc,info
    env=None
    class SaveCallback(BaseCallback):
        def _on_rollout_start(self):
            pauser.checkpoint(flush_training)
            if self.model.num_timesteps>0 and self.model.num_timesteps%1008==0:
                training_log.flush()
                save_bundle(OUT/f'recovery_{self.model.num_timesteps}',self.model,env)
        def _on_step(self):
            assert self.model.num_timesteps<=ACTUAL
            if self.model.num_timesteps in CHECKPOINTS:
                self.model.save(str(OUT/f'forecast_seed{SEED}_ckpt{self.model.num_timesteps}.zip'))
                dump(f'checkpoint_{self.model.num_timesteps}_metadata.json',dict(num_timesteps=self.model.num_timesteps,n_updates=self.model._n_updates))
            if self.model.num_timesteps%144==0:
                dump('training_progress.json',dict(num_timesteps=self.model.num_timesteps,n_updates=self.model._n_updates,episodes=len(env.switch_log)))
                training_log.flush()
                pd.DataFrame(env.switch_log).to_csv(OUT/'training_year_switch_log.csv',index=False)
            return True
    try:
        env=SingleActiveRandomYear(runtime,ec,STATION,TRAIN_YEARS,SEED)
        model=MaskablePPO('MlpPolicy',env,verbose=0,seed=SEED,device='cpu',**engine.base03222.base.ppo_kwargs(runtime))
        model.set_optimization_log(OUT/'ppo_optimization_metrics.csv')
        assert model.observation_space.shape==(BASE_DIM+12,)
        dump('formal_policy.json',dict(policy=str(model.policy),parameters=sum(p.numel() for p in model.policy.parameters()),hidden_layers=[64,64],weather_encoding=e1.NORM_FEATURES))
        dump('train_wrapper_chain.json',chain(env.current_env))
        model.learn(total_timesteps=REQUESTED,reset_num_timesteps=True,callback=SaveCallback())
        signal.alarm(0)
        assert model.num_timesteps==ACTUAL and counter[0]==ACTUAL
        model.save(str(OUT/f'forecast_seed{SEED}_final{ACTUAL}.zip'))
        pd.DataFrame(env.switch_log).to_csv(OUT/'training_year_switch_log.csv',index=False)
        pd.DataFrame([dict(year=y,episodes=env.reset_counts[y]) for y in env.years]).to_csv(OUT/'training_year_counts.csv',index=False)
        observed=[r['year'] for r in env.switch_log]
        expected_years=np.random.default_rng(SEED).choice(TRAIN_YEARS,size=len(observed)).tolist()
        assert observed==expected_years,'Independent year RNG sequence changed'
    finally:
        signal.alarm(0);training_log.close()
        if env is not None:env.close()
    dump('training_finished.json',dict(seed=SEED,requested_steps=REQUESTED,actual_final_steps=ACTUAL,main_checkpoint_steps=REQUESTED,main_checkpoint_updates=((REQUESTED-1)//144)*5,independent_year_rng_verified=True,baseline_year_sequence_pairing_claimed=False))

def main():
    if '--worker' in sys.argv:
        try:worker(sys.argv[-1])
        except Exception:
            dump(f'FAILED_{sys.argv[-1]}.json',dict(traceback=traceback.format_exc()));raise
        return
    OUT.mkdir(exist_ok=False,parents=True)
    refs=json.loads((ROOT/'results/sy_forecast_action_date_smoke_087_20261022/protected_sha256_after.json').read_text())
    for rel,h in refs.items():assert sha(ROOT/rel)==h,rel
    protected=[ROOT/rel for rel in refs]
    protected.extend(ROOT/p for p in PLAN['frozen_sha256'])
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    dump('protected_sha256_before.json',before)
    dump('frozen_plan.json',dict(site=SITE,station=STATION,seed=SEED,smoke=SMOKE,requested_steps=REQUESTED,actual_steps=ACTUAL,checkpoints=CHECKPOINTS,train_years=TRAIN_YEARS,validation_years=list(range(2014,2024)),runner_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'configs/four_site_forecast_116.json'),no_performance_gate=True))
    try:
        dump('manager_status.json',dict(status='training',seed=SEED,evaluations_completed=0,pid=os.getpid()))
        watch('contract',768,120)
        if SMOKE:
            watch('memory_probe',768,300)
            watch('date_probe',768,210)
        else:
            gate=json.loads((BATCH/SITE/'smoke_seed0/result.json').read_text());assert gate['status']=='passed_technical_smoke'
            assert json.loads((BATCH/SITE/'smoke_seed0/protected_files_check.json').read_text())['unchanged']
        watch('train',1024,600 if SMOKE else 6*3600)
        with (OUT/'training_steps.csv').open() as f:
            logrows=list(csv.DictReader(f))
        assert len(logrows)==ACTUAL
        assert all(len(row)==10+BASE_DIM+12 and None not in row and int(row['legal_count'])>0 for row in logrows)
        with (OUT/'ppo_optimization_metrics.csv').open() as f:metric_rows=list(csv.DictReader(f))
        assert len(metric_rows)==ACTUAL//144
        assert [int(r['timesteps']) for r in metric_rows]==list(range(144,ACTUAL+1,144))
        assert int(metric_rows[-1]['n_updates'])==(ACTUAL//144)*5
        dump('optimization_log_gate.json',dict(status='passed',rows=len(metric_rows),last_step=int(metric_rows[-1]['timesteps']),last_n_updates=int(metric_rows[-1]['n_updates']),entropy_definition='negative of SB3 entropy_loss'))
        dump('training_log_gate.json',dict(status='passed',rows=len(logrows),base_dim=BASE_DIM,forecast_dim=12,total_obs_dim=BASE_DIM+12))
        del logrows
        completed=[]
        for step in CHECKPOINTS:
            for year in ([2014,2018,2023] if SMOKE else range(2014,2024)):
                stage=f'eval_{step}_{year}'
                dump('manager_status.json',dict(status='evaluating',seed=SEED,stage=stage,evaluations_completed=len(completed),pid=os.getpid()))
                watch(stage,768,210)
                r=json.loads((OUT/f'{stage}_summary.json').read_text());r['seed']=SEED
                completed.append(r);dump('evaluation_summaries_partial.json',completed)
        dump('evaluation_summaries.json',completed)
        if not SMOKE:
            baselines=[]
            for year in range(2014,2024):
                stage=f'baseline_{year}';dump('manager_status.json',dict(status='baseline_replay',seed=SEED,stage=stage,pid=os.getpid()))
                watch(stage,768,210);baselines.append(json.loads((OUT/f'{stage}_summary.json').read_text()));dump('baseline_replays.json',baselines)
        dump('result.json',dict(site=SITE,station=STATION,status='passed_technical_smoke' if SMOKE else 'completed_100k_and_40_evaluations',seed=SEED,smoke=SMOKE,formal_training=not SMOKE,evaluations=len(completed)))
        dump('manager_status.json',dict(status='completed',seed=SEED,evaluations_completed=len(completed)))
    except Exception:
        dump('FAILED.json',dict(traceback=traceback.format_exc(),seed=SEED));dump('manager_status.json',dict(status='failed',seed=SEED));raise
    finally:
        after={str(p.relative_to(ROOT)):sha(p) for p in protected}
        dump('protected_sha256_after.json',after);dump('protected_files_check.json',dict(unchanged=before==after))
        assert before==after,'Protected files changed'

if __name__=='__main__':main()
