"""Sequential site-specific WGEN season smoke with runtime evidence and fail-stop."""
import os
os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
from pathlib import Path
import sys,json,copy,shutil,re,time,hashlib,argparse
from datetime import date
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_yc_wgen_seed_pilot as helpers
OUT=ROOT/'results/sylc_external_weather_054'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run_one(site,tag,seed,full_year=False,target_year=2005):
    import gym
    import gym_dssat_pdi.envs.dssat_pdi as module
    directory=OUT/site/'runtime'/tag; directory.mkdir(parents=True,exist_ok=False)
    source=ROOT/('benchmark_results/231_sy_seed7_sya_originIC_expanded_action_maskableppo_seed7_extension/rendered_inputs/SYA/2005/SYA_2005_032_22_train/SYA_2005_SYA_2005_032_22_train.jinja2' if site=='SY' else 'benchmark_results/231_lc_seed7_lca_lowIC_expanded_action_maskableppo_seed7_extension/rendered_inputs/LCA/2005/LCA_2005_032_22_train/LCA_2005_LCA_2005_032_22_train.jinja2')
    inputs=ROOT/('DSSAT_auto_validation/multisite_new_cultivar_inputs_013/SY' if site=='SY' else 'DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/LC')
    climate=ROOT/'results/sylc_external_weather_20261012/processed_stage/CNSY.CLI' if site=='SY' else ROOT/'results/sylc_external_weather_054/LC_v2/CNLC.CLI'
    station='CN'+site
    text=source.read_text()
    if full_year:
        text=text.replace('\x1a','')
        text=re.sub(r'(?ms)^\*HARVEST DETAILS\s*\n.*?(?=^\*|\Z)','',text)
        text=re.sub(r'(?m)^(\s*1\s+GE\s+\d+\s+\d+\s+\S+\s+)\d{5}',r'\g<1>05001',text,count=1)
        text=re.sub(r'(?m)^(\s*1\s+MZ\s+)\d{5}',r'\g<1>05001',text,count=1)
        text=re.sub(r'(?m)^(\s*1\s+MA\s+\S+\s+\S+\s+\S+\s+\S+\s+)\S+',r'\g<1>R',text,count=1)
        lines=text.splitlines()
        for i,line in enumerate(lines):
            if re.match(r'^\s*1\s+1\s+1\s+0\s+Sim2005',line):
                lines[i]=re.sub(r'\s+0\s+1\s*$', '  1  1',line)
        text='\n'.join(lines)+'\n'
        text=text.replace('*SIMULATION CONTROLS','*HARVEST DETAILS\n@H HDATE  HSTG  HCOM  HSIZ   HPC  HBPC HNAME\n 1 05365 GS000   -99   -99   100     0 FullYearDiagnostic\n\n*SIMULATION CONTROLS',1)
        if target_year!=2005:
            import calendar
            text=text.replace('2005',str(target_year))
            text=re.sub(r'(?<!\d)05(\d{3})(?!\d)',lambda m:f'{target_year%100:02d}'+m.group(1),text)
            if calendar.isleap(target_year):text=text.replace(f'{target_year%100:02d}365 GS000',f'{target_year%100:02d}366 GS000')
    text,count=re.subn(r'(?m)^(\s*1\s+ME\s+)\S+',r'\g<1>{{ wther }}',text,count=1)
    assert count==1
    template=directory/'wgen_template.jinja2';template.write_text(text)
    cls=helpers._runtime_class(module); old_get=cls._get_state; old_sockets=cls._get_sockets_
    states=[]; context={}; env=None; start=time.monotonic();peak=0
    def capture(instance):
        result=old_get(instance)
        if isinstance(getattr(instance,'_state',None),dict):states.append(copy.deepcopy(instance._state))
        return result
    def sockets(instance):
        instance._rseed1=seed;context['instance']=instance
        result=old_sockets(instance)
        if getattr(instance,'_server',None) is not None:
            import zmq
            instance._server.setsockopt(zmq.RCVTIMEO,30000)
        return result
    def snapshot(instance):
        if not instance:return
        target=directory/'snapshot';target.mkdir(exist_ok=True)
        for p in Path(instance._tmp_folder).iterdir():
            if p.is_file():shutil.copy2(p,target/p.name)
    cls._get_state=capture;cls._get_sockets_=sockets
    result={'site':site,'tag':tag,'weather_seed':seed,'scope':'full_year_diagnostic' if full_year else 'season_only','status':'FAILED'}
    try:
        env=gym.make('gym_dssat_pdi:GymDssatPdi-v0',log_saving_path=str(directory/'dssat.log'),mode='all',auxiliary_file_paths=[str(inputs/'MZCER048.CUL'),str(inputs/'SOIL.SOL'),str(climate)],random_weather=True,seed=None,fileX_template_path=str(template),experiment_number=1,evaluation=True,cultivar='maize',run_dssat_location='/opt/dssat_pdi/run_dssat').unwrapped
        temp=Path(env._tmp_folder); runtime_cli=temp/(station+'.CLI')
        assert digest(runtime_cli)==digest(climate)
        runtime=(temp/'fileX.MZX').read_text()
        assert helpers.runtime_filex_mode(runtime)=='W'
        assert helpers.pdi_seed_configured((temp/'dssat-pdi.yml').read_text(),seed)
        steps=0
        while not env.done and steps<400:
            env.step({k:0. for k in env.action_variables});steps+=1
            peak=max(peak,helpers._process_tree_rss_mb())
            if peak>1536 or time.monotonic()-start>300:raise RuntimeError('Resource limit exceeded')
        assert env.done
        snapshot(env)
        # Date reconstruction uses simulation start only as fallback. Preserve
        # raw state for calendar semantics audit rather than claiming full year.
        sim=re.search(r'(?m)^\s*1\s+GE\s+\d+\s+\d+\s+\S+\s+(\d{5})',runtime)
        import datetime
        startdate=datetime.datetime.strptime(sim.group(1),'%y%j').date()
        rows=helpers.daily_weather_from_states(states,startdate)
        if full_year:
            rows=[]
            for state in states:
                token=int(state['yrdoy'])
                day=datetime.datetime.strptime(str(token),'%Y%j').date()
                rows.append({'DATE':day.isoformat(),'DOY':day.timetuple().tm_yday,**{v:float(state[v.lower()]) for v in ['RAIN','SRAD','TMAX','TMIN']}})
        canonical=helpers.canonical_weather_bytes(rows)
        (directory/'weather.csv').write_bytes(canonical)
        (directory/'raw_states.json').write_text(json.dumps(states,default=str))
        qc=helpers.physical_sanity(rows)
        result.update({'status':'PASS' if qc['status']=='PASS' else 'WEATHER_QC_FAILED','steps':steps,'rows':len(rows),'physical_qc':qc,'cli_sha256':digest(runtime_cli),'runtime_WTHER':'W','runtime_seed_verified':True,'weather_sha256':hashlib.sha256(canonical).hexdigest()})
        if full_year:
            import calendar
            expected=[(date(target_year,1,1)+datetime.timedelta(days=i)).isoformat() for i in range(366 if calendar.isleap(target_year) else 365)]
            result['full_year_dates_complete']=[r['DATE'] for r in rows]==expected
            result['actual_start_date']=rows[0]['DATE'] if rows else None
            result['actual_end_date']=rows[-1]['DATE'] if rows else None
            if not result['full_year_dates_complete']:result['status']='FULL_YEAR_INCOMPLETE'
    except Exception as exc:
        snapshot(env or context.get('instance'));result['error']=f'{type(exc).__name__}: {exc}'
    finally:
        result.update({'elapsed_seconds':time.monotonic()-start,'peak_tree_rss_mb':peak})
        (directory/'result.json').write_text(json.dumps(result,indent=2))
        if env:env.close()
        cls._get_state=old_get;cls._get_sockets_=old_sockets
    print(json.dumps(result,indent=2),flush=True)
    return result
def main(site):
    results=[]
    for tag,seed in [('seed1001_a',1001),('seed1001_b',1001),('seed1002',1002)]:
        r=run_one(site,tag,seed);results.append(r)
        if r['status']!='PASS':break
    repeat=len(results)==3 and results[0].get('weather_sha256')==results[1].get('weather_sha256')
    different=len(results)==3 and results[0].get('weather_sha256')!=results[2].get('weather_sha256')
    gate={'site':site,'runs':results,'same_seed_reproduced':repeat,'different_seed_changed':different,'season_runtime_pass':len(results)==3 and all(r['status']=='PASS' for r in results) and repeat and different,'full_year_QC':'NOT_RUN','PPO_smoke':'NOT_RUN','formal_training_allowed':False}
    (OUT/site/'runtime_gate.json').write_text(json.dumps(gate,indent=2))
    if not gate['season_runtime_pass']:raise RuntimeError('Season smoke failed; evidence retained')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--site',choices=['SY','LC'],required=True);main(parser.parse_args().site)
