"""Sequential calendar-correct WGEN bank and reproducible episode contracts."""
import os
os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
from pathlib import Path
import sys,json,csv,calendar,hashlib,datetime,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_sylc_wgen_runtime_smoke_054 as runtime
import ppo_safe_rendering as rendering
OUT=ROOT/'results/sylc_external_weather_057'
OLD=ROOT/'results/sylc_external_weather_055'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def parse(path,year):
    lines=path.read_text().splitlines();header=next(x for x in lines if x.startswith('@YEAR')).lstrip('@').split()
    rows=[]
    for line in lines:
        parts=line.split()
        if len(parts)!=len(header) or not parts[0].isdigit():continue
        r=dict(zip(header,parts));assert int(r['YEAR'])==year
        day=datetime.date(year,1,1)+datetime.timedelta(days=int(r['DOY'])-1)
        rows.append({'DATE':day.isoformat(),'RAIN':float(r['PRED']),'SRAD':float(r['SRAD']),'TMAX':float(r['TMXD']),'TMIN':float(r['TMND'])})
    dates=[(datetime.date(year,1,1)+datetime.timedelta(days=i)).isoformat() for i in range(366 if calendar.isleap(year) else 365)]
    assert [r['DATE'] for r in rows]==dates
    assert all(r['RAIN']>=0 and 0<=r['SRAD']<=45 and -60<=r['TMIN']<=r['TMAX']<=60 for r in rows)
    return rows
def write_csv(path,rows):
    if path.exists():
        existing=list(csv.DictReader(path.open()))
        assert existing==[{k:str(v) for k,v in r.items()} for r in rows],path
        return
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    OUT.mkdir(exist_ok=True);runtime.OUT=OUT
    manifest=[];summaries=[]
    for site in ['SY','LC']:
        bank=OUT/site/'bank';bank.mkdir(parents=True,exist_ok=True)
        samples={};source_records=[]
        for year in [2005,2008]:
            for seed in range(1001,1011):
                if year==2005:
                    tag=('seed1001_a' if site=='SY' else 'attempt003_seed1001_a') if seed==1001 else ('seed1002' if site=='SY' else 'attempt003_seed1002') if seed==1002 else f'seed{seed}'
                    folder=OLD/site/'runtime'/tag
                elif seed==1001:folder=OLD/site/'runtime/leap2008_seed1001'
                else:
                    tag=f'calendar2008_seed{seed}'
                    prior=OUT/site/'runtime'/tag/'result.json'
                    result=json.loads(prior.read_text()) if prior.exists() else runtime.run_one(site,tag,seed,full_year=True,target_year=2008)
                    assert result['status']=='PASS',result
                    folder=OUT/site/'runtime'/tag
                source=folder/'snapshot/Weather.OUT'
                rows=parse(source,year);samples[(calendar.isleap(year),seed)]=rows
                corrected=bank/f'calendar{year}_seed{seed}.csv';write_csv(corrected,rows)
                source_records.append({'site':site,'calendar_year':year,'weather_seed':seed,'source':str(source.relative_to(ROOT)),'source_sha256':sha(source),'corrected_csv':str(corrected.relative_to(ROOT)),'corrected_sha256':sha(corrected),'days':len(rows)})
        for year in range(2005,2014):
            for seed in range(1001,1011):
                rows=samples[(calendar.isleap(year),seed)]
                folder=bank/str(year)/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
                path=folder/f'{rendering.target_weather_stem(site+"A",year)}.WTH'
                header=rendering.source_weather_path(site+'A',year).read_text().splitlines()[:5]
                body=[f'{year}{i:03d} {r["SRAD"]:5.1f} {r["TMAX"]:5.1f} {r["TMIN"]:5.1f} {r["RAIN"]:5.1f}' for i,r in enumerate(rows,1)]
                content='\n'.join(header+body)+'\n'
                if path.exists():assert path.read_text()==content,path
                else:path.write_text(content)
                manifest.append({'site':site,'year':year,'weather_seed':seed,'days':len(rows),'WTH':str(path.relative_to(ROOT)),'sha256':sha(path)})
        write_csv(OUT/site/'source_manifest.csv',source_records)
        configs=OUT/site/'configs';configs.mkdir(exist_ok=True)
        plans=OUT/site/'sampling';plans.mkdir(exist_ok=True)
        for ppo_seed in range(8):
            cfg=json.loads((ROOT/f'results/sylc_external_weather_056/attempt004/{site}/config.json').read_text())
            cfg['seed']=ppo_seed;cfg['task_id']=f'057{site}s{ppo_seed}';cfg['task_name']='external_weather_augmented_100k'
            config_path=configs/f'seed{ppo_seed}.json';config_path.write_text(json.dumps(cfg,indent=2))
            years_rng=np.random.default_rng(ppo_seed);weather_rng=np.random.default_rng(57000+ppo_seed)
            buckets={False:[],True:[]};plan=[]
            for episode in range(2048):
                year=int(years_rng.choice(list(range(2005,2014))));leap=calendar.isleap(year)
                if not buckets[leap]:buckets[leap]=weather_rng.permutation(np.arange(1001,1011)).tolist()
                seed=int(buckets[leap].pop());plan.append({'episode':episode,'year':year,'weather_seed':seed,'leap':leap})
            assert set(r['year'] for r in plan)==set(range(2005,2014))
            for leap in [False,True]:
                values=[r['weather_seed'] for r in plan if r['leap']==leap]
                for i in range(0,len(values)-9,10):assert set(values[i:i+10])==set(range(1001,1011))
            write_csv(plans/f'seed{ppo_seed}.csv',plan)
        summaries.append({'site':site,'source_calendars':20,'mapped_WTH_files':90,'sampling_plans':8,'steps':100000,'checkpoints':[25000,50000,75000,100000],'train_years':list(range(2005,2014)),'validation_years':list(range(2014,2024))})
        (OUT/'progress.json').write_text(json.dumps(summaries,indent=2))
    write_csv(OUT/'weather_manifest.csv',manifest)
    gate={'status':'PASS_CALENDAR_CORRECT_BANK_AND_SAMPLING_AUDIT','sites':summaries,'authoritative_calendar':'DSSAT Weather.OUT YEAR/DOY','previous_055_state_csv':'ONE_DAY_LATE_LABEL; retained; do not use for formal bank','rounding':'native Weather.OUT one decimal, matching WTH','weather_rng':'57000+PPOseed, shuffled cycles per calendar type','unused_initial_environment':'must not consume sampling plan','formal_allowed':False,'next_gate':'2K smoke against this frozen bank, runtime and calendar checks, exact frozen-contract audit'}
    (OUT/'stage_decision.json').write_text(json.dumps(gate,indent=2))
    doc=ROOT/'docs/sylc_external_weather_057_execution.md'
    doc.write_text('# 057 正式天气库与八种子抽样合同\n\n以DSSAT Weather.OUT的YEAR/DOY为日历依据，避免055旧状态CSV晚标一天。原结果保留。每站20个源日历（10平年、10真实闰年），映射2005–2013共90份WTH；各站seed0–7独立天气RNG、每类日历10样本打乱无放回循环，2048回合计划已生成并检查覆盖。历史年份抽样保持原算法。\n\n配置继承056所用站点合同并恢复100K与25/50/75/100K checkpoints，验证2014–2023历史天气。配置是下一阶段审核草案，未运行正式训练。天气库采用Weather.OUT原生0.1精度；10个样本不证明气候分布等价。下一步须在此库上重新2K smoke并核对合同，届时才能形成正式执行门槛。\n',encoding='utf-8')
    files=sorted(p for p in OUT.rglob('*') if p.is_file())+[doc,Path(__file__).resolve(),ROOT/'prompt_02/057_sylc_formal_weather_bank_and_sampling_contract.md']
    records=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in files];write_csv(OUT/'artifact_manifest.csv',records)
    backup=ROOT/'backups/sylc_057_weather_bank';backup.mkdir(exist_ok=False)
    with zipfile.ZipFile(backup/'weather_bank_057.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[OUT/'artifact_manifest.csv']:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(backup/'weather_bank_057.zip') as z:assert all(hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256'] for r in records)
    (backup/'verification.json').write_text(json.dumps({'all_sha256_verified':True,'files':len(files)+1,'zip_sha256':sha(backup/'weather_bank_057.zip')},indent=2))
    print(json.dumps(gate,indent=2))
if __name__=='__main__':main()
