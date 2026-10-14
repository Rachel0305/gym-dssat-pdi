"""Minimal official-runtime full-year path trial; stop on first failed gate."""
import os,sys,json,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_sylc_wgen_runtime_smoke_054 as runtime
OUT=ROOT/'results/sylc_external_weather_055'
def main():
    OUT.mkdir(exist_ok=False)
    tmp=OUT/'tmp';tmp.mkdir();os.environ['TMPDIR']=str(tmp);tempfile.tempdir=str(tmp)
    runtime.OUT=OUT
    results=[]
    for tag,seed in [('seed1001_a',1001),('seed1001_b',1001),('seed1002',1002)]:
        r=runtime.run_one('SY',tag,seed,full_year=True);results.append(r)
        if r['status']!='PASS':break
    gate={'runs':results,'full_year_path_pass':len(results)==3 and all(r['status']=='PASS' for r in results) and results[0]['weather_sha256']==results[1]['weather_sha256'] and results[0]['weather_sha256']!=results[2]['weather_sha256'],'formal_training_allowed':False,'PPO_smoke':'NOT_RUN','ensemble_quality':'NOT_RUN'}
    (OUT/'path_gate.json').write_text(json.dumps(gate,indent=2))
    print(json.dumps({'full_year_path_pass':gate['full_year_path_pass'],'attempts':len(results)},indent=2))
if __name__=='__main__':main()
