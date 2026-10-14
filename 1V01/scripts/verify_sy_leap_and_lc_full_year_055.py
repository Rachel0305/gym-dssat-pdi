"""One SY leap-year check then LC full-year repeatability, after SY quality screen."""
from pathlib import Path
import sys,os,json,tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_sylc_wgen_runtime_smoke_054 as runtime
OUT=ROOT/'results/sylc_external_weather_055'
def main():
    assert json.loads((OUT/'ensemble_gate.json').read_text())['status']=='PASS_FULL_YEAR_ENGINEERING_AND_PHYSICAL_SCREEN'
    runtime.OUT=OUT;os.environ['TMPDIR']=str(OUT/'tmp');tempfile.tempdir=str(OUT/'tmp')
    leap=json.loads((OUT/'leap_year_gate.json').read_text())
    if leap['status']!='PASS':return
    results=[]
    for tag,seed in [('attempt003_seed1001_a',1001),('attempt003_seed1001_b',1001),('attempt003_seed1002',1002)]:
        r=runtime.run_one('LC',tag,seed,True);results.append(r)
        if r['status']!='PASS':break
    gate={'runs':results,'full_year_path_pass':len(results)==3 and all(r['status']=='PASS' for r in results) and results[0]['weather_sha256']==results[1]['weather_sha256'] and results[0]['weather_sha256']!=results[2]['weather_sha256'],'ensemble_quality':'NOT_RUN','PPO_smoke':'NOT_RUN','formal_training_allowed':False}
    (OUT/'LC_path_gate_attempt003.json').write_text(json.dumps(gate,indent=2))
if __name__=='__main__':main()
