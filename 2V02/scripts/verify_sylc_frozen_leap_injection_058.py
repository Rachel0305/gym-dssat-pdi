"""Full-calendar diagnostic injection for the leap year absent from 2K RNG prefix."""
from pathlib import Path
import sys,inspect,json,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_sylc_wgen_runtime_smoke_054 as runtime
source=inspect.getsource(runtime.run_one)
start=source.index('    climate=');end=source.index('\n',start)
source=source[:start]+"    climate=ROOT/f'results/sylc_external_weather_057/{site}/bank/2008/seed1001/CN{site}0801.WTH'"+source[end:]
source=source.replace('random_weather=True,seed=None','random_weather=False,seed=0')
source=source.replace("runtime_cli=temp/(station+'.CLI')","runtime_cli=temp/(station+'0801.WTH')")
source=source.replace("runtime_filex_mode(runtime)=='W'","runtime_filex_mode(runtime)=='M'")
source=source.replace("        assert helpers.pdi_seed_configured((temp/'dssat-pdi.yml').read_text(),seed)","        # Generated draw already frozen in WTH; runtime weather RNG not used.")
source=source.replace("'runtime_WTHER':'W'","'runtime_WTHER':'M'")
source=source.replace("    template=directory/'wgen_template.jinja2';template.write_text(text)","    text=text.replace('CN'+site+'0501','CN'+site+'0801')\n    template=directory/'wgen_template.jinja2';template.write_text(text)")
exec(compile(source,__file__,'exec'),runtime.__dict__)
runtime.OUT=ROOT/'results/sylc_external_weather_058/leap_injection_attempt002'
output=[]
for site in ['SY','LC']:
    r=runtime.run_one(site,'frozen2008_seed1001',1001,True,2008);assert r['status']=='PASS',r
    source=ROOT/f'results/sylc_external_weather_057/{site}/bank/2008/seed1001/CN{site}0801.WTH'
    expected={int(f[0]):list(map(float,f[1:5])) for f in (line.split() for line in source.read_text().splitlines()[5:])}
    path=runtime.OUT/site/'runtime/frozen2008_seed1001/snapshot/Weather.OUT'
    lines=path.read_text().splitlines();header=next(s for s in lines if s.startswith('@YEAR')).lstrip('@').split();checked=[]
    for line in lines:
        f=line.split()
        if len(f)!=len(header) or not f[0].isdigit():continue
        row=dict(zip(header,f));token=int(row['YEAR'])*1000+int(row['DOY'])
        actual=[float(row[k]) for k in ['SRAD','TMXD','TMND','PRED']]
        assert actual==expected[token],(token,actual,expected[token]);checked.append(token)
    assert len(checked)==366 and 2008060 in checked
    output.append({'site':site,'days_matched':366,'Feb29_matched':True,'WTHER':'M','WTH_sha256':runtime.digest(source),'peak_rss_mb':r['peak_tree_rss_mb']})
gate={'status':'PASS_FROZEN_366_DAY_RUNTIME_INJECTION','sites':output,'management_output':'DIAGNOSTIC_ONLY; does not replace station PPO results'}
(runtime.OUT/'gate.json').write_text(json.dumps(gate,indent=2));print(json.dumps(gate,indent=2))
