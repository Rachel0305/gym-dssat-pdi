"""Independent source-contract and reproducibility audit of frozen inputs."""
from pathlib import Path
import json,csv,hashlib,calendar
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/sylc_external_weather_057'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
records=[]
for site,original in [('SY','046_10_sya_originIC_expanded_action_maskableppo.json'),('LC','053_00_lca_lowIC_expanded_action_maskableppo.json')]:
    baseline=ROOT/'configs'/original;base=json.loads(baseline.read_text())
    for seed in range(8):
        path=OUT/site/'configs'/f'seed{seed}.json';cfg=json.loads(path.read_text())
        left=json.loads(json.dumps(base));right=json.loads(json.dumps(cfg))
        for key in ['task_id','task_name','seed']:left.pop(key,None);right.pop(key,None)
        left.get('paths',{}).pop('output_root',None);right.get('paths',{}).pop('output_root',None)
        assert left==right,(site,seed,'contract difference')
        assert cfg['training']['total_timesteps']==100000
        assert cfg['training']['checkpoint_steps']==[25000,50000,75000,100000]
        yr=np.random.default_rng(seed);wr=np.random.default_rng(57000+seed);buckets={False:[],True:[]}
        plan=list(csv.DictReader((OUT/site/'sampling'/f'seed{seed}.csv').open()))
        for episode,row in enumerate(plan):
            year=int(yr.choice(range(2005,2014)));leap=calendar.isleap(year)
            if not buckets[leap]:buckets[leap]=wr.permutation(np.arange(1001,1011)).tolist()
            weather=int(buckets[leap].pop())
            assert int(row['episode'])==episode and int(row['year'])==year and int(row['weather_seed'])==weather
        records.append({'site':site,'PPO_seed':seed,'contract_matches_original_except_identity_output_seed':True,'original_config_sha256':sha(baseline),'config_sha256':sha(path),'sampling_reproduced_episodes':len(plan)})
    samples=list(csv.DictReader((OUT/site/'source_manifest.csv').open()))
    assert len({r['corrected_sha256'] for r in samples if r['calendar_year']=='2005'})==10
    assert len({r['corrected_sha256'] for r in samples if r['calendar_year']=='2008'})==10
result={'status':'PASS','contracts':records,'formal_allowed':False,'corrected_bank_smoke_required':True,'bank_size_notes':'10 samples per calendar, not a claim of sufficient climatological diversity'}
(OUT/'independent_contract_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'status':'PASS','contracts':len(records),'episodes_reproduced':sum(r['sampling_reproduced_episodes'] for r in records)},indent=2))
