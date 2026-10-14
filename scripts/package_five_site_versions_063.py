"""Copy curated five-site projects, preserving source paths and original outputs."""
from pathlib import Path
import json,csv,shutil,hashlib,subprocess,argparse
ROOT=Path(__file__).resolve().parents[1]
CODE_EXT={'.py','.ps1','.sh','.json','.yaml','.yml','.md','.jinja2','.toml','.txt','.ini'}
DATA_EXT={'.csv','.json','.yaml','.yml','.md','.py','.png','.svg','.CLI','.WTH','.SOL','.CUL','.MZX','.jinja2','.mat','.txt','.OUT'}
OMIT={'__pycache__','cache','tmp','temp','tensorboard','tb_logs','rendered_inputs','runtime','stale_snapshots','failed_attempts'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def roots():
    paths=[]
    for top in ['src','configs','scripts','experiments','Leave_One_experiments/ppo_observed_years']:
        paths.extend((ROOT/top).rglob('*'))
    for top in ['DSSAT_auto_validation/multisite_new_cultivar_inputs_013','DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual','data/external/SYLC_气象站数据']:
        paths.extend((ROOT/top).rglob('*'))
    groups=['results/sy_lc_8seed_extension_231','results/sylc_external_weather_20261012','results/sylc_external_weather_054','results/sylc_external_weather_055','results/sylc_external_weather_057','results/sylc_external_weather_058','results/sylc_external_weather_059','results/sylc_external_weather_060','results/sylc_augmented_strategy_screen_061','results/fq_seed7_completion_20261012','results/fq_historical_seed7_screen_062','results/hl_fq_8seed_cross_site/055_03_five_scenario','results/hla_wgen_8seed_053','results/hla_wgen_8seed_053_no_auto_baseline_231','results/fqa_multiyear_wgen_ppo_051','results/fqa_multiyear_wgen_ppo_052_8seed','results/five_site_strategy_screen_233']
    groups += ['results/yc_random_weather_ppo/'+x for x in ['004_03','004_04','004_05','004_05_review_figures','004_07_02_seed345_etcp_replay','004_15_75mm_all_seed_five_scenario_figures']]
    # Include the training and final frozen baseline output roots actually used.
    names=['046_10_sya_originIC_expanded_action_maskableppo','053_00_lca_lowIC_expanded_action_maskableppo','054_00_hla_lowIC_expanded_action_maskableppo','051_00_fqa_originIC_expanded_action_maskableppo','20261012fq7_fq_seed7_historical_completion','046_06_sya_originIC_046_10_sya_originIC_expanded_action_maskableppo_five_scenario_daily_ckpt100000','053_03_lca_lowIC_053_00_lca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000','054_03_hla_lowIC_054_00_hla_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000','051_03_fqa_originIC_051_00_fqa_originIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000','055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000']
    for site in ['SY','LC']:
        for seed in range(8):
            names.append(f'059{site}s{seed}_frozen_weather_augmented_100k')
            if seed:
                c=json.loads((ROOT/f'results/sy_lc_8seed_extension_231/configs/{site.lower()}_seed{seed}.json').read_text());names.append(f"{c['task_id']}_{c['task_name']}")
    for site in ['HL','FQ']:
        for seed in range(1,8 if site=='HL' else 7):
            c=json.loads((ROOT/f'configs/hl_fq_8seed_cross_site/{site.lower()}_seed{seed}.json').read_text());names.append(f"{c['task_id']}_{c['task_name']}")
    groups+=['benchmark_results/'+n for n in names]
    for group in groups:
        p=ROOT/group;assert p.exists(),p;paths.extend(p.rglob('*'))
    paths.extend(p for p in (ROOT/'docs').glob('*') if p.is_file() and any(s in p.name for s in ['sylc','sy_lc','hla_wgen','fqa_wgen','yc_random_weather','hl_fq','fq_seed7']))
    paths.extend(p for p in ROOT.iterdir() if p.is_file() and (p.suffix in {'.py','.yaml','.yml','.toml'} or p.name=='requirements.txt'))
    return paths
def allowed(p):
    if not p.is_file():return False
    rel=p.relative_to(ROOT)
    if set(rel.parts)&OMIT:return False
    if p.suffix.lower() in {'.pdf','.pptx','.xlsx','.xls','.mp4','.gif','.pyc'}:return False
    if p.suffix=='.zip':return ('models' in p.parts or 'checkpoints' in p.parts) and p.stat().st_size<10*1024**2
    if p.suffix=='.OUT':return p.name.upper() in {'SUMMARY.OUT','PLANTGRO.OUT','SOILWAT.OUT','WEATHER.OUT','MGMTEVENT.OUT'} and ('validation' in p.parts or 'snapshots' in p.parts)
    if rel.parts[0] in ['src','scripts','configs','experiments','Leave_One_experiments']:return p.suffix in CODE_EXT
    return p.suffix in DATA_EXT
def catalog():return sorted(set(p for p in roots() if allowed(p)))
def main(version):
    files=catalog()
    if version=='plan':
        print(json.dumps({'files':len(files),'MiB':sum(p.stat().st_size for p in files)/1024**2,'large_files':[{'path':p.relative_to(ROOT).as_posix(),'MiB':p.stat().st_size/1024**2} for p in files if p.stat().st_size>48*1024**2]},indent=2));return
    import os
    target=Path('\\\\?\\'+str(ROOT/version)) if os.name=='nt' else ROOT/version
    target.mkdir(exist_ok=True);records=[];chunks=[]
    if version=='2V02':
        for top in ['results/version2_yc_wp_et_063_attempt002','results/version2_fq_seed7_corrected2018_063','results/fqa_wgen_052_corrected2018_231','results/hl_fq_8seed_cross_site/055_03_native_auto_corrected2018','benchmark_results/088_00_fq2018_input_quality_correction','benchmark_results/088_00_fq2018_input_quality_correction_run_corrected2018']:
            files+=sorted(p for p in (ROOT/top).rglob('*') if allowed(p))
        files=sorted(set(files))
    for count,p in enumerate(files):
        rel=p.relative_to(ROOT);dest=target/rel;digest=sha(p)
        if p.stat().st_size>48*1024**2:
            parts=[]
            with p.open('rb') as f:
                index=0
                while True:
                    block=f.read(24*1024**2)
                    if not block:break
                    part=target/'data_chunks'/digest/f'{index:04d}.part';part.parent.mkdir(parents=True,exist_ok=True);part.write_bytes(block)
                    parts.append(part.relative_to(target).as_posix());records.append({'path':part.relative_to(target).as_posix(),'bytes':len(block),'sha256':sha(part),'source':rel.as_posix()});index+=1
            chunks.append({'path':rel.as_posix(),'sha256':digest,'parts':parts,'bytes':p.stat().st_size})
        else:
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha(dest)==digest,('Existing copy differs',dest)
            else:shutil.copyfile(p,dest)
            assert sha(dest)==digest
            records.append({'path':rel.as_posix(),'bytes':p.stat().st_size,'sha256':digest,'source':rel.as_posix()})
        if count%2000==0:print(version,'copied',count,'/',len(files),flush=True)
    (target/'large_file_chunks.json').write_text(json.dumps(chunks,indent=2))
    restore='''from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parent
for item in json.loads((root/'large_file_chunks.json').read_text()):
    dest=(root/item['path']).resolve();assert dest.is_relative_to(root)
    if dest.exists():assert hashlib.sha256(dest.read_bytes()).hexdigest()==item['sha256'];continue
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:
        for part in item['parts']:
            with (root/part).open('rb') as src:
                for block in iter(lambda:src.read(1024*1024),b''):f.write(block)
    h=hashlib.sha256()
    with dest.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    assert h.hexdigest()==item['sha256']
print('Large inputs restored and verified')
'''
    (target/'restore_large_files.py').write_text(restore)
    (target/'VERSION.json').write_text(json.dumps({'version':version,'site_count':5,'regimes':['historical','augmented'],'seeds':list(range(8)),'policy_count':80,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'FQ2018':'original frozen weather' if version=='1V01' else 'corrected validation weather; training unchanged','YC_WP_ET':'preserve original partial availability' if version=='1V01' else 'complete exact observed-weather 160-year endpoints','excluded':'large caches/repeated rendered inputs/unrelated research; original source directories remain intact'},indent=2))
    for p in target.iterdir():
        if p.is_file():records.append({'path':p.relative_to(target).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p),'source':'generated'})
    with (target/'SOURCE_MANIFEST.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','sha256','source']);w.writeheader();w.writerows(records)
    (target/'COPY_VERIFICATION.json').write_text(json.dumps({'all_copied_sha256_verified':True,'files':len(records),'chunks':len(chunks),'bytes':sum(r['bytes'] for r in records)},indent=2))
    print(version,'COPY_COMPLETE',len(records),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('version',choices=['plan','1V01','2V02']);a=p.parse_args();main(a.version)
