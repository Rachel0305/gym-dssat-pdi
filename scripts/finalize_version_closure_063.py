"""Supplement compact reproduction dependencies; index all release files."""
from pathlib import Path
import json,hashlib,subprocess,csv,sys,importlib.metadata
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main(version):
    target=ROOT/version;qa=json.loads((target/'RELEASE_QA.json').read_text());assert qa['status']=='PASS'
    entries=[];sources=[]
    # Dynamic weather-gate imports live in results; copy source code without caches.
    pyfiles=(ROOT/'results/version_release_063/dynamic_source_paths.txt').read_text(encoding='utf-8').splitlines()
    sources += [ROOT/p for p in pyfiles if not any(x in Path(p).parts for x in ['__pycache__','cache','tmp','temp','rendered_inputs','runtime'])]
    for top in ['weather_clean','results/hla_weather_enhancement_029','results/fqa_weather_resume_041','results/fqa_wgen_multiyear_heldout_gate_045','results/fqa_wgen_ppo_smoke_044','results/yc_wgen_cli_pilot/003_06_05_02','results/yc_weather_gapfill_finalize','results/yc_random_weather_episode_validation/004_02']:
        folder=ROOT/top;assert folder.exists(),folder
        sources += [p for p in folder.rglob('*') if p.is_file() and p.suffix in ['.py','.csv','.json','.CLI','.WTH','.SOL','.CUL','.MZX','.jinja2','.yaml','.yml','.md'] and not any(x in p.relative_to(ROOT).parts for x in ['cache','tmp','temp','runtime','rendered_inputs','__pycache__'])]
    # Exact YC validators require only these frozen evaluation input files.
    for group in ['historical','random_weather']:
        for seed in range(8):
            parent=ROOT/('results/yc_random_weather_ppo/004_03/evaluation/verified_weather_refresh/observed_weather' if seed<=2 else 'results/yc_random_weather_ppo/004_05/evaluation/runs/observed_weather')/group/f'ppo_seed_{seed}'/'rendered_inputs'
            assert parent.exists(),parent
            sources += [p for p in parent.rglob('*') if p.is_file() and p.suffix in ['.WTH','.jinja2']]
    for p in sorted(set(sources)):
        rel=p.relative_to(ROOT);dest=target/rel
        if dest.exists():assert sha(dest)==sha(p),rel
        else:
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
            entries.append({'path':rel.as_posix(),'bytes':dest.stat().st_size,'sha256':sha(dest)})
    # Capture the verified replay runtime, while keeping system binaries external.
    libs={}
    for name in ['torch','stable-baselines3','sb3-contrib','gym','gymnasium','numpy','pandas','matplotlib','scipy','PyYAML','psutil']:
        try:libs[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:libs[name]='not_available'
    (target/'REPLAY_ENVIRONMENT.json').write_text(json.dumps({'Python':sys.version,'libraries':libs,'device':'CPU','DSSAT':'4.8.0.024 stable; external installed runtime','scope':'verified current replay environment; not a claim of original training lockfile'},indent=2))
    images=list((target/'deliverables').glob('*/*/seed_*/figures/*.png'));assert len(images)==400
    for p in images:
        with Image.open(p) as im:assert min(im.size)>400;im.verify()
    registry=pd.read_csv(target/'MODEL_REGISTRY.csv');assert len(registry)==80
    for r in registry.itertuples():assert sha(target/r.model_path)==r.model_sha256
    f=pd.read_csv(target/'deliverables/all_80_policy_800_year_metrics.csv');assert len(f)==800 and not f.duplicated(['site','regime','seed','year']).any()
    counts=f.groupby(['site','regime','seed'])[['grain_yield_kg_ha','PFP_N','WP_ET_kg_m3']].count();counts.to_csv(target/'METRIC_DEFINED_YEAR_COUNTS.csv')
    if version=='2V02':assert f.WP_ET_kg_m3.notna().all()
    (target/'CLOSURE_NOTES.md').write_text('''# Reproduction scope and version safeguards

Compact weather fitting CLI/processed datasets, dynamic result-folder Python helpers, and the frozen YC evaluation WTH/templates required by hash validators are included. The installed DSSAT executable and Python environment are external; REPLAY_ENVIRONMENT.json records the current verified CPU replay versions.

deliverables is the active 100K result set. Original five-scenario/daily archives remain under results, and may include old FQ2018 or alternative automatic-irrigation reports. Do not substitute these archive panels for the active 2V02 corrected tables. Historical selection tables are retained as provenance; they have not been silently recomputed using revised endpoints.

YC exact WP_ET completion covers observed-weather 2014-2023 for both eight-seed regimes. Supplemental heldout-WGEN episode outputs are retained in their original form; their WP_ET was not added in this release. Undefined PFP_N when N=0 remains undefined. V01 retains the original partial YC efficiency coverage.

Large CSVs are stored as byte-exact chunks for GitHub transport. Run restore_large_files.py after checkout before workflows that consume those logical source CSV paths; sha256 is checked automatically. Unrelated research, repeated input copies, system executables and large failed caches are not part of the curated release.

The code is a source snapshot of the current workspace; the model/source manifests and release hashes are authoritative. Some extra helper scripts refer to unrelated historical experiments; this package audits the listed 80 policies, not every exploratory script in the repository.
''',encoding='utf-8')
    pd.DataFrame(entries).to_csv(target/'SUPPLEMENT_MANIFEST.csv',index=False)
    # Preserve the already verified hashes instead of re-reading 1.8 GB twice.
    original=pd.read_csv(target/'RELEASE_MANIFEST.csv').to_dict('records');known={r['path']:r for r in original}
    known.update({r['path']:r for r in entries})
    for p in target.iterdir():
        if p.is_file() and p.name not in ['ARCHIVE_MANIFEST.csv','FINAL_RELEASE_QA.json']:
            known[p.name]={'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)}
    pd.DataFrame(sorted(known.values(),key=lambda r:r['path'])).to_csv(target/'ARCHIVE_MANIFEST.csv',index=False)
    (target/'FINAL_RELEASE_QA.json').write_text(json.dumps({'status':'PASS','models':80,'primary_year_rows':800,'primary_png_verified':400,'YC_exact_WP_ET_years':int(f[f.site.eq('YC')].WP_ET_kg_m3.notna().sum()),'compact_dependencies_added':len(entries),'archive_files':len(known),'large_file_chunk_count':len(json.loads((target/'large_file_chunks.json').read_text())),'zero_N_PFP_N_not_fabricated':True},indent=2))
    print(version,'FINAL_QA_PASS',flush=True)
if __name__=='__main__':main(sys.argv[1])
