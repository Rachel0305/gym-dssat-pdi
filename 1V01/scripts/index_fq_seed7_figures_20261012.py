"""Retain prior 15-seed gallery and append verified FQ seed7, then back up index."""
from pathlib import Path
import json,zipfile,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/hl_fq_8seed_cross_site/055_03_five_scenario'
BACKUP=ROOT/'backups/fq_seed7_figures_20261012_preexisting'
def main():
    prior=(BACKUP/'GALLERY.md').read_text(encoding='utf-8')
    new=(OUT/'GALLERY.md').read_text(encoding='utf-8')
    pos=new.find('## FQ')
    if pos<0:raise ValueError('Missing FQ gallery section')
    (OUT/'GALLERY.md').write_text(prior.replace('15 个 seed','16 个 seed')+'\n'+new[pos:],encoding='utf-8')
    manifest=json.loads((BACKUP/'figure_completion_manifest.json').read_text())
    addition=json.loads((OUT/'figure_completion_manifest.json').read_text())
    assert len(manifest)==15 and len(addition)==1
    (OUT/'figure_completion_manifest.json').write_text(json.dumps(manifest+addition,indent=2))
    readme=(BACKUP/'README.md').read_text(encoding='utf-8')
    readme=readme.replace('- FQ: PPO seeds 0–6 (7 completed seeds). Seed 7 remains excluded because it did not pass the existing formal smoke gate.','- FQ: PPO seeds 0–7 (8 completed seeds). Seed7 completed in isolated 20261012fq7 attempt; original behavioral smoke failure retained.')
    readme=readme.replace('complete 15-seed','complete 16-seed')
    readme+='\n- [FQ seed7 figures and tables](FQ/best_seed_seed7/README.md): 23 PNG; 10 exact ETCP replays.\n- [FQ eight-seed yearly metrics](FQ/best_seed_seed7/fq_8seed_100k_yearly_metrics.csv): 80 rows, fixed 100K checkpoint.\n'
    (OUT/'README.md').write_text(readme,encoding='utf-8')
    paths=[OUT/n for n in ['GALLERY.md','README.md','figure_completion_manifest.json']]+[Path(__file__).resolve()]
    with zipfile.ZipFile(ROOT/'backups/fq_seed7_figures_20261012/gallery_index.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(ROOT/'backups/fq_seed7_figures_20261012/gallery_index.zip') as z:
        assert z.testzip() is None
        for p in paths:assert hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).digest()==hashlib.sha256(p.read_bytes()).digest()
    print('PASS: 16-seed gallery, 368 PNG total; old per-seed results preserved')
if __name__=='__main__':main()
