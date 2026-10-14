"""Stage only a verified version snapshot, leaving all other workspace changes alone."""
from pathlib import Path
import csv,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
version=sys.argv[1];assert version in ['1V01','2V02']
qa=json.loads((ROOT/version/'FINAL_RELEASE_QA.json').read_text());assert qa['status']=='PASS'
rows=list(csv.DictReader((ROOT/version/'ARCHIVE_MANIFEST.csv').open(encoding='utf-8')))
paths=[version+'/'+r['path'] for r in rows]+[version+'/ARCHIVE_MANIFEST.csv',version+'/FINAL_RELEASE_QA.json']
if version=='1V01':
    paths += [p.relative_to(ROOT).as_posix() for p in ROOT.glob('scripts/*063.py')]
    paths += ['prompt_02/063_two_version_five_site_release.md']
plan=ROOT/f'results/version_release_063/stage_{version}.paths'
plan.write_bytes(b'\0'.join(p.encode('utf-8') for p in sorted(set(paths)))+b'\0')
print('Staging explicit files:',len(set(paths)),flush=True)
subprocess.run(['git','--literal-pathspecs','-c','core.longpaths=true','add','-f','--pathspec-from-file='+str(plan),'--pathspec-file-nul'],cwd=ROOT,check=True)
staged=subprocess.check_output(['git','-c','core.longpaths=true','diff','--cached','--name-only','-z'],cwd=ROOT).decode('utf-8').split('\0')
allowed=set(paths);assert all(p in allowed for p in staged if p),[p for p in staged if p and p not in allowed][:10]
print('STAGE_VERIFIED',len([p for p in staged if p]),flush=True)
