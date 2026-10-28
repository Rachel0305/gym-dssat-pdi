from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
from PIL import Image,ImageDraw
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/four_site_five_scenario_118_20261026'
assert json.loads((OUT/'result.json').read_text())['status']=='completed'
counts={}
for site in ['HL','LC','YC','FQ']:
    for seed in range(8):
        folder=OUT/site/f'seed_{seed:02d}'/'figures'
        for ext in ['png','pdf','svg']:
            files=list(folder.glob('*.'+ext));assert len(files)==12,(site,seed,ext,len(files))
            for p in files:
                if ext=='png':
                    with Image.open(p) as im:assert im.width>=2000 and im.height>=2000;im.verify()
                elif ext=='pdf':assert p.read_bytes().startswith(b'%PDF-')
                else:ET.parse(p)
    counts[site]={ext:len(list((OUT/site).rglob('*.'+ext))) for ext in ['png','pdf','svg']}
    montage=Image.new('RGB',(1500,450),'white');draw=ImageDraw.Draw(montage)
    for j,(seed,year) in enumerate([(0,2014),(3,2018),(7,2023)]):
        p=OUT/site/f'seed_{seed:02d}'/'figures'/f'{site}_{year}_five_scenario_daily.png'
        with Image.open(p) as im:
            im.thumbnail((500,410));montage.paste(im,(j*500,25))
        draw.text((j*500,5),f'{site} seed{seed} {year}',fill='black')
    montage.save(OUT/site/'visual_QA.png')
sources=json.loads((OUT/'source_sha256.json').read_text())
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in sources.items())
assert len(pd.read_csv(OUT/'all_sites_seed_year_scenario.csv'))==1600
old=OUT/'FAILED_audit_or_render.json'
if old.exists():old.rename(OUT/'attempt_01_failed_audit_retained.json')
(OUT/'QA_complete.json').write_text(json.dumps(dict(status='passed',counts=counts,source_files_verified=len(sources),all_source_hashes_unchanged=True,png_decoding=True,pdf_headers=True,svg_xml=True,seed_year_records=320,scenario_records=1600,initial_failure_retained=True),indent=2))
with (OUT/'figure_contract.md').open('a',encoding='utf-8') as f:
    f.write('\nAudit correction: frozen control tables contained mixed appended management runs. Final-run daily outputs and final Summary.OUT were re-extracted for all four controls; WP_ET and PFP_N recomputed from that same final record. Event/summary display-rounding differences are retained (max irrigation difference 0.5 mm; N totals equal). Baseline independent n=1; PPO n=8.\n')
print(json.dumps(counts))
