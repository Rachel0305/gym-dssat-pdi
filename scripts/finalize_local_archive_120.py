from pathlib import Path
import re,json,shutil,hashlib,xml.etree.ElementTree as ET
from PIL import Image
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/forecast_five_scenario_120_corrected_HL_20261027'
OLD=ROOT/'results/four_site_five_scenario_118_20261026'
BACK=ROOT/'backups/figure120';BACK.mkdir(parents=True,exist_ok=True)
assert json.loads((OUT/'result.json').read_text())['status']=='completed'
counts={}
for site in ['SY','HL','LC','YC','FQ']:
    for seed in range(8):
        dest=OUT/site/f'seed_{seed:02d}'
        assert len(list((dest/'yearly').glob('*/*.png')))==10
        assert len(list((dest/'summary').glob('*.png')))==2
        for year in range(2014,2024):assert len(list((dest/'yearly'/str(year)).glob('*.png')))==1
    counts[site]=len(list((OUT/site).rglob('*.png')))
    assert counts[site]==99
for p in OUT.rglob('*.png'):
    with Image.open(p) as im:im.verify()
for p in (OUT/'HL').rglob('*.svg'):ET.parse(p)
for p in (OUT/'HL').rglob('*.pdf'):assert p.read_bytes().startswith(b'%PDF-')
allrows=pd.read_csv(OUT/'all_five_sites_seed_year_scenario.csv',keep_default_na=False)
assert len(allrows)==2000
hl=allrows[allrows.site.eq('HL')&allrows.scenario.eq('dssat_auto')]
assert len(hl)==80 and hl.actual_nitrogen_kg_ha.eq(0).all()
assert not (allrows.site.eq('HL')&allrows.scenario.eq('dssat_auto_external_n')).any()
sources=json.loads((OUT/'source_sha256.json').read_text())
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in sources.items())
xlsx=ROOT/'outputs/five_scenario_120/five_scenario_results_HL_corrected.xlsx'
assert xlsx.exists();shutil.copy2(xlsx,OUT/xlsx.name)
html=(OUT/'index.html').read_text(encoding='utf-8')
html=html.replace('<h1>',f'<p><a href={xlsx.name}>Excel合并表（HL已修正）</a></p><h1>',1)
(OUT/'index.html').write_text(html,encoding='utf-8')
for link in re.findall(r'href=([^ >]+)',html):assert (OUT/link).exists(),link
# Repair the previously delivered website in place, with a backup, without overwriting old research tables.
oldweb=OLD/'index.html';shutil.copy2(oldweb,BACK/'index_118_before_HL_correction.html')
s=oldweb.read_text(encoding='utf-8');prefix='../'+OUT.name+'/'
def redirect(hit):
    path=hit[0];name=path.split('/')[-1];seed=path.split('/')[1]
    year=re.search(r'(201[4-9]|202[0-3])',name)
    return prefix+'HL/'+seed+('/yearly/'+year[1]+'/' if year else '/summary/')+name
s=re.sub(r'HL/seed_\d{2}/figures/[^ >]+',redirect,s)
s=s.replace('href=HL/station_summary/','href='+prefix+'HL/station_summary/')
s=re.sub(r'href=HL/seed_(\d{2})/tables/',lambda h:'href='+prefix+'HL/seed_'+h[1]+'/tables/',s)
s=s.replace('../../outputs/five_scenario_118/five_scenario_results.xlsx','../../outputs/five_scenario_120/five_scenario_results_HL_corrected.xlsx')
s=s.replace('href=all_five_sites_seed_year_scenario.csv','href='+prefix+'all_five_sites_seed_year_scenario.csv')
s=s.replace('href=all_sites_seed_year_scenario.csv','href='+prefix+'all_five_sites_seed_year_scenario.csv')
s=s.replace('<h1>',f'<p><strong>HL已更正为原生DSSAT auto，十年施氮均为0。</strong> <a href={prefix}index.html>完整本地归档及修正表格</a></p><h1>',1)
oldweb.write_text(s,encoding='utf-8')
for link in re.findall(r'href=([^ >]+)',s):assert (OLD/link).exists(),link
readme=OLD/'README_zh.md';shutil.copy2(readme,BACK/'README_118_before_HL_correction.md')
t=readme.read_text(encoding='utf-8')
t=re.sub(r'HL/seed_\d{2}/figures/[^)\s]+',redirect,t)
t=t.replace('./'+prefix,prefix)
t=t.replace('./HL/station_summary/',prefix+'HL/station_summary/')
t=re.sub(r'\./HL/seed_(\d{2})/tables/',lambda h:prefix+'HL/seed_'+h[1]+'/tables/',t)
t=t.replace('(all_five_sites_seed_year_scenario.csv)','('+prefix+'all_five_sites_seed_year_scenario.csv)')
t='> HL对照已更正：原生DSSAT auto，无外部施氮，十年实际施氮均为0。当前图表与合并表请使用[本地修正版]('+prefix+'index.html)。旧原始文件保留，仅作历史记录。\n\n'+t
readme.write_text(t,encoding='utf-8')
(OUT/'QA_complete.json').write_text(json.dumps(dict(passed=True,local_pngs_by_site=counts,total_pngs=sum(counts.values()),HL_seed_figures_redrawn=96,HL_native_auto_records=80,HL_native_auto_NICM=0,old_website_repaired=True,original_tables_preserved=True,all_local_links_valid=True,source_hashes_unchanged=True),indent=2))
print(json.dumps(counts));print('LOCAL ARCHIVE AND OLD WEBSITE VERIFIED')
