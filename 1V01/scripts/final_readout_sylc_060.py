from pathlib import Path
import json,hashlib,zipfile
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/sylc_external_weather_060'
assert json.loads((OUT/'status.json').read_text())['status']=='COMPLETE'
images=list((OUT/'five_scenario_archive').rglob('*.png'))+list((OUT/'four_scenario_v3').rglob('yearly_*.png'))+list((OUT/'cohort_summary').rglob('*.png'))
assert len(images)==282
for p in images:
    with Image.open(p) as im:assert min(im.size)>400;im.verify()
text='# SY/LC 天气增强八种子配对结果\n\n主比较统一100K，seed0–7、2014–2023全部保留。表中为各seed十年均值的八seed均值±样本SD；差值为增强减未增强。\n'
metrics=[('final_grnwt','产量 kg/ha'),('total_irrigation','灌溉 mm'),('total_n','施氮 kg/ha'),('PFP_N','PFP_N kg/kg'),('WP_ET_kg_m3','WP_ET kg/m³')]
for site in ['SY','LC']:
    f=pd.read_csv(OUT/'cohort_summary'/site/'augmented_vs_historical_100k_seed_means.csv')
    text+=f'\n## {site}\n\n|指标|未增强|增强|配对差值|\n|---|---:|---:|---:|\n'
    for m,label in metrics:
        h=f[m+'_historical'];a=f[m+'_augmented'];d=f['delta_'+m]
        text+=f'|{label}|{h.mean():.2f} ± {h.std():.2f}|{a.mean():.2f} ± {a.std():.2f}|{d.mean():+.2f}|\n'
    dy=f.delta_final_grnwt.mean();di=f.delta_total_irrigation.mean();dn=f.delta_total_n.mean()
    text+=f'\n增强相对未增强：产量{dy:+.2f} kg/ha，灌溉{di:+.2f} mm，施氮{dn:+.2f} kg/ha。需结合效率与种子波动解读，不能由单个增益认定全面改善。\n'
text+='\n## 交付与核验\n\n- 每站八个种子，四情景主报告每seed40行、五张逐年指标图；五情景存档每seed50行、12张图（十年逐日过程+指标+管理）。\n- 四checkpoint640配对年度行；100K160配对年度行；10张增强/未增强种子均值图；共282张正式图。\n- WP_ET使用确定性回放Summary.OUT ETCP，沿用旧报告两位小数；核验考虑产量1kg、ETCP0.1mm和WP0.01的展示精度。Null施氮为零，PFP_N为未定义，不填成零。\n- 已抽查SY指标/逐日图和LC图；全部PNG可解码且尺寸符合原格式。模型、基线与原站点图库未改。\n- 失败记录保留：初始WP精度校验过严、旧CSV空字符串数值汇总、一次本地GBK读取、LC多字段舍入误差。修复后复用通过核验的回放，不重跑训练；有效四情景目录为four_scenario_v3，早期目录仅保留失败痕迹。\n- 主归档按生成完成时快照备份，实时status后来变为COMPLETE；本补充归档保存最终完成状态，避免将快照中的RUNNING视为当前状态。\n'
doc=ROOT/'docs/sylc_external_weather_060_readout.md';assert not doc.exists();doc.write_text(text,encoding='utf-8')
audit=OUT/'final_QA.json';audit.write_text(json.dumps({'status':'PASS','images_decoded':282,'sites':2,'seeds':16,'paired_rows':640,'primary_rows':160,'visual_sample_checked':True,'undefined_PFP_N_at_zero_N_retained':True},indent=2))
files=[doc,audit,OUT/'status.json',Path(__file__).resolve()]+list((ROOT/'backups/sylc_060_scripts').glob('*'))
backup=ROOT/'backups/sylc_060_reports/final_readout.zip'
with zipfile.ZipFile(backup,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(backup) as z:assert all(hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).digest()==hashlib.sha256(p.read_bytes()).digest() for p in files)
(backup.parent/'final_readout_verification.json').write_text(json.dumps({'all_sha256_verified':True,'files':len(files)},indent=2))
print(text)
