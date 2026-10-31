"""Record existing missing-value counts; no estimates or intervals are recomputed."""
from pathlib import Path
import shutil,hashlib,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/section32_figures_20261031_attempt02';BACK=ROOT/'backups/section32_figures136/final_qa';BACK.mkdir(parents=True,exist_ok=False)
table=pd.read_csv(OUT/'table_section32_supplementary.csv');p=pd.read_csv(ROOT/'results/three_route_results_126_20261027/policy_120.csv')
mapping={'Yield':'grain_yield_kg_ha','Irrigation':'total_irrigation_mm','N application':'total_n_kg_ha','PFP_N':'PFP_N','WP_ET':'WP_ET_kg_m3'}
for i,r in table.iterrows():
 for group,route in [('historical','historical'),('augmented','augmented')]:
  q=p[p.site.eq(r.site)&p.regime.eq(route)];table.loc[i,group+'_n_defined_policies']=int(q[mapping[r.metric]].notna().sum()) if r.metric in mapping else 8
file=OUT/'table_section32_supplementary.csv';shutil.copy2(file,BACK/file.name);table.to_csv(file,index=False)
note='PFP_N有效策略数：FQ历史组7/8（seed5零施氮）、YC增强组7/8（seed6零施氮）；其余组8/8。所有其他核心指标均8/8。未定义不补零；因此PFP_N百分比是各组有效策略均值的描述性比较，缺失模式不同，不能直接当作全部8策略整体氮效率改善。'
english='For PFP_N, the FQ historical group and YC augmented group each have seven defined policies: FQ H seed5 and YC A seed6 applied no N and their PFP_N values are undefined. All other groups have eight defined PFP_N policies. Group means and relative changes retain this missingness convention and should not be read as a complete-eight-policy N-efficiency improvement.'
for file,addition in [(OUT/'figure_captions.md',english),(OUT/'table_section32_supplementary.md',english),(OUT/'figure_quality_check.md',note),(ROOT/'docs/2026-10-31_section32_figure_design.md',note)]:
 shutil.copy2(file,BACK/file.name);file.write_text(file.read_text(encoding='utf-8')+'\n\n'+addition+'\n',encoding='utf-8')
pd.read_csv(OUT/'table_section32_supplementary.csv').to_csv(OUT/'plotting_data.csv',index=False)
(OUT/'execution_notes.md').write_text('首版主图色标文字靠近边界；第二版只修正色标标题位置，所有值不变。首版输出保留。两次环境检查分别遇到本地缺fitz及默认GBK解码，改用bundled pypdf与显式UTF-8完成文件格式核验，没有安装软件。已逐页检查4页PPT及2张最终图，PPT包含两张嵌入图片，未做PowerPoint原生应用验证。\n'+note,encoding='utf-8')
shutil.copy2(Path(__file__),OUT/Path(__file__).name)
hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.iterdir() if f.is_file() and f.name!='deliverable_sha256.json'}
(OUT/'deliverable_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
print(json.dumps({'PFP_N_FQ_H':7,'PFP_N_YC_A':7,'statistics_unchanged':True}))
