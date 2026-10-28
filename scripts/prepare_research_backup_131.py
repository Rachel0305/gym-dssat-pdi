"""Curate research backup without staging unrelated changes or large models."""
from pathlib import Path
import ast,json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_backups/20261028_weather_ppo_131';OUT.mkdir(parents=True,exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(262144),b''):h.update(b)
 return h.hexdigest()
tracked=set(subprocess.check_output(['git','ls-tree','-r','HEAD','--name-only'],cwd=ROOT,text=True).splitlines());selected=set();local=[]
scripts=['run_sylc_formal_queue_059.py','audit_evidence_forecast_075.py','assemble_results_tables_074.py','run_sy_forecast_seed_extension_095.py','run_four_site_forecast_116.py','manage_four_site_forecast_116.py','monitor_forecast_host_memory_117.py']
scripts += [p.name for p in (ROOT/'scripts').glob('*.py') if any(f'_{i}' in p.stem for i in range(118,131))]
queue=[ROOT/'scripts'/s for s in scripts];queue+=list((ROOT/'prompts').glob('2026-10-*.md'));queue+=list((ROOT/'configs').glob('*forecast*.json'))
seen=set()
while queue:
 p=queue.pop()
 if not p.exists() or p in seen:continue
 seen.add(p);rel=p.relative_to(ROOT).as_posix()
 if p.suffix in ['.py','.md','.json','.yaml']:selected.add(p)
 if p.suffix=='.py':
  try:tree=ast.parse(p.read_text(encoding='utf-8-sig'))
  except (SyntaxError,UnicodeError):continue
  for node in ast.walk(tree):
   mods=([node.module] if isinstance(node,ast.ImportFrom) else [n.name for n in node.names] if isinstance(node,ast.Import) else [])
   for mod in mods:
    if not mod:continue
    for folder in ['src','scripts','']:
     candidate=ROOT/folder/(mod.replace('.','/')+'.py')
     if candidate.exists():queue.append(candidate)
roots=['paper_results_tables_074_20261022','forecast_analysis_119_20261027','forecast_publication_audit_122_20261027','weather_publication_audit_123_20261027/attempt02','fq2018_generator_audit_124_20261027','fq2018_forecast_reassessment_125_20261027','three_route_results_126_20261027','forecast_policy_response_127_20261027','weather_mechanism_evidence_128_20261027','rain_timing_mechanism_129_20261027','single_decision_value_130_20261027']
for root in roots:
 folder=ROOT/'results'/root
 for p in folder.glob('*'):
  if p.is_file() and p.suffix in ['.csv','.json','.md','.html','.png','.pdf','.svg'] and p.stat().st_size<3*1024**2 and p.name not in ['state_response.csv','host_memory.json','guest_memory.json']:selected.add(p)
# Current five-scenario package: consolidated tables; detailed figure paths stay local.
folder=ROOT/'results/forecast_five_scenario_120_corrected_HL_20261027'
for p in folder.glob('*'):
 if p.is_file() and p.suffix in ['.csv','.json','.md'] and p.stat().st_size<4*1024**2:selected.add(p)
for p in folder.rglob('*.png'):local.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p),storage='local detailed figure; source and consolidated data curated in git'))
for name in ['results/forecast_policy_response_127_20261027/frozen_plan.json','results/evidence_forecast_audit_075_20261022/model_identity_metadata_80.csv','results/evidence_forecast_audit_075_20261022/saved_contract_and_input_lineage_80.csv']:
 p=ROOT/name
 if p.exists():selected.add(p)
modified=set(subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines())
# Snapshot dependency modifications rather than silently including unrelated working-tree edits.
for p in list(selected):
 if p.relative_to(ROOT).as_posix() in modified:
  selected.remove(p);dest=OUT/'dependency_snapshots'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());selected.add(dest)
manifest=[];stage=[]
for p in sorted(selected):
 rel=p.relative_to(ROOT).as_posix();same=False
 if rel in tracked:
  localblob=subprocess.check_output(['git','hash-object','--',rel],cwd=ROOT,text=True).strip();headblob=subprocess.check_output(['git','rev-parse','HEAD:'+rel],cwd=ROOT,text=True).strip();same=localblob==headblob
 manifest.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha(p),already_in_head=same))
 if not same:stage.append(rel)
(OUT/'curated_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');(OUT/'local_detailed_figures_manifest.json').write_text(json.dumps(local,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'README_zh.md').write_text('''# 三路线研究备份 2026-10-28

原PPO/天气增强80策略以已在Git的2V02为基础；新预报40策略及126统一FQ2018版本统计、图表、127响应、128机制讨论、129/130有限实验分别保留来源。最新入口results/three_route_results_126_20261027/index.html与四条结论修订稿；全部原始120图集仍是早期版本，不能与126的修正计数混淆。

结果：120策略20增产、46节资、并集64，77→47→29；描述性事后门槛。预报40模型置换动作响应弱；129阈值规则未通过；130仅局部hindsight改进机会，未学得新策略。FQ2018采用088启发式TMIN质量控制版，不称独立验证天气真值。HL自动对照实际N=0。PFP_N零氮未定义，不能称NUE。

备份包括代码依赖、配置、prompts、年度/策略表、统计、主要PNG/PDF/SVG与说明。已经存在于HEAD且一致的文件不重复提交。模型ZIP/PT、逐日原始DSSAT快照、巨大日志/状态CSV与详细495幅图保留本地；图的路径/大小/SHA256见local_detailed_figures_manifest.json。该清单不代表大文件已上传。完整模型身份索引见075及127 frozen_plan。

源数据/原结果不覆盖，未提交无关的工作区修改；被修改的依赖以dependency_snapshots另存当前版本。所有研究均属模拟条件的探索，不能据此宣称田间部署或预报普遍有效。
''',encoding='utf-8')
stage += [p.relative_to(ROOT).as_posix() for p in OUT.rglob('*') if p.is_file()];stage=sorted(set(stage));(OUT/'stage_paths.json').write_text(json.dumps(stage,ensure_ascii=False,indent=2),encoding='utf-8');stage.append((OUT/'stage_paths.json').relative_to(ROOT).as_posix())
for i in range(0,len(stage),40):subprocess.run(['git','add','-f','--',*stage[i:i+40]],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
print(json.dumps(dict(curated=len(manifest),unchanged=sum(r['already_in_head'] for r in manifest),staged_paths=len(stage),staged_MiB=sum((ROOT/p).stat().st_size for p in stage)/1024**2,local_figures=len(local))))
