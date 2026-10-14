"""Independent frozen-input, episode, validation and resource smoke gates."""
from pathlib import Path
import json,csv,hashlib,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/sylc_external_weather_058'
sys.path.insert(0,str(ROOT/'src'))
import ppo_safe_rendering as rendering
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
for r in rows(ROOT/'results/sylc_external_weather_057/artifact_manifest.csv'):
    assert sha(ROOT/r['path'])==r['sha256'],r['path']
sites=[];files=[]
for site in ['SY','LC']:
    folder=OUT/site;gate=json.loads((folder/'gate.json').read_text());resource=json.loads((OUT/f'{site}_process.json').read_text())
    assert resource['exit_code']==0 and resource['failure'] is None and resource['peak_tree_rss_mb']<1536
    assert gate['injection_pass'] and gate['result']['smoke_gate']['next_step_allowed']
    assert all(v==[-1] for v in gate['runtime_weather_date_offsets'].values())
    plan=rows(ROOT/f'results/sylc_external_weather_057/{site}/sampling/seed0.csv')
    manifest=json.loads((folder/'weather_manifest.json').read_text())
    training=[r for r in manifest if not r['unused_initialization']]
    assert len([r for r in manifest if r['unused_initialization']])==1
    assert [r['sampling_episode'] for r in training]==list(range(len(training)))
    for r,p in zip(training,plan):
        assert r['year']==int(p['year']) and r['weather_seed']==int(p['weather_seed'])
        assert sha(ROOT/r['bank_WTH'])==sha(ROOT/r['WTH'])==r['source_sha256']==r['WTH_sha256']
    run=ROOT/f'benchmark_results/058{site}_frozen_weather_seed0_smoke_smoke2k'
    switches=rows(run/'logs'/f'032_22_{site}A_training_year_switch_log.csv')
    assert [int(r['year']) for r in switches]==[r['year'] for r in training]
    models=list((run/'models').rglob('*.zip'));assert len(models)==2
    inventory=rows(run/'evaluation/032_22_training_checkpoint_inventory.csv')
    assert {int(r['checkpoint_step']) for r in inventory}=={1000,2000}
    assert all(r['run_status']=='ok' and sha(ROOT/r['model_path'])==r['model_sha256'] for r in inventory)
    original={r['WTH'] for r in manifest};historical=0
    for p in (run/'rendered_inputs').rglob('*.WTH'):
        if p.relative_to(ROOT).as_posix() in original:continue
        year=2000+int(p.stem[4:6]);assert 2014<=year<=2023
        assert sha(p)==sha(rendering.source_weather_path(site+'A',year));historical+=1
    assert historical==20
    cfg=json.loads((folder/'config.json').read_text());frozen=json.loads((ROOT/f'results/sylc_external_weather_057/{site}/configs/seed0.json').read_text())
    for obj in [cfg,frozen]:
        for key in ['task_id','task_name']:obj.pop(key)
    assert cfg==frozen
    sites.append({'site':site,'status':'PASS','requested_PPO_steps':2000,'checkpoints':[1000,2000],'historical_validation_WTH_verified':historical,'weather_states_matched':gate['training_weather_states_matched'],'sampling_episodes_verified':len(training),'weather_seeds_observed':gate['weather_seeds'],'training_years_observed':sorted({r['year'] for r in training}),'peak_rss_mb':round(resource['peak_tree_rss_mb'],2),'seconds':round(resource['seconds'],2)})
    files.extend(p for p in run.rglob('*') if p.is_file())
decision={'status':'PASS_FROZEN_BANK_SMOKE_AND_CONTRACT','sites':sites,'next_step':'SY seed0-7 sequential 100K then LC seed0-7 sequential 100K; retain all seeds and four checkpoints','formal_run_started':False,'formal_preflight_requirement':'recheck frozen hashes; runner must consume frozen sampling without initial-env draw, enforce runtime weather and RSS gate','scope_limit':'2K smoke validates execution and observed behavior, not yield benefit or climate equivalence'}
(OUT/'stage_decision.json').write_text(json.dumps(decision,indent=2))
doc=ROOT/'docs/sylc_external_weather_058_execution.md'
text='# 058 冻结天气库2K smoke执行报告\n\n两站均通过冻结库注入、固定日期偏移、抽样顺序、模型hash与历史验证天气核验，1K/2K模型及20个验证年度结果/站已保存。原核心脚本和原输入未改。正式100K尚未启动。\n\n'
for r in sites:text+=f"- {r['site']}：抽样回合{r['sampling_episodes_verified']}，天气seed {r['weather_seeds_observed']}，峰值{r['peak_rss_mb']}MB，{r['seconds']}秒。\n"
text+='\n实际状态四项天气严格对应yrdoy前一天WTH，1e-4容差，未放宽偏移；此处匹配计数包括初始化与内部重放，不代表PPO步数。训练回合逐行对应057采样计划及原年份切换日志，初始化环境不消费计划。057冻结文件全部hash再验证，smoke配置与冻结配置仅任务标识不同。\n\n下一步可执行正式队列，先SY seed0–7逐个100K，再LC seed0–7。每个种子25/50/75/100K模型均进行2014–2023历史验证，保持单环境与RSS限制。正式收益须汇总全部种子并与未增强同合同结果比较，不能由smoke推断。\n'
doc.write_text(text,encoding='utf-8')
files.extend(p for p in OUT.rglob('*') if p.is_file());files.extend([doc,Path(__file__).resolve(),ROOT/'scripts/run_sylc_frozen_weather_smoke_058.py',ROOT/'prompt_02/058_sylc_frozen_weather_bank_2k_smoke.md'])
files=sorted(set(files));backup=ROOT/'backups/sylc_058_frozen_weather_smoke';backup.mkdir(exist_ok=False)
with zipfile.ZipFile(backup/'smoke_058.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(backup/'smoke_058.zip') as z:assert all(hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).hexdigest()==sha(p) for p in files)
(backup/'verification.json').write_text(json.dumps({'files':len(files),'all_sha256_verified':True,'zip_sha256':sha(backup/'smoke_058.zip')},indent=2))
print(json.dumps(decision,indent=2))
