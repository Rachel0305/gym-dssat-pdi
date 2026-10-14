"""Audit retained smoke artifacts and make a compact verified backup."""
from pathlib import Path
import json,hashlib,csv,zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/sylc_external_weather_056'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for site in ['SY','LC']:
    directory=OUT/'attempt004'/site
    gate=json.loads((directory/'gate.json').read_text())
    process=json.loads((directory.parent/f'{site}_process.json').read_text())
    assert gate['injection_pass'] and process['exit_code']==0
    assert all(v==[-1] for v in gate['runtime_weather_date_offsets'].values())
    assert gate['result']['smoke_gate']['next_step_allowed']
    run=ROOT/f'benchmark_results/056d{site}_external_weather_seed0_smoke_smoke2k'
    models=list((run/'models').rglob('*.zip'));assert len(models)==2
    checked=0
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    import ppo_safe_rendering as rendering
    manifest=json.loads((directory/'weather_manifest.json').read_text())
    augmented={r['WTH'] for r in manifest}
    for path in (run/'rendered_inputs').rglob('*.WTH'):
        if path.relative_to(ROOT).as_posix() in augmented:continue
        year=2000+int(path.stem[4:6])
        assert 2014<=year<=2023
        assert sha(path)==sha(rendering.source_weather_path(site+'A',year)),path
        checked+=1
    assert checked>=10
    rows.append({'site':site,'weather_states_matched':gate['training_weather_states_matched'],'weather_environments':gate['weather_environment_count'],'weather_seeds':gate['weather_seeds'],'validation_WTH_hashes_matched':checked,'models':len(models),'peak_rss_mb':round(process['peak_tree_rss_mb'],2),'seconds':round(process['seconds'],2),'technical_and_behavior_gate':'PASS','formal_started':False})
summary={'status':'PASS_FIXED_WEATHER_INJECTION_AND_2K_SMOKE','sites':rows,'runtime_weather_date_offset_days':-1,'formal_started':False,'scope':'Two fixed generated weather draws; smoke establishes execution, not training improvement or sufficient diversity for eight seeds.'}
(OUT/'stage_decision.json').write_text(json.dumps(summary,indent=2))
doc=ROOT/'docs/sylc_external_weather_056_execution.md'
text='# 056 SY/LC天气增强注入与2K smoke\n\n'
text+='SY与LC均完成seed0的2K PPO训练，1K/2K两个模型和每个checkpoint十年历史验证；技术与行为门槛均通过。正式八种子训练尚未启动。\n\n'
text+='实际DSSAT返回的四项天气与状态日期前一天的WTH一致；每个环境固定该偏移并逐状态校验四变量，容差1e-4，所有环境最终唯一偏移均为-1。终止空状态不计天气匹配。状态匹配次数包含环境初始化和内部重放，不能当作PPO步数。\n\n'
text+='训练仅使用055的两个固定天气样本1001/1002；闰年使用实际366天样本。验证2014–2023所有新渲染WTH均与冻结历史源文件SHA256一致。原始数据与核心训练脚本未改。\n\n'
for r in rows:text+=f"- {r['site']}: 天气状态匹配{r['weather_states_matched']}次，环境{r['weather_environments']}个，历史验证天气哈希{r['validation_WTH_hashes_matched']}份，峰值{r['peak_rss_mb']}MB，耗时{r['seconds']}秒。\n"
text+='\n失败记录：attempt001初始化读取未赋值_state；attempt002发现日期偏移；attempt003终止空状态访问yrdoy；attempt004完成。误判短暂停顿启动attempt005后已主动终止，该尝试无效，不用于结果。备份保留各版本；后续脚本增加socket超时以限制异常等待。\n\n下一步：冻结正式训练天气库与episode抽样方案，审核历史/增强八种子合同和年份、缓存、日期语义及样本覆盖，再按SY先行的队列执行。当前两个天气样本不足以代表正式增强域，不据2K结果声明收益。\n'
doc.write_text(text,encoding='utf-8')
backup=ROOT/'backups/sylc_056_smoke';backup.mkdir(exist_ok=False)
files=list(OUT.rglob('*'))+[doc,ROOT/'prompt_02/056_sylc_fixed_weather_injection_and_ppo_smoke.md',ROOT/'scripts/run_sylc_augmented_smoke_056.py',Path(__file__).resolve()]
for site in ['SY','LC']:files+=list((ROOT/f'benchmark_results/056d{site}_external_weather_seed0_smoke_smoke2k').rglob('*'))
files=sorted(set(p for p in files if p.is_file()))
with zipfile.ZipFile(backup/'sylc_056_smoke.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(backup/'sylc_056_smoke.zip') as z:
    assert all(hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).hexdigest()==sha(p) for p in files)
(backup/'verification.json').write_text(json.dumps({'files':len(files),'all_sha256_verified':True,'archive_sha256':sha(backup/'sylc_056_smoke.zip')},indent=2))
print(json.dumps(summary,indent=2))
