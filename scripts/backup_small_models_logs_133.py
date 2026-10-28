"""User-authorized small reporting checkpoints and compact training logs."""
from pathlib import Path
import pandas as pd,json,hashlib,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_backups/20261028_weather_ppo_131/models_and_logs';OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=pd.read_csv(ROOT/'results/evidence_forecast_audit_075_20261022/model_identity_metadata_80.csv');lineage=pd.read_csv(ROOT/'results/evidence_forecast_audit_075_20261022/saved_contract_and_input_lineage_80.csv')
models=[dict(site=r.site,regime=r.regime,seed=int(r.seed),path=r.model_path,expected=r.sha256) for r in old.itertuples()]
models += [dict(site=r['site'],regime='forecast',seed=r['seed'],path=r['model'],expected=r['model_sha256']) for r in json.loads((ROOT/'results/forecast_policy_response_127_20261027/frozen_plan.json').read_text())['models']]
copied=[];skipped=[];roots=[]
for r in models:
    p=ROOT/r['path'];assert sha(p)==r['expected']
    assert p.stat().st_size<10*1024**2
    dest=OUT/'models'/f"{r['site']}_{r['regime']}_seed{r['seed']:02d}{p.suffix}";dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);assert sha(dest)==r['expected']
    copied.append(dict(kind='100000_step_reporting_model',source=p.relative_to(ROOT).as_posix(),backup=dest.relative_to(ROOT).as_posix(),sha256=sha(dest),bytes=dest.stat().st_size))
    if r['regime']=='forecast':roots.append(p.parent)
roots += [ROOT/p for p in lineage.run_root]
seen=set()
for base in roots:
    candidates=list(base.glob('*.csv'))+list(base.glob('*train*.log'))+list((base/'logs').glob('*training*.csv'))
    for p in candidates:
        if p in seen:continue
        seen.add(p)
        if not any(k in p.name for k in ['train','optim','episode']):continue
        if p.stat().st_size>2*1024**2:
            skipped.append(dict(source=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,reason='individual training log >2MiB; retained local'));continue
        rel=p.relative_to(ROOT);dest=OUT/'logs'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        copied.append(dict(kind='compact_training_log',source=rel.as_posix(),backup=dest.relative_to(ROOT).as_posix(),sha256=sha(dest),bytes=dest.stat().st_size))
(OUT/'manifest.json').write_text(json.dumps(copied,indent=2));(OUT/'large_logs_local_only.json').write_text(json.dumps(skipped,indent=2))
(OUT/'README_zh.md').write_text('用户明确授权上传体积小的模型与训练日志。这里保存120个论文主报告100000步模型，复制内容与原模型SHA256一致；不收集重复中间检查点。紧凑训练/优化/年份/回合记录同时备份，单个大于2MiB的日志保留本地并列清单。模型按site/regime/seed命名，原始路径见manifest。恢复模型应使用对应代码与依赖；仅加载可信的自有模型。此补充取代131早期“模型均本地保留”的范围说明，原件没有移动或覆盖。',encoding='utf-8')
paths=[p.relative_to(ROOT).as_posix() for p in OUT.rglob('*') if p.is_file()]
for i in range(0,len(paths),40):subprocess.run(['git','add','-f','--',*paths[i:i+40]],cwd=ROOT,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
print(json.dumps(dict(models=sum(x['kind']=='100000_step_reporting_model' for x in copied),compact_logs=sum(x['kind']=='compact_training_log' for x in copied),total_MiB=sum(x['bytes'] for x in copied)/1024**2,large_logs_local=len(skipped))))
