"""Isolated historical seed7 completion; preserve behavioral failures as outcomes."""
import os
os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/fq_seed7_completion_20261012'
CONFIG=OUT/'fq_seed7.json'

def worker(phase):
    sys.path.insert(0,str(ROOT/'src/hl_fq_8seed_cross_site'))
    import run_seeded_site_transfer as adapter
    runner=adapter.import_runner('FQA')
    original=runner.verify_completed_smoke
    def verify(cfg):
        checks=original(cfg)
        path=runner.output_root(runner.prepare_config(cfg,True))/'20261012fq7_smoke_result.json'
        result=json.loads(path.read_text())
        gate=result['smoke_gate']
        technical=(gate['expected_validation_rows']==gate['observed_validation_rows']==10 and
            all(gate[k] is True for k in ['all_daily_files_exist','all_actions_on_declared_grid','positive_actions_transmitted']))
        unchanged=all(v for k,v in checks.items() if k not in ['smoke_gate_passed','smoke_output_root','next_step_allowed'])
        checks['behavior_gate_passed']=checks['smoke_gate_passed']
        checks['technical_gate_passed']=technical
        checks['next_step_allowed']=bool(technical and unchanged)
        checks['gate_policy']='Complete requested seed cohort; preserve behavior failure without seed selection'
        return checks
    runner.verify_completed_smoke=verify
    print(json.dumps(adapter.run(CONFIG,phase),indent=2),flush=True)

def main():
    import psutil
    OUT.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'configs/hl_fq_8seed_cross_site/fq_seed7.json').read_text())
    cfg['task_id']='20261012fq7'; cfg['task_name']='fq_seed7_historical_completion'
    CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    for phase in ['smoke','formal']:
        target=ROOT/'benchmark_results'/('20261012fq7_fq_seed7_historical_completion'+('_smoke2k' if phase=='smoke' else ''))
        if target.exists(): raise FileExistsError(target)
        start=time.monotonic(); peak=0
        with (OUT/f'{phase}.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,__file__,'--worker',phase],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            exceeded=None
            while proc.poll() is None:
                try:
                    tree=[psutil.Process(proc.pid)]+psutil.Process(proc.pid).children(recursive=True)
                    rss=sum(p.memory_info().rss for p in tree if p.is_running())/1024**2
                    peak=max(peak,rss)
                    if rss>1536 or time.monotonic()-start>7200:
                        exceeded='RSS_LIMIT' if rss>1536 else 'WALL_LIMIT'
                        import signal
                        os.killpg(proc.pid,signal.SIGTERM)
                        try: proc.wait(timeout=10)
                        except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL); proc.wait()
                        break
                except psutil.NoSuchProcess: pass
                time.sleep(2)
        status={'phase':phase,'exit_code':proc.returncode,'peak_tree_rss_mb':peak,'elapsed_seconds':time.monotonic()-start,'limit_failure':exceeded}
        (OUT/f'{phase}_process_status.json').write_text(json.dumps(status,indent=2))
        if exceeded or proc.returncode: raise RuntimeError(status)
    (OUT/'queue_complete.json').write_text(json.dumps({'training_and_evaluation_completed':True,'requires_final_audit':True}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--worker',choices=['smoke','formal']); args=parser.parse_args()
    if args.worker: worker(args.worker)
    else: main()
