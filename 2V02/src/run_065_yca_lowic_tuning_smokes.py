"""Isolated 2K YC one-factor PPO tuning runner; never launches formal training."""
from __future__ import annotations
import argparse, importlib.util, json, time
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'src'/'055_yca_lowIC_site_transfer'/'run_055_00_yca_lowIC_expanded_action_maskableppo.py'
spec=importlib.util.spec_from_file_location('base055',P); base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
BASE={'learning_rate':3e-4,'gamma':1.0,'n_steps':144,'batch_size':144,'n_epochs':5,'ent_coef':0.01,'net_arch':[64,64]}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True,type=Path); p.add_argument('--dry-run',action='store_true'); p.add_argument('--rescue25k',action='store_true'); a=p.parse_args()
 cfg=json.loads(a.config.read_text(encoding='utf-8')); override=cfg.pop('ppo_overrides'); cfg['report_checkpoint']=2000; cfg['reference_run']='046_10_sya_originIC_expanded_action_maskableppo'
 if set(override) not in ({'net_arch'},{'learning_rate'},{'ent_coef'}): raise ValueError('exactly one permitted PPO factor')
 effective={**BASE,**override}; original=base.engine.load_config
 def patched():
  c=original(); c['ppo'].update(effective); return c
 base.engine.load_config=patched
 try:
  if a.dry_run:
   pf=base.preflight(cfg); print(json.dumps({'preflight':pf,'effective_ppo':effective,'mode':'dry_run'},ensure_ascii=False,indent=2)); return
  if a.rescue25k:
   if cfg['training'] != {'total_timesteps':25000,'checkpoint_steps':[5000,10000,25000]}: raise ValueError('registered rescue must be exactly 25K/5K,10K,25K')
   base.SMOKE_TIMESTEPS=25000; base.SMOKE_CHECKPOINTS=[5000,10000,25000]
  out=base.output_root(base.prepare_config(cfg,smoke=True))
  if out.exists(): raise FileExistsError(out)
  start=time.perf_counter(); rss0=psutil.Process().memory_info().rss
  tmp=ROOT/'benchmark_results'/'_065_tmp.json'; tmp.write_text(json.dumps(cfg),encoding='utf-8')
  try: result=base.run(tmp,dry_run=False,smoke=True,formal=False)
  finally: tmp.unlink(missing_ok=True)
  result['effective_ppo']=effective; result['elapsed_s']=time.perf_counter()-start; result['rss_delta_bytes']=psutil.Process().memory_info().rss-rss0
  (out/'065_execution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2))
 finally: base.engine.load_config=original
if __name__=='__main__': main()
