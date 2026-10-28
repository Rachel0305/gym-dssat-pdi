"""Record real thread identities; never kill a worker on a name-only sample."""
import json,os,time,signal
from pathlib import Path
from forecast_memory_pause_108 import MemoryPause as BasePause

def classify(records):
 return dict(main=[r for r in records if r['tid']==r['pid']],ipc=[r for r in records if r['tid']!=r['pid'] and r['name'].startswith('ZMQbg/')],unknown=[r for r in records if r['tid']!=r['pid'] and not r['name'].startswith('ZMQbg/')])
class ThreadGuard:
 def __init__(self,folder):
  self.folder=Path(folder);self.bad=0;self.good=0
 def observe(self,tree):
  processes=[];records=[]
  for p in tree:
   try:
    if not p.name().startswith('dscsm'):continue
    processes.append(dict(pid=p.pid,created=p.create_time()))
    for thread in p.threads():
     try:name=Path(f'/proc/{p.pid}/task/{thread.id}/comm').read_text().strip()
     except FileNotFoundError:continue
     records.append(dict(pid=p.pid,tid=thread.id,name=name,user_seconds=thread.user_time,system_seconds=thread.system_time))
   except (ProcessLookupError,FileNotFoundError):continue
  classified=classify(records)
  suspect=len(processes)>1 or any(sum(r['pid']==p['pid'] for r in records)>3 for p in processes)
  self.bad=self.bad+1 if suspect else 0
  self.good=self.good+1 if not suspect else 0
  evidence=dict(unix=time.time(),processes=processes,threads=records,classified=classified,suspect=suspect,consecutive_suspect=self.bad)
  if suspect or classified['unknown']:
   with (self.folder/'thread_guard_samples.jsonl').open('a') as f:f.write(json.dumps(evidence)+'\n')
  hold=self.folder/'thread_guard_hold.json'
  if len(processes)>1 or self.bad>=3:
   hold.write_text(json.dumps(evidence))
  elif self.good>=3 and hold.exists():
   hold.unlink() # Only this newly generated control flag, never research data.
class MemoryPause(BasePause):
 def checkpoint(self,flush=lambda:None):
  hold=self.state.parent/'thread_guard_hold.json'
  if hold.exists():
   flush();remaining=signal.alarm(0);started=time.monotonic();self.write(True,started)
   try:
    while hold.exists():time.sleep(2)
   finally:
    self.total+=time.monotonic()-started;self.write(False)
    if remaining:signal.alarm(remaining)
  super().checkpoint(flush)
