"""Cooperative safe-point pause. Retains the live DSSAT process; never kills it."""
import json,os,signal,time
from pathlib import Path

class MemoryPause:
    def __init__(self,telemetry,state,low_mib=1024,resume_mib=2048,stable_seconds=60,poll_seconds=2):
        self.telemetry=Path(telemetry);self.state=Path(state)
        self.low=low_mib*1024**2;self.high=resume_mib*1024**2
        self.stable=stable_seconds;self.poll=poll_seconds;self.total=0.
    def available(self):
        try:
            v=json.loads(self.telemetry.read_text())
            return v['available_bytes'] if 0<=time.time()-v['updated_unix']<=120 else None
        except (OSError,ValueError,KeyError):return None
    def write(self,paused,started=None):
        value=dict(paused=paused,pid=os.getpid(),paused_seconds=self.total,pause_started_monotonic=started,updated_unix=time.time())
        tmp=self.state.with_name(self.state.name+f'.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(value));os.replace(tmp,self.state)
    def checkpoint(self,flush=lambda:None):
        available=self.available()
        if available is not None and available>=self.low:return
        flush();remaining=signal.alarm(0);started=time.monotonic();healthy=None
        self.write(True,started)
        try:
            while True:
                available=self.available();now=time.monotonic()
                if available is not None and available>=self.high:
                    if healthy is None:healthy=now
                    if now-healthy>=self.stable:break
                else:healthy=None
                time.sleep(self.poll)
        finally:
            self.total+=time.monotonic()-started;self.write(False)
            if remaining:signal.alarm(remaining)

def paused_seconds(folder):
    total=0.
    for p in Path(folder).glob('pause_*.json'):
        v=json.loads(p.read_text());total+=v['paused_seconds']
        if v['paused']:total+=max(0,time.monotonic()-v['pause_started_monotonic'])
    return total
