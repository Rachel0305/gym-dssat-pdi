"""One active-time definition shared by both supervisor levels."""
import time
from pathlib import Path
from forecast_memory_pause_099 import paused_seconds

def total_paused(folder,queue=False):
    folder=Path(folder)
    if not queue:return paused_seconds(folder)
    return sum(paused_seconds(p) for p in folder.glob('*/*') if p.is_dir())

class ActiveClock:
    def __init__(self,folder,queue=False):
        self.folder=Path(folder);self.queue=queue
        self.start=time.monotonic();self.initial=total_paused(folder,queue)
    def elapsed(self):
        return max(0,time.monotonic()-self.start-(total_paused(self.folder,self.queue)-self.initial))
