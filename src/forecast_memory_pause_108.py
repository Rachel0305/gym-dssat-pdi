"""Debounce host pressure for evaluation workers; no process termination."""
import time
from forecast_memory_pause_099 import MemoryPause as OriginalPause
class MemoryPause(OriginalPause):
 def __init__(self,telemetry,state):
  super().__init__(telemetry,state,low_mib=1024,resume_mib=1280,stable_seconds=30)
  self.low_since=None
 def must_pause(self,available,now):
  if available is None or available<256*1024**2:
   self.low_since=None
   return True
  if available>=self.low:
   self.low_since=None
   return False
  if self.low_since is None:self.low_since=now
  return now-self.low_since>=60
 def checkpoint(self,flush=lambda:None):
  if self.must_pause(self.available(),time.monotonic()):
   super().checkpoint(flush)
   self.low_since=None
