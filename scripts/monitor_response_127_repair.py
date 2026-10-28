"""Replace failed host telemetry writer without restarting offline inference."""
from pathlib import Path
import os,time,json,psutil
OUT=Path(__file__).resolve().parents[1]/'results/forecast_policy_response_127_20261027'
(OUT/'telemetry_repair.json').write_text(json.dumps({'reason':'Windows transient replace PermissionError in original telemetry thread','worker_restarted':False,'replacement_pid':os.getpid()}))
while not (OUT/'result.json').exists() and not (OUT/'FAILED.json').exists():
    p=OUT/'host_memory.json';tmp=OUT/f'host_memory.{os.getpid()}.tmp'
    tmp.write_text(json.dumps(dict(updated_unix=time.time(),available_bytes=psutil.virtual_memory().available,pid=os.getpid())))
    for attempt in range(50):
        try:os.replace(tmp,p);break
        except PermissionError:time.sleep(.05)
    time.sleep(2)
