"""Independent guest telemetry; avoids stale samples during host pauses."""
from pathlib import Path
import json,time,os,psutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/double_dqn_pilot_132_20261028/attempt02'
while not (OUT/'result.json').exists() and not (OUT/'FAILED.json').exists():
    try:
        tag=json.loads((OUT/'manager_status.json').read_text())['tag'];p=OUT/tag/'guest_memory.json';tmp=p.with_name(f'guest_memory.monitor{os.getpid()}.tmp')
        tmp.write_text(json.dumps(dict(updated_unix=time.time(),available_bytes=psutil.virtual_memory().available,pid=os.getpid(),independent_monitor=True)));os.replace(tmp,p)
    except (OSError,KeyError,ValueError):pass
    time.sleep(2)
