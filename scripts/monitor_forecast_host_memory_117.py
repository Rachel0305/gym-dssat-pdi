"""Independent, single-instance Windows telemetry for the existing 116 queue."""
from pathlib import Path
import ctypes, json, os, time, msvcrt, traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/four_site_forecast_116_20261025'

class MemoryStatus(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),*[(n,ctypes.c_ulonglong) for n in ['total_physical','available_physical','total_page_file','available_page_file','total_virtual','available_virtual','available_extended_virtual']]]

def publish(name,value):
    target=OUT/name
    temporary=target.with_name(target.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    for _ in range(40):
        try:
            os.replace(temporary,target)
            return
        except PermissionError:
            time.sleep(.25)
    raise PermissionError(f'Cannot publish {name}; retry next sample')

def main():
    assert os.name=='nt'
    with (OUT/'monitor117.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):lock.write(b'0');lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:return
        while True:
            try:
                state=json.loads((OUT/'queue_status.json').read_text(encoding='utf-8-sig'))
                if state.get('status') in ('completed','failed'):return
                memory=MemoryStatus();memory.length=ctypes.sizeof(memory)
                if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                    raise ctypes.WinError()
                now=time.time()
                publish('host_memory.json',dict(updated_unix=now,pid=os.getpid(),available_bytes=memory.available_physical,total_bytes=memory.total_physical,load_percent=memory.load,commit_used_bytes=memory.total_page_file-memory.available_page_file,monitor_version=117))
                # Auxiliary status failure must not stop future memory sampling.
                try:publish('HOST_MONITOR_REPLACEMENT_117.json',dict(status='healthy',pid=os.getpid(),updated_unix=now,launch='independent WMI process',supersedes_pid=5920))
                except OSError:pass
            except Exception:
                try:
                    with (OUT/'monitor117_errors.jsonl').open('a',encoding='utf-8') as log:
                        log.write(json.dumps(dict(unix=time.time(),error=traceback.format_exc()))+'\n')
                except OSError:pass
            time.sleep(10)

if __name__=='__main__':main()
