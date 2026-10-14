from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parent
for item in json.loads((root/'large_file_chunks.json').read_text()):
    dest=(root/item['path']).resolve();assert dest.is_relative_to(root)
    if dest.exists():assert hashlib.sha256(dest.read_bytes()).hexdigest()==item['sha256'];continue
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:
        for part in item['parts']:
            with (root/part).open('rb') as src:
                for block in iter(lambda:src.read(1024*1024),b''):f.write(block)
    h=hashlib.sha256()
    with dest.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    assert h.hexdigest()==item['sha256']
print('Large inputs restored and verified')
