"""Race-tolerant output size accounting; disappearing temp entries are normal."""
import os
from pathlib import Path

def directory_bytes(root):
    total=0
    def walk_error(error):
        if not isinstance(error,(FileNotFoundError,NotADirectoryError)):
            raise error
    for base,dirs,files in os.walk(root,onerror=walk_error,followlinks=False):
        for name in files:
            path=Path(base)/name
            try:
                if not path.is_symlink():total+=path.stat().st_size
            except (FileNotFoundError,NotADirectoryError):
                continue
    return total
