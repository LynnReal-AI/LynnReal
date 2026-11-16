"""Integrity verification for complete inference weight packages."""
from pathlib import Path
import json
import hashlib


def verify_checksums(directory):
    directory = Path(directory); manifest = directory/'SHA256SUMS.json'
    if not manifest.is_file(): return False
    for name, expected in json.loads(manifest.read_text()).items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()): raise ValueError('Invalid checksum path')
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(16<<20),b''): h.update(chunk)
        if h.hexdigest() != expected: raise ValueError(f'Weight checksum mismatch: {name}')
    return True
