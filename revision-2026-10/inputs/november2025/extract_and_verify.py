"""Extract the published November inputs and verify every original file hash."""
from pathlib import Path
import hashlib
import json
import zipfile

base = Path(__file__).resolve().parent
report = json.loads((base / 'verification.json').read_text())
expected = json.loads((base.parents[1] / 'manifests' / 'november_file_hashes.json').read_text())
output = base / 'extracted'

def digest(data):
    return hashlib.sha256(data).hexdigest()

# Validate all archives and members before writing anything.
found = set()
for part in report['parts']:
    path = base / part['file']
    if digest(path.read_bytes()) != part['sha256']:
        raise RuntimeError(f'Archive checksum mismatch: {path.name}')
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name not in expected or name in found:
                raise RuntimeError(f'Unexpected or duplicate member: {name}')
            data = z.read(name)
            if digest(data) != expected[name]:
                raise RuntimeError(f'Input checksum mismatch: {name}')
            target = output / name
            if target.exists() and digest(target.read_bytes()) != expected[name]:
                raise RuntimeError(f'Refusing to overwrite different file: {target}')
            found.add(name)
if found != set(expected):
    raise RuntimeError('Incomplete November input set')
output.mkdir(exist_ok=True)
for part in report['parts']:
    with zipfile.ZipFile(base / part['file']) as z:
        for name in z.namelist():
            target = output / name
            if not target.exists():
                target.write_bytes(z.read(name))
print(f'Verified and extracted {len(found)} original files to {output}')
