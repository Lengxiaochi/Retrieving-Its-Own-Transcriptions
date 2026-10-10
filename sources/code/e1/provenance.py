"""Run identities and atomic result writes; no model imports."""
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from common import NORMALIZATION_VERSION

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def run_metadata(args, data=None, inputs=()):
    root = Path(__file__).resolve().parents[1]
    packages = {}
    for name in ('torch', 'numpy', 'transformers', 'faster-whisper', 'ctranslate2', 'jiwer'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    result = dict(config=vars(args), argv=sys.argv, python=platform.python_version(),
                  normalization_version=NORMALIZATION_VERSION, packages=packages,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  code_sha256={str(p.relative_to(root)): sha256(p) for p in root.rglob('*.py')},
                  input_sha256={str(p): sha256(p) for p in inputs if p and Path(p).is_file()})
    if data is not None:
        result['evaluation_manifest'] = data
        result['evaluation_manifest_sha256'] = hashlib.sha256(
            json.dumps(data, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    return result
