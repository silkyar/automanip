"""Build small, explicit Modal image contexts without credentials or run archives."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SHA256='fe2cc5987576a84b7a66ae9c8c9c4099b688c682d5736c5388217f5372391764'
FROZEN_PACKAGE = (
    '__init__.py', 'evaluator.py', 'metrics.py', 'numpy_inference.py',
    'policy_io.py', 'stackcube.py', 'stackcube_adapter.py', 'task_spec.py',
)


def source_files(role):
    if role not in ('validation', 'trainer', 'converter'):
        raise ValueError('unknown image role')
    if role == 'trainer':
        return ('third_party/maniskill/ppo.py',)
    if role == 'converter':
        return ('scripts/export_official_policy.py','src/automanip/__init__.py',
                'src/automanip/policy_io.py','src/automanip/stackcube.py','src/automanip/task_spec.py')
    raw=(ROOT/'benchmarks/stackcube-v1/protocol.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PROTOCOL_SHA256:raise ValueError('frozen protocol digest changed')
    protocol = json.loads(raw)
    for name,digest in protocol['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('frozen source digest changed: '+name)
    return tuple(sorted(set(protocol['files']) | {
        'benchmarks/stackcube-v1/protocol.json', 'scripts/check_stackcube.py',
        'scripts/evaluate_stackcube.py',
        *('src/automanip/'+name for name in FROZEN_PACKAGE),
    }))


def build_context(role, destination):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    # Only recreate this explicitly named staging directory; never walk the repository.
    for old in destination.iterdir():
        if old.is_dir():shutil.rmtree(old)
        else:old.unlink()
    manifest={}
    for name in source_files(role):
        source=ROOT/name
        if source.is_symlink() or not source.is_file():raise ValueError('regular source file required')
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes())
        manifest[name]=hashlib.sha256(target.read_bytes()).hexdigest()
    (destination/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return destination
