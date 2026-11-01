"""One cumulative StackCube track: contracts, immutable assets and assessment data.

No trainer, deployment, simulator or candidate-code imports. Existing safe actor
validation remains unchanged. These manifests are provenance, not authentication.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path

from automanip.policy_io import PolicySchema, load_policy, _json, _read
from automanip.stackcube import STACK_TASK

ROOT = Path(__file__).resolve().parents[2]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


@dataclass(frozen=True)
class Interface:
    robot: str = 'panda_wristcam'
    controller: str = 'pd_joint_delta_pos'
    observation_dim: int = 48
    action_dim: int = 8
    control_hz: int = 20
    simulation_hz: int = 100
    horizon: int = 50
    action_semantics: str = 'clip[-1,1]; arm7:q_target=q_current+0.1*a radians; gripper:absolute_mimic_target=0.015+0.025*a meters; use_target=False'
    observation_semantics: str = 'qpos arm[0:7]rad fingers[7:9]m; qvel arm[9:16]rad/s fingers[16:18]m/s; TCPpose[18:25],cubeApose[25:32],cubeBpose[32:39] per-environment scene coordinates(m,wxyz; GPU scene offsets removed); A-TCP[39:42],B-TCP[42:45],B-A[45:48] per-environment scene meters'
    goal_encoding: str = 'cubeB per-environment scene pose and relative vectors; stack center 0.04m above cubeB; no extra goal dimensions'
    boundaries: str = 'full50 control steps; success does not reset; true termination and time truncation distinct; restart at explicit episode boundary'
    information: str = 'state48 only; no object velocities/contact-force inputs, simulator lookahead, demonstrations or evaluator signals unless separately declared'
    runtime: str = 'stateless tanh-hidden MLP,linear output clipped[-1,1];automanip-json-f32-v1'

    @property
    def fingerprint(self):
        return sha(asdict(self))


STACK_INTERFACE = Interface()


@dataclass(frozen=True)
class TaskContract:
    goal: str = 'source'
    interface: Interface = field(default_factory=Interface)
    native_fingerprint: str = STACK_TASK.fingerprint

    def validate(self):
        if self.goal not in ('source', 'north'):
            raise ValueError('Only registered source/north variation goals are implemented')
        if len(self.native_fingerprint) != 64:
            raise ValueError('Invalid native task fingerprint')
        return self

    @property
    def task_id(self):
        self.validate()
        return 'stackcube-' + self.goal + '-' + sha(self.to_dict())[:16]

    @property
    def schema(self):
        return PolicySchema(self.native_fingerprint, self.interface.observation_dim, self.interface.action_dim)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if set(value) != {'goal', 'interface', 'native_fingerprint'}:
            raise ValueError('Unexpected task contract fields')
        return cls(value['goal'], Interface(**value['interface']), value['native_fingerprint']).validate()

    def require_stack(self):
        self.validate()
        if self.interface != STACK_INTERFACE or self.native_fingerprint != STACK_TASK.fingerprint:
            raise ValueError('Unsupported interface; no cross-controller/observation adapter')
        return self


SOURCE = TaskContract()
NORTH = TaskContract('north')


def directory_digest(directory):
    directory=Path(directory)
    entries=[]
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file():raise ValueError('Expected a flat data-only episode store')
        raw=path.read_bytes();entries.append(dict(name=path.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    return sha(entries)


def relative_path(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Asset storage must remain under the supplied storage root')
    return str(path.relative_to(root))


def cost(transitions=None, seconds=None, dollars=None):
    if transitions is not None and (type(transitions) is not int or transitions < 0):
        raise ValueError('Invalid interaction cost')
    for value in (seconds, dollars):
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError('Invalid compute/money cost')
    return dict(environment_transitions=transitions, wall_seconds=seconds, dollars=dollars)


@dataclass(frozen=True)
class Asset:
    record: dict
    manifest_path: Path
    policy_dir: Path
    policy: object

    @property
    def asset_id(self):
        return self.record['asset_id']

    @property
    def task(self):
        return TaskContract.from_dict(self.record['source_task'])


def register_asset(destination, policy_dir, task, acquisition, *, root=ROOT, parents=(), evidence=(), data_assets=()):
    task.validate()
    actor = load_policy(policy_dir, task.schema)
    acquisition = cost(acquisition['environment_transitions'], acquisition['wall_seconds'], acquisition['dollars'])
    record = dict(format='automanip-cumulative-asset-v1', version=1,
        capability='whole StackCube policy; not an independently validated grasp skill',
        source_task=task.to_dict(), interface_id=task.interface.fingerprint,
        policy_path=relative_path(policy_dir, root), weights_sha256=actor.weights_sha256,
        policy_manifest_sha256=hashlib.sha256((Path(policy_dir)/'manifest.json').read_bytes()).hexdigest(),
        parents=[dict(asset_id=p.asset_id, manifest=relative_path(p.manifest_path, root)) for p in parents],
        acquisition=acquisition, applicability=list(evidence), data_assets=list(data_assets))
    record['asset_id'] = sha(record)
    write_json(destination, record)
    return load_asset(destination, root=root)


def load_asset(path, *, root=ROOT):
    path = Path(path)
    relative_path(path, root)
    record = _json(_read(path, 1024*1024))
    required = {'format','version','capability','source_task','interface_id','policy_path','weights_sha256',
                'policy_manifest_sha256','parents','acquisition','applicability','data_assets','asset_id'}
    if type(record) is not dict or set(record) != required or record['format'] != 'automanip-cumulative-asset-v1' or type(record['version']) is not int or record['version'] != 1:
        raise ValueError('Unsupported asset manifest')
    if sha({k:v for k,v in record.items() if k != 'asset_id'}) != record['asset_id']:
        raise ValueError('Asset manifest digest mismatch')
    task = TaskContract.from_dict(record['source_task'])
    if record['interface_id'] != task.interface.fingerprint:
        raise ValueError('Asset interface identity differs')
    acquisition = record['acquisition']
    if set(acquisition) != {'environment_transitions','wall_seconds','dollars'}:
        raise ValueError('Invalid acquisition fields')
    cost(acquisition['environment_transitions'], acquisition['wall_seconds'], acquisition['dollars'])
    if type(record['parents']) is not list or type(record['applicability']) is not list or type(record['data_assets']) is not list:
        raise ValueError('Invalid asset provenance')
    policy_dir = Path(root)/record['policy_path']
    relative_path(policy_dir, root)
    actor = load_policy(policy_dir, task.schema)
    if actor.weights_sha256 != record['weights_sha256'] or hashlib.sha256((policy_dir/'manifest.json').read_bytes()).hexdigest() != record['policy_manifest_sha256']:
        raise ValueError('Asset policy changed')
    return Asset(record, path, policy_dir, actor)


def asset_status(path, *, root=ROOT):
    try:
        asset = load_asset(path, root=root)
        return dict(state='available', asset_id=asset.asset_id, weights_sha256=asset.policy.weights_sha256)
    except FileNotFoundError as exc:
        return dict(state='missing', path=str(exc.filename))
    except (ValueError, TypeError, KeyError) as exc:
        return dict(state='invalid', error=str(exc))


def lifetime_cost(asset, *, root=ROOT):
    seen, active, available_data = {}, set(), {}
    def visit(item):
        if item.asset_id in active:
            raise ValueError('Cyclic asset lineage')
        if item.asset_id in seen:
            return
        active.add(item.asset_id)
        for parent in item.record['parents']:
            loaded = load_asset(Path(root)/parent['manifest'], root=root)
            if loaded.asset_id != parent['asset_id']:
                raise ValueError('Parent identity differs')
            visit(loaded)
        active.remove(item.asset_id)
        seen[item.asset_id] = item.record['acquisition']
        for data in item.record['data_assets']:
            if 'acquisition' in data:
                digest=data['sha256'];entry=data['acquisition']
                cost(entry['environment_transitions'],entry['wall_seconds'],entry['dollars'])
                if digest in available_data and available_data[digest]!=entry:raise ValueError('Conflicting data acquisition cost')
                available_data[digest]=entry
    visit(asset)
    totals = {k:None if any(v[k] is None for v in seen.values()) else sum(v[k] for v in seen.values()) for k in ('environment_transitions','wall_seconds','dollars')}
    all_costs=list(seen.values())+list(available_data.values())
    library_totals={k:None if any(v[k] is None for v in all_costs) else sum(v[k] for v in all_costs) for k in ('environment_transitions','wall_seconds','dollars')}
    return dict(unique_assets=len(seen), asset_costs=seen, totals=totals,available_unused_experience_costs=available_data,library_totals_including_available_experience=library_totals,
                attribution='Each ancestor charged once; unknown source costs remain unknown, not zero')


def compatible(asset, target):
    target.require_stack()
    if asset.task.interface.fingerprint != target.interface.fingerprint or asset.task.native_fingerprint != target.native_fingerprint:
        raise ValueError('Incompatible controller/observation/timing/goal encoding; no padding adapter')


def retention_decision(target, retained, reference, *, controller_sha256, reference_sha256, source_task=SOURCE, target_task=NORTH,
                       partition='development', target_threshold=.9, retention_threshold=.9, max_drop=.05):
    if any(r is None for r in (target, retained, reference)):
        return dict(state='unassessed', promoted=False, retention_semantics='same_controller')
    for report, task in ((target,target_task),(retained,source_task),(reference,source_task)):
        if report['task_id'] != task.task_id or report['partition'] != partition:
            raise ValueError('Wrong assessment task or partition')
        n,s = report['episodes'],report['successes']
        if type(n) is not int or n < 1 or type(s) is not int or not 0 <= s <= n:
            raise ValueError('Invalid outcome counts')
    if target['weights_sha256'] != controller_sha256 or retained['weights_sha256'] != controller_sha256:
        raise ValueError('Retention must score the SAME adapted controller, not an archived router branch')
    if reference['weights_sha256'] != reference_sha256:
        raise ValueError('Reference is not the declared unchanged source controller')
    if retained['episodes'] != reference['episodes']:
        raise ValueError('Paired retention/reference episode counts differ')
    if retained['episode_set_id'] != reference['episode_set_id']:
        raise ValueError('Retention/reference must use the same paired episode set')
    t,r,b = (x['successes']/x['episodes'] for x in (target,retained,reference))
    passed = not any(x.get('failed_episodes',0) for x in (target,retained,reference)) and t >= target_threshold and r >= retention_threshold and b-r <= max_drop+1e-12
    return dict(state='assessed', promoted=passed, target_success=t, retained_success=r, reference_success=b,
                retention_drop=b-r, retention_semantics='same_controller',
                qualification='One screening decision; independent replication/final assessment still required')
