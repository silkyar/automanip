"""Prepare one bounded three-control from-scratch mechanism experiment."""
import ast
import hashlib
import json
from pathlib import Path
import time

from scripts.prepare_scratch_credit import BASE, BASE_SHA, build_sources


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT/'.runtime/gripper-hold-20261031'
METHODS = ('full', 'masked', 'hold')


def sources():
    common = build_sources(BASE.read_bytes())['ppo-gae']
    common = common.replace('(0,512000,1024000,2048000,4096000,6144000,8192000,10240000)',
                            '(0,512000,1024000,2048000,4096000)')
    result = {}
    for method in METHODS:
        text = common
        if method != 'full':
            needle = '    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)\n'
            replacement = ('    from gripper_hold_initialization import configure, verify_optimizer, maximum_fixed_change\n'
                f'    automanip_masks, automanip_handles = configure(agent.actor_mean, {method == "hold"!r})\n'
                '    automanip_fixed = [p.detach().clone() for p, _ in automanip_masks]\n'+needle+
                '    verify_optimizer(optimizer)\n')
            if text.count(needle) != 1:
                raise ValueError('Optimizer insertion differs')
            text = text.replace(needle, replacement)
            needle = '        if global_step in automanip_checkpoints:\n'
            text = text.replace(needle,
                "        if maximum_fixed_change(automanip_masks, automanip_fixed) != 0:\n"
                "            raise ValueError('Fixed actuator prior moved')\n"+needle, 1)
        ast.parse(text)
        result[method] = text
    return result


def prepare():
    out = RUNTIME/'inputs'
    out.mkdir(parents=True, exist_ok=False)
    programs = sources()
    for method, text in programs.items():
        (out/(method+'.py')).write_text(text)
    # Reuse the existing plain numeric actor exporter exactly.
    exporter = ROOT/'.runtime/scratch-credit-20261030/inputs/export_helper.py'
    (out/'export_helper.py').write_bytes(exporter.read_bytes())
    (out/'gripper_hold_initialization.py').write_bytes((ROOT/'research/gripper_hold_initialization.py').read_bytes())
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
    cpu = json.loads((ROOT/'cumulative/results/gripper-hold-CPU-20261031.json').read_text())
    if not cpu['passed']:
        raise ValueError('CPU mechanism gate did not pass')
    plan = dict(format='automanip-gripper-hold-training-v1', execution_authorized=True,
        methods=list(METHODS), seed=41, files_sha256=hashes, source_sha256=BASE_SHA,
        max_training_transitions_per_trial=4096000, trainer_seconds=300,
        checkpoints=[0,512000,1024000,2048000,4096000],
        production_pins={'mani_skill':'3.0.0b21','sapien':'3.0.0b1','torch':'2.6.0+cu124','numpy':'1.26.4','gymnasium':'0.29.1'})
    (out/'training-plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    score = dict(format='automanip-gripper-hold-score-v1', execution_authorized=True,
        development_seed_base=96900000, episodes=128, checkpoints=plan['checkpoints'],
        max_public_transitions_per_trial=32000, confirmation_queries=0, final_queries=0)
    (RUNTIME/'score-plan.json').write_text(json.dumps(score, indent=2)+'\n')
    protocol = dict(format='automanip-gripper-hold-experiment-v1', registered_at=time.time(),
        execution_authorized=True, authorization='User continuous autonomous research and GPU authorization; separate bounded study after closed negative GPD pilot',
        hypothesis='A known gripper hold mean reduces unintended closing under the existing absolute gripper controller and may improve early from-scratch PPO exploration; it may instead impair acquiring grasps.',
        competing_hypotheses=['Hold coordinate improves sequencing and release before sufficient early experience accumulates',
            'Initial closing bias helps grasp discovery; removing it delays learning',
            'Previous proposal benefit was only a local late-stage search effect and will not transfer to from-scratch learning'],
        intervention='One fixed isolated unit per hidden layer approximates40*q7-.6; learned255-unit residual uses fresh production random initialization. Fixed gradient masks, no pretrained weights or data. Existing48obs/8actions/3x256tanh safe MLP.',
        controls='Full original256-unit strong PPO; identical masked255-unit residual with fixed hold path output0; candidate uses same masked residual with output1/.003. Critic/logstd/random draws and original GAE/reward unchanged. All shared instrumentation identical.',
        initialization='Fresh seed41, no source actors, demonstrations, search queries or prior simulator training data. Analytic actuator prior explicitly disclosed; not a newly learned control law.',
        training=plan, assessment=score,
        budget=dict(max_new_training_transitions=12288000,max_new_public_transitions=96000,max_trainer_seconds_per_trial=300,GPU='L40S',trials=3,automatic_retries=0),
        primary='Normalized success AUC over common completed registered checkpoints: hold must exceed BOTH full and masked controls by>=.10, with final success no lower. Report every curve, stage metric and compute; this is a single-training-seed screen only.',
        secondary='T80/T90 require two consecutive registered checkpoints; unreached thresholds censored without interpolation. Stage rates diagnose grasp impairment or later release changes, not substitute primary success.',
        falsification='If candidate fails primary screen, do not infer sample efficiency or launch automatic seeds/variants. Preserve negative and reassess. If material signal passes, replicate matched training seeds before promotion; no10x claim from this pilot.',
        compute='Equal process/transition ceilings; report consumed transitions, optimizer calls, process/wrapper/learner time, exports/traces and scoring. Masking overhead counted; equal ceilings are not equal consumed compute.',
        limitations='Actuator-specific prior on one task/seed. Gaussian noise plus clipping still induces inward drift near limits; hold is a mean map, not zero physical drift. Frozen wire may alter capacity; both masked and full controls required.',
        CPU_gate='cumulative/results/gripper-hold-CPU-20261031.json',preserve='All trusted evaluator/task/seeds/pins and historical artifacts unchanged; scoring accepts safe MLPs only in separate image.')
    target = ROOT/'cumulative/experiments/gripper-hold-v1.json'
    with target.open('x') as f:
        json.dump(protocol, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(state='prepared',runtime=str(RUNTIME),protocol=str(target)),indent=2))


if __name__ == '__main__':
    prepare()
