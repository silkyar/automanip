"""Once-only, fixed-policy confirmation admission for the td3-utd16 recipe; no research-agent inputs."""
EXPERIMENT = 'stackcube-td3-utd16-confirmation-20261101'
PROTOCOL = 'fe2cc5987576a84b7a66ae9c8c9c4099b688c682d5736c5388217f5372391764'
SEEDS = [21, 22, 23, 24, 25]
DIGESTS = [
    '2af097c2dc1e3c4cb7e6498ad9adab6eefec7524e109ac54a829d63ccb45593e',
    'aed562e322f46e0e4684ca28c4b29a5a1c23c7021d75caaae274d8142e302a7c',
    'd0744ec8b4f855d6e609bb75f5aa8b3b66b93bc44cc496c196c80682e8265294',
    '91d1055897bf1a22ae4699226983d80995f2381933d0f180b6edf1a0ed1070f2',
    'f3d385a4f93c29faa59569d7415be12933d7c21e4059d6a9d8c16a1dd3b53388',
]
RULE = {'mean_success_at_least': .95, 'every_seed_success_at_least': .9}


def validate(plan):
    if not plan.get('launch_enabled') or not plan.get('root_reviewed'):
        raise PermissionError('Explicit reviewed admission required')
    if plan.get('experiment_id') != EXPERIMENT or plan.get('protocol_sha256') != PROTOCOL:
        raise ValueError('Fixed campaign/protocol required')
    if plan.get('confirmation_weights_sha256') != DIGESTS or plan.get('training_seeds') != SEEDS:
        raise ValueError('Only the five final r4 policies may be confirmed')
    if [(p.get('seed'), p.get('weights_sha256')) for p in plan.get('policies', [])] != list(zip(SEEDS, DIGESTS)):
        raise ValueError('Policy list differs from admission')
    for key, value in dict(split='confirmation', episodes_per_policy=512, total_episodes=2560,
                           max_parallel=3, gpu='L40S', evaluation_timeout_seconds=220,
                           gpu_container_timeout_seconds=240, coordinator_timeout_seconds=1800,
                           budget_cap_usd=2, new_training_runs=0, retries=0).items():
        if plan.get(key) != value:
            raise ValueError('Fixed confirmation scope changed: '+key)
    if plan.get('acceptance_rule') != RULE:
        raise ValueError('Predeclared target changed')
    return DIGESTS
