"""Once-only, fixed-policy confirmation admission; no research-agent inputs."""
EXPERIMENT = 'stackcube-precision-confirmation-20261028'
PROTOCOL = 'fe2cc5987576a84b7a66ae9c8c9c4099b688c682d5736c5388217f5372391764'
DIGESTS = [
    'ebe92bbc1ce5dec3f8a271792dd4befb81296b36a5f131345c84e8715a39ea16',
    'c3abf0e70973e39c3616f0a217f75583aeff56aea86ad253370e501a24472959',
    'ec3202403c8cb7c705c1591ce61d61a242d4d9ab957241b4b98d8697bef5cb10',
]


def validate(plan):
    if not plan.get('launch_enabled') or not plan.get('root_reviewed'):
        raise PermissionError('Explicit reviewed admission required')
    if plan.get('experiment_id') != EXPERIMENT or plan.get('protocol_sha256') != PROTOCOL:
        raise ValueError('Fixed campaign/protocol required')
    if plan.get('confirmation_weights_sha256') != DIGESTS or plan.get('training_seeds') != [1,2,3]:
        raise ValueError('Only the three final policies may be confirmed')
    for key, value in dict(split='confirmation', episodes_per_policy=512, total_episodes=1536,
                           max_parallel=3, gpu='L40S', evaluation_timeout_seconds=220,
                           gpu_container_timeout_seconds=240, coordinator_timeout_seconds=1800,
                           budget_cap_usd=1, new_training_runs=0, retries=0).items():
        if plan.get(key) != value:
            raise ValueError('Fixed confirmation scope changed: '+key)
    if plan.get('acceptance_rule') != {'mean_success_at_least':.99, 'every_seed_success_at_least':.98}:
        raise ValueError('Predeclared target changed')
    return DIGESTS
