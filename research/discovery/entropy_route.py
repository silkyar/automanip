"""Strict two-route early-grasp experiment; no task or benchmark changes."""
import argparse
from dataclasses import asdict
import hashlib
import os
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--route', choices=('high-actor-low-backup', 'low-actor-high-backup'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['AUTOMANIP_ENTROPY_ROUTE'] = args.route
    from replay_robot import Config, train, atomic_json

    class RouteConfig(Config):
        def validate(self):
            expected = asdict(Config())
            expected.update(method='target-sac', seed=1, gamma=.99, total_transitions=102400, max_seconds=150.)
            if asdict(self) != expected: raise ValueError('Only registered route test config')
            return self

    report = train(RouteConfig(total_transitions=102400, max_seconds=150.), args.output)
    report['admission'] = dict(experiment='entropy-route-20261029', route=args.route,
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_learner_sha256=hashlib.sha256(Path(__file__).with_name('replay_learner_base.py').read_bytes()).hexdigest(),
        primary_transitions=102400, scope='Adaptive entropy-route sufficiency; grasp-only causal question')
    atomic_json(args.output/'report.json', report)
