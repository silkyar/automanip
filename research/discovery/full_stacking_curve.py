"""Strict fresh-seed complete-task comparison, not an old-run continuation."""
import argparse
from dataclasses import asdict
import hashlib
import os
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--route', choices=('baseline', 'coordinate'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['AUTOMANIP_FULL_CURVE_ARM'] = args.route
    from replay_robot import Config, train, atomic_json

    class CurveConfig(Config):
        def validate(self):
            expected = asdict(Config())
            expected.update(method='target-sac', seed=2, gamma=.99, total_transitions=1024000, max_seconds=600.)
            if asdict(self) != expected: raise ValueError('Only registered seed2 full-task comparison')
            return self

    report = train(CurveConfig(seed=2, total_transitions=1024000, max_seconds=600.), args.output)
    report['admission'] = dict(experiment='full-stacking-curve-20261029', route=args.route,
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        collector_sha256=hashlib.sha256(Path(__file__).with_name('joint_exploration.py').read_bytes()).hexdigest(),
        primary_transitions=512000, scope='Freshseed2 full-stacking curves; no pretrained continuation or automatic promotion')
    atomic_json(args.output/'report.json', report)
