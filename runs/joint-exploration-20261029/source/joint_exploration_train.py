"""One matched-marginal exploration question, strict fresh-learner budget."""
import argparse
from dataclasses import asdict
import hashlib
import os
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--route', choices=('joint', 'coordinate'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['AUTOMANIP_EXPLORATION_MODE'] = args.route
    from replay_robot import Config, train, atomic_json

    class ExplorationConfig(Config):
        def validate(self):
            expected = asdict(Config())
            expected.update(method='target-sac', seed=1, gamma=.99, total_transitions=102400, max_seconds=150.)
            if asdict(self) != expected: raise ValueError('Only registered exploration config')
            return self

    report = train(ExplorationConfig(total_transitions=102400, max_seconds=150.), args.output)
    report['admission'] = dict(experiment='joint-exploration-20261029', route=args.route,
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        collector_sha256=hashlib.sha256(Path(__file__).with_name('joint_exploration.py').read_bytes()).hexdigest(),
        primary_transitions=102400, scope='Fresh SAC behavior dependence, not a new target objective or assisted policy')
    atomic_json(args.output/'report.json', report)
