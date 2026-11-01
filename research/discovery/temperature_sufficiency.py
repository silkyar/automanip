"""One explicit learner-line change; robot and actor code unchanged."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
from replay_robot import Config,train,atomic_json


class TemperatureConfig(Config):
    def validate(self):
        expected=asdict(Config())
        expected.update(method='target-sac',seed=1,gamma=.99,total_transitions=256000,max_seconds=240.)
        if asdict(self)!=expected:raise ValueError('Only preregistered temperature-sufficiency config')
        return self


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=TemperatureConfig(method='target-sac',seed=1,gamma=.99,total_transitions=256000,max_seconds=240.)
    report=train(config,args.output)
    report['admission']=dict(experiment='temperature-sufficiency-20261028',
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        initial_alpha=.005,primary_transitions=102400,primary_metrics=['grasp','lift'],
        scope='Single-seed initial-temperature sufficiency screen; not new algorithm or strong baseline claim')
    atomic_json(args.output/'report.json',report)
