"""Single-factor discount restoration; original learner/robot code unchanged."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
from replay_robot import Config,train,atomic_json


class GammaConfig(Config):
    def validate(self):
        expected=asdict(Config())
        expected.update(method='target-sac',seed=1,gamma=.9,total_transitions=256000,max_seconds=240.)
        if asdict(self)!=expected:raise ValueError('Only preregistered gamma-sufficiency config')
        return self


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=GammaConfig(method='target-sac',seed=1,gamma=.9,total_transitions=256000,max_seconds=240.)
    report=train(config,args.output)
    report['admission']=dict(experiment='gamma-sufficiency-20261028',
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        primary_transitions=102400,primary_metrics=['grasp','lift'],
        scope='Single-seed discount sufficiency screen; not new algorithm or strong baseline claim')
    atomic_json(args.output/'report.json',report)
