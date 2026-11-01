"""Register local source evidence and compile the UNLAUNCHED pilot; never run it."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from automanip.cumulative import write_json
from automanip.cumulative_plan import register_existing_source,make_plan,compile_plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True,help='New preparation directory; existing output is never overwritten')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        source=register_existing_source()
        plan=make_plan(source);write_json(args.output/'plan.json',plan)
        result=compile_plan(plan,args.output/'prepared.json')
        print(json.dumps(dict(state=result['state'],source_asset=source.asset_id,output=str(args.output),execution_authorized=False)))
    except FileNotFoundError as exc:
        result=dict(state='missing_asset',path=str(exc.filename),execution_authorized=False)
        write_json(args.output/'missing-assets.json',result);print(json.dumps(result));raise SystemExit(2)


if __name__=='__main__':main()
