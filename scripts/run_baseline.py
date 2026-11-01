"""Print a reproducible multi-seed GPU recipe; training requires --execute."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output-root", type=Path, default=Path("runs/reference"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**31 for s in args.seeds):
        parser.error("seeds must be unique integers in [0, 2**31)")
    root = Path(__file__).resolve().parents[1]
    for seed in args.seeds:
        command = [sys.executable, str(root/"train.py"), "--config", str(root/"configs/baseline.json"),
                   "--backend", "physx_cuda", "--seed", str(seed),
                   "--output", str(args.output_root/f"seed-{seed}")]
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
