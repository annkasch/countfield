"""python -m countfield train CONFIG.yaml"""

import argparse

from countfield.config import load_config
from countfield.train import run


def main() -> None:
    parser = argparse.ArgumentParser(prog="countfield")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train, select on holdout, score evaluation groups")
    train.add_argument("config")
    args = parser.parse_args()
    if args.command == "train":
        print(f"output: {run(load_config(args.config))}", flush=True)


if __name__ == "__main__":
    main()
