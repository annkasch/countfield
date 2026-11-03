"""python -m countfield train CONFIG.yaml | plot RUN_DIRECTORY | rescore RUN_DIRECTORY"""

import argparse

from countfield.config import load_config
from countfield.train import run


def main() -> None:
    parser = argparse.ArgumentParser(prog="countfield")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train, select on holdout, score evaluation groups")
    train.add_argument("config")
    plot = sub.add_parser("plot", help="(re)draw the plots of a finished run")
    plot.add_argument("run_directory")
    plot.add_argument("--coverage-ylim", type=float, nargs=2, metavar=("LOW", "HIGH"))
    rescore = sub.add_parser(
        "rescore", help="recompute metrics and plots of a finished run from its saved latents"
    )
    rescore.add_argument("run_directory")
    args = parser.parse_args()
    if args.command == "train":
        print(f"output: {run(load_config(args.config))}", flush=True)
    elif args.command == "plot":
        from countfield.plots import plot_run

        for path in plot_run(args.run_directory, args.coverage_ylim):
            print(path)
    elif args.command == "rescore":
        from countfield.train import rescore as rescore_run

        for row in rescore_run(args.run_directory):
            print(row["group"], {k: row[k] for k in ("predictive_nll", "crps", "coverage")})


if __name__ == "__main__":
    main()
