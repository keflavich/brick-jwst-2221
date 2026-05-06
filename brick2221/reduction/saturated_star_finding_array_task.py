#!/usr/bin/env python

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from saturated_star_finding import remove_saturated_stars


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run saturated star finding/removal for one array-task exposure."
    )
    parser.add_argument("--filter", required=True, dest="filtername",
                        help="Filter name, e.g. F410M")
    parser.add_argument("--module", required=True,
                        help="Module name, e.g. nrca, nrcb, nrca1")
    parser.add_argument("--target", default="brick",
                        help="Target directory under /orange/adamginsburg/jwst/")
    parser.add_argument("--each-suffix", default="align*crf",
                        help="Filename glob suffix to match one exposure set")
    parser.add_argument("--array-index", type=int, default=None,
                        help="Override SLURM_ARRAY_TASK_ID")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.array_index is None:
        array_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    else:
        array_index = args.array_index

    pattern = (
        f"/orange/adamginsburg/jwst/{args.target}/{args.filtername}/pipeline/"
        f"*{args.module}*{args.each_suffix}.fits"
    )
    matches = sorted(glob.glob(pattern))

    print(f"filter={args.filtername} module={args.module} target={args.target}", flush=True)
    print(f"suffix={args.each_suffix}", flush=True)
    print(f"matched {len(matches)} files", flush=True)

    if array_index < 0 or array_index >= len(matches):
        print(
            f"array_index={array_index} is outside available files [0, {len(matches)-1}]",
            flush=True,
        )
        return

    filename = matches[array_index]
    print(f"Running remove_saturated_stars on {filename}", flush=True)
    remove_saturated_stars(filename)


if __name__ == "__main__":
    main()
