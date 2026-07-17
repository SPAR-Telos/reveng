#!/usr/bin/env python3
"""Build Experiment 1 and 2 outputs from local direct-answer readouts."""

from __future__ import annotations

import argparse
import json

from reveng.experiments.local_immediate_analysis import build_local_immediate_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readout-dir", required=True)
    parser.add_argument("--experiment2-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_local_immediate_analysis(
                readout_dir=args.readout_dir,
                experiment2_dir=args.experiment2_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
