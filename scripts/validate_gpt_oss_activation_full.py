#!/usr/bin/env python3
"""Validate the complete GPT-OSS boundary-activation dataset."""

from __future__ import annotations

import argparse
import json

from reveng.experiments.gpt_oss_activation_full import validate_gpt_oss_activation_full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="outputs/activation_collection/gpt_oss_20b_boundary_v1",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate_gpt_oss_activation_full(args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
