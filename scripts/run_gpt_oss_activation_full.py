#!/usr/bin/env python3
"""Run resumable GPT-OSS activation extraction over the full DoorKey corpus."""

from __future__ import annotations

import argparse
import json

from reveng.experiments.gpt_oss_activation_full import run_gpt_oss_activation_full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="outputs/activation_collection/gpt_oss_20b_boundary_v1",
    )
    parser.add_argument(
        "--model-snapshot",
        default=(
            "/root/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/"
            "6cee5e81ee83917806bbde320786a8fb61efebee"
        ),
    )
    parser.add_argument("--forward-chunk-size", type=int, default=512)
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 15, 23])
    parser.add_argument("--minimum-free-gb", type=float, default=8.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = run_gpt_oss_activation_full(
        output_dir=args.output_dir,
        model_snapshot=args.model_snapshot,
        layers=args.layers,
        forward_chunk_size=args.forward_chunk_size,
        resume=not args.no_resume,
        minimum_free_bytes=int(args.minimum_free_gb * 1e9),
        limit=args.limit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
