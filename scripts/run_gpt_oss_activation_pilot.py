#!/usr/bin/env python3
"""Run the four-state GPT-OSS boundary-activation calibration pilot."""

from __future__ import annotations

import argparse
import json

from reveng.experiments.gpt_oss_activation_pilot import run_gpt_oss_activation_pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="outputs/activation_collection_pilot/gpt_oss_20b_boundary_v1",
    )
    parser.add_argument(
        "--model-snapshot",
        default=(
            "/root/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/"
            "6cee5e81ee83917806bbde320786a8fb61efebee"
        ),
    )
    parser.add_argument("--benchmark-chunk-sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--vram-limit-gib", type=float, default=22.0)
    args = parser.parse_args()
    result = run_gpt_oss_activation_pilot(
        output_dir=args.output_dir,
        model_snapshot=args.model_snapshot,
        benchmark_chunk_sizes=args.benchmark_chunk_sizes,
        vram_limit_gib=args.vram_limit_gib,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
