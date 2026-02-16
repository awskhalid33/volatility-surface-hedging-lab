#!/usr/bin/env python3
import argparse
from pathlib import Path

from vol_surface_hedging_lab.data_io import load_option_quotes, write_json
from vol_surface_hedging_lab.pipeline import render_markdown_report, run_surface_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run volatility surface research pipeline."
    )
    parser.add_argument(
        "--input",
        default="data/sample_option_chain.csv",
        help="Path to option-chain CSV input",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for report artifacts",
    )
    parser.add_argument(
        "--seed",
        default=11,
        type=int,
        help="Random seed for SVI calibration initialization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quotes = load_option_quotes(args.input)
    result = run_surface_pipeline(quotes=quotes, seed=args.seed)
    report = render_markdown_report(result)

    json_path = output_dir / "surface_results.json"
    report_path = output_dir / "surface_report.md"
    write_json(result, json_path)
    report_path.write_text(report, encoding="utf-8")

    print(f"Wrote JSON results to {json_path}")
    print(f"Wrote markdown report to {report_path}")


if __name__ == "__main__":
    main()
