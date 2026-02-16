#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path

from vol_surface_hedging_lab.data_io import load_option_quotes, write_json
from vol_surface_hedging_lab.model_comparison import (
    render_model_comparison_markdown,
    run_model_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SVI vs SABR volatility-model comparison."
    )
    parser.add_argument(
        "--input",
        default="data/sample_option_chain.csv",
        help="Input option-chain CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--valuation-date",
        default=None,
        help="Specific valuation date (YYYY-MM-DD). Default: latest date in input.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Calibration seed",
    )
    parser.add_argument(
        "--sabr-beta",
        type=float,
        default=1.0,
        help="SABR beta in [0,1]",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Number of CV folds for out-of-sample comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quotes = load_option_quotes(args.input)
    grouped: dict[str, list] = defaultdict(list)
    for q in quotes:
        grouped[q.valuation_date].append(q)
    available_dates = sorted(grouped.keys())
    if not available_dates:
        raise RuntimeError("No valuation dates found in input CSV")

    chosen_date = args.valuation_date or available_dates[-1]
    if chosen_date not in grouped:
        raise ValueError(
            f"valuation date {chosen_date} not found. Available: {available_dates[:3]}...{available_dates[-3:]}"
        )

    result = run_model_comparison(
        grouped[chosen_date],
        seed=args.seed,
        sabr_beta=args.sabr_beta,
        folds=args.folds,
    )
    report = render_model_comparison_markdown(result)

    json_path = output_dir / "model_comparison_results.json"
    report_path = output_dir / "model_comparison_report.md"
    write_json(result, json_path)
    report_path.write_text(report, encoding="utf-8")

    print(f"Valuation date used: {chosen_date}")
    print(f"Wrote JSON results to {json_path}")
    print(f"Wrote markdown report to {report_path}")


if __name__ == "__main__":
    main()
