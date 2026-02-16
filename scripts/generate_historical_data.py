#!/usr/bin/env python3
import argparse
from datetime import date

from lse_fm_vol_project.synthetic_data import (
    SyntheticHistoryConfig,
    generate_historical_option_rows,
    write_option_rows_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic historical option-chain snapshots."
    )
    parser.add_argument(
        "--output",
        default="data/historical_option_chain.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--valuation-days",
        type=int,
        default=180,
        help="Number of business-day snapshots to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed",
    )
    parser.add_argument(
        "--start-date",
        default="2025-01-02",
        help="Start date (YYYY-MM-DD)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SyntheticHistoryConfig(
        start_date=date.fromisoformat(args.start_date),
        valuation_days=args.valuation_days,
        seed=args.seed,
    )
    rows = generate_historical_option_rows(cfg)
    write_option_rows_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
