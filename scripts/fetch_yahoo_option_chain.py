#!/usr/bin/env python3
import argparse
from pathlib import Path

from vol_surface_hedging_lab.synthetic_data import write_option_rows_csv
from vol_surface_hedging_lab.yahoo_data import fetch_yahoo_option_chain_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch option-chain snapshots from Yahoo Finance and save to CSV."
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Ticker symbol, e.g. SPY",
    )
    parser.add_argument(
        "--output",
        default="data/live_option_chain.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--max-expiries",
        type=int,
        default=4,
        help="Number of expiries to fetch",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = fetch_yahoo_option_chain_rows(
        ticker=args.ticker,
        max_expiries=args.max_expiries,
        timeout_sec=args.timeout_sec,
    )
    write_option_rows_csv(rows, output_path)

    unique_expiries = sorted(set(r["expiry"] for r in rows))
    print(f"Ticker: {args.ticker.upper()}")
    print(f"Rows written: {len(rows)}")
    print(f"Expiries captured: {len(unique_expiries)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
