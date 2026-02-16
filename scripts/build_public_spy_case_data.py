#!/usr/bin/env python3
import argparse
import csv
import ssl
import statistics
import urllib.request
from datetime import date
from pathlib import Path

from vol_surface_hedging_lab.synthetic_data import write_option_rows_csv

PUBLIC_SOURCES = (
    (
        "2020-07-10",
        (
            "https://raw.githubusercontent.com/cantaro86/"
            "Financial-Models-Numerical-Methods/master/data/"
            "spy-options-exp-2020-07-10-weekly-show-all-stacked-07-05-2020.csv"
        ),
    ),
    (
        "2021-01-15",
        (
            "https://raw.githubusercontent.com/cantaro86/"
            "Financial-Models-Numerical-Methods/master/data/"
            "spy-options-exp-2021-01-15-weekly-show-all-stacked-07-05-2020.csv"
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a real SPY option snapshot case study from public CSV snapshots."
    )
    parser.add_argument(
        "--valuation-date",
        default="2020-07-05",
        help="Valuation date for the snapshot (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.001,
        help="Flat continuously-compounded risk-free rate.",
    )
    parser.add_argument(
        "--dividend",
        type=float,
        default=0.0,
        help="Flat continuously-compounded dividend yield.",
    )
    parser.add_argument(
        "--min-moneyness",
        type=float,
        default=0.70,
        help="Minimum strike/spot ratio retained.",
    )
    parser.add_argument(
        "--max-moneyness",
        type=float,
        default=1.35,
        help="Maximum strike/spot ratio retained.",
    )
    parser.add_argument(
        "--output",
        default="data/public_spy_snapshot_2020-07-05.csv",
        help="Output CSV path in project schema.",
    )
    parser.add_argument(
        "--min-days-to-expiry",
        type=int,
        default=2,
        help="Minimum days-to-expiry retained in the output dataset.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=20.0,
        help="HTTP timeout per source file.",
    )
    return parser.parse_args()


def _download_text(url: str, timeout_sec: float) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
            )
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return resp.read().decode("utf-8")
    except Exception:
        insecure = ssl._create_unverified_context()
        with urllib.request.urlopen(req, timeout=timeout_sec, context=insecure) as resp:
            return resp.read().decode("utf-8")


def _parse_call_rows(csv_text: str) -> list[tuple[float, float]]:
    reader = csv.DictReader(csv_text.splitlines())
    rows: list[tuple[float, float]] = []
    for row in reader:
        if (row.get("Type") or "").strip().lower() != "call":
            continue
        strike_raw = row.get("Strike") or ""
        midpoint_raw = row.get("Midpoint") or ""
        try:
            strike = float(strike_raw)
            midpoint = float(midpoint_raw)
        except ValueError:
            continue
        if strike <= 0.0 or midpoint <= 0.0:
            continue
        rows.append((strike, midpoint))
    rows.sort(key=lambda x: x[0])
    return rows


def _estimate_spot(near_expiry_calls: list[tuple[float, float]]) -> float:
    if len(near_expiry_calls) < 5:
        raise RuntimeError("Not enough near-expiry call rows to estimate spot")
    n = max(5, min(14, len(near_expiry_calls) // 5))
    deepest = near_expiry_calls[:n]
    estimates = [strike + midpoint for strike, midpoint in deepest]
    spot = float(statistics.median(estimates))
    if spot <= 0.0:
        raise RuntimeError("Estimated spot is non-positive")
    return spot


def _convert_rows(
    valuation_date: date,
    expiry: date,
    spot: float,
    calls: list[tuple[float, float]],
    rate: float,
    dividend: float,
    min_moneyness: float,
    max_moneyness: float,
    min_days_to_expiry: int,
) -> list[dict[str, str]]:
    maturity_days = (expiry - valuation_date).days
    if maturity_days <= min_days_to_expiry:
        return []
    maturity = maturity_days / 365.0

    out: list[dict[str, str]] = []
    for strike, midpoint in calls:
        moneyness = strike / spot
        if moneyness < min_moneyness or moneyness > max_moneyness:
            continue
        out.append(
            {
                "valuation_date": valuation_date.isoformat(),
                "expiry": expiry.isoformat(),
                "maturity": f"{maturity:.10f}",
                "spot": f"{spot:.8f}",
                "rate": f"{rate:.8f}",
                "dividend": f"{dividend:.8f}",
                "strike": f"{strike:.6f}",
                "call_mid": f"{midpoint:.10f}",
            }
        )
    return out


def main() -> None:
    args = parse_args()
    valuation_dt = date.fromisoformat(args.valuation_date)

    downloaded: dict[str, list[tuple[float, float]]] = {}
    for expiry_s, url in PUBLIC_SOURCES:
        text = _download_text(url, timeout_sec=args.timeout_sec)
        downloaded[expiry_s] = _parse_call_rows(text)

    near_expiry = min(PUBLIC_SOURCES, key=lambda p: p[0])[0]
    spot = _estimate_spot(downloaded[near_expiry])

    all_rows: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for expiry_s, _ in PUBLIC_SOURCES:
        expiry_dt = date.fromisoformat(expiry_s)
        converted = _convert_rows(
            valuation_date=valuation_dt,
            expiry=expiry_dt,
            spot=spot,
            calls=downloaded[expiry_s],
            rate=args.rate,
            dividend=args.dividend,
            min_moneyness=args.min_moneyness,
            max_moneyness=args.max_moneyness,
            min_days_to_expiry=args.min_days_to_expiry,
        )
        counts[expiry_s] = len(converted)
        all_rows.extend(converted)

    if not all_rows:
        raise RuntimeError("No rows were produced from public data sources")

    output_path = Path(args.output)
    write_option_rows_csv(all_rows, output_path)

    print(f"Estimated spot: {spot:.6f}")
    print(f"Output rows: {len(all_rows)}")
    for expiry_s in sorted(counts):
        print(f"  Expiry {expiry_s}: {counts[expiry_s]} rows")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
