#!/usr/bin/env python3
import csv
import math
from datetime import date, timedelta
from pathlib import Path

from lse_fm_vol_project.black_scholes import bs_call_price


def smile_vol(spot: float, strike: float, maturity: float) -> float:
    x = strike / spot
    log_m = 0.0 if x <= 0.0 else math.log(x)
    base = 0.16 + 0.03 * maturity**0.5
    skew = -0.08 * log_m
    curvature = 0.22 * abs(log_m)
    vol = base + skew + curvature
    return max(0.08, min(vol, 1.20))


def build_rows() -> list[dict]:
    valuation = date(2026, 2, 15)
    spot = 100.0
    rate = 0.02
    dividend = 0.0
    maturity_days = [30, 90, 180, 365]
    strikes = [70, 80, 90, 95, 100, 105, 110, 120, 130]

    rows = []
    for days in maturity_days:
        maturity = days / 365.0
        expiry = valuation + timedelta(days=days)
        for strike in strikes:
            vol = smile_vol(spot=spot, strike=float(strike), maturity=maturity)
            call_mid = bs_call_price(
                spot=spot,
                strike=float(strike),
                maturity=maturity,
                rate=rate,
                dividend=dividend,
                vol=vol,
            )
            rows.append(
                {
                    "valuation_date": valuation.isoformat(),
                    "expiry": expiry.isoformat(),
                    "maturity": f"{maturity:.10f}",
                    "spot": f"{spot:.6f}",
                    "rate": f"{rate:.8f}",
                    "dividend": f"{dividend:.8f}",
                    "strike": f"{float(strike):.6f}",
                    "call_mid": f"{call_mid:.10f}",
                }
            )
    return rows


def main() -> None:
    out_path = Path("data/sample_option_chain.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
