import csv
import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from .black_scholes import bs_call_price


@dataclass(frozen=True)
class SyntheticHistoryConfig:
    start_date: date = date(2025, 1, 2)
    valuation_days: int = 180
    spot0: float = 100.0
    rate: float = 0.02
    dividend: float = 0.0
    low_regime_vol: float = 0.16
    high_regime_vol: float = 0.32
    transition_p00: float = 0.96
    transition_p11: float = 0.84
    initial_regime: int = 0
    seed: int = 123
    expiries_days: tuple[int, ...] = (120, 210, 300, 390)
    strikes: tuple[int, ...] = (70, 80, 90, 95, 100, 105, 110, 120, 130)
    noise_std: float = 0.004


def _iter_business_days(start: date, count: int) -> list[date]:
    days: list[date] = []
    cursor = start
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _market_surface_vol(
    base_vol: float,
    spot: float,
    strike: float,
    maturity: float,
) -> float:
    log_m = math.log(strike / spot)
    skew = -0.12 * log_m
    curvature = 0.20 * abs(log_m)
    term = 0.02 * math.exp(-2.0 * maturity)
    vol = base_vol * (1.0 + skew + curvature) + term
    return max(0.05, min(2.0, vol))


def _next_regime(current_regime: int, cfg: SyntheticHistoryConfig, rng: random.Random) -> int:
    u = rng.random()
    if current_regime == 0:
        return 0 if u < cfg.transition_p00 else 1
    return 1 if u < cfg.transition_p11 else 0


def generate_historical_option_rows(cfg: SyntheticHistoryConfig) -> list[dict[str, str]]:
    rng = random.Random(cfg.seed)
    valuation_dates = _iter_business_days(cfg.start_date, cfg.valuation_days)
    fixed_expiries = [cfg.start_date + timedelta(days=d) for d in cfg.expiries_days]

    spot = cfg.spot0
    regime = cfg.initial_regime
    rows: list[dict[str, str]] = []
    prev_date = valuation_dates[0]

    for idx, valuation in enumerate(valuation_dates):
        if idx > 0:
            regime = _next_regime(regime, cfg, rng)
            dt = max((valuation - prev_date).days / 365.0, 1.0 / 365.0)
            sqrt_dt = math.sqrt(dt)
            base_vol = cfg.low_regime_vol if regime == 0 else cfg.high_regime_vol
            z = rng.gauss(0.0, 1.0)
            spot *= math.exp(
                (cfg.rate - cfg.dividend - 0.5 * base_vol * base_vol) * dt
                + base_vol * sqrt_dt * z
            )
        prev_date = valuation

        for expiry in fixed_expiries:
            days_to_expiry = (expiry - valuation).days
            if days_to_expiry <= 7:
                continue
            maturity = days_to_expiry / 365.0
            base_vol = cfg.low_regime_vol if regime == 0 else cfg.high_regime_vol

            for strike_int in cfg.strikes:
                strike = float(strike_int)
                vol = _market_surface_vol(
                    base_vol=base_vol,
                    spot=spot,
                    strike=strike,
                    maturity=maturity,
                )
                noisy_vol = max(0.03, vol * (1.0 + cfg.noise_std * rng.gauss(0.0, 1.0)))
                call_mid = bs_call_price(
                    spot=spot,
                    strike=strike,
                    maturity=maturity,
                    rate=cfg.rate,
                    dividend=cfg.dividend,
                    vol=noisy_vol,
                )
                rows.append(
                    {
                        "valuation_date": valuation.isoformat(),
                        "expiry": expiry.isoformat(),
                        "maturity": f"{maturity:.10f}",
                        "spot": f"{spot:.8f}",
                        "rate": f"{cfg.rate:.8f}",
                        "dividend": f"{cfg.dividend:.8f}",
                        "strike": f"{strike:.6f}",
                        "call_mid": f"{call_mid:.10f}",
                    }
                )

    return rows


def write_option_rows_csv(rows: list[dict[str, str]], output_path: str | Path) -> None:
    if not rows:
        raise ValueError("rows cannot be empty")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
