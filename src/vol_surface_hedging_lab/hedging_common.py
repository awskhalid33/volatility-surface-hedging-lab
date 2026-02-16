import math

from .black_scholes import bs_call_delta, bs_call_vega
from .surface import SVISurfaceModel


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_desired_positions(
    strategy: str,
    surface: SVISurfaceModel,
    spot: float,
    rate: float,
    dividend: float,
    target_strike: float,
    hedge_strike: float,
    tau_target: float,
    tau_hedge: float,
    max_abs_stock_position: float,
    max_abs_hedge_option_position: float,
) -> tuple[float, float]:
    if strategy == "unhedged" or tau_target <= 0.0:
        return 0.0, 0.0

    tau_target_eff = max(1e-6, tau_target)
    model_vol_target = surface.implied_volatility(
        spot=spot,
        strike=target_strike,
        maturity=tau_target_eff,
        rate=rate,
        dividend=dividend,
    )
    delta_target = bs_call_delta(
        spot=spot,
        strike=target_strike,
        maturity=tau_target_eff,
        rate=rate,
        dividend=dividend,
        vol=model_vol_target,
    )

    if strategy == "delta":
        return clamp(delta_target, -max_abs_stock_position, max_abs_stock_position), 0.0

    if strategy != "delta-vega":
        raise ValueError(f"Unknown strategy: {strategy}")

    tau_hedge_eff = max(1e-6, tau_hedge)
    model_vol_hedge = surface.implied_volatility(
        spot=spot,
        strike=hedge_strike,
        maturity=tau_hedge_eff,
        rate=rate,
        dividend=dividend,
    )
    vega_target = bs_call_vega(
        spot=spot,
        strike=target_strike,
        maturity=tau_target_eff,
        rate=rate,
        dividend=dividend,
        vol=model_vol_target,
    )
    delta_hedge = bs_call_delta(
        spot=spot,
        strike=hedge_strike,
        maturity=tau_hedge_eff,
        rate=rate,
        dividend=dividend,
        vol=model_vol_hedge,
    )
    vega_hedge = bs_call_vega(
        spot=spot,
        strike=hedge_strike,
        maturity=tau_hedge_eff,
        rate=rate,
        dividend=dividend,
        vol=model_vol_hedge,
    )
    if abs(vega_hedge) < 1e-8:
        return clamp(delta_target, -max_abs_stock_position, max_abs_stock_position), 0.0

    hedge_units = clamp(
        vega_target / vega_hedge,
        -max_abs_hedge_option_position,
        max_abs_hedge_option_position,
    )
    stock_units = clamp(
        delta_target - hedge_units * delta_hedge,
        -max_abs_stock_position,
        max_abs_stock_position,
    )
    return stock_units, hedge_units


def quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def summary_stats(values: list[float]) -> dict:
    if not values:
        raise ValueError("values must not be empty")
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(variance)
    ordered = sorted(values)
    var95 = quantile(ordered, 0.05)
    tail = [x for x in ordered if x <= var95]
    es95 = (sum(tail) / len(tail)) if tail else var95
    return {
        "mean_pnl": mean,
        "std_pnl": std,
        "min_pnl": ordered[0],
        "max_pnl": ordered[-1],
        "var_95": var95,
        "expected_shortfall_95": es95,
        "positive_ratio": sum(1 for x in values if x > 0.0) / n,
    }
