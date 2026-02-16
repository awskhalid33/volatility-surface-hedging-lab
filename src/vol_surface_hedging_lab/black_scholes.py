import math


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_d1(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    vol: float,
) -> float:
    sqrt_t = math.sqrt(maturity)
    return (
        math.log(spot / strike) + (rate - dividend + 0.5 * vol * vol) * maturity
    ) / (vol * sqrt_t)


def bs_call_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    vol: float,
) -> float:
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if maturity <= 0.0:
        raise ValueError("maturity must be positive")
    if vol <= 0.0:
        raise ValueError("vol must be positive")

    sqrt_t = math.sqrt(maturity)
    d1 = _bs_d1(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend=dividend,
        vol=vol,
    )
    d2 = d1 - vol * sqrt_t
    discounted_spot = spot * math.exp(-dividend * maturity)
    discounted_strike = strike * math.exp(-rate * maturity)
    return discounted_spot * normal_cdf(d1) - discounted_strike * normal_cdf(d2)


def no_arbitrage_call_bounds(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
) -> tuple[float, float]:
    discounted_spot = spot * math.exp(-dividend * maturity)
    discounted_strike = strike * math.exp(-rate * maturity)
    lower = max(0.0, discounted_spot - discounted_strike)
    upper = discounted_spot
    return lower, upper


def implied_volatility_call(
    call_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    tol: float = 1e-9,
    max_iter: int = 300,
    vol_low: float = 1e-6,
    vol_high: float = 5.0,
) -> float:
    if call_price < 0.0:
        raise ValueError("call price must be non-negative")
    if vol_low <= 0.0:
        raise ValueError("vol_low must be positive")
    if vol_high <= vol_low:
        raise ValueError("vol_high must exceed vol_low")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    lower, upper = no_arbitrage_call_bounds(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend=dividend,
    )
    if call_price < lower - 1e-12 or call_price > upper + 1e-12:
        raise ValueError(
            "call price outside no-arbitrage bounds "
            f"[{lower:.10f}, {upper:.10f}] with price={call_price:.10f}"
        )
    if call_price <= lower + tol:
        return vol_low

    low = vol_low
    high = vol_high

    low_price = bs_call_price(spot, strike, maturity, rate, dividend, low)
    high_price = bs_call_price(spot, strike, maturity, rate, dividend, high)
    while high_price < call_price and high < 20.0:
        high *= 2.0
        high_price = bs_call_price(spot, strike, maturity, rate, dividend, high)

    if low_price > call_price + tol:
        return low
    if high_price < call_price - tol:
        return high
    if abs(low_price - call_price) < tol:
        return low
    if abs(high_price - call_price) < tol:
        return high

    # Fast path: Newton-Raphson with vega, always bracketed by [low, high].
    vol = 0.5 * (low + high)
    for _ in range(max_iter):
        price = bs_call_price(spot, strike, maturity, rate, dividend, vol)
        diff = price - call_price
        if abs(diff) < tol:
            return vol
        if diff > 0.0:
            high = min(high, vol)
        else:
            low = max(low, vol)

        vega = bs_call_vega(spot, strike, maturity, rate, dividend, vol)
        if vega > 1e-12:
            newton = vol - diff / vega
            if low < newton < high:
                next_vol = newton
            else:
                next_vol = 0.5 * (low + high)
        else:
            next_vol = 0.5 * (low + high)

        if abs(next_vol - vol) < tol * max(1.0, vol):
            vol = next_vol
            break
        vol = next_vol

    # Guaranteed convergence fallback.
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        mid_price = bs_call_price(spot, strike, maturity, rate, dividend, mid)
        diff = mid_price - call_price
        if abs(diff) < tol:
            return mid
        if diff > 0.0:
            high = mid
        else:
            low = mid

    return 0.5 * (low + high)


def bs_call_delta(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    vol: float,
) -> float:
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if maturity <= 0.0:
        raise ValueError("maturity must be positive")
    if vol <= 0.0:
        raise ValueError("vol must be positive")

    d1 = _bs_d1(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend=dividend,
        vol=vol,
    )
    return math.exp(-dividend * maturity) * normal_cdf(d1)


def bs_call_vega(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    vol: float,
) -> float:
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if maturity <= 0.0:
        raise ValueError("maturity must be positive")
    if vol <= 0.0:
        raise ValueError("vol must be positive")

    d1 = _bs_d1(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend=dividend,
        vol=vol,
    )
    return spot * math.exp(-dividend * maturity) * math.sqrt(maturity) * normal_pdf(d1)
