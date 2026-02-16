import pytest

from vol_surface_hedging_lab.black_scholes import (
    bs_call_delta,
    bs_call_price,
    bs_call_vega,
    implied_volatility_call,
    no_arbitrage_call_bounds,
)


def test_implied_vol_roundtrip():
    spot = 100.0
    strike = 105.0
    maturity = 0.75
    rate = 0.02
    dividend = 0.0
    vol = 0.28

    price = bs_call_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend=dividend,
        vol=vol,
    )

    solved = implied_volatility_call(
        call_price=price,
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend=dividend,
    )
    assert abs(solved - vol) < 1e-6


def test_delta_matches_finite_difference():
    spot = 100.0
    strike = 100.0
    maturity = 0.5
    rate = 0.01
    dividend = 0.0
    vol = 0.25
    h = 1e-4

    analytical = bs_call_delta(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend=dividend,
        vol=vol,
    )
    bumped_up = bs_call_price(
        spot=spot + h,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend=dividend,
        vol=vol,
    )
    bumped_down = bs_call_price(
        spot=spot - h,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend=dividend,
        vol=vol,
    )
    finite_diff = (bumped_up - bumped_down) / (2.0 * h)
    assert abs(analytical - finite_diff) < 1e-5


def test_vega_matches_finite_difference():
    spot = 95.0
    strike = 100.0
    maturity = 0.8
    rate = 0.02
    dividend = 0.0
    vol = 0.22
    h = 1e-5

    analytical = bs_call_vega(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend=dividend,
        vol=vol,
    )
    bumped_up = bs_call_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend=dividend,
        vol=vol + h,
    )
    bumped_down = bs_call_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend=dividend,
        vol=vol - h,
    )
    finite_diff = (bumped_up - bumped_down) / (2.0 * h)
    assert abs(analytical - finite_diff) < 1e-4


def test_implied_vol_returns_floor_near_intrinsic_bound():
    spot = 100.0
    strike = 140.0
    maturity = 0.4
    rate = 0.01
    dividend = 0.0
    lower, _ = no_arbitrage_call_bounds(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend=dividend,
    )
    solved = implied_volatility_call(
        call_price=lower + 1e-11,
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend=dividend,
    )
    assert solved <= 5e-6


def test_implied_vol_rejects_invalid_bracket_input():
    price = bs_call_price(spot=100.0, strike=100.0, maturity=1.0, rate=0.02, dividend=0.0, vol=0.2)
    with pytest.raises(ValueError):
        implied_volatility_call(
            call_price=price,
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.02,
            dividend=0.0,
            vol_low=0.3,
            vol_high=0.2,
        )
