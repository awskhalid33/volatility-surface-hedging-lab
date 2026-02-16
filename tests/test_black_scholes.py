from lse_fm_vol_project.black_scholes import (
    bs_call_delta,
    bs_call_price,
    bs_call_vega,
    implied_volatility_call,
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
