from vol_surface_hedging_lab.black_scholes import bs_call_price
from vol_surface_hedging_lab.model_comparison import run_model_comparison
from vol_surface_hedging_lab.types import OptionQuote


def _smile_vol(spot: float, strike: float, maturity: float) -> float:
    import math

    log_m = math.log(strike / spot)
    return max(0.08, 0.17 + 0.03 * maturity + 0.06 * abs(log_m) - 0.02 * log_m)


def test_model_comparison_outputs_summary():
    spot = 100.0
    rate = 0.02
    dividend = 0.0
    maturities = [0.25, 0.5]
    strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
    quotes = []

    for maturity in maturities:
        for strike in strikes:
            vol = _smile_vol(spot, strike, maturity)
            price = bs_call_price(
                spot=spot,
                strike=strike,
                maturity=maturity,
                rate=rate,
                dividend=dividend,
                vol=vol,
            )
            quotes.append(
                OptionQuote(
                    valuation_date="2026-02-15",
                    expiry="2026-12-31",
                    maturity=maturity,
                    spot=spot,
                    rate=rate,
                    dividend=dividend,
                    strike=strike,
                    call_mid=price,
                )
            )

    result = run_model_comparison(quotes, seed=7, sabr_beta=1.0, folds=3)
    assert result["metadata"]["num_maturities"] == 2
    assert result["summary"]["winner_in_sample"] in {"SVI", "SABR"}
    assert result["summary"]["avg_oos_cv_rmse_iv"]["svi"] is not None
    assert result["summary"]["avg_oos_cv_rmse_iv"]["sabr"] is not None
