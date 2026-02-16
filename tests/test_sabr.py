import math

from vol_surface_hedging_lab.sabr import fit_sabr_from_observations, sabr_implied_vol


def test_sabr_fit_roundtrip_for_generated_smile():
    forward = 100.0
    maturity = 0.5
    strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]

    ivs = []
    for k in strikes:
        log_m = math.log(k / forward)
        ivs.append(0.18 + 0.04 * abs(log_m) - 0.01 * log_m)

    params, rmse = fit_sabr_from_observations(
        forward=forward,
        strikes=strikes,
        maturity=maturity,
        implied_vols=ivs,
        beta=1.0,
        seed=31,
    )
    fitted = [sabr_implied_vol(forward, k, maturity, params) for k in strikes]
    mse = sum((a - b) ** 2 for a, b in zip(fitted, ivs)) / len(strikes)
    assert rmse < 0.05
    assert mse < 0.0025
