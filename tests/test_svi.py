import math

from vol_surface_hedging_lab.svi import SVIParams, fit_raw_svi, svi_total_variance


def test_svi_fit_on_synthetic_data():
    params_true = SVIParams(a=0.02, b=0.12, rho=-0.35, m=0.01, sigma=0.20)
    ks = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]
    ws = [svi_total_variance(k, params_true) for k in ks]

    fitted = fit_raw_svi(ks, ws, seed=13, random_trials=3000, local_iters=200)
    mse = sum((svi_total_variance(k, fitted) - w) ** 2 for k, w in zip(ks, ws)) / len(ks)
    rmse = math.sqrt(mse)
    assert rmse < 1e-3


def test_svi_fit_with_extreme_moneyness():
    params_true = SVIParams(a=0.018, b=0.14, rho=-0.45, m=-0.01, sigma=0.24)
    ks = [-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9]
    ws = [svi_total_variance(k, params_true) for k in ks]
    fitted = fit_raw_svi(ks, ws, seed=9, random_trials=2000, local_iters=160)
    mse = sum((svi_total_variance(k, fitted) - w) ** 2 for k, w in zip(ks, ws)) / len(ks)
    assert math.sqrt(mse) < 3e-3
