import math

from lse_fm_vol_project.svi import SVIParams, fit_raw_svi, svi_total_variance


def test_svi_fit_on_synthetic_data():
    params_true = SVIParams(a=0.02, b=0.12, rho=-0.35, m=0.01, sigma=0.20)
    ks = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]
    ws = [svi_total_variance(k, params_true) for k in ks]

    fitted = fit_raw_svi(ks, ws, seed=13, random_trials=3000, local_iters=200)
    mse = sum((svi_total_variance(k, fitted) - w) ** 2 for k, w in zip(ks, ws)) / len(ks)
    rmse = math.sqrt(mse)
    assert rmse < 1e-3
