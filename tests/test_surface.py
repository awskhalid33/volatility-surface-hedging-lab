from lse_fm_vol_project.svi import SVIParams
from lse_fm_vol_project.surface import SVISurfaceModel


def test_surface_implied_vol_positive_and_stable():
    surface = SVISurfaceModel(
        {
            0.25: SVIParams(a=0.01, b=0.08, rho=-0.2, m=0.0, sigma=0.2),
            0.50: SVIParams(a=0.015, b=0.10, rho=-0.25, m=0.0, sigma=0.22),
            1.00: SVIParams(a=0.02, b=0.12, rho=-0.30, m=0.0, sigma=0.25),
        }
    )

    iv_short = surface.implied_volatility(
        spot=100.0,
        strike=95.0,
        maturity=0.30,
        rate=0.02,
        dividend=0.0,
    )
    iv_long = surface.implied_volatility(
        spot=100.0,
        strike=105.0,
        maturity=0.90,
        rate=0.02,
        dividend=0.0,
    )
    assert iv_short > 0.0
    assert iv_long > 0.0
