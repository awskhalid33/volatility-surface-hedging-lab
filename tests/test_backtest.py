from lse_fm_vol_project.backtest import HedgingConfig, run_hedging_experiment
from lse_fm_vol_project.svi import SVIParams
from lse_fm_vol_project.surface import SVISurfaceModel


def test_delta_hedging_reduces_terminal_pnl_std_in_controlled_case():
    surface = SVISurfaceModel(
        {
            0.25: SVIParams(a=0.01, b=0.09, rho=-0.2, m=0.0, sigma=0.2),
            0.50: SVIParams(a=0.015, b=0.11, rho=-0.25, m=0.0, sigma=0.22),
            1.00: SVIParams(a=0.02, b=0.13, rho=-0.30, m=0.0, sigma=0.25),
        }
    )
    cfg = HedgingConfig(
        spot0=100.0,
        rate=0.01,
        dividend=0.0,
        target_strike=100.0,
        hedge_strike=110.0,
        target_maturity=0.5,
        hedge_maturity=0.8,
        steps=63,
        paths=200,
        seed=42,
        low_regime_vol=0.20,
        high_regime_vol=0.20,
        transition_p00=1.0,
        transition_p11=1.0,
        initial_regime=0,
        transaction_cost_stock=0.0,
        transaction_cost_option=0.0,
    )
    result = run_hedging_experiment(surface=surface, cfg=cfg)
    unhedged_std = result["strategy_metrics"]["unhedged"]["std_pnl"]
    delta_std = result["strategy_metrics"]["delta"]["std_pnl"]
    assert delta_std < unhedged_std
