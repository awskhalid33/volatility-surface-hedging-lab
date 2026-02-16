from datetime import date

from vol_surface_hedging_lab.data_io import load_option_quotes
from vol_surface_hedging_lab.rolling import (
    RollingRecalibrationConfig,
    run_rolling_recalibration_experiment,
)
from vol_surface_hedging_lab.synthetic_data import (
    SyntheticHistoryConfig,
    generate_historical_option_rows,
    write_option_rows_csv,
)


def test_rolling_recalibration_runs_and_hedging_reduces_dispersion(tmp_path):
    data_path = tmp_path / "historical.csv"
    synth_cfg = SyntheticHistoryConfig(
        start_date=date(2025, 1, 2),
        valuation_days=120,
        seed=17,
        low_regime_vol=0.20,
        high_regime_vol=0.20,
        transition_p00=1.0,
        transition_p11=1.0,
        noise_std=0.0,
    )
    rows = generate_historical_option_rows(synth_cfg)
    write_option_rows_csv(rows, data_path)

    quotes = load_option_quotes(data_path)
    rolling_cfg = RollingRecalibrationConfig(
        max_windows=12,
        max_rebalance_dates=80,
        transaction_cost_stock=0.0,
        transaction_cost_option=0.0,
    )
    result = run_rolling_recalibration_experiment(quotes, rolling_cfg)
    assert result["window_study"]["num_windows"] > 0
    unhedged_std = result["strategy_metrics"]["unhedged"]["std_pnl"]
    delta_std = result["strategy_metrics"]["delta"]["std_pnl"]
    assert delta_std <= unhedged_std + 1e-12
