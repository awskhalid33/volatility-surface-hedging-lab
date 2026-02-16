from datetime import date

from vol_surface_hedging_lab.backtest import HedgingConfig, run_hedging_experiment
from vol_surface_hedging_lab.black_scholes import bs_call_price
from vol_surface_hedging_lab.data_io import load_option_quotes
from vol_surface_hedging_lab.model_comparison import run_model_comparison
from vol_surface_hedging_lab.pipeline import run_surface_pipeline
from vol_surface_hedging_lab.rolling import (
    RollingRecalibrationConfig,
    run_rolling_recalibration_experiment,
)
from vol_surface_hedging_lab.surface import SVISurfaceModel
from vol_surface_hedging_lab.synthetic_data import (
    SyntheticHistoryConfig,
    generate_historical_option_rows,
    write_option_rows_csv,
)
from vol_surface_hedging_lab.visualisation import generate_visual_artifacts


def _build_sample_rows() -> list[dict[str, str]]:
    import math
    from datetime import timedelta

    valuation = date(2026, 2, 15)
    spot = 100.0
    rate = 0.02
    dividend = 0.0
    maturity_days = [30, 90, 180, 365]
    strikes = [70, 80, 90, 95, 100, 105, 110, 120, 130]

    rows = []
    for days in maturity_days:
        maturity = days / 365.0
        expiry = valuation + timedelta(days=days)
        for strike in strikes:
            log_m = math.log(strike / spot)
            vol = max(0.08, min(0.16 + 0.03 * maturity**0.5 - 0.08 * log_m + 0.22 * abs(log_m), 1.2))
            call_mid = bs_call_price(
                spot=spot,
                strike=float(strike),
                maturity=maturity,
                rate=rate,
                dividend=dividend,
                vol=vol,
            )
            rows.append(
                {
                    "valuation_date": valuation.isoformat(),
                    "expiry": expiry.isoformat(),
                    "maturity": f"{maturity:.10f}",
                    "spot": f"{spot:.6f}",
                    "rate": f"{rate:.8f}",
                    "dividend": f"{dividend:.8f}",
                    "strike": f"{float(strike):.6f}",
                    "call_mid": f"{call_mid:.10f}",
                }
            )
    return rows


def test_end_to_end_pipeline_small_dataset(tmp_path):
    sample_path = tmp_path / "sample.csv"
    write_option_rows_csv(_build_sample_rows(), sample_path)
    sample_quotes = load_option_quotes(sample_path)

    surface = run_surface_pipeline(sample_quotes, seed=11)
    model = run_model_comparison(sample_quotes, seed=17, sabr_beta=1.0, folds=3)

    surface_model = SVISurfaceModel.from_pipeline_result(surface)
    spot0 = sample_quotes[0].spot
    hedge_cfg = HedgingConfig(
        spot0=spot0,
        rate=sample_quotes[0].rate,
        dividend=sample_quotes[0].dividend,
        target_strike=spot0,
        hedge_strike=1.10 * spot0,
        target_maturity=0.5,
        hedge_maturity=0.75,
        steps=40,
        paths=80,
        seed=22,
    )
    hedging = run_hedging_experiment(surface_model, hedge_cfg)
    assert "terminal_pnl_by_strategy" in hedging

    hist_path = tmp_path / "historical.csv"
    hist_cfg = SyntheticHistoryConfig(
        start_date=date(2025, 1, 2),
        valuation_days=80,
        seed=55,
    )
    write_option_rows_csv(generate_historical_option_rows(hist_cfg), hist_path)
    hist_quotes = load_option_quotes(hist_path)
    rolling_cfg = RollingRecalibrationConfig(max_windows=6, max_rebalance_dates=45)
    rolling = run_rolling_recalibration_experiment(hist_quotes, rolling_cfg)

    generated = generate_visual_artifacts(
        surface_result=surface,
        model_result=model,
        hedging_result=hedging,
        rolling_result=rolling,
        output_dir=tmp_path / "outputs",
    )
    assert len(generated) >= 4
    assert (tmp_path / "outputs" / "fig_model_rmse.svg").exists()
