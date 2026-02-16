#!/usr/bin/env python3
from pathlib import Path

from vol_surface_hedging_lab.backtest import (
    HedgingConfig,
    render_backtest_markdown,
    run_hedging_experiment,
)
from vol_surface_hedging_lab.data_io import load_option_quotes, write_json
from vol_surface_hedging_lab.black_scholes import bs_call_price
from vol_surface_hedging_lab.model_comparison import (
    render_model_comparison_markdown,
    run_model_comparison,
)
from vol_surface_hedging_lab.pipeline import render_markdown_report, run_surface_pipeline
from vol_surface_hedging_lab.rolling import (
    RollingRecalibrationConfig,
    render_rolling_markdown,
    run_rolling_recalibration_experiment,
)
from vol_surface_hedging_lab.surface import SVISurfaceModel
from vol_surface_hedging_lab.synthetic_data import (
    SyntheticHistoryConfig,
    generate_historical_option_rows,
    write_option_rows_csv,
)
from vol_surface_hedging_lab.visualisation import generate_visual_artifacts


def _write_rows(rows: list[dict[str, str]], path: Path) -> None:
    write_option_rows_csv(rows, path)


def _build_sample_rows() -> list[dict[str, str]]:
    from datetime import date, timedelta
    import math

    valuation = date(2026, 2, 15)
    spot = 100.0
    rate = 0.02
    dividend = 0.0
    maturity_days = [30, 90, 180, 365]
    strikes = [70, 80, 90, 95, 100, 105, 110, 120, 130]

    def smile_vol(strike: float, maturity: float) -> float:
        log_m = math.log(strike / spot)
        base = 0.16 + 0.03 * maturity**0.5
        skew = -0.08 * log_m
        curvature = 0.22 * abs(log_m)
        return max(0.08, min(base + skew + curvature, 1.20))

    rows = []
    for days in maturity_days:
        maturity = days / 365.0
        expiry = valuation + timedelta(days=days)
        for strike in strikes:
            vol = smile_vol(float(strike), maturity)
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


def main() -> None:
    root = Path(".")
    data_dir = root / "data"
    outputs_dir = root / "outputs"
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1) sample cross-sectional data
    sample_path = data_dir / "sample_option_chain.csv"
    _write_rows(_build_sample_rows(), sample_path)
    sample_quotes = load_option_quotes(sample_path)

    # 2) surface calibration report
    surface_result = run_surface_pipeline(sample_quotes, seed=11)
    write_json(surface_result, outputs_dir / "surface_results.json")
    (outputs_dir / "surface_report.md").write_text(
        render_markdown_report(surface_result),
        encoding="utf-8",
    )

    # 3) model comparison report
    model_result = run_model_comparison(sample_quotes, seed=17, sabr_beta=1.0, folds=3)
    write_json(model_result, outputs_dir / "model_comparison_results.json")
    (outputs_dir / "model_comparison_report.md").write_text(
        render_model_comparison_markdown(model_result),
        encoding="utf-8",
    )

    # 4) dynamic backtest report
    surface = SVISurfaceModel.from_pipeline_result(surface_result)
    spot0 = sample_quotes[0].spot
    maturities = sorted(set(q.maturity for q in sample_quotes))
    target_maturity = min(maturities, key=lambda t: abs(t - 0.5))
    hedge_maturity = max(target_maturity + 0.20, min(1.0, target_maturity * 1.5))
    hedge_cfg = HedgingConfig(
        spot0=spot0,
        rate=sample_quotes[0].rate,
        dividend=sample_quotes[0].dividend,
        target_strike=spot0,
        hedge_strike=1.10 * spot0,
        target_maturity=target_maturity,
        hedge_maturity=hedge_maturity,
        steps=126,
        paths=400,
        seed=21,
    )
    hedge_result = run_hedging_experiment(surface, hedge_cfg)
    write_json(hedge_result, outputs_dir / "hedging_backtest_results.json")
    (outputs_dir / "hedging_backtest_report.md").write_text(
        render_backtest_markdown(hedge_result),
        encoding="utf-8",
    )

    # 5) historical rolling study report
    hist_path = data_dir / "historical_option_chain.csv"
    hist_cfg = SyntheticHistoryConfig(valuation_days=180, seed=123)
    hist_rows = generate_historical_option_rows(hist_cfg)
    write_option_rows_csv(hist_rows, hist_path)
    hist_quotes = load_option_quotes(hist_path)
    rolling_cfg = RollingRecalibrationConfig(max_windows=25, max_rebalance_dates=100)
    rolling_result = run_rolling_recalibration_experiment(hist_quotes, rolling_cfg)
    write_json(rolling_result, outputs_dir / "rolling_recalibration_results.json")
    (outputs_dir / "rolling_recalibration_report.md").write_text(
        render_rolling_markdown(rolling_result),
        encoding="utf-8",
    )

    generated_figures = generate_visual_artifacts(
        surface_result=surface_result,
        model_result=model_result,
        hedging_result=hedge_result,
        rolling_result=rolling_result,
        output_dir=outputs_dir,
    )

    print("Wrote all experiment artefacts to outputs/")
    print(f"Generated {len(generated_figures)} visual files.")


if __name__ == "__main__":
    main()
