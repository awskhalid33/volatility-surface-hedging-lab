#!/usr/bin/env python3
import argparse
from pathlib import Path

from lse_fm_vol_project.backtest import (
    HedgingConfig,
    render_backtest_markdown,
    run_hedging_experiment,
)
from lse_fm_vol_project.data_io import load_option_quotes, write_json
from lse_fm_vol_project.pipeline import run_surface_pipeline
from lse_fm_vol_project.surface import SVISurfaceModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dynamic hedging backtest from calibrated SVI surface."
    )
    parser.add_argument(
        "--input",
        default="data/sample_option_chain.csv",
        help="Path to option-chain CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for backtest artifacts",
    )
    parser.add_argument(
        "--paths",
        default=400,
        type=int,
        help="Number of Monte Carlo scenarios",
    )
    parser.add_argument(
        "--steps",
        default=126,
        type=int,
        help="Number of rebalancing steps over target maturity",
    )
    parser.add_argument(
        "--seed",
        default=21,
        type=int,
        help="Random seed for scenario generation",
    )
    return parser.parse_args()


def _select_maturities(observed_maturities: list[float]) -> tuple[float, float]:
    observed = sorted(set(observed_maturities))
    target = min(observed, key=lambda t: abs(t - 0.5))
    hedge = max(target + 0.20, min(1.0, target * 1.5))
    return target, hedge


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quotes = load_option_quotes(args.input)
    surface_result = run_surface_pipeline(quotes=quotes, seed=11)
    surface = SVISurfaceModel.from_pipeline_result(surface_result)

    spot0 = quotes[0].spot
    maturities = [q.maturity for q in quotes]
    target_maturity, hedge_maturity = _select_maturities(maturities)

    cfg = HedgingConfig(
        spot0=spot0,
        rate=quotes[0].rate,
        dividend=quotes[0].dividend,
        target_strike=spot0,
        hedge_strike=1.10 * spot0,
        target_maturity=target_maturity,
        hedge_maturity=hedge_maturity,
        steps=args.steps,
        paths=args.paths,
        seed=args.seed,
    )
    backtest_result = run_hedging_experiment(surface=surface, cfg=cfg)
    report = render_backtest_markdown(backtest_result)

    json_path = output_dir / "hedging_backtest_results.json"
    report_path = output_dir / "hedging_backtest_report.md"
    write_json(backtest_result, json_path)
    report_path.write_text(report, encoding="utf-8")

    print(f"Wrote JSON results to {json_path}")
    print(f"Wrote markdown report to {report_path}")


if __name__ == "__main__":
    main()
