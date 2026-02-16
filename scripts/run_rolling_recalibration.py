#!/usr/bin/env python3
import argparse
from pathlib import Path

from vol_surface_hedging_lab.data_io import load_option_quotes, write_json
from vol_surface_hedging_lab.rolling import (
    RollingRecalibrationConfig,
    render_rolling_markdown,
    run_rolling_recalibration_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rolling recalibration and out-of-sample hedging experiment."
    )
    parser.add_argument(
        "--input",
        default="data/historical_option_chain.csv",
        help="Path to historical option-chain CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for experiment artifacts",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=40,
        help="Maximum number of rolling windows",
    )
    parser.add_argument(
        "--max-rebalance-dates",
        type=int,
        default=120,
        help="Maximum number of rebalance dates per window",
    )
    parser.add_argument(
        "--calibration-seed",
        type=int,
        default=11,
        help="Seed for SVI calibration initializations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quotes = load_option_quotes(args.input)
    cfg = RollingRecalibrationConfig(
        max_windows=args.max_windows,
        max_rebalance_dates=args.max_rebalance_dates,
        calibration_seed=args.calibration_seed,
    )
    result = run_rolling_recalibration_experiment(quotes, cfg)
    report = render_rolling_markdown(result)

    json_path = output_dir / "rolling_recalibration_results.json"
    report_path = output_dir / "rolling_recalibration_report.md"
    write_json(result, json_path)
    report_path.write_text(report, encoding="utf-8")

    print(f"Wrote JSON results to {json_path}")
    print(f"Wrote markdown report to {report_path}")


if __name__ == "__main__":
    main()
