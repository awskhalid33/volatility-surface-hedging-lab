#!/usr/bin/env python3
import argparse
from pathlib import Path

from vol_surface_hedging_lab.backtest import (
    HedgingConfig,
    render_backtest_markdown,
    run_hedging_experiment,
)
from vol_surface_hedging_lab.data_io import load_option_quotes, write_json
from vol_surface_hedging_lab.model_comparison import (
    render_model_comparison_markdown,
    run_model_comparison,
)
from vol_surface_hedging_lab.pipeline import render_markdown_report, run_surface_pipeline
from vol_surface_hedging_lab.surface import SVISurfaceModel
from vol_surface_hedging_lab.synthetic_data import write_option_rows_csv
from vol_surface_hedging_lab.visualisation import generate_visual_artifacts
from vol_surface_hedging_lab.yahoo_data import fetch_yahoo_option_chain_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch one live option snapshot and run surface/model/hedging analysis."
    )
    parser.add_argument(
        "--ticker",
        default="SPY",
        help="Ticker symbol, e.g. SPY (used for fetch mode and output labelling)",
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Path to an existing option CSV. If provided, Yahoo fetch is skipped.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--max-expiries", type=int, default=4, help="Number of expiries to fetch")
    parser.add_argument("--paths", type=int, default=300, help="MC paths for hedging simulation")
    parser.add_argument("--steps", type=int, default=120, help="Rebalancing steps for hedging simulation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.input_csv:
        live_csv = Path(args.input_csv)
        if not live_csv.exists():
            raise FileNotFoundError(f"input CSV not found: {live_csv}")
    else:
        rows = fetch_yahoo_option_chain_rows(ticker=args.ticker, max_expiries=args.max_expiries)
        live_csv = data_dir / f"live_option_chain_{args.ticker.upper()}.csv"
        write_option_rows_csv(rows, live_csv)
    quotes = load_option_quotes(live_csv)

    surface = run_surface_pipeline(quotes, seed=11)
    write_json(surface, out / "live_surface_results.json")
    (out / "live_surface_report.md").write_text(render_markdown_report(surface), encoding="utf-8")

    model = run_model_comparison(quotes, seed=17, sabr_beta=1.0, folds=3)
    write_json(model, out / "live_model_comparison_results.json")
    (out / "live_model_comparison_report.md").write_text(
        render_model_comparison_markdown(model),
        encoding="utf-8",
    )

    surface_model = SVISurfaceModel.from_pipeline_result(surface)
    spot0 = quotes[0].spot
    maturities = sorted(set(q.maturity for q in quotes))
    target_maturity = min(maturities, key=lambda t: abs(t - 0.5))
    hedge_maturity = max(target_maturity + 0.20, min(1.0, target_maturity * 1.5))
    hedge_cfg = HedgingConfig(
        spot0=spot0,
        rate=quotes[0].rate,
        dividend=quotes[0].dividend,
        target_strike=spot0,
        hedge_strike=1.10 * spot0,
        target_maturity=target_maturity,
        hedge_maturity=hedge_maturity,
        steps=args.steps,
        paths=args.paths,
        seed=27,
    )
    hedging = run_hedging_experiment(surface_model, hedge_cfg)
    write_json(hedging, out / "live_hedging_backtest_results.json")
    (out / "live_hedging_backtest_report.md").write_text(
        render_backtest_markdown(hedging),
        encoding="utf-8",
    )

    visuals = generate_visual_artifacts(
        surface_result=surface,
        model_result=model,
        hedging_result=hedging,
        rolling_result={"calibration_quality": {"per_date_rmse": {}}},
        output_dir=out,
        file_prefix="live_",
    )

    print(f"Ticker: {args.ticker.upper()}")
    if args.input_csv:
        print(f"Using input CSV: {live_csv}")
    else:
        print(f"Wrote live snapshot CSV: {live_csv}")
    print(f"Wrote live study artefacts in {out}")
    print(f"Generated {len(visuals)} live visual files.")


if __name__ == "__main__":
    main()
