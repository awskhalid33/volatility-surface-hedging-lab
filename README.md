# Volatility Surface and Dynamic Hedging Lab

Personal quantitative research project for option-surface modelling, model comparison, dynamic hedging, and rolling out-of-sample evaluation.

## Project Overview

This repository builds an end-to-end derivatives research pipeline:

1. Convert option prices to implied volatility.
2. Run static no-arbitrage checks.
3. Calibrate smile models (`SVI`, `SABR`) using optimisation-based fitting.
4. Compare models in-sample and out-of-sample.
5. Backtest hedging policies (`unhedged`, `delta`, `delta-vega`) with transaction costs.
6. Recalibrate surfaces through rolling windows and evaluate stability.
7. Generate visual artefacts (SVG plots) alongside JSON and Markdown reports.

## What Was Upgraded

- Replaced random-search-only calibration with bounded Levenberg–Marquardt optimisation (multi-start).
- Refactored duplicated hedging/statistics logic into a shared module.
- Added visualisation outputs:
  - smile fit chart
  - model RMSE comparison chart
  - terminal P&L histograms
  - rolling calibration RMSE trend
- Added stronger tests:
  - near-ATM SABR stability case
  - extreme-moneyness SVI case
  - integration test covering the full workflow on a compact dataset

## Synthetic Benchmark Results

These figures come from the latest deterministic synthetic run:

```bash
PYTHONPATH=src python3 scripts/run_full_study.py
```

### 1) Static arbitrage diagnostics

- Monotonicity violations: `0`
- Convexity violations: `0`
- Calendar total-variance violations: `0`

### 2) Model comparison (`SVI` vs `SABR`, 3-fold CV)

- In-sample RMSE (IV): `SVI=0.00025257`, `SABR=0.00406937`
- Out-of-sample CV RMSE (IV): `SVI=0.00267168`, `SABR=0.00666528`
- In-sample winner: `SVI`
- Out-of-sample winner: `SVI`

### 3) Dynamic hedging (400 paths)

- `delta` standard-deviation reduction vs unhedged: `89.65%`
- `delta-vega` standard-deviation reduction vs unhedged: `86.38%`
- ES 95%:
  - unhedged: `-27.406009`
  - delta: `-4.111110`
  - delta-vega: `-2.749126`

### 4) Rolling recalibration (25 windows)

- Average calibration RMSE(w): `0.00013942`
- Standard-deviation reduction vs unhedged:
  - delta: `41.65%`
  - delta-vega: `88.95%`

## Synthetic vs Live Results

- The benchmark numbers above are **synthetic** and deterministic (reproducible).
- A dedicated **live snapshot** workflow is included for real market data.
- Keep synthetic and live reports separate when presenting conclusions.

## One-Command Reproduction

Run the full synthetic study:

```bash
PYTHONPATH=src python3 scripts/run_full_study.py
```

Produced reports:

- `outputs/surface_report.md`
- `outputs/model_comparison_report.md`
- `outputs/hedging_backtest_report.md`
- `outputs/rolling_recalibration_report.md`

Produced visual artefacts:

- `outputs/fig_smile_fit.svg`
- `outputs/fig_model_rmse.svg`
- `outputs/fig_terminal_pnl_hist_unhedged.svg`
- `outputs/fig_terminal_pnl_hist_delta.svg`
- `outputs/fig_terminal_pnl_hist_delta-vega.svg`
- `outputs/fig_rolling_rmse_trend.svg`

## Live Snapshot Workflow

Run a live single-snapshot study from Yahoo Finance:

```bash
PYTHONPATH=src python3 scripts/run_live_snapshot_study.py \
  --ticker SPY \
  --output-dir outputs \
  --max-expiries 4 \
  --paths 300 \
  --steps 120
```

This produces:

- `outputs/live_surface_report.md`
- `outputs/live_model_comparison_report.md`
- `outputs/live_hedging_backtest_report.md`

## Component Commands

Generate synthetic sample chain:

```bash
PYTHONPATH=src python3 scripts/generate_sample_data.py
```

Surface report:

```bash
PYTHONPATH=src python3 scripts/run_research_pipeline.py \
  --input data/sample_option_chain.csv \
  --output-dir outputs
```

Model comparison report:

```bash
PYTHONPATH=src python3 scripts/run_model_comparison.py \
  --input data/sample_option_chain.csv \
  --output-dir outputs
```

Hedging report:

```bash
PYTHONPATH=src python3 scripts/run_hedging_backtest.py \
  --input data/sample_option_chain.csv \
  --output-dir outputs
```

Rolling study:

```bash
PYTHONPATH=src python3 scripts/generate_historical_data.py \
  --output data/historical_option_chain.csv \
  --valuation-days 180

PYTHONPATH=src python3 scripts/run_rolling_recalibration.py \
  --input data/historical_option_chain.csv \
  --output-dir outputs
```

Generate charts from existing JSON outputs:

```bash
PYTHONPATH=src python3 scripts/generate_visuals.py --output-dir outputs
```

## Data Format

Required CSV fields:

- `valuation_date` (YYYY-MM-DD)
- `expiry` (YYYY-MM-DD)
- `maturity` (years)
- `spot`
- `rate` (continuous risk-free rate)
- `dividend` (continuous dividend yield)
- `strike`
- `call_mid`

## Test Suite

Run tests:

```bash
PYTHONPATH=src pytest -q
```

The suite includes unit tests, calibration edge cases, and a compact end-to-end integration test.

## Current Limitations

- Synthetic history is still the default benchmark dataset.
- Live-data workflow is snapshot-based rather than a long historical archive.
- `SABR` calibration currently uses fixed `beta`.
- Microstructure details (full bid-ask filtering, liquidity constraints, trading calendars) are simplified.
