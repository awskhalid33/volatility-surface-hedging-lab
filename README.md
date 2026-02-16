# Volatility Surface and Dynamic Hedging Research Platform

This repository is a complete quant project designed to demonstrate MSc Financial Mathematics readiness: no-arbitrage surface construction, model calibration, model comparison, dynamic hedging, and rolling out-of-sample evaluation.

## Executive Summary

The project builds and tests an end-to-end derivatives research stack:

1. Convert option prices into implied-volatility surfaces.
2. Enforce static no-arbitrage diagnostics.
3. Calibrate and compare `SVI` and `SABR` smiles.
4. Backtest `unhedged`, `delta`, and `delta-vega` hedging with transaction costs.
5. Recalibrate surfaces across rolling historical snapshots and evaluate out-of-sample robustness.

This README contains the project summary and empirical findings in place of a separate research paper.

## Key Results

Results below come from the current generated artifacts in `outputs/`:

- `outputs/surface_report.md`
- `outputs/model_comparison_report.md`
- `outputs/hedging_backtest_report.md`
- `outputs/rolling_recalibration_report.md`

### 1) Static Arbitrage Diagnostics (sample chain)

- Monotonicity violations: `0`
- Convexity violations: `0`
- Calendar total-variance violations: `0`

Interpretation: the generated sample surface passes core static no-arbitrage checks.

### 2) Volatility Model Comparison (SVI vs SABR, CV folds = 3)

- In-sample RMSE(IV): `SVI = 0.00215406`, `SABR = 0.00406937`
- Out-of-sample CV RMSE(IV): `SVI = 0.02177206`, `SABR = 0.00788022`
- In-sample winner: `SVI`
- Out-of-sample winner: `SABR`

Interpretation: SVI fits training data tighter; SABR generalizes better in cross-validation on this sample.

### 3) Dynamic Hedging Backtest (regime-switching scenarios, 400 paths)

- `delta` hedging std reduction vs unhedged: `90.39%`
- `delta-vega` hedging std reduction vs unhedged: `86.61%`
- Tail risk (ES 95%):
  - unhedged: `-27.406009`
  - delta: `-3.861296`
  - delta-vega: `-2.577188`

Interpretation: both hedging policies materially reduce risk; delta-vega improves extreme-loss tail behavior in this setup.

### 4) Rolling Recalibration Out-of-Sample Study (25 windows)

- Average calibration RMSE(w): `0.00057172`
- Std reduction vs unhedged:
  - delta: `41.67%`
  - delta-vega: `88.44%`
- Average transaction cost:
  - delta: `0.103290`
  - delta-vega: `0.084790`

Interpretation: with re-calibration and contract roll windows, delta-vega remains robust and risk-efficient.

## What Was Implemented

- Option-chain ingestion and validation from CSV.
- Black-Scholes pricing, implied-vol inversion, and Greeks (`delta`, `vega`).
- Static no-arbitrage diagnostics:
  - call monotonicity by strike
  - call convexity (butterfly condition)
  - calendar monotonicity of total variance
- `SVI` calibration and maturity interpolation in total-variance space.
- `SABR` (Hagan approximation) calibration.
- Cross-validated model comparison (`SVI` vs `SABR`).
- Dynamic hedging simulation:
  - `unhedged`
  - `delta`
  - `delta-vega`
  - transaction cost accounting
- Rolling re-calibration and out-of-sample window study.
- Optional live option-chain ingestion from Yahoo Finance.
- Reproducible artifact generation to JSON + Markdown reports.

## One-Command Reproduction

Run everything and regenerate all outputs:

```bash
PYTHONPATH=src python3 scripts/run_full_study.py
```

This writes:

- `outputs/surface_results.json`
- `outputs/surface_report.md`
- `outputs/model_comparison_results.json`
- `outputs/model_comparison_report.md`
- `outputs/hedging_backtest_results.json`
- `outputs/hedging_backtest_report.md`
- `outputs/rolling_recalibration_results.json`
- `outputs/rolling_recalibration_report.md`

Note: `run_full_study.py` uses deterministic synthetic datasets for reproducibility.

## Component Runs

### Generate sample option chain

```bash
PYTHONPATH=src python3 scripts/generate_sample_data.py
```

### Surface calibration report

```bash
PYTHONPATH=src python3 scripts/run_research_pipeline.py \
  --input data/sample_option_chain.csv \
  --output-dir outputs
```

### SVI vs SABR comparison report

```bash
PYTHONPATH=src python3 scripts/run_model_comparison.py \
  --input data/sample_option_chain.csv \
  --output-dir outputs \
  --seed 17 \
  --sabr-beta 1.0 \
  --folds 3
```

### Dynamic hedging report

```bash
PYTHONPATH=src python3 scripts/run_hedging_backtest.py \
  --input data/sample_option_chain.csv \
  --output-dir outputs \
  --paths 400 \
  --steps 126
```

### Historical rolling study

```bash
PYTHONPATH=src python3 scripts/generate_historical_data.py \
  --output data/historical_option_chain.csv \
  --valuation-days 180

PYTHONPATH=src python3 scripts/run_rolling_recalibration.py \
  --input data/historical_option_chain.csv \
  --output-dir outputs \
  --max-windows 25 \
  --max-rebalance-dates 100
```

### Live market snapshot workflow (optional)

```bash
PYTHONPATH=src python3 scripts/fetch_yahoo_option_chain.py \
  --ticker SPY \
  --output data/live_option_chain.csv \
  --max-expiries 4

PYTHONPATH=src python3 scripts/run_research_pipeline.py \
  --input data/live_option_chain.csv \
  --output-dir outputs

PYTHONPATH=src python3 scripts/run_model_comparison.py \
  --input data/live_option_chain.csv \
  --output-dir outputs
```

## Dataset Format

Required CSV columns:

- `valuation_date` (YYYY-MM-DD)
- `expiry` (YYYY-MM-DD)
- `maturity` (years, float)
- `spot` (float)
- `rate` (continuous risk-free rate, float)
- `dividend` (continuous dividend yield, float)
- `strike` (float)
- `call_mid` (float)

## Limitations

- Current historical data source is synthetic (designed for controlled evaluation).
- Real exchange microstructure effects (bid-ask, liquidity filters, early close days) are not yet included.
- SABR calibration currently uses fixed `beta`; dynamic `beta` calibration is not implemented.

## Next Practical Upgrade

- Replace synthetic history with real option snapshots and rerun the same pipeline unchanged.
