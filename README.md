# Volatility Surface and Dynamic Hedging Lab

Personal quantitative research project for option-surface modelling, model comparison, dynamic hedging, and rolling out-of-sample evaluation.

## What this project does

1. Converts option prices to implied volatility.
2. Runs static no-arbitrage diagnostics.
3. Calibrates `SVI` and `SABR` smiles using bounded multi-start optimisation.
4. Compares models in-sample and out-of-sample with cross-validation.
5. Backtests `unhedged`, `delta`, and `delta-vega` hedging with transaction costs.
6. Runs a rolling recalibration experiment.
7. Produces JSON, Markdown, and SVG artefacts.

## Core upgrades in this version

- **Implied-vol solver upgrade**: `implied_volatility_call` now uses hybrid Newton-Raphson (with vega) plus bracketed bisection fallback.
- **Optimisation upgrade**: bounded Levenberg-Marquardt now includes Marquardt diagonal scaling and trust-region-style step control.
- **Real-data pathway upgrade**:
  - Yahoo workflow now includes retry/backoff and endpoint failover.
  - `run_live_snapshot_study.py` accepts `--input-csv` so real-data case studies can run from local snapshots.
  - Added `build_public_spy_case_data.py` to build a real SPY snapshot dataset from public files.
- **Engineering quality upgrade**:
  - CI workflow added: `.github/workflows/ci.yml`
  - lint config added: `ruff.toml`
  - type-check config added: `mypy.ini`
  - stronger tests for optimisation, visualisation, IV edge cases, data I/O failures, and Yahoo SSL fallback.

## Quick start

Install project + dev tooling:

```bash
python3 -m pip install -e ".[dev]"
```

Run synthetic full study:

```bash
PYTHONPATH=src python3 scripts/run_full_study.py
```

Run quality checks:

```bash
ruff check .
mypy
pytest -q
```

## Reproducible real-data case study

Build a public real SPY snapshot dataset (valuation date `2020-07-05`):

```bash
PYTHONPATH=src python3 scripts/build_public_spy_case_data.py \
  --output data/public_spy_snapshot_2020-07-05.csv
```

Run the live-study pipeline from that local CSV:

```bash
PYTHONPATH=src python3 scripts/run_live_snapshot_study.py \
  --ticker SPY \
  --input-csv data/public_spy_snapshot_2020-07-05.csv \
  --output-dir outputs \
  --paths 300 \
  --steps 120
```

Public source files used by the builder script:

- [SPY 2020-07-10 options snapshot](https://raw.githubusercontent.com/cantaro86/Financial-Models-Numerical-Methods/master/data/spy-options-exp-2020-07-10-weekly-show-all-stacked-07-05-2020.csv)
- [SPY 2021-01-15 options snapshot](https://raw.githubusercontent.com/cantaro86/Financial-Models-Numerical-Methods/master/data/spy-options-exp-2021-01-15-weekly-show-all-stacked-07-05-2020.csv)

## Key results

### A) Synthetic benchmark (`scripts/run_full_study.py`)

- Arbitrage diagnostics:
  - Monotonicity violations: `0`
  - Convexity violations: `0`
  - Calendar total-variance violations: `0`
- Model comparison (`SVI` vs `SABR`, 3-fold CV):
  - In-sample RMSE (IV): `SVI=0.00025257`, `SABR=0.00406938`
  - Out-of-sample CV RMSE (IV): `SVI=0.00267081`, `SABR=0.00666528`
  - Winners: in-sample `SVI`, out-of-sample `SVI`
- Dynamic hedging (400 paths):
  - `delta` standard-deviation reduction vs unhedged: `89.65%`
  - `delta-vega` standard-deviation reduction vs unhedged: `86.38%`
  - ES 95%:
    - unhedged: `-27.406009`
    - delta: `-4.111097`
    - delta-vega: `-2.749096`
- Rolling recalibration (25 windows):
  - Average calibration RMSE(w): `0.00014027`
  - Standard-deviation reduction vs unhedged:
    - delta: `41.65%`
    - delta-vega: `88.95%`

### B) Real-data case (`data/public_spy_snapshot_2020-07-05.csv`)

- Dataset summary:
  - Quotes: `264`
  - Maturities: `2`
- Arbitrage diagnostics:
  - Monotonicity violations: `1`
  - Convexity violations: `54`
  - Calendar total-variance violations: `0`
- Model comparison (`SVI` vs `SABR`, 3-fold CV):
  - In-sample RMSE (IV): `SVI=0.00065403`, `SABR=0.03483466`
  - Out-of-sample CV RMSE (IV): `SVI=0.01275583`, `SABR=0.03482615`
  - Winners: in-sample `SVI`, out-of-sample `SVI`
- Dynamic hedging (300 paths):
  - `delta` standard-deviation reduction vs unhedged: `90.01%`
  - `delta-vega` standard-deviation reduction vs unhedged: `85.44%`
  - ES 95%:
    - unhedged: `-103.527269`
    - delta: `-12.439165`
    - delta-vega: `-10.127254`

This real-data run intentionally demonstrates that market data is messy and can contain static-shape violations, while the pipeline still runs end-to-end.

## Output artefacts

Synthetic outputs:

- `outputs/surface_report.md`
- `outputs/model_comparison_report.md`
- `outputs/hedging_backtest_report.md`
- `outputs/rolling_recalibration_report.md`
- `outputs/fig_smile_fit.svg`
- `outputs/fig_model_rmse.svg`
- `outputs/fig_terminal_pnl_hist_unhedged.svg`
- `outputs/fig_terminal_pnl_hist_delta.svg`
- `outputs/fig_terminal_pnl_hist_delta-vega.svg`
- `outputs/fig_rolling_rmse_trend.svg`

Real-case outputs:

- `outputs/live_surface_report.md`
- `outputs/live_model_comparison_report.md`
- `outputs/live_hedging_backtest_report.md`
- `outputs/live_fig_smile_fit.svg`
- `outputs/live_fig_model_rmse.svg`
- `outputs/live_fig_terminal_pnl_hist_unhedged.svg`
- `outputs/live_fig_terminal_pnl_hist_delta.svg`
- `outputs/live_fig_terminal_pnl_hist_delta-vega.svg`

## Data schema

Required CSV columns:

- `valuation_date` (YYYY-MM-DD)
- `expiry` (YYYY-MM-DD)
- `maturity` (years)
- `spot`
- `rate` (continuous risk-free rate)
- `dividend` (continuous dividend yield)
- `strike`
- `call_mid`

## Limitations

- Public real-data case uses static snapshots rather than a full live archive.
- `SABR` calibration currently uses fixed `beta`.
- Market microstructure handling is simplified (for example, no full liquidity filter stack).
- Yahoo endpoints can rate-limit (`HTTP 429`); use `--input-csv` mode for reproducible runs.
