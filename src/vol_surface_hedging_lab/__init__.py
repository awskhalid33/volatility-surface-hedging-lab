"""Research toolkit for volatility surface modeling and hedging."""

from .black_scholes import (
    bs_call_delta,
    bs_call_price,
    bs_call_vega,
    implied_volatility_call,
)
from .model_comparison import render_model_comparison_markdown, run_model_comparison
from .rolling import (
    RollingRecalibrationConfig,
    render_rolling_markdown,
    run_rolling_recalibration_experiment,
)
from .sabr import SABRParams, fit_sabr_from_observations, sabr_implied_vol
from .svi import SVIParams, fit_svi_from_observations, svi_total_variance
from .surface import SVISurfaceModel
from .visualisation import generate_visual_artifacts
from .yahoo_data import fetch_yahoo_option_chain_rows

__all__ = [
    "SABRParams",
    "SVIParams",
    "SVISurfaceModel",
    "RollingRecalibrationConfig",
    "bs_call_delta",
    "bs_call_price",
    "bs_call_vega",
    "fit_sabr_from_observations",
    "fit_svi_from_observations",
    "implied_volatility_call",
    "render_model_comparison_markdown",
    "render_rolling_markdown",
    "run_model_comparison",
    "run_rolling_recalibration_experiment",
    "sabr_implied_vol",
    "svi_total_variance",
    "fetch_yahoo_option_chain_rows",
    "generate_visual_artifacts",
]
