import math
from dataclasses import dataclass

from .optimisation import (
    halton_points,
    levenberg_marquardt,
    map_unit_to_bounds,
)


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float


def svi_total_variance(k: float, params: SVIParams) -> float:
    x = k - params.m
    return params.a + params.b * (
        params.rho * x + math.sqrt(x * x + params.sigma * params.sigma)
    )


def _is_valid(params: SVIParams) -> bool:
    if params.b <= 0.0:
        return False
    if params.sigma <= 0.0:
        return False
    if abs(params.rho) >= 1.0:
        return False
    min_total_var = params.a + params.b * params.sigma * math.sqrt(1.0 - params.rho * params.rho)
    return min_total_var > 0.0


def _sse(log_moneyness: list[float], total_vars: list[float], params: SVIParams) -> float:
    err = 0.0
    for k, w_obs in zip(log_moneyness, total_vars):
        w_hat = svi_total_variance(k, params)
        diff = w_hat - w_obs
        err += diff * diff
    return err


def _from_vector(x: list[float]) -> SVIParams:
    return SVIParams(a=x[0], b=x[1], rho=x[2], m=x[3], sigma=x[4])


def fit_raw_svi(
    log_moneyness: list[float],
    total_vars: list[float],
    seed: int = 7,
    random_trials: int = 4000,
    local_iters: int = 250,
) -> SVIParams:
    if len(log_moneyness) != len(total_vars):
        raise ValueError("log_moneyness and total_vars must have same length")
    if len(log_moneyness) < 5:
        raise ValueError("need at least 5 points to fit SVI robustly")

    k_min = min(log_moneyness)
    k_max = max(log_moneyness)
    w_min = min(total_vars)
    w_max = max(total_vars)
    w_mid = total_vars[len(total_vars) // 2]

    bounds = [
        (-0.25 * max(0.01, w_max), max(0.01, 1.5 * w_max)),  # a
        (1e-6, 5.0),  # b
        (-0.999, 0.999),  # rho
        (k_min - 0.6, k_max + 0.6),  # m
        (1e-4, 3.0),  # sigma
    ]

    def residual_fn(x: list[float]) -> list[float]:
        p = _from_vector(x)
        residuals = [svi_total_variance(k, p) - w for k, w in zip(log_moneyness, total_vars)]
        min_total_var = p.a + p.b * p.sigma * math.sqrt(max(1e-12, 1.0 - p.rho * p.rho))
        penalty = 0.0
        if p.b <= 0.0:
            penalty += abs(p.b) + 1.0
        if p.sigma <= 0.0:
            penalty += abs(p.sigma) + 1.0
        if abs(p.rho) >= 1.0:
            penalty += abs(p.rho) - 0.999 + 1.0
        if min_total_var <= 0.0:
            penalty += abs(min_total_var) + 1.0
        residuals.append(25.0 * penalty)
        return residuals

    deterministic_starts = [
        [0.5 * w_min, 0.15, -0.25, 0.0, 0.2],
        [0.2 * w_mid, 0.35, -0.5, 0.0, 0.35],
        [0.8 * w_min, 0.08, -0.1, 0.0, 0.12],
        [0.0, 0.5 * max(0.03, w_mid), -0.2, 0.05, 0.3],
    ]
    deterministic_starts = [
        [max(bounds[i][0], min(bounds[i][1], x[i])) for i in range(5)]
        for x in deterministic_starts
    ]

    # Preserve compatibility with historical parameters while moving to proper optimisation.
    n_halton = max(6, min(20, random_trials // 600))
    max_iter = max(60, min(220, local_iters))
    halton = halton_points(n_halton, dimension=5)
    if halton:
        shift = seed % len(halton)
        halton = halton[shift:] + halton[:shift]
    starts = deterministic_starts + [map_unit_to_bounds(p, bounds) for p in halton]

    best_result = None
    best_cost = float("inf")
    for idx, x0 in enumerate(starts):
        result = levenberg_marquardt(
            residual_fn=residual_fn,
            x0=x0,
            bounds=bounds,
            max_iter=max_iter,
            damping0=1e-2 if idx < len(deterministic_starts) else 5e-2,
        )
        if result.cost < best_cost:
            best_result = result
            best_cost = result.cost

    if best_result is None:
        raise RuntimeError("SVI calibration failed: optimisation did not produce a candidate")

    final = _from_vector(best_result.x)
    if not _is_valid(final):
        raise RuntimeError("SVI calibration failed: best candidate violated constraints")
    return final


def fit_svi_from_observations(
    forward: float,
    strikes: list[float],
    maturity: float,
    implied_vols: list[float],
    seed: int = 7,
) -> tuple[SVIParams, float]:
    if forward <= 0.0:
        raise ValueError("forward must be positive")
    if maturity <= 0.0:
        raise ValueError("maturity must be positive")
    if len(strikes) != len(implied_vols):
        raise ValueError("strikes and implied_vols length mismatch")

    log_moneyness = [math.log(k / forward) for k in strikes]
    total_vars = [iv * iv * maturity for iv in implied_vols]
    params = fit_raw_svi(log_moneyness=log_moneyness, total_vars=total_vars, seed=seed)
    rmse = math.sqrt(_sse(log_moneyness, total_vars, params) / len(log_moneyness))
    return params, rmse
