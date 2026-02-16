import math
from dataclasses import dataclass

from .optimisation import (
    halton_points,
    levenberg_marquardt,
    map_unit_to_bounds,
)


@dataclass(frozen=True)
class SABRParams:
    alpha: float
    beta: float
    rho: float
    nu: float


def _is_valid(params: SABRParams) -> bool:
    if params.alpha <= 0.0:
        return False
    if params.nu <= 0.0:
        return False
    if abs(params.rho) >= 1.0:
        return False
    if not (0.0 <= params.beta <= 1.0):
        return False
    return True


def _x_of_z(z: float, rho: float) -> float:
    numerator = math.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho
    denominator = 1.0 - rho
    if denominator <= 0.0:
        denominator = 1e-12
    ratio = numerator / denominator
    if ratio <= 1e-12:
        ratio = 1e-12
    return math.log(ratio)


def sabr_implied_vol(
    forward: float,
    strike: float,
    maturity: float,
    params: SABRParams,
) -> float:
    if forward <= 0.0:
        raise ValueError("forward must be positive")
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if maturity <= 0.0:
        raise ValueError("maturity must be positive")
    if not _is_valid(params):
        raise ValueError("invalid SABR parameters")

    alpha = params.alpha
    beta = params.beta
    rho = params.rho
    nu = params.nu
    one_minus_beta = 1.0 - beta

    # ATM closed form
    if abs(forward - strike) < 1e-12:
        f_beta = forward ** one_minus_beta
        term1 = (one_minus_beta * one_minus_beta / 24.0) * (alpha * alpha) / (f_beta * f_beta)
        term2 = 0.25 * rho * beta * nu * alpha / f_beta
        term3 = (2.0 - 3.0 * rho * rho) * (nu * nu) / 24.0
        atm = alpha / f_beta * (1.0 + (term1 + term2 + term3) * maturity)
        return max(1e-8, atm)

    fk = forward * strike
    fk_beta = fk ** (0.5 * one_minus_beta)
    log_fk = math.log(forward / strike)
    log_fk_sq = log_fk * log_fk

    z = (nu / alpha) * fk_beta * log_fk
    xz = _x_of_z(z, rho)
    ratio = 1.0 if abs(z) < 1e-8 else z / xz

    denom = fk_beta * (
        1.0
        + (one_minus_beta * one_minus_beta / 24.0) * log_fk_sq
        + (one_minus_beta**4 / 1920.0) * (log_fk_sq * log_fk_sq)
    )
    if denom <= 1e-12:
        denom = 1e-12

    term1 = (one_minus_beta * one_minus_beta / 24.0) * (alpha * alpha) / (fk ** one_minus_beta)
    term2 = 0.25 * rho * beta * nu * alpha / fk_beta
    term3 = (2.0 - 3.0 * rho * rho) * (nu * nu) / 24.0
    time_correction = 1.0 + (term1 + term2 + term3) * maturity

    sigma = (alpha / denom) * ratio * time_correction
    return max(1e-8, sigma)


def _sse(
    forward: float,
    strikes: list[float],
    maturity: float,
    implied_vols: list[float],
    params: SABRParams,
) -> float:
    err = 0.0
    for k, iv in zip(strikes, implied_vols):
        fitted = sabr_implied_vol(forward, k, maturity, params)
        d = fitted - iv
        err += d * d
    return err


def _from_vector(x: list[float], beta: float) -> SABRParams:
    return SABRParams(alpha=x[0], beta=beta, rho=x[1], nu=x[2])


def fit_sabr_from_observations(
    forward: float,
    strikes: list[float],
    maturity: float,
    implied_vols: list[float],
    beta: float = 1.0,
    seed: int = 19,
    random_trials: int = 3500,
    local_iters: int = 250,
) -> tuple[SABRParams, float]:
    if forward <= 0.0:
        raise ValueError("forward must be positive")
    if maturity <= 0.0:
        raise ValueError("maturity must be positive")
    if len(strikes) != len(implied_vols):
        raise ValueError("strikes and implied_vols length mismatch")
    if len(strikes) < 4:
        raise ValueError("need at least 4 points for SABR calibration")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0, 1]")

    iv_min = min(implied_vols)
    iv_max = max(implied_vols)
    iv_mean = sum(implied_vols) / len(implied_vols)

    bounds = [
        (max(1e-4, 0.2 * iv_min), max(5.0, 3.0 * iv_max)),  # alpha
        (-0.999, 0.999),  # rho
        (1e-4, 5.0),  # nu
    ]

    def residual_fn(x: list[float]) -> list[float]:
        p = _from_vector(x, beta=beta)
        residuals = [sabr_implied_vol(forward, k, maturity, p) - iv for k, iv in zip(strikes, implied_vols)]
        # Keep a fixed-length residual vector and softly penalise near-boundary instability.
        boundary_penalty = 0.0
        boundary_penalty += max(0.0, abs(p.rho) - 0.98)
        boundary_penalty += max(0.0, 1e-3 - p.alpha)
        boundary_penalty += max(0.0, 1e-3 - p.nu)
        residuals.append(10.0 * boundary_penalty)
        return residuals

    deterministic_starts = [
        [max(1e-4, iv_mean), -0.1, 0.5],
        [max(1e-4, 0.7 * iv_mean), -0.4, 0.8],
        [max(1e-4, 1.2 * iv_mean), 0.2, 0.4],
    ]
    deterministic_starts = [
        [max(bounds[i][0], min(bounds[i][1], x[i])) for i in range(3)]
        for x in deterministic_starts
    ]

    n_halton = max(5, min(16, random_trials // 700))
    max_iter = max(60, min(200, local_iters))
    halton = halton_points(n_halton, dimension=3)
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
            damping0=1e-2 if idx < len(deterministic_starts) else 4e-2,
        )
        if result.cost < best_cost:
            best_result = result
            best_cost = result.cost

    if best_result is None:
        raise RuntimeError("SABR calibration failed: optimisation did not produce a candidate")

    best = _from_vector(best_result.x, beta=beta)
    if not _is_valid(best):
        raise RuntimeError("SABR calibration failed: best candidate violated bounds")

    rmse = math.sqrt(_sse(forward, strikes, maturity, implied_vols, best) / len(strikes))
    return best, rmse
