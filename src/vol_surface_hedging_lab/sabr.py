import math
import random
from dataclasses import dataclass


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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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

    bounds = {
        "alpha": (max(1e-4, 0.2 * iv_min), max(5.0, 3.0 * iv_max)),
        "rho": (-0.999, 0.999),
        "nu": (1e-4, 5.0),
    }

    rng = random.Random(seed)
    best = None
    best_err = float("inf")

    for _ in range(random_trials):
        candidate = SABRParams(
            alpha=rng.uniform(*bounds["alpha"]),
            beta=beta,
            rho=rng.uniform(*bounds["rho"]),
            nu=rng.uniform(*bounds["nu"]),
        )
        if not _is_valid(candidate):
            continue
        err = _sse(forward, strikes, maturity, implied_vols, candidate)
        if err < best_err:
            best = candidate
            best_err = err

    if best is None:
        # deterministic fallback around average vol level
        best = SABRParams(alpha=max(1e-4, iv_mean), beta=beta, rho=0.0, nu=0.5)
        best_err = _sse(forward, strikes, maturity, implied_vols, best)

    step = {
        "alpha": max(1e-4, 0.08 * iv_mean),
        "rho": 0.06,
        "nu": 0.08,
    }

    for _ in range(local_iters):
        improved = False
        for key in ("alpha", "rho", "nu"):
            for direction in (-1.0, 1.0):
                kwargs = {
                    "alpha": best.alpha,
                    "beta": beta,
                    "rho": best.rho,
                    "nu": best.nu,
                }
                proposed = kwargs[key] + direction * step[key]
                low, high = bounds[key]
                kwargs[key] = _clamp(proposed, low, high)
                candidate = SABRParams(**kwargs)
                if not _is_valid(candidate):
                    continue
                err = _sse(forward, strikes, maturity, implied_vols, candidate)
                if err + 1e-16 < best_err:
                    best = candidate
                    best_err = err
                    improved = True
        if not improved:
            for key in step:
                step[key] *= 0.7
            if max(step.values()) < 1e-5:
                break

    rmse = math.sqrt(best_err / len(strikes))
    return best, rmse
