import math
import random
from dataclasses import dataclass


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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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

    bounds = {
        "a": (-0.25 * max(0.01, w_max), max(0.01, 1.5 * w_max)),
        "b": (1e-6, 5.0),
        "rho": (-0.999, 0.999),
        "m": (k_min - 0.6, k_max + 0.6),
        "sigma": (1e-4, 3.0),
    }

    rng = random.Random(seed)
    best = None
    best_err = float("inf")

    for _ in range(random_trials):
        candidate = SVIParams(
            a=rng.uniform(*bounds["a"]),
            b=rng.uniform(*bounds["b"]),
            rho=rng.uniform(*bounds["rho"]),
            m=rng.uniform(*bounds["m"]),
            sigma=rng.uniform(*bounds["sigma"]),
        )
        if not _is_valid(candidate):
            continue
        err = _sse(log_moneyness, total_vars, candidate)
        if err < best_err:
            best = candidate
            best_err = err

    if best is None:
        raise RuntimeError("SVI calibration failed: no valid random initializations")

    step = {
        "a": max(1e-4, 0.05 * max(w_max, 0.01)),
        "b": 0.08,
        "rho": 0.05,
        "m": 0.03,
        "sigma": 0.03,
    }

    for _ in range(local_iters):
        improved = False
        for key in ("a", "b", "rho", "m", "sigma"):
            for direction in (-1.0, 1.0):
                kwargs = {
                    "a": best.a,
                    "b": best.b,
                    "rho": best.rho,
                    "m": best.m,
                    "sigma": best.sigma,
                }
                proposed = kwargs[key] + direction * step[key]
                low, high = bounds[key]
                kwargs[key] = _clamp(proposed, low, high)
                candidate = SVIParams(**kwargs)
                if not _is_valid(candidate):
                    continue
                err = _sse(log_moneyness, total_vars, candidate)
                if err + 1e-16 < best_err:
                    best = candidate
                    best_err = err
                    improved = True
        if not improved:
            for key in step:
                step[key] *= 0.65
            if max(step.values()) < 1e-5:
                break

    return best


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
