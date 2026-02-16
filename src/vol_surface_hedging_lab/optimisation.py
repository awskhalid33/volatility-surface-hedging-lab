import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OptimisationResult:
    x: list[float]
    cost: float
    converged: bool
    iterations: int


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp_to_bounds(x: list[float], bounds: list[tuple[float, float]]) -> list[float]:
    return [_clamp(v, b[0], b[1]) for v, b in zip(x, bounds)]


def _l2_norm(x: list[float]) -> float:
    return math.sqrt(sum(v * v for v in x))


def _objective_cost(residuals: list[float]) -> float:
    return 0.5 * sum(r * r for r in residuals)


def _transpose_multiply_jacobian(jacobian: list[list[float]]) -> list[list[float]]:
    m = len(jacobian)
    n = len(jacobian[0]) if m > 0 else 0
    out = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(m):
                s += jacobian[k][i] * jacobian[k][j]
            out[i][j] = s
    return out


def _transpose_multiply_vector(jacobian: list[list[float]], vec: list[float]) -> list[float]:
    m = len(jacobian)
    n = len(jacobian[0]) if m > 0 else 0
    out = [0.0 for _ in range(n)]
    for i in range(n):
        s = 0.0
        for k in range(m):
            s += jacobian[k][i] * vec[k]
        out[i] = s
    return out


def _solve_linear_system(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(a)
    if n == 0:
        return []
    aug = [row[:] + [rhs] for row, rhs in zip(a, b)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-14:
            raise ValueError("Singular linear system in LM step")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        pivot_val = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_val

        for row in range(col + 1, n):
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    x = [0.0 for _ in range(n)]
    for row in range(n - 1, -1, -1):
        s = aug[row][n]
        for col in range(row + 1, n):
            s -= aug[row][col] * x[col]
        x[row] = s
    return x


def finite_difference_jacobian(
    residual_fn,
    x: list[float],
    bounds: list[tuple[float, float]],
    residuals_at_x: list[float] | None = None,
    epsilon: float = 1e-6,
) -> tuple[list[list[float]], list[float]]:
    base = residuals_at_x if residuals_at_x is not None else residual_fn(x)
    m = len(base)
    n = len(x)
    jacobian = [[0.0 for _ in range(n)] for _ in range(m)]

    for j in range(n):
        h = epsilon * (1.0 + abs(x[j]))
        x_up = x[:]
        x_dn = x[:]
        x_up[j] = _clamp(x[j] + h, bounds[j][0], bounds[j][1])
        x_dn[j] = _clamp(x[j] - h, bounds[j][0], bounds[j][1])

        if abs(x_up[j] - x_dn[j]) < 1e-14:
            continue

        r_up = residual_fn(x_up)
        r_dn = residual_fn(x_dn)
        denom = x_up[j] - x_dn[j]
        for i in range(m):
            jacobian[i][j] = (r_up[i] - r_dn[i]) / denom

    return jacobian, base


def levenberg_marquardt(
    residual_fn,
    x0: list[float],
    bounds: list[tuple[float, float]],
    max_iter: int = 120,
    damping0: float = 1e-2,
    tol_step: float = 1e-9,
    tol_cost: float = 1e-12,
) -> OptimisationResult:
    if len(x0) != len(bounds):
        raise ValueError("x0 and bounds must have same length")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    x = clamp_to_bounds(x0, bounds)
    residuals = residual_fn(x)
    cost = _objective_cost(residuals)
    damping = damping0
    converged = False

    for iteration in range(1, max_iter + 1):
        jacobian, residuals = finite_difference_jacobian(
            residual_fn=residual_fn,
            x=x,
            bounds=bounds,
            residuals_at_x=residuals,
        )
        jtj = _transpose_multiply_jacobian(jacobian)
        grad = _transpose_multiply_vector(jacobian, residuals)
        n = len(x)
        for i in range(n):
            jtj[i][i] += damping

        try:
            step = _solve_linear_system(jtj, [-g for g in grad])
        except ValueError:
            damping = min(1e12, damping * 10.0)
            continue

        if _l2_norm(step) < tol_step:
            converged = True
            break

        candidate = clamp_to_bounds([x_i + dx_i for x_i, dx_i in zip(x, step)], bounds)
        cand_residuals = residual_fn(candidate)
        cand_cost = _objective_cost(cand_residuals)

        if cand_cost + tol_cost < cost:
            improvement = cost - cand_cost
            x = candidate
            residuals = cand_residuals
            cost = cand_cost
            damping = max(1e-12, damping * 0.5)
            if improvement < tol_cost:
                converged = True
                break
        else:
            damping = min(1e12, damping * 2.0)

    return OptimisationResult(
        x=x,
        cost=cost,
        converged=converged,
        iterations=iteration if "iteration" in locals() else 0,
    )


def _halton_value(index: int, base: int) -> float:
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


def halton_points(
    n_points: int,
    dimension: int,
    bases: tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17, 19),
) -> list[list[float]]:
    if dimension > len(bases):
        raise ValueError("dimension exceeds available Halton bases")
    points = []
    for idx in range(1, n_points + 1):
        point = [_halton_value(idx, bases[d]) for d in range(dimension)]
        points.append(point)
    return points


def map_unit_to_bounds(unit_point: list[float], bounds: list[tuple[float, float]]) -> list[float]:
    if len(unit_point) != len(bounds):
        raise ValueError("unit_point and bounds length mismatch")
    out = []
    for u, (low, high) in zip(unit_point, bounds):
        out.append(low + u * (high - low))
    return out
