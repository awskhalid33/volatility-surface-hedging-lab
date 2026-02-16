import math

from vol_surface_hedging_lab.optimisation import (
    halton_points,
    levenberg_marquardt,
    map_unit_to_bounds,
)


def test_halton_points_stay_inside_unit_cube():
    points = halton_points(n_points=8, dimension=3)
    assert len(points) == 8
    assert all(0.0 < value < 1.0 for point in points for value in point)


def test_map_unit_to_bounds_respects_intervals():
    mapped = map_unit_to_bounds(
        unit_point=[0.1, 0.5, 0.9],
        bounds=[(-2.0, 2.0), (10.0, 12.0), (100.0, 200.0)],
    )
    assert abs(mapped[0] + 1.6) < 1e-12
    assert abs(mapped[1] - 11.0) < 1e-12
    assert abs(mapped[2] - 190.0) < 1e-12


def test_levenberg_marquardt_converges_on_simple_problem():
    def residual_fn(x: list[float]) -> list[float]:
        return [x[0] - 1.5, 2.0 * (x[1] + 0.75)]

    result = levenberg_marquardt(
        residual_fn=residual_fn,
        x0=[4.0, -4.0],
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
        max_iter=80,
    )
    assert result.converged
    assert abs(result.x[0] - 1.5) < 1e-6
    assert abs(result.x[1] + 0.75) < 1e-6
    assert result.cost < 1e-10


def test_levenberg_marquardt_handles_near_singular_geometry():
    def residual_fn(x: list[float]) -> list[float]:
        return [
            x[0] + x[1] - 2.0,
            x[0] + x[1] - 2.000001,
            1e-3 * (x[0] - x[1]),
        ]

    result = levenberg_marquardt(
        residual_fn=residual_fn,
        x0=[3.0, -1.0],
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
        max_iter=120,
        damping0=1e-1,
    )
    assert result.converged
    assert abs((result.x[0] + result.x[1]) - 2.0000005) < 1e-4
    assert abs(result.x[0] - result.x[1]) < 1e-3
    assert math.isfinite(result.cost)
