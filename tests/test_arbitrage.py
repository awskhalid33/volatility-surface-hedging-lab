from lse_fm_vol_project.arbitrage import (
    check_calendar_total_variance,
    check_call_convexity,
    check_call_monotonicity,
)


def test_no_static_arbitrage_for_smooth_curve():
    strikes = [90.0, 100.0, 110.0]
    calls = [15.0, 9.0, 5.0]
    assert check_call_monotonicity(strikes, calls) == []
    assert check_call_convexity(strikes, calls) == []


def test_calendar_total_variance_violation_detected():
    maturity_to_iv = {
        0.5: {100.0: 0.30},
        1.0: {100.0: 0.20},
    }
    issues = check_calendar_total_variance(maturity_to_iv)
    assert len(issues) == 1
