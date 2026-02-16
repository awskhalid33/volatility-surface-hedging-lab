from typing import Iterable


def check_call_monotonicity(
    strikes: Iterable[float],
    call_prices: Iterable[float],
    tolerance: float = 1e-10,
) -> list[str]:
    ordered = sorted(zip(strikes, call_prices), key=lambda x: x[0])
    issues: list[str] = []
    for idx in range(1, len(ordered)):
        prev_k, prev_c = ordered[idx - 1]
        curr_k, curr_c = ordered[idx]
        if curr_c > prev_c + tolerance:
            issues.append(
                "Monotonicity violation: call increased with strike "
                f"(K {prev_k:.4f}->{curr_k:.4f}, C {prev_c:.6f}->{curr_c:.6f})"
            )
    return issues


def check_call_convexity(
    strikes: Iterable[float],
    call_prices: Iterable[float],
    tolerance: float = 1e-10,
) -> list[str]:
    ordered = sorted(zip(strikes, call_prices), key=lambda x: x[0])
    issues: list[str] = []
    if len(ordered) < 3:
        return issues

    for idx in range(1, len(ordered) - 1):
        k_prev, c_prev = ordered[idx - 1]
        k_curr, c_curr = ordered[idx]
        k_next, c_next = ordered[idx + 1]

        left = (c_curr - c_prev) / (k_curr - k_prev)
        right = (c_next - c_curr) / (k_next - k_curr)
        if right < left - tolerance:
            issues.append(
                "Convexity violation: slope became more negative "
                f"(K={k_curr:.4f}, left={left:.8f}, right={right:.8f})"
            )
    return issues


def check_calendar_total_variance(
    maturity_to_strike_iv: dict[float, dict[float, float]],
    tolerance: float = 1e-10,
) -> list[str]:
    issues: list[str] = []
    maturities = sorted(maturity_to_strike_iv.keys())
    if len(maturities) < 2:
        return issues

    strike_sets = [set(maturity_to_strike_iv[t].keys()) for t in maturities]
    common_strikes = sorted(set.intersection(*strike_sets)) if strike_sets else []
    if not common_strikes:
        issues.append("Calendar check skipped: no common strikes across maturities.")
        return issues

    for strike in common_strikes:
        prev_tw = None
        prev_t = None
        for maturity in maturities:
            iv = maturity_to_strike_iv[maturity][strike]
            total_var = iv * iv * maturity
            if prev_tw is not None and total_var < prev_tw - tolerance:
                issues.append(
                    "Calendar total variance violation at "
                    f"K={strike:.4f}: w(T={maturity:.6f})={total_var:.8f} "
                    f"< w(T={prev_t:.6f})={prev_tw:.8f}"
                )
            prev_tw = total_var
            prev_t = maturity

    return issues
