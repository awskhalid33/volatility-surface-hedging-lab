import math
from dataclasses import asdict, dataclass
from datetime import date

from .hedging_common import compute_desired_positions, summary_stats
from .pipeline import run_surface_pipeline
from .surface import SVISurfaceModel
from .types import OptionQuote


@dataclass(frozen=True)
class RollingRecalibrationConfig:
    target_maturity_days: int = 180
    hedge_maturity_days: int = 300
    target_moneyness: float = 1.0
    hedge_moneyness: float = 1.10
    transaction_cost_stock: float = 0.0005
    transaction_cost_option: float = 0.0015
    max_abs_stock_position: float = 8.0
    max_abs_hedge_option_position: float = 8.0
    max_rebalance_dates: int = 120
    max_windows: int = 40
    calibration_seed: int = 11


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def _group_quotes_by_valuation_date(quotes: list[OptionQuote]) -> dict[str, list[OptionQuote]]:
    grouped: dict[str, list[OptionQuote]] = {}
    for q in quotes:
        grouped.setdefault(q.valuation_date, []).append(q)
    return grouped


def _nearest(values: list[float], target: float) -> float:
    return min(values, key=lambda x: abs(x - target))


def _unique_expiry_days(snapshot: list[OptionQuote], valuation: date) -> list[tuple[int, str]]:
    items = []
    for expiry in sorted(set(q.expiry for q in snapshot)):
        d = (_parse_date(expiry) - valuation).days
        if d > 7:
            items.append((d, expiry))
    return items


def _choose_contracts(
    snapshot: list[OptionQuote], cfg: RollingRecalibrationConfig
) -> tuple[str, str, float, float]:
    valuation = _parse_date(snapshot[0].valuation_date)
    expiries = _unique_expiry_days(snapshot, valuation)
    if len(expiries) < 2:
        raise ValueError("Need at least two valid expiries to build rolling hedge setup")

    target_expiry = min(expiries, key=lambda x: abs(x[0] - cfg.target_maturity_days))[1]
    target_days = (_parse_date(target_expiry) - valuation).days
    candidates_hedge = [x for x in expiries if x[0] > target_days + 20]
    hedge_pool = candidates_hedge if candidates_hedge else expiries
    hedge_expiry = min(hedge_pool, key=lambda x: abs(x[0] - cfg.hedge_maturity_days))[1]
    if hedge_expiry == target_expiry:
        hedge_expiry = max(expiries, key=lambda x: x[0])[1]

    spot = snapshot[0].spot
    target_rows = [q for q in snapshot if q.expiry == target_expiry]
    hedge_rows = [q for q in snapshot if q.expiry == hedge_expiry]
    if not target_rows or not hedge_rows:
        raise ValueError("Could not find quotes for selected expiries")

    target_strikes = sorted(set(q.strike for q in target_rows))
    hedge_strikes = sorted(set(q.strike for q in hedge_rows))
    target_strike = _nearest(target_strikes, spot * cfg.target_moneyness)
    hedge_strike = _nearest(hedge_strikes, spot * cfg.hedge_moneyness)
    return target_expiry, hedge_expiry, target_strike, hedge_strike


def _interpolate_call_price(rows: list[OptionQuote], strike: float) -> float:
    ordered = sorted(rows, key=lambda q: q.strike)
    if not ordered:
        raise ValueError("Cannot interpolate call price with empty rows")
    if strike <= ordered[0].strike:
        return ordered[0].call_mid
    if strike >= ordered[-1].strike:
        return ordered[-1].call_mid

    for idx in range(1, len(ordered)):
        left = ordered[idx - 1]
        right = ordered[idx]
        if left.strike <= strike <= right.strike:
            if abs(right.strike - left.strike) < 1e-12:
                return left.call_mid
            alpha = (strike - left.strike) / (right.strike - left.strike)
            return (1.0 - alpha) * left.call_mid + alpha * right.call_mid
    return ordered[-1].call_mid


def _strategy_backtest(
    strategy: str,
    snapshots: list[tuple[date, str, list[OptionQuote], SVISurfaceModel]],
    cfg: RollingRecalibrationConfig,
    target_expiry: date,
    hedge_expiry: date,
    target_strike: float,
    hedge_strike: float,
) -> dict:
    if not snapshots:
        raise ValueError("snapshots cannot be empty")

    cash = 0.0
    stock_units = 0.0
    hedge_units = 0.0
    total_costs = 0.0
    values: list[float] = []
    rebalances = 0
    prev_date = snapshots[0][0]

    for idx, (valuation_dt, _, quotes, surface) in enumerate(snapshots):
        spot = quotes[0].spot
        rate = quotes[0].rate
        dividend = quotes[0].dividend
        dt_years = max((valuation_dt - prev_date).days / 365.0, 0.0) if idx > 0 else 0.0
        if dt_years > 0.0:
            cash *= math.exp(rate * dt_years)
        prev_date = valuation_dt

        tau_target = max((target_expiry - valuation_dt).days / 365.0, 0.0)
        tau_hedge = max((hedge_expiry - valuation_dt).days / 365.0, 0.0)

        target_rows = [q for q in quotes if q.expiry == target_expiry.isoformat()]
        hedge_rows = [q for q in quotes if q.expiry == hedge_expiry.isoformat()]
        target_price = (
            _interpolate_call_price(target_rows, target_strike)
            if tau_target > 0.0 and target_rows
            else max(spot - target_strike, 0.0)
        )
        hedge_price = (
            _interpolate_call_price(hedge_rows, hedge_strike)
            if tau_hedge > 0.0 and hedge_rows
            else max(spot - hedge_strike, 0.0)
        )

        if idx == 0:
            cash += target_price
            desired_stock, desired_hedge = compute_desired_positions(
                strategy=strategy,
                surface=surface,
                spot=spot,
                rate=rate,
                dividend=dividend,
                target_strike=target_strike,
                hedge_strike=hedge_strike,
                tau_target=tau_target,
                tau_hedge=tau_hedge,
                max_abs_stock_position=cfg.max_abs_stock_position,
                max_abs_hedge_option_position=cfg.max_abs_hedge_option_position,
            )
            stock_trade = desired_stock - stock_units
            hedge_trade = desired_hedge - hedge_units
            cash -= stock_trade * spot + hedge_trade * hedge_price
            trade_cost = (
                abs(stock_trade) * spot * cfg.transaction_cost_stock
                + abs(hedge_trade) * hedge_price * cfg.transaction_cost_option
            )
            cash -= trade_cost
            total_costs += trade_cost
            stock_units = desired_stock
            hedge_units = desired_hedge
            rebalances += 1

        value = cash + stock_units * spot + hedge_units * hedge_price - target_price
        values.append(value)

        is_last = idx == len(snapshots) - 1
        if is_last or tau_target <= 0.0:
            break

        desired_stock, desired_hedge = compute_desired_positions(
            strategy=strategy,
            surface=surface,
            spot=spot,
            rate=rate,
            dividend=dividend,
            target_strike=target_strike,
            hedge_strike=hedge_strike,
            tau_target=tau_target,
            tau_hedge=tau_hedge,
            max_abs_stock_position=cfg.max_abs_stock_position,
            max_abs_hedge_option_position=cfg.max_abs_hedge_option_position,
        )
        stock_trade = desired_stock - stock_units
        hedge_trade = desired_hedge - hedge_units
        cash -= stock_trade * spot + hedge_trade * hedge_price
        trade_cost = (
            abs(stock_trade) * spot * cfg.transaction_cost_stock
            + abs(hedge_trade) * hedge_price * cfg.transaction_cost_option
        )
        cash -= trade_cost
        total_costs += trade_cost
        stock_units = desired_stock
        hedge_units = desired_hedge
        rebalances += 1

    terminal_spot = snapshots[min(len(values) - 1, len(snapshots) - 1)][2][0].spot
    terminal_tau_hedge = max((hedge_expiry - snapshots[min(len(values) - 1, len(snapshots) - 1)][0]).days / 365.0, 0.0)
    terminal_quotes = snapshots[min(len(values) - 1, len(snapshots) - 1)][2]
    terminal_hedge_rows = [q for q in terminal_quotes if q.expiry == hedge_expiry.isoformat()]
    terminal_hedge_price = (
        _interpolate_call_price(terminal_hedge_rows, hedge_strike)
        if terminal_tau_hedge > 0.0 and terminal_hedge_rows
        else max(terminal_spot - hedge_strike, 0.0)
    )
    unwind_cost = (
        abs(stock_units) * terminal_spot * cfg.transaction_cost_stock
        + abs(hedge_units) * terminal_hedge_price * cfg.transaction_cost_option
    )
    values[-1] = values[-1] - unwind_cost
    total_costs += unwind_cost

    return {
        "path_values": values,
        "terminal_pnl": values[-1],
        "total_transaction_cost": total_costs,
        "rebalances": rebalances,
    }


def run_rolling_recalibration_experiment(
    quotes: list[OptionQuote],
    cfg: RollingRecalibrationConfig,
) -> dict:
    grouped = _group_quotes_by_valuation_date(quotes)
    valuation_dates = sorted(grouped.keys())
    if len(valuation_dates) < 5:
        raise ValueError("Need at least 5 valuation dates for rolling experiment")

    calibrated_surface_by_date: dict[str, SVISurfaceModel] = {}
    calibration_rmse_by_date: dict[str, float] = {}
    for valuation_s in valuation_dates:
        snapshot_quotes = grouped[valuation_s]
        try:
            surface_result = run_surface_pipeline(snapshot_quotes, seed=cfg.calibration_seed)
            calibrated_surface_by_date[valuation_s] = SVISurfaceModel.from_pipeline_result(
                surface_result
            )
            rmses = [
                block["svi_fit"]["rmse_total_variance"]
                for block in surface_result["per_maturity"]
            ]
            calibration_rmse_by_date[valuation_s] = sum(rmses) / len(rmses) if rmses else 0.0
        except Exception:
            continue

    if len(calibrated_surface_by_date) < 5:
        raise RuntimeError("Rolling experiment failed: not enough calibrated dates")

    strategies = ["unhedged", "delta", "delta-vega"]
    terminal_by_strategy: dict[str, list[float]] = {s: [] for s in strategies}
    cost_by_strategy: dict[str, list[float]] = {s: [] for s in strategies}
    rebalance_by_strategy: dict[str, list[int]] = {s: [] for s in strategies}
    representative_path_values: dict[str, list[float]] = {}
    representative_contracts = None
    window_summaries = []
    used_calibration_dates: set[str] = set()

    for start_idx, start_date_s in enumerate(valuation_dates):
        if len(window_summaries) >= cfg.max_windows:
            break
        start_snapshot = grouped[start_date_s]
        try:
            target_expiry_s, hedge_expiry_s, target_strike, hedge_strike = _choose_contracts(
                start_snapshot, cfg
            )
        except Exception:
            continue

        target_expiry = _parse_date(target_expiry_s)
        hedge_expiry = _parse_date(hedge_expiry_s)
        snapshots: list[tuple[date, str, list[OptionQuote], SVISurfaceModel]] = []

        for valuation_s in valuation_dates[start_idx:]:
            if len(snapshots) >= cfg.max_rebalance_dates:
                break
            valuation_dt = _parse_date(valuation_s)
            if valuation_dt >= target_expiry:
                break
            if valuation_s not in calibrated_surface_by_date:
                break
            snapshot_quotes = grouped[valuation_s]
            if not any(q.expiry == target_expiry_s for q in snapshot_quotes):
                break
            snapshots.append(
                (
                    valuation_dt,
                    valuation_s,
                    snapshot_quotes,
                    calibrated_surface_by_date[valuation_s],
                )
            )

        if len(snapshots) < 3:
            continue

        run_for_window = {}
        for strategy in strategies:
            run = _strategy_backtest(
                strategy=strategy,
                snapshots=snapshots,
                cfg=cfg,
                target_expiry=target_expiry,
                hedge_expiry=hedge_expiry,
                target_strike=target_strike,
                hedge_strike=hedge_strike,
            )
            run_for_window[strategy] = run
            terminal_by_strategy[strategy].append(run["terminal_pnl"])
            cost_by_strategy[strategy].append(run["total_transaction_cost"])
            rebalance_by_strategy[strategy].append(run["rebalances"])
            if strategy not in representative_path_values:
                representative_path_values[strategy] = run["path_values"]

        if representative_contracts is None:
            representative_contracts = {
                "target_expiry": target_expiry_s,
                "hedge_expiry": hedge_expiry_s,
                "target_strike": target_strike,
                "hedge_strike": hedge_strike,
            }

        for _, valuation_s, _, _ in snapshots:
            used_calibration_dates.add(valuation_s)

        window_summaries.append(
            {
                "start_date": snapshots[0][1],
                "end_date": snapshots[-1][1],
                "num_dates": len(snapshots),
                "contracts": {
                    "target_expiry": target_expiry_s,
                    "hedge_expiry": hedge_expiry_s,
                    "target_strike": target_strike,
                    "hedge_strike": hedge_strike,
                },
                "strategy_terminal_pnl": {
                    s: run_for_window[s]["terminal_pnl"] for s in strategies
                },
            }
        )

    if not window_summaries:
        raise RuntimeError("Rolling experiment failed: no valid windows were produced")

    metrics = {}
    for strategy in strategies:
        stats = summary_stats(terminal_by_strategy[strategy])
        stats["avg_transaction_cost"] = sum(cost_by_strategy[strategy]) / len(
            cost_by_strategy[strategy]
        )
        stats["avg_rebalances"] = sum(rebalance_by_strategy[strategy]) / len(
            rebalance_by_strategy[strategy]
        )
        metrics[strategy] = stats

    unhedged_std = metrics["unhedged"]["std_pnl"]
    unhedged_abs_mean = abs(metrics["unhedged"]["mean_pnl"])
    for strategy in ["delta", "delta-vega"]:
        strategy_std = metrics[strategy]["std_pnl"]
        metrics[strategy]["std_reduction_vs_unhedged"] = (
            (unhedged_std - strategy_std) / unhedged_std if unhedged_std > 1e-12 else 0.0
        )
        strategy_abs_mean = abs(metrics[strategy]["mean_pnl"])
        metrics[strategy]["abs_mean_pnl_reduction_vs_unhedged"] = (
            (unhedged_abs_mean - strategy_abs_mean) / unhedged_abs_mean
            if unhedged_abs_mean > 1e-12
            else 0.0
        )

    used_rmse = [calibration_rmse_by_date[d] for d in sorted(used_calibration_dates)]
    return {
        "config": asdict(cfg),
        "window_study": {
            "num_windows": len(window_summaries),
            "first_window_start": window_summaries[0]["start_date"],
            "last_window_start": window_summaries[-1]["start_date"],
            "representative_contracts": representative_contracts,
        },
        "calibration_quality": {
            "avg_rmse_total_variance": sum(used_rmse) / len(used_rmse) if used_rmse else 0.0,
            "per_date_rmse": {d: calibration_rmse_by_date[d] for d in sorted(used_calibration_dates)},
        },
        "strategy_metrics": metrics,
        "representative_path_values": representative_path_values,
        "window_summaries": window_summaries,
    }


def render_rolling_markdown(result: dict) -> str:
    window_study = result["window_study"]
    contracts = window_study["representative_contracts"]
    metrics = result["strategy_metrics"]
    quality = result["calibration_quality"]

    lines = []
    lines.append("# Rolling Recalibration + Out-of-Sample Hedging Report")
    lines.append("")
    lines.append("## Window")
    lines.append(
        f"- Number of rolling windows: `{window_study['num_windows']}`"
    )
    lines.append(
        f"- Window starts: `{window_study['first_window_start']}` to `{window_study['last_window_start']}`"
    )
    lines.append(
        f"- Target contract: expiry `{contracts['target_expiry']}`, strike `{contracts['target_strike']:.2f}`"
    )
    lines.append(
        f"- Hedge contract: expiry `{contracts['hedge_expiry']}`, strike `{contracts['hedge_strike']:.2f}`"
    )
    lines.append(
        f"- Average calibration RMSE(w): `{quality['avg_rmse_total_variance']:.8f}`"
    )
    lines.append("")
    lines.append("## Strategy Comparison")
    lines.append(
        "| Strategy | Mean PnL | Std PnL | VaR 95% | ES 95% | Avg Tx Cost | Avg Rebalances | Std Reduction vs Unhedged |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for strategy in ["unhedged", "delta", "delta-vega"]:
        m = metrics[strategy]
        reduction = m.get("std_reduction_vs_unhedged", 0.0)
        lines.append(
            f"| {strategy} | {m['mean_pnl']:.6f} | {m['std_pnl']:.6f} | {m['var_95']:.6f} | "
            f"{m['expected_shortfall_95']:.6f} | {m['avg_transaction_cost']:.6f} | "
            f"{m['avg_rebalances']:.2f} | {reduction:.2%} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "- This test recalibrates the surface at every snapshot and hedges out-of-sample in the next interval."
    )
    lines.append(
        "- Use this as the core empirical section for model-risk and execution-cost discussion."
    )
    lines.append("")
    return "\n".join(lines)
