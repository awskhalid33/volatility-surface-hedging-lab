import math
import random
from dataclasses import asdict, dataclass

from .black_scholes import bs_call_price
from .hedging_common import clamp, compute_desired_positions, summary_stats
from .surface import SVISurfaceModel


@dataclass(frozen=True)
class HedgingConfig:
    spot0: float = 100.0
    rate: float = 0.02
    dividend: float = 0.0
    target_strike: float = 100.0
    hedge_strike: float = 110.0
    target_maturity: float = 0.5
    hedge_maturity: float = 0.75
    steps: int = 126
    paths: int = 400
    seed: int = 21
    low_regime_vol: float = 0.16
    high_regime_vol: float = 0.34
    transition_p00: float = 0.94
    transition_p11: float = 0.78
    initial_regime: int = 0
    transaction_cost_stock: float = 0.0005
    transaction_cost_option: float = 0.0015
    max_abs_stock_position: float = 8.0
    max_abs_hedge_option_position: float = 8.0


@dataclass(frozen=True)
class MarketScenario:
    spots: list[float]
    regimes: list[int]


def _regime_transition(current_regime: int, cfg: HedgingConfig, rng: random.Random) -> int:
    u = rng.random()
    if current_regime == 0:
        return 0 if u < cfg.transition_p00 else 1
    return 1 if u < cfg.transition_p11 else 0


def _market_call_vol(base_vol: float, spot: float, strike: float, tau: float) -> float:
    if tau <= 0.0:
        return 0.0001
    log_m = math.log(strike / spot)
    skew = -0.10 * log_m
    curvature = 0.18 * abs(log_m)
    term_bump = 0.015 * math.exp(-3.0 * tau)
    vol = base_vol * (1.0 + skew + curvature) + term_bump
    return clamp(vol, 0.05, 2.0)


def _market_call_price(
    base_vol: float,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    dividend: float,
) -> float:
    if tau <= 0.0:
        return max(spot - strike, 0.0)
    iv = _market_call_vol(base_vol=base_vol, spot=spot, strike=strike, tau=tau)
    return bs_call_price(
        spot=spot,
        strike=strike,
        maturity=tau,
        rate=rate,
        dividend=dividend,
        vol=iv,
    )


def _simulate_market_scenarios(cfg: HedgingConfig) -> list[MarketScenario]:
    if cfg.steps <= 0:
        raise ValueError("steps must be positive")
    if cfg.paths <= 0:
        raise ValueError("paths must be positive")

    dt = cfg.target_maturity / cfg.steps
    sqrt_dt = math.sqrt(dt)
    rng = random.Random(cfg.seed)

    scenarios: list[MarketScenario] = []
    for _ in range(cfg.paths):
        spots = [cfg.spot0]
        regimes = [cfg.initial_regime]
        spot = cfg.spot0
        regime = cfg.initial_regime

        for _step in range(1, cfg.steps + 1):
            regime = _regime_transition(regime, cfg, rng)
            base_vol = cfg.low_regime_vol if regime == 0 else cfg.high_regime_vol
            z = rng.gauss(0.0, 1.0)
            spot *= math.exp(
                (cfg.rate - cfg.dividend - 0.5 * base_vol * base_vol) * dt
                + base_vol * sqrt_dt * z
            )
            spots.append(spot)
            regimes.append(regime)
        scenarios.append(MarketScenario(spots=spots, regimes=regimes))

    return scenarios


def _portfolio_value(
    cash: float,
    stock_units: float,
    hedge_option_units: float,
    spot: float,
    hedge_option_price: float,
    target_option_price: float,
) -> float:
    return (
        cash + stock_units * spot + hedge_option_units * hedge_option_price - target_option_price
    )


def _evaluate_strategy_on_scenario(
    strategy: str,
    surface: SVISurfaceModel,
    cfg: HedgingConfig,
    scenario: MarketScenario,
) -> dict:
    dt = cfg.target_maturity / cfg.steps
    cash = 0.0
    stock_units = 0.0
    hedge_option_units = 0.0
    transaction_costs = 0.0
    values: list[float] = []

    spot0 = scenario.spots[0]
    regime0 = scenario.regimes[0]
    base_vol0 = cfg.low_regime_vol if regime0 == 0 else cfg.high_regime_vol
    target_price0 = _market_call_price(
        base_vol=base_vol0,
        spot=spot0,
        strike=cfg.target_strike,
        tau=cfg.target_maturity,
        rate=cfg.rate,
        dividend=cfg.dividend,
    )
    hedge_price0 = _market_call_price(
        base_vol=base_vol0,
        spot=spot0,
        strike=cfg.hedge_strike,
        tau=cfg.hedge_maturity,
        rate=cfg.rate,
        dividend=cfg.dividend,
    )

    cash += target_price0
    desired_stock, desired_hedge = compute_desired_positions(
        strategy=strategy,
        surface=surface,
        spot=spot0,
        rate=cfg.rate,
        dividend=cfg.dividend,
        target_strike=cfg.target_strike,
        hedge_strike=cfg.hedge_strike,
        tau_target=cfg.target_maturity,
        tau_hedge=cfg.hedge_maturity,
        max_abs_stock_position=cfg.max_abs_stock_position,
        max_abs_hedge_option_position=cfg.max_abs_hedge_option_position,
    )
    stock_trade = desired_stock - stock_units
    hedge_trade = desired_hedge - hedge_option_units
    cash -= stock_trade * spot0 + hedge_trade * hedge_price0
    trade_cost = (
        abs(stock_trade) * spot0 * cfg.transaction_cost_stock
        + abs(hedge_trade) * hedge_price0 * cfg.transaction_cost_option
    )
    cash -= trade_cost
    transaction_costs += trade_cost
    stock_units = desired_stock
    hedge_option_units = desired_hedge

    value0 = _portfolio_value(
        cash=cash,
        stock_units=stock_units,
        hedge_option_units=hedge_option_units,
        spot=spot0,
        hedge_option_price=hedge_price0,
        target_option_price=target_price0,
    )
    values.append(value0)

    for step in range(1, cfg.steps + 1):
        cash *= math.exp(cfg.rate * dt)
        t = step * dt
        tau_target = max(cfg.target_maturity - t, 0.0)
        tau_hedge = max(cfg.hedge_maturity - t, 0.0)

        spot = scenario.spots[step]
        regime = scenario.regimes[step]
        base_vol = cfg.low_regime_vol if regime == 0 else cfg.high_regime_vol

        target_price = _market_call_price(
            base_vol=base_vol,
            spot=spot,
            strike=cfg.target_strike,
            tau=tau_target,
            rate=cfg.rate,
            dividend=cfg.dividend,
        )
        hedge_price = _market_call_price(
            base_vol=base_vol,
            spot=spot,
            strike=cfg.hedge_strike,
            tau=tau_hedge,
            rate=cfg.rate,
            dividend=cfg.dividend,
        )

        value = _portfolio_value(
            cash=cash,
            stock_units=stock_units,
            hedge_option_units=hedge_option_units,
            spot=spot,
            hedge_option_price=hedge_price,
            target_option_price=target_price,
        )
        values.append(value)

        if step < cfg.steps and tau_target > 0.0:
            desired_stock, desired_hedge = compute_desired_positions(
                strategy=strategy,
                surface=surface,
                spot=spot,
                rate=cfg.rate,
                dividend=cfg.dividend,
                target_strike=cfg.target_strike,
                hedge_strike=cfg.hedge_strike,
                tau_target=tau_target,
                tau_hedge=tau_hedge,
                max_abs_stock_position=cfg.max_abs_stock_position,
                max_abs_hedge_option_position=cfg.max_abs_hedge_option_position,
            )
            stock_trade = desired_stock - stock_units
            hedge_trade = desired_hedge - hedge_option_units
            cash -= stock_trade * spot + hedge_trade * hedge_price
            trade_cost = (
                abs(stock_trade) * spot * cfg.transaction_cost_stock
                + abs(hedge_trade) * hedge_price * cfg.transaction_cost_option
            )
            cash -= trade_cost
            transaction_costs += trade_cost
            stock_units = desired_stock
            hedge_option_units = desired_hedge

    terminal_spot = scenario.spots[-1]
    terminal_regime = scenario.regimes[-1]
    terminal_base_vol = cfg.low_regime_vol if terminal_regime == 0 else cfg.high_regime_vol
    terminal_tau_hedge = max(cfg.hedge_maturity - cfg.target_maturity, 0.0)
    terminal_hedge_price = _market_call_price(
        base_vol=terminal_base_vol,
        spot=terminal_spot,
        strike=cfg.hedge_strike,
        tau=terminal_tau_hedge,
        rate=cfg.rate,
        dividend=cfg.dividend,
    )
    unwind_cost = (
        abs(stock_units) * terminal_spot * cfg.transaction_cost_stock
        + abs(hedge_option_units) * terminal_hedge_price * cfg.transaction_cost_option
    )
    terminal_value = values[-1] - unwind_cost
    values[-1] = terminal_value
    transaction_costs += unwind_cost

    return {
        "terminal_pnl": terminal_value,
        "values": values,
        "transaction_costs": transaction_costs,
    }


def run_hedging_experiment(surface: SVISurfaceModel, cfg: HedgingConfig) -> dict:
    scenarios = _simulate_market_scenarios(cfg)
    strategies = ["unhedged", "delta", "delta-vega"]
    strategy_to_terminal: dict[str, list[float]] = {s: [] for s in strategies}
    strategy_to_costs: dict[str, list[float]] = {s: [] for s in strategies}
    representative_paths: dict[str, list[float]] = {}

    for strategy in strategies:
        for idx, scenario in enumerate(scenarios):
            evaluation = _evaluate_strategy_on_scenario(
                strategy=strategy,
                surface=surface,
                cfg=cfg,
                scenario=scenario,
            )
            strategy_to_terminal[strategy].append(evaluation["terminal_pnl"])
            strategy_to_costs[strategy].append(evaluation["transaction_costs"])
            if idx == 0:
                representative_paths[strategy] = evaluation["values"]

    metrics = {}
    for strategy in strategies:
        stats = summary_stats(strategy_to_terminal[strategy])
        avg_cost = sum(strategy_to_costs[strategy]) / len(strategy_to_costs[strategy])
        stats["avg_transaction_cost"] = avg_cost
        metrics[strategy] = stats

    unhedged_std = metrics["unhedged"]["std_pnl"]
    for strategy in ["delta", "delta-vega"]:
        strategy_std = metrics[strategy]["std_pnl"]
        reduction = (
            (unhedged_std - strategy_std) / unhedged_std if unhedged_std > 1e-12 else 0.0
        )
        metrics[strategy]["std_reduction_vs_unhedged"] = reduction

    return {
        "config": asdict(cfg),
        "strategy_metrics": metrics,
        "terminal_pnl_by_strategy": strategy_to_terminal,
        "representative_path_values": representative_paths,
    }


def render_backtest_markdown(result: dict) -> str:
    cfg = result["config"]
    metrics = result["strategy_metrics"]

    lines = []
    lines.append("# Dynamic Hedging Backtest Report")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- Paths: `{cfg['paths']}`")
    lines.append(f"- Rebalance steps: `{cfg['steps']}`")
    lines.append(
        f"- Target option: `K={cfg['target_strike']:.2f}`, `T={cfg['target_maturity']:.4f}`"
    )
    lines.append(
        f"- Hedge option: `K={cfg['hedge_strike']:.2f}`, `T={cfg['hedge_maturity']:.4f}`"
    )
    lines.append(
        "- Regime vols: "
        f"`low={cfg['low_regime_vol']:.3f}`, `high={cfg['high_regime_vol']:.3f}`"
    )
    lines.append("")
    lines.append("## Strategy Comparison")
    lines.append(
        "| Strategy | Mean PnL | Std PnL | VaR 95% | ES 95% | Avg Tx Cost | Std Reduction vs Unhedged |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    for strategy in ["unhedged", "delta", "delta-vega"]:
        m = metrics[strategy]
        reduction = m.get("std_reduction_vs_unhedged", 0.0)
        lines.append(
            f"| {strategy} | {m['mean_pnl']:.6f} | {m['std_pnl']:.6f} | "
            f"{m['var_95']:.6f} | {m['expected_shortfall_95']:.6f} | "
            f"{m['avg_transaction_cost']:.6f} | {reduction:.2%} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "- Compare risk reduction against transaction costs to evaluate whether vega hedging adds net value."
    )
    lines.append(
        "- Use this table in your report to justify the hedging policy under regime-dependent volatility."
    )
    lines.append("")
    return "\n".join(lines)
