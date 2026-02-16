import math
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone

from .arbitrage import (
    check_calendar_total_variance,
    check_call_convexity,
    check_call_monotonicity,
)
from .black_scholes import implied_volatility_call
from .svi import fit_svi_from_observations
from .types import OptionQuote, QuoteWithIV


def _to_quote_with_iv(quote: OptionQuote) -> QuoteWithIV:
    iv = implied_volatility_call(
        call_price=quote.call_mid,
        spot=quote.spot,
        strike=quote.strike,
        maturity=quote.maturity,
        rate=quote.rate,
        dividend=quote.dividend,
    )
    forward = quote.spot * math.exp((quote.rate - quote.dividend) * quote.maturity)
    log_moneyness = math.log(quote.strike / forward)
    total_variance = iv * iv * quote.maturity
    return QuoteWithIV(
        quote=quote,
        implied_vol=iv,
        forward=forward,
        log_moneyness=log_moneyness,
        total_variance=total_variance,
    )


def run_surface_pipeline(quotes: list[OptionQuote], seed: int = 11) -> dict:
    with_iv: list[QuoteWithIV] = [_to_quote_with_iv(q) for q in quotes]
    by_maturity: dict[float, list[QuoteWithIV]] = defaultdict(list)
    for row in with_iv:
        by_maturity[row.quote.maturity].append(row)

    per_maturity = []
    maturity_to_strike_iv: dict[float, dict[float, float]] = {}
    total_monotonicity_issues = 0
    total_convexity_issues = 0

    for maturity in sorted(by_maturity.keys()):
        rows = sorted(by_maturity[maturity], key=lambda x: x.quote.strike)
        strikes = [r.quote.strike for r in rows]
        call_prices = [r.quote.call_mid for r in rows]
        ivs = [r.implied_vol for r in rows]
        forward = rows[0].forward

        monotonicity_issues = check_call_monotonicity(strikes, call_prices)
        convexity_issues = check_call_convexity(strikes, call_prices)
        total_monotonicity_issues += len(monotonicity_issues)
        total_convexity_issues += len(convexity_issues)

        svi_params, svi_rmse = fit_svi_from_observations(
            forward=forward,
            strikes=strikes,
            maturity=maturity,
            implied_vols=ivs,
            seed=seed,
        )

        maturity_to_strike_iv[maturity] = {
            strike: iv for strike, iv in zip(strikes, ivs)
        }
        per_maturity.append(
            {
                "maturity": maturity,
                "num_quotes": len(rows),
                "forward": forward,
                "arbitrage_issues": {
                    "monotonicity": monotonicity_issues,
                    "convexity": convexity_issues,
                },
                "svi_fit": {
                    "params": asdict(svi_params),
                    "rmse_total_variance": svi_rmse,
                },
                "rows": [
                    {
                        "strike": r.quote.strike,
                        "call_mid": r.quote.call_mid,
                        "implied_vol": r.implied_vol,
                        "log_moneyness": r.log_moneyness,
                        "total_variance": r.total_variance,
                    }
                    for r in rows
                ],
            }
        )

    calendar_issues = check_calendar_total_variance(maturity_to_strike_iv)

    result = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "num_quotes": len(quotes),
            "num_maturities": len(by_maturity),
            "seed": seed,
        },
        "summary": {
            "monotonicity_violations": total_monotonicity_issues,
            "convexity_violations": total_convexity_issues,
            "calendar_violations": len(calendar_issues),
        },
        "calendar_issues": calendar_issues,
        "per_maturity": per_maturity,
    }
    return result


def render_markdown_report(result: dict) -> str:
    lines = []
    summary = result["summary"]
    metadata = result["metadata"]

    lines.append("# Volatility Surface Report")
    lines.append("")
    lines.append(f"- Generated (UTC): `{metadata['created_utc']}`")
    lines.append(f"- Number of quotes: `{metadata['num_quotes']}`")
    lines.append(f"- Number of maturities: `{metadata['num_maturities']}`")
    lines.append("")
    lines.append("## Arbitrage Diagnostics")
    lines.append(f"- Monotonicity violations: `{summary['monotonicity_violations']}`")
    lines.append(f"- Convexity violations: `{summary['convexity_violations']}`")
    lines.append(f"- Calendar total variance violations: `{summary['calendar_violations']}`")
    lines.append("")

    if result["calendar_issues"]:
        lines.append("### Calendar Issues")
        for issue in result["calendar_issues"]:
            lines.append(f"- {issue}")
        lines.append("")

    lines.append("## SVI Fits")
    for block in result["per_maturity"]:
        fit = block["svi_fit"]
        p = fit["params"]
        lines.append(
            f"- `T={block['maturity']:.6f}`: "
            f"RMSE(w)={fit['rmse_total_variance']:.8f}, "
            f"params=(a={p['a']:.6f}, b={p['b']:.6f}, rho={p['rho']:.6f}, "
            f"m={p['m']:.6f}, sigma={p['sigma']:.6f})"
        )

    lines.append("")
    lines.append("## Next Step")
    lines.append(
        "- Replace synthetic inputs with live option snapshots and rerun the same diagnostics."
    )
    lines.append("")
    return "\n".join(lines)
