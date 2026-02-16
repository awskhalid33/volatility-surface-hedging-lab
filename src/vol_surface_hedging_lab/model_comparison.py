import math
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone

from .pipeline import _to_quote_with_iv  # reuse one consistent IV transformation
from .sabr import fit_sabr_from_observations, sabr_implied_vol
from .svi import fit_svi_from_observations, svi_total_variance
from .types import OptionQuote


def _rmse(values: list[float], targets: list[float]) -> float:
    err = 0.0
    for v, t in zip(values, targets):
        d = v - t
        err += d * d
    return math.sqrt(err / len(values)) if values else 0.0


def _kfold_splits(n: int, folds: int = 3) -> list[tuple[list[int], list[int]]]:
    if n < 2:
        return []
    effective_folds = min(folds, n)
    splits: list[tuple[list[int], list[int]]] = []
    for fold in range(effective_folds):
        test = [i for i in range(n) if i % effective_folds == fold]
        train = [i for i in range(n) if i % effective_folds != fold]
        if not train or not test:
            continue
        splits.append((train, test))
    return splits


def _eval_svi(
    forward: float,
    maturity: float,
    strikes: list[float],
    ivs: list[float],
    train_idx: list[int],
    test_idx: list[int],
    seed: int,
) -> float | None:
    if len(train_idx) < 5:
        return None
    train_strikes = [strikes[i] for i in train_idx]
    train_ivs = [ivs[i] for i in train_idx]
    params, _ = fit_svi_from_observations(
        forward=forward,
        strikes=train_strikes,
        maturity=maturity,
        implied_vols=train_ivs,
        seed=seed,
    )
    predicted = []
    observed = []
    for i in test_idx:
        k = strikes[i]
        iv = ivs[i]
        log_m = math.log(k / forward)
        total_var = max(1e-12, svi_total_variance(log_m, params))
        pred_iv = math.sqrt(total_var / maturity)
        predicted.append(pred_iv)
        observed.append(iv)
    return _rmse(predicted, observed)


def _eval_sabr(
    forward: float,
    maturity: float,
    strikes: list[float],
    ivs: list[float],
    train_idx: list[int],
    test_idx: list[int],
    seed: int,
    beta: float,
) -> float | None:
    if len(train_idx) < 4:
        return None
    train_strikes = [strikes[i] for i in train_idx]
    train_ivs = [ivs[i] for i in train_idx]
    params, _ = fit_sabr_from_observations(
        forward=forward,
        strikes=train_strikes,
        maturity=maturity,
        implied_vols=train_ivs,
        beta=beta,
        seed=seed,
    )
    predicted = [sabr_implied_vol(forward, strikes[i], maturity, params) for i in test_idx]
    observed = [ivs[i] for i in test_idx]
    return _rmse(predicted, observed)


def run_model_comparison(
    quotes: list[OptionQuote],
    seed: int = 17,
    sabr_beta: float = 1.0,
    folds: int = 3,
) -> dict:
    transformed = [_to_quote_with_iv(q) for q in quotes]
    by_maturity: dict[float, list] = defaultdict(list)
    for row in transformed:
        by_maturity[row.quote.maturity].append(row)

    per_maturity = []
    svi_in_rmse_values = []
    sabr_in_rmse_values = []
    svi_oos_rmse_values = []
    sabr_oos_rmse_values = []

    for maturity in sorted(by_maturity.keys()):
        rows = sorted(by_maturity[maturity], key=lambda x: x.quote.strike)
        strikes = [r.quote.strike for r in rows]
        ivs = [r.implied_vol for r in rows]
        forward = rows[0].forward

        svi_params, svi_in_rmse = fit_svi_from_observations(
            forward=forward,
            strikes=strikes,
            maturity=maturity,
            implied_vols=ivs,
            seed=seed,
        )
        sabr_params, sabr_in_rmse = fit_sabr_from_observations(
            forward=forward,
            strikes=strikes,
            maturity=maturity,
            implied_vols=ivs,
            beta=sabr_beta,
            seed=seed + 1,
        )
        splits = _kfold_splits(len(strikes), folds=folds)
        svi_oos = []
        sabr_oos = []
        for split_idx, (train, test) in enumerate(splits):
            svi_fold = _eval_svi(
                forward=forward,
                maturity=maturity,
                strikes=strikes,
                ivs=ivs,
                train_idx=train,
                test_idx=test,
                seed=seed + 10 + split_idx,
            )
            sabr_fold = _eval_sabr(
                forward=forward,
                maturity=maturity,
                strikes=strikes,
                ivs=ivs,
                train_idx=train,
                test_idx=test,
                seed=seed + 100 + split_idx,
                beta=sabr_beta,
            )
            if svi_fold is not None:
                svi_oos.append(svi_fold)
            if sabr_fold is not None:
                sabr_oos.append(sabr_fold)

        svi_oos_rmse = sum(svi_oos) / len(svi_oos) if svi_oos else None
        sabr_oos_rmse = sum(sabr_oos) / len(sabr_oos) if sabr_oos else None

        per_maturity.append(
            {
                "maturity": maturity,
                "num_quotes": len(rows),
                "forward": forward,
                "svi": {
                    "params": asdict(svi_params),
                    "in_sample_rmse_iv": svi_in_rmse,
                    "oos_cv_rmse_iv": svi_oos_rmse,
                },
                "sabr": {
                    "params": asdict(sabr_params),
                    "in_sample_rmse_iv": sabr_in_rmse,
                    "oos_cv_rmse_iv": sabr_oos_rmse,
                },
            }
        )

        svi_in_rmse_values.append(svi_in_rmse)
        sabr_in_rmse_values.append(sabr_in_rmse)
        if svi_oos_rmse is not None:
            svi_oos_rmse_values.append(svi_oos_rmse)
        if sabr_oos_rmse is not None:
            sabr_oos_rmse_values.append(sabr_oos_rmse)

    svi_in_avg = sum(svi_in_rmse_values) / len(svi_in_rmse_values)
    sabr_in_avg = sum(sabr_in_rmse_values) / len(sabr_in_rmse_values)
    svi_oos_avg = sum(svi_oos_rmse_values) / len(svi_oos_rmse_values) if svi_oos_rmse_values else None
    sabr_oos_avg = sum(sabr_oos_rmse_values) / len(sabr_oos_rmse_values) if sabr_oos_rmse_values else None

    winner_in = "SVI" if svi_in_avg < sabr_in_avg else "SABR"
    winner_oos = None
    if svi_oos_avg is not None and sabr_oos_avg is not None:
        winner_oos = "SVI" if svi_oos_avg < sabr_oos_avg else "SABR"

    return {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "num_quotes": len(quotes),
            "num_maturities": len(per_maturity),
            "seed": seed,
            "sabr_beta": sabr_beta,
            "folds": folds,
        },
        "summary": {
            "avg_in_sample_rmse_iv": {
                "svi": svi_in_avg,
                "sabr": sabr_in_avg,
            },
            "avg_oos_cv_rmse_iv": {
                "svi": svi_oos_avg,
                "sabr": sabr_oos_avg,
            },
            "winner_in_sample": winner_in,
            "winner_oos_cv": winner_oos,
        },
        "per_maturity": per_maturity,
    }


def render_model_comparison_markdown(result: dict) -> str:
    meta = result["metadata"]
    summary = result["summary"]

    lines = []
    lines.append("# Volatility Model Comparison Report (SVI vs SABR)")
    lines.append("")
    lines.append(f"- Generated (UTC): `{meta['created_utc']}`")
    lines.append(f"- Number of quotes: `{meta['num_quotes']}`")
    lines.append(f"- Number of maturities: `{meta['num_maturities']}`")
    lines.append(f"- SABR beta: `{meta['sabr_beta']}`")
    lines.append(f"- CV folds: `{meta['folds']}`")
    lines.append("")
    lines.append("## Aggregate Results")
    lines.append(
        f"- Average in-sample RMSE (IV): `SVI={summary['avg_in_sample_rmse_iv']['svi']:.8f}`, "
        f"`SABR={summary['avg_in_sample_rmse_iv']['sabr']:.8f}`"
    )
    svi_oos = summary["avg_oos_cv_rmse_iv"]["svi"]
    sabr_oos = summary["avg_oos_cv_rmse_iv"]["sabr"]
    if svi_oos is not None and sabr_oos is not None:
        lines.append(
            f"- Average out-of-sample CV RMSE (IV): `SVI={svi_oos:.8f}`, `SABR={sabr_oos:.8f}`"
        )
    lines.append(f"- In-sample winner: `{summary['winner_in_sample']}`")
    if summary["winner_oos_cv"] is not None:
        lines.append(f"- Out-of-sample winner: `{summary['winner_oos_cv']}`")
    lines.append("")
    lines.append("## Per-Maturity Results")
    lines.append("| Maturity | SVI RMSE (IV) | SABR RMSE (IV) | SVI OOS RMSE | SABR OOS RMSE |")
    lines.append("|---:|---:|---:|---:|---:|")
    for block in result["per_maturity"]:
        lines.append(
            f"| {block['maturity']:.6f} | {block['svi']['in_sample_rmse_iv']:.8f} | "
            f"{block['sabr']['in_sample_rmse_iv']:.8f} | "
            f"{(block['svi']['oos_cv_rmse_iv'] if block['svi']['oos_cv_rmse_iv'] is not None else float('nan')):.8f} | "
            f"{(block['sabr']['oos_cv_rmse_iv'] if block['sabr']['oos_cv_rmse_iv'] is not None else float('nan')):.8f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- Use out-of-sample RMSE to select the production smile model for hedging.")
    lines.append("- Use in-sample RMSE only as a calibration quality diagnostic.")
    lines.append("")
    return "\n".join(lines)
