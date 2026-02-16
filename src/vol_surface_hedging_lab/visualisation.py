import math
from pathlib import Path

from .sabr import SABRParams, sabr_implied_vol
from .svi import SVIParams, svi_total_variance


def _safe_min_max(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        pad = 1.0 if abs(lo) < 1e-12 else 0.1 * abs(lo)
        return lo - pad, hi + pad
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def _map(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if abs(hi - lo) < 1e-12:
        return 0.5 * (out_lo + out_hi)
    t = (value - lo) / (hi - lo)
    return out_lo + t * (out_hi - out_lo)


def _write_svg(path: Path, width: int, height: int, elements: list[str]) -> None:
    text = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        *elements,
        "</svg>",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def _line_chart(
    path: Path,
    title: str,
    x_values: list[float],
    series: list[tuple[str, str, list[float]]],
    x_label: str,
    y_label: str,
) -> None:
    width, height = 920, 560
    m_left, m_right, m_top, m_bottom = 85, 26, 64, 78
    x0, x1 = m_left, width - m_right
    y0, y1 = height - m_bottom, m_top

    all_y = []
    for _, _, ys in series:
        all_y.extend(ys)
    x_lo, x_hi = _safe_min_max(x_values)
    y_lo, y_hi = _safe_min_max(all_y)

    elems = []
    elems.append(f'<text x="{width/2:.1f}" y="36" text-anchor="middle" font-family="Arial" font-size="22">{title}</text>')
    elems.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#111" stroke-width="1.3"/>')
    elems.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#111" stroke-width="1.3"/>')

    for i in range(6):
        xv = x_lo + (x_hi - x_lo) * i / 5.0
        px = _map(xv, x_lo, x_hi, x0, x1)
        elems.append(f'<line x1="{px:.2f}" y1="{y0}" x2="{px:.2f}" y2="{y0+6}" stroke="#222" stroke-width="1"/>')
        elems.append(f'<text x="{px:.2f}" y="{y0+24}" text-anchor="middle" font-family="Arial" font-size="12">{xv:.3f}</text>')
        elems.append(f'<line x1="{px:.2f}" y1="{y0}" x2="{px:.2f}" y2="{y1}" stroke="#eee" stroke-width="1"/>')

    for i in range(6):
        yv = y_lo + (y_hi - y_lo) * i / 5.0
        py = _map(yv, y_lo, y_hi, y0, y1)
        elems.append(f'<line x1="{x0-6}" y1="{py:.2f}" x2="{x0}" y2="{py:.2f}" stroke="#222" stroke-width="1"/>')
        elems.append(f'<text x="{x0-10}" y="{py+4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{yv:.4f}</text>')
        elems.append(f'<line x1="{x0}" y1="{py:.2f}" x2="{x1}" y2="{py:.2f}" stroke="#eee" stroke-width="1"/>')

    elems.append(f'<text x="{(x0+x1)/2:.1f}" y="{height-22}" text-anchor="middle" font-family="Arial" font-size="14">{x_label}</text>')
    elems.append(
        f'<text x="24" y="{(y0+y1)/2:.1f}" transform="rotate(-90 24 {(y0+y1)/2:.1f})" text-anchor="middle" font-family="Arial" font-size="14">{y_label}</text>'
    )

    legend_x = x0 + 12
    legend_y = y1 + 10
    for idx, (name, colour, ys) in enumerate(series):
        points = []
        for xv, yv in zip(x_values, ys):
            px = _map(xv, x_lo, x_hi, x0, x1)
            py = _map(yv, y_lo, y_hi, y0, y1)
            points.append(f"{px:.2f},{py:.2f}")
        elems.append(f'<polyline fill="none" stroke="{colour}" stroke-width="2.2" points="{" ".join(points)}"/>')
        ly = legend_y + 22 * idx
        elems.append(f'<rect x="{legend_x}" y="{ly-10}" width="18" height="6" fill="{colour}"/>')
        elems.append(f'<text x="{legend_x+26}" y="{ly-4}" font-family="Arial" font-size="12">{name}</text>')

    _write_svg(path, width, height, elems)


def _bar_chart(
    path: Path,
    title: str,
    categories: list[str],
    series: list[tuple[str, str, list[float]]],
    y_label: str,
) -> None:
    width, height = 920, 560
    m_left, m_right, m_top, m_bottom = 85, 26, 64, 78
    x0, x1 = m_left, width - m_right
    y0, y1 = height - m_bottom, m_top

    all_vals = [v for _, _, vals in series for v in vals if v is not None and not math.isnan(v)]
    y_lo = 0.0
    y_hi = max(all_vals) if all_vals else 1.0
    if y_hi <= 0.0:
        y_hi = 1.0
    y_hi *= 1.15

    elems = []
    elems.append(f'<text x="{width/2:.1f}" y="36" text-anchor="middle" font-family="Arial" font-size="22">{title}</text>')
    elems.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#111" stroke-width="1.3"/>')
    elems.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#111" stroke-width="1.3"/>')

    n_cat = len(categories)
    n_series = len(series)
    group_w = (x1 - x0) / max(1, n_cat)
    bar_w = group_w / (n_series + 1.6)

    for i in range(6):
        yv = y_lo + (y_hi - y_lo) * i / 5.0
        py = _map(yv, y_lo, y_hi, y0, y1)
        elems.append(f'<line x1="{x0}" y1="{py:.2f}" x2="{x1}" y2="{py:.2f}" stroke="#eee" stroke-width="1"/>')
        elems.append(f'<text x="{x0-10}" y="{py+4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{yv:.4f}</text>')

    for c_idx, cat in enumerate(categories):
        gx = x0 + c_idx * group_w
        elems.append(f'<text x="{gx+group_w/2:.2f}" y="{y0+24}" text-anchor="middle" font-family="Arial" font-size="12">{cat}</text>')
        for s_idx, (name, colour, vals) in enumerate(series):
            if c_idx >= len(vals):
                continue
            val = vals[c_idx]
            if val is None or math.isnan(val):
                continue
            left = gx + 0.8 * bar_w + s_idx * bar_w
            top = _map(val, y_lo, y_hi, y0, y1)
            elems.append(f'<rect x="{left:.2f}" y="{top:.2f}" width="{bar_w*0.9:.2f}" height="{(y0-top):.2f}" fill="{colour}"/>')

    elems.append(f'<text x="24" y="{(y0+y1)/2:.1f}" transform="rotate(-90 24 {(y0+y1)/2:.1f})" text-anchor="middle" font-family="Arial" font-size="14">{y_label}</text>')

    legend_x = x0 + 12
    legend_y = y1 + 10
    for idx, (name, colour, _) in enumerate(series):
        ly = legend_y + 22 * idx
        elems.append(f'<rect x="{legend_x}" y="{ly-10}" width="18" height="10" fill="{colour}"/>')
        elems.append(f'<text x="{legend_x+26}" y="{ly-2}" font-family="Arial" font-size="12">{name}</text>')

    _write_svg(path, width, height, elems)


def _histogram(
    path: Path,
    title: str,
    values: list[float],
    bins: int = 24,
    colour: str = "#4f8a8b",
) -> None:
    width, height = 920, 560
    m_left, m_right, m_top, m_bottom = 85, 26, 64, 78
    x0, x1 = m_left, width - m_right
    y0, y1 = height - m_bottom, m_top
    if not values:
        _write_svg(path, width, height, ['<text x="120" y="120">No values available</text>'])
        return

    vmin, vmax = _safe_min_max(values)
    step = (vmax - vmin) / bins
    counts = [0 for _ in range(bins)]
    for v in values:
        idx = int((v - vmin) / step) if step > 1e-14 else 0
        idx = max(0, min(bins - 1, idx))
        counts[idx] += 1
    max_count = max(counts) if counts else 1

    elems = []
    elems.append(f'<text x="{width/2:.1f}" y="36" text-anchor="middle" font-family="Arial" font-size="22">{title}</text>')
    elems.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#111" stroke-width="1.3"/>')
    elems.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#111" stroke-width="1.3"/>')

    for i in range(6):
        xv = vmin + (vmax - vmin) * i / 5.0
        px = _map(xv, vmin, vmax, x0, x1)
        elems.append(f'<line x1="{px:.2f}" y1="{y0}" x2="{px:.2f}" y2="{y1}" stroke="#eee" stroke-width="1"/>')
        elems.append(f'<text x="{px:.2f}" y="{y0+24}" text-anchor="middle" font-family="Arial" font-size="12">{xv:.3f}</text>')

    for i in range(6):
        cv = max_count * i / 5.0
        py = _map(cv, 0.0, max_count, y0, y1)
        elems.append(f'<line x1="{x0}" y1="{py:.2f}" x2="{x1}" y2="{py:.2f}" stroke="#eee" stroke-width="1"/>')
        elems.append(f'<text x="{x0-10}" y="{py+4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{cv:.0f}</text>')

    bar_w = (x1 - x0) / bins
    for i, c in enumerate(counts):
        left = x0 + i * bar_w
        top = _map(c, 0.0, max_count, y0, y1)
        elems.append(f'<rect x="{left:.2f}" y="{top:.2f}" width="{bar_w*0.9:.2f}" height="{(y0-top):.2f}" fill="{colour}" opacity="0.9"/>')

    _write_svg(path, width, height, elems)


def generate_visual_artifacts(
    surface_result: dict,
    model_result: dict,
    hedging_result: dict,
    rolling_result: dict,
    output_dir: str | Path,
) -> list[str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    # 1) Smile fit visual for first maturity.
    first_surface = surface_result["per_maturity"][0]
    maturity = first_surface["maturity"]
    strikes = [row["strike"] for row in first_surface["rows"]]
    market_ivs = [row["implied_vol"] for row in first_surface["rows"]]
    svi_params_raw = first_surface["svi_fit"]["params"]
    svi_params = SVIParams(
        a=svi_params_raw["a"],
        b=svi_params_raw["b"],
        rho=svi_params_raw["rho"],
        m=svi_params_raw["m"],
        sigma=svi_params_raw["sigma"],
    )
    forward = first_surface["forward"]
    svi_ivs = []
    for strike in strikes:
        log_m = math.log(strike / forward)
        total_var = max(1e-12, svi_total_variance(log_m, svi_params))
        svi_ivs.append(math.sqrt(total_var / maturity))

    sabr_block = None
    for block in model_result["per_maturity"]:
        if abs(block["maturity"] - maturity) < 1e-9:
            sabr_block = block
            break
    sabr_ivs = None
    if sabr_block is not None:
        p = sabr_block["sabr"]["params"]
        sabr_params = SABRParams(alpha=p["alpha"], beta=p["beta"], rho=p["rho"], nu=p["nu"])
        sabr_ivs = [sabr_implied_vol(forward, k, maturity, sabr_params) for k in strikes]

    series = [("Market IV", "#1f77b4", market_ivs), ("SVI fit", "#ff7f0e", svi_ivs)]
    if sabr_ivs is not None:
        series.append(("SABR fit", "#2ca02c", sabr_ivs))
    smile_path = out / "fig_smile_fit.svg"
    _line_chart(
        path=smile_path,
        title=f"Implied Volatility Smile Fit (T={maturity:.3f})",
        x_values=strikes,
        series=series,
        x_label="Strike",
        y_label="Implied volatility",
    )
    generated.append(str(smile_path))

    # 2) Model RMSE bar chart.
    summary = model_result["summary"]
    categories = ["In-sample", "Out-of-sample CV"]
    rmse_path = out / "fig_model_rmse.svg"
    _bar_chart(
        path=rmse_path,
        title="SVI vs SABR RMSE Comparison",
        categories=categories,
        series=[
            (
                "SVI",
                "#ff7f0e",
                [
                    summary["avg_in_sample_rmse_iv"]["svi"],
                    summary["avg_oos_cv_rmse_iv"]["svi"] if summary["avg_oos_cv_rmse_iv"]["svi"] is not None else float("nan"),
                ],
            ),
            (
                "SABR",
                "#2ca02c",
                [
                    summary["avg_in_sample_rmse_iv"]["sabr"],
                    summary["avg_oos_cv_rmse_iv"]["sabr"] if summary["avg_oos_cv_rmse_iv"]["sabr"] is not None else float("nan"),
                ],
            ),
        ],
        y_label="RMSE (IV)",
    )
    generated.append(str(rmse_path))

    # 3) Hedging terminal P&L histograms.
    terminals = hedging_result.get("terminal_pnl_by_strategy", {})
    if terminals:
        for strategy, vals in terminals.items():
            hist_path = out / f"fig_terminal_pnl_hist_{strategy}.svg"
            _histogram(
                path=hist_path,
                title=f"Terminal P&L Distribution ({strategy})",
                values=vals,
                bins=24,
                colour="#4f8a8b" if strategy == "unhedged" else "#5b8c5a",
            )
            generated.append(str(hist_path))

    # 4) Rolling calibration RMSE trend.
    rmse_by_date = rolling_result["calibration_quality"]["per_date_rmse"]
    if rmse_by_date:
        ordered_dates = sorted(rmse_by_date.keys())
        xs = list(range(len(ordered_dates)))
        ys = [rmse_by_date[d] for d in ordered_dates]
        rolling_path = out / "fig_rolling_rmse_trend.svg"
        _line_chart(
            path=rolling_path,
            title="Rolling Calibration RMSE Trend",
            x_values=xs,
            series=[("RMSE(w)", "#9467bd", ys)],
            x_label="Recalibration index",
            y_label="RMSE of total variance",
        )
        generated.append(str(rolling_path))

    return generated
