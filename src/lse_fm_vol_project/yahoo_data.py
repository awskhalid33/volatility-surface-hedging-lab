import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, timezone


YAHOO_OPTIONS_URL = "https://query2.finance.yahoo.com/v7/finance/options/{ticker}"


def _safe_mid(bid: float, ask: float, last: float) -> float | None:
    if bid > 0.0 and ask > 0.0:
        return 0.5 * (bid + ask)
    if last > 0.0:
        return last
    if bid > 0.0:
        return bid
    if ask > 0.0:
        return ask
    return None


def _fetch_json(url: str, timeout_sec: float = 20.0) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _extract_result(payload: dict) -> dict:
    chain = payload.get("optionChain", {})
    results = chain.get("result", [])
    if not results:
        raise RuntimeError("Yahoo response had no optionChain.result payload")
    return results[0]


def _extract_rows_from_result(result: dict, ticker: str) -> tuple[list[dict[str, str]], list[int]]:
    quote = result.get("quote", {})
    options = result.get("options", [])
    if not options:
        raise RuntimeError("Yahoo response has no options list")

    calls = options[0].get("calls", [])
    valuation_epoch = int(quote.get("regularMarketTime") or datetime.now(timezone.utc).timestamp())
    valuation_dt = datetime.fromtimestamp(valuation_epoch, tz=timezone.utc).date()
    spot = float(quote.get("regularMarketPrice") or 0.0)
    if spot <= 0.0:
        raise RuntimeError("Yahoo response did not provide positive regularMarketPrice")

    rows = []
    for call in calls:
        strike = float(call.get("strike") or 0.0)
        bid = float(call.get("bid") or 0.0)
        ask = float(call.get("ask") or 0.0)
        last = float(call.get("lastPrice") or 0.0)
        expiry_epoch = int(call.get("expiration") or 0)
        if strike <= 0.0 or expiry_epoch <= 0:
            continue
        call_mid = _safe_mid(bid=bid, ask=ask, last=last)
        if call_mid is None or call_mid <= 0.0:
            continue
        expiry_dt = datetime.fromtimestamp(expiry_epoch, tz=timezone.utc).date()
        maturity_days = (expiry_dt - valuation_dt).days
        if maturity_days <= 7:
            continue
        maturity = maturity_days / 365.0
        rows.append(
            {
                "valuation_date": valuation_dt.isoformat(),
                "expiry": expiry_dt.isoformat(),
                "maturity": f"{maturity:.10f}",
                "spot": f"{spot:.8f}",
                "rate": f"{0.0:.8f}",
                "dividend": f"{0.0:.8f}",
                "strike": f"{strike:.6f}",
                "call_mid": f"{call_mid:.10f}",
            }
        )

    expiration_dates = [int(x) for x in result.get("expirationDates", [])]
    return rows, expiration_dates


def fetch_yahoo_option_chain_rows(
    ticker: str,
    max_expiries: int = 4,
    timeout_sec: float = 20.0,
) -> list[dict[str, str]]:
    if not ticker:
        raise ValueError("ticker cannot be empty")
    if max_expiries <= 0:
        raise ValueError("max_expiries must be positive")

    encoded_ticker = urllib.parse.quote(ticker.upper())
    base_url = YAHOO_OPTIONS_URL.format(ticker=encoded_ticker)

    first_payload = _fetch_json(base_url, timeout_sec=timeout_sec)
    first_result = _extract_result(first_payload)
    first_rows, expiration_dates = _extract_rows_from_result(first_result, ticker)
    rows = list(first_rows)

    for expiry_epoch in expiration_dates[: max(0, max_expiries - 1)]:
        url = f"{base_url}?date={expiry_epoch}"
        payload = _fetch_json(url, timeout_sec=timeout_sec)
        result = _extract_result(payload)
        more_rows, _ = _extract_rows_from_result(result, ticker)
        rows.extend(more_rows)

    dedup = {}
    for r in rows:
        key = (r["valuation_date"], r["expiry"], r["strike"])
        dedup[key] = r
    final_rows = list(dedup.values())
    final_rows.sort(key=lambda r: (r["valuation_date"], r["expiry"], float(r["strike"])))
    if not final_rows:
        raise RuntimeError("No valid option rows were extracted from Yahoo payload")
    return final_rows


def infer_flat_rate_from_spot_forward(
    spot: float,
    forward: float,
    maturity: float,
) -> float:
    if spot <= 0.0 or forward <= 0.0 or maturity <= 0.0:
        raise ValueError("spot, forward, maturity must be positive")
    return math.log(forward / spot) / maturity
