import ssl
import urllib.error

import pytest

from vol_surface_hedging_lab import yahoo_data
from vol_surface_hedging_lab.yahoo_data import _extract_rows_from_result, _fetch_json


def test_extract_rows_from_result_parses_calls_and_filters_invalid():
    result = {
        "quote": {
            "regularMarketTime": 1736985600,  # 2025-01-16 UTC
            "regularMarketPrice": 100.0,
        },
        "expirationDates": [1741824000],
        "options": [
            {
                "calls": [
                    {
                        "strike": 95.0,
                        "bid": 6.0,
                        "ask": 6.4,
                        "lastPrice": 6.1,
                        "expiration": 1741824000,  # 2025-03-13 UTC
                    },
                    {
                        "strike": 105.0,
                        "bid": 0.0,
                        "ask": 0.0,
                        "lastPrice": 2.2,
                        "expiration": 1741824000,
                    },
                    {
                        "strike": 0.0,
                        "bid": 1.0,
                        "ask": 1.2,
                        "lastPrice": 1.1,
                        "expiration": 1741824000,
                    },
                ]
            }
        ],
    }
    rows, expiries = _extract_rows_from_result(result, "SPY")
    assert len(rows) == 2
    assert expiries == [1741824000]
    assert rows[0]["valuation_date"] == "2025-01-16"
    assert rows[0]["expiry"] == "2025-03-13"
    assert float(rows[0]["spot"]) == 100.0
    assert float(rows[0]["call_mid"]) > 0.0


def test_fetch_json_retries_on_ssl_cert_error(monkeypatch):
    class _Response:
        def __init__(self, payload: str):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return self._payload.encode("utf-8")

    calls = {"count": 0}

    def fake_urlopen(req, timeout=0.0, context=None):  # noqa: ARG001
        calls["count"] += 1
        if calls["count"] == 1:
            raise urllib.error.URLError(ssl.SSLCertVerificationError("cert failed"))
        assert context is not None
        return _Response('{"ok": true}')

    monkeypatch.setattr(yahoo_data.urllib.request, "urlopen", fake_urlopen)
    payload = _fetch_json("https://example.com")
    assert payload == {"ok": True}
    assert calls["count"] == 2


def test_fetch_json_raises_if_insecure_fallback_disabled(monkeypatch):
    def fake_urlopen(req, timeout=0.0, context=None):  # noqa: ARG001
        raise urllib.error.URLError(ssl.SSLCertVerificationError("cert failed"))

    monkeypatch.setattr(yahoo_data.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.URLError):
        _fetch_json("https://example.com", allow_insecure_ssl_fallback=False)
