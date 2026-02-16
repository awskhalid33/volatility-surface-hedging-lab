from lse_fm_vol_project.yahoo_data import _extract_rows_from_result


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
