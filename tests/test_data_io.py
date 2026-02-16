import pytest

from vol_surface_hedging_lab.data_io import load_option_quotes


def test_load_option_quotes_raises_on_missing_columns(tmp_path):
    csv_path = tmp_path / "missing_columns.csv"
    csv_path.write_text(
        "valuation_date,expiry,maturity,spot,rate,dividend,strike\n"
        "2026-02-15,2026-03-15,0.0821917808,100,0.02,0.0,100\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing columns"):
        load_option_quotes(csv_path)


def test_load_option_quotes_raises_on_empty_rows(tmp_path):
    csv_path = tmp_path / "header_only.csv"
    csv_path.write_text(
        "valuation_date,expiry,maturity,spot,rate,dividend,strike,call_mid\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="zero option rows"):
        load_option_quotes(csv_path)
