import csv
import json
from pathlib import Path

from .types import OptionQuote

REQUIRED_COLUMNS = {
    "valuation_date",
    "expiry",
    "maturity",
    "spot",
    "rate",
    "dividend",
    "strike",
    "call_mid",
}


def load_option_quotes(csv_path: str | Path) -> list[OptionQuote]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    quotes: list[OptionQuote] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        for row in reader:
            quotes.append(
                OptionQuote(
                    valuation_date=row["valuation_date"],
                    expiry=row["expiry"],
                    maturity=float(row["maturity"]),
                    spot=float(row["spot"]),
                    rate=float(row["rate"]),
                    dividend=float(row["dividend"]),
                    strike=float(row["strike"]),
                    call_mid=float(row["call_mid"]),
                )
            )

    if not quotes:
        raise ValueError("CSV contained zero option rows")
    return quotes


def write_json(payload: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
