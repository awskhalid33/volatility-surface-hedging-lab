from dataclasses import dataclass


@dataclass(frozen=True)
class OptionQuote:
    valuation_date: str
    expiry: str
    maturity: float
    spot: float
    rate: float
    dividend: float
    strike: float
    call_mid: float


@dataclass(frozen=True)
class QuoteWithIV:
    quote: OptionQuote
    implied_vol: float
    forward: float
    log_moneyness: float
    total_variance: float
