import math

from .svi import SVIParams, svi_total_variance


class SVISurfaceModel:
    def __init__(self, maturity_to_params: dict[float, SVIParams]):
        if not maturity_to_params:
            raise ValueError("maturity_to_params cannot be empty")
        self._maturity_to_params = dict(sorted(maturity_to_params.items(), key=lambda x: x[0]))
        self._maturities = list(self._maturity_to_params.keys())

    @classmethod
    def from_pipeline_result(cls, result: dict) -> "SVISurfaceModel":
        maturity_to_params: dict[float, SVIParams] = {}
        for block in result.get("per_maturity", []):
            t = float(block["maturity"])
            p = block["svi_fit"]["params"]
            maturity_to_params[t] = SVIParams(
                a=float(p["a"]),
                b=float(p["b"]),
                rho=float(p["rho"]),
                m=float(p["m"]),
                sigma=float(p["sigma"]),
            )
        return cls(maturity_to_params)

    def _total_variance_for_maturity(self, maturity: float, log_moneyness: float) -> float:
        params = self._maturity_to_params[maturity]
        return max(1e-12, svi_total_variance(log_moneyness, params))

    def total_variance(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend: float,
    ) -> float:
        if spot <= 0.0:
            raise ValueError("spot must be positive")
        if strike <= 0.0:
            raise ValueError("strike must be positive")
        if maturity <= 0.0:
            raise ValueError("maturity must be positive")

        forward = spot * math.exp((rate - dividend) * maturity)
        log_m = math.log(strike / forward)
        maturities = self._maturities

        if maturity <= maturities[0]:
            t0 = maturities[0]
            w0 = self._total_variance_for_maturity(t0, log_m)
            return max(1e-12, w0 * maturity / t0)

        if maturity >= maturities[-1]:
            t1 = maturities[-1]
            w1 = self._total_variance_for_maturity(t1, log_m)
            if len(maturities) == 1:
                return max(1e-12, w1 * maturity / t1)
            t0 = maturities[-2]
            w0 = self._total_variance_for_maturity(t0, log_m)
            slope = (w1 - w0) / (t1 - t0)
            extrapolated = w1 + slope * (maturity - t1)
            if extrapolated <= 1e-12:
                return max(1e-12, w1 * maturity / t1)
            return extrapolated

        for idx in range(1, len(maturities)):
            left = maturities[idx - 1]
            right = maturities[idx]
            if left <= maturity <= right:
                w_left = self._total_variance_for_maturity(left, log_m)
                w_right = self._total_variance_for_maturity(right, log_m)
                alpha = (maturity - left) / (right - left)
                return max(1e-12, (1.0 - alpha) * w_left + alpha * w_right)

        raise RuntimeError("maturity interpolation failed unexpectedly")

    def implied_volatility(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend: float,
    ) -> float:
        total_var = self.total_variance(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            dividend=dividend,
        )
        return math.sqrt(total_var / maturity)
