
"""
market_forecast.py  (FREE-tier friendly)

Alpha Vantage helpers + lightweight forecasting
- Primary: TIME_SERIES_DAILY (free)
- Optional fallback: WEEKLY or MONTHLY (free)
- Avoid TIME_SERIES_DAILY_ADJUSTED by default (often premium in 2025)
- Statsmodels Exponential Smoothing forecaster

Install:
  pip install requests pandas numpy statsmodels plotly
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import requests
import pandas as pd
import numpy as np

# Forecasting
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:
    ExponentialSmoothing = None  # type: ignore


def _normalize_df_from_series(ts: dict, granularity: str) -> pd.DataFrame:
    rows = []
    # Keys differ: Daily -> "Time Series (Daily)", Weekly -> "Weekly Time Series", Monthly -> "Monthly Time Series"
    for d, v in ts.items():
        # Field names differ across endpoints; guard with get
        rows.append({
            "date": pd.to_datetime(d),
            "open": float(v.get("1. open") or v.get("1. Open")),
            "high": float(v.get("2. high") or v.get("2. High")),
            "low": float(v.get("3. low") or v.get("3. Low")),
            "close": float(v.get("4. close") or v.get("4. Close")),
            "volume": float(v.get("5. volume") or v.get("5. Volume") or 0.0),
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["adjusted_close"] = df["close"]  # for non-adjusted endpoints
    df["granularity"] = granularity
    return df


def fetch_alpha_vantage_free(symbol: str, api_key: str, granularity: str = "daily", outputsize: str = "compact") -> pd.DataFrame:
    """
    FREE-tier friendly fetcher. Uses non-adjusted endpoints.
      granularity: "daily" | "weekly" | "monthly"
      outputsize: "compact" | "full"  (daily only)
    Returns standardized columns: [date, open, high, low, close, adjusted_close, volume, granularity]
    """
    url = "https://www.alphavantage.co/query"
    if granularity == "daily":
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": api_key,
            "outputsize": outputsize,
        }
        key_name = "Time Series (Daily)"
    elif granularity == "weekly":
        params = {
            "function": "TIME_SERIES_WEEKLY",
            "symbol": symbol,
            "apikey": api_key,
        }
        key_name = "Weekly Time Series"
    elif granularity == "monthly":
        params = {
            "function": "TIME_SERIES_MONTHLY",
            "symbol": symbol,
            "apikey": api_key,
        }
        key_name = "Monthly Time Series"
    else:
        raise ValueError("granularity must be 'daily', 'weekly', or 'monthly'")

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if "Error Message" in data:
        raise ValueError(data["Error Message"])
    if "Note" in data:
        # Rate limited; surface the message
        raise RuntimeError(data["Note"])
    if "Information" in data:
        # Premium endpoint or other info message
        raise RuntimeError(data["Information"])
    if key_name not in data:
        raise RuntimeError(f"Alpha Vantage response missing '{key_name}'. Raw keys: {list(data.keys())[:5]}")

    ts = data[key_name]
    df = _normalize_df_from_series(ts, granularity=granularity)
    return df


HORIZON_PAT = re.compile(r"(\d+)\s*(day|days|week|weeks|month|months|quarter|quarters|year|years)", re.I)

def parse_horizon(prompt: str, default_days: int = 20) -> int:
    """
    Parse natural language like 'forecast next 6 weeks' -> 30 business days.
    """
    if not prompt:
        return default_days
    m = HORIZON_PAT.search(prompt)
    if not m:
        return default_days
    n = int(m.group(1))
    unit = m.group(2).lower()
    if "day" in unit:
        return n
    if "week" in unit:
        return n * 5
    if "month" in unit:
        return n * 21
    if "quarter" in unit:
        return n * 63
    if "year" in unit:
        return n * 252
    return default_days


def exp_smoothing_forecast(df_prices: pd.DataFrame, value_col: str = "adjusted_close", periods: int = 20,
                           seasonal: Optional[str] = None, seasonal_periods: Optional[int] = None) -> pd.DataFrame:
    """
    Fit Exponential Smoothing with optional trend and seasonal components.
    Returns DataFrame with columns: date, yhat, yhat_lower, yhat_upper
    """
    if ExponentialSmoothing is None:
        raise ImportError("statsmodels is required. pip install statsmodels")

    s = df_prices.set_index("date")[value_col].asfreq("B")  # business days
    s = s.interpolate(limit_direction="both")  # fill any gaps

    model = ExponentialSmoothing(
        s,
        trend="add",
        seasonal=seasonal,                 # None by default
        seasonal_periods=seasonal_periods  # e.g., 5 for weekly pattern on business days
    )
    fit = model.fit(optimized=True, use_brute=True)

    resid = fit.resid.dropna()
    sigma = float(resid.std(ddof=1)) if len(resid) > 1 else 0.0
    z = 1.96

    future_idx = pd.bdate_range(start=s.index[-1] + pd.offsets.BDay(), periods=periods)
    yhat = fit.forecast(periods)
    yhat.index = future_idx

    out = pd.DataFrame({
        "date": yhat.index,
        "yhat": yhat.values,
        "yhat_lower": yhat.values - z * sigma,
        "yhat_upper": yhat.values + z * sigma,
    })
    return out
