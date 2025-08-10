"""
Simple Yahoo Finance fetcher

What it does
- Downloads OHLCV price history for a given ticker and date range
- Saves the data as CSV (auto-creates `api/cache/yahoo/` by default)

Quick usage
- Defaults (no args): uses AAPL, 2024-01-01..2024-12-31, interval 1d
  python api/yahoo_fin.py

- Specify ticker and date range:
  python api/yahoo_fin.py AAPL 2024-01-01 2024-06-30 --interval 60m

- Save to a custom path:
  python api/yahoo_fin.py MSFT 2024-01-01 2024-03-01 --interval 1d --output api/cache/yahoo/MSFT_Q1_1d.csv

- If your network blocks Yahoo, use a proxy:
  python api/yahoo_fin.py KO 2024-01-01 2024-12-31 --interval 1d --proxy http://user:pass@host:port

Notes
- Supported intervals include: 1d, 1wk, 1mo, 60m (and others supported by yfinance)
- Use --no-auto-adjust to disable split/dividend adjustments
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

# Defaults used when running without CLI arguments
DEFAULT_TICKER = "AAPL"
DEFAULT_START = "2024-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_INTERVAL = "1d"


def fetch_history_to_csv(
    ticker: str,
    start: str,
    end: str,
    *,
    interval: str = "1d",
    auto_adjust: bool = True,
    output: Optional[Path] = None,
    proxy: Optional[str] = None,
) -> Path:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        proxy=proxy,
    )
    if df.empty:
        raise RuntimeError("No data returned from Yahoo Finance.")

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    out = output or (Path(__file__).parent / "cache" / "yahoo" / f"{ticker}_{start}_{end}_{interval}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple Yahoo Finance fetch: save OHLCV to CSV")
    p.add_argument("ticker", nargs="?", default=DEFAULT_TICKER, help="Ticker symbol, e.g., KO")
    p.add_argument("start", nargs="?", default=DEFAULT_START, help="Start date YYYY-MM-DD")
    p.add_argument("end", nargs="?", default=DEFAULT_END, help="End date YYYY-MM-DD")
    p.add_argument("--interval", default=DEFAULT_INTERVAL, help="Interval, e.g., 1d, 1wk, 1mo, 60m")
    p.add_argument("--no-auto-adjust", action="store_true", help="Disable price auto-adjust")
    p.add_argument("--output", help="Output CSV path")
    p.add_argument("--proxy", help="HTTP(S) proxy URL, e.g. http://user:pass@host:port")
    return p.parse_args()


def main() -> None:
    # If called with no CLI args, use defaults; otherwise parse provided args
    if len(sys.argv) == 1:
        ticker = DEFAULT_TICKER
        start = DEFAULT_START
        end = DEFAULT_END
        interval = DEFAULT_INTERVAL
        output = Path(__file__).parent / "cache" / "yahoo" / f"{ticker}_{start}_{end}_{interval}.csv"
        out = fetch_history_to_csv(
            ticker,
            start,
            end,
            interval=interval,
            auto_adjust=True,
            output=output,
            proxy=None,
        )
        print(f"Saved to: {out}")
        return

    args = parse_args()
    out = fetch_history_to_csv(
        args.ticker,
        args.start,
        args.end,
        interval=args.interval,
        auto_adjust=not args.no_auto_adjust,
        output=Path(args.output) if args.output else None,
        proxy=args.proxy,
    )
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
