"""
Simple Quandl data fetcher

Usage:
    python api/quandl_fin.py [asset] [start_date] [end_date] [--timeframe <tf>] [--output <csv_path>] [--api-key <key>]

Defaults (if no positional args):
    asset=FRED/GDP, start=2024-01-01, end=2024-12-31

Example:
    python api/quandl_fin.py WIKI/AAPL 2020-01-01 2020-12-31 --timeframe daily --output api/cache/quandl/AAPL_2020.csv

Notes:
- Asset should be a valid Quandl code, e.g., "WIKI/AAPL" or "FRED/GDP"
- Timeframe is optional and depends on the dataset (e.g., daily, weekly, monthly)
- Requires the 'quandl' and 'pandas' packages
- You can set your API key via --api-key (recommended for quick tests)
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import quandl
except ImportError:
    raise ImportError("You must install the 'quandl' package: pip install quandl")

# Default values to allow running without positional args
DEFAULT_ASSET = "FRED/GDP"
DEFAULT_START = "2024-01-01"
DEFAULT_END = "2024-12-31"

def fetch_quandl_to_csv(
    asset: str,
    start: str,
    end: str,
    *,
    timeframe: Optional[str] = None,
    output: Optional[Path] = None,
    api_key: Optional[str] = None,
) -> Path:
    # Set API key if provided
    if api_key:
        quandl.ApiConfig.api_key = api_key
    elif "QUANDL_API_KEY" in os.environ:
        quandl.ApiConfig.api_key = os.environ["QUANDL_API_KEY"]

    # Map timeframe to collapse argument if provided
    collapse = None
    if timeframe:
        tf_map = {
            "daily": "daily",
            "weekly": "weekly",
            "monthly": "monthly",
            "quarterly": "quarterly",
            "annual": "annual",
            "yearly": "annual",
        }
        collapse = tf_map.get(timeframe.lower())
        if collapse is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")

    df = quandl.get(
        asset,
        start_date=start,
        end_date=end,
        collapse=collapse,
    )

    if df.empty:
        raise RuntimeError("No data returned from Quandl.")

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    asset_code = asset.replace("/", "_")
    out = output or (Path(__file__).parent / "cache" / "quandl" / f"{asset_code}_{start}_{end}{'_' + collapse if collapse else ''}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    return out

def main():
    parser = argparse.ArgumentParser(description="Download Quandl data to CSV.")
    parser.add_argument("asset", nargs="?", default=DEFAULT_ASSET, help=f"Quandl asset code (default: {DEFAULT_ASSET})")
    parser.add_argument("start", nargs="?", default=DEFAULT_START, help=f"Start date (YYYY-MM-DD, default: {DEFAULT_START})")
    parser.add_argument("end", nargs="?", default=DEFAULT_END, help=f"End date (YYYY-MM-DD, default: {DEFAULT_END})")
    parser.add_argument("--timeframe", help="Timeframe: daily, weekly, monthly, quarterly, annual", default=None)
    parser.add_argument("--output", help="Output CSV file path", default=None)
    parser.add_argument("--api-key", help="Quandl API key", default="EW6Y4WFzxxDDBEaxUQzH")

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None

    try:
        out = fetch_quandl_to_csv(
            args.asset,
            args.start,
            args.end,
            timeframe=args.timeframe,
            output=output_path,
            api_key=args.api_key,
        )
        print(f"Saved Quandl data to {out}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
