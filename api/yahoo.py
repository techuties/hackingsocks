import os
from typing import Any, Dict, Tuple

import yfinance as yf
import pandas as pd

try:
    # Prefer relative when used as a package
    from .csv_cache import CsvCache
except Exception:  # fallback for direct script execution
    try:
        from api.csv_cache import CsvCache  # type: ignore
    except Exception:
        from csv_cache import CsvCache  # type: ignore

asset = os.environ.get("YF_ASSET", "MSFT")
start_date = os.environ.get("YF_START", "2024-01-01")
end_date = os.environ.get("YF_END", "2024-12-31")
difference = pd.to_datetime(end_date) - pd.to_datetime(start_date)
print(f'Difference: {difference.days}d')

dat = yf.Ticker(asset)

# Cache setup
CACHE_TTL_SECONDS = int(os.environ.get("YF_CACHE_TTL", "86400"))  # default 24h
cache = CsvCache()  # defaults to api/cache/


def _df_to_json(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "__type__": "dataframe",
        "orient": "split",
        "payload": df.to_dict(orient="split"),
    }


def _json_to_df(obj: Dict[str, Any]) -> pd.DataFrame:
    data = obj.get("payload", {})
    return pd.DataFrame(data=data.get("data", []), index=data.get("index", []), columns=data.get("columns", []))


def cached_fetch(table: str, source: str, params: Dict[str, Any], fetch_fn, is_dataframe: bool) -> Tuple[Any, bool]:
    def _fetch_wrapped():
        value = fetch_fn()
        if is_dataframe and isinstance(value, pd.DataFrame):
            return _df_to_json(value)
        return value

    data, from_cache = cache.get_or_fetch(
        table_name=f"yahoo/{table}",
        source=source,
        params=params,
        fetch_fn=_fetch_wrapped,
        ttl_seconds=CACHE_TTL_SECONDS,
    )
    if is_dataframe and isinstance(data, dict) and data.get("__type__") == "dataframe":
        return _json_to_df(data), from_cache
    return data, from_cache

## Overall
info, _ = cached_fetch("overall", "info", {"asset": asset}, lambda: dat.info, is_dataframe=False)
print(f'dat.info: {info}')

calendar_df, _ = cached_fetch("overall", "calendar", {"asset": asset}, lambda: dat.calendar, is_dataframe=True)
print(f'dat.calendar: {calendar_df}')

## Financials (cached as DataFrames)
income_stmt_df, _ = cached_fetch("financials", "income_stmt", {"asset": asset}, lambda: dat.income_stmt, is_dataframe=True)
print(f'Income Statement: {income_stmt_df}')

quarterly_income_stmt_df, _ = cached_fetch("financials", "quarterly_income_stmt", {"asset": asset}, lambda: dat.quarterly_income_stmt, is_dataframe=True)
print(f'Quarterly Income Statement: {quarterly_income_stmt_df}')

financials_df, _ = cached_fetch("financials", "financials", {"asset": asset}, lambda: dat.financials, is_dataframe=True)
print(f'Financials: {financials_df}')

quarterly_financials_df, _ = cached_fetch("financials", "quarterly_financials", {"asset": asset}, lambda: dat.quarterly_financials, is_dataframe=True)
print(f'Quarterly Financials: {quarterly_financials_df}')

balance_sheet_df, _ = cached_fetch("financials", "balance_sheet", {"asset": asset}, lambda: dat.balance_sheet, is_dataframe=True)
print(f'Balance Sheet: {balance_sheet_df}')

quarterly_balance_sheet_df, _ = cached_fetch("financials", "quarterly_balance_sheet", {"asset": asset}, lambda: dat.quarterly_balance_sheet, is_dataframe=True)
print(f'Quarterly Balance Sheet: {quarterly_balance_sheet_df}')

cashflow_df, _ = cached_fetch("financials", "cashflow", {"asset": asset}, lambda: dat.cashflow, is_dataframe=True)
print(f'Cash Flow: {cashflow_df}')

quarterly_cashflow_df, _ = cached_fetch("financials", "quarterly_cashflow", {"asset": asset}, lambda: dat.quarterly_cashflow, is_dataframe=True)
print(f'Quarterly Cash Flow: {quarterly_cashflow_df}')

earnings_df, _ = cached_fetch("financials", "earnings", {"asset": asset}, lambda: dat.earnings, is_dataframe=True)
print(f'Earnings: {earnings_df}')

quarterly_earnings_df, _ = cached_fetch("financials", "quarterly_earnings", {"asset": asset}, lambda: dat.quarterly_earnings, is_dataframe=True)
print(f'Quarterly Earnings: {quarterly_earnings_df}')

## Other Financials (cached as DataFrames)
analyst_price_targets_df, _ = cached_fetch("other_financials", "analyst_price_targets", {"asset": asset}, lambda: dat.analyst_price_targets, is_dataframe=True)
print(f'analyst_price_targets: {analyst_price_targets_df}')

institutional_holders_df, _ = cached_fetch("other_financials", "institutional_holders", {"asset": asset}, lambda: dat.institutional_holders, is_dataframe=True)
print(f'Institutional Holders: {institutional_holders_df}')

major_holders_df, _ = cached_fetch("other_financials", "major_holders", {"asset": asset}, lambda: dat.major_holders, is_dataframe=True)
print(f'Major Holders: {major_holders_df}')

insider_transactions_df, _ = cached_fetch("other_financials", "insider_transactions", {"asset": asset}, lambda: dat.insider_transactions, is_dataframe=True)
print(f'Insider Transactions: {insider_transactions_df}')

## Other (cached as DataFrames)
sustainability_df, _ = cached_fetch("other", "sustainability", {"asset": asset}, lambda: dat.sustainability, is_dataframe=True)
print(f'Sustainability: {sustainability_df}')

recommendations_df, _ = cached_fetch("other", "recommendations", {"asset": asset}, lambda: dat.recommendations, is_dataframe=True)
print(f'Recommendations: {recommendations_df}')

#print(f"dat.history(period=\'1mo\'): {dat.history(period='1mo')}")
#print(f'dat.option_chain(dat.options[0]).calls: {dat.option_chain(dat.options[0]).calls}')

