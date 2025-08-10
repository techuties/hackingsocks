import yfinance as yf
import pandas as pd
import numpy as np
from typing import Any, Dict, Tuple, Union
from datetime import date, datetime

# Use CsvCache to persist results under api/cache/
try:
    from .csv_cache import CsvCache
except Exception:
    try:
        from api.csv_cache import CsvCache  # type: ignore
    except Exception:
        from csv_cache import CsvCache  # type: ignore

asset = "MSFT"
start_date = "2024-01-01"
end_date = "2024-12-31"
cache_ttl_seconds = 24 * 60 * 60  # 24h
difference = pd.to_datetime(end_date) - pd.to_datetime(start_date)
print(f'Difference: {difference.days}d')

dat = yf.Ticker(asset)
cache = CsvCache()  # defaults to api/cache/


def _df_to_json(df: pd.DataFrame) -> Dict[str, Any]:
    payload = df.to_dict(orient="split")
    # Ensure payload is fully JSON-serializable (timestamps, numpy types, etc.)
    payload = _make_json_safe(payload)
    return {
        "__type__": "dataframe",
        "orient": "split",
        "payload": payload,
    }


def _json_to_df(obj: Dict[str, Any]) -> pd.DataFrame:
    data = obj.get("payload", {})
    return pd.DataFrame(
        data=data.get("data", []),
        index=data.get("index", []),
        columns=data.get("columns", []),
    )


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    # pandas containers
    if isinstance(value, pd.Series):
        return _make_json_safe(value.to_dict())
    # numpy containers
    if isinstance(value, np.ndarray):
        return [_make_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    # datetime-like
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    try:
        # numpy datetime64
        if isinstance(value, np.datetime64):
            return pd.Timestamp(value).isoformat()
    except Exception:
        pass
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta,)):
        return value.isoformat()
    return value


def cached_fetch(
    table: str,
    source: str,
    params: Dict[str, Any],
    fetch_fn,
    returns_dataframe: bool,
) -> Tuple[Union[pd.DataFrame, Dict[str, Any], list, None], bool]:
    def _fetch_wrapped():
        v = fetch_fn()
        if returns_dataframe and isinstance(v, pd.DataFrame):
            return _df_to_json(v)
        return _make_json_safe(v)

    data, from_cache = cache.get_or_fetch(
        table_name=f"yahoo/{asset}_{table}",
        source=source,
        params=params,
        fetch_fn=_fetch_wrapped,
        ttl_seconds=cache_ttl_seconds,
    )
    if returns_dataframe and isinstance(data, dict) and data.get("__type__") == "dataframe":
        return _json_to_df(data), from_cache
    return data, from_cache

## Overall
info, _ = cached_fetch("overall", "info", {"asset": asset}, lambda: dat.info, returns_dataframe=False)
print(f'dat.info: {info}')

calendar_df, _ = cached_fetch("overall", "calendar", {"asset": asset}, lambda: dat.calendar, returns_dataframe=True)
print(f'dat.calendar: {calendar_df}')

history_df, _ = cached_fetch(
    "overall",
    "history",
    {"asset": asset, "period_days": difference.days},
    lambda: dat.history(period=f"{difference.days}d"),
    returns_dataframe=True,
)
print(f"dat.history(period='{difference.days}d'): {history_df}")

## Financials (cached)
income_stmt_df, _ = cached_fetch("financials", "income_stmt", {"asset": asset}, lambda: dat.income_stmt, returns_dataframe=True)
print(f'Income Statement: {income_stmt_df}')

quarterly_income_stmt_df, _ = cached_fetch("financials", "quarterly_income_stmt", {"asset": asset}, lambda: dat.quarterly_income_stmt, returns_dataframe=True)
print(f'Quarterly Income Statement: {quarterly_income_stmt_df}')

financials_df, _ = cached_fetch("financials", "financials", {"asset": asset}, lambda: dat.financials, returns_dataframe=True)
print(f'Financials: {financials_df}')

quarterly_financials_df, _ = cached_fetch("financials", "quarterly_financials", {"asset": asset}, lambda: dat.quarterly_financials, returns_dataframe=True)
print(f'Quarterly Financials: {quarterly_financials_df}')

balance_sheet_df, _ = cached_fetch("financials", "balance_sheet", {"asset": asset}, lambda: dat.balance_sheet, returns_dataframe=True)
print(f'Balance Sheet: {balance_sheet_df}')

quarterly_balance_sheet_df, _ = cached_fetch("financials", "quarterly_balance_sheet", {"asset": asset}, lambda: dat.quarterly_balance_sheet, returns_dataframe=True)
print(f'Quarterly Balance Sheet: {quarterly_balance_sheet_df}')

cashflow_df, _ = cached_fetch("financials", "cashflow", {"asset": asset}, lambda: dat.cashflow, returns_dataframe=True)
print(f'Cash Flow: {cashflow_df}')

quarterly_cashflow_df, _ = cached_fetch("financials", "quarterly_cashflow", {"asset": asset}, lambda: dat.quarterly_cashflow, returns_dataframe=True)
print(f'Quarterly Cash Flow: {quarterly_cashflow_df}')

earnings_df, _ = cached_fetch("financials", "earnings", {"asset": asset}, lambda: dat.earnings, returns_dataframe=True)
print(f'Earnings: {earnings_df}')

quarterly_earnings_df, _ = cached_fetch("financials", "quarterly_earnings", {"asset": asset}, lambda: dat.quarterly_earnings, returns_dataframe=True)
print(f'Quarterly Earnings: {quarterly_earnings_df}')

## Other Financials (cached)
analyst_price_targets_df, _ = cached_fetch("other_financials", "analyst_price_targets", {"asset": asset}, lambda: dat.analyst_price_targets, returns_dataframe=True)
print(f'dat.analyst_price_targets: {analyst_price_targets_df}')

institutional_holders_df, _ = cached_fetch("other_financials", "institutional_holders", {"asset": asset}, lambda: dat.institutional_holders, returns_dataframe=True)
print(f'Institutional Holders: {institutional_holders_df}')

major_holders_df, _ = cached_fetch("other_financials", "major_holders", {"asset": asset}, lambda: dat.major_holders, returns_dataframe=True)
print(f'Major Holders: {major_holders_df}')

insider_transactions_df, _ = cached_fetch("other_financials", "insider_transactions", {"asset": asset}, lambda: dat.insider_transactions, returns_dataframe=True)
print(f'Insider Transactions: {insider_transactions_df}')

## Other (cached)
sustainability_df, _ = cached_fetch("other", "sustainability", {"asset": asset}, lambda: dat.sustainability, returns_dataframe=True)
print(f'Sustainability: {sustainability_df}')

recommendations_df, _ = cached_fetch("other", "recommendations", {"asset": asset}, lambda: dat.recommendations, returns_dataframe=True)
print(f'Recommendations: {recommendations_df}')

# Options list and first option chain
options_list, _ = cached_fetch("options", "options_list", {"asset": asset}, lambda: list(dat.options), returns_dataframe=False)
print(f"dat.options: {options_list}")
if options_list:
    first_exp = options_list[0]
    option_calls_df, _ = cached_fetch(
        "options",
        "option_chain_calls",
        {"asset": asset, "date": first_exp},
        lambda: dat.option_chain(first_exp).calls,
        returns_dataframe=True,
    )
    print(f'dat.option_chain({first_exp}).calls: {option_calls_df}')

# News (list of dicts)
news, _ = cached_fetch("news", "get_news", {"asset": asset}, lambda: dat.get_news(), returns_dataframe=False)
print(f'get_news(): {news}')