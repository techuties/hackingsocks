"""
Alpha Vantage API helper for fundamental and news endpoints.

Implements:
- News & Sentiments Trending
- Company Overview Trending
- ETF Profile & Holdings
- Corporate Action - Dividends
- Corporate Action - Splits
- Income Statement
- Balance Sheet
- Cash Flow
- Earnings History
- Earnings Estimates Trending
- Listing & Delisting Status
- Earnings Calendar
- IPO Calendar

Docs: https://www.alphavantage.co/documentation/
"""

from typing import Optional, Dict, Any, List
import os
import requests
try:
    # Prefer relative import when used as a package (api.alphavantag)
    from .csv_cache import CsvCache
except Exception:  # fallback for running as a script from repo root
    try:
        from api.csv_cache import CsvCache  # type: ignore
    except Exception:
        from csv_cache import CsvCache  # type: ignore

BASE_URL = "https://www.alphavantage.co/query"

class AlphaVantageClient:
    def __init__(
        self,
        api_key: str,
        timeout: float = 15.0,
        enable_cache: bool = True,
        cache_ttl_seconds: Optional[int] = 24 * 60 * 60,
        cache_table_name: str = "alphavantage",
        cache_directory: Optional[str] = None,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.enable_cache = enable_cache
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_table_name = cache_table_name
        self.cache = CsvCache(cache_directory=cache_directory) if enable_cache else None

    def _get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Do not mutate caller's params
        cache_params: Dict[str, Any] = dict(params)
        source = cache_params.get("function", "GENERIC")

        def fetch() -> Dict[str, Any]:
            req_params: Dict[str, Any] = dict(cache_params)
            req_params["apikey"] = self.api_key
            resp = requests.get(BASE_URL, params=req_params, timeout=self.timeout)
            resp.raise_for_status()
            try:
                return resp.json()
            except Exception:
                # Return a structured error without caching broken content
                text_preview = resp.text[:500] if hasattr(resp, "text") else ""
                raise RuntimeError(
                    f"AlphaVantage non-JSON response (status={resp.status_code}): {text_preview}"
                )

        if self.enable_cache and self.cache is not None:
            try:
                data, _ = self.cache.get_or_fetch(
                    table_name=self.cache_table_name,
                    source=source,
                    params=cache_params,
                    fetch_fn=fetch,
                    ttl_seconds=self.cache_ttl_seconds,
                )
                return data
            except Exception as exc:
                return {"error": str(exc)}
        else:
            try:
                return fetch()
            except Exception as exc:
                return {"error": str(exc)}

    def get_news_sentiment(self, tickers: Optional[List[str]] = None, topics: Optional[List[str]] = None, time_from: Optional[str] = None, time_to: Optional[str] = None) -> Dict[str, Any]:
        """
        News & Sentiments Trending
        Docs: https://www.alphavantage.co/documentation/#news
        """
        params = {
            "function": "NEWS_SENTIMENT"
        }
        if tickers:
            params["tickers"] = ",".join(tickers)
        if topics:
            params["topics"] = ",".join(topics)
        if time_from:
            params["time_from"] = time_from
        if time_to:
            params["time_to"] = time_to
        return self._get(params)

    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """
        Company Overview Trending
        Docs: https://www.alphavantage.co/documentation/#company-overview
        """
        params = {
            "function": "OVERVIEW",
            "symbol": symbol
        }
        return self._get(params)

    def get_etf_profile(self, symbol: str) -> Dict[str, Any]:
        """
        ETF Profile & Holdings
        Docs: https://www.alphavantage.co/documentation/#etf-profile
        """
        params = {
            "function": "ETF_PROFILE",
            "symbol": symbol
        }
        return self._get(params)

    def get_etf_holdings(self, symbol: str) -> Dict[str, Any]:
        """
        ETF Holdings
        Docs: https://www.alphavantage.co/documentation/#etf-holdings
        """
        params = {
            "function": "ETF_HOLDINGS",
            "symbol": symbol
        }
        return self._get(params)

    def get_dividends(self, symbol: str) -> Dict[str, Any]:
        """
        Corporate Action - Dividends
        Docs: https://www.alphavantage.co/documentation/#dividends
        """
        params = {
            "function": "DIVIDEND_HISTORY",
            "symbol": symbol
        }
        return self._get(params)

    def get_splits(self, symbol: str) -> Dict[str, Any]:
        """
        Corporate Action - Splits
        Docs: https://www.alphavantage.co/documentation/#splits
        """
        params = {
            "function": "SPLIT_HISTORY",
            "symbol": symbol
        }
        return self._get(params)

    def get_income_statement(self, symbol: str) -> Dict[str, Any]:
        """
        Income Statement
        Docs: https://www.alphavantage.co/documentation/#income-statement
        """
        params = {
            "function": "INCOME_STATEMENT",
            "symbol": symbol
        }
        return self._get(params)

    def get_balance_sheet(self, symbol: str) -> Dict[str, Any]:
        """
        Balance Sheet
        Docs: https://www.alphavantage.co/documentation/#balance-sheet
        """
        params = {
            "function": "BALANCE_SHEET",
            "symbol": symbol
        }
        return self._get(params)

    def get_cash_flow(self, symbol: str) -> Dict[str, Any]:
        """
        Cash Flow
        Docs: https://www.alphavantage.co/documentation/#cash-flow
        """
        params = {
            "function": "CASH_FLOW",
            "symbol": symbol
        }
        return self._get(params)

    def get_earnings(self, symbol: str) -> Dict[str, Any]:
        """
        Earnings History
        Docs: https://www.alphavantage.co/documentation/#earnings
        """
        params = {
            "function": "EARNINGS",
            "symbol": symbol
        }
        return self._get(params)

    def get_earnings_estimates(self, symbol: str) -> Dict[str, Any]:
        """
        Earnings Estimates Trending
        Docs: https://www.alphavantage.co/documentation/#earnings-estimates
        """
        params = {
            "function": "EARNINGS_ESTIMATE",
            "symbol": symbol
        }
        return self._get(params)

    def get_listing_status(self, state: str = "active") -> Dict[str, Any]:
        """
        Listing & Delisting Status
        Docs: https://www.alphavantage.co/documentation/#listing-status
        state: "active", "delisted", or "suspended"
        """
        params = {
            "function": "LISTING_STATUS",
            "state": state
        }
        return self._get(params)

    def get_earnings_calendar(self, horizon: str = "3month", symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Earnings Calendar
        Docs: https://www.alphavantage.co/documentation/#earnings-calendar
        horizon: "3month", "6month", "12month"
        """
        params = {
            "function": "EARNINGS_CALENDAR",
            "horizon": horizon
        }
        if symbol:
            params["symbol"] = symbol
        return self._get(params)

    def get_ipo_calendar(self, horizon: str = "3month") -> Dict[str, Any]:
        """
        IPO Calendar
        Docs: https://www.alphavantage.co/documentation/#ipo-calendar
        horizon: "3month", "6month", "12month"
        """
        params = {
            "function": "IPO_CALENDAR",
            "horizon": horizon
        }
        return self._get(params)

if __name__ == "__main__":
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY", "")
    if not api_key:
        print("Set ALPHAVANTAGE_API_KEY env var to run examples.")
    else:
        av = AlphaVantageClient(api_key=api_key)
        print(av.get_company_overview("AAPL"))
        print(av.get_news_sentiment(tickers=["AAPL", "MSFT"]))
        print(av.get_etf_profile("SPY"))
        print(av.get_etf_holdings("SPY"))
        print(av.get_dividends("AAPL"))
        print(av.get_splits("AAPL"))
        print(av.get_income_statement("AAPL"))
        print(av.get_balance_sheet("AAPL"))
        print(av.get_cash_flow("AAPL"))
        print(av.get_earnings("AAPL"))
        print(av.get_earnings_estimates("AAPL"))
        print(av.get_listing_status("active"))
        print(av.get_earnings_calendar(horizon="3month", symbol="AAPL"))
        print(av.get_ipo_calendar(horizon="3month"))
