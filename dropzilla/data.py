"""
Handles all data acquisition and caching from the Polygon.io API.
"""

import os
import joblib
import pandas as pd
from datetime import datetime, timedelta
from polygon import RESTClient
from polygon.exceptions import BadResponse
from typing import Optional


class PolygonDataClient:
    """A client for fetching and caching financial data from Polygon.io."""

    def __init__(
        self, api_key: str, cache_dir: str = ".cache", cache_ttl_seconds: int = 3600
    ):
        if not api_key:
            raise ValueError("Polygon API key is required.")

        self.client = RESTClient(api_key=api_key)
        self.cache_dir = cache_dir
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self._verify_connection()

    def _verify_connection(self):
        """Verify the API key is valid by making a test call."""
        try:
            self.client.get_ticker_details("AAPL")
            print("Polygon API connection successful.")
        except BadResponse as e:
            print(f"Polygon API key validation failed: {e}")
            raise

    def _get_cache_path(self, symbol: str, from_date: str, to_date: str) -> str:
        """Generates a consistent file path for a given cache request."""
        filename = f"{symbol}_{from_date}_{to_date}.joblib"
        return os.path.join(self.cache_dir, filename)

    def get_aggs(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        multiplier: int = 1,
        timespan: str = "minute",
        max_retries: int = 3,
    ) -> Optional[pd.DataFrame]:
        """
        Fetches aggregate bars for a ticker over a date range, with caching.

        Args:
            symbol (str): The ticker symbol.
            from_date (str): The start date in YYYY-MM-DD format.
            to_date (str): The end date in YYYY-MM-DD format.
            multiplier (int): The size of the time window.
            timespan (str): The size of the time window (e.g., 'minute', 'day').
            max_retries (int): Number of times to retry on network failure.

        Returns:
            Optional[pd.DataFrame]: A DataFrame of the aggregate bars, or None if failed.
        """
        cache_path = self._get_cache_path(symbol, from_date, to_date)

        # Check cache first
        if os.path.exists(cache_path):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - file_mod_time < self.cache_ttl:
                print(f"Cache HIT for {symbol}. Loading from {cache_path}")
                return joblib.load(cache_path)

        # Fetch from API if not in cache or expired
        for attempt in range(max_retries):
            try:
                print(f"Cache MISS for {symbol}. Fetching from Polygon API...")
                aggs = self.client.list_aggs(
                    ticker=symbol,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_=from_date,
                    to=to_date,
                    limit=50000,
                )
                df = pd.DataFrame([a.__dict__ for a in aggs])
                if df.empty:
                    return pd.DataFrame()  # Return empty DF for no data

                df = df.rename(
                    columns={
                        "timestamp": "Timestamp",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                        "vwap": "Vwap",
                    }
                )
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
                df.set_index("Timestamp", inplace=True)

                joblib.dump(df, cache_path)  # Save to cache
                return df

            except Exception as e:
                print(
                    f"Error fetching {symbol} (attempt {attempt+1}/{max_retries}): {e}"
                )
                if attempt == max_retries - 1:
                    return None
        return None
