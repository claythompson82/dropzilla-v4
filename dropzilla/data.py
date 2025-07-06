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

    def __init__(self, api_key: str, cache_dir: str = ".cache", cache_ttl_seconds: int = 3600):
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

    def _get_cache_path(self, symbol: str, from_date: str, to_date: str, timespan: str) -> str:
        """Generates a consistent file path for a given cache request."""
        filename = f"{symbol}_{from_date}_{to_date}_{timespan}.joblib"
        return os.path.join(self.cache_dir, filename)

    def get_aggs(self, symbol: str, from_date: str, to_date: str, multiplier: int = 1, timespan: str = 'minute', max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Fetches aggregate bars for a ticker over a date range, with caching.
        """
        cache_path = self._get_cache_path(symbol, from_date, to_date, timespan)
        
        if os.path.exists(cache_path):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - file_mod_time < self.cache_ttl:
                print(f"Cache HIT for {symbol} ({timespan}). Loading from {cache_path}")
                return joblib.load(cache_path)

        for attempt in range(max_retries):
            try:
                print(f"Cache MISS for {symbol} ({timespan}). Fetching from Polygon API...")
                aggs = self.client.list_aggs(
                    ticker=symbol, multiplier=multiplier, timespan=timespan,
                    from_=from_date, to=to_date, limit=50000
                )
                df = pd.DataFrame([a.__dict__ for a in aggs])
                if df.empty:
                    return pd.DataFrame()
                
                df = df.rename(columns={'timestamp': 'Timestamp', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume', 'vwap': 'Vwap'})
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
                df.set_index("Timestamp", inplace=True)
                
                joblib.dump(df, cache_path)
                return df

            except Exception as e:
                print(f"Error fetching {symbol} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None
        return None

    def get_tick_data(self, symbol: str, date_str: str) -> pd.DataFrame | None:
        """
        Fetches all tick-level trade data for a given symbol and a single date.
        Handles pagination automatically.

        Args:
            symbol (str): The ticker symbol.
            date_str (str): The date to fetch data for in YYYY-MM-DD format.

        Returns:
            pd.DataFrame | None: A DataFrame of all trades for that day, or None.
        """
        cache_path = self._get_cache_path(symbol, date_str, date_str, "ticks")
        
        if os.path.exists(cache_path):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - file_mod_time < self.cache_ttl:
                print(f"Cache HIT for {symbol} (ticks on {date_str}). Loading from {cache_path}")
                return joblib.load(cache_path)

        print(f"Fetching tick data for {symbol} on {date_str}...")
        all_trades = []
        try:
            for trade in self.client.list_trades(symbol, date_str, limit=50000):
                all_trades.append(trade)
            
            if not all_trades:
                return pd.DataFrame()

            df = pd.DataFrame([t.__dict__ for t in all_trades])
            df = df.rename(columns={'participant_timestamp': 'Timestamp', 'price': 'Price', 'size': 'Volume'})
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df.set_index('Timestamp', inplace=True)
            
            joblib.dump(df, cache_path)
            return df[['Price', 'Volume']].sort_index()

        except Exception as e:
            print(f"Error fetching tick data for {symbol}: {e}")
            return None
