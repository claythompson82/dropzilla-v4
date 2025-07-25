# Filename: dropzilla/data.py
"""
Light wrapper around Polygon.io’s REST client with disk‑based caching & retries.

Key points
----------
* Compatible with polygon-api-client 1.13 → latest (aliases exceptions that changed in 1.14+)
* Uses epoch‑milliseconds for any intraday window, YYYY‑MM‑DD for true daily bars
* Local joblib caching with TTL
* Full logging (no print spam)
"""

from __future__ import annotations

import os
import time as _time
import logging
from datetime import datetime, timedelta
from typing import Optional, Union

import joblib
import pandas as pd
from polygon import RESTClient
from polygon.exceptions import BadResponse

LOG = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Exception aliases – polygon 1.14 collapsed HTTPError / NoResultsError
# -----------------------------------------------------------------------------
try:
    from polygon.exceptions import NoResultsError  # type: ignore
except ImportError:  # 1.14+
    NoResultsError = BadResponse  # type: ignore[assignment]

try:
    from polygon.exceptions import HTTPError  # type: ignore
except ImportError:  # 1.14+
    HTTPError = BadResponse  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _fmt_ts(ts: Union[str, datetime], *, intraday: bool) -> Union[str, int]:
    """
    Convert flexible datetime input → Polygon-friendly timestamp.

    * For **intraday** (timespan != "day") or if the datetime includes a time component,
      we return **epoch milliseconds** (int).
    * For strict **daily** bars (and midnight-only dates), we return "YYYY-MM-DD".
    """
    if isinstance(ts, str):
        try:
            ts_dt = datetime.fromisoformat(ts.replace("Z", ""))
        except ValueError:
            ts_dt = datetime.strptime(ts, "%Y-%m-%d")
    else:
        ts_dt = ts

    if intraday or ts_dt.time() != datetime.min.time():
        return int(ts_dt.timestamp() * 1000)
    return ts_dt.strftime("%Y-%m-%d")


def _cache_path(
    cache_dir: str,
    symbol: str,
    from_v: Union[str, int],
    to_v: Union[str, int],
    multiplier: int,
    timespan: str,
    adjusted: bool,
) -> str:
    from_key = str(from_v)
    to_key = str(to_v)
    fname = (
        f"{symbol}_{from_key}_{to_key}_{multiplier}x{timespan}"
        + ("_adj" if adjusted else "_raw")
        + ".joblib"
    )
    return os.path.join(cache_dir, fname)


# -----------------------------------------------------------------------------
# Main client
# -----------------------------------------------------------------------------
class PolygonDataClient:
    """
    Thin wrapper that:

    • Verifies the key on startup
    • Caches queries locally to save credits / speed up retries
    • Provides retry/back‑off on 429 / transient HTTP errors
    """

    def __init__(
        self,
        api_key: str,
        *,
        cache_dir: str = ".cache",
        cache_ttl_seconds: int = 3_600,
        max_retries: int = 3,
        retry_backoff_sec: float = 1.5,
        http_timeout: int = 10,
    ) -> None:
        if not api_key:
            raise ValueError("Polygon API key required.")

        # Do NOT pass unknown kwargs like timeout= to older client versions
        self.client = RESTClient(api_key)

        self.cache_dir = cache_dir
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.http_timeout = http_timeout  # kept for parity / future

        os.makedirs(self.cache_dir, exist_ok=True)
        self._verify_connection()

    # -------------------------------------------------------------------------
    # connection sanity‑check
    # -------------------------------------------------------------------------
    def _verify_connection(self) -> None:
        try:
            self.client.get_ticker_details("AAPL")
            LOG.debug("Polygon API connection successful.")
        except BadResponse as exc:
            raise RuntimeError(f"Polygon API key failed validation: {exc}") from exc

    # -------------------------------------------------------------------------
    # aggregate bars
    # -------------------------------------------------------------------------
    def get_aggs(
        self,
        symbol: str,
        from_date: Union[str, datetime],
        to_date: Union[str, datetime],
        *,
        multiplier: int = 1,
        timespan: str = "minute",
        adjusted: bool = True,
        limit: int = 50_000,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch & cache aggregate bars.

        Parameters
        ----------
        adjusted : bool
            Passed straight through to Polygon.  Valid for SDK ≥ 1.13.

        Returns
        -------
        pd.DataFrame or None
            On total failure returns None; on “no data” returns empty DF.
        """
        intraday = timespan != "day"
        from_v = _fmt_ts(from_date, intraday=intraday)
        to_v = _fmt_ts(to_date, intraday=intraday)

        path = _cache_path(
            self.cache_dir, symbol, from_v, to_v, multiplier, timespan, adjusted
        )
        if (
            os.path.exists(path)
            and datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
            < self.cache_ttl
        ):
            LOG.debug("Cache HIT for %s (%s) – %s", symbol, timespan, path)
            return joblib.load(path)

        # --- hit network with retry loop ------------------------------------- #
        for attempt in range(1, self.max_retries + 1):
            try:
                LOG.debug(
                    "Cache MISS for %s (%s) – polygon attempt %d/%d …",
                    symbol,
                    timespan,
                    attempt,
                    self.max_retries,
                )
                aggs = self.client.list_aggs(
                    ticker=symbol,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_=from_v,
                    to=to_v,
                    adjusted=adjusted,
                    limit=limit,
                )

                # Convert to DataFrame
                df = pd.DataFrame([bar.__dict__ for bar in aggs])
                if df.empty:
                    joblib.dump(df, path)
                    return df

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
                # polygon returns ms since epoch
                if "Timestamp" in df.columns:
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
                    df = df.set_index("Timestamp").sort_index()

                joblib.dump(df, path)
                return df

            except NoResultsError:
                # Treat "no results" as empty data, not a hard failure
                empty = pd.DataFrame()
                joblib.dump(empty, path)
                return empty

            except HTTPError as exc:
                status = getattr(exc, "status", None)
                LOG.warning("[%s] HTTP %s: %s", symbol, status, exc)
                if status == 429 and attempt < self.max_retries:
                    sleep_for = self.retry_backoff_sec * attempt
                    LOG.warning("Hit rate‑limit (429). Sleeping %.1fs …", sleep_for)
                    _time.sleep(sleep_for)
                    continue
                # fallthrough to retry/backoff below

            except Exception as exc:  # noqa: BLE001
                LOG.exception("[%s] unexpected error", symbol, exc_info=exc)

            _time.sleep(self.retry_backoff_sec)

        # completely exhausted retries
        return None

    # -------------------------------------------------------------------------
    # tick data
    # -------------------------------------------------------------------------
    def get_tick_data(self, symbol: str, date_str: str) -> Optional[pd.DataFrame]:
        """
        Download & cache full trade tape for a single day.
        """
        path = _cache_path(self.cache_dir, symbol, date_str, date_str, 1, "ticks", False)
        if (
            os.path.exists(path)
            and datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
            < self.cache_ttl
        ):
            LOG.debug("Cache HIT for %s ticks %s – %s", symbol, date_str, path)
            return joblib.load(path)

        LOG.debug("Fetching tick data for %s on %s …", symbol, date_str)
        trades: list[dict] = []
        try:
            for t in self.client.list_trades(symbol, date_str, limit=50_000):
                trades.append(t.__dict__)

            if not trades:
                empty = pd.DataFrame()
                joblib.dump(empty, path)
                return empty

            df = (
                pd.DataFrame(trades)
                .rename(
                    columns={
                        "participant_timestamp": "Timestamp",
                        "price": "Price",
                        "size": "Volume",
                    }
                )
            )

            if "Timestamp" not in df:
                LOG.warning("No 'participant_timestamp' found for %s on %s", symbol, date_str)
                empty = pd.DataFrame()
                joblib.dump(empty, path)
                return empty

            df = (
                df.assign(Timestamp=lambda d: pd.to_datetime(d["Timestamp"]))
                .set_index("Timestamp")
                .sort_index()
            )
            out = df[["Price", "Volume"]]
            joblib.dump(out, path)
            return out

        except Exception as exc:  # noqa: BLE001
            LOG.exception("Tick‑data fetch failed for %s", symbol, exc_info=exc)
            return None
