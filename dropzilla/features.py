"""Feature engineering logic for Dropzilla v4."""

from __future__ import annotations

import numpy as np
import pandas as pd

# pandas-ta expects ``numpy.NaN`` which was removed in newer numpy versions.
# Provide the alias for compatibility before importing pandas_ta.
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # type: ignore[attr-defined]

import pandas_ta as ta


def calculate_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Calculate all v4 features for the given OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with columns ``['Open', 'High', 'Low', 'Close', 'Volume']``
        and a ``DatetimeIndex``.
    config : dict | None, optional
        Optional configuration dictionary, by default ``None``.

    Returns
    -------
    pd.DataFrame
        The original ``DataFrame`` augmented with new feature columns.
    """

    if df.empty:
        return df

    # --- Feature Group: Relative Volume ---
    avg_vol_period = 50
    df["avg_volume"] = (
        df["Volume"].rolling(window=avg_vol_period, min_periods=avg_vol_period).mean()
    )
    df["relative_volume"] = (df["Volume"] / df["avg_volume"]).fillna(1.0)

    # --- Feature Group: VWAP-centric Features ---
    df_reset = df.reset_index().set_index("index")
    vwap = ta.vwap(
        high=df_reset["High"],
        low=df_reset["Low"],
        close=df_reset["Close"],
        volume=df_reset["Volume"],
    )
    if vwap is not None:
        df["vwap"] = vwap.values
        df["distance_from_vwap_pct"] = (df["Close"] - df["vwap"]) / df["vwap"]
        df["vwap_slope"] = ta.slope(df["vwap"], length=10)
    else:
        df["vwap"] = np.nan
        df["distance_from_vwap_pct"] = np.nan
        df["vwap_slope"] = np.nan

    # --- Feature Group: Momentum and Acceleration (Example) ---
    df["roc_60"] = ta.roc(df["Close"], length=60)

    # --- Clean up ---
    df = df.drop(columns=["avg_volume"])
    df = df.fillna(method="bfill").fillna(method="ffill")

    return df

