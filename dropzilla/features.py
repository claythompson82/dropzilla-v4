"""
Handles all feature engineering logic for Dropzilla v4.

This module takes a raw OHLCV DataFrame and returns a DataFrame with all
the calculated features needed for the model.
"""
import numpy as np
import pandas as pd

# pandas-ta expects the deprecated `numpy.NaN` alias which was removed in
# numpy 2.0. Recreate it if missing for backward compatibility.
if not hasattr(np, "NaN"):
    setattr(np, "NaN", np.nan)

import pandas_ta as ta

def calculate_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """
    Calculates all v4 features for the given OHLCV data.

    Args:
        df (pd.DataFrame): Input DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume'].
                           Must have a DatetimeIndex.
        config (dict, optional): Configuration dictionary. Defaults to None.

    Returns:
        pd.DataFrame: The original DataFrame augmented with new feature columns.
    """
    if df.empty:
        return df

    # --- Feature Group: Relative Volume ---
    # Calculate a rolling average of volume to compare against
    avg_vol_period = 50
    df['avg_volume'] = df['Volume'].rolling(window=avg_vol_period, min_periods=avg_vol_period).mean()
    # Calculate relative volume. Fill initial NaNs with 1 (neutral).
    df['relative_volume'] = (df['Volume'] / df['avg_volume']).fillna(1.0)

    # --- Feature Group: VWAP-centric Features ---
    # pandas-ta's vwap function correctly handles daily resets when passed
    # Series with a DatetimeIndex.
    vwap = ta.vwap(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
    if vwap is not None:
        df['vwap'] = vwap # Assign the resulting Series directly
        # Calculate distance from VWAP as a percentage
        df['distance_from_vwap_pct'] = (df['Close'] - df['vwap']) / df['vwap']
        # Calculate slope of VWAP
        df['vwap_slope'] = ta.slope(df['vwap'], length=10)
    else:
        df['vwap'] = np.nan
        df['distance_from_vwap_pct'] = np.nan
        df['vwap_slope'] = np.nan


    # --- Feature Group: Momentum and Acceleration ---
    # Multi-timescale Rate of Change (ROC) for velocity
    df['roc_30'] = ta.roc(df['Close'], length=30)
    df['roc_60'] = ta.roc(df['Close'], length=60)
    df['roc_120'] = ta.roc(df['Close'], length=120)

    # RSI and its smoothed version
    df['rsi_14'] = ta.rsi(df['Close'], length=14)
    df['rsi_14_sma_5'] = ta.sma(df['rsi_14'], length=5)  # Smoothed RSI trend

    # MACD for trend and momentum
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df['macd_line'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
        # Acceleration of momentum (change in the histogram)
        df['macd_hist_diff'] = df['macd_hist'].diff()

    # --- Feature Group: Volume and Money Flow ---
    # Money Flow Index (MFI) - the volume-weighted RSI
    df['mfi_14'] = ta.mfi(
        high=df['High'], low=df['Low'], close=df['Close'],
        volume=df['Volume'], length=14
    )

    # On-Balance Volume (OBV) to measure cumulative pressure
    obv = ta.obv(df['Close'], df['Volume'])
    if obv is not None:
        df['obv'] = obv
        # The trend of OBV is often more useful than its raw value
        df['obv_slope'] = ta.slope(df['obv'], length=10)

    # --- Feature Group: Contextual ---
    # The regime is calculated separately and merged in.
    # We assume 'market_regime' column is already present if this feature is used.
    if 'market_regime' in df.columns:
        # No calculation needed here, just ensure it's kept
        pass

    # --- Clean up ---
    # Drop intermediate columns and handle NaNs
    if 'avg_volume' in df.columns:
        df = df.drop(columns=['avg_volume'])
    df = df.fillna(method='bfill').fillna(method='ffill')

    return df
