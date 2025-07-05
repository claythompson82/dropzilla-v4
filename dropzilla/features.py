"""
Handles all feature engineering logic for Dropzilla v4.

This module takes a raw OHLCV DataFrame and returns a DataFrame with all
the calculated features needed for the model.
"""
import numpy as np
np.NaN = np.nan  # Compatibility alias for pandas_ta
import pandas as pd
import pandas_ta as ta

def calculate_features(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
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
    # VWAP calculation using pandas-ta requires a DatetimeIndex
    vwap = ta.vwap(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
    if vwap is not None:
        df['vwap'] = vwap.values
        # Calculate distance from VWAP as a percentage
        df['distance_from_vwap_pct'] = (df['Close'] - df['vwap']) / df['vwap']
        # Calculate slope of VWAP
        df['vwap_slope'] = ta.slope(df['vwap'], length=10)
    else:
        df['vwap'] = np.nan
        df['distance_from_vwap_pct'] = np.nan
        df['vwap_slope'] = np.nan


    # --- Feature Group: Momentum and Acceleration (Example) ---
    df['roc_60'] = ta.roc(df['Close'], length=60) # 1-hour rate of change
    
    # --- Clean up ---
    # Drop intermediate columns and handle NaNs
    df = df.drop(columns=['avg_volume'])
    df = df.fillna(method='bfill').fillna(method='ffill')

    return df
