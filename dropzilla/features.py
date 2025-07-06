"""
Handles all feature engineering logic for Dropzilla v4.

This module takes a raw OHLCV DataFrame and returns a DataFrame with all
the calculated features needed for the model.
"""
import numpy as np
import pandas as pd
import pandas_ta as ta

# Import our new advanced volatility model
from dropzilla.volatility import get_mc_garch_volatility_forecast

# pandas-ta expects the deprecated `numpy.NaN` alias which was removed in
# numpy 2.0. Recreate it if missing for backward compatibility.
if not hasattr(np, "NaN"):
    np.NaN = np.nan


def calculate_features(df: pd.DataFrame,
                       daily_log_returns: pd.Series,
                       config: dict | None = None) -> pd.DataFrame:
    """
    Calculates all v4 features for the given OHLCV data.

    Args:
        df (pd.DataFrame): Input DataFrame with minute-level OHLCV data.
        daily_log_returns (pd.Series): A Series of daily log returns for the same asset.
        config (dict, optional): Configuration dictionary. Defaults to None.

    Returns:
        pd.DataFrame: The original DataFrame augmented with new feature columns.
    """
    if df.empty:
        return df

    # --- All existing feature calculations remain the same ---
    # (Relative Volume, VWAP, Momentum, Volume/Flow)
    # ... (code for these features is unchanged) ...
    avg_vol_period = 50
    df['avg_volume'] = df['Volume'].rolling(window=avg_vol_period, min_periods=avg_vol_period).mean()
    df['relative_volume'] = (df['Volume'] / df['avg_volume']).fillna(1.0)
    vwap = ta.vwap(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
    if vwap is not None:
        df['vwap'] = vwap
        df['distance_from_vwap_pct'] = (df['Close'] - df['vwap']) / df['vwap']
        df['vwap_slope'] = ta.slope(df['vwap'], length=10)
    else:
        df['vwap'] = np.nan
        df['distance_from_vwap_pct'] = np.nan
        df['vwap_slope'] = np.nan
    df['roc_30'] = ta.roc(df['Close'], length=30)
    df['roc_60'] = ta.roc(df['Close'], length=60)
    df['roc_120'] = ta.roc(df['Close'], length=120)
    df['rsi_14'] = ta.rsi(df['Close'], length=14)
    df['rsi_14_sma_5'] = ta.sma(df['rsi_14'], length=5)
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df['macd_line'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
        df['macd_hist_diff'] = df['macd_hist'].diff()
    df['mfi_14'] = ta.mfi(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], length=14)
    obv = ta.obv(df['Close'], df['Volume'])
    if obv is not None:
        df['obv'] = obv
        df['obv_slope'] = ta.slope(df['obv'], length=10)


    # --- Feature Group: Advanced Volatility (MC-GARCH) ---
    # --- THIS SECTION IS CORRECTED ---
    try:
        intraday_log_returns = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
        mc_garch_forecast = get_mc_garch_volatility_forecast(daily_log_returns, intraday_log_returns)
        
        # The forecast is a single value for the next period. We align it to the last known timestamp.
        if mc_garch_forecast is not None and not mc_garch_forecast.empty:
            last_timestamp = mc_garch_forecast.index[0] - pd.Timedelta(minutes=1)
            realized_vol_at_forecast = intraday_log_returns.rolling(window=21).std().get(last_timestamp)
            
            if realized_vol_at_forecast is not None and mc_garch_forecast.iloc[0] > 0:
                surprise = (realized_vol_at_forecast - mc_garch_forecast.iloc[0]) / mc_garch_forecast.iloc[0]
                # Initialize the column with a neutral 0
                df['volatility_surprise'] = 0.0
                # Set the calculated surprise only for the specific timestamp it applies to
                df.loc[last_timestamp, 'volatility_surprise'] = surprise
            else:
                df['volatility_surprise'] = 0.0
        else:
            df['volatility_surprise'] = 0.0
    except Exception as e:
        print(f"Warning: Could not calculate GARCH feature. Setting to 0. Error: {e}")
        df['volatility_surprise'] = 0.0
    # --- END CORRECTION ---
    
    # --- Clean up ---
    columns_to_drop = ['avg_volume']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    df = df.fillna(method='bfill').fillna(method='ffill')

    return df
