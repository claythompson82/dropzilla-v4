"""
Handles all feature engineering for the Dropzilla model.
This version includes robust data type handling to prevent errors.
"""
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", message="pkg_resources is deprecated*", category=UserWarning)
import pandas_ta as ta
from dropzilla.volatility import get_mc_garch_volatility_forecast
from dropzilla.config import FEATURE_CONFIG

def calculate_features(
    df: pd.DataFrame,
    tick_data: pd.DataFrame,
    daily_log_returns: pd.Series,
    config: dict = FEATURE_CONFIG,
) -> pd.DataFrame:
    """
    Calculates all features for the model.
    """
    # Work on a copy to prevent SettingWithCopyWarning
    features_df = df.copy()
    
    # --- ROBUSTNESS FIX: Ensure OHLCV data are floats to prevent dtype errors ---
    for col in ["Open", "High", "Low", "Close", "Volume", "Vwap"]:
        if col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce")
            if col == "Volume":
                features_df[col] = features_df[col].astype("float64")
    # --- END FIX ---

    if 'Vwap' not in features_df.columns:
        features_df['Vwap'] = features_df['Close']

    features_df.ta.rsi(length=config['rsi_period'], append=True)
    features_df.ta.macd(fast=config['macd_fast'], slow=config['macd_slow'], signal=config['macd_signal'], append=True)
    features_df.ta.mfi(length=config['mfi_period'], append=True)
    features_df.ta.obv(append=True)

    features_df['relative_volume'] = features_df['Volume'] / features_df['Volume'].rolling(window=config['relative_volume_period']).mean()
    if 'Vwap' in features_df.columns:
        features_df['vwap_slope'] = features_df['Vwap'].rolling(
            window=config['vwap_slope_period']
        ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)
        features_df['distance_from_vwap_pct'] = (
            (features_df['Close'] - features_df['Vwap']) / features_df['Vwap'] * 100
        )
    else:
        features_df['vwap_slope'] = np.nan
        features_df['distance_from_vwap_pct'] = np.nan
    features_df["roc_30"] = features_df["Close"].pct_change(periods=30, fill_method=None)
    features_df["roc_60"] = features_df["Close"].pct_change(periods=60, fill_method=None)
    features_df["roc_120"] = features_df["Close"].pct_change(periods=120, fill_method=None)

    features_df['rsi_14_sma_5'] = features_df[f'RSI_{config["rsi_period"]}'].rolling(window=5).mean()
    features_df['macd_hist_diff'] = features_df[f'MACDh_{config["macd_fast"]}_{config["macd_slow"]}_{config["macd_signal"]}'].diff()
    features_df['obv_slope'] = features_df['OBV'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )

    try:
        clean_daily_returns = daily_log_returns.replace([np.inf, -np.inf], np.nan).dropna()
        if not clean_daily_returns.empty:
            intraday_log_returns = (
                np.log(features_df["Close"].replace(0, np.nan))
                .diff()
                .dropna()
            )
            if not intraday_log_returns.empty:
                garch_forecast = get_mc_garch_volatility_forecast(clean_daily_returns, intraday_log_returns)
                realized_vol = intraday_log_returns.rolling(window=config['garch_realized_vol_period']).std()
                surprise = (realized_vol - garch_forecast.iloc[0]) / garch_forecast.iloc[0]
                features_df['volatility_surprise'] = surprise
            else:
                features_df['volatility_surprise'] = 0
        else:
            features_df['volatility_surprise'] = 0
    except Exception as e:
        print(f"Warning: Could not calculate GARCH feature. Setting to 0. Error: {e}", flush=True)
        features_df['volatility_surprise'] = 0
        
    features_df['volatility_surprise'] = features_df['volatility_surprise'].fillna(0)

    features_df = features_df.rename(columns={
        f'RSI_{config["rsi_period"]}': 'rsi_14',
        f'MACD_{config["macd_fast"]}_{config["macd_slow"]}_{config["macd_signal"]}': 'macd_line',
        f'MACDs_{config["macd_fast"]}_{config["macd_slow"]}_{config["macd_signal"]}': 'macd_signal',
        f'MACDh_{config["macd_fast"]}_{config["macd_slow"]}_{config["macd_signal"]}': 'macd_hist',
        f'MFI_{config["mfi_period"]}': 'mfi_14',
        'Vwap': 'vwap'
    })
    
    return features_df
