"""
Handles all feature engineering for the Dropzilla model.
This version includes robust data type handling to prevent errors.
"""

import warnings
from typing import Any

import pandas as pd
import numpy as np
import pandas_ta as ta

from dropzilla.volatility import get_mc_garch_volatility_forecast
from dropzilla.config import FEATURE_CONFIG

# suppress noisy deprecation warnings from other libraries
warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated*", category=UserWarning
)


def calculate_features(
    df: pd.DataFrame,
    tick_data: pd.DataFrame,
    daily_log_returns: pd.Series,
    config: dict[str, Any] = FEATURE_CONFIG,
) -> pd.DataFrame:
    """
    Calculates all features for the model.
    """
    features_df = df.copy()

    # Ensure numeric types for OHLCV/Vwap
    for col in ["Open", "High", "Low", "Close", "Volume", "Vwap"]:
        if col in features_df:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce")
            if col == "Volume":
                features_df[col] = features_df[col].astype("float64")

    # If Vwap is missing, default to Close
    if "Vwap" not in features_df:
        features_df["Vwap"] = features_df["Close"].astype("float64")

    # Core TA indicators
    features_df.ta.rsi(length=config["rsi_period"], append=True)
    features_df.ta.macd(
        fast=config["macd_fast"],
        slow=config["macd_slow"],
        signal=config["macd_signal"],
        append=True,
    )
    # Compute MFI into a standalone Series, cast, then assign
    mfi_series = features_df.ta.mfi(
        length=config["mfi_period"], append=False
    )
    features_df[f"MFI_{config['mfi_period']}"] = mfi_series.astype("float64")

    # Compute OBV and cast
    obv_series = features_df.ta.obv(append=False)
    features_df["OBV"] = obv_series.astype("float64")

    # Relative volume
    features_df["relative_volume"] = (
        features_df["Volume"]
        / features_df["Volume"]
        .rolling(window=config["relative_volume_period"])
        .mean()
    )

    # VWAP‐based features
    features_df["vwap_slope"] = (
        features_df["Vwap"]
        .rolling(window=config["vwap_slope_period"])
        .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)
    )
    features_df["distance_from_vwap_pct"] = (
        (features_df["Close"] - features_df["Vwap"])
        / features_df["Vwap"]
        * 100
    )

    # ROC features
    for p in (30, 60, 120):
        features_df[f"roc_{p}"] = features_df["Close"].pct_change(
            periods=p, fill_method=None
        )

    # SMA and diff
    rsi_col = f"RSI_{config['rsi_period']}"
    features_df[f"{rsi_col}_sma_5"] = (
        features_df[rsi_col].rolling(window=5).mean()
    )
    macd_hist_col = (
        f"MACDh_{config['macd_fast']}_{config['macd_slow']}_{config['macd_signal']}"
    )
    features_df["macd_hist_diff"] = features_df[macd_hist_col].diff()
    features_df["obv_slope"] = features_df["OBV"].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )

    # GARCH‐based volatility surprise
    try:
        clean_dr = (
            daily_log_returns.replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if not clean_dr.empty:
            intra = (
                np.log(features_df["Close"].replace(0, np.nan))
                .diff()
                .dropna()
            )
            if not intra.empty:
                garch_fc = get_mc_garch_volatility_forecast(clean_dr, intra)
                realized = intra.rolling(window=config["garch_realized_vol_period"]).std()
                features_df["volatility_surprise"] = (
                    (realized - garch_fc.iloc[0]) / garch_fc.iloc[0]
                )
            else:
                features_df["volatility_surprise"] = 0.0
        else:
            features_df["volatility_surprise"] = 0.0
    except Exception as e:
        print(f"Warning: GARCH feature error: {e}", flush=True)
        features_df["volatility_surprise"] = 0.0

    features_df["volatility_surprise"] = features_df["volatility_surprise"].fillna(0.0)

    # Rename for model consumption
    rename_map = {
        rsi_col: f"rsi_{config['rsi_period']}",
        f"MACD_{config['macd_fast']}_{config['macd_slow']}_{config['macd_signal']}": "macd_line",
        f"MACDs_{config['macd_fast']}_{config['macd_slow']}_{config['macd_signal']}": "macd_signal",
        macd_hist_col: "macd_hist",
        f"MFI_{config['mfi_period']}": f"mfi_{config['mfi_period']}",
        "Vwap": "vwap",
    }
    features_df = features_df.rename(columns=rename_map)

    return features_df
