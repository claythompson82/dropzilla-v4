# Filename: dropzilla/features.py
"""
Feature engineering for Dropzilla v4.

Design goals
------------
* Never crash in production: every external dependency is wrapped with safe fallbacks.
* Strong typing, consistent NaN/Inf handling, and rich logging.
* Provides a fast 10‑day Volatility Regime Anomaly (VRA) score, with a safe fallback.
* Keeps vectors inside; GUI can read the latest values it needs.

This file is self-contained: if pandas_ta or dropzilla.volatility aren't importable,
we degrade gracefully to safe defaults.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger("dropzilla.features")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------
# Optional deps & safe wrappers
# ---------------------------------------------------------------------
try:
    import pandas_ta as ta  # type: ignore
except Exception:  # pragma: no cover
    ta = None  # sentinel


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    if ta is not None:
        try:
            return ta.rsi(close=close, length=length)
        except Exception as e:  # pragma: no cover
            logger.warning("pandas_ta.rsi failed, falling back: %s", e)
    # Fallback implementation
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).ewm(alpha=1 / length, adjust=False).mean()
    roll_down = pd.Series(down, index=close.index).ewm(alpha=1 / length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    if ta is not None:
        try:
            return ta.macd(close=close, fast=fast, slow=slow, signal=signal)
        except Exception as e:  # pragma: no cover
            logger.warning("pandas_ta.macd failed, falling back: %s", e)
    # Fallback implementation
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return pd.DataFrame(
        {
            "MACD_12_26_9": macd_line,
            "MACDs_12_26_9": macd_signal,
            "MACDh_12_26_9": macd_hist,
        }
    )


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
    if ta is not None:
        try:
            return ta.mfi(high=high, low=low, close=close, volume=volume, length=length)
        except Exception as e:  # pragma: no cover
            logger.warning("pandas_ta.mfi failed, falling back: %s", e)
    # Fallback implementation
    tp = (high + low + close) / 3.0
    mf = tp * volume
    delta_tp = tp.diff()
    pos_mf = np.where(delta_tp > 0, mf, 0.0)
    neg_mf = np.where(delta_tp < 0, mf, 0.0)
    pos_roll = pd.Series(pos_mf, index=tp.index).rolling(length, min_periods=1).sum()
    neg_roll = pd.Series(neg_mf, index=tp.index).rolling(length, min_periods=1).sum()
    mfr = pos_roll / (neg_roll + 1e-12)
    return 100.0 - (100.0 / (1.0 + mfr))


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    if ta is not None:
        try:
            return ta.obv(close=close, volume=volume)
        except Exception as e:  # pragma: no cover
            logger.warning("pandas_ta.obv failed, falling back: %s", e)
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


# ---------------------------------------------------------------------
# Config with safe default
# ---------------------------------------------------------------------
try:
    from dropzilla.config import FEATURE_CONFIG  # type: ignore
except Exception:  # pragma: no cover
    FEATURE_CONFIG = {"relative_volume_period": 30}

# ---------------------------------------------------------------------
# Volatility helpers with safe fallbacks
# ---------------------------------------------------------------------
try:
    from dropzilla.volatility import (  # type: ignore
        get_mc_garch_volatility_forecast,
        get_volatility_regime_anomaly,
    )
except Exception:  # pragma: no cover

    def get_mc_garch_volatility_forecast(*_args, **_kwargs) -> pd.Series:
        logger.debug("get_mc_garch_volatility_forecast fallback engaged.")
        return pd.Series(dtype=float)

    def get_volatility_regime_anomaly(series: pd.Series) -> pd.Series:
        logger.debug("get_volatility_regime_anomaly fallback engaged.")
        return pd.Series(0.0, index=series.index)


__all__ = ["calculate_features"]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _slope(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    idx = np.arange(len(values), dtype=float)
    try:
        return float(np.polyfit(idx, values.values, 1)[0])
    except Exception:
        return 0.0


def _to_series(data: Any, index: Optional[pd.Index] = None) -> pd.Series:
    if isinstance(data, pd.Series):
        return data
    return pd.Series(data, index=index)


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def calculate_features(
    df: pd.DataFrame,
    daily_log_returns: Any | None,
    config: dict | None = FEATURE_CONFIG,
) -> pd.DataFrame:
    """
    Compute all Dropzilla v4 features for a single OHLCV dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Intraday OHLCV data. Index must be a DatetimeIndex.
        Required columns: Open, High, Low, Close, Volume.
        Optional: Vwap (falls back to Close).
    daily_log_returns : array-like | pd.Series | None
        Daily log returns for the volatility (GARCH) piece. If None, they are
        computed from df by downsampling to daily.
    config : dict, optional
        Feature configuration.

    Returns
    -------
    pd.DataFrame
        Same index as `df` with all feature columns appended.
    """
    if df is None or df.empty:
        logger.warning("Empty dataframe passed to calculate_features; returning as-is.")
        return df.copy()

    if config is None:
        config = FEATURE_CONFIG

    features_df = df.copy()
    logger.debug("Starting feature calculation for dataframe with shape: %s", features_df.shape)

    # -----------------------------------------------------------------
    # Ensure numeric / clean OHLCV
    # -----------------------------------------------------------------
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in features_df.columns:
            raise KeyError(f"Required column '{col}' missing from dataframe.")
        features_df[col] = pd.to_numeric(features_df[col], errors="coerce").astype("float64").fillna(0.0)
        logger.debug("Coerced %s to float64 with NaN fill.", col)

    # VWAP (optional)
    if "Vwap" not in features_df or features_df["Vwap"].isna().all():
        features_df["Vwap"] = features_df["Close"]
        logger.debug("Vwap missing/all-NaN; defaulted to Close.")
    else:
        features_df["Vwap"] = pd.to_numeric(features_df["Vwap"], errors="coerce").astype("float64").fillna(
            features_df["Close"]
        )

    # -----------------------------------------------------------------
    # Relative Volume
    # -----------------------------------------------------------------
    rv_period = int(config.get("relative_volume_period", 30))
    vol_mean = features_df["Volume"].rolling(window=rv_period, min_periods=1).mean().replace(0.0, 1.0)
    features_df["relative_volume"] = (features_df["Volume"] / vol_mean).fillna(1.0)
    logger.debug("Computed relative_volume.")

    # -----------------------------------------------------------------
    # VWAP derived
    # -----------------------------------------------------------------
    features_df["distance_from_vwap_pct"] = (
        (features_df["Close"] - features_df["Vwap"]) / features_df["Vwap"].replace(0.0, np.nan)
    ).fillna(0.0)

    features_df["vwap_slope"] = (
        features_df["Vwap"]
        .rolling(window=10, min_periods=1)
        .apply(lambda x: _slope(pd.Series(x)), raw=False)
        .fillna(0.0)
    )
    logger.debug("Computed VWAP features.")

    # -----------------------------------------------------------------
    # ROC features
    # -----------------------------------------------------------------
    for p in (30, 60, 120):
        features_df[f"roc_{p}"] = features_df["Close"].pct_change(periods=p).fillna(0.0)
    logger.debug("Computed ROC features.")

    # -----------------------------------------------------------------
    # RSI & SMA(RSI)
    # -----------------------------------------------------------------
    rsi = _rsi(features_df["Close"], length=14)
    features_df["rsi_14"] = _to_series(rsi, index=features_df.index).astype("float64").fillna(0.0)
    features_df["rsi_14_sma_5"] = (
        features_df["rsi_14"].rolling(window=5, min_periods=1).mean().astype("float64").fillna(0.0)
    )
    logger.debug("Computed RSI and its SMA.")

    # -----------------------------------------------------------------
    # MACD
    # -----------------------------------------------------------------
    macd_df = _macd(features_df["Close"])
    if macd_df is not None and not macd_df.empty:
        features_df["macd_line"] = macd_df["MACD_12_26_9"].astype("float64").fillna(0.0)
        features_df["macd_signal"] = macd_df["MACDs_12_26_9"].astype("float64").fillna(0.0)
        features_df["macd_hist"] = macd_df["MACDh_12_26_9"].astype("float64").fillna(0.0)
        features_df["macd_hist_diff"] = features_df["macd_hist"].diff().fillna(0.0)
    else:
        features_df["macd_line"] = 0.0
        features_df["macd_signal"] = 0.0
        features_df["macd_hist"] = 0.0
        features_df["macd_hist_diff"] = 0.0
        logger.warning("MACD calculation returned None/empty - defaulted to zeros.")
    logger.debug("Computed MACD features.")

    # -----------------------------------------------------------------
    # MFI
    # -----------------------------------------------------------------
    mfi = _mfi(
        high=features_df["High"],
        low=features_df["Low"],
        close=features_df["Close"],
        volume=features_df["Volume"],
        length=14,
    )
    features_df["mfi_14"] = _to_series(mfi, index=features_df.index).astype("float64").fillna(0.0)
    logger.debug("Computed MFI.")

    # -----------------------------------------------------------------
    # OBV & slope
    # -----------------------------------------------------------------
    obv = _obv(features_df["Close"], features_df["Volume"])
    features_df["obv"] = _to_series(obv, index=features_df.index).astype("float64").fillna(0.0)
    features_df["obv_slope"] = (
        features_df["obv"]
        .rolling(window=10, min_periods=1)
        .apply(lambda x: _slope(pd.Series(x)), raw=False)
        .fillna(0.0)
    )
    logger.debug("Computed OBV and slope.")

    # -----------------------------------------------------------------
    # Intraday log returns (keep for consumers)
    # -----------------------------------------------------------------
    intraday_log_returns = np.log1p(features_df["Close"].pct_change()).astype("float64").fillna(0.0)
    features_df["intraday_log_returns"] = intraday_log_returns

    # -----------------------------------------------------------------
    # Volatility Surprise (GARCH)
    # -----------------------------------------------------------------
    try:
        if daily_log_returns is None:
            if not isinstance(features_df.index, pd.DatetimeIndex):
                raise ValueError("df.index must be a DatetimeIndex to auto-build daily_log_returns.")
            dly = (
                features_df["Close"]
                .resample("D")
                .last()
                .pct_change()
                .apply(lambda x: np.log1p(x))
                .dropna()
                .astype("float64")
            )
            daily_log_returns = dly
        else:
            daily_log_returns = _to_series(daily_log_returns).fillna(0.0).astype("float64")

        if daily_log_returns.var() < 1e-10:
            daily_log_returns = daily_log_returns + np.random.normal(0.0, 1e-8, len(daily_log_returns))

        intraday_lr = intraday_log_returns.copy()
        if intraday_lr.var() < 1e-10:
            intraday_lr = intraday_lr + np.random.normal(0.0, 1e-8, len(intraday_lr))

        forecast = get_mc_garch_volatility_forecast(daily_log_returns, intraday_lr)
        features_df["volatility_surprise"] = 0.0

        if forecast is not None and len(forecast) > 0:
            ts = forecast.index[0] if hasattr(forecast, "index") and len(forecast.index) > 0 else None
            if ts is not None:
                realized = intraday_lr.rolling(window=21, min_periods=1).std()
                realized_at = realized.asof(ts - pd.Timedelta(minutes=1))
                fval = float(forecast.iloc[0])
                surprise = ((realized_at or 0.0) - fval) / fval if fval > 0 else 0.0
                # align or drop on the last row
                if ts in features_df.index:
                    features_df.loc[ts, "volatility_surprise"] = surprise
                else:
                    features_df.iloc[-1, features_df.columns.get_loc("volatility_surprise")] = surprise
    except Exception as e:  # pragma: no cover
        logger.warning("Could not calculate GARCH feature: %s", e)
        features_df["volatility_surprise"] = 0.0

    # -----------------------------------------------------------------
    # Fast VRA (10-day window)
    # -----------------------------------------------------------------
    try:
        if not isinstance(features_df.index, pd.DatetimeIndex):
            raise ValueError("DatetimeIndex required to compute fast VRA.")
        daily_pct = features_df["Close"].resample("D").last().pct_change()
        daily_rv_fast = daily_pct.rolling(10).std() * np.sqrt(252)
        daily_rv_fast = daily_rv_fast.dropna()
        if not daily_rv_fast.empty:
            vra_scores_fast = get_volatility_regime_anomaly(daily_rv_fast)
            features_df["vra_score_fast"] = float(vra_scores_fast.iloc[-1])
        else:
            features_df["vra_score_fast"] = 0.0
    except Exception as e:  # pragma: no cover
        logger.warning("Could not calculate fast VRA: %s", e)
        features_df["vra_score_fast"] = 0.0

    # -----------------------------------------------------------------
    # Final sanitize
    # -----------------------------------------------------------------
    features_df.replace([np.inf, -np.inf], 0.0, inplace=True)
    features_df.fillna(0.0, inplace=True)

    logger.debug("Final NaN fill completed. Feature calculation done.")
    return features_df
