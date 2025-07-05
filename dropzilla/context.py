"""Market context utilities such as regime detection."""

from __future__ import annotations

import pandas as pd
import numpy as np


def get_market_regimes(df: pd.DataFrame, fast_period: int = 20, slow_period: int = 50) -> pd.Series:
    """Simple market regime classifier based on moving average crossover.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least a ``Close`` column with a ``DatetimeIndex``.
    fast_period : int, optional
        Window for the fast moving average, by default 20.
    slow_period : int, optional
        Window for the slow moving average, by default 50.

    Returns
    -------
    pd.Series
        Series of ``1`` for bullish regime and ``-1`` for bearish regime,
        indexed the same as ``df``.
    """
    if df.empty:
        return pd.Series(dtype=float)

    fast_ma = df['Close'].rolling(window=fast_period, min_periods=fast_period).mean()
    slow_ma = df['Close'].rolling(window=slow_period, min_periods=slow_period).mean()
    regime = np.where(fast_ma > slow_ma, 1, -1)
    return pd.Series(regime, index=df.index, name='market_regime')
