"""
Implements the triple-barrier labeling method for financial time series.

This module provides the logic for generating labels (1 for a drop, 0 otherwise)
based on whether price hits a profit-take, stop-loss, or time-based barrier first.
The v4 implementation uses dynamic, volatility-adjusted barriers.
"""
import numpy as np
import pandas as pd  # type: ignore
import pandas_ta as ta  # type: ignore


def get_volatility_adjusted_barriers(
    close_prices: pd.Series, atr_period: int = 14, atr_multiplier: float = 2.0
) -> pd.Series:
    """Calculates the size of the price move for the barrier based on ATR.

    Args:
        close_prices (pd.Series): Series of close prices.
        atr_period (int): The lookback period for calculating ATR.
        atr_multiplier (float): The multiplier to apply to the ATR value.

    Returns:
        pd.Series: A series containing the dynamic barrier size for each timestamp.
    """

    # We need high/low prices to calculate ATR, so we assume they are available
    # in a DataFrame that `close_prices` came from. This is a common pattern.
    # For this function, we'll assume a proxy for H/L if not available.
    # In the main data pipeline, we will pass the full OHLCV DataFrame.
    # This is a simplified version for modularity.
    high_proxy = close_prices * 1.01
    low_proxy = close_prices * 0.99

    atr = ta.atr(high=high_proxy, low=low_proxy, close=close_prices, length=atr_period)
    atr = atr.fillna(method="bfill")  # Backfill NaNs at the beginning

    return atr * atr_multiplier


def get_triple_barrier_labels(
    prices: pd.Series,
    t_events: pd.Index,
    pt_sl: list,  # [profit_take_multiplier, stop_loss_multiplier]
    target: pd.Series,
    min_ret: float = 0.0,
    num_threads: int = 1,  # Placeholder for future parallelization
    vertical_barrier_times: pd.Series | None = None,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """Generates labels for each event based on the triple-barrier method.

    This is a more robust implementation based on de Prado's work.

    Args:
        prices (pd.Series): Series of prices.
        t_events (pd.Index): Timestamps of the events to be labeled.
        pt_sl (list): A list of two non-negative floats, [pt, sl], where pt
            is the multiplier for the profit-take barrier and sl is for the
            stop-loss. A value of 0 means the barrier is disabled.
        target (pd.Series): A series of volatility estimates for setting barrier
            widths.
        min_ret (float): The minimum return required for the profit-take barrier
            to trigger.
        vertical_barrier_times (pd.Series | None): Timestamps for the vertical
            barrier. If None, no time limit is applied.
        side (pd.Series | None): The side of the bet (1 for long, -1 for short).
            If None, it assumes a short-only (-1) strategy.

    Returns:
        pd.DataFrame: A DataFrame with columns ['ret', 'bin', 't1'], where 'ret' is
            the return, 'bin' is the label (1, -1, or 0), and 't1' is the
            timestamp of the barrier touch.
    """
    if side is None:
        side = pd.Series(-1, index=t_events)  # Assume short-only

    # 1. Get barrier levels
    events = pd.concat({"t1": vertical_barrier_times, "trgt": target, "side": side}, axis=1)
    events = events.reindex(t_events).dropna()

    # Profit-take and stop-loss levels
    pt_level = events["trgt"] * pt_sl[0]
    sl_level = events["trgt"] * pt_sl[1]

    # 2. Find barrier touch times
    out = events[["t1"]].copy(deep=True)
    out["bin"] = 0
    out["ret"] = 0.0

    for loc, t1 in events["t1"].items():
        path_prices = prices.loc[loc:t1]
        price_ret = (path_prices / prices[loc] - 1) * events.at[loc, "side"]

        # Profit take
        pt_touch_idx = path_prices[price_ret > pt_level[loc]].index.min()
        # Stop loss
        sl_touch_idx = path_prices[price_ret < -sl_level[loc]].index.min()

        touch_times = pd.Series([t1, pt_touch_idx, sl_touch_idx]).dropna().sort_values()
        first_touch = touch_times.iloc[0]

        out.at[loc, "t1"] = first_touch
        if first_touch == pt_touch_idx:
            out.at[loc, "bin"] = 1
        elif first_touch == sl_touch_idx:
            out.at[loc, "bin"] = -1  # Stop loss hit

        # Calculate return at first touch
        out.at[loc, "ret"] = (prices[first_touch] / prices[loc] - 1) * events.at[loc, "side"]

    return out
