"""
Implements the triple-barrier labeling method for financial time series.

This module provides the logic for generating labels (1 for a drop, 0 otherwise)
based on whether price hits a profit-take, stop-loss, or time-based barrier first.
The v4 implementation uses dynamic, volatility-adjusted barriers.
"""
import numpy as np
import pandas as pd
import pandas_ta as ta

def get_volatility_adjusted_barriers(close_prices: pd.Series,
                                     atr_period: int = 14,
                                     atr_multiplier: float = 2.0) -> pd.Series:
    """
    Calculates the size of the price move for the barrier based on ATR.

    Args:
        close_prices (pd.Series): Series of close prices.
        atr_period (int): The lookback period for calculating ATR.
        atr_multiplier (float): The multiplier to apply to the ATR value.

    Returns:
        pd.Series: A series containing the dynamic barrier size for each timestamp.
    """
    # This function is a placeholder for a more robust implementation
    # that would ideally receive the full OHLC DataFrame.
    high_proxy = close_prices * 1.01
    low_proxy = close_prices * 0.99
    
    atr = ta.atr(high=high_proxy, low=low_proxy, close=close_prices, length=atr_period)
    atr = atr.fillna(method='bfill')
    
    return atr * atr_multiplier

def get_triple_barrier_labels(
    prices: pd.Series,
    t_events: pd.Index,
    pt_sl: list,
    target: pd.Series,
    min_ret: float = 0.0,
    num_threads: int = 1,
    vertical_barrier_times: pd.Series = None,
    side: pd.Series = None
) -> pd.DataFrame:
    """
    Generates labels for each event based on the triple-barrier method.
    This is a more robust implementation based on de Prado's work.

    Args:
        prices (pd.Series): Series of prices for a SINGLE instrument.
        t_events (pd.Index): Timestamps of the events to be labeled.
        pt_sl (list): A list of two non-negative floats, [pt, sl].
        target (pd.Series): A series of volatility estimates for setting barrier widths.
        min_ret (float): The minimum return required for the profit-take barrier.
        vertical_barrier_times (pd.Series): Timestamps for the vertical barrier.
        side (pd.Series): The side of the bet (1 for long, -1 for short).

    Returns:
        pd.DataFrame: A DataFrame with columns ['ret', 'bin', 't1'].
    """
    if side is None:
        side = pd.Series(-1, index=t_events)

    events = pd.concat({'t1': vertical_barrier_times, 'trgt': target, 'side': side}, axis=1)
    events = events.reindex(t_events).dropna()

    pt_level = events['trgt'] * pt_sl[0]
    sl_level = events['trgt'] * pt_sl[1]

    out = events[['t1']].copy(deep=True)
    out['bin'] = 0
    out['ret'] = 0.

    for loc, t1_event in events.iterrows():
        t1 = t1_event['t1']
        
        # Slice the path of prices from the event start to the vertical barrier.
        # This slice will automatically end at the last available price if t1 is out of bounds.
        path_prices = prices.loc[loc:t1]
        
        if path_prices.empty:
            continue

        price_ret = (path_prices / prices[loc] - 1) * t1_event['side']

        pt_touch_idx = path_prices[price_ret > pt_level[loc]].index.min()
        sl_touch_idx = path_prices[price_ret < -sl_level[loc]].index.min()

        touch_times = pd.Series([t1, pt_touch_idx, sl_touch_idx]).dropna().sort_values()
        first_touch = touch_times.iloc[0]
        
        out.at[loc, 't1'] = first_touch
        if first_touch == pt_touch_idx:
            out.at[loc, 'bin'] = 1
        elif first_touch == sl_touch_idx:
            out.at[loc, 'bin'] = -1

        # --- THE FIX ---
        # If the barrier touch was the vertical barrier (t1), its timestamp might be
        # outside of market hours and thus not in the `prices` index.
        # In that case, we use the price of the *last available bar* in our path.
        # Otherwise, we use the price at the exact touch time.
        if first_touch == t1:
            ret_price = path_prices.iloc[-1]
        else:
            ret_price = prices[first_touch]
        
        out.at[loc, 'ret'] = (ret_price / prices[loc] - 1) * t1_event['side']

    return out
