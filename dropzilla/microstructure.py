"""
Handles advanced market microstructure analysis, such as order flow imbalance.
"""
import pandas as pd
import numpy as np

def get_order_flow_imbalance(tick_data: pd.DataFrame, window_minutes: int = 30) -> pd.Series:
    """
    Calculates the buy/sell imbalance ratio based on the tick rule.

    This implementation uses the zero-tick rule for higher accuracy, as
    described in the research literature.

    Args:
        tick_data (pd.DataFrame): DataFrame of tick data with 'Price' and 'Volume'.
        window_minutes (int): The rolling window size in minutes.

    Returns:
        pd.Series: A Series of the order flow imbalance ratio, resampled to 1-minute frequency.
    """
    if tick_data.empty:
        return pd.Series(dtype=float)

    # 1. Classify Ticks
    price_diff = tick_data['Price'].diff()
    
    # Initialize tick direction series
    tick_direction = pd.Series(np.nan, index=tick_data.index)
    tick_direction[price_diff > 0] = 1  # Up-tick = buy
    tick_direction[price_diff < 0] = -1 # Down-tick = sell
    
    # Propagate last known tick for zero-tick trades
    tick_direction = tick_direction.fillna(method='ffill').fillna(0)

    # 2. Calculate Directional Volume
    buy_volume = tick_data['Volume'][tick_direction == 1]
    sell_volume = tick_data['Volume'][tick_direction == -1]

    # 3. Resample to 1-minute frequency and calculate rolling sums
    # This gives us the total buy/sell volume in each minute
    buy_vol_minute = buy_volume.resample('1min').sum()
    sell_vol_minute = sell_volume.resample('1min').sum()
    
    # Calculate rolling pressure over the specified window
    rolling_buy_pressure = buy_vol_minute.rolling(window=f'{window_minutes}min').sum()
    rolling_sell_pressure = sell_vol_minute.rolling(window=f'{window_minutes}min').sum()

    # 4. Calculate Imbalance Ratio
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-9
    imbalance_ratio = rolling_buy_pressure / (rolling_sell_pressure + epsilon)
    
    return imbalance_ratio.rename('order_flow_imbalance')

