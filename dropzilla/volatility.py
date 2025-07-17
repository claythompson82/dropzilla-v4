# Filename: dropzilla/volatility.py
"""
Handles advanced volatility modeling, such as GARCH analysis and VRA.
"""
import numpy as np
import pandas as pd
from arch import arch_model

def get_mc_garch_volatility_forecast(daily_returns: pd.Series,
                                     intraday_returns: pd.Series) -> pd.Series:
    """
    Fits an MC-GARCH model and returns the one-step-ahead conditional volatility forecast.

    Args:
        daily_returns (pd.Series): A pandas Series of daily log returns for the asset.
        intraday_returns (pd.Series): A pandas Series of high-frequency (e.g., 1-minute)
                                      log returns for the asset.

    Returns:
        pd.Series: A pandas Series containing the forecasted intraday volatility.
    """
    if daily_returns.empty or intraday_returns.empty:
        return pd.Series(dtype=float)

    # Add variance check and jitter for low/zero vol
    if np.var(daily_returns) < 1e-6 or np.var(intraday_returns) < 1e-6:
        return pd.Series([0.0], index=[intraday_returns.index[-1] + pd.Timedelta(minutes=1)])  # Early fallback

    daily_returns = daily_returns + np.random.normal(0, 1e-8, len(daily_returns))  # Small jitter
    intraday_returns = intraday_returns + np.random.normal(0, 1e-8, len(intraday_returns))

    # Rescale inputs
    scale_factor = 100
    daily_returns_scaled = daily_returns.dropna() * scale_factor
    intraday_returns_scaled = intraday_returns * scale_factor  # For diurnal/s_i too

    # 1. Daily GARCH on scaled
    daily_garch = arch_model(daily_returns_scaled, p=1, q=1, vol='Garch', dist='Normal')
    daily_res = daily_garch.fit(disp='off', options={'maxiter': 100})  # Cap iterations
    daily_forecast_var = daily_res.forecast(horizon=1).variance.iloc[-1, 0] / (scale_factor ** 2)  # Scale back

    # 2. Diurnal on scaled
    intraday_returns_sq = intraday_returns_scaled**2
    diurnal_factors = intraday_returns_sq.groupby(intraday_returns_sq.index.time).mean()
    if diurnal_factors.mean() > 0:
        diurnal_factors /= diurnal_factors.mean()
    
    s_i = intraday_returns.index.to_series().apply(lambda x: diurnal_factors.get(x.time(), 1.0))

    # 3. Intraday GARCH on scaled deasonalized
    deasonalized_returns = intraday_returns_scaled / (s_i**0.5)
    intraday_garch = arch_model(deasonalized_returns.dropna(), p=1, q=1, vol='Garch', dist='Normal')
    intraday_res = intraday_garch.fit(disp='off', options={'maxiter': 100})
    q_forecast_var = intraday_res.forecast(horizon=1).variance.iloc[-1, 0] / (scale_factor ** 2)

    # 4. Combine the components for the final forecast
    final_forecast_var = daily_forecast_var * s_i.iloc[-1] * q_forecast_var
    final_forecast_vol = np.sqrt(final_forecast_var)

    forecast_series = pd.Series(final_forecast_vol, index=[intraday_returns.index[-1] + pd.Timedelta(minutes=1)])
    return forecast_series

def get_volatility_regime_anomaly(daily_realized_vol: pd.Series,
                                  short_window: int = 10,
                                  long_window: int = 180) -> pd.Series:
    """
    Calculates the Volatility Regime Anomaly (VRA) score.

    The VRA is the percentile rank of the recent short-term average volatility
    within the long-term historical distribution of volatility.

    Args:
        daily_realized_vol (pd.Series): A Series of daily realized volatility.
        short_window (int): The lookback period for the short-term average.
        long_window (int): The lookback period for the historical distribution.

    Returns:
        pd.Series: A Series of VRA scores (0-100).
    """
    if daily_realized_vol.empty or len(daily_realized_vol) < long_window:
        return pd.Series(dtype=float)

    short_term_vol = daily_realized_vol.rolling(window=short_window).mean()
    
    vra_scores = []
    # Start loop after the initial burn-in period for the long window
    for i in range(long_window, len(daily_realized_vol)):
        current_short_vol = short_term_vol.iloc[i]
        historical_dist = daily_realized_vol.iloc[i-long_window:i]
        
        if pd.isna(current_short_vol):
            vra_scores.append(np.nan)
            continue
            
        # Calculate percentile rank of current value against the historical distribution
        percentile = np.sum(historical_dist < current_short_vol) / len(historical_dist) if len(historical_dist) > 0 else np.nan
        vra_scores.append(percentile * 100)
        
    return pd.Series(vra_scores, index=daily_realized_vol.index[long_window:], name="vra_score")
