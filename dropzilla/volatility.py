"""
Handles advanced volatility modeling, such as GARCH analysis.

This module implements a Multiplicative Component GARCH (MC-GARCH) model
to accurately forecast intraday volatility by decomposing it into daily,
diurnal, and stochastic components.
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

    # 1. Model the daily component h_t with a standard GARCH(1,1)
    # We use daily returns to capture the day-to-day volatility clustering.
    # We scale by 100 to help the GARCH optimizer converge more reliably.
    daily_garch = arch_model(daily_returns.dropna() * 100, p=1, q=1, vol='Garch', dist='Normal')
    daily_res = daily_garch.fit(disp='off')
    
    # Forecast the next day's variance and scale it back.
    daily_forecast_var = daily_res.forecast(horizon=1).variance.iloc[-1, 0] / 10000

    # 2. Model the diurnal component s_i (the U-shape of intraday volatility)
    intraday_returns_sq = intraday_returns**2
    diurnal_factors = intraday_returns_sq.groupby(intraday_returns_sq.index.time).mean()
    # Normalize so the average diurnal factor is 1, preserving the total variance.
    if diurnal_factors.mean() > 0:
        diurnal_factors /= diurnal_factors.mean()
    
    # Map the calculated diurnal factor to each intraday timestamp.
    s_i = intraday_returns.index.to_series().apply(lambda x: diurnal_factors.get(x.time(), 1.0))

    # 3. Model the stochastic intraday component q_t,i
    # De-seasonalize the intraday returns by removing the diurnal component.
    deasonalized_returns = intraday_returns / (s_i**0.5)
    
    # Fit a GARCH(1,1) to these de-seasonalized residuals.
    intraday_garch = arch_model(deasonalized_returns.dropna() * 100, p=1, q=1, vol='Garch', dist='Normal')
    intraday_res = intraday_garch.fit(disp='off')
    
    # Forecast the next period's stochastic variance and scale it back.
    q_forecast_var = intraday_res.forecast(horizon=1).variance.iloc[-1, 0] / 10000

    # 4. Combine the components for the final forecast
    # The final forecast is the product of the daily, next-period diurnal, and next-period stochastic components.
    final_forecast_var = daily_forecast_var * s_i.iloc[-1] * q_forecast_var
    final_forecast_vol = np.sqrt(final_forecast_var)

    # Create a series with the forecast to be merged back into the main dataframe.
    # We will use this single forecast value for the next period's surprise calculation.
    forecast_series = pd.Series(final_forecast_vol, index=[intraday_returns.index[-1] + pd.Timedelta(minutes=1)])

    return forecast_series
