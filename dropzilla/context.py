"""
Provides application-wide context, including configuration management
and market regime detection.
"""
import pandas as pd
import numpy as np  # <-- THE FIX: Added missing import
from hmmlearn.hmm import GaussianHMM
from .config import DATA_CONFIG

def get_market_regimes(market_data: pd.DataFrame, n_regimes: int = 3) -> pd.Series:
    """
    Identifies market regimes using a Gaussian Hidden Markov Model (HMM).

    Args:
        market_data (pd.DataFrame): DataFrame of market data (e.g., SPY) with 'Close' prices.
        n_regimes (int): The number of hidden states (regimes) to identify.

    Returns:
        pd.Series: A Series with the identified regime for each timestamp.
    """
    # Use daily log returns as the primary observable for the HMM
    log_returns = np.log(market_data['Close']).diff().dropna().to_frame()

    if log_returns.empty:
        return pd.Series(index=market_data.index, dtype=int)

    model = GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(log_returns)
    
    regimes = model.predict(log_returns)
    regime_series = pd.Series(regimes, index=log_returns.index, name="market_regime")

    # Use .ffill() to address the FutureWarning
    return regime_series.reindex(market_data.index).ffill()
