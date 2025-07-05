"""
Handles market context analysis, such as regime detection.
"""
import numpy as np
import pandas as pd
from hmmlearn import hmm


def get_market_regimes(market_data: pd.DataFrame,
                       n_regimes: int = 2) -> pd.Series:
    """
    Identifies market regimes using a Gaussian Hidden Markov Model (HMM).

    The HMM is trained on daily log returns and realized volatility to classify
    each day into one of several unobserved states (regimes).

    Args:
        market_data (pd.DataFrame): A DataFrame with a DatetimeIndex and a 'Close'
                                    column, typically for a market index like SPY.
        n_regimes (int): The number of regimes to identify (e.g., 2 for
                         Low/High Volatility).

    Returns:
        pd.Series: A Series with the same index as market_data, where the values
                   are the identified regime for each day.
    """
    if market_data.empty:
        return pd.Series(dtype=int)

    # Calculate daily log returns
    log_returns = np.log(market_data['Close'] / market_data['Close'].shift(1))
    
    # Calculate daily realized volatility
    realized_volatility = log_returns.rolling(window=21).std() * np.sqrt(252)

    # Prepare the feature matrix for the HMM
    hmm_features = pd.concat([log_returns, realized_volatility], axis=1).dropna()
    
    if hmm_features.empty:
        return pd.Series(dtype=int)

    # Create and train the Gaussian HMM
    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="diag",
        n_iter=1000,
        random_state=42
    )
    model.fit(hmm_features.values)

    # Predict the hidden states (regimes)
    hidden_states = model.predict(hmm_features.values)

    # Create a Series to return, aligned with the original data's index
    regime_series = pd.Series(hidden_states, index=hmm_features.index, name="market_regime")
    
    # Reindex to match the full input dataframe, forward-filling regimes
    return regime_series.reindex(market_data.index).fillna(method='ffill')
