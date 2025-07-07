"""
Handles analysis of correlation structures, such as PCA and systemic risk.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_systemic_absorption_ratio(returns_df: pd.DataFrame,
                                  n_components: int = 1,
                                  window: int = 60) -> pd.Series:
    """
    Calculates the Systemic Absorption Ratio (SAR) using a rolling PCA.

    SAR measures the proportion of variance explained by the first N principal
    components, indicating risk concentration. A high SAR means risk is systemic.

    Args:
        returns_df (pd.DataFrame): A DataFrame where columns are asset tickers
                                   and rows are daily returns.
        n_components (int): The number of principal components to use for the numerator.
        window (int): The rolling window size in days for the PCA.

    Returns:
        pd.Series: A Series of SAR scores, indexed by date.
    """
    if returns_df.empty or len(returns_df) < window:
        return pd.Series(dtype=float)
        
    scaler = StandardScaler()
    sar_scores = []
    
    # --- THE FIX ---
    # We manually loop through the DataFrame to create rolling windows.
    # This ensures that at each step, we pass a 2D DataFrame to the PCA.
    for i in range(window, len(returns_df)):
        # Get the current window of data (e.g., the last 60 days)
        window_data = returns_df.iloc[i-window:i]
        
        # Drop any columns that might be all NaN in this specific window
        window_data = window_data.dropna(axis=1, how='all')
        
        if window_data.shape[1] < n_components:
            # Not enough assets in this window to perform PCA
            sar_scores.append(np.nan)
            continue

        try:
            # Scale the data and perform PCA
            scaled_data = scaler.fit_transform(window_data)
            pca = PCA(n_components=n_components)
            pca.fit(scaled_data)
            
            # Append the sum of explained variance for the top N components
            sar_scores.append(pca.explained_variance_ratio_.sum())
        except Exception:
            # If PCA fails for any reason on a given window, append NaN
            sar_scores.append(np.nan)
    # --- END FIX ---

    # Create a pandas Series with the correct index
    return pd.Series(sar_scores, index=returns_df.index[window:], name="sar_score")
