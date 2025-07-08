"""
Unit tests for the feature engineering module.
"""
import pandas as pd
import numpy as np

# Make sure to use the correct path if your test file is in the tests/ directory
from dropzilla.features import calculate_features

def test_calculate_features_basic():
    """
    Tests that calculate_features runs without error and adds new columns.
    """
    # Create 60 rows of sample OHLCV data
    periods = 60
    index = pd.to_datetime(
        pd.date_range("2023-01-01", periods=periods, freq="min")
    )
    data = {
        "Open": pd.Series(range(periods), index=index, dtype=float),
        "High": pd.Series(range(periods), index=index, dtype=float) + 0.5,
        "Low": pd.Series(range(periods), index=index, dtype=float) - 0.5,
        "Close": pd.Series(range(periods), index=index, dtype=float),
        "Volume": pd.Series([100] * periods, index=index, dtype=float),
    }
    df = pd.DataFrame(data, index=index)

    # --- THE FIX ---
    # Create mock data for the new required arguments
    mock_daily_returns = pd.Series(np.random.randn(2), index=pd.to_datetime(pd.date_range("2023-01-01", periods=2, freq="D")))
    mock_tick_data = pd.DataFrame() # Can be empty for this test
    # --- END FIX ---

    # Call the function with the required arguments
    result = calculate_features(df, mock_tick_data, mock_daily_returns)

    # Assertions
    assert isinstance(result, pd.DataFrame)
    assert 'relative_volume' in result.columns
    assert 'vwap' in result.columns
    assert not result.empty
