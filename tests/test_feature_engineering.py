import pandas as pd

from dropzilla.features import calculate_features


def test_calculate_features_basic():
    # Create 60 rows of sample OHLCV data
    periods = 60
    index = pd.date_range("2023-01-01", periods=periods, freq="T")
    data = {
        "Open": pd.Series(range(periods), index=index, dtype=float),
        "High": pd.Series(range(periods), index=index, dtype=float) + 0.5,
        "Low": pd.Series(range(periods), index=index, dtype=float) - 0.5,
        "Close": pd.Series(range(periods), index=index, dtype=float),
        "Volume": pd.Series([100] * periods, index=index, dtype=float),
    }
    df = pd.DataFrame(data, index=index)

    result = calculate_features(df)

    assert "relative_volume" in result.columns
    assert "vwap" in result.columns
    assert "distance_from_vwap_pct" in result.columns
    assert "vwap_slope" in result.columns
    assert result["relative_volume"].iloc[-1] == 1

