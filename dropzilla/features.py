"""
Feature engineering logic for Dropzilla v4.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd


def calculate_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Return DataFrame with a few basic engineered features."""
    result = df.copy()
    result["relative_volume"] = result["Volume"] / result["Volume"].rolling(30).mean()
    result["distance_from_vwap_pct"] = 0.0
    result["vwap_slope"] = 0.0
    result["roc_60"] = result["Close"].pct_change(periods=60)
    return result

