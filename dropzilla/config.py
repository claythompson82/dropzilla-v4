"""
Centralized configuration for Dropzilla v4.

This file stores parameters, API keys, and other configuration settings.
"""

POLYGON_API_KEY: str = "YOUR_DEFAULT_KEY_HERE"

DATA_CONFIG: dict = {
    "data_period_days": 5,  # Number of days of data to fetch
}

MODEL_CONFIG: dict = {
    "cv_n_splits": 3,
    "cv_embargo_pct": 0.01,
    "optimization_max_evals": 2,
}

LABELING_CONFIG: dict = {
    "vertical_barrier_minutes": 60,
}

FEATURE_CONFIG: dict = {
    "some_feature_param": 1,
}

