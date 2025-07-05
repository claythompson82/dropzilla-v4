"""
Centralized configuration for Dropzilla v4.

This file stores all parameters, paths, API keys, and other
configuration settings for the system.
"""
import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

# --- API Configuration ---
# It's best practice to load secrets from environment variables
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "YOUR_DEFAULT_KEY_HERE")

# --- Data Configuration ---
DATA_CONFIG = {
    "cache_dir": ".cache",
    "cache_ttl_seconds": 60 * 60 * 4, # 4 hours
    "data_period_days": 365, # Lookback period for fetching training data
}

# --- Labeling Configuration (Phase 2) ---
LABELING_CONFIG = {
    "atr_period": 14,
    "profit_take_atr_multiplier": 2.0,
    "stop_loss_atr_multiplier": 1.5,
    "vertical_barrier_minutes": 240, # 4 hours
}

# --- Feature Engineering Configuration (Phase 2) ---
FEATURE_CONFIG = {
    "relative_volume_period": 50,
    "vwap_slope_period": 10,
    "roc_period": 60,
}

# --- Model & Optimization Configuration (Phase 1) ---
MODEL_CONFIG = {
    "model_filename": "dropzilla_v4_lgbm.pkl",
    "optimization_max_evals": 50,
    "cv_n_splits": 5,
    "cv_embargo_pct": 0.01,
}
