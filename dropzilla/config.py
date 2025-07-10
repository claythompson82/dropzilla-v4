"""
Centralized configuration for the Dropzilla v4 application.
"""
import os
from dotenv import load_dotenv
import lightgbm as lgb


def _gpu_available() -> bool:
    """Return True if LightGBM was built with CUDA support."""
    try:
        return "cuda" in lgb.get_device_name(0).lower()
    except Exception:
        return False

# Load environment variables from a .env file if it exists
load_dotenv()

# --- API Configuration ---
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# --- Data Configuration ---
DATA_CONFIG = {
    "data_period_days": 365,
    "cache_dir": ".cache",
    "cache_ttl_seconds": 86400, # 24 hours
}

# --- Feature Engineering Configuration ---
# This dictionary provides all parameters needed by the feature calculation functions.
FEATURE_CONFIG = {
    "rsi_period": 14,
    "mfi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "relative_volume_period": 50,
    "vwap_slope_period": 10,
    "garch_realized_vol_period": 21
}

# --- Model Training Configuration ---
MODEL_CONFIG = {
    "use_gpu": _gpu_available(),
    "model_filename": "dropzilla_v4_lgbm.pkl",
    "cv_n_splits": 5,
    "cv_embargo_pct": 0.01,
    "optimization_max_evals": 50,
    "suffix_gpu": "_gpu",
    "suffix_cpu": "_cpu",
}

# --- Labeling Configuration ---
LABELING_CONFIG = {
    "atr_period": 14,
    "vertical_barrier_minutes": 240, # 4 hours
    "profit_take_atr_multiplier": 2.0,
    "stop_loss_atr_multiplier": 1.5
}
