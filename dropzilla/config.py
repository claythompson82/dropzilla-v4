# Filename: dropzilla/config.py
"""
Centralised configuration for the Dropzilla v4 application.

Highlights
----------
* Sensible, production-friendly defaults (e.g., rel_vol_threshold=1.5, not 15)
* Liquidity guard moved to DATA_VALIDATION_CFG and lowered to a practical baseline
* Scanner gains an optional minute confirmation window (confirm_window_minutes)
* Data client knobs for retries / backoff / HTTP timeouts
* Feature config includes every knob used by features.py (incl. fast VRA + ROC periods)
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
import lightgbm as lgb

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _gpu_available() -> bool:
    """Return True iff LightGBM was compiled with CUDA support."""
    try:
        return "cuda" in lgb.get_device_name(0).lower()
    except Exception:  # pragma: no cover  (LightGBM not initialised / no GPU)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Environment variables
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()  # read “.env” if present

POLYGON_API_KEY: str = os.getenv("POLYGON_API_KEY", "").strip()

# Fail fast option (toggle to True if you want to hard-stop when the key is missing)
REQUIRE_POLYGON_KEY: bool = False
if REQUIRE_POLYGON_KEY and not POLYGON_API_KEY:
    raise RuntimeError(
        "POLYGON_API_KEY not set. Create a .env file or export it in your environment."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Data acquisition
# ──────────────────────────────────────────────────────────────────────────────
DATA_CONFIG = {
    "data_period_days":   365,
    "cache_dir":          ".cache",
    "cache_ttl_seconds":  86_400,   # 24h
    # polygon-data client knobs
    "max_retries":        3,
    "retry_backoff_sec":  1.5,
    "http_timeout":       10,
}

# ──────────────────────────────────────────────────────────────────────────────
# Data validation (v4.1+)
# ──────────────────────────────────────────────────────────────────────────────
DATA_VALIDATION_CFG = {
    "min_data_points":    60,      # lowered from 120 to be more inclusive
    "liquidity_min_vol":  250.0,   # average minute-volume baseline cut (prevents dead tickers)
}

# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_CONFIG = {
    # Momentum / oscillators
    "rsi_period":                 14,
    "mfi_period":                 14,
    "roc_periods":               [30, 60, 120],  # used to generate roc_30/60/120
    # MACD
    "macd_fast":                  12,
    "macd_slow":                  26,
    "macd_signal":                 9,
    # Volume / VWAP
    "relative_volume_period":     50,
    "vwap_slope_period":          10,
    "obv_slope_period":           10,
    # Volatility related
    "garch_realised_vol_period":  21,
    "vra_fast_period":            10,   # fast VRA window
}

# ──────────────────────────────────────────────────────────────────────────────
# Model training
# ──────────────────────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "use_gpu":              _gpu_available(),
    "model_filename":       "dropzilla_v4_lgbm.pkl",
    "cv_n_splits":          5,
    "cv_embargo_pct":       0.01,
    "optim_max_evals":      50,
    "suffix_gpu":          "_gpu",
    "suffix_cpu":          "_cpu",
    # LightGBM defaults (override in training if needed)
    "lgb_params": {
        "num_leaves":        64,
        "learning_rate":     0.05,
        "n_estimators":      2000,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.0,
        "reg_lambda":        0.0,
        "objective":        "binary",
        "metric":           ["auc", "binary_logloss"],
        "device_type":      "gpu" if _gpu_available() else "cpu",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Event‑labelling
# ──────────────────────────────────────────────────────────────────────────────
LABELING_CONFIG = {
    "atr_period":                   14,
    "vertical_barrier_minutes":    240,   # 4h
    "profit_take_atr_multiplier":  2.0,
    "stop_loss_atr_multiplier":    1.5,
}

# ──────────────────────────────────────────────────────────────────────────────
# Auto‑scanner (intraday universe)
# ──────────────────────────────────────────────────────────────────────────────
SCANNER_CONFIG = {
    # scheduling
    "interval_min":            10,     # minutes between scan passes
    "start_buffer_min":        20,     # wait X min after open before first pass

    # price filters
    "min_price":               1.0,    # ignore sub-penny/illiquid junk
    "max_price":               500.0,

    # volume / liquidity filters
    "rel_vol_threshold":       1.5,    # min relative-volume multiple (so-far vs baseline)
    "confirm_window_minutes":  10,     # 0 to disable minute re-check stage
    "min_bars":                200,    # if you need a minimum number of minute bars for features

    # polygon / infra
    "max_calls_per_min":       100,    # Polygon Starter quota
    "polygon_delay_minutes":   16,     # 15-min embargo + 1 min cushion
    "precompute_days":         50,     # grouped-daily sessions to compute baselines

    # logging
    "log_level":              "INFO",  # DEBUG / INFO / WARN / ERROR
    "signal_fade_misses":       3,     # retain signals until N consecutive scan misses
}
