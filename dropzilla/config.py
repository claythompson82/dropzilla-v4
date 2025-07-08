"""
Centralised, typed configuration for Dropzilla v4.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Union, cast        # added Union and cast

from dotenv import load_dotenv
import lightgbm as lgb


# --------------------------------------------------------------------------- #
# 1. Helper – detect CUDA build *safely*
# --------------------------------------------------------------------------- #
def _gpu_available() -> bool:
    """
    Return ``True`` iff this LightGBM build supports CUDA.

    • ``lgb.get_device_name`` only exists in CUDA builds – so we ask for it
      inside ``getattr`` to avoid an ``AttributeError`` in CPU wheels.
    """
    name = getattr(lgb, "get_device_name", lambda *_: "")(0)  # type: ignore[attr-defined]
    return bool(name and "cuda" in name.lower())


# --------------------------------------------------------------------------- #
# 2. Load environment variables
# --------------------------------------------------------------------------- #
load_dotenv()  # .env is optional; silently ignored if absent

POLYGON_API_KEY: Final[str | None] = os.getenv("POLYGON_API_KEY")


# --------------------------------------------------------------------------- #
# 3. Sub-configs
# --------------------------------------------------------------------------- #
DATA_CONFIG: Final[dict[str, Union[int, str]]] = {
    "data_period_days": 365,
    "cache_dir": ".cache",
    "cache_ttl_seconds": 86_400,  # 24 h
}

FEATURE_CONFIG: Final[dict[str, int]] = {
    "rsi_period": 14,
    "mfi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "relative_volume_period": 50,
    "vwap_slope_period": 10,
    "garch_realized_vol_period": 21,
}

MODEL_CONFIG: Final[dict[str, Union[int, bool, str, float]]] = {  # now allows float
    "use_gpu": _gpu_available(),
    "model_filename": "dropzilla_v4_lgbm.pkl",
    "cv_n_splits": 5,
    "cv_embargo_pct": 0.01,                # float is now permitted
    "optimization_max_evals": 50,
}

LABELING_CONFIG: Final[dict[str, Union[int, float]]] = {
    "atr_period": 14,
    "vertical_barrier_minutes": 240,
    "profit_take_atr_multiplier": 2.0,
    "stop_loss_atr_multiplier": 1.5,
}


# --------------------------------------------------------------------------- #
# 4. Ensure cache directory exists (with explicit cast for mypy)
# --------------------------------------------------------------------------- #
cache_root: str = cast(str, DATA_CONFIG["cache_dir"])
Path(cache_root).mkdir(exist_ok=True, parents=True)
