"""Verify GPU-enabled LightGBM build."""

import sys

try:
    import cupy  # noqa: F401
    import lightgbm as lgb
except Exception as exc:  # pragma: no cover - simple sanity check
    print(f"Import failed: {exc}")
    sys.exit(1)

if hasattr(lgb, "DaskDeviceQuantileDMatrix"):
    print("LightGBM built with CUDA ✅")
else:
    print("LightGBM built without CUDA ❌")
    sys.exit(1)
