import numpy as np, lightgbm as lgb

try:
    print("LightGBM shared lib:", lgb.__file__)
    print("CUDA device name   :", lgb.get_device_name(0))  # raises if CPU build
    ds = lgb.Dataset(np.random.rand(2, 2), label=[0, 1])
    lgb.train({"device_type": "cuda", "objective": "binary", "verbose": -1},
              ds, num_boost_round=1)
    print("✅  CUDA learner works")
except (lgb.basic.LightGBMError, AttributeError) as e:
    raise SystemExit(f"❌  GPU unavailable: {e}")
