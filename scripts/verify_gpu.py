import numpy as np, lightgbm as lgb

try:
    data = lgb.Dataset(np.random.rand(2, 2), label=[0, 1])
    lgb.train(
        {
            "device_type": "cuda",   # CUDA learner ▶ OpenCL learner needs 'gpu'
            "objective":   "binary",
            "verbose":     -1
        },
        data,
        num_boost_round=1
    )
    print("LightGBM built with CUDA ✅")
except lgb.basic.LightGBMError as err:
    print("❌ CUDA learner unavailable:", err)
    raise SystemExit(1)
