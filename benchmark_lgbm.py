# benchmark_lgbm.py
import argparse, json, os, time, lightgbm as lgb
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
parser.add_argument("--rows",   type=int, default=6_000_000)
parser.add_argument("--cols",   type=int, default=180)
parser.add_argument("--iters",  type=int, default=300)
parser.add_argument("--max_bin",type=int, default=63)
parser.add_argument("--seed",   type=int, default=42)
args = parser.parse_args()

data_dir = Path("data"); data_dir.mkdir(exist_ok=True)
bin_path = data_dir / f"train_{args.rows}x{args.cols}_{args.max_bin}.bin"

# ----------------------------------------------------------------------
# 0️⃣  Build (or load) a binary dataset *once* – time excluded
# ----------------------------------------------------------------------
if bin_path.exists():
    dataset_source = "cached"
    dtrain = lgb.Dataset(str(bin_path), params={"max_bin": args.max_bin})
else:
    rng  = np.random.default_rng(args.seed)
    X    = rng.random((args.rows, args.cols), dtype=np.float32)
    y    = rng.integers(0, 2, size=args.rows)
    dtrain = lgb.Dataset(X, label=y, params={"max_bin": args.max_bin})
    dtrain.save_binary(str(bin_path))
    dataset_source = "generated"

# ----------------------------------------------------------------------
# 1️⃣  GPU-/CPU-only parameters
# ----------------------------------------------------------------------
common = dict(
    objective         = "binary",
    metric            = "auc",
    max_bin           = args.max_bin,
    force_col_wise    = True,                     # skip 40 s auto-test 
    num_leaves        = 255,
    learning_rate     = 0.05,
    feature_fraction  = 1.0,                      # keep identical workload
    bagging_fraction  = 1.0,
    num_threads       = os.cpu_count()//2,        # physical cores only :contentReference[oaicite:5]{index=5}
    verbose           = -1
)
if args.device == "cuda":
    common.update(dict(device_type="cuda",
                       gpu_device_id=0,
                       gpu_use_dp=False))

# ----------------------------------------------------------------------
# 2️⃣  Time the training loop only
# ----------------------------------------------------------------------
tic = time.perf_counter()
booster = lgb.train(common, dtrain, num_boost_round=args.iters)
elapsed = round(time.perf_counter() - tic, 2)

# ----------------------------------------------------------------------
# 3️⃣  Emit result in a well-known file for the shell script
# ----------------------------------------------------------------------
out = Path(f"/tmp/{args.device}.json")
out.write_text(json.dumps({"device": args.device,
                           "seconds": elapsed,
                           "dataset_source": dataset_source,
                           "status": "ok"}))
print(json.dumps({"device": args.device, "seconds": elapsed,
                  "dataset_source": dataset_source}))
