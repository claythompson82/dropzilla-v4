"""
bench_lightgbm.py  –  Compare CPU vs CUDA training speed

Examples
--------
# quick sanity (100 k rows, 50 cols, 200 rounds – defaults)
python benchmarks/bench_lightgbm.py

# larger synthetic set
python benchmarks/bench_lightgbm.py --nrows 10_000_000 --ncols 200 --rounds 1000 --cpu_threads 8
"""

import argparse
import multiprocessing as mp
import time

import numpy as np
import lightgbm as lgb


def run(device: str, X: np.ndarray, y: np.ndarray, rounds: int, threads: int | None):
    """Train one LightGBM booster and return wall-time in seconds."""
    params: dict[str, int | float | str] = {
        "device_type": device,          # "cpu"  or "cuda"
        "objective":   "binary",
        "verbose":     -1,
    }

    if device == "cpu":
        params["nthread"] = threads or mp.cpu_count()
    else:  # CUDA tree-learner tuning
        params.update(
            {
                "max_bin":          63,     # smaller histograms → faster GPU
                "gpu_platform_id":  0,
                "gpu_device_id":    0,
                "feature_fraction": 1.0,    # no column sampling
                "bagging_fraction": 1.0,    # no row bagging
                "min_data_in_bin":  1,
            }
        )

    start = time.perf_counter()
    lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=rounds)
    return time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrows", type=int, default=100_000, help="Rows of synthetic data")
    ap.add_argument("--ncols", type=int, default=50, help="Feature columns")
    ap.add_argument("--rounds", type=int, default=200, help="Boosting rounds")
    ap.add_argument("--cpu_threads", type=int, default=mp.cpu_count(), help="Threads for CPU run")
    args = ap.parse_args()

    X = np.random.rand(args.nrows, args.ncols).astype(np.float32)
    y = np.random.randint(0, 2, size=args.nrows)

    t_cpu = run("cpu", X, y, args.rounds, args.cpu_threads)
    t_gpu = run("cuda", X, y, args.rounds, None)

    print("\n===== LightGBM Benchmark =====")
    print(f"Rows  : {args.nrows:,}")
    print(f"Cols  : {args.ncols}")
    print(f"Rounds: {args.rounds}")
    print(f"CPU   : {t_cpu:9.2f}s  ({args.cpu_threads} threads)")
    print(f"CUDA  : {t_gpu:9.2f}s")
    print(f"Speed-up: {t_cpu / t_gpu:5.2f}× 🚀")


if __name__ == "__main__":
    main()
