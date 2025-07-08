"""
Benchmark LightGBM on CPU vs CUDA.

Run defaults:
    python benchmarks/bench_lightgbm.py
Big run:
    python benchmarks/bench_lightgbm.py --nrows 1000000 --ncols 100 --rounds 500
"""
import argparse, time, multiprocessing as mp, numpy as np, lightgbm as lgb

def run(device, X, y, rounds, threads=None):
    params = {"device_type": device, "objective": "binary", "verbose": -1}
    if device == "cpu":
        params["nthread"] = threads or mp.cpu_count()
    start = time.perf_counter()
    lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=rounds)
    return time.perf_counter() - start

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nrows", type=int, default=100_000)
    p.add_argument("--ncols", type=int, default=50)
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--cpu_threads", type=int, default=mp.cpu_count())
    a = p.parse_args()

    X = np.random.rand(a.nrows, a.ncols).astype(np.float32)
    y = np.random.randint(0, 2, size=a.nrows)

    t_cpu  = run("cpu",  X, y, a.rounds, a.cpu_threads)
    t_cuda = run("cuda", X, y, a.rounds)

    print(f"\nCPU  : {t_cpu:7.2f}s ({a.cpu_threads} thr)")
    print(f"CUDA : {t_cuda:7.2f}s")
    print(f"Speed-up: {t_cpu/t_cuda:5.2f}× 🚀")
