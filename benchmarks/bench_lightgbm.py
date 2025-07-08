import argparse
import numpy as np
import lightgbm as lgb
import time


def train(params, data, num_round):
    start = time.time()
    lgb.train(params, data, num_boost_round=num_round)
    return time.time() - start


def main():
    parser = argparse.ArgumentParser(description="Benchmark LightGBM CPU vs CUDA")
    parser.add_argument("--nrows", type=float, default=1e6, help="Number of rows")
    parser.add_argument("--ncols", type=int, default=100, help="Number of columns")
    parser.add_argument("--num_round", type=int, default=200, help="Boosting rounds")
    parser.add_argument("--cpu_threads", type=int, default=4, help="CPU threads")
    args = parser.parse_args()

    nrows = int(args.nrows)
    ncols = args.ncols
    num_round = args.num_round

    X = np.random.random((nrows, ncols)).astype(np.float32)
    y = np.random.randint(0, 2, size=nrows)
    data = lgb.Dataset(X, label=y)

    common_params = {"objective": "binary"}

    cpu_params = dict(common_params)
    cpu_params.update({"device_type": "cpu", "nthread": args.cpu_threads})
    cpu_time = train(cpu_params, data, num_round)

    cuda_params = dict(common_params)
    cuda_params.update({"device_type": "cuda"})
    cuda_time = train(cuda_params, data, num_round)

    speedup = cpu_time / cuda_time if cuda_time else float("inf")

    print(f"CPU training time: {cpu_time:.2f}s")
    print(f"CUDA training time: {cuda_time:.2f}s")
    print(f"Speed-up: {speedup:.2f}x")


if __name__ == "__main__":
    main()

