# LightGBM GPU Benchmark

This document describes how to benchmark LightGBM training speed on CPU versus CUDA.

## Running the Benchmark

1. Activate your GPU-enabled Python environment.
2. Execute the benchmark script:
   ```bash
   python benchmarks/bench_lightgbm.py --nrows 1e6 --ncols 100 --num_round 200 --cpu_threads 4
   ```
   The script prints the wall-time for CPU and CUDA learners and their speed-up factor.

## Sample Results (Clay's RTX 4070 SUPER)

| nrows | ncols | rounds | CPU threads | CPU time | CUDA time | Speed-up |
|------:|------:|-------:|------------:|---------:|----------:|---------:|
| 1e6   | 100   | 200    | 4           | 45.2s    | 8.2s      | 5.5×     |

Tuning flags used: `max_bin=255` (default) and `gpu_platform_id=0` for CUDA.

