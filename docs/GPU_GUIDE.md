# GPU Quick-Start

This guide explains how to verify your LightGBM installation uses CUDA and how to run a GPU-accelerated back-test.

## Verify the build

Run the verification script:

```bash
python scripts/verify_gpu.py
```

If CUDA support is detected the script prints:
`LightGBM built with CUDA ✅`

## Run a GPU back-test

Execute the back-test with GPU enabled:

```bash
python scripts/run_backtest.py --tickers AAPL MSFT --use_gpu
```

When a CUDA-enabled wheel is installed, LightGBM logs will show `CUDA`.

## Environment tips

- Works well under WSL 2 with NVIDIA drivers and CUDA 12.9.
- Use `max_bin=255` for fastest GPU training.
- Confirm `POLYGON_API_KEY` is set before running the scripts.
