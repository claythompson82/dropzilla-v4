Dropzilla v4: An Institutional‑Grade Intraday Signal Engine

Last Updated: July 9 2025GPU ✔ | CUDA speed‑up ×5.5 · Quick GPU guide

1. Project Overview & Status

This repository contains the source code for Dropzilla v4, a proprietary signal engine that identifies high‑conviction, short‑biased trading opportunities in liquid US equities on an intraday (minutes‑to‑hours) horizon.

As of July 2025 the core research and engineering work is complete and scientifically validated. All metrics were produced with a Purged K‑Fold CV to eliminate look‑ahead bias.

Metric

Value

ROC AUC (broad universe)

0.7009

ROC AUC (SEDG & SYM sanity test)

0.6718

Profit Factor @ 65 % conf.

1.15

Win Rate

53.93 %

Annualised Sharpe

0.98

2. Development Journey

Foundational Rectification (7 Jul 2025) – replaced random splits with PurgedKFold.

Context & Edge (7 Jul 2025) – added VRA & SAR; hit 0.7009 AUC.

Meta‑Model Pivot – switched to deterministic multi‑factor confidence score.

Economic Validation – back‑tested → optimal 65 % threshold.

3. System Architecture

3.1 Context‑Awareness Engine

Gaussian HMM market‑regime detector (SPY).

VRA – Volatility regime anomaly.

SAR – Systemic absorption ratio.

3.2 Primary Model

LightGBM, device = cuda (after GPU upgrade).

Price / volume derivatives + contextual features.

3.3 Conviction Engine

Deterministic multi‑factor score (probability, regime, volume, stability).

4. Path Forward

Package modules → /dropzilla namespace.

Live deployment script run_live.py.

GUI dashboard (Tkinter/Flet) for real‑time monitoring.

5. Installation (CPU‑only and GPU)

5.1 Prerequisites

Component

Version

Ubuntu 22.04 LTS (native or WSL2)

 

Python

3.12+

NVIDIA driver (Windows host)

575.xx +

CUDA toolkit (inside WSL)

12.0+

nvcc compatible GCC

12.x

Windows 11 + WSL GPU caveat  Do not install a second Linux GPU driver; WSL proxies the Windows driver via /usr/lib/wsl/lib/libcuda.so.

5.2 Quick CPU‑only setup

git clone <repo>
cd dropzilla-v4
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # pulls LightGBM CPU wheel
pip install -e .                     # project itself

6. GPU Installation – Definitive Checklist

 The steps below reproduce the exact build that finally succeeded after a long debugging session on 9 Jul 2025.

conda create -n dropzilla python=3.12
conda activate dropzilla
conda install -c conda-forge "lightgbm>=4.6.0=*cuda*"  # auto‑selects CUDA build
pip install -r requirements.txt  # installs rest of deps (will skip LightGBM)

Pros: 3‑minute install, no compiler dancing.Cons: conda env instead of venv.

# 0  Activate your venv inside dropzilla‑v4
cd ~/dropzilla-v4 && source .venv/bin/activate

# 1  Remove any pre‑installed LightGBM wheel
pip uninstall -y lightgbm

# 2  Clone source **with submodules**
rm -rf ~/LightGBM-src
git clone --recursive --branch v4.6.0 \
    https://github.com/microsoft/LightGBM.git  ~/LightGBM-src
cd ~/LightGBM-src

# 3  nvcc ↔ GCC handshake (CUDA 12 hates GCC 13)
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
printf 'set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")\n' > ~/cuda_allow.cmake

# 4  One‑shot build & install (creates wheel then pip installs)
python3 ./build.py --cuda -- \
    -DCMAKE_TOOLCHAIN_FILE=$HOME/cuda_allow.cmake \
    -DCMAKE_BUILD_TYPE=Release

# 5  Smoke test
python - <<'PY'
import ctypes, lightgbm, pathlib, os
so = pathlib.Path(lightgbm.__file__).parent/'lib'/'lib_lightgbm.so'
ctypes.CDLL(os.fspath(so))
print('✅  lib_lightgbm.so loaded')
print('Backend →', 'CUDA' if not hasattr(ctypes.CDLL(os.fspath(so)), 'LGBM_GetGPUInfo') else 'OpenCL')
PY

Why this works

build.py (preferred over build-python.sh) handles both CMake + wheel.

Submodules (fast_double_parser, fmt, compute) are fetched via --recursive – no more missing‑header errors.

The toolchain file injects -allow-unsupported-compiler so nvcc accepts GCC 12.

CUDA architectures auto‑detected → includes sm_89 for Ada, so no -DCUDA_ARCH_BIN needed.

Common pitfalls & fixes

Symptom

Root cause

Fix

fatal: header fast_double_parser.h not found

Submodules missing

re‑clone with --recursive

Invalid ELF header when cdll.LoadLibrary

You copied __init__.py path, not .so

load lib/lib_lightgbm.so

LGBM_GetGPUInfo missing but expect OpenCL

You built CUDA (device=cuda)

use device='cuda' params – that symbol is OpenCL only

nvcc fatal: unsupported gcc

Ubuntu default gcc‑13

sudo apt install gcc‑12 g++‑12 & set CC,CXX

out of memory instantly under WSL2

Old Windows / WSL GPU layer bug

Update Windows driver + reboot

7. Usage

# Train
python scripts/run_training.py --use_gpu --rounds 300

# Back‑test
python scripts/run_backtest.py --model models/latest.pkl --use_gpu

# Live
python scripts/run_live.py  --symbols SPY,QQQ --interval 5m --use_gpu

--use_gpu flips LightGBM parameter device='cuda' and enables multi‑thread feature‑engineering.

8. Troubleshooting / FAQ

GPU idle, training on CPU?  → ensure device='cuda' in params and model says so in .model_file header; watch nvidia-smi.

Editable install breaks after git pull LightGBM → just rerun the compile recipe (#6.2).

Why not OpenCL build?  CUDA is ~10‑30 % faster on NVIDIA; OpenCL left as fallback.

9. References & Further Reading

LightGBM GPU guide – https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html

scikit‑build‑core skip flags – https://scikit-build-core.readthedocs.io

WSL2 CUDA setup – NVIDIA docs https://docs.nvidia.com/cuda/wsl-user-guide/index.html
