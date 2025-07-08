# Dropzilla v4: An Institutional-Grade Intraday Signal Engine
**Last Updated:** July 7, 2025
[GPU ✔ | CUDA speed-up x5.5](docs/GPU_BENCHMARK.md)

## 1. Project Overview & Status

This repository contains the source code for Dropzilla v4, a proprietary signal engine designed to identify high-conviction, short-biased trading opportunities in liquid US equities on a "minutes-to-hours" intraday time horizon.

As of July 7, 2025, the core research and development phase of Dropzilla v4 is **complete, scientifically rigorous, and methodologically sound**. The training pipeline has been rigorously validated using a PurgedKFold cross-validation framework to eliminate lookahead bias and ensure the integrity of all performance metrics.

**Key Validated Performance Metrics:**
* **ROC AUC (Broad, Liquid Universe):** 0.70089  
* **ROC AUC (Two-Ticker Test – SEDG & SYM):** 0.6718  
* **Profit Factor (65% Confidence Threshold):** 1.15  
* **Win Rate:** 53.93%  
* **Annualized Sharpe Ratio:** 0.98  

## 2. The Development Journey: A Methodological Evolution

### Phase 1: Foundational Rectification (July 6, 2025)
- Replaced randomized splits with a `PurgedKFold` cross-validator to eliminate lookahead bias.

### Phase 2: Advanced Feature Engineering & Statistical Edge (July 6, 2025)
- Added Volatility Regime Anomaly (VRA) and Systemic Absorption Ratio (SAR) contextual features.
- Achieved a **best validation ROC AUC of 0.7009** on a diversified universe.

### Phase 3: Meta-Model Pivot (July 6–7, 2025)
- Abandoned the imbalanced secondary ML meta-model in favor of a deterministic Multi-Factor Confidence Score.

### Phase 4: Economic Validation & Optimization (July 7, 2025)
- Conducted backtests across confidence thresholds; identified the optimal **65% threshold** yielding a positive economic edge.

## 3. Current System Architecture

The final, validated architecture of Dropzilla v4 consists of three core components:

### 3.1 Context-Awareness Engine
- **Market Regime Detection:** Gaussian HMM on SPY log-returns to classify regimes.
- **Volatility Regime Anomaly (VRA):** Captures deviations in realized volatility.
- **Systemic Absorption Ratio (SAR):** Measures market fragility and risk concentration.
- **Function:** Supplies adaptive, real-time context signals to both the Primary Model and Conviction Engine.

### 3.2 Primary Model (The "Candidate Generator")
- **Algorithm:** LightGBM classifier.
- **Features:** Price/volume derivatives, momentum, plus regime, VRA, and SAR scores.
- **Performance:** Validated ROC AUC of **0.70089**.

### 3.3 Conviction Engine (The "Signal Filter")
- **Algorithm:** Deterministic Multi-Factor Confidence Score.
- **Components:** Calibrated Probability (40%), Regime Context (25%), Volume Confirmation (20%), Signal Stability (15%).
- **Function:** Ranks and filters primary model candidates into final signals.

## 4. Path Forward

### Productionizing
1. **Refactor** into a modular Python package.  
2. **Deploy** with a `run_live.py` script for live data.  
3. **Build** a GUI to visualize signals and system health.

### Future Research
* **Kalman Filter** for Signal Stability  
* **Universe Expansion** to stress-test generalization  
* **Wavelet & LSTM Features** for non-linear dynamics  

## 5. Installation & Usage

### Prerequisites
* Ubuntu (WSL2 on Windows 11 supported)  
* Python 3.12+

### Setup
```bash
git clone <your-private-repo-url>
cd dropzilla-v4
python3 -m venv .venv
source .venv/bin/activate
echo 'export POLYGON_API_KEY="YOUR_API_KEY_HERE"' >> ~/.bashrc && source ~/.bashrc
pip install -r requirements.txt
pip install -e .
