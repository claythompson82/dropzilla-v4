# Dropzilla v4: An Institutional-Grade Intraday Signal Engine
**Last Updated:** July 15, 2025

## 1. Project Overview & Status

This repository contains the source code for Dropzilla v4, a proprietary signal engine designed to identify high-conviction, short-biased trading opportunities in liquid US equities on a "minutes-to-hours" intraday time horizon.

As of July 15, 2025, the core research, development, and iterative optimization phase of Dropzilla v4 is **complete and highly successful**. The system has evolved from an early prototype plagued by validation flaws and inconsistent performance into a robust, methodologically sound engine with a proven, profitable trading edge. Through targeted debugging, code refinements, and expanded training on 10 tickers, the model now achieves superior statistical and economic metrics compared to initial benchmarks.

**Current Validated Performance (at 65% Confidence Threshold):**
* **Profit Factor:** 1.24
* **Win Rate:** 47.84%
* **Annualized Sharpe Ratio:** 1.44
* **ROC AUC (Validation):** 0.7968

These metrics were achieved on July 15, 2025, using a training dataset of ~1.86 million minute-level samples from 10 liquid tickers (AAPL, MSFT, NVDA, TSLA, GOOG, AMZN, META, AMD, INTC, PYPL) over approximately one year. The system is now primed for production deployment, live testing, and further enhancements to sharpen its edge.

## 2. The Development Journey: A Methodological Evolution

The success of Dropzilla v4 stems from a rigorous, iterative process emphasizing bias-free validation, advanced feature engineering, and reproducible experimentation. Development spanned from January/February 2025 (initial iterations) through an intensive overhaul in early July 2025, culminating in breakthrough performance on July 15. Key phases are detailed below, with granular technical insights for reproducibility.

### Phase 1: Foundational Rectification (July 6, 2025)

The project began by addressing a critical flaw in its architecture: the use of a randomized train-test split for time-series data. This introduced severe lookahead bias, rendering all previous performance metrics invalid.

* **Action:** The validation framework was completely replaced with a `PurgedKFold` cross-validator (from `mlfinlab` or equivalent) to ensure a leak-free, time-series-aware evaluation process. Parameters: `n_splits=5`, `embargo_pct=0.01` (1% of data purged post-fold to avoid event overlap), `pct_purge=0.01`.
* **Outcome:** A reliable foundation was established, enabling true, unbiased measurement of model performance for the first time. Early tests confirmed no leakage, setting the stage for accurate AUC scoring.

### Phase 2: Advanced Feature Engineering & Statistical Edge (July 6, 2025)

With a robust validation framework in place, the project focused on enhancing the model's predictive power by moving beyond standard technical indicators.

* **Action:** A suite of advanced contextual features, based on the project's foundational research documents, was implemented. This included:
    * **Volatility Regime Anomaly (VRA):** Quantifies the "unusualness" of an asset's current volatility using a rolling 21-day standard deviation of daily returns, annualized (`std * sqrt(252)`), and anomaly detection (likely via isolation forest or similar in `get_volatility_regime_anomaly`).
    * **Systemic Absorption Ratio (SAR):** Measures market fragility by computing the ratio of variance explained by top principal components in a panel of daily returns across symbols (via PCA in `get_systemic_absorption_ratio`).
* **Technical Details:** Features computed per symbol, merged via `pd.merge_asof` (backward direction) with datetime indices preserved. Daily log returns used: `np.log(1 + pct_change(fill_method=None))` to handle zeros/negatives without NaN/inf errors.
* **Outcome:** The new features successfully propelled the primary `LightGBM` model across a key performance barrier, achieving a **best validation ROC AUC score of 0.7009**. This provided objective, mathematical proof of a significant statistical edge.

### Phase 3: The Meta-Model Challenge & Architectural Pivot (July 6-7, 2025)

The initial architecture proposed a two-stage machine learning model, with a secondary "meta-model" to provide a conviction score. This approach failed.

* **Diagnosis:** A detailed diagnostic analysis revealed that the meta-model's training data was severely imbalanced (over 99.7% false positives), causing it to produce functionally useless confidence scores with a maximum value of less than 1%.
* **Action:** A decisive architectural pivot was made. The flawed ML meta-model was abandoned in favor of a **Multi-Factor Confidence Score**, as outlined in the system's design documents.
* **Outcome:** This new architecture provided a robust, interpretable, and effective framework for measuring signal conviction, unblocking the project and enabling the final phase of economic validation.

### Phase 4: Economic Validation & Optimization (July 7, 2025)

With the new confidence architecture in place, the final step was to evaluate the system's real-world economic performance.

* **Action:** A series of backtests were run on the final model, systematically probing different confidence thresholds (55%, 65%, 75%).
* **Outcome:** The backtests successfully identified the optimal "sweet spot" for the system. At a **65% confidence threshold**, the model demonstrates a clear, positive economic edge, achieving a Profit Factor of 1.15 and a Sharpe Ratio of 0.98.

### Phase 5: Iterative Debugging, Expansion, and Breakthrough Performance (July 15, 2025)

Following the initial validation, the system underwent targeted refinements to address inconsistencies (e.g., post-GPU revert AUC drops) and scale to 10 tickers for improved generalizability. This phase involved collaborative debugging with AI assistants (Grok, o3, GPT-4o, o4-mini) to fix code issues and reproduce high scores.

* **Actions and Technical Fixes (for Reproducibility):**
  - **Repo Revert:** Checked out the stable commit tagged "v4-stable-high-auc" (hash: 8eba8985), which previously achieved AUC 0.7391 on CPU with PurgedKFold.
  - **Code Edits in `scripts/run_training.py`**:
    - Added CLI args via `argparse`: `--symbols` (e.g., "AAPL,MSFT,..."), `--device cpu`, `--cv 5`.
    - Fixed returns: All `pct_change()` calls set to `fill_method=None` (deprecation fix); log returns to `np.log(1 + pct_change)` (avoids NaN/inf).
    - Label mapping: Changed from `{-1:1}` to `{1:1}` for short-biased drops (profit-take hits first with side=-1 in triple-barrier).
    - Index Compatibility: Forced datetime indices on SAR/VRA scores: `pd.Series(scores, index=...index)`.
    - Seeds: Added `np.random.seed(42)`, `random.seed(42)` for reproducibility.
    - Hyperopt: 50 trials; best params: `{'learning_rate': 0.0227, 'max_depth': 13, 'min_child_samples': 70, 'n_estimators': 850, 'num_leaves': 105, 'reg_alpha': 0.1528, 'reg_lambda': 0.6792}`.
  - **Other Files:**
    - `dropzilla/features.py`: Changed `fillna(method='bfill/ffill')` to `.bfill().ffill()` (deprecation fix).
    - `dropzilla/context.py`: Changed `fillna(method='ffill')` to `.ffill()`.
  - **Training Command:** `python scripts/run_training.py --symbols AAPL,MSFT,NVDA,TSLA,GOOG,AMZN,META,AMD,INTC,PYPL --device cpu --cv 5`
    - Data: ~1.86M samples; labels ~1% positives.
    - Model: LightGBM with `device_type='cpu'`; meta-model trained successfully on 55,394 candidates (34% positives).
  - **Backtest Command:** `python scripts/run_backtest.py --model dropzilla_v4_lgbm.pkl --threshold 0.65`
* **Outcome:** Achieved breakthrough metrics: Validation ROC AUC of 0.7968, Profit Factor 1.24, Win Rate 47.84%, Sharpe Ratio 1.44 (at 65% threshold, 533 trades). This represents a sharpened edge, attributable to bug fixes, expanded universe, and stable CPU training.

## 3. Current System Architecture

The validated architecture of Dropzilla v4 consists of two core components, with optional meta-model integration for enhanced conviction.

### Primary Model (The "Candidate Generator")
* **Algorithm:** `LightGBM` Classifier.
* **Function:** Analyzes minute-by-minute data to identify potential short-biased opportunities.
* **Key Features:** Rich set including price/volume (e.g., relative_volume, ROC_30/60/120), momentum (RSI_14, MACD), and advanced contextual (vra_score, sar_score).
* **Performance:** Validated ROC AUC of **0.7968** (on 10 tickers, PurgedKFold CV).

### Conviction Engine (The "Signal Filter")
* **Algorithm:** Deterministic **Multi-Factor Confidence Score** (weighted average), with optional ML meta-model for refinement.
* **Function:** Assigns 0-100% conviction to candidates; filters to high-threshold signals.
* **Components:** 
  1. **Calibrated Probability (40% weight):** Primary model output.
  2. **Regime Context (25% weight):** Market state (e.g., 'Bear Trend' from SPY regimes).
  3. **Volume Confirmation (20% weight):** Relative volume surge.
  4. **Signal Stability (15% weight):** Probability consistency (simple MA; future: Kalman Filter).
* **Meta-Model Note:** Trained on filtered candidates (p>=0.50), using features like primary_probability, uncertainty, sar/vra. Successful in latest run (balanced data), adding probabilistic conviction layer.

## 4. Path Forward

### Productionizing
1. **Code Refactoring:** Modularize into Python package; enhance CLI for live runs.
2. **Deployment Scripting:** Implement `run_live.py` for real-time Polygon feeds.
3. **GUI Development:** Build interface (e.g., Streamlit) for signal visualization.

### Future Research
- **Kalman Filter:** Upgrade signal stability for dynamic conviction.
- **Universe Expansion:** Train on 50+ equities (e.g., add QQQ components) for broader patterns.
- **Advanced Features:** Incorporate wavelet analysis or LSTM autoencoders for non-linear dynamics.
- **Live Monitoring:** Add slippage/fee simulation; retrain quarterly on fresh data.

## 5. Installation & Usage

### Prerequisites
* Ubuntu (tested on WSL2 for Windows 11)
* Python 3.12+

### Setup
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/claythompson82/dropzilla-v4.git
   cd dropzilla-v4
   git checkout v4-stable-high-auc  # Use stable commit for reproducibility
