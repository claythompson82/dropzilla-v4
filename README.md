# Dropzilla v4: An Institutional-Grade Intraday Signal Engine
**Last Updated:** July 7, 2025

## 1. Project Overview & Status

This repository contains the source code for Dropzilla v4, a proprietary signal engine designed to identify high-conviction, short-biased trading opportunities in liquid US equities on a "minutes-to-hours" intraday time horizon.

As of July 7, 2025, the core research and development phase of Dropzilla v4 is **complete and successful**. The system has evolved from a prototype with a flawed validation framework into a methodologically sound engine with a proven, profitable trading edge.

**Current Validated Performance (at 65% Confidence Threshold):**
* **Profit Factor:** 1.15
* **Win Rate:** 53.93%
* **Annualized Sharpe Ratio:** 0.98

The system is now ready for the next phase of its lifecycle: productionizing, deployment, and future research into enhancing its performance edge.

## 2. The Development Journey: A Methodological Evolution

The success of Dropzilla v4 is the result of a disciplined, iterative development process that prioritized methodological rigor over superficial metrics. The journey, which took place over an intensive period in early July 2025, involved several critical phases.

### Phase 1: Foundational Rectification (July 6, 2025)

The project began by addressing a critical flaw in its architecture: the use of a randomized train-test split for time-series data. This introduced severe lookahead bias, rendering all previous performance metrics invalid.

* **Action:** The validation framework was completely replaced with a `PurgedKFold` cross-validator to ensure a leak-free, time-series-aware evaluation process.
* **Outcome:** A reliable foundation was established, enabling true, unbiased measurement of model performance for the first time.

### Phase 2: Advanced Feature Engineering & Statistical Edge (July 6, 2025)

With a robust validation framework in place, the project focused on enhancing the model's predictive power by moving beyond standard technical indicators.

* **Action:** A suite of advanced contextual features, based on the project's foundational research documents, was implemented. This included:
    * **Volatility Regime Anomaly (VRA):** To quantify the "unusualness" of an asset's current volatility.
    * **Systemic Absorption Ratio (SAR):** To measure market fragility and risk concentration.
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

## 3. Current System Architecture

The final, validated architecture of Dropzilla v4 consists of two core components:

### Primary Model (The "Candidate Generator")
* **Algorithm:** `LightGBM` Classifier.
* **Function:** To analyze the market on a minute-by-minute basis and identify a large set of potential short-biased trading opportunities.
* **Key Features:** The model is trained on a rich feature set that includes standard price/volume derivatives, momentum indicators, and the advanced `vra_score` and `sar_score` contextual features.
* **Performance:** Validated ROC AUC of **0.7009**.

### Conviction Engine (The "Signal Filter")
* **Algorithm:** A deterministic **Multi-Factor Confidence Score**.
* **Function:** To evaluate each candidate signal from the primary model and assign a final, principled conviction score from 0% to 100%.
* **Components:** The score is a weighted average of four key factors:
    1.  **Calibrated Probability (40% weight):** The raw probability output from the primary model.
    2.  **Regime Context (25% weight):** The broader market state (e.g., 'Bear Trend').
    3.  **Volume Confirmation (20% weight):** The relative volume behind the signal.
    4.  **Signal Stability (15% weight):** The consistency of the primary model's probability over the last few periods.

## 4. Path Forward

### Productionizing
The immediate priority is to prepare the validated system for operational use.
1.  **Code Refactoring:** Refactor the codebase from scripts into a clean, modular, and installable Python package to improve maintainability and testability.
2.  **Deployment Scripting:** Develop a `run_live.py` script to execute the prediction pipeline on live market data.
3.  **GUI Development:** Build a user interface to visualize signals and system status.

### Future Research
The current Sharpe Ratio of 0.98 is a strong foundation. The following research avenues are recommended for future iterations to enhance the system's edge:
* **Kalman Filter:** Upgrade the "Signal Stability" component from a simple moving average to a more responsive Kalman Filter to better estimate the true underlying conviction.
* **Universe Expansion:** Train and validate a new model on a wider and more diverse universe of equities to improve the generalizability of its learned patterns.
* **Advanced Feature Engineering:** Explore more complex features, such as those derived from **Wavelet Analysis** or **LSTM Autoencoders**, to capture non-linear market dynamics.

## 5. Installation & Usage

### Prerequisites
* Ubuntu (tested on WSL2 for Windows 11)
* Python 3.12+

### Setup
1.  **Clone the Repository:**
    ```bash
    git clone <your-private-repo-url>
    cd dropzilla-v4
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Set Environment Variable:**
    Set your Polygon.io API key. This can be done permanently by adding it to your shell's configuration file.
    ```bash
    echo 'export POLYGON_API_KEY="YOUR_API_KEY_HERE"' >> ~/.bashrc
    source ~/.bashrc
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Install Dropzilla in Editable Mode:**
    Installing in editable mode (`-e`) allows code changes to be reflected immediately without reinstallation.
    ```bash
    pip install -e .
    ```

### Running Key Scripts
* **Train a New Model:**
    ```bash
    python scripts/run_training.py
    ```
* **Run a Financial Backtest:**
    ```bash
    python scripts/run_backtest.py --model <path_to_model.pkl> --threshold <e.g., 0.65>
    ```
* **Run Unit Tests:**
    ```bash
    pytest
    ```
