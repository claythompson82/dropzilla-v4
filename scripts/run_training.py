"""
Main script to run a full model training and optimization pipeline.

This script orchestrates the entire process:
1. Loads configuration.
2. Initializes the data client.
3. Fetches training data for a list of symbols.
4. Performs feature engineering and labeling.
5. Runs Bayesian hyperparameter optimization.
6. Saves the best model and supplementary artifacts.
"""
import pandas as pd
import pandas_ta as ta
import joblib
from datetime import datetime, timedelta

# Import project modules
from dropzilla.config import POLYGON_API_KEY, DATA_CONFIG, MODEL_CONFIG, LABELING_CONFIG, FEATURE_CONFIG
from dropzilla.data import PolygonDataClient
from dropzilla.features import calculate_features
from dropzilla.labeling import get_triple_barrier_labels
from dropzilla.validation import PurgedKFold
from dropzilla.models import optimize_hyperparameters


def main() -> None:
    """Main training pipeline execution function."""
    print("--- Starting Dropzilla v4 Training Pipeline ---")

    # 1. Initialization
    if not POLYGON_API_KEY or POLYGON_API_KEY == "YOUR_DEFAULT_KEY_HERE":
        raise ValueError("POLYGON_API_KEY is not set. Please set it in your .env file.")

    data_client = PolygonDataClient(api_key=POLYGON_API_KEY)

    # Define symbols and date range for training data
    symbols_to_train = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOG"]
    to_date = datetime.now()
    from_date = to_date - timedelta(days=DATA_CONFIG['data_period_days'])

    # 2. Data Collection
    all_data = []
    for symbol in symbols_to_train:
        print(f"\nFetching data for {symbol}...")
        df = data_client.get_aggs(
            symbol,
            from_date=from_date.strftime('%Y-%m-%d'),
            to_date=to_date.strftime('%Y-%m-%d')
        )
        if df is not None and not df.empty:
            df['symbol'] = symbol
            all_data.append(df)

    if not all_data:
        print("No data collected. Exiting.")
        return

    combined_df = pd.concat(all_data)
    print(f"\nTotal data collected: {len(combined_df)} rows.")

    # 3. Feature Engineering & Labeling (Simplified for now)
    print("Calculating features...")
    features_df = calculate_features(combined_df, FEATURE_CONFIG)

    # --- Dynamic Labeling ---
    atr = ta.atr(
        high=features_df['High'],
        low=features_df['Low'],
        close=features_df['Close'],
        length=LABELING_CONFIG['atr_period']
    )
    target_volatility = atr.reindex(features_df.index).fillna(method='bfill')

    vertical_barrier_times = (
        features_df.index.to_series() +
        pd.Timedelta(minutes=LABELING_CONFIG['vertical_barrier_minutes'])
    )

    labels = get_triple_barrier_labels(
        prices=features_df['Close'],
        t_events=features_df.index,
        pt_sl=[
            LABELING_CONFIG['profit_take_atr_multiplier'],
            LABELING_CONFIG['stop_loss_atr_multiplier']
        ],
        target=target_volatility,
        vertical_barrier_times=vertical_barrier_times,
        side=pd.Series(-1, index=features_df.index)
    )

    features_df['drop_label'] = labels['bin'].replace(-1, 0)
    features_df['label_time'] = labels['t1']
    # --- End Dynamic Labeling ---

    # Prepare final data for model
    features_to_use = ['relative_volume', 'distance_from_vwap_pct', 'vwap_slope', 'roc_60']
    final_df = features_df.dropna(subset=features_to_use + ['drop_label', 'label_time'])

    X = final_df[features_to_use]
    y = final_df['drop_label']
    label_times = final_df['label_time']

    print(f"Data ready for training. Shape: {X.shape}")
    print(f"Label distribution:\n{y.value_counts(normalize=True)}")

    # 4. Model Optimization
    print("\n--- Starting Hyperparameter Optimization ---")
    cv_validator = PurgedKFold(
        n_splits=MODEL_CONFIG['cv_n_splits'],
        label_times=label_times,
        embargo_pct=MODEL_CONFIG['cv_embargo_pct']
    )

    best_params, trials = optimize_hyperparameters(
        X, y, cv_validator, max_evals=MODEL_CONFIG['optimization_max_evals']
    )

    # 5. TODO: Final Model Training and Serialization
    print("\n--- Pipeline Complete (Model saving not yet implemented) ---")
    print(f"Best parameters found: {best_params}")


if __name__ == "__main__":
    main()
