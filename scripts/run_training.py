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
from datetime import datetime, timedelta

# Import project modules
from dropzilla.config import POLYGON_API_KEY, DATA_CONFIG, MODEL_CONFIG, LABELING_CONFIG, FEATURE_CONFIG
from dropzilla.data import PolygonDataClient
from dropzilla.features import calculate_features
# from dropzilla.labeling import get_triple_barrier_labels # To be used once fully implemented
from dropzilla.validation import PurgedKFold
from dropzilla.models import optimize_hyperparameters, train_lightgbm_model
import joblib


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

    # --- Placeholder Labeling ---
    # This section will be replaced by the output of the real labeling module
    # For now, create a dummy label for testing the pipeline
    features_df['drop_label'] = (features_df['Close'].pct_change(periods=60) < -0.01).astype(int)
    features_df['label_time'] = features_df.index + pd.Timedelta(minutes=LABELING_CONFIG['vertical_barrier_minutes'])
    # --- End Placeholder ---

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

    # 5. Final Model Training and Serialization
    print("\n--- Training Final Model on All Data ---")
    final_params = best_params.copy()
    for param in ['n_estimators', 'num_leaves', 'max_depth', 'min_child_samples']:
        if param in final_params:
            final_params[param] = int(final_params[param])

    final_model = train_lightgbm_model(X.values, y.values, params=final_params)
    print("Final model training complete.")

    model_artifact = {
        "model": final_model,
        "best_params": best_params,
        "features_to_use": features_to_use,
        "training_timestamp_utc": datetime.utcnow().isoformat(),
        "dropzilla_version": "4.0"
    }

    model_filename = MODEL_CONFIG['model_filename']
    joblib.dump(model_artifact, model_filename)
    print(f"✅ Model artifact successfully saved to: {model_filename}")

    print("\n--- Pipeline Complete ---")
    print(f"Best parameters found: {best_params}")


if __name__ == "__main__":
    main()
