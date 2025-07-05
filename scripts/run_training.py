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
from typing import cast

# Import project modules
from dropzilla.config import POLYGON_API_KEY, DATA_CONFIG, MODEL_CONFIG, LABELING_CONFIG, FEATURE_CONFIG
from dropzilla.data import PolygonDataClient
from dropzilla.features import calculate_features
from dropzilla.labeling import get_triple_barrier_labels
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
    from_date = to_date - timedelta(days=cast(int, DATA_CONFIG['data_period_days']))

    # --- NEW: Fetch Market Context Data (SPY) ---
    print("\nFetching market context data (SPY)...")
    spy_df = data_client.get_aggs(
        "SPY",
        from_date=(from_date - timedelta(days=50)).strftime('%Y-%m-%d'),  # Fetch extra for rolling calcs
        to_date=to_date.strftime('%Y-%m-%d'),
        timespan='day'
    )
    # --- End NEW ---

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
    print("Calculating features and labels...")
    processed = []
    for symbol, group_df in combined_df.groupby('symbol'):
        print(f"\nProcessing features and labels for {symbol}...")

        # --- NEW: Merge Market Regime ---
        if 'market_regime' not in spy_df.columns:
            from dropzilla.context import get_market_regimes
            spy_df['market_regime'] = get_market_regimes(spy_df)

        group_df = pd.merge_asof(
            group_df.sort_index(),
            spy_df[['market_regime']].dropna(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        # --- End NEW ---

        # Calculate features for the single symbol
        features_df = calculate_features(group_df, FEATURE_CONFIG)

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

        processed.append(features_df)

    features_df = pd.concat(processed)

    # Prepare final data for model
    features_to_use = [
        'relative_volume', 'distance_from_vwap_pct', 'vwap_slope',
        'roc_30', 'roc_60', 'roc_120',
        'rsi_14', 'rsi_14_sma_5',
        'macd_line', 'macd_signal', 'macd_hist', 'macd_hist_diff',
        'mfi_14', 'obv_slope',
        'market_regime'
    ]
    final_df = features_df.dropna(subset=features_to_use + ['drop_label', 'label_time'])

    X = final_df[features_to_use]
    y = final_df['drop_label']
    label_times = final_df['label_time']

    print(f"Data ready for training. Shape: {X.shape}")
    print(f"Label distribution:\n{y.value_counts(normalize=True)}")

    # 4. Model Optimization
    print("\n--- Starting Hyperparameter Optimization ---")
    cv_validator = PurgedKFold(
        n_splits=cast(int, MODEL_CONFIG['cv_n_splits']),
        label_times=label_times,
        embargo_pct=cast(float, MODEL_CONFIG['cv_embargo_pct'])
    )

    best_params, trials = optimize_hyperparameters(
        X, y, cv_validator, max_evals=cast(int, MODEL_CONFIG['optimization_max_evals'])
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


def generate_meta_dataset(model_artifact_path: str, data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a dataset for training the meta-model.

    It loads the primary model, makes predictions, and creates a new target
    variable where 1 means the primary model was correct, and 0 means it was not.

    Args:
        model_artifact_path (str): Path to the saved primary model artifact.
        data_df (pd.DataFrame): The fully featured and labeled DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame ready for training the meta-model.
    """
    print(f"\n--- Generating Meta-Model Dataset from {model_artifact_path} ---")

    # 1. Load the trained primary model and its details
    artifact = joblib.load(model_artifact_path)
    primary_model = artifact['model']
    features_to_use = artifact['features_to_use']

    # 2. Get predictions and probabilities from the primary model
    X = data_df[features_to_use]
    primary_predictions = primary_model.predict(X)
    primary_probabilities = primary_model.predict_proba(X)[:, 1]  # Probability of class 1

    # 3. Create the meta-model features
    meta_df = data_df.copy()
    meta_df['primary_model_prediction'] = primary_predictions
    meta_df['primary_model_probability'] = primary_probabilities

    # 4. Create the meta-model target label
    # The target is 1 if the primary model's prediction matched the true label
    meta_df['meta_target'] = (
        meta_df['primary_model_prediction'] == meta_df['drop_label']
    ).astype(int)

    # We only train the meta-model on instances where the primary model predicted a drop
    meta_df = meta_df[meta_df['primary_model_prediction'] == 1]

    print(f"Meta-dataset created with {len(meta_df)} samples.")
    return meta_df


if __name__ == "__main__":
    main()
    # Example of how to run the new function after main() completes:
    # final_df = ... (this would need to be passed from main)
    # generate_meta_dataset("dropzilla_v4_lgbm.pkl", final_df)
