# --- Compatibility Shim ---
# This must be at the very top of the file.
# It ensures that pandas_ta, which expects an older version of numpy,
# can run without errors.
import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan

"""
Main script to run a full model training and optimization pipeline.

This script orchestrates the entire process:
1. Loads configuration.
2. Initializes the data client.
3. Fetches training data for a list of symbols.
4. Performs feature engineering and labeling on a per-symbol basis.
5. Runs Bayesian hyperparameter optimization to find the best primary model.
6. Trains the primary model and a secondary meta-model for conviction.
7. Saves the final, complete artifact with both models.
"""
# --- Standard Library Imports ---
from datetime import datetime, timedelta

# --- Third-Party Imports ---
import pandas as pd
import pandas_ta as ta
import joblib

# --- Local Application Imports ---
from dropzilla.config import POLYGON_API_KEY, DATA_CONFIG, MODEL_CONFIG, LABELING_CONFIG, FEATURE_CONFIG
from dropzilla.data import PolygonDataClient
from dropzilla.features import calculate_features
from dropzilla.labeling import get_triple_barrier_labels
from dropzilla.validation import PurgedKFold
from dropzilla.models import optimize_hyperparameters, train_lightgbm_model
from dropzilla.context import get_market_regimes
from dropzilla.signal import train_meta_model


def generate_meta_dataset(model_artifact_path: str, data_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Generates a dataset for training the meta-model.

    It loads the primary model, makes predictions, and creates a new target
    variable where 1 means the primary model was correct, and 0 means it was not.

    Args:
        model_artifact_path (str): Path to the saved primary model artifact.
        data_df (pd.DataFrame): The fully featured and labeled DataFrame.

    Returns:
        pd.DataFrame | None: A new DataFrame ready for training the meta-model, or None if fails.
    """
    print(f"\n--- Generating Meta-Model Dataset from {model_artifact_path} ---")
    
    try:
        artifact = joblib.load(model_artifact_path)
    except FileNotFoundError:
        print(f"Error: Model artifact not found at {model_artifact_path}")
        return None

    primary_model = artifact['model']
    features_to_use = artifact['features_to_use']

    X = data_df[features_to_use]
    primary_predictions = primary_model.predict(X)
    primary_probabilities = primary_model.predict_proba(X)[:, 1]

    meta_df = data_df.copy()
    meta_df['primary_model_prediction'] = primary_predictions
    meta_df['primary_model_probability'] = primary_probabilities
    meta_df['meta_target'] = (meta_df['primary_model_prediction'] == meta_df['drop_label']).astype(int)

    # We only train the meta-model on instances where the primary model predicted a drop
    meta_df_filtered = meta_df[meta_df['primary_model_prediction'] == 1].copy()

    print(f"Meta-dataset created with {len(meta_df_filtered)} samples for training.")
    return meta_df_filtered


def main() -> None:
    """Main training pipeline execution function."""
    print("--- Starting Dropzilla v4 Training Pipeline ---")

    # 1. Initialization
    if not POLYGON_API_KEY or POLYGON_API_KEY == "YOUR_DEFAULT_KEY_HERE":
        raise ValueError("POLYGON_API_KEY is not set. Please set it in your .env file.")

    data_client = PolygonDataClient(api_key=POLYGON_API_KEY)

    symbols_to_train = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOG"]
    to_date = datetime.now()
    from_date = to_date - timedelta(days=DATA_CONFIG['data_period_days'])

    print("\nFetching market context data (SPY)...")
    spy_df = data_client.get_aggs(
        "SPY",
        from_date=(from_date - timedelta(days=50)).strftime('%Y-%m-%d'),
        to_date=to_date.strftime('%Y-%m-%d'),
        timespan='day'
    )
    if spy_df is not None and not spy_df.empty:
        spy_df['market_regime'] = get_market_regimes(spy_df)

    # 2. Data Collection & Processing
    all_labeled_data = []
    for symbol in symbols_to_train:
        print(f"\nFetching and processing data for {symbol}...")
        df = data_client.get_aggs(
            symbol,
            from_date=from_date.strftime('%Y-%m-%d'),
            to_date=to_date.strftime('%Y-%m-%d')
        )
        if df is None or df.empty:
            continue
        
        df['symbol'] = symbol
        
        group_with_context = pd.merge_asof(
            df.sort_index(), spy_df[['market_regime']].dropna(),
            left_index=True, right_index=True, direction='backward'
        )
        
        features_df = calculate_features(group_with_context, FEATURE_CONFIG)

        atr = ta.atr(high=features_df['High'], low=features_df['Low'], close=features_df['Close'], length=LABELING_CONFIG['atr_period'])
        target_volatility = atr.reindex(features_df.index).fillna(method='bfill')
        vertical_barrier_times = features_df.index.to_series() + pd.Timedelta(minutes=LABELING_CONFIG['vertical_barrier_minutes'])

        labels = get_triple_barrier_labels(
            prices=features_df['Close'], t_events=features_df.index,
            pt_sl=[LABELING_CONFIG['profit_take_atr_multiplier'], LABELING_CONFIG['stop_loss_atr_multiplier']],
            target=target_volatility, vertical_barrier_times=vertical_barrier_times,
            side=pd.Series(-1, index=features_df.index)
        )

        features_df['drop_label'] = labels['bin'].replace(-1, 0)
        features_df['label_time'] = labels['t1']
        
        all_labeled_data.append(features_df)

    if not all_labeled_data:
        print("No data processed. Exiting.")
        return

    # 3. Final Data Preparation
    final_df = pd.concat(all_labeled_data)
    features_to_use = [
        'relative_volume', 'distance_from_vwap_pct', 'vwap_slope', 'roc_30', 'roc_60', 'roc_120',
        'rsi_14', 'rsi_14_sma_5', 'macd_line', 'macd_signal', 'macd_hist', 'macd_hist_diff',
        'mfi_14', 'obv_slope', 'market_regime'
    ]
    final_df = final_df.dropna(subset=features_to_use + ['drop_label', 'label_time'])
    X = final_df[features_to_use]
    y = final_df['drop_label']
    label_times = final_df['label_time']

    print(f"\nData ready for training. Shape: {X.shape}")
    print(f"Label distribution:\n{y.value_counts(normalize=True)}")

    # 4. Primary Model Optimization
    print("\n--- Starting Hyperparameter Optimization ---")
    cv_validator = PurgedKFold(n_splits=MODEL_CONFIG['cv_n_splits'], label_times=label_times, embargo_pct=MODEL_CONFIG['cv_embargo_pct'])
    best_params, _ = optimize_hyperparameters(X, y, cv_validator, max_evals=MODEL_CONFIG['optimization_max_evals'])

    # 5. Final Primary Model Training
    print("\n--- Training Final Primary Model on All Data ---")
    final_params = best_params.copy()
    for param in ['n_estimators', 'num_leaves', 'max_depth', 'min_child_samples']:
        if param in final_params:
            final_params[param] = int(final_params[param])
    final_model = train_lightgbm_model(X.values, y.values, params=final_params)
    print("Final primary model training complete.")

    model_artifact = {"model": final_model, "best_params": best_params, "features_to_use": features_to_use, "training_timestamp_utc": datetime.utcnow().isoformat(), "dropzilla_version": "4.0"}
    model_filename = MODEL_CONFIG['model_filename']
    joblib.dump(model_artifact, model_filename)
    print(f"✅ Primary model artifact saved to: {model_filename}")

    # 6. Meta-Model Training
    meta_dataset = generate_meta_dataset(model_filename, final_df)
    if meta_dataset is not None and not meta_dataset.empty:
        meta_model = train_meta_model(meta_dataset)
        model_artifact['meta_model'] = meta_model
        model_artifact['meta_model_features'] = ['primary_model_probability', 'relative_volume', 'market_regime']
        joblib.dump(model_artifact, model_filename)
        print(f"✅ Meta-model trained and added to artifact: {model_filename}")
    else:
        print("Could not generate meta-dataset. Skipping meta-model training.")

    print("\n--- Full Pipeline Complete ---")


if __name__ == "__main__":
    main()
