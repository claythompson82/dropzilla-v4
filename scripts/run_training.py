# --- Compatibility Shim ---
import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan

"""
Main script to run a full model training and optimization pipeline.
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
from dropzilla.correlation import get_systemic_absorption_ratio
from dropzilla.volatility import get_volatility_regime_anomaly


def generate_meta_dataset(model_artifact_path: str, data_df: pd.DataFrame, probability_threshold: float = 0.20) -> pd.DataFrame | None:
    """
    Generates a dataset for training the meta-model.
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
    primary_probabilities = primary_model.predict_proba(X)[:, 1]
    model_uncertainty = 1 - 2 * np.abs(primary_probabilities - 0.5)
    primary_candidate_signals = (primary_probabilities >= probability_threshold).astype(int)
    meta_df = data_df.copy()
    meta_df['primary_model_candidate'] = primary_candidate_signals
    meta_df['primary_model_probability'] = primary_probabilities
    meta_df['model_uncertainty'] = model_uncertainty
    meta_df['meta_target'] = (meta_df['primary_model_candidate'] == meta_df['drop_label']).astype(int)
    meta_df_filtered = meta_df[meta_df['primary_model_candidate'] == 1].copy()
    print(f"Found {len(meta_df_filtered)} candidate signals with p > {probability_threshold} for meta-model training.")
    return meta_df_filtered

def main() -> None:
    """Main training pipeline execution function."""
    print("--- Starting Dropzilla v4 ADVANCED Training Pipeline ---")

    # 1. Initialization
    data_client = PolygonDataClient(api_key=POLYGON_API_KEY)
    symbols_to_train = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOG"]
    to_date = datetime.now()
    from_date = to_date - timedelta(days=DATA_CONFIG['data_period_days'])

    # 2. Data Collection
    print("Fetching all market data...")
    market_data = {
        symbol: data_client.get_aggs(
            symbol,
            from_date=from_date.strftime('%Y-%m-%d'),
            to_date=to_date.strftime('%Y-%m-%d') # THE FIX: Added 'to_date='
        )
        for symbol in symbols_to_train
    }
    market_data = {k: v for k, v in market_data.items() if v is not None and not v.empty}
    
    spy_df = data_client.get_aggs(
        "SPY",
        from_date=(from_date - timedelta(days=50)).strftime('%Y-%m-%d'),
        to_date=to_date.strftime('%Y-%m-%d'), # THE FIX: Added 'to_date='
        timespan='day'
    )
    if spy_df is not None and not spy_df.empty:
        spy_df['market_regime'] = get_market_regimes(spy_df)

    # 3. Cross-Sectional & Per-Symbol Feature Engineering
    all_labeled_data = []
    
    daily_returns_panel = pd.DataFrame({
        symbol: df['Close'].resample('D').last().pct_change() for symbol, df in market_data.items()
    }).dropna(how='all')
    sar_scores = get_systemic_absorption_ratio(daily_returns_panel)

    for symbol, df in market_data.items():
        print(f"\nProcessing features and labels for {symbol}...")
        
        daily_rv = df['Close'].resample('D').last().pct_change().rolling(window=21).std() * np.sqrt(252)
        vra_scores = get_volatility_regime_anomaly(daily_rv.dropna())
        
        df_context = pd.merge_asof(df.sort_index(), spy_df[['market_regime']].dropna(), left_index=True, right_index=True, direction='backward')
        df_context = pd.merge_asof(df_context, sar_scores.to_frame(name='sar_score'), left_index=True, right_index=True, direction='backward')
        df_context = pd.merge_asof(df_context, vra_scores.to_frame(name='vra_score'), left_index=True, right_index=True, direction='backward')
        df_context[['sar_score', 'vra_score']] = df_context[['sar_score', 'vra_score']].ffill().fillna(0)

        daily_log_returns = np.log(df['Close'].resample('D').last().pct_change()).dropna()
        features_df = calculate_features(df_context, daily_log_returns, FEATURE_CONFIG)
        
        atr = ta.atr(high=features_df['High'], low=features_df['Low'], close=features_df['Close'], length=LABELING_CONFIG['atr_period'])
        target_volatility = atr.reindex(features_df.index).fillna(method='bfill')
        vertical_barrier_times = features_df.index.to_series() + pd.Timedelta(minutes=LABELING_CONFIG['vertical_barrier_minutes'])
        labels = get_triple_barrier_labels(prices=features_df['Close'], t_events=features_df.index, pt_sl=[LABELING_CONFIG['profit_take_atr_multiplier'], LABELING_CONFIG['stop_loss_atr_multiplier']], target=target_volatility, vertical_barrier_times=vertical_barrier_times, side=pd.Series(-1, index=features_df.index))
        features_df['drop_label'] = labels['bin'].replace(-1, 0)
        features_df['label_time'] = labels['t1']
        all_labeled_data.append(features_df)

    # 4. Final Data Preparation
    final_df = pd.concat(all_labeled_data)
    features_to_use = [
        'relative_volume', 'distance_from_vwap_pct', 'vwap_slope', 'roc_30', 'roc_60', 'roc_120',
        'rsi_14', 'rsi_14_sma_5', 'macd_line', 'macd_signal', 'macd_hist', 'macd_hist_diff',
        'mfi_14', 'obv_slope', 'market_regime', 'volatility_surprise', 'sar_score', 'vra_score'
    ]
    final_df = final_df.dropna(subset=features_to_use + ['drop_label', 'label_time'])
    
    if final_df.empty:
        print("ERROR: No data remaining after feature calculation and cleanup. Exiting.")
        return

    X = final_df[features_to_use]
    y = final_df['drop_label']
    label_times = final_df['label_time']

    print(f"\nData ready for training. Shape: {X.shape}")
    print(f"Label distribution:\n{y.value_counts(normalize=True)}")

    # 5. Primary Model Optimization & Training
    print("\n--- Starting Hyperparameter Optimization ---")
    cv_validator = PurgedKFold(n_splits=MODEL_CONFIG['cv_n_splits'], label_times=label_times, embargo_pct=MODEL_CONFIG['cv_embargo_pct'])
    best_params, _ = optimize_hyperparameters(X, y, cv_validator, max_evals=MODEL_CONFIG['optimization_max_evals'])
    
    print("\n--- Training Final Primary Model on All Data ---")
    final_params = best_params.copy()
    for param in ['n_estimators', 'num_leaves', 'max_depth', 'min_child_samples']:
        if param in final_params:
            final_params[param] = int(final_params[param])
    final_model = train_lightgbm_model(X, y, params=final_params)
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
        model_artifact['meta_model_features'] = ['primary_model_probability', 'relative_volume', 'market_regime', 'model_uncertainty', 'sar_score', 'vra_score']
        joblib.dump(model_artifact, model_filename)
        print(f"✅ Meta-model trained and added to artifact: {model_filename}")
    else:
        print("Could not generate meta-dataset. Skipping meta-model training.")

    print("\n--- Full Pipeline Complete ---")

if __name__ == "__main__":
    main()
