"""
Main script to run a full model training and optimization pipeline.
This version is instrumented to emit structured progress updates for the GUI.
"""
import argparse
import pandas as pd
import numpy as np
import joblib
import pandas_ta as ta
from datetime import datetime, timedelta

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

def generate_meta_dataset(model_artifact_path: str, data_df: pd.DataFrame, probability_threshold: float = 0.5) -> pd.DataFrame | None:
    """Generates a dataset for training the meta-model."""
    print(f"PROGRESS::STATUS::Generating meta-model dataset...")
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

def main(tickers: list[str], model_name: str) -> None:
    """Main training pipeline execution function."""
    total_steps = 6 # Total number of major steps in the training process
    
    print(f"PROGRESS::OVERALL::{1}/{total_steps}::Initializing and Fetching Market Data")
    data_client = PolygonDataClient(api_key=POLYGON_API_KEY)
    to_date = datetime.now()
    from_date = to_date - timedelta(days=DATA_CONFIG['data_period_days'])
    market_data = {
        symbol: data_client.get_aggs(symbol, from_date.strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d'))
        for symbol in tickers
    }
    market_data = {k: v for k, v in market_data.items() if v is not None and not v.empty}
    spy_df = data_client.get_aggs("SPY", (from_date - timedelta(days=50)).strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d'), timespan='day')
    if spy_df is not None and not spy_df.empty:
        spy_df['market_regime'] = get_market_regimes(spy_df)

    print(f"PROGRESS::OVERALL::{2}/{total_steps}::Engineering Features and Labels")
    all_labeled_data = []
    daily_returns_panel = pd.DataFrame({s: df['Close'].resample('D').last().pct_change() for s, df in market_data.items()}).dropna(how='all')
    sar_scores = get_systemic_absorption_ratio(daily_returns_panel)
    
    num_tickers = len(market_data)
    for i, (symbol, df) in enumerate(market_data.items()):
        print(f"PROGRESS::DETAIL::{i+1}/{num_tickers}::Processing {symbol}")
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

    print(f"PROGRESS::STATUS::Finalizing data preparation...")
    final_df = pd.concat(all_labeled_data)
    features_to_use = [
        'relative_volume', 'distance_from_vwap_pct', 'vwap_slope', 'roc_30', 'roc_60', 'roc_120',
        'rsi_14', 'rsi_14_sma_5', 'macd_line', 'macd_signal', 'macd_hist', 'macd_hist_diff',
        'mfi_14', 'obv_slope', 'market_regime', 'volatility_surprise', 'sar_score', 'vra_score'
    ]
    final_df = final_df.dropna(subset=features_to_use + ['drop_label', 'label_time'])
    if final_df.empty:
        print("ERROR: No data remaining after feature calculation. Exiting.")
        return

    X = final_df[features_to_use]
    y = final_df['drop_label']
    label_times = final_df['label_time']

    print(f"PROGRESS::OVERALL::{3}/{total_steps}::Optimizing Hyperparameters")
    cv_validator = PurgedKFold(n_splits=MODEL_CONFIG['cv_n_splits'], label_times=label_times, embargo_pct=MODEL_CONFIG['cv_embargo_pct'])
    best_params, _ = optimize_hyperparameters(X, y, cv_validator, max_evals=MODEL_CONFIG['optimization_max_evals'])

    print(f"PROGRESS::OVERALL::{4}/{total_steps}::Training Final Primary Model")
    final_params = best_params.copy()
    for param in ['n_estimators', 'num_leaves', 'max_depth', 'min_child_samples']:
        if param in final_params:
            final_params[param] = int(final_params[param])
    final_model = train_lightgbm_model(X, y, params=final_params)
    
    print(f"PROGRESS::STATUS::Saving primary model artifact...")
    model_artifact = {"model": final_model, "best_params": best_params, "features_to_use": features_to_use, "training_timestamp_utc": datetime.utcnow().isoformat(), "dropzilla_version": "4.0"}
    joblib.dump(model_artifact, model_name)
    print(f"✅ Primary model artifact saved to: {model_name}")

    print(f"PROGRESS::OVERALL::{5}/{total_steps}::Training Meta-Model")
    meta_dataset = generate_meta_dataset(model_name, final_df)
    if meta_dataset is not None and not meta_dataset.empty:
        meta_model = train_meta_model(meta_dataset)
        model_artifact['meta_model'] = meta_model
        model_artifact['meta_model_features'] = [
            'primary_model_probability', 'relative_volume', 'market_regime',
            'model_uncertainty', 'sar_score', 'vra_score'
        ]
        joblib.dump(model_artifact, model_name)
        print(f"✅ Meta-model trained and added to artifact: {model_name}")
    else:
        print("Could not generate meta-dataset. Skipping meta-model training.")

    print(f"PROGRESS::OVERALL::{6}/{total_steps}::Training Complete!")
    print("\n--- Full Pipeline Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dropzilla v4 Training Pipeline.")
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated list of tickers to train on.")
    parser.add_argument("--model_name", type=str, required=True, help="Filename for the new model artifact (e.g., my_model.pkl).")
    args = parser.parse_args()
    ticker_list = [ticker.strip().upper() for ticker in args.tickers.split(',')]
    main(tickers=ticker_list, model_name=args.model_name)
