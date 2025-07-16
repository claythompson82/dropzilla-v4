#!/usr/bin/env python
# --- Compatibility Shim ---
import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan

"""
Main script to run a full model training and optimization pipeline.
"""
# --- Standard Library Imports ---
from datetime import datetime, timedelta
import random

# --- Third-Party Imports ---
import pandas as pd
import pandas_ta as ta
import joblib

# --- Local Application Imports ---
from dropzilla.config import (
    POLYGON_API_KEY, DATA_CONFIG, MODEL_CONFIG,
    LABELING_CONFIG, FEATURE_CONFIG
)
from dropzilla.data import PolygonDataClient
from dropzilla.features import calculate_features
from dropzilla.labeling import get_triple_barrier_labels
from dropzilla.validation import PurgedKFold
from dropzilla.models import optimize_hyperparameters, train_lightgbm_model
from dropzilla.context import get_market_regimes
from dropzilla.signal import train_meta_model
from dropzilla.correlation import get_systemic_absorption_ratio
from dropzilla.volatility import get_volatility_regime_anomaly

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

import argparse

def generate_meta_dataset(
    model_artifact_path: str,
    data_df: pd.DataFrame,
    probability_threshold: float = 0.20
) -> pd.DataFrame | None:
    """
    Generates a dataset for training the meta-model.
    """
    print(f"\n--- Generating Meta-Model Dataset from {model_artifact_path} ---")
    try:
        artifact = joblib.load(model_artifact_path)
    except FileNotFoundError:
        print(f"Error: Model artifact not found at {model_artifact_path}")
        return None

    primary_model   = artifact['model']
    features_to_use = artifact['features_to_use']

    X = data_df[features_to_use]
    # get primary probabilities, uncertainty, candidate mask
    primary_probabilities   = primary_model.predict_proba(X)[:, 1]
    model_uncertainty       = 1 - 2 * np.abs(primary_probabilities - 0.5)
    primary_candidates      = (primary_probabilities >= probability_threshold).astype(int)

    meta_df = data_df.copy()
    meta_df['primary_model_candidate']   = primary_candidates
    meta_df['primary_model_probability'] = primary_probabilities
    meta_df['model_uncertainty']         = model_uncertainty
    meta_df['meta_target']               = (
        meta_df['primary_model_candidate'] == meta_df['drop_label']
    ).astype(int)

    filtered = meta_df[meta_df['primary_model_candidate'] == 1].copy()
    print(
        f"Found {len(filtered)} candidate signals with p ≥ "
        f"{probability_threshold:.2f} for meta-model training."
    )
    return filtered


def main() -> None:
    """Main training pipeline execution function."""
    parser = argparse.ArgumentParser(description="Dropzilla v4 Training Pipeline")
    parser.add_argument('--symbols', type=str, default="AAPL,MSFT,NVDA,TSLA,GOOG", help="Comma-separated list of symbols to train on")
    parser.add_argument('--device', type=str, default="cpu", choices=["cpu", "gpu"], help="Device for LightGBM: cpu or gpu")
    parser.add_argument('--cv', type=int, default=5, help="Number of CV splits for PurgedKFold")
    args = parser.parse_args()

    # Update configs based on args
    symbols_to_train = [s.strip() for s in args.symbols.split(',')]
    MODEL_CONFIG['cv_n_splits'] = args.cv

    print("--- Starting Dropzilla v4 ADVANCED Training Pipeline ---")
    print(f"Using symbols: {symbols_to_train}")
    print(f"Device: {args.device}")
    print(f"CV splits: {args.cv}")

    # 1. Initialization
    client           = PolygonDataClient(api_key=POLYGON_API_KEY)
    to_date          = datetime.now()
    from_date        = to_date - timedelta(days=DATA_CONFIG['data_period_days'])

    # 2. Data Collection
    print("Fetching all market data...")
    market_data = {
        sym: client.get_aggs(
            sym,
            from_date=from_date.strftime('%Y-%m-%d'),
            to_date=to_date.strftime('%Y-%m-%d')
        )
        for sym in symbols_to_train
    }
    market_data = {k: v for k, v in market_data.items() if v is not None and not v.empty}

    spy_df = client.get_aggs(
        "SPY",
        from_date=(from_date - timedelta(days=50)).strftime('%Y-%m-%d'),
        to_date=to_date.strftime('%Y-%m-%d'),
        timespan='day'
    )
    if spy_df is not None and not spy_df.empty:
        spy_df['market_regime'] = get_market_regimes(spy_df)

    # 3. Feature Engineering + Labeling per Symbol
    all_labeled = []
    daily_returns_panel = pd.DataFrame({
        sym: df['Close'].resample('D').last().pct_change(fill_method=None)
        for sym, df in market_data.items()
    }).dropna(how='all')
    sar_scores = get_systemic_absorption_ratio(daily_returns_panel)
    sar_scores = pd.Series(sar_scores, index=daily_returns_panel.index)  # Ensure datetime index

    for sym, df in market_data.items():
        print(f"\nProcessing features and labels for {sym}...")

        # volatility-regime anomaly
        daily_rv   = df['Close'].resample('D').last().pct_change(fill_method=None) \
                     .rolling(window=21).std() * np.sqrt(252)
        vra_scores = get_volatility_regime_anomaly(daily_rv.dropna())
        vra_scores = pd.Series(vra_scores, index=daily_rv.dropna().index)  # Ensure datetime index

        # merge context: market regime, SAR, VRA
        ctx = df.sort_index()
        ctx = pd.merge_asof(ctx, spy_df[['market_regime']].dropna(),
                            left_index=True, right_index=True, direction='backward')
        ctx = pd.merge_asof(ctx, sar_scores.to_frame('sar_score'),
                            left_index=True, right_index=True, direction='backward')
        ctx = pd.merge_asof(ctx, vra_scores.to_frame('vra_score'),
                            left_index=True, right_index=True, direction='backward')
        ctx[['sar_score','vra_score']] = ctx[['sar_score','vra_score']].ffill().fillna(0)

        # compute features
        daily_close = df['Close'].resample('D').last()
        dlogret = np.log(1 + daily_close.pct_change(fill_method=None)).dropna()
        features_df = calculate_features(ctx, dlogret, FEATURE_CONFIG)

        # single-sided drop-only triple-barrier labels
        atr                   = ta.atr(
            high=features_df['High'], low=features_df['Low'],
            close=features_df['Close'], length=LABELING_CONFIG['atr_period']
        )
        target_volatility     = atr.reindex(features_df.index).bfill()
        vbar                  = features_df.index.to_series() + pd.Timedelta(
                                   minutes=LABELING_CONFIG['vertical_barrier_minutes']
                               )
        side_series = pd.Series(-1, index=features_df.index)

        labels = get_triple_barrier_labels(
            prices=features_df['Close'],
            t_events=features_df.index,
            pt_sl=[
                LABELING_CONFIG['profit_take_atr_multiplier'],
                LABELING_CONFIG['stop_loss_atr_multiplier']
            ],
            target=target_volatility,
            vertical_barrier_times=vbar,
            side=side_series,
        )

        # Print bin counts for debugging
        print(f"Label bin counts for {sym}:")
        print(labels['bin'].value_counts())

        # map: bin = 1 → 1 (good drop for short), others → 0
        features_df['drop_label'] = labels['bin'].map({1: 1}).fillna(0).astype(int)
        features_df['label_time'] = labels['t1']

        all_labeled.append(features_df)

    # 4. Assemble & Clean
    final_df = pd.concat(all_labeled)
    features_to_use = [
        'relative_volume','distance_from_vwap_pct','vwap_slope',
        'roc_30','roc_60','roc_120','rsi_14','rsi_14_sma_5',
        'macd_line','macd_signal','macd_hist','macd_hist_diff',
        'mfi_14','obv_slope','market_regime','volatility_surprise',
        'sar_score','vra_score'
    ]
    final_df = final_df.dropna(subset=features_to_use + ['drop_label','label_time'])
    if final_df.empty:
        print("ERROR: No data remaining after feature calc. Exiting.")
        return

    X           = final_df[features_to_use]
    y           = final_df['drop_label']
    label_times = final_df['label_time']

    print(f"\nData ready for training. Shape: {X.shape}")
    print(f"Label distribution:\n{y.value_counts(normalize=True)}")

    # 5. Hyperopt + Primary model
    print("\n--- Starting Hyperparameter Optimization ---")
    cv_validator = PurgedKFold(
        n_splits=MODEL_CONFIG['cv_n_splits'],
        label_times=label_times,
        embargo_pct=MODEL_CONFIG['cv_embargo_pct']
    )
    best_params, _ = optimize_hyperparameters(
        X, y, cv_validator,
        max_evals=MODEL_CONFIG['optimization_max_evals']
    )

    print("\n--- Training Final Primary Model on All Data ---")
    final_params = best_params.copy()
    final_params['device_type'] = args.device  # Set device for LightGBM
    for p in ['n_estimators','num_leaves','max_depth','min_child_samples']:
        if p in final_params:
            final_params[p] = int(final_params[p])
    final_model = train_lightgbm_model(X, y, params=final_params)
    print("Final primary model training complete.")

    # save artifact
    artifact = {
        "model": final_model,
        "best_params": best_params,
        "features_to_use": features_to_use,
        "training_timestamp_utc": datetime.utcnow().isoformat(),
        "dropzilla_version": "4.0"
    }
    fname = MODEL_CONFIG['model_filename']
    joblib.dump(artifact, fname)
    print(f"✅ Primary model artifact saved to: {fname}")

    # 6. Meta-model (optional)
    meta_df = generate_meta_dataset(fname, final_df, probability_threshold=0.50)
    if meta_df is not None and not meta_df.empty:
        meta_model = train_meta_model(meta_df)
        artifact['meta_model']          = meta_model
        artifact['meta_model_features'] = [
            'primary_model_probability','relative_volume','market_regime',
            'model_uncertainty','sar_score','vra_score'
        ]
        joblib.dump(artifact, fname)
        print(f"✅ Meta-model trained and added to artifact: {fname}")
    else:
        print("Could not generate meta-dataset. Skipping meta-model training.")

    print("\n--- Full Pipeline Complete ---")


if __name__ == "__main__":
    main()
