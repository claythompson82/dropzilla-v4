"""
Main script for running live predictions with a trained Dropzilla v4 model.

This script loads the complete model artifact (primary + meta-model),
fetches the latest data for a given symbol, and generates a final,
principled confidence score.
"""
# --- Compatibility Shim ---
import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan

# --- Standard Library Imports ---
import argparse
from datetime import datetime, timedelta

# --- Third-Party Imports ---
import pandas as pd
import joblib

# --- Local Application Imports ---
from dropzilla.config import POLYGON_API_KEY
from dropzilla.data import PolygonDataClient
from dropzilla.features import calculate_features
from dropzilla.context import get_market_regimes

def get_prediction(symbol: str, model_artifact_path: str) -> dict | None:
    """
    Generates a drop prediction and confidence score for a single symbol.

    Args:
        symbol (str): The stock ticker to predict on.
        model_artifact_path (str): The path to the saved model artifact.

    Returns:
        dict | None: A dictionary with prediction details or None if an error occurs.
    """
    print(f"\n--- Generating Prediction for {symbol} ---")

    # 1. Load Model Artifact
    try:
        artifact = joblib.load(model_artifact_path)
        primary_model = artifact['model']
        meta_model = artifact['meta_model']
        features_to_use = artifact['features_to_use']
        meta_features_to_use = artifact['meta_model_features']
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading model artifact: {e}")
        return None

    # 2. Fetch Latest Data
    data_client = PolygonDataClient(api_key=POLYGON_API_KEY)
    to_date = datetime.now()
    from_date = to_date - timedelta(days=5)
    
    latest_data = data_client.get_aggs(
        symbol,
        from_date=from_date.strftime('%Y-%m-%d'),
        to_date=to_date.strftime('%Y-%m-%d')
    )
    if latest_data is None or latest_data.empty:
        print(f"Could not fetch recent data for {symbol}.")
        return None

    # 3. Add Market Context
    print("Fetching market context (SPY)...")
    spy_df = data_client.get_aggs("SPY", from_date=(to_date - timedelta(days=50)).strftime('%Y-%m-%d'), to_date=to_date.strftime('%Y-%m-%d'), timespan='day')
    if spy_df is not None and not spy_df.empty:
        spy_df['market_regime'] = get_market_regimes(spy_df)
        latest_data = pd.merge_asof(
            latest_data.sort_index(), spy_df[['market_regime']].dropna(),
            left_index=True, right_index=True, direction='backward'
        )

    # 4. Calculate Features
    features_df = calculate_features(latest_data)
    
    latest_features = features_df.iloc[-1]
    X_live = latest_features[features_to_use].to_frame().T

    # --- THE FIX ---
    # Ensure all feature columns are the correct numeric type before prediction.
    X_live = X_live.astype(float)
    # --- END FIX ---

    # 5. Run Primary Model
    primary_prob = primary_model.predict_proba(X_live)[:, 1][0]
    print(f"Primary model probability: {primary_prob:.4f}")

    # 6. Run Meta-Model for Final Confidence
    meta_features_data = {
        'primary_model_probability': primary_prob,
        'relative_volume': latest_features.get('relative_volume', 1.0),
        'market_regime': latest_features.get('market_regime', 0)
    }
    meta_features = pd.DataFrame([meta_features_data])
    # Ensure dtypes match for meta-model as well
    meta_features = meta_features[meta_features_to_use].astype(float)
    
    final_confidence = meta_model.predict_proba(meta_features)[:, 1][0]

    result = {
        "symbol": symbol,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "primary_probability": primary_prob,
        "final_confidence_score": final_confidence,
        "market_regime": latest_features.get('market_regime', 'N/A'),
        "details": latest_features.to_dict()
    }
    
    print(f"\n--- Prediction Complete for {symbol} ---")
    print(f"Final Confidence Score: {final_confidence:.2%}")
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dropzilla v4 Live Prediction.")
    parser.add_argument("symbol", type=str, help="The stock ticker symbol to predict (e.g., AAPL).")
    parser.add_argument("--model", type=str, default="dropzilla_v4_lgbm.pkl", help="Path to the model artifact file.")
    args = parser.parse_args()

    get_prediction(args.symbol.upper(), args.model)
