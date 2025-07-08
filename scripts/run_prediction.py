"""
Main script for running live predictions with a trained Dropzilla v4 model.
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
from dropzilla.config import POLYGON_API_KEY, FEATURE_CONFIG
from dropzilla.data import PolygonDataClient
from dropzilla.features import calculate_features
from dropzilla.context import get_market_regimes
from dropzilla.correlation import get_systemic_absorption_ratio

def get_prediction(symbol: str, model_artifact_path: str) -> dict | None:
    """
    Generates a drop prediction and confidence score for a single symbol.
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

    # 2. Fetch Data
    data_client = PolygonDataClient(api_key=POLYGON_API_KEY)
    to_date = datetime.now()
    # Fetch enough historical data for rolling calculations
    from_date = to_date - timedelta(days=90) 
    
    # Fetch data for the target symbol and the peer group for SAR
    symbols_to_fetch = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOG"]
    if symbol.upper() not in symbols_to_fetch:
        symbols_to_fetch.append(symbol.upper())

    all_minute_data = {
        s: data_client.get_aggs(s, from_date.strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d'))
        for s in symbols_to_fetch
    }
    
    latest_data = all_minute_data[symbol.upper()]
    if latest_data is None or latest_data.empty:
        print(f"Could not fetch recent data for {symbol}.")
        return None
        
    daily_df = latest_data.resample('D').agg({'Close': 'last'}).dropna()
    daily_log_returns = np.log(daily_df['Close'] / daily_df['Close'].shift(1)).dropna()

    # 3. Add Context Features
    # SAR Score
    daily_returns_panel = pd.DataFrame({
        s: df['Close'].resample('D').last().pct_change() for s, df in all_minute_data.items() if df is not None
    }).dropna(how='all')
    sar_scores = get_systemic_absorption_ratio(daily_returns_panel)
    latest_data = pd.merge_asof(latest_data.sort_index(), sar_scores.to_frame(name='sar_score'), left_index=True, right_index=True, direction='backward')

    # Market Regime
    spy_df = data_client.get_aggs(
        "SPY",
        from_date=(from_date - timedelta(days=50)).strftime('%Y-%m-%d'),
        to_date=to_date.strftime('%Y-%m-%d'),
        timespan='day'
    )
    if spy_df is not None and not spy_df.empty:
        spy_df['market_regime'] = get_market_regimes(spy_df)
        latest_data = pd.merge_asof(latest_data.sort_index(), spy_df[['market_regime']].dropna(), left_index=True, right_index=True, direction='backward')

    # 4. Calculate Features
    features_df = calculate_features(latest_data, daily_log_returns, FEATURE_CONFIG)
    
    latest_features = features_df.iloc[-1]
    X_live = latest_features[features_to_use].to_frame().T
    X_live = X_live.astype(float)

    # 5. Run Primary Model
    primary_prob = primary_model.predict_proba(X_live)[:, 1][0]
    print(f"Primary model probability: {primary_prob:.4f}")

    # 6. Run Meta-Model for Final Confidence
    model_uncertainty = 1 - 2 * np.abs(primary_prob - 0.5)
    meta_features_data = {
        'primary_model_probability': primary_prob,
        'relative_volume': latest_features.get('relative_volume', 1.0),
        'market_regime': latest_features.get('market_regime', 0),
        'model_uncertainty': model_uncertainty,
        'sar_score': latest_features.get('sar_score', 0.5) # Add SAR
    }
    meta_features = pd.DataFrame([meta_features_data])
    meta_features = meta_features[meta_features_to_use].astype(float)
    
    final_confidence = meta_model.predict_proba(meta_features)[:, 1][0]

    result = {
        "symbol": symbol, "timestamp_utc": datetime.utcnow().isoformat(),
        "primary_probability": primary_prob, "final_confidence_score": final_confidence,
        "market_regime": int(latest_features.get('market_regime', -1)),
        "sar_score": latest_features.get('sar_score', -1),
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
