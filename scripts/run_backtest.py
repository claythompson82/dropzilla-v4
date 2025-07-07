"""
Runs a financial backtest using a Multi-Factor Confidence Score,
bypassing the meta-model to create a more robust conviction signal.
"""
import argparse
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# --- Local Application Imports ---
from dropzilla.config import POLYGON_API_KEY, DATA_CONFIG, FEATURE_CONFIG, MODEL_CONFIG
from dropzilla.data import PolygonDataClient
from dropzilla.features import calculate_features
from dropzilla.context import get_market_regimes
from dropzilla.correlation import get_systemic_absorption_ratio
from dropzilla.volatility import get_volatility_regime_anomaly

def run_backtest(model_artifact_path: str, confidence_threshold: float):
    """
    Runs a vector-based backtest on historical data using a multi-factor score.
    """
    print("--- Starting Dropzilla v4 Financial Backtest (Multi-Factor Score) ---")

    # 1. Load Model (We only need the primary model now)
    print(f"Loading model artifact from: {model_artifact_path}")
    try:
        artifact = joblib.load(model_artifact_path)
        primary_model = artifact['model']
        features_to_use = artifact['features_to_use']
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading model artifact: {e}.")
        return

    # 2. Data Collection
    data_client = PolygonDataClient(api_key=POLYGON_API_KEY)
    symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOG"]
    to_date = datetime.now()
    from_date = to_date - timedelta(days=DATA_CONFIG.get('data_period_days', 365) + 50)
    market_data = {s: data_client.get_aggs(s, from_date.strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d')) for s in symbols}
    market_data = {k: v for k, v in market_data.items() if v is not None and not v.empty}
    spy_df = data_client.get_aggs("SPY", from_date.strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d'), timespan='day')
    if spy_df is not None and not spy_df.empty:
        spy_df['market_regime'] = get_market_regimes(spy_df)

    # 3. Feature Engineering
    print("Engineering features for backtest data...")
    all_featured_data = []
    # ... (This section remains the same as it correctly builds the features needed)
    daily_returns_panel = pd.DataFrame({s: df['Close'].resample('D').last().pct_change() for s, df in market_data.items()}).dropna(how='all')
    sar_scores = get_systemic_absorption_ratio(daily_returns_panel)
    for symbol, df in market_data.items():
        daily_rv = df['Close'].resample('D').last().pct_change().rolling(window=21).std() * np.sqrt(252)
        vra_scores = get_volatility_regime_anomaly(daily_rv.dropna())
        df_context = pd.merge_asof(df.sort_index(), spy_df[['market_regime']].dropna(), left_index=True, right_index=True, direction='backward')
        df_context = pd.merge_asof(df_context, sar_scores.to_frame(name='sar_score'), left_index=True, right_index=True, direction='backward')
        df_context = pd.merge_asof(df_context, vra_scores.to_frame(name='vra_score'), left_index=True, right_index=True, direction='backward')
        df_context[['sar_score', 'vra_score']] = df_context[['sar_score', 'vra_score']].ffill().fillna(0)
        daily_log_returns = np.log(df['Close'].resample('D').last().pct_change()).dropna()
        features_df = calculate_features(df_context, daily_log_returns, FEATURE_CONFIG)
        features_df['symbol'] = symbol
        all_featured_data.append(features_df)

    backtest_df = pd.concat(all_featured_data).dropna(subset=features_to_use)
    print(f"\nLoaded {len(backtest_df)} total data points for backtest.")
    if backtest_df.empty:
        print("No data available for backtesting. Exiting.")
        return

    # 4. Generate Primary Probabilities
    print("Generating primary model probabilities...")
    X_primary = backtest_df[features_to_use]
    primary_probs = primary_model.predict_proba(X_primary)[:, 1]

    # --- NEW: Multi-Factor Confidence Score Calculation ---
    print("Calculating Multi-Factor Confidence Score...")
    # Component 1: Calibrated Probability (from the primary model)
    p_calibrated = pd.Series(primary_probs, index=backtest_df.index)

    # Component 2: Signal Stability (lower standard deviation = higher stability)
    # We use a simple rolling standard deviation as a proxy for the Kalman Filter for now
    prob_stability = 1 - (p_calibrated.rolling(window=5).std() / 0.5).clip(0, 1)
    prob_stability.fillna(0.5, inplace=True) # Fill initial NaNs with neutral stability

    # Component 3: Regime Context
    # We give a higher score to signals that occur in a bearish regime
    regime_map = {0: 0.8, 1: 0.5, 2: 1.0} # Assuming 2 might be 'Crisis' or 'Bear Trend'
    r_context = backtest_df['market_regime'].map(regime_map).fillna(0.5)

    # Component 4: Volume Confirmation
    # Score is clipped at 1.0, rewarding signals on high relative volume
    c_confirmation = (backtest_df['relative_volume'] / 2.0).clip(0, 1).fillna(0.5)

    # Combine components with weights
    W_prob = 0.40
    W_stab = 0.15
    W_regime = 0.25
    W_conf = 0.20

    final_confidences = (W_prob * p_calibrated + 
                         W_stab * prob_stability + 
                         W_regime * r_context + 
                         W_conf * c_confirmation)
    # --- END NEW LOGIC ---

    backtest_df['confidence'] = final_confidences
    backtest_df['signal'] = (backtest_df['confidence'] > confidence_threshold).astype(int)

    # 5. Simulate Trades
    print("Simulating trades...")
    backtest_df['forward_return'] = backtest_df.groupby('symbol')['Close'].pct_change(periods=-240).shift(240)
    trade_returns = backtest_df[backtest_df['signal'] == 1]['forward_return']
    trade_returns = -trade_returns

    # 6. Calculate and Print Performance Metrics
    print("\n--- Backtest Performance Metrics ---")
    num_trades = len(trade_returns)
    if num_trades == 0:
        print(f"No trades were generated at this confidence threshold ({confidence_threshold:.2%}).")
        return

    gross_profit = trade_returns[trade_returns > 0].sum()
    gross_loss = np.abs(trade_returns[trade_returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    win_rate = (trade_returns > 0).mean()
    avg_trade_return = trade_returns.mean()
    sharpe_ratio = (trade_returns.mean() / trade_returns.std()) * np.sqrt(252 * (390/240)) if trade_returns.std() > 0 else 0

    print(f"Confidence Threshold: {confidence_threshold:.2%}")
    print(f"Total Trades Generated: {num_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Trade Return: {avg_trade_return:.4%}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print("------------------------------------")

    # Also print the new confidence distribution for analysis
    print("\n--- New Multi-Factor Score Distribution ---")
    print(final_confidences.describe(percentiles=[.25, .5, .75, .9, .95, .99]))
    print("-----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dropzilla v4 Financial Backtest.")
    parser.add_argument("--model", type=str, default="dropzilla_v4_lgbm.pkl", help="Path to the model artifact file.")
    parser.add_argument("--threshold", type=float, default=0.55, help="The minimum confidence score to simulate a trade.")
    parser.add_argument("--use_gpu", action="store_true", help="Enable GPU acceleration")
    args = parser.parse_args()
    MODEL_CONFIG["use_gpu"] = args.use_gpu
    run_backtest(args.model, args.threshold)
