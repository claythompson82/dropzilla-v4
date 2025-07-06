"""
Runs a financial backtest of a trained Dropzilla v4 model to evaluate
the economic performance of its signals.
"""
import argparse
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

from dropzilla.config import POLYGON_API_KEY, DATA_CONFIG
from dropzilla.data import PolygonDataClient
from dropzilla.features import calculate_features
from dropzilla.context import get_market_regimes

def run_backtest(model_artifact_path: str, confidence_threshold: float):
    """
    Runs a vector-based backtest on historical data.

    Args:
        model_artifact_path (str): Path to the saved model artifact.
        confidence_threshold (float): The confidence level required to simulate a trade.
    """
    print("--- Starting Dropzilla v4 Financial Backtest ---")

    # 1. Load Model and Data
    print("Loading model and historical data...")
    try:
        artifact = joblib.load(model_artifact_path)
        primary_model = artifact['model']
        meta_model = artifact['meta_model']
        features_to_use = artifact['features_to_use']
        meta_features_to_use = artifact['meta_model_features']
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading model artifact: {e}")
        return

    data_client = PolygonDataClient(api_key=POLYGON_API_KEY)
    symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOG"]
    to_date = datetime.now()
    from_date = to_date - timedelta(days=DATA_CONFIG['data_period_days'])
    
    print("Fetching context data...")
    spy_df = data_client.get_aggs("SPY", from_date=(from_date - timedelta(days=50)).strftime('%Y-%m-%d'), to_date=to_date.strftime('%Y-%m-%d'), timespan='day')
    if spy_df is not None and not spy_df.empty:
        spy_df['market_regime'] = get_market_regimes(spy_df)

    all_data = []
    for symbol in symbols:
        print(f"Fetching full dataset for {symbol}...")
        df = data_client.get_aggs(symbol, from_date=from_date.strftime('%Y-%m-%d'), to_date=to_date.strftime('%Y-%m-%d'))
        if df is None or df.empty: continue
        
        daily_df = df.resample('D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
        daily_log_returns = np.log(daily_df['Close'] / daily_df['Close'].shift(1)).dropna()
        
        group_with_context = pd.merge_asof(df.sort_index(), spy_df[['market_regime']].dropna(), left_index=True, right_index=True, direction='backward')
        
        features_df = calculate_features(group_with_context, daily_log_returns)
        features_df['symbol'] = symbol
        all_data.append(features_df)
    
    backtest_df = pd.concat(all_data).dropna(subset=features_to_use)
    print(f"Loaded {len(backtest_df)} total data points for backtest.")

    # 2. Generate Signals for Entire Dataset
    print("Generating historical signals...")
    X = backtest_df[features_to_use]
    primary_probs = primary_model.predict_proba(X)[:, 1]
    
    model_uncertainty = 1 - 2 * np.abs(primary_probs - 0.5)

    meta_features_df = pd.DataFrame({
        'primary_model_probability': primary_probs,
        'relative_volume': backtest_df['relative_volume'],
        'market_regime': backtest_df['market_regime'],
        'model_uncertainty': model_uncertainty
    })
    final_confidences = meta_model.predict_proba(meta_features_df[meta_features_to_use])[:, 1]
    
    backtest_df['confidence'] = final_confidences
    backtest_df['signal'] = (backtest_df['confidence'] > confidence_threshold).astype(int)

    # 3. Simulate Trades and Calculate Returns
    print("Simulating trades...")
    # Calculate the forward return for each minute to evaluate trade profitability
    backtest_df['forward_return'] = backtest_df.groupby('symbol')['Close'].pct_change(periods=-60).shift(60)
    
    trade_returns = backtest_df[backtest_df['signal'] == 1]['forward_return']
    trade_returns = -trade_returns # Invert for short positions

    # 4. Calculate and Print Performance Metrics
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
    sharpe_ratio = (trade_returns.mean() / trade_returns.std()) * np.sqrt(252 * 390) if trade_returns.std() > 0 else 0

    print(f"Confidence Threshold: {confidence_threshold:.2%}")
    print(f"Total Trades Generated: {num_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Trade Return: {avg_trade_return:.4%}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print("------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dropzilla v4 Financial Backtest.")
    parser.add_argument(
        "--model",
        type=str,
        default="dropzilla_v4_lgbm.pkl",
        help="Path to the model artifact file."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55, # Default to a 55% confidence threshold for analysis
        help="The minimum confidence score (as a decimal, e.g., 0.55) to simulate a trade."
    )
    args = parser.parse_args()
    run_backtest(args.model, args.threshold)
