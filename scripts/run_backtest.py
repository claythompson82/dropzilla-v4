"""
Runs a financial backtest of a trained Dropzilla v4 model to evaluate
the economic performance of its signals.
"""
import argparse
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# --- Local Application Imports ---
from dropzilla.config import POLYGON_API_KEY, DATA_CONFIG, FEATURE_CONFIG
from dropzilla.data import PolygonDataClient
from dropzilla.features import calculate_features
from dropzilla.context import get_market_regimes
from dropzilla.correlation import get_systemic_absorption_ratio
from dropzilla.volatility import get_volatility_regime_anomaly

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

    # 2. Data Collection (Mirrors the training script logic)
    data_client = PolygonDataClient(api_key=POLYGON_API_KEY)
    symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOG"]
    to_date = datetime.now()
    from_date = to_date - timedelta(days=DATA_CONFIG['data_period_days'])
    
    print("Fetching all market data...")
    market_data = {
        symbol: data_client.get_aggs(symbol, from_date.strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d'))
        for symbol in symbols
    }
    market_data = {k: v for k, v in market_data.items() if v is not None and not v.empty}
    
    spy_df = data_client.get_aggs("SPY", from_date=(from_date - timedelta(days=50)).strftime('%Y-%m-%d'), to_date=to_date.strftime('%Y-%m-%d'), timespan='day')
    if spy_df is not None and not spy_df.empty:
        spy_df['market_regime'] = get_market_regimes(spy_df)

    # 3. Feature Engineering (Mirrors the training script logic)
    all_featured_data = []
    daily_returns_panel = pd.DataFrame({
        symbol: df['Close'].resample('D').last().pct_change() for symbol, df in market_data.items()
    }).dropna(how='all')
    sar_scores = get_systemic_absorption_ratio(daily_returns_panel)

    for symbol, df in market_data.items():
        print(f"\nProcessing features for {symbol}...")
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

    # 4. Generate Signals for Entire Dataset
    print("Generating historical signals...")
    X = backtest_df[features_to_use]
    primary_probs = primary_model.predict_proba(X)[:, 1]
    
    model_uncertainty = 1 - 2 * np.abs(primary_probs - 0.5)

    meta_features_df = pd.DataFrame({
        'primary_model_probability': primary_probs,
        'relative_volume': backtest_df['relative_volume'],
        'market_regime': backtest_df['market_regime'],
        'model_uncertainty': model_uncertainty,
        'sar_score': backtest_df['sar_score'],
        'vra_score': backtest_df['vra_score']
    })
    final_confidences = meta_model.predict_proba(meta_features_df[meta_features_to_use])[:, 1]
    
    backtest_df['confidence'] = final_confidences
    backtest_df['signal'] = (backtest_df['confidence'] > confidence_threshold).astype(int)

    # 5. Simulate Trades and Calculate Returns
    print("Simulating trades...")
    backtest_df['forward_return'] = backtest_df.groupby('symbol')['Close'].pct_change(periods=-60).shift(60)
    
    trade_returns = backtest_df[backtest_df['signal'] == 1]['forward_return']
    trade_returns = -trade_returns # Invert for short positions

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
    parser.add_argument("--model", type=str, default="dropzilla_v4_lgbm.pkl", help="Path to the model artifact file.")
    parser.add_argument("--threshold", type=float, default=0.55, help="The minimum confidence score (as a decimal, e.g., 0.55) to simulate a trade.")
    args = parser.parse_args()
    run_backtest(args.model, args.threshold)
