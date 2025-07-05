"""
Runs the Dropzilla v4 prediction pipeline in a continuous loop to monitor
a watchlist of stocks for trading signals.
"""
import time
import argparse
from datetime import datetime

# We can import the get_prediction function directly from our existing script
from run_prediction import get_prediction

def live_monitor(watchlist: list[str], model_path: str, interval_minutes: int, confidence_threshold: float):
    """
    Continuously monitors a watchlist of stocks for trading signals.

    Args:
        watchlist (list[str]): A list of stock tickers to monitor.
        model_path (str): The path to the saved model artifact.
        interval_minutes (int): The number of minutes to wait between each loop.
        confidence_threshold (float): The minimum confidence score to report.
    """
    print("--- Starting Dropzilla v4 Live Monitor ---")
    print(f"Watchlist: {watchlist}")
    print(f"Monitoring Interval: {interval_minutes} minutes")
    print(f"Confidence Threshold: {confidence_threshold:.2%}")
    print("-----------------------------------------")

    while True:
        try:
            print(f"\n--- Running new scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
            
            for symbol in watchlist:
                prediction_result = get_prediction(symbol, model_path)
                
                if prediction_result and prediction_result.get('final_confidence_score', 0) >= confidence_threshold:
                    print("\n" + "="*40)
                    print(f"🔥🔥🔥 HIGH CONFIDENCE SIGNAL DETECTED 🔥🔥🔥")
                    print(f"  Symbol: {prediction_result['symbol']}")
                    print(f"  Timestamp (UTC): {prediction_result['timestamp_utc']}")
                    print(f"  Final Confidence: {prediction_result['final_confidence_score']:.2%}")
                    print(f"  Primary Probability: {prediction_result['primary_probability']:.4f}")
                    print(f"  Market Regime: {prediction_result['market_regime']}")
                    print("="*40 + "\n")
            
            print(f"\nScan complete. Waiting for {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
            break
        except Exception as e:
            print(f"An error occurred in the monitoring loop: {e}")
            print("Restarting loop in 60 seconds...")
            time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dropzilla v4 Live Monitor.")
    parser.add_argument(
        "--symbols",
        type=str,
        default="AAPL,MSFT,NVDA,TSLA,GOOG",
        help="A comma-separated list of stock tickers to monitor."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dropzilla_v4_lgbm.pkl",
        help="Path to the model artifact file."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="The monitoring interval in minutes."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50, # Default to reporting signals with > 50% confidence
        help="The minimum confidence score (as a decimal, e.g., 0.5) to report."
    )
    args = parser.parse_args()
    
    watchlist = [symbol.strip().upper() for symbol in args.symbols.split(',')]
    live_monitor(watchlist, args.model, args.interval, args.threshold)
