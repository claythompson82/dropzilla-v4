# Filename: dropzilla/run_gui.py
#!/usr/bin/env python
"""
Dropzilla v4 GUI: A live signal engine interface for intraday short-biased trading.
Fetches the latest data from Polygon, processes features, runs the model, and displays
signals with confidence scores and key features. Supports off-hours prediction testing
(minutes-to-hours holds, no overnight positions).
"""

# --- Standard Library Imports ---
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import subprocess
import time
import queue
from datetime import datetime, timedelta
import os
import pytz
import logging

# --- Third-Party Imports ---
import pandas as pd
import numpy as np
import joblib
import pandas_ta as ta
from PIL import Image, ImageTk
from joblib import Parallel, delayed

# --- Local Application Imports ---
from dropzilla.config import POLYGON_API_KEY, FEATURE_CONFIG, LABELING_CONFIG, DATA_CONFIG
from dropzilla.data import PolygonDataClient
from dropzilla.features import calculate_features
from dropzilla.context import get_market_regimes
from dropzilla.correlation import get_systemic_absorption_ratio
from dropzilla.volatility import get_volatility_regime_anomaly

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
IMAGE_PATH = "dropzilla_v4_logo.png"
SYMBOLS_DEFAULT = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOG", "AMZN", "META", "AMD", "INTC", "PYPL"]
CONVICTION_THRESHOLD = 0.60  # Optimal from backtests
MARKET_OPEN_HOUR = 9   # ET
MARKET_CLOSE_HOUR = 16 # ET
ET_TZ = pytz.timezone('US/Eastern')
CHECK_INTERVAL_SEC = 480  # 8 minutes for auto-refresh
MIN_DATA_POINTS = 120  # Min rows for features like roc_120

class DropzillaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dropzilla v4 GUI")
        self.root.geometry("1000x700")
        self.root.configure(bg='#D3D3D3')  # Light grey background
        
        self.monitoring = False
        self.training = False
        self.symbols = []
        self.signal_data = []  # List of dicts for table
        self.models = [f for f in os.listdir(".") if f.endswith(".pkl")] + [f for f in os.listdir("models/") if f.endswith(".pkl")]
        self.debug_mode = tk.BooleanVar(value=True)  # Default to debug (show all signals)
        self.parallel_mode = tk.BooleanVar(value=False)  # Default off for parallel
        self.data_cache = {}  # To store and append live data
        self.context_cache = {}  # For caching VRA, etc.
        
        self.create_widgets()
    
    def create_widgets(self):
        # Ticker Input
        tk.Label(self.root, text="Tickers (comma-separated):", bg='#D3D3D3').pack(pady=5)
        self.ticker_entry = tk.Entry(self.root, width=50, bg='#E0E0E0')
        self.ticker_entry.pack()
        
        # Model Dropdown
        tk.Label(self.root, text="Select Model:", bg='#D3D3D3').pack(pady=5)
        self.model_var = tk.StringVar(value=self.models[0] if self.models else "")
        self.model_dropdown = ttk.Combobox(self.root, textvariable=self.model_var, values=self.models, state="readonly", style="TCombobox")
        self.model_dropdown.pack()
        
        # Debug Mode Checkbox
        tk.Checkbutton(self.root, text="Debug Mode (Show All Signals)", variable=self.debug_mode, bg='#D3D3D3').pack(pady=5)
        
        # Parallel Mode Checkbox
        tk.Checkbutton(self.root, text="Enable Parallel Processing", variable=self.parallel_mode, bg='#D3D3D3').pack(pady=5)
        
        # Image
        try:
            img = Image.open(IMAGE_PATH)
            img = img.resize((200, 100), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)
            self.image_label = tk.Label(self.root, image=self.photo, bg='#D3D3D3')
            self.image_label.pack(pady=5)
        except Exception as e:
            self.image_label = tk.Label(self.root, text=f"Image Error: {e}", bg='#D3D3D3')
            self.image_label.pack(pady=5)

        # Buttons Frame
        btn_frame = tk.Frame(self.root, bg='#D3D3D3')
        btn_frame.pack(pady=10)
        
        self.train_btn = tk.Button(btn_frame, text="Train Model", command=self.train_model, bg='#A9A9A9')
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.toggle_btn = tk.Button(btn_frame, text="Start Monitoring", command=self.toggle_monitoring, bg='#A9A9A9')
        self.toggle_btn.pack(side=tk.LEFT, padx=5)
        
        self.refresh_btn = tk.Button(btn_frame, text="Refresh Data", command=self.refresh_data, bg='#A9A9A9')
        self.refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # Signal Table with Feature Columns
        self.tree = ttk.Treeview(self.root, columns=("Ticker", "Timestamp", "Confidence", "Last Price", "Rel Vol", "RSI", "MACD", "SAR"), show="headings")
        self.tree.heading("Ticker", text="Ticker")
        self.tree.heading("Timestamp", text="Timestamp")
        self.tree.heading("Confidence", text="Confidence")
        self.tree.heading("Last Price", text="Last Price")
        self.tree.heading("Rel Vol", text="Rel Vol")
        self.tree.heading("RSI", text="RSI")
        self.tree.heading("MACD", text="MACD")
        self.tree.heading("SAR", text="SAR")
        self.tree.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Highlight tags
        self.tree.tag_configure('yellow', background='#FFFF00', foreground='black')  # 40-50%
        self.tree.tag_configure('orange', background='#FFA500', foreground='black')  # 50-60%
        self.tree.tag_configure('red', background='#FF0000', foreground='white')    # 60%+
        
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self.tree.bind("<Double-1>", self.on_double_click)  # New bind for popup
        
        # Detail Panel
        self.detail_frame = tk.Frame(self.root, bg='#D3D3D3')
        self.detail_frame.pack(fill=tk.X, pady=5)
        
        self.prob_label = tk.Label(self.detail_frame, text="Probability: N/A", bg='#D3D3D3')
        self.prob_label.pack(anchor="w")
        
        self.stab_label = tk.Label(self.detail_frame, text="Stability: N/A", bg='#D3D3D3')
        self.stab_label.pack(anchor="w")
        
        # Context Labels
        self.regime_label = tk.Label(self.root, text="Market Regime: N/A", bg='#D3D3D3')
        self.regime_label.pack(anchor="w")
        
        self.sar_label = tk.Label(self.root, text="SAR: N/A", bg='#D3D3D3')
        self.sar_label.pack(anchor="w")
        
        # Insights Text
        self.insights_text = scrolledtext.ScrolledText(self.root, height=5, wrap=tk.WORD, bg='#E0E0E0')
        self.insights_text.pack(fill=tk.X, pady=5)
        self.insights_text.insert(tk.END, "No insights yet.")
    
    def parse_tickers(self):
        tickers = self.ticker_entry.get().upper().split(',')
        self.symbols = [t.strip() for t in tickers if t.strip()]
    
    def toggle_monitoring(self):
        self.parse_tickers()
        if not self.symbols:
            messagebox.showerror("Error", "Enter tickers first!")
            return
        
        self.monitoring = not self.monitoring
        self.toggle_btn.config(text="Stop Monitoring" if self.monitoring else "Start Monitoring")
        
        if self.monitoring:
            model_path = os.path.join(".", self.model_var.get()) if self.model_var.get() in os.listdir(".") else os.path.join("models", self.model_var.get())
            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model {self.model_var.get()} not found!")
                self.monitoring = False
                self.toggle_btn.config(text="Start Monitoring")
                return
            threading.Thread(target=self.monitor_thread, args=(model_path,), daemon=True).start()
    
    def refresh_data(self):
        """Manually refresh data and update table."""
        self.parse_tickers()
        if not self.symbols:
            messagebox.showerror("Error", "Enter tickers first!")
            return
        
        model_path = os.path.join(".", self.model_var.get()) if self.model_var.get() in os.listdir(".") else os.path.join("models", self.model_var.get())
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model {self.model_var.get()} not found!")
            return
        self.monitor_loop(model_path, refresh_only=True)
    
    def monitor_thread(self, model_path):
        while self.monitoring:
            self.monitor_loop(model_path)
            if self.is_market_open():
                time.sleep(CHECK_INTERVAL_SEC)
            else:
                time.sleep(3600)  # Hourly off-hours
    
    def monitor_loop(self, model_path, refresh_only=False):
        self.signal_data = []
        logging.debug(f"Starting monitor_loop for symbols: {self.symbols}")
        print(f"Loading model from {model_path}")
        artifact = joblib.load(model_path)
        primary_model = artifact['model']
        meta_model = artifact.get('meta_model')
        features_to_use = artifact['features_to_use']
        meta_features = artifact.get('meta_model_features', [])
        has_meta = meta_model is not None and meta_features
        print(f"Model loaded. Features: {features_to_use}. Has meta-model: {has_meta}")
        client = PolygonDataClient(POLYGON_API_KEY)
        
        # Fetch data: Historical base + live append if market open
        to_date = datetime.now()
        market_data = {}
        for sym in self.symbols:
            try:
                logging.debug(f"Fetching data for {sym}")
                # For live, fetch from last cached timestamp or short history
                from_date_str = (self.data_cache[sym].index[-1] + timedelta(minutes=1)).strftime('%Y-%m-%d') if sym in self.data_cache and not self.data_cache[sym].empty else (to_date - timedelta(days=1)).strftime('%Y-%m-%d')
                df_new = client.get_aggs(sym, from_date_str, to_date.strftime('%Y-%m-%d'))
                if df_new is not None and not df_new.empty:
                    df_new = df_new.sort_index()
                    # Append to cache
                    if sym in self.data_cache:
                        df = pd.concat([self.data_cache[sym], df_new]).drop_duplicates(keep='last')
                    else:
                        df = df_new
                    self.data_cache[sym] = df
                    market_data[sym] = df
                elif sym in self.data_cache:
                    market_data[sym] = self.data_cache[sym]  # Use cache if no new
                else:
                    logging.error(f"No data for {sym}")
            except Exception as e:
                logging.error(f"Error fetching data for {sym}: {e}")
                if sym in self.data_cache:
                    market_data[sym] = self.data_cache[sym]  # Fallback to cache
        
        # Fetch SPY for regimes with retries
        spy_from_date = to_date - timedelta(days=50)
        spy_df = pd.DataFrame()  # Empty init
        max_retries = 3
        for attempt in range(max_retries):
            try:
                spy_df = client.get_aggs("SPY", spy_from_date.strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d'), timespan='day')
                if spy_df is not None and not spy_df.empty:
                    spy_df = spy_df.sort_index()
                    spy_df['market_regime'] = get_market_regimes(spy_df)
                    print(f"Debug: Last 5 SPY regimes: {spy_df['market_regime'].tail()}")
                    break
                else:
                    logging.warning(f"SPY fetch empty on attempt {attempt+1}/{max_retries}")
                    time.sleep(5 * (attempt + 1))  # Exponential backoff
            except Exception as e:
                logging.error(f"Error fetching SPY data on attempt {attempt+1}: {e}")
                time.sleep(5 * (attempt + 1))

        if spy_df.empty:
            # Longer history fallback
            spy_from_date_extended = to_date - timedelta(days=100)  # Double to 100 days
            try:
                spy_df = client.get_aggs("SPY", spy_from_date_extended.strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d'), timespan='day')
                if spy_df is not None and not spy_df.empty:
                    spy_df = spy_df.sort_index()
                    spy_df['market_regime'] = get_market_regimes(spy_df)
                else:
                    logging.error("Failed all SPY fetches - regimes will be 0.0")
                    print("Error: No SPY data - regimes will be 0.0")
            except Exception as e:
                logging.error(f"Extended SPY fetch failed: {e}")
                spy_df = pd.DataFrame()  # Final empty
        
        # Compute SAR
        try:
            daily_returns_panel = pd.DataFrame({
                sym: df['Close'].resample('D').last().pct_change(fill_method=None)
                for sym, df in market_data.items() if not df.empty
            }).dropna(how='all')
            sar_scores = get_systemic_absorption_ratio(daily_returns_panel)
            sar_scores = pd.Series(sar_scores, index=daily_returns_panel.index)
        except Exception as e:
            logging.error(f"Error computing SAR: {e}")
            sar_scores = pd.Series()  # Empty fallback
        
        def process_symbol(sym, market_data, spy_df, sar_scores, primary_model, features_to_use, has_meta, meta_model, meta_features, MIN_DATA_POINTS, FEATURE_CONFIG):
            try:
                if sym not in market_data or market_data[sym].empty:
                    logging.error(f"No data for {sym}, skipping")
                    return None
                df = market_data[sym]
                if len(df) < MIN_DATA_POINTS:
                    logging.warning(f"Insufficient data for {sym} ({len(df)} rows < {MIN_DATA_POINTS}), skipping")
                    return None
                
                logging.debug(f"Computing features for {sym}")
                # Compute VRA with caching
                if sym in self.context_cache and 'vra_scores' in self.context_cache[sym] and len(df) == self.context_cache[sym]['data_len']:
                    vra_scores = self.context_cache[sym]['vra_scores']
                else:
                    daily_rv = df['Close'].resample('D').last().pct_change(fill_method=None).rolling(21).std() * np.sqrt(252)
                    vra_scores = get_volatility_regime_anomaly(daily_rv.dropna())
                    vra_scores = pd.Series(vra_scores, index=daily_rv.dropna().index)
                    self.context_cache[sym] = {'vra_scores': vra_scores, 'data_len': len(df)}
                
                # Add pre-merge validation
                nan_pct_returns = df['Close'].resample('D').last().pct_change(fill_method=None).isna().mean() * 100
                if nan_pct_returns > 20:  # Configurable threshold
                    logging.warning(f"High NaNs ({nan_pct_returns:.1f}%) in {sym} daily returns - using last valid for contexts")
                
                # Merge contextual features
                ctx = df.sort_index()
                ctx = pd.merge_asof(ctx, spy_df[['market_regime']].dropna(), left_index=True, right_index=True, direction='backward')
                ctx['market_regime'] = ctx['market_regime'].interpolate(method='linear').ffill().bfill().fillna(0.0)  # Numeric
                ctx = pd.merge_asof(ctx, sar_scores.to_frame('sar_score'), left_index=True, right_index=True, direction='backward')
                ctx = pd.merge_asof(ctx, vra_scores.to_frame('vra_score'), left_index=True, right_index=True, direction='backward')
                ctx[['sar_score', 'vra_score']] = ctx[['sar_score', 'vra_score']].interpolate(method='linear').ffill().bfill().fillna(0)
                
                # Add post-merge validation
                if (ctx['market_regime'] == 0.0).all() or (ctx['sar_score'] == 0.0).all() or (ctx['vra_score'] == 0.0).all():
                    logging.warning(f"All contexts 0.0 for {sym} - possible stale data")
                
                # Compute features
                daily_close = ctx['Close'].resample('D').last()
                dlogret = np.log(1 + daily_close.pct_change(fill_method=None)).dropna()
                features_df = calculate_features(ctx, dlogret, FEATURE_CONFIG)
                print(f"Debug: Features for {sym}: {features_df.columns.tolist()}")
                
                # Ensure features
                missing_feats = [feat for feat in features_to_use if feat not in features_df.columns]
                if missing_feats:
                    print(f"Warning: Missing features for {sym}: {missing_feats}. Filling with defaults.")
                for feat in features_to_use:
                    if feat not in features_df.columns:
                        if feat == 'rsi_14' and 'Close' in ctx.columns:
                            features_df['rsi_14'] = ta.rsi(ctx['Close'], length=14)
                        elif feat.startswith('macd') and 'Close' in ctx.columns:
                            macd = ta.macd(ctx['Close'])
                            if macd is not None and not macd.empty:
                                features_df['macd_line'] = macd['MACD_12_26_9']
                                features_df['macd_signal'] = macd['MACDs_12_26_9']
                                features_df['macd_hist'] = macd['MACDh_12_26_9']
                                features_df['macd_hist_diff'] = macd['MACDh_12_26_9'].diff()
                            else:
                                features_df['macd_line'] = pd.Series(np.nan, index=features_df.index)
                                features_df['macd_signal'] = pd.Series(np.nan, index=features_df.index)
                                features_df['macd_hist'] = pd.Series(np.nan, index=features_df.index)
                                features_df['macd_hist_diff'] = pd.Series(np.nan, index=features_df.index)
                        elif feat == 'sar_score':
                            features_df['sar_score'] = ctx['sar_score']
                        elif feat == 'vra_score':
                            features_df['vra_score'] = ctx['vra_score']
                        else:
                            features_df[feat] = pd.Series(0.0, index=features_df.index)
                
                if len(missing_feats) > len(features_to_use) // 2:
                    logging.warning(f"Too many missing features for {sym}. Skipping.")
                    return None
                
                # Predict
                X = features_df[features_to_use].iloc[-1:].fillna(0)
                if 'market_regime' in features_to_use:
                    regime_value = features_df['market_regime'].iloc[-1]
                    print(f"Debug: {sym} market_regime type/value: {type(regime_value)} / {regime_value}")
                if not X.empty and not X.isnull().all().all():
                    logging.debug(f"Predicting for {sym}")
                    prob = primary_model.predict_proba(X)[0, 1]
                    
                    # Multi-Factor (same as before)
                    regime_value = features_df['market_regime'].iloc[-1] if 'market_regime' in features_df.columns else 0.0
                    if regime_value > 0:
                        regime_score = 1.0
                    elif regime_value < 0:
                        regime_score = 0.0
                    else:
                        regime_score = 0.5
                    volume_score = min(features_df['relative_volume'].iloc[-1] / 2.0, 1.0) if 'relative_volume' in features_df.columns else 0.5
                    stability_score = np.mean(primary_model.predict_proba(features_df[features_to_use].tail(10).fillna(0))[:, 1]) if len(features_df) >= 10 else 0.5
                    
                    multi_conf = 0.70 * prob + 0.10 * regime_score + 0.10 * volume_score + 0.10 * stability_score
                    
                    conf = multi_conf
                    if has_meta:
                        meta_X = pd.DataFrame({
                            'primary_model_probability': [prob],
                            'relative_volume': features_df['relative_volume'].iloc[-1] if 'relative_volume' in features_df.columns else 1.0,
                            'market_regime': regime_value,
                            'model_uncertainty': 1 - 2 * np.abs(prob - 0.5),
                            'sar_score': features_df['sar_score'].iloc[-1] if 'sar_score' in features_df.columns else 0.5,
                            'vra_score': features_df['vra_score'].iloc[-1] if 'vra_score' in features_df.columns else 0.5
                        })[meta_features].fillna(0)
                        meta_prob = meta_model.predict_proba(meta_X)[0, 1]
                        print(f"{sym}: Using meta-model. Meta prob = {meta_prob:.4f}")
                        conf = 0.50 * multi_conf + 0.50 * meta_prob
                    
                    print(f"{sym}: Primary prob = {prob:.4f}, Regime score = {regime_score:.4f} ({regime_value}), Volume score = {volume_score:.4f}, Stability score = {stability_score:.4f}, Multi conf = {multi_conf:.4f}, Final conf = {conf:.4f}")
                    print(f"{sym}: Latest features: {X.to_dict(orient='records')[0]}")
                    
                    threshold = 0.0 if self.debug_mode.get() else CONVICTION_THRESHOLD
                    if conf >= threshold or not self.is_market_open():
                        entry = {
                            "ticker": sym,
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "confidence": conf,
                            "price": df['Close'].iloc[-1] if not df.empty else np.nan,
                            "rel_vol": features_df['relative_volume'].iloc[-1] if 'relative_volume' in features_df.columns else np.nan,
                            "rsi": features_df['rsi_14'].iloc[-1] if 'rsi_14' in features_df.columns else np.nan,
                            "macd": features_df['macd_line'].iloc[-1] if 'macd_line' in features_df.columns else np.nan,
                            "sar": features_df['sar_score'].iloc[-1] if 'sar_score' in features_df.columns else np.nan,
                            "market_regime": regime_value
                        }
                        if conf >= 0.60:
                            entry["features"] = X.to_dict(orient='records')[0]  # Store features for popup
                        print(f"{sym}: Added to table (conf >= {threshold:.2f})")
                        return entry
                else:
                    logging.error(f"Prediction failed for {sym} due to empty or all-NaN X")
                    print(f"{sym}: Prediction failed - empty features")
                    return None
            except Exception as e:
                logging.error(f"Error processing {sym}: {e}")
                print(f"Error processing {sym}: {e}")
                return None
        
        if self.parallel_mode.get():
            results = Parallel(n_jobs=-1)(delayed(process_symbol)(sym, market_data, spy_df, sar_scores, primary_model, features_to_use, has_meta, meta_model, meta_features, MIN_DATA_POINTS, FEATURE_CONFIG) for sym in self.symbols)
            self.signal_data = [r for r in results if r is not None]
        else:
            for sym in self.symbols:
                entry = process_symbol(sym, market_data, spy_df, sar_scores, primary_model, features_to_use, has_meta, meta_model, meta_features, MIN_DATA_POINTS, FEATURE_CONFIG)
                if entry:
                    self.signal_data.append(entry)
        
        self.root.after(0, self.update_table)
        if not self.is_market_open() and not refresh_only:
            print("Off-hours mode: Data fetched. Use 'Refresh Data' to update.")
    
    def is_market_open(self):
        now_et = datetime.now(ET_TZ)
        return now_et.weekday() < 5 and MARKET_OPEN_HOUR <= now_et.hour < MARKET_CLOSE_HOUR
    
    def update_table(self):
        self.tree.delete(*self.tree.get_children())
        for data in sorted(self.signal_data, key=lambda x: x['confidence'] if x['confidence'] is not None else 0, reverse=True):
            conf_pct = data['confidence'] * 100 if data['confidence'] is not None else 0
            tags = ()
            if conf_pct >= 60:
                tags = ('red',)
            elif conf_pct >= 50:
                tags = ('orange',)
            elif conf_pct >= 40:
                tags = ('yellow',)
            self.tree.insert("", "end", values=(
                data['ticker'], data['timestamp'], f"{conf_pct:.2f}%" if data['confidence'] is not None else "N/A",
                f"{data['price']:.2f}" if pd.notna(data['price']) else "N/A",
                f"{data['rel_vol']:.2f}" if pd.notna(data['rel_vol']) else "N/A",
                f"{data['rsi']:.2f}" if pd.notna(data['rsi']) else "N/A",
                f"{data['macd']:.2f}" if pd.notna(data['macd']) else "N/A",
                f"{data['sar']:.2f}" if pd.notna(data['sar']) else "N/A"
            ), tags=tags)
    
    def on_select(self, event):
        selected = self.tree.selection()
        if selected:
            item = self.tree.item(selected[0])
            ticker = item['values'][0]
            conf_pct = float(item['values'][2].rstrip('%')) if item['values'][2] != "N/A" else 0
            for data in self.signal_data:
                if data['ticker'] == ticker:
                    self.prob_label.config(text=f"Probability: {data['confidence']:.2%}" if data['confidence'] is not None else "N/A")
                    self.stab_label.config(text=f"Stability: {np.random.uniform(0.5, 1.0):.2%}")  # Placeholder
                    self.regime_label.config(text=f"Market Regime: {data['market_regime']}")
                    self.sar_label.config(text=f"SAR: {data['sar']:.2f}" if pd.notna(data['sar']) else "N/A")
                    self.insights_text.delete(1.0, tk.END)
                    insight = f"{ticker}: {conf_pct:.1f}% confidence – strong prob." if conf_pct > 50 else f"{ticker}: {conf_pct:.1f}% confidence – low signal; neutral market."
                    self.insights_text.insert(tk.END, insight)
                    break
    
    def on_double_click(self, event):
        selected = self.tree.selection()
        if selected:
            item = self.tree.item(selected[0])
            conf_pct = float(item['values'][2].rstrip('%')) if item['values'][2] != "N/A" else 0
            if conf_pct >= 60:
                ticker = item['values'][0]
                for data in self.signal_data:
                    if data['ticker'] == ticker and 'features' in data:
                        features_str = "\n".join(f"{k}: {v}" for k, v in data['features'].items())
                        popup = tk.Toplevel(self.root)
                        popup.title(f"Features for {ticker}")
                        popup.geometry("400x300")
                        text = tk.Text(popup, wrap=tk.WORD)
                        text.pack(fill=tk.BOTH, expand=True)
                        text.insert(tk.END, features_str)
                        # Make selectable/copyable but non-editable
                        text.config(state=tk.NORMAL)
                        text.bind("<Key>", lambda e: 'break')  # Block typing
                        text.bind("<Button-3>", lambda e: 'break')  # Optional: Block right-click menu if needed
                        popup.mainloop()
                        break
    
    def train_model(self):
        self.parse_tickers()
        if not self.symbols:
            messagebox.showerror("Error", "Enter tickers first!")
            return
        
        # Pop-up status window
        status_win = tk.Toplevel(self.root)
        status_win.title("Training Progress")
        status_win.geometry("400x300")
        status_win.configure(bg='#D3D3D3')
        
        progress = ttk.Progressbar(status_win, mode='determinate', maximum=100)
        progress.pack(fill=tk.X, pady=10)
        
        status_label = tk.Label(status_win, text="Starting training...", bg='#D3D3D3')
        status_label.pack()
        
        log_text = scrolledtext.ScrolledText(status_win, height=10, bg='#E0E0E0')
        log_text.pack(fill=tk.BOTH, expand=True)
        
        def train_thread():
            try:
                cmd = ['python', 'scripts/run_training.py', '--symbols', ','.join(self.symbols), '--device', 'cpu', '--cv', '5']
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for i in range(101):  # Simulate progress
                    time.sleep(0.05)
                    progress['value'] = i
                    status_win.update()
                    for line in process.stdout:
                        log_text.insert(tk.END, line)
                        log_text.see(tk.END)
                    for line in process.stderr:
                        log_text.insert(tk.END, f"ERROR: {line}")
                        log_text.see(tk.END)
                stdout, stderr = process.communicate()
                log_text.insert(tk.END, stdout)
                if stderr:
                    log_text.insert(tk.END, f"ERROR: {stderr}")
                status_label.config(text="Training Complete!")
            except Exception as e:
                log_text.insert(tk.END, f"Error: {e}\n")
        
        threading.Thread(target=train_thread, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = DropzillaGUI(root)
    root.mainloop()
