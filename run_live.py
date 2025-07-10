import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import os
import subprocess
import threading
import queue
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Local Application Imports ---
from dropzilla.config import POLYGON_API_KEY, DATA_CONFIG, FEATURE_CONFIG
from dropzilla.data import PolygonDataClient
from dropzilla.features import calculate_features
from dropzilla.context import get_market_regimes
from dropzilla.correlation import get_systemic_absorption_ratio
from dropzilla.volatility import get_volatility_regime_anomaly

# --- NEW: Custom Error Window ---
class ErrorWindow(tk.Toplevel):
    def __init__(self, parent, error_message):
        super().__init__(parent)
        self.title("Training Error")
        self.geometry("800x600")
        self.configure(bg="#2E2E2E")

        label = ttk.Label(self, text="The training process failed with the following error:", font=("Arial", 12, "bold"))
        label.pack(pady=10, padx=10)

        log_text = ScrolledText(self, wrap=tk.WORD, bg="#1E1E1E", fg="#FF5555", font=("Courier New", 10), relief=tk.FLAT)
        log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        log_text.insert(tk.END, error_message)
        log_text.config(state=tk.DISABLED) # Make it read-only

        close_button = ttk.Button(self, text="Close", command=self.destroy)
        close_button.pack(pady=10)

# --- Training Progress Window Class (Unchanged but included for completeness) ---
class TrainingProgressWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Training Progress")
        self.geometry("700x500")
        self.configure(bg="#2E2E2E")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.progress_queue = queue.Queue()
        self.parent = parent

        style = ttk.Style(self)
        style.configure("Progress.TLabel", background="#2E2E2E", foreground="white", font=("Arial", 11))
        style.configure("Overall.Horizontal.TProgressbar", background='#007ACC', troughcolor='#4A4A4A')
        style.configure("Detail.Horizontal.TProgressbar", background='#2ECC71', troughcolor='#4A4A4A')

        main_frame = ttk.Frame(self, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Overall Progress:", style="Progress.TLabel").pack(fill=tk.X, pady=(0,5))
        self.overall_progress = ttk.Progressbar(main_frame, orient="horizontal", length=100, mode="determinate", style="Overall.Horizontal.TProgressbar")
        self.overall_progress.pack(fill=tk.X, pady=(0, 10))
        self.overall_status_label = ttk.Label(main_frame, text="Initializing...", style="Progress.TLabel")
        self.overall_status_label.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(main_frame, text="Detailed Progress:", style="Progress.TLabel").pack(fill=tk.X, pady=(0,5))
        self.detail_progress = ttk.Progressbar(main_frame, orient="horizontal", length=100, mode="determinate", style="Detail.Horizontal.TProgressbar")
        self.detail_progress.pack(fill=tk.X, pady=(0, 10))
        self.detail_status_label = ttk.Label(main_frame, text="", style="Progress.TLabel")
        self.detail_status_label.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(main_frame, text="Training Log:", style="Progress.TLabel").pack(fill=tk.X, pady=(0,5))
        self.log_text = ScrolledText(main_frame, wrap=tk.WORD, bg="#1E1E1E", fg="white", font=("Courier New", 9), relief=tk.FLAT)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.after(100, self.process_queue)

    def process_queue(self):
        try:
            while True:
                message = self.progress_queue.get_nowait()
                self.handle_message(message)
        except queue.Empty:
            self.after(100, self.process_queue)

    def handle_message(self, message):
        if message.startswith("PROGRESS::"):
            parts = message.strip().split("::")
            p_type = parts[1]

            if p_type in ["OVERALL", "DETAIL"] and len(parts) == 4:
                _, _, progress_str, status = parts
                current, total = map(int, progress_str.split('/'))
                progress_value = (current / total) * 100
                if p_type == "OVERALL":
                    self.overall_progress['value'] = progress_value
                    self.overall_status_label.config(text=f"Step {current}/{total}: {status}")
                    self.detail_progress['value'] = 0
                    self.detail_status_label.config(text="")
                else:
                    self.detail_progress['value'] = progress_value
                    self.detail_status_label.config(text=status)
            elif p_type == "STATUS" and len(parts) == 3:
                _, _, status = parts
                self.detail_status_label.config(text=status)
        elif message == "TRAINING_COMPLETE":
            self.overall_progress['value'] = 100
            self.detail_progress['value'] = 100
            self.overall_status_label.config(text="Training Complete!")
            self.detail_status_label.config(text="You can now close this window.")
            messagebox.showinfo("Success", "Training process finished successfully.", parent=self)
            self.parent.on_training_finish(success=True)
        elif message.startswith("TRAINING_ERROR"):
            error_msg = message.split("::", 1)[1]
            ErrorWindow(self.parent, error_msg) # Use the new error window
            self.parent.on_training_finish(success=False)
            self.destroy() # Close the progress window on error
        else:
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to abort the training process?", parent=self):
            self.parent.on_training_finish(success=False)
            self.destroy()

# --- Main Application Class ---
class DropzillaApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dropzilla v4 - Live Signal Engine")
        self.geometry("1200x800")
        self.style = ttk.Style(self)
        self.configure(bg="#2E2E2E")
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#2E2E2E")
        self.style.configure("TLabel", background="#2E2E2E", foreground="white", font=("Arial", 10))
        self.style.configure("TButton", background="#4A4A4A", foreground="white", font=("Arial", 10, "bold"), borderwidth=0)
        self.style.map("TButton", background=[("active", "#6A6A6A")])
        self.style.configure("TEntry", fieldbackground="#4A4A4A", foreground="white", insertcolor="white", borderwidth=0)
        self.style.configure("Treeview", background="#2E2E2E", fieldbackground="#4A4A4A", foreground="white",rowheight=25,font=("Arial", 10))
        self.style.configure("Treeview.Heading", background="#4A4A4A", foreground="white", font=("Arial", 11, "bold"))
        self.style.map("Treeview.Heading", background=[("active", "#6A6A6A")])

        self.MODELS_DIR = "models"
        if not os.path.exists(self.MODELS_DIR):
            os.makedirs(self.MODELS_DIR)

        self.current_model_artifact = None
        self.is_running = False
        self.after_id = None

        self._create_widgets()
        self.update_model_dropdown()

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.pack(fill=tk.X)
        ttk.Label(control_frame, text="Tickers (comma-separated):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.ticker_entry = ttk.Entry(control_frame, width=60)
        self.ticker_entry.insert(0, "AAPL,MSFT,NVDA,TSLA,GOOG,CVI,PGY,PLAY,SATS,SEDG,SYM")
        self.ticker_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(control_frame, text="New Model Name:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.model_name_entry = ttk.Entry(control_frame, width=40)
        self.model_name_entry.insert(0, f"model_{datetime.now().strftime('%Y%m%d')}.pkl")
        self.model_name_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.train_button = ttk.Button(control_frame, text="Train New Model", command=self.start_training_thread)
        self.train_button.grid(row=1, column=2, padx=10, pady=5)
        monitor_frame = ttk.Frame(main_frame, padding="10")
        monitor_frame.pack(fill=tk.X, pady=10)
        ttk.Label(monitor_frame, text="Select Model for Live Monitoring:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(monitor_frame, textvariable=self.model_var, state="readonly", width=38)
        self.model_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.model_dropdown.bind("<<ComboboxSelected>>", self.on_model_select)
        self.start_button = ttk.Button(monitor_frame, text="Start Live Monitoring", command=self.start_monitoring)
        self.start_button.grid(row=0, column=2, padx=5, pady=5)
        self.stop_button = ttk.Button(monitor_frame, text="Stop Monitoring", command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=3, padx=5, pady=5)
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Select a model and start monitoring.")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w", padding=5)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        table_frame = ttk.Frame(main_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        columns = ("ticker", "timestamp", "confidence", "price")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        self.tree.heading("ticker", text="Ticker")
        self.tree.heading("timestamp", text="Timestamp (UTC)")
        self.tree.heading("confidence", text="Confidence")
        self.tree.heading("price", text="Last Price")
        self.tree.column("ticker", width=100, anchor="center")
        self.tree.column("timestamp", width=200, anchor="center")
        self.tree.column("confidence", width=150, anchor="center")
        self.tree.column("price", width=150, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True)

    def start_training_thread(self):
        tickers = self.ticker_entry.get()
        model_name = self.model_name_entry.get()
        if not tickers:
            messagebox.showerror("Error", "Ticker list cannot be empty.")
            return
        if not model_name.endswith(".pkl"):
            messagebox.showerror("Error", "Model name must end with .pkl")
            return
        model_path = os.path.join(self.MODELS_DIR, model_name)

        self.train_button.config(state=tk.DISABLED)

        self.progress_window = TrainingProgressWindow(self)

        thread = threading.Thread(target=self.run_training_process, args=(tickers, model_path, self.progress_window.progress_queue))
        thread.daemon = True
        thread.start()

    def run_training_process(self, tickers, model_path, q):
        try:
            command = ["python", "scripts/run_training.py", "--tickers", tickers, "--model_name", model_path]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)

            for line in process.stdout:
                q.put(line.strip())

            process.wait()

            stderr_output = process.stderr.read()
            if process.returncode != 0:
                q.put(f"TRAINING_ERROR::{stderr_output}")
            else:
                q.put("TRAINING_COMPLETE")
        except Exception as e:
            q.put(f"TRAINING_ERROR::Failed to start training process: {e}")

    def on_training_finish(self, success: bool):
        self.train_button.config(state=tk.NORMAL)
        self.status_var.set("Ready.")
        if success:
            self.update_model_dropdown()

    def update_model_dropdown(self):
        models = sorted([f for f in os.listdir(self.MODELS_DIR) if f.endswith(".pkl")], reverse=True)
        self.model_dropdown['values'] = models
        if models and not self.model_var.get():
            self.model_var.set(models[0])
            self.on_model_select()

    def on_model_select(self, event=None):
        model_file = self.model_var.get()
        if not model_file: return
        model_path = os.path.join(self.MODELS_DIR, model_file)
        try:
            self.current_model_artifact = joblib.load(model_path)
            self.status_var.set(f"Loaded model: {model_file}. Ready to start monitoring.")
        except Exception as e:
            ErrorWindow(self, f"Failed to load model:\n\n{e}")
            self.current_model_artifact = None

    def start_monitoring(self):
        if not self.current_model_artifact:
            self.on_model_select()
            if not self.current_model_artifact:
                messagebox.showerror("Error", "Please select a valid model to use.")
                return
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set(f"Live monitoring started with {self.model_var.get()}. Refreshing every 5 minutes.")
        self.run_prediction_loop()

    def stop_monitoring(self):
        if self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Monitoring stopped. Ready.")

    def run_prediction_loop(self):
        if not self.is_running: return
        thread = threading.Thread(target=self.fetch_and_predict)
        thread.daemon = True
        thread.start()
        self.after_id = self.after(300000, self.run_prediction_loop)

    def fetch_and_predict(self):
        try:
            self.status_var.set(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching latest data...")
            primary_model = self.current_model_artifact['model']
            features_to_use = self.current_model_artifact['features_to_use']

            data_client = PolygonDataClient(api_key=POLYGON_API_KEY)
            symbols = [s.strip().upper() for s in self.ticker_entry.get().split(',')]
            to_date = datetime.now()
            from_date = to_date - timedelta(days=DATA_CONFIG.get('data_period_days', 365) + 50)

            market_data = {s: data_client.get_aggs(s, from_date.strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d')) for s in symbols}
            market_data = {k: v for k, v in market_data.items() if v is not None and not v.empty}

            if not market_data:
                self.status_var.set("Could not fetch data for any tickers.")
                return

            spy_df = data_client.get_aggs("SPY", from_date.strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d'), timespan='day')
            if spy_df is not None and not spy_df.empty:
                spy_df['market_regime'] = get_market_regimes(spy_df)

            self.status_var.set(f"[{datetime.now().strftime('%H:%M:%S')}] Generating features and signals...")

            daily_returns_panel = pd.DataFrame({s: df['Close'].resample('D').last().pct_change() for s, df in market_data.items()}).dropna(how='all')
            sar_scores = get_systemic_absorption_ratio(daily_returns_panel)

            latest_signals = []
            for symbol, df in market_data.items():
                daily_rv = df['Close'].resample('D').last().pct_change().rolling(window=21).std() * np.sqrt(252)
                vra_scores = get_volatility_regime_anomaly(daily_rv.dropna())

                df_context = pd.merge_asof(df.sort_index(), spy_df[['market_regime']].dropna(), left_index=True, right_index=True, direction='backward')
                df_context = pd.merge_asof(df_context, sar_scores.to_frame(name='sar_score'), left_index=True, right_index=True, direction='backward')
                df_context = pd.merge_asof(df_context, vra_scores.to_frame(name='vra_score'), left_index=True, right_index=True, direction='backward')
                df_context[['sar_score', 'vra_score']] = df_context[['sar_score', 'vra_score']].ffill().fillna(0)

                daily_log_returns = np.log(df['Close'][df['Close'] > 0]).resample('D').last().diff().dropna()
                features_df = calculate_features(df_context, daily_log_returns, FEATURE_CONFIG)

                latest_features = features_df.reindex(columns=features_to_use).iloc[-1:]
                if latest_features.isnull().values.any():
                    print(f"Skipping {symbol}, not enough data for all features.")
                    continue

                primary_prob = primary_model.predict_proba(latest_features)[0, 1]

                p_calibrated = primary_prob
                prob_stability = 1 - (pd.Series(primary_model.predict_proba(features_df[features_to_use].tail(5))[:,1]).std() / 0.5).clip(0,1)
                regime_map = {0: 0.8, 1: 0.5, 2: 1.0}
                r_context = regime_map.get(latest_features['market_regime'].iloc[0], 0.5)
                c_confirmation = (latest_features['relative_volume'].iloc[0] / 2.0).clip(0, 1)

                W_prob, W_stab, W_regime, W_conf = 0.40, 0.15, 0.25, 0.20
                confidence = (W_prob * p_calibrated + W_stab * prob_stability + W_regime * r_context + W_conf * c_confirmation)

                latest_signals.append({
                    "ticker": symbol,
                    "timestamp": latest_features.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                    "confidence": f"{confidence:.2%}",
                    "price": f"{df['Close'].iloc[-1]:.2f}"
                })

            self.update_table(latest_signals)
            self.status_var.set(f"[{datetime.now().strftime('%H:%M:%S')}] Update complete. Next refresh in 5 minutes.")

        except Exception as e:
            ErrorWindow(self, f"An error occurred during prediction:\n\n{e}")
            print(f"Prediction error: {e}")

    def update_table(self, signals):
        sorted_signals = sorted(signals, key=lambda x: float(x['confidence'].strip('%')), reverse=True)

        for i in self.tree.get_children():
            self.tree.delete(i)
        for signal in sorted_signals:
            self.tree.insert("", tk.END, values=list(signal.values()))

if __name__ == "__main__":
    app = DropzillaApp()
    app.mainloop()
