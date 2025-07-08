"""
Dropzilla v4.1 - Unified Training Pipeline
"""

from __future__ import annotations
import argparse
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, cast

import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib

from dropzilla.config import (
    POLYGON_API_KEY,
    DATA_CONFIG,
    FEATURE_CONFIG,
    LABELING_CONFIG,
    MODEL_CONFIG,
)
from dropzilla.data import PolygonDataClient
from dropzilla.features import calculate_features
from dropzilla.labeling import get_triple_barrier_labels
from dropzilla.validation import PurgedKFold
from dropzilla.models import optimize_hyperparameters, train_lightgbm_model
from dropzilla.context import get_market_regimes
from dropzilla.correlation import get_systemic_absorption_ratio
from dropzilla.volatility import get_volatility_regime_anomaly

DIVERSIFIED_UNIVERSE: list[str] = [ 
    # … your tickers … 
]


def main(tickers: list[str], model_name: str) -> None:
    api_key = cast(str, POLYGON_API_KEY)
    client = PolygonDataClient(api_key=api_key)

    to_date = datetime.now(timezone.utc)
    days = int(cast(int, DATA_CONFIG["data_period_days"]))
    from_date = to_date - timedelta(days=days)

    # Fetch market data
    raw = {
        s: client.get_aggs(s,
          from_date.strftime("%Y-%m-%d"),
          to_date.strftime("%Y-%m-%d"))
        for s in tickers
    }
    market_data = {
        s: df for s, df in raw.items()
        if isinstance(df, pd.DataFrame) and not df.empty
    }

    # SPY regime
    spy_raw = client.get_aggs(
        "SPY",
        (from_date - timedelta(days=50)).strftime("%Y-%m-%d"),
        to_date.strftime("%Y-%m-%d"),
        timespan="day"
    )
    if isinstance(spy_raw, pd.DataFrame) and not spy_raw.empty:
        spy_raw["market_regime"] = get_market_regimes(spy_raw)
        spy_df = spy_raw
    else:
        spy_df = pd.DataFrame(columns=["market_regime"])

    # Feature & label prep
    all_labeled: list[pd.DataFrame] = []
    daily_panel = pd.DataFrame({
        sym: df["Close"].resample("D").last().pct_change(fill_method=None)
        for sym, df in market_data.items()
    }).ffill().dropna(how="all", axis=1)
    sar_scores = get_systemic_absorption_ratio(daily_panel)

    for symbol, df in market_data.items():
        # daily log returns
        daily_close = df["Close"].resample("D").last()
        log_ret = np.log(daily_close[daily_close > 0]).diff().dropna()
        if log_ret.empty:
            continue

        # volatility anomaly
        rv = (daily_close.pct_change(fill_method=None)
              .ffill()
              .rolling(window=21)
              .std()
              * np.sqrt(252))
        vra_scores = get_volatility_regime_anomaly(rv.dropna())

        # merge context
        ctx = pd.merge_asof(
            df.sort_index(),
            spy_df[["market_regime"]],
            left_index=True, right_index=True,
            direction="backward"
        )
        ctx = pd.merge_asof(
            ctx,
            sar_scores.to_frame("sar_score"),
            left_index=True, right_index=True,
            direction="backward"
        )
        ctx = pd.merge_asof(
            ctx,
            vra_scores.to_frame("vra_score"),
            left_index=True, right_index=True,
            direction="backward"
        )
        ctx[["sar_score", "vra_score"]] = ctx[["sar_score", "vra_score"]].ffill().fillna(0)

        # calculate features & labels
        feats = calculate_features(ctx, pd.DataFrame(), log_ret, FEATURE_CONFIG)
        atr = ta.atr(
            high=feats["High"],
            low=feats["Low"],
            close=feats["Close"],
            length=LABELING_CONFIG["atr_period"]
        )
        if isinstance(atr, pd.DataFrame):
            atr = atr.iloc[:, 0]
        target_vol = atr.reindex(feats.index).bfill()
        t_events = feats.index.to_series() + pd.Timedelta(
            minutes=LABELING_CONFIG["vertical_barrier_minutes"]
        )
        labels = get_triple_barrier_labels(
            prices=feats["Close"],
            t_events=feats.index,
            pt_sl=[
                LABELING_CONFIG["profit_take_atr_multiplier"],
                LABELING_CONFIG["stop_loss_atr_multiplier"],
            ],
            target=target_vol,
            vertical_barrier_times=t_events,
            side=pd.Series(-1, index=feats.index),
        )
        feats["drop_label"] = labels["bin"].replace(-1, 0)
        feats["label_time"] = labels["t1"]
        all_labeled.append(feats)

    # define actual feature-column names
    features_to_use = [
        "relative_volume",
        "distance_from_vwap_pct",
        "vwap_slope",
        "roc_30",
        "roc_60",
        "roc_120",
        f"rsi_{FEATURE_CONFIG['rsi_period']}",
        f"rsi_{FEATURE_CONFIG['rsi_period']}_sma_5",
        "macd_line",
        "macd_signal",
        "macd_hist",
        "macd_hist_diff",
        f"mfi_{FEATURE_CONFIG['mfi_period']}",
        "obv_slope",
        "market_regime",
        "volatility_surprise",
        "sar_score",
        "vra_score",
    ]

    # drop rows missing any of these true feature columns or labels
    final_df = pd.concat(all_labeled).dropna(
        subset=features_to_use + ["drop_label", "label_time"]
    )

    # proceed with CV, training, etc. …
    # (unchanged beyond this point)
    # …
