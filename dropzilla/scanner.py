# Filename: dropzilla/scanner.py
"""
Intraday universe scanner tuned for the Polygon *Stocks Starter* plan.

What’s new vs. your previous version
------------------------------------
• Sane defaults (MIN_REL_VOL ~ 1.5, not 15)
• Stage-by-stage gate logging via GateCounts
• Optional minute-window confirmation (can be disabled)
• Liquidity guard on baseline minute volume
• Works even if dropzilla.config is missing (falls back to sensible defaults)
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional

import pytz
import requests

# ────────────────── config fallbacks (so this file is paste‑and‑run) ──────────
try:
    from dropzilla.config import POLYGON_API_KEY, SCANNER_CONFIG, DATA_VALIDATION_CFG
except Exception:  # noqa: BLE001
    POLYGON_API_KEY = ""  # must be provided by caller
    SCANNER_CONFIG = {
        "min_price": 1.0,
        "max_price": 500.0,
        "rel_vol_threshold": 1.5,
        "confirm_window_minutes": 10,   # 0 to disable 3rd stage
        "min_bars": 200,
        "log_level": "INFO",
    }
    DATA_VALIDATION_CFG = {
        "liquidity_min_vol": 250.0,     # average minute vol baseline
    }

from dropzilla.data import PolygonDataClient

# Logging
LOG = logging.getLogger(__name__)
_level = getattr(logging, str(SCANNER_CONFIG.get("log_level", "INFO")).upper(), logging.INFO)
logging.basicConfig(
    level=_level,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

NY_TZ = pytz.timezone("US/Eastern")
NY_OPEN = NY_TZ.localize(datetime.combine(date.today(), time(9, 30)))
NY_CLOSE = NY_TZ.localize(datetime.combine(date.today(), time(16, 0)))

# ────────────────── starter‑plan timing windows ──────────────────
_INTRADAY_DELAY = timedelta(minutes=16)  # 15‑min embargo + safety

# ────────────────── stage accounting ─────────────────────────────
@dataclass
class GateCounts:
    total: int = 0
    priced: int = 0
    price_band: int = 0
    liquidity_ok: int = 0
    rel_vol: int = 0
    minute_confirmed: int = 0
    qualified: int = 0  # alias of minute_confirmed, kept for readability


def log_gate_counts(tag: str, c: GateCounts) -> None:
    LOG.info(
        "[%s] total=%d priced=%d price_band=%d liquidity_ok=%d rel_vol=%d minute_confirmed=%d qualified=%d",
        tag,
        c.total,
        c.priced,
        c.price_band,
        c.liquidity_ok,
        c.rel_vol,
        c.minute_confirmed,
        c.qualified,
    )


# ────────────────── thin REST helper ─────────────────────────────
def _get_json(path_or_url: str, *, retries: int = 3, pause: float = 1.0) -> dict:
    if not POLYGON_API_KEY:
        raise RuntimeError("POLYGON_API_KEY is not configured.")

    url = (
        f"https://api.polygon.io{path_or_url}"
        if path_or_url.startswith("/")
        else path_or_url
    )
    if "apiKey=" not in url:
        url += ("&" if "?" in url else "?") + f"apiKey={POLYGON_API_KEY}"

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                raise RuntimeError(f"Polygon request failed: {url}") from exc
            LOG.warning("Polygon request failed (%s) retry %d/%d", exc, attempt, retries)
            _time.sleep(pause * attempt)


# ────────────────── scanner class ────────────────────────────────
class PolygonScanner:
    """Intraday universe scanner tuned for the Polygon *Stocks Starter* plan."""

    def __init__(self, api_key: Optional[str] = None, cfg: dict | None = None) -> None:
        if api_key is None:
            if not POLYGON_API_KEY:
                raise ValueError("Polygon API key not provided and not found in config.")
            api_key = POLYGON_API_KEY

        self.cfg = {
            "min_price": 1.0,
            "max_price": 500.0,
            "rel_vol_threshold": 1.5,
            "confirm_window_minutes": 10,
            "min_bars": 200,
            "log_level": "INFO",
            **(cfg or SCANNER_CONFIG),
        }
        self.liq_cfg = {
            "liquidity_min_vol": 250.0,
            **(DATA_VALIDATION_CFG or {}),
        }

        LOG.info("Booting PolygonScanner …")
        self.client = PolygonDataClient(api_key)

        # Universe & baselines
        self.all_tickers: List[str] = self._fetch_active_tickers()
        self.avg_min_vol: Dict[str, float] = self._precompute_baselines(days=50)

        LOG.info(
            "Scanner ready (%d symbols, baselines for %d)",
            len(self.all_tickers),
            len(self.avg_min_vol),
        )

    # ────────────────── public API ───────────────────────────────
    def scan(self) -> List[str]:
        """
        Returns a list of qualified tickers for the current pass.
        """
        now_et = datetime.now(NY_TZ)
        if now_et.weekday() >= 5 or not (time(9, 30) <= now_et.time() < time(16, 0)):
            LOG.debug("Market closed – scanner idle.")
            return []

        counts = GateCounts()

        # ------------------------------------------------------------------
        # 1️⃣ grouped snapshot (ONE call) – *Starter plan friendly*
        # ------------------------------------------------------------------
        snap = _get_json(f"/v2/aggs/grouped/locale/us/market/stocks/{now_et:%Y-%m-%d}")
        rows = {r["T"]: r for r in snap.get("results", [])}
        counts.total = len(rows)

        # 2️⃣ price screen
        min_px = float(self.cfg["min_price"])
        max_px = float(self.cfg.get("max_price", float("inf")))
        priced = {
            t: r
            for t, r in rows.items()
            if r.get("c", 0) > 0  # has price
        }
        counts.priced = len(priced)

        price_band = {
            t: r for t, r in priced.items()
            if min_px <= r.get("c", 0.0) <= max_px
        }
        counts.price_band = len(price_band)
        if not price_band:
            log_gate_counts("scan", counts)
            return []

        # ------------------------------------------------------------------
        # 3️⃣ relative‑volume‑so‑far screen (no extra API calls)
        # ------------------------------------------------------------------
        elapsed_min = max(
            1,
            int(((now_et - _INTRADAY_DELAY) - NY_OPEN).total_seconds() / 60),
        )
        rel_thr = float(self.cfg["rel_vol_threshold"])
        min_vol_baseline = float(self.liq_cfg["liquidity_min_vol"])

        short_list: List[str] = []
        for sym, row in price_band.items():
            base = self.avg_min_vol.get(sym)
            if not base:
                base = self._compute_baseline_single(sym)
                if not base:
                    continue
                self.avg_min_vol[sym] = base

            # liquidity filter: skip low-baseline symbols
            if base < min_vol_baseline:
                continue

            rv = row.get("v", 0) / (base * elapsed_min)
            if rv >= rel_thr:
                short_list.append(sym)
        counts.liquidity_ok = len(short_list)
        counts.rel_vol = len(short_list)

        if not short_list:
            log_gate_counts("scan", counts)
            return []

        # ------------------------------------------------------------------
        # 4️⃣ OPTIONAL fine‑grained minute check (few dozen API calls)
        # ------------------------------------------------------------------
        qualifiers: List[str] = []
        confirm_window = int(self.cfg.get("confirm_window_minutes", 10))
        if confirm_window > 0:
            end = now_et - _INTRADAY_DELAY
            start = end - timedelta(minutes=confirm_window)
            for sym in short_list:
                try:
                    bars = self.client.get_aggs(
                        sym,
                        start.isoformat(),
                        end.isoformat(),
                        timespan="minute",
                        adjusted=True,
                    )
                    if bars is None or bars.empty:
                        continue
                    vol_window = bars["Volume"].sum()
                    base = self.avg_min_vol.get(sym, 0.0)
                    if base <= 0:
                        continue
                    if vol_window / (base * confirm_window) >= rel_thr:
                        qualifiers.append(sym)
                except Exception as exc:  # noqa: BLE001
                    LOG.debug("symbol %s skipped (%s)", sym, exc)
            counts.minute_confirmed = len(qualifiers)
            counts.qualified = len(qualifiers)
        else:
            qualifiers = short_list
            counts.minute_confirmed = len(qualifiers)
            counts.qualified = len(qualifiers)

        log_gate_counts("scan", counts)
        return qualifiers

    # ────────────────── helpers ──────────────────────────────────
    def _fetch_active_tickers(self) -> List[str]:
        tickers: List[str] = []
        url = "/v3/reference/tickers?active=true&market=stocks&limit=1000"
        while url:
            data = _get_json(url)
            tickers.extend(r["ticker"] for r in data.get("results", []))
            url = data.get("next_url")
            _time.sleep(0.25)  # be polite
        return tickers

    def _precompute_baselines(self, *, days: int = 50) -> Dict[str, float]:
        """
        Compute an *average minute volume baseline*:
        baseline = average(daily volume over N sessions) / 390
        """
        baselines, days_done = defaultdict(float), defaultdict(int)
        sess = date.today()
        remaining = days

        while remaining:
            # back up until a weekday
            while sess.weekday() >= 5:
                sess -= timedelta(days=1)

            g = _get_json(f"/v2/aggs/grouped/locale/us/market/stocks/{sess}")
            for r in g.get("results", []):
                baselines[r["T"]] += r.get("v", 0.0)
                days_done[r["T"]] += 1

            remaining -= 1
            sess -= timedelta(days=1)
            _time.sleep(0.25)  # polite

        out = {
            t: (vol / days_done[t]) / 390.0
            for t, vol in baselines.items()
            if days_done[t]
        }
        return out

    def _compute_baseline_single(self, sym: str) -> float | None:
        """
        Fallback baseline computation (for a single symbol) when not covered
        by the precomputed 50-day pack. Uses last 50 daily bars.
        """
        try:
            end = datetime.now(NY_TZ) - _INTRADAY_DELAY
            start = end - timedelta(days=60)
            daily = self.client.get_aggs(
                sym,
                start.isoformat(),
                end.isoformat(),
                timespan="day",
                adjusted=True,
                limit=60,
            )
            if daily is None or daily.empty:
                return None
            return float(daily["Volume"].tail(50).mean() / 390.0)
        except Exception:  # noqa: BLE001
            return None
