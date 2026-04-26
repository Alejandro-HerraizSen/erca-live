"""
Market data layer — yfinance wrapper with Streamlit caching.
All functions are cache-safe and return plain Python / pandas objects.
"""

from __future__ import annotations
import time
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional
import streamlit as st


# ── Shared requests session with browser-like headers ─────────────────────────
# Yahoo Finance aggressively rate-limits bare Python user-agents on cloud IPs.
# A realistic UA + cookie consent header prevents most rejections.

def _yf_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    })
    return s


def _ticker(sym: str) -> yf.Ticker:
    return yf.Ticker(sym, session=_yf_session())


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_get(d: dict, *keys, default=None):
    for k in keys:
        v = d.get(k)
        if v is not None and v != 0:
            return v
    return default


# ── Stock info ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=180, show_spinner=False)
def get_stock_info(ticker: str) -> dict:
    """Current snapshot: price, change, volume, fundamentals."""
    try:
        t = _ticker(ticker)

        # fast_info is the reliable price source in recent yfinance versions
        fast = t.fast_info
        price = float(fast.last_price or 0)
        prev  = float(fast.previous_close or price)

        # Supplement with .info for fundamentals (may be empty / rate-limited)
        try:
            info = t.info or {}
        except Exception:
            info = {}

        # Ultimate price fallback: last close from 5-day history
        if price == 0:
            hist = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
            if not hist.empty:
                close = hist["Close"].squeeze()
                price = float(close.iloc[-1])
                prev  = float(close.iloc[-2]) if len(close) >= 2 else price

        change_pct = ((price - prev) / prev * 100) if prev else 0.0

        # Volume: prefer fast_info (always populated), fall back to info
        volume    = int(fast.last_volume or 0) or int(info.get("regularMarketVolume") or 0)
        avg_vol   = int(getattr(fast, "three_month_average_volume", None) or
                        info.get("averageVolume") or 0)
        mkt_cap   = float(getattr(fast, "market_cap", None) or info.get("marketCap") or 0)
        high_52w  = float(getattr(fast, "year_high", None) or info.get("fiftyTwoWeekHigh") or 0)
        low_52w   = float(getattr(fast, "year_low",  None) or info.get("fiftyTwoWeekLow")  or 0)

        return {
            "ticker":      ticker,
            "name":        info.get("longName") or info.get("shortName") or ticker,
            "sector":      info.get("sector", "—"),
            "price":       price,
            "prev_close":  prev,
            "change_pct":  change_pct,
            "volume":      volume,
            "avg_volume":  avg_vol,
            "market_cap":  mkt_cap,
            "pe_ratio":    info.get("trailingPE"),
            "beta":        info.get("beta"),
            "52w_high":    high_52w,
            "52w_low":     low_52w,
            "short_ratio": info.get("shortRatio"),
            "short_float": info.get("shortPercentOfFloat"),
            "description": info.get("longBusinessSummary", ""),
        }
    except Exception:
        return {"ticker": ticker, "name": ticker, "price": 0, "change_pct": 0}


# ── Price history ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def get_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """OHLCV history."""
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


# ── Earnings ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_earnings_info(ticker: str) -> dict:
    """Next/last earnings date, EPS surprise history."""
    try:
        t = _ticker(ticker)
        cal = t.calendar
        next_earnings: Optional[date] = None
        if cal is not None:
            dates = cal.get("Earnings Date", [])
            if hasattr(dates, "__iter__"):
                future = [d for d in dates if hasattr(d, "date") and d.date() >= date.today()]
                if future:
                    next_earnings = future[0].date()
        hist = t.earnings_history
        return {
            "next_earnings": next_earnings,
            "history": hist if hist is not None else pd.DataFrame(),
        }
    except Exception:
        return {"next_earnings": None, "history": pd.DataFrame()}


# ── Options ────────────────────────────────────────────────────────────────────

def _fetch_options_expiries(ticker: str, retries: int = 3) -> tuple[yf.Ticker, list[str]]:
    """Return (Ticker, expiry_list) with retry on empty result."""
    for attempt in range(retries):
        try:
            t = _ticker(ticker)
            exps = list(t.options or [])
            if exps:
                return t, exps
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    return _ticker(ticker), []


@st.cache_data(ttl=300, show_spinner=False)
def get_options_chain(ticker: str, expiry: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Returns (calls_df, puts_df, all_expiries).
    Columns include: strike, lastPrice, bid, ask, iv,
                     volume, openInterest, inTheMoney
    """
    try:
        t, exps = _fetch_options_expiries(ticker)
        if not exps:
            return pd.DataFrame(), pd.DataFrame(), []
        if expiry is None or expiry not in exps:
            expiry = exps[0]
        chain = t.option_chain(expiry)
        calls, puts = chain.calls.copy(), chain.puts.copy()
        for df in (calls, puts):
            df.rename(columns={"impliedVolatility": "iv"}, inplace=True)
        return calls, puts, exps
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), []


@st.cache_data(ttl=600, show_spinner=False)
def get_all_options(ticker: str) -> pd.DataFrame:
    """
    Fetch options across up to 8 expiries — used for IV surface.
    Returns DataFrame with columns: expiry, dte, strike, type, iv, volume, oi
    """
    try:
        t, exps = _fetch_options_expiries(ticker)
        exps = exps[:8]
        rows = []
        today = date.today()
        for exp in exps:
            try:
                chain = t.option_chain(exp)
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                dte = max((exp_date - today).days, 1)
                for _, row in chain.calls.iterrows():
                    rows.append({
                        "expiry": exp, "dte": dte,
                        "strike": row["strike"], "type": "call",
                        "iv": row.get("impliedVolatility", np.nan),
                        "volume": row.get("volume", 0) or 0,
                        "oi": row.get("openInterest", 0) or 0,
                    })
                for _, row in chain.puts.iterrows():
                    rows.append({
                        "expiry": exp, "dte": dte,
                        "strike": row["strike"], "type": "put",
                        "iv": row.get("impliedVolatility", np.nan),
                        "volume": row.get("volume", 0) or 0,
                        "oi": row.get("openInterest", 0) or 0,
                    })
                time.sleep(0.3)   # be polite between expiry fetches
            except Exception:
                continue
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ── News ───────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def get_news(ticker: str, limit: int = 20) -> list[dict]:
    """Yahoo Finance news items for ticker."""
    try:
        t = _ticker(ticker)
        items = t.news or []
        out = []
        for item in items[:limit]:
            content = item.get("content", {})
            title = content.get("title") or item.get("title", "")
            summary = content.get("summary") or ""
            pub = content.get("pubDate") or item.get("providerPublishTime", 0)
            if isinstance(pub, int):
                pub = datetime.fromtimestamp(pub).strftime("%Y-%m-%d %H:%M")
            url = ""
            clinks = content.get("canonicalUrl", {})
            if isinstance(clinks, dict):
                url = clinks.get("url", "")
            if not url:
                url = item.get("link", "")
            out.append({
                "title": title,
                "summary": summary,
                "date": str(pub)[:16],
                "url": url,
                "text": f"{title}. {summary}",
            })
        return out
    except Exception:
        return []
