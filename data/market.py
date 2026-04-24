"""
Market data layer — yfinance wrapper with Streamlit caching.
All functions are cache-safe and return plain Python / pandas objects.
"""

from __future__ import annotations
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional
import streamlit as st


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
        t = yf.Ticker(ticker)
        info = t.info
        fast = t.fast_info
        price = _safe_get(info, "currentPrice", "regularMarketPrice") or float(fast.last_price or 0)
        prev  = _safe_get(info, "previousClose", "regularMarketPreviousClose") or float(fast.previous_close or price)
        change_pct = ((price - prev) / prev * 100) if prev else 0.0
        return {
            "ticker": ticker,
            "name":        info.get("longName", ticker),
            "sector":      info.get("sector", "—"),
            "price":       price,
            "prev_close":  prev,
            "change_pct":  change_pct,
            "volume":      _safe_get(info, "regularMarketVolume", default=0),
            "avg_volume":  _safe_get(info, "averageVolume", default=0),
            "market_cap":  info.get("marketCap", 0),
            "pe_ratio":    info.get("trailingPE"),
            "beta":        info.get("beta"),
            "52w_high":    info.get("fiftyTwoWeekHigh", 0),
            "52w_low":     info.get("fiftyTwoWeekLow", 0),
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
        t = yf.Ticker(ticker)
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

@st.cache_data(ttl=300, show_spinner=False)
def get_options_chain(ticker: str, expiry: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Returns (calls_df, puts_df, all_expiries).
    Each df has: strike, lastPrice, bid, ask, impliedVolatility,
                 delta (approx), volume, openInterest, inTheMoney
    """
    try:
        t = yf.Ticker(ticker)
        exps = list(t.options)
        if not exps:
            return pd.DataFrame(), pd.DataFrame(), []
        if expiry is None or expiry not in exps:
            expiry = exps[0]
        chain = t.option_chain(expiry)
        calls, puts = chain.calls.copy(), chain.puts.copy()
        # Rename for consistency
        for df in (calls, puts):
            df.rename(columns={"impliedVolatility": "iv"}, inplace=True)
        return calls, puts, exps
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), []


@st.cache_data(ttl=600, show_spinner=False)
def get_all_options(ticker: str) -> pd.DataFrame:
    """
    Fetch options across up to 8 expiries — used for IV surface.
    Returns DataFrame: expiry, days_to_exp, strike, type, iv, volume, oi
    """
    try:
        t = yf.Ticker(ticker)
        exps = list(t.options)[:8]
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
        t = yf.Ticker(ticker)
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
