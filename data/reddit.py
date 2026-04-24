"""
Reddit & StockTwits scraping — no API key required.
Uses Reddit's public JSON endpoint and StockTwits public API.
"""

from __future__ import annotations
import requests
import streamlit as st
from datetime import datetime
from typing import Optional

_HEADERS = {
    "User-Agent": "ERCA-Live/1.0 (educational research; contact: research@psu.edu)"
}


# ── Reddit ─────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def get_wsb_posts(ticker: str, limit: int = 30) -> list[dict]:
    """
    Scrape r/wallstreetbets posts mentioning the ticker via Reddit's
    public JSON search endpoint — no OAuth required.
    """
    posts = []
    for subreddit in ["wallstreetbets", "options", "stocks"]:
        url = (
            f"https://www.reddit.com/r/{subreddit}/search.json"
            f"?q={ticker}&sort=new&limit={limit}&restrict_sr=1&t=week"
        )
        try:
            r = requests.get(url, headers=_HEADERS, timeout=8)
            if r.status_code != 200:
                continue
            children = r.json()["data"]["children"]
            for p in children:
                d = p["data"]
                posts.append({
                    "title":    d.get("title", ""),
                    "score":    d.get("score", 0),
                    "comments": d.get("num_comments", 0),
                    "created":  datetime.utcfromtimestamp(d.get("created_utc", 0)),
                    "url":      f"https://reddit.com{d.get('permalink', '')}",
                    "text":     d.get("title", "") + " " + d.get("selftext", "")[:300],
                    "source":   f"r/{subreddit}",
                    "upvote_ratio": d.get("upvote_ratio", 0.5),
                })
        except Exception:
            continue
    # Sort by recency
    posts.sort(key=lambda x: x["created"], reverse=True)
    return posts[:limit]


# ── StockTwits ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=180, show_spinner=False)
def get_stocktwits_posts(ticker: str) -> list[dict]:
    """
    Fetch StockTwits stream via the public symbol API.
    Returns up to 30 recent messages with sentiment labels.
    """
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        r = requests.get(url, timeout=8, headers=_HEADERS)
        if r.status_code != 200:
            return []
        messages = r.json().get("messages", [])
        out = []
        for m in messages:
            body = m.get("body", "")
            sentiment_raw = (m.get("entities", {}) or {}).get("sentiment", {}) or {}
            st_sentiment = sentiment_raw.get("basic", "")   # "Bullish" | "Bearish" | ""
            likes = (m.get("likes") or {}).get("total", 0)
            created = m.get("created_at", "")
            out.append({
                "title":    body,
                "text":     body,
                "score":    likes,
                "comments": 0,
                "created":  created,
                "source":   "StockTwits",
                "st_sentiment": st_sentiment,
                "url":      f"https://stocktwits.com/symbol/{ticker}",
            })
        return out
    except Exception:
        return []


# ── Combined ───────────────────────────────────────────────────────────────────

def get_all_social(ticker: str) -> list[dict]:
    """Merge Reddit + StockTwits posts, sorted by recency."""
    reddit  = get_wsb_posts(ticker)
    twits   = get_stocktwits_posts(ticker)
    combined = reddit + twits
    return combined
