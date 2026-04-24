"""
SEC EDGAR 8-K filings — free public RSS/ATOM feed, no API key required.
"""

from __future__ import annotations
import requests
import streamlit as st
from xml.etree import ElementTree as ET
from typing import Optional

_HEADERS = {"User-Agent": "ERCA-Live research@psu.edu"}


@st.cache_data(ttl=3600, show_spinner=False)
def get_8k_filings(ticker: str, limit: int = 8) -> list[dict]:
    """
    Fetch recent 8-K filings for a ticker from SEC EDGAR ATOM feed.
    Falls back to EDGAR full-text search JSON if ATOM fails.
    """
    # ── Method 1: EDGAR company search ATOM feed ────────────────────────────
    atom_url = (
        f"https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcompany&company={ticker}&type=8-K"
        f"&dateb=&owner=include&count={limit}&search_text=&output=atom"
    )
    try:
        r = requests.get(atom_url, headers=_HEADERS, timeout=10)
        if r.status_code == 200:
            root = ET.fromstring(r.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)
            filings = []
            for e in entries[:limit]:
                title   = e.findtext("atom:title",   default="8-K Filing", namespaces=ns)
                updated = e.findtext("atom:updated", default="",            namespaces=ns)
                link_el = e.find("atom:link", ns)
                href    = link_el.get("href", "") if link_el is not None else ""
                filings.append({
                    "title": title,
                    "date":  updated[:10] if updated else "",
                    "url":   href,
                    "text":  title,
                    "source": "SEC EDGAR",
                })
            if filings:
                return filings
    except Exception:
        pass

    # ── Method 2: EDGAR full-text search JSON ───────────────────────────────
    search_url = (
        "https://efts.sec.gov/LATEST/search-index"
        f"?q=%22{ticker}%22&forms=8-K&dateRange=custom"
        f"&startdt=2024-01-01&hits.hits.total.value=true"
    )
    try:
        r = requests.get(search_url, headers=_HEADERS, timeout=10)
        if r.status_code == 200:
            hits = r.json().get("hits", {}).get("hits", [])
            filings = []
            for h in hits[:limit]:
                src = h.get("_source", {})
                filings.append({
                    "title":  src.get("display_names", ticker) + " — 8-K",
                    "date":   src.get("file_date", ""),
                    "url":    f"https://www.sec.gov{src.get('file_url', '')}",
                    "text":   src.get("period_of_report", ""),
                    "source": "SEC EDGAR",
                })
            return filings
    except Exception:
        pass

    return []
