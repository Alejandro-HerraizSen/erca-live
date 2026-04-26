"""
SEC EDGAR 8-K filings — free public APIs, no API key required.

Three methods tried in order:
  1. EDGAR ATOM feed using ticker= parameter (most reliable)
  2. EDGAR EFTS full-text search JSON
  3. EDGAR submissions JSON (data.sec.gov) via CIK lookup
"""

from __future__ import annotations
import requests
import streamlit as st
from xml.etree import ElementTree as ET
from typing import Optional
import time

_HEADERS = {
    "User-Agent": "ERCA-Live research@psu.edu",
    "Accept":     "application/json, application/atom+xml, text/xml, */*",
}

# Known CIKs for the 5 tickers (avoids a round-trip lookup)
_CIK_MAP = {
    "AAPL": "0000320193",
    "TSLA": "0001318605",
    "NVDA": "0001045810",
    "AMZN": "0001018724",
    "COIN": "0001679788",
}


def _get(url: str, timeout: int = 10, stream: bool = False) -> Optional[requests.Response]:
    """GET with retries and polite delay."""
    for attempt in range(3):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 429:           # rate-limited
                time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(1)
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_8k_filings(ticker: str, limit: int = 8) -> list[dict]:
    """
    Fetch recent 8-K filings for ticker. Tries three methods.
    Returns list of dicts: {title, date, url, text, source}.
    """
    ticker = ticker.upper()

    # ── Method 1: EDGAR ATOM feed with ticker= (NOT company=) ──────────────
    atom_url = (
        "https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcompany&ticker={ticker}&type=8-K"
        f"&dateb=&owner=include&count={limit}&search_text=&output=atom"
    )
    r = _get(atom_url)
    if r is not None:
        try:
            root = ET.fromstring(r.content)
            ns   = {"atom": "http://www.w3.org/2005/Atom"}
            filings = []
            for e in root.findall("atom:entry", ns)[:limit]:
                title   = e.findtext("atom:title",   default="8-K Filing", namespaces=ns)
                updated = e.findtext("atom:updated", default="",            namespaces=ns)
                link_el = e.find("atom:link", ns)
                href    = link_el.get("href", "") if link_el is not None else ""
                filings.append({
                    "title":  title,
                    "date":   updated[:10] if updated else "",
                    "url":    href,
                    "text":   title,
                    "source": "SEC EDGAR",
                })
            if filings:
                return filings
        except ET.ParseError:
            pass

    # ── Method 2: EDGAR EFTS full-text search JSON ──────────────────────────
    efts_url = (
        "https://efts.sec.gov/LATEST/search-index"
        f"?q=%22{ticker}%22&forms=8-K"
        f"&dateRange=custom&startdt=2024-01-01"
        f"&category=form-type"
    )
    r = _get(efts_url)
    if r is not None:
        try:
            data = r.json()
            hits = data.get("hits", {}).get("hits", [])
            filings = []
            for h in hits[:limit]:
                src = h.get("_source", {})
                names = src.get("display_names", [ticker])
                title = (names[0] if isinstance(names, list) else str(names)) + " — 8-K"
                filings.append({
                    "title":  title,
                    "date":   src.get("file_date", ""),
                    "url":    "https://www.sec.gov" + src.get("file_url", ""),
                    "text":   src.get("period_of_report", title),
                    "source": "SEC EDGAR",
                })
            if filings:
                return filings
        except Exception:
            pass

    # ── Method 3: data.sec.gov submissions JSON via CIK ─────────────────────
    cik = _CIK_MAP.get(ticker)
    if cik:
        sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        r = _get(sub_url, timeout=15)
        if r is not None:
            try:
                data   = r.json()
                recent = data.get("filings", {}).get("recent", {})
                forms  = recent.get("form", [])
                dates  = recent.get("filingDate", [])
                accs   = recent.get("accessionNumber", [])
                cik_num = cik.lstrip("0")
                filings = []
                for form, date, acc in zip(forms, dates, accs):
                    if form.startswith("8-K") and len(filings) < limit:
                        acc_clean = acc.replace("-", "")
                        url = (
                            f"https://www.sec.gov/Archives/edgar/full-index/"
                            f"{date[:4]}/{date[5:7]}/{acc_clean}"
                        )
                        filings.append({
                            "title":  f"{ticker} {form} — {date}",
                            "date":   date,
                            "url":    f"https://www.sec.gov/cgi-bin/browse-edgar"
                                      f"?action=getcompany&CIK={cik}&type=8-K&dateb=&owner=include&count=10",
                            "text":   f"{ticker} {form} filing dated {date}",
                            "source": "SEC EDGAR",
                        })
                if filings:
                    return filings
            except Exception:
                pass

    return []
