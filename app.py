"""
ERCA Live — Earnings Call Risk & Confidence Analyzer
Streamlit dashboard — 5 tickers, live data, full ERCA pipeline

Run:  streamlit run app.py
"""

from __future__ import annotations

import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy.interpolate import griddata

# ── ERCA core ──────────────────────────────────────────────────────────────────
from erca import HawkesProcess, LatentProfileAnalysis, DivergenceDetector, FractionalKelly

# ── Data layer ─────────────────────────────────────────────────────────────────
from data.market import (
    get_stock_info, get_price_history, get_options_chain,
    get_all_options, get_news, get_earnings_info,
)
from data.sentiment import score_text, score_batch, sentiment_color, sentiment_label
from data.reddit import get_wsb_posts, get_stocktwits_posts
from data.edgar import get_8k_filings

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

TICKERS = ["AAPL", "TSLA", "NVDA", "AMZN", "COIN"]

TICKER_META = {
    "AAPL": {"color": "#A8B5C7"},
    "TSLA": {"color": "#E82127"},
    "NVDA": {"color": "#76B900"},
    "AMZN": {"color": "#FF9900"},
    "COIN": {"color": "#0052FF"},
}

st.set_page_config(
    page_title="ERCA Live",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Header banner ── */
.erca-header {
    background: linear-gradient(135deg, #0E1117 0%, #161C2D 50%, #0E1117 100%);
    border-bottom: 1px solid #1E2740;
    padding: 18px 28px 14px 28px;
    margin: -1rem -1rem 1.5rem -1rem;
}
.erca-title {
    font-size: 1.7rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(90deg, #00D4FF, #7B61FF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.erca-sub { font-size: 0.78rem; color: #5A6478; margin-top: 2px; }

/* ── Metric cards ── */
.metric-card {
    background: #161C2D; border: 1px solid #1E2740;
    border-radius: 10px; padding: 14px 18px;
    text-align: center;
}
.metric-label { font-size: 0.72rem; color: #5A6478; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { font-size: 1.5rem; font-weight: 700; color: #E8EDF5; margin-top: 4px; }
.metric-delta-pos { color: #00C853; font-size: 0.85rem; }
.metric-delta-neg { color: #D50000; font-size: 0.85rem; }

/* ── Signal badge ── */
.signal-fire {
    background: linear-gradient(135deg, #D50000, #FF6D00);
    color: white; padding: 6px 14px; border-radius: 20px;
    font-weight: 700; font-size: 0.9rem; display: inline-block;
    animation: pulse 1.2s infinite;
}
.signal-quiet {
    background: #1E2740; color: #5A6478;
    padding: 6px 14px; border-radius: 20px;
    font-size: 0.9rem; display: inline-block;
}
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.7;} }

/* ── Social post card ── */
.post-card {
    background: #161C2D; border: 1px solid #1E2740;
    border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;
}
.post-title { font-size: 0.88rem; color: #E8EDF5; }
.post-meta  { font-size: 0.72rem; color: #5A6478; margin-top: 6px; }

/* ── Options table ── */
.options-call { background: rgba(0,200,83,0.08); }
.options-put  { background: rgba(213,0,0,0.08); }

/* ── Tab strip ── */
button[data-baseweb="tab"] { font-size: 0.88rem; }

/* ── Countdown ── */
.countdown {
    font-size: 2rem; font-weight: 800; color: #00D4FF;
    font-variant-numeric: tabular-nums; letter-spacing: -1px;
}
.countdown-label { font-size: 0.72rem; color: #5A6478; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER + TICKER SELECTOR
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="erca-header">
  <div class="erca-title">ERCA Live</div>
  <div class="erca-sub">Earnings Call Risk &amp; Confidence Analyzer &nbsp;·&nbsp;
  Hawkes · LPA · Z<sub>short</sub> · Fractional Kelly &nbsp;·&nbsp;
  Penn State 2026 &nbsp;·&nbsp; 5 tickers · live data</div>
</div>
""", unsafe_allow_html=True)

# Ticker selector row
col_tickers, col_refresh, col_ts = st.columns([5, 1, 2])
with col_tickers:
    ticker = st.radio(
        "Select ticker",
        TICKERS,
        horizontal=True,
        label_visibility="collapsed",
        format_func=lambda t: t,
    )
with col_refresh:
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with col_ts:
    st.markdown(
        f"<div style='text-align:right;color:#5A6478;font-size:0.78rem;padding-top:8px;'>"
        f"Last update: {datetime.now(ET).strftime('%H:%M:%S')} ET</div>",
        unsafe_allow_html=True,
    )

color = TICKER_META[ticker]["color"]

# ── Fetch all data ─────────────────────────────────────────────────────────────
with st.spinner(f"Loading {ticker} data…"):
    info       = get_stock_info(ticker)
    price_hist = get_price_history(ticker, period="1y")
    calls, puts, exps = get_options_chain(ticker)
    all_opts   = get_all_options(ticker)
    news_items = get_news(ticker)
    earnings   = get_earnings_info(ticker)
    wsb_posts  = get_wsb_posts(ticker)
    st_posts   = get_stocktwits_posts(ticker)
    filings    = get_8k_filings(ticker)

price   = info.get("price", 0) or 0
chg_pct = info.get("change_pct", 0) or 0

# ── Price summary bar ──────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)

def _metric(col, label, value, delta=None, fmt=None):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          {"" if delta is None else
           f'<div class="metric-delta-{"pos" if delta>=0 else "neg"}">{"▲" if delta>=0 else "▼"} {abs(delta):.2f}%</div>'}
        </div>""", unsafe_allow_html=True)

_metric(c1, "Price",       f"${price:,.2f}", delta=chg_pct)
_metric(c2, "Market Cap",  f"${info.get('market_cap',0)/1e9:.1f}B" if info.get('market_cap') else "—")
_metric(c3, "Volume",      f"{info.get('volume',0)/1e6:.1f}M" if info.get('volume') else "—")
_metric(c4, "52W High",    f"${info.get('52w_high',0):,.2f}")
_metric(c5, "52W Low",     f"${info.get('52w_low',0):,.2f}")

next_e = earnings.get("next_earnings")
if next_e:
    days_to = (next_e - date.today()).days
    _metric(c6, "Next Earnings", f"{days_to}d" if days_to >= 0 else "Passed",
            delta=None)
else:
    _metric(c6, "Next Earnings", "—")

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dashboard",
    "Options Chain",
    "IV Surface",
    "Social Sentiment",
    "News & Filings",
    "ERCA Signal",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    left, right = st.columns([3, 1])

    # ── Price chart ────────────────────────────────────────────────────────────
    with left:
        st.markdown(f"#### {info.get('name', ticker)} — 1 Year Price")
        if not price_hist.empty:
            close = price_hist["Close"].squeeze()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=close.index, y=close.values,
                mode="lines", name="Close",
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
            ))
            # Add volume on secondary y
            vol = price_hist["Volume"].squeeze()
            fig.add_trace(go.Bar(
                x=vol.index, y=vol.values,
                name="Volume", yaxis="y2",
                marker_color="#1E2740", opacity=0.5,
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117", height=340,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
                yaxis=dict(title="Price ($)", gridcolor="#1E2740"),
                yaxis2=dict(overlaying="y", side="right", showgrid=False,
                            title="Volume", tickformat=".2s"),
                xaxis=dict(gridcolor="#1E2740"),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Price history unavailable.")

    # ── Right panel ────────────────────────────────────────────────────────────
    with right:
        # Earnings countdown
        st.markdown("#### Earnings Countdown")
        if next_e:
            days_to = (next_e - date.today()).days
            if days_to >= 0:
                st.markdown(f"""
                <div style='text-align:center;background:#161C2D;border:1px solid #1E2740;
                            border-radius:10px;padding:20px;'>
                  <div class='countdown'>{days_to}</div>
                  <div class='countdown-label'>DAYS TO EARNINGS</div>
                  <div style='color:#5A6478;font-size:0.78rem;margin-top:8px;'>{next_e}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:#5A6478;'>Earnings passed.</div>",
                            unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#5A6478;'>Date unavailable.</div>",
                        unsafe_allow_html=True)

        # P/C Ratio
        st.markdown("#### Put/Call Ratio")
        if not calls.empty and not puts.empty:
            c_vol = calls["volume"].fillna(0).sum()
            p_vol = puts["volume"].fillna(0).sum()
            pc_ratio = p_vol / max(c_vol, 1)
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(pc_ratio, 2),
                title={"text": "P/C (volume)", "font": {"size": 12, "color": "#5A6478"}},
                gauge={
                    "axis":     {"range": [0, 3], "tickcolor": "#5A6478"},
                    "bar":      {"color": "#00D4FF"},
                    "bgcolor":  "#161C2D",
                    "bordercolor": "#1E2740",
                    "steps": [
                        {"range": [0,   0.7], "color": "rgba(0,200,83,0.15)"},
                        {"range": [0.7, 1.3], "color": "rgba(100,100,100,0.1)"},
                        {"range": [1.3, 3],   "color": "rgba(213,0,0,0.15)"},
                    ],
                    "threshold": {"line": {"color": "#FFB300", "width": 2}, "value": 1.0},
                },
            ))
            gauge.update_layout(
                template="plotly_dark", paper_bgcolor="#0E1117",
                height=180, margin=dict(l=20, r=20, t=30, b=10),
            )
            st.plotly_chart(gauge, use_container_width=True)

        # Top OI strikes
        st.markdown("#### Highest Open Interest")
        if not calls.empty and not puts.empty:
            top_calls = calls.nlargest(3, "openInterest")[["strike", "openInterest", "iv"]].copy()
            top_puts  = puts.nlargest(3, "openInterest")[["strike", "openInterest", "iv"]].copy()
            top_calls.columns = ["Strike", "OI", "IV"]
            top_puts.columns  = ["Strike", "OI", "IV"]
            top_calls["IV"] = top_calls["IV"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
            top_puts["IV"]  = top_puts["IV"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
            st.markdown("**Calls**")
            st.dataframe(top_calls, use_container_width=True, hide_index=True)
            st.markdown("**Puts**")
            st.dataframe(top_puts, use_container_width=True, hide_index=True)

    # ── OI & IV smile ──────────────────────────────────────────────────────────
    if not calls.empty and not puts.empty and exps:
        st.markdown(f"#### IV Smile — Nearest Expiry ({exps[0]})")
        atm = price
        c_filt = calls[(calls["strike"] > atm * 0.8) & (calls["strike"] < atm * 1.2)].copy()
        p_filt = puts [(puts ["strike"] > atm * 0.8) & (puts ["strike"] < atm * 1.2)].copy()
        c_filt = c_filt.dropna(subset=["iv"])
        p_filt = p_filt.dropna(subset=["iv"])

        fig2 = go.Figure()
        if not c_filt.empty:
            fig2.add_trace(go.Scatter(
                x=c_filt["strike"], y=c_filt["iv"] * 100,
                name="Calls IV", mode="lines+markers",
                line=dict(color="#00C853", width=2),
                marker=dict(size=6),
            ))
        if not p_filt.empty:
            fig2.add_trace(go.Scatter(
                x=p_filt["strike"], y=p_filt["iv"] * 100,
                name="Puts IV", mode="lines+markers",
                line=dict(color="#D50000", width=2),
                marker=dict(size=6),
            ))
        fig2.add_vline(x=atm, line=dict(color="#FFB300", dash="dash", width=1),
                       annotation_text=f"ATM ${atm:.0f}", annotation_font_color="#FFB300")
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117", height=260,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Strike", yaxis_title="Implied Vol (%)",
            legend=dict(orientation="h", y=1.05),
            yaxis=dict(gridcolor="#1E2740"),
            xaxis=dict(gridcolor="#1E2740"),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — OPTIONS CHAIN
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("#### Options Chain")
    if not exps:
        st.warning("No options data available.")
    else:
        sel_expiry = st.selectbox("Expiry date", exps, key="chain_expiry")
        calls2, puts2, _ = get_options_chain(ticker, sel_expiry)

        if not calls2.empty and not puts2.empty:
            atm = price

            # Style helpers
            def _style_calls(df):
                def _color(row):
                    itm = "#0D2818" if row.get("inTheMoney", False) else ""
                    return [f"background:{itm}" if itm else ""] * len(row)
                return df.style.apply(_color, axis=1).format({
                    "strike": "${:,.2f}",
                    "lastPrice": "${:.2f}",
                    "bid": "${:.2f}", "ask": "${:.2f}",
                    "iv": lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—",
                    "volume": "{:,.0f}", "openInterest": "{:,.0f}",
                }, na_rep="—")

            CALL_COLS = ["strike", "lastPrice", "bid", "ask", "iv", "delta", "gamma", "volume", "openInterest"]
            PUT_COLS  = CALL_COLS.copy()

            def _trim(df, cols):
                available = [c for c in cols if c in df.columns]
                return df[available].copy()

            c_show = _trim(calls2, CALL_COLS)
            p_show = _trim(puts2, PUT_COLS)

            # Filter near ATM
            show_atm = st.checkbox("Show ATM ±20% only", value=True, key="atm_filter")
            if show_atm:
                c_show = c_show[(c_show["strike"] >= atm * 0.80) & (c_show["strike"] <= atm * 1.20)]
                p_show = p_show[(p_show["strike"] >= atm * 0.80) & (p_show["strike"] <= atm * 1.20)]

            lc, rc = st.columns(2)
            with lc:
                st.markdown("<span style='color:#00C853;font-weight:700;'>CALLS</span>",
                            unsafe_allow_html=True)
                st.dataframe(
                    c_show.style.format({
                        "strike": "${:,.2f}", "lastPrice": "${:.2f}",
                        "bid": "${:.2f}", "ask": "${:.2f}",
                        "iv": lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—",
                        "volume": "{:,.0f}", "openInterest": "{:,.0f}",
                    }, na_rep="—").background_gradient(
                        subset=["openInterest"] if "openInterest" in c_show.columns else [],
                        cmap="Greens", low=0, high=0.6,
                    ),
                    use_container_width=True, hide_index=True,
                )

            with rc:
                st.markdown("<span style='color:#D50000;font-weight:700;'>PUTS</span>",
                            unsafe_allow_html=True)
                st.dataframe(
                    p_show.style.format({
                        "strike": "${:,.2f}", "lastPrice": "${:.2f}",
                        "bid": "${:.2f}", "ask": "${:.2f}",
                        "iv": lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—",
                        "volume": "{:,.0f}", "openInterest": "{:,.0f}",
                    }, na_rep="—").background_gradient(
                        subset=["openInterest"] if "openInterest" in p_show.columns else [],
                        cmap="Reds", low=0, high=0.6,
                    ),
                    use_container_width=True, hide_index=True,
                )

            # OI bar chart
            st.markdown(f"#### Open Interest by Strike — {sel_expiry}")
            fig3 = go.Figure()
            if "openInterest" in calls2.columns and "openInterest" in puts2.columns:
                c_oi = calls2[["strike", "openInterest"]].dropna()
                p_oi = puts2[["strike", "openInterest"]].dropna()
                if show_atm:
                    c_oi = c_oi[(c_oi["strike"] >= atm*0.8) & (c_oi["strike"] <= atm*1.2)]
                    p_oi = p_oi[(p_oi["strike"] >= atm*0.8) & (p_oi["strike"] <= atm*1.2)]
                fig3.add_trace(go.Bar(x=c_oi["strike"], y=c_oi["openInterest"],
                                      name="Call OI", marker_color="#00C853", opacity=0.8))
                fig3.add_trace(go.Bar(x=p_oi["strike"], y=-p_oi["openInterest"],
                                      name="Put OI", marker_color="#D50000", opacity=0.8))
                fig3.add_vline(x=atm, line=dict(color="#FFB300", dash="dash", width=1),
                               annotation_text=f"${atm:.0f}", annotation_font_color="#FFB300")
                fig3.update_layout(
                    template="plotly_dark", paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117", height=300, barmode="relative",
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis_title="Open Interest", xaxis_title="Strike",
                    legend=dict(orientation="h", y=1.05),
                    xaxis=dict(gridcolor="#1E2740"),
                    yaxis=dict(gridcolor="#1E2740"),
                )
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Options data not available for this expiry.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — IV SURFACE
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("#### Implied Volatility Surface")
    if all_opts.empty:
        st.warning("No multi-expiry options data available.")
    else:
        opt_type = st.radio("Option type", ["call", "put", "both"], horizontal=True, key="iv_type")
        filt = all_opts if opt_type == "both" else all_opts[all_opts["type"] == opt_type]
        filt = filt.dropna(subset=["iv"])
        filt = filt[(filt["iv"] > 0.01) & (filt["iv"] < 5.0)]

        # Filter near ATM
        atm = price
        filt = filt[(filt["strike"] >= atm * 0.70) & (filt["strike"] <= atm * 1.30)]

        if len(filt) >= 6:
            strikes = filt["strike"].values
            dtes    = filt["dte"].values
            ivs     = filt["iv"].values * 100  # percent

            # Interpolate onto regular grid for smooth surface
            strike_grid = np.linspace(strikes.min(), strikes.max(), 40)
            dte_grid    = np.linspace(dtes.min(), dtes.max(), 25)
            S, D = np.meshgrid(strike_grid, dte_grid)
            try:
                IV_grid = griddata((strikes, dtes), ivs, (S, D), method="cubic")
                IV_grid = np.nan_to_num(IV_grid, nan=float(np.nanmedian(ivs)))
            except Exception:
                IV_grid = np.full_like(S, np.nanmedian(ivs))

            col_tab = "Viridis" if opt_type != "put" else "Reds"
            fig4 = go.Figure(data=[go.Surface(
                x=S, y=D, z=IV_grid,
                colorscale=col_tab,
                colorbar=dict(title="IV (%)", tickfont=dict(color="#E8EDF5")),
                contours=dict(
                    z=dict(show=True, usecolormap=True, project_z=True),
                ),
                opacity=0.9,
            )])

            # ATM line
            fig4.add_trace(go.Scatter3d(
                x=[atm] * len(dte_grid), y=dte_grid,
                z=griddata((strikes, dtes), ivs, ([atm]*len(dte_grid), dte_grid), method="nearest"),
                mode="lines", line=dict(color="#FFB300", width=5), name="ATM",
            ))

            fig4.update_layout(
                template="plotly_dark", paper_bgcolor="#0E1117",
                height=520, margin=dict(l=0, r=0, t=30, b=0),
                scene=dict(
                    xaxis=dict(title="Strike ($)", backgroundcolor="#0E1117",
                               gridcolor="#1E2740", showbackground=True),
                    yaxis=dict(title="Days to Expiry", backgroundcolor="#0E1117",
                               gridcolor="#1E2740", showbackground=True),
                    zaxis=dict(title="IV (%)", backgroundcolor="#0E1117",
                               gridcolor="#1E2740", showbackground=True),
                    bgcolor="#0E1117",
                    camera=dict(eye=dict(x=1.8, y=-1.6, z=0.9)),
                ),
                title=dict(text=f"{ticker} IV Surface — {opt_type.title()}s",
                           font=dict(color="#E8EDF5", size=14)),
            )
            st.plotly_chart(fig4, use_container_width=True)

            # Heatmap view
            with st.expander("IV Heatmap (flat view)"):
                fig5 = go.Figure(go.Heatmap(
                    x=strike_grid, y=dte_grid, z=IV_grid,
                    colorscale=col_tab,
                    colorbar=dict(title="IV (%)"),
                    hoverongaps=False,
                ))
                fig5.add_vline(x=atm, line=dict(color="#FFB300", dash="dash", width=2))
                fig5.update_layout(
                    template="plotly_dark", paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117", height=350,
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="Strike ($)", yaxis_title="DTE",
                    xaxis=dict(gridcolor="#1E2740"),
                    yaxis=dict(gridcolor="#1E2740"),
                )
                st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("Not enough data points to render the IV surface. Try a larger ticker.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SOCIAL RADAR
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("#### Social Sentiment Radar")

    all_posts = wsb_posts + st_posts
    all_posts = score_batch(all_posts, text_key="text")

    if not all_posts:
        st.warning("No social data available right now. Reddit may be rate-limiting.")
    else:
        # ── Aggregate sentiment bar ────────────────────────────────────────────
        scores = [p["sentiment"] for p in all_posts]
        avg_s = float(np.mean(scores)) if scores else 0.0
        lbl   = sentiment_label(avg_s)
        clr   = sentiment_color(avg_s)

        sc1, sc2, sc3 = st.columns([1, 2, 1])
        with sc2:
            gauge2 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(avg_s, 3),
                delta={"reference": 0, "valueformat": ".3f"},
                title={"text": f"Aggregate Social Sentiment — {lbl}",
                       "font": {"size": 13, "color": "#E8EDF5"}},
                number={"font": {"color": clr, "size": 28}},
                gauge={
                    "axis": {"range": [-1, 1], "tickcolor": "#5A6478"},
                    "bar":  {"color": clr},
                    "bgcolor": "#161C2D",
                    "bordercolor": "#1E2740",
                    "steps": [
                        {"range": [-1, -0.25], "color": "rgba(213,0,0,0.2)"},
                        {"range": [-0.25, 0.25], "color": "rgba(100,100,100,0.1)"},
                        {"range": [0.25, 1],  "color": "rgba(0,200,83,0.2)"},
                    ],
                    "threshold": {"line": {"color": "#FFB300", "width": 2}, "value": avg_s},
                },
            ))
            gauge2.update_layout(
                template="plotly_dark", paper_bgcolor="#0E1117",
                height=220, margin=dict(l=40, r=40, t=30, b=10),
            )
            st.plotly_chart(gauge2, use_container_width=True)

        # ── Hawkes intensity from post timestamps ──────────────────────────────
        st.markdown("#### Hawkes Social Intensity λ(t)")
        if len(all_posts) >= 3:
            times_raw = []
            for p in all_posts:
                c = p.get("created")
                if isinstance(c, datetime):
                    times_raw.append(c.timestamp())
                elif isinstance(c, str):
                    try:
                        times_raw.append(datetime.fromisoformat(c.replace("Z","")).timestamp())
                    except Exception:
                        pass

            if len(times_raw) >= 3:
                t_min = min(times_raw)
                rel_times = sorted([(t - t_min) / 3600 for t in times_raw])  # hours
                T = rel_times[-1] + 0.1

                hawkes = HawkesProcess(mu=0.1, alpha=0.5, beta=1.0)
                hawkes.fit_to_timestamps(rel_times)
                t_grid, lam_grid = hawkes.simulate_path(T, n_points=400, seed=42)

                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(
                    x=t_grid, y=lam_grid, mode="lines",
                    name="λ_soc(t)", line=dict(color="#00D4FF", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0,212,255,0.08)",
                ))
                for rt in rel_times:
                    fig_h.add_vline(x=rt, line=dict(color="#FFB300", width=0.6, dash="dot"))

                fig_h.add_annotation(
                    text=f"Branching ratio n={hawkes.branching_ratio:.2f}  "
                         f"{'[Near-critical]' if hawkes.branching_ratio > 0.8 else '[Stationary]'}",
                    xref="paper", yref="paper", x=0.01, y=0.97,
                    showarrow=False, font=dict(color="#FFB300", size=11),
                    bgcolor="#161C2D", bordercolor="#1E2740",
                )
                fig_h.update_layout(
                    template="plotly_dark", paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117", height=260,
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="Time (hours)", yaxis_title="λ_soc(t)",
                    xaxis=dict(gridcolor="#1E2740"),
                    yaxis=dict(gridcolor="#1E2740"),
                )
                st.plotly_chart(fig_h, use_container_width=True)

                # LPA profile weights
                st.markdown("#### LPA Profile Weights")
                lpa = LatentProfileAnalysis(K=8)
                for p in all_posts:
                    lpa.update(p["sentiment"])
                weights = lpa.weights

                fig_lpa = go.Figure(go.Bar(
                    x=lpa.names, y=weights,
                    marker_color=lpa.colors,
                    text=[f"{w:.1%}" for w in weights],
                    textposition="outside",
                ))
                fig_lpa.update_layout(
                    template="plotly_dark", paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117", height=260,
                    margin=dict(l=0, r=0, t=10, b=30),
                    xaxis_title="", yaxis_title="Weight π_k(t)",
                    yaxis=dict(tickformat=".0%", gridcolor="#1E2740"),
                    xaxis=dict(tickangle=-25),
                )
                st.plotly_chart(fig_lpa, use_container_width=True)

        # ── Post feed ──────────────────────────────────────────────────────────
        st.markdown(f"#### Latest Posts — {len(all_posts)} found")
        col_filter, _ = st.columns([1, 3])
        with col_filter:
            show_only = st.selectbox("Filter", ["All", "Bullish", "Bearish", "Neutral"], key="post_filter")
        filtered = all_posts if show_only == "All" else [
            p for p in all_posts
            if show_only.lower() in p.get("sentiment_label", "").lower()
        ]
        for p in filtered[:25]:
            s = p.get("sentiment", 0)
            bar_w = int(abs(s) * 100)
            bar_color = "#00C853" if s > 0 else "#D50000" if s < 0 else "#9E9E9E"
            src = p.get("source", "Reddit")
            score_disp = p.get("score", 0)
            created = p.get("created", "")
            if isinstance(created, datetime):
                created = created.strftime("%m/%d %H:%M")
            elif isinstance(created, str):
                created = created[:16]
            url = p.get("url", "")
            title = p.get("title", "")[:120]
            link_html = f'<a href="{url}" target="_blank" style="color:#00D4FF;text-decoration:none;">↗</a>' if url else ""
            st.markdown(f"""
            <div class="post-card">
              <div class="post-title">{title} {link_html}</div>
              <div style="margin-top:6px;height:3px;background:#1E2740;border-radius:2px;">
                <div style="width:{bar_w}%;height:100%;background:{bar_color};border-radius:2px;"></div>
              </div>
              <div class="post-meta">
                {src} &nbsp;·&nbsp; {p.get('sentiment_label','—')} ({s:+.2f})
                &nbsp;·&nbsp; Score: {score_disp} &nbsp;·&nbsp; {created}
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — NEWS & EDGAR
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("#### News & Official Filings")

    scored_news    = score_batch(news_items,  text_key="text")
    scored_filings = score_batch(filings,     text_key="text")

    lnews, lfil = st.columns(2)

    with lnews:
        st.markdown(f"**Yahoo Finance News** — {len(scored_news)} items")
        if not scored_news:
            st.info("No news available.")
        else:
            # Sentiment over time bar
            sentiments_n = [n["sentiment"] for n in scored_news]
            dates_n      = [n.get("date", "")[:10] for n in scored_news]
            fig_n = go.Figure(go.Bar(
                x=list(range(len(sentiments_n))),
                y=sentiments_n,
                marker_color=[sentiment_color(s) for s in sentiments_n],
                text=[d for d in dates_n],
                textposition="outside",
                hovertext=[n["title"][:60] for n in scored_news],
            ))
            fig_n.update_layout(
                template="plotly_dark", paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117", height=180,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(range=[-1, 1], gridcolor="#1E2740", title="Sentiment"),
                xaxis=dict(showticklabels=False),
                showlegend=False,
            )
            fig_n.add_hline(y=0, line=dict(color="#5A6478", dash="dash", width=1))
            st.plotly_chart(fig_n, use_container_width=True)

            for n in scored_news[:12]:
                s = n.get("sentiment", 0)
                clr = sentiment_color(s)
                url = n.get("url", "")
                lnk = f'<a href="{url}" target="_blank" style="color:#00D4FF;">↗</a>' if url else ""
                st.markdown(f"""
                <div class="post-card">
                  <div class="post-title">{n['title'][:100]} {lnk}</div>
                  <div class="post-meta">
                    {n.get('date','')} &nbsp;·&nbsp;
                    <span style="color:{clr};">{n.get('sentiment_label','')}</span>
                    ({s:+.2f})
                  </div>
                </div>""", unsafe_allow_html=True)

    with lfil:
        st.markdown(f"**SEC EDGAR 8-K Filings** — {len(scored_filings)} recent")
        if not scored_filings:
            st.info("No filings found. EDGAR may be slow.")
        else:
            for f in scored_filings:
                s = f.get("sentiment", 0)
                clr = sentiment_color(s)
                url = f.get("url", "")
                lnk = f'<a href="{url}" target="_blank" style="color:#00D4FF;">↗ View filing</a>' if url else ""
                st.markdown(f"""
                <div class="post-card">
                  <div class="post-title">{f['title'][:90]}</div>
                  <div class="post-meta">
                    {f.get('date','')} &nbsp;·&nbsp; {f.get('source','SEC EDGAR')}
                    &nbsp;·&nbsp; {lnk}
                  </div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ERCA SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.markdown("#### ERCA Signal Engine")
    st.markdown(
        "<div style='color:#5A6478;font-size:0.82rem;margin-bottom:16px;'>"
        "Live computation of Z<sub>short</sub>(t) · Hawkes · LPA · Fractional Kelly "
        "on real social + news data</div>",
        unsafe_allow_html=True,
    )

    # ── Parameter panel ────────────────────────────────────────────────────────
    with st.expander("Model Parameters", expanded=False):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            theta1 = st.slider("θ₁ (price weight)",    0.0, 3.0, 1.0, 0.1)
            theta2 = st.slider("θ₂ (IV grad weight)",  0.0, 3.0, 0.5, 0.1)
        with pc2:
            gamma_thresh = st.slider("Γ_thresh (signal threshold)", 0.01, 5.0, 0.5, 0.05)
            kelly_c      = st.slider("Kelly fraction c",            0.05, 0.50, 0.25, 0.05)
        with pc3:
            mu_h    = st.slider("Hawkes μ (baseline)", 0.01, 1.0, 0.1, 0.01)
            alpha_h = st.slider("Hawkes α (excitation)", 0.0, 2.0, 0.5, 0.05)
            beta_h  = st.slider("Hawkes β (decay)",    0.1, 5.0, 1.0, 0.1)

    # ── Build pipeline on real data (ERCA Algorithm 1) ────────────────────────
    all_social  = score_batch(wsb_posts + st_posts, text_key="text")
    all_news_sc = score_batch(news_items, text_key="text")

    # S_off: mean news sentiment (official channel)
    S_off_avg = float(np.mean([n["sentiment"] for n in all_news_sc])) if all_news_sc else 0.0

    # IV gradient — approx from ATM options snapshot
    grad_iv = 0.0
    if not calls.empty and "iv" in calls.columns:
        atm_opts = calls[abs(calls["strike"] - price) < price * 0.05]["iv"].dropna()
        if len(atm_opts) >= 2:
            grad_iv = float(atm_opts.iloc[-1] - atm_opts.iloc[0])

    # ΔP — yesterday-to-today price return
    delta_P = 0.0
    if not price_hist.empty:
        close = price_hist["Close"].squeeze()
        if len(close) >= 2:
            delta_P = float((close.iloc[-1] - close.iloc[-2]) / (close.iloc[-2] + 1e-9))

    # ── Sequential event loop (matches ERCA Algorithm 1 exactly) ──────────────
    # Each social post is one event: update Hawkes → update LPA →
    # get S̃_soc → compute Z_short → update Kelly with Z_short
    lpa_sig  = LatentProfileAnalysis(K=8)
    detector = DivergenceDetector(theta1=theta1, theta2=theta2, gamma_thresh=gamma_thresh)
    kelly    = FractionalKelly(c=kelly_c, window=20)
    hawkes_sig = HawkesProcess(mu=mu_h, alpha=alpha_h, beta=beta_h)

    posts_to_run = all_social[:60] if all_social else []
    S_soc_agg = 0.0

    for i, post in enumerate(posts_to_run):
        t_now = float(i * 120)          # 2-min inter-arrival (seconds)

        # Phase A: update Hawkes with new social event
        hawkes_sig.update(t_now)

        # Phase B: update LPA posterior with this post's VADER score
        s_raw = post.get("sentiment", 0.0)
        lpa_sig.update(s_raw)

        # Phase C: get current LPA-weighted aggregate S̃_soc(t)
        S_soc_agg = lpa_sig.aggregate()

        # Phase D: compute Z_short(t) from evolving sentiment
        z = detector.compute(
            S_soc=S_soc_agg,
            t=t_now,
            delta_P=delta_P,
            grad_iv=grad_iv,
        )

        # Phase E: update Kelly window with Z_short (not raw sentiment)
        kelly.update(z=z)

    z_current = detector.current_z
    z_max     = detector.max_z
    n_signals = detector.n_signals
    firing    = detector.is_firing
    f_star    = kelly.compute()

    # ── Signal status ──────────────────────────────────────────────────────────
    sig_html = (
        '<span class="signal-fire">SIGNAL ACTIVE — Enter Short-Vol Position</span>'
        if firing else
        '<span class="signal-quiet">● Monitoring — No signal</span>'
    )
    st.markdown(sig_html, unsafe_allow_html=True)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Key metrics row ────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    _metric(m1, "Z_short (current)", f"{z_current:.3f}")
    _metric(m2, "Z_short (max)",     f"{z_max:.3f}")
    _metric(m3, "Signals fired",     str(n_signals))
    _metric(m4, "Kelly f*(t)",       f"{f_star:.1%}")
    _metric(m5, "S̃_soc agg",        f"{S_soc_agg:.3f}")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Z_short timeline ───────────────────────────────────────────────────────
    st.markdown("#### Z_short Timeline")
    t_arr, z_arr = detector.history_arrays()
    if len(z_arr) > 1:
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(
            x=t_arr / 60, y=z_arr,
            mode="lines", name="Z_short(t)",
            line=dict(color="#00D4FF", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
        ))
        # Threshold line
        fig_z.add_hline(y=gamma_thresh, line=dict(color="#FFB300", dash="dash", width=1.5),
                        annotation_text=f"Γ_thresh={gamma_thresh}",
                        annotation_font_color="#FFB300", annotation_position="top left")
        # Signal markers
        signal_ts = [t for t, z in zip(t_arr, z_arr) if z > gamma_thresh]
        signal_zs = [z for t, z in zip(t_arr, z_arr) if z > gamma_thresh]
        if signal_ts:
            fig_z.add_trace(go.Scatter(
                x=np.array(signal_ts) / 60, y=signal_zs,
                mode="markers", name="Signal",
                marker=dict(color="#D50000", size=9, symbol="circle"),
            ))
        fig_z.update_layout(
            template="plotly_dark", paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117", height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Time (minutes)", yaxis_title="Z_short(t)",
            legend=dict(orientation="h", y=1.05),
            xaxis=dict(gridcolor="#1E2740"),
            yaxis=dict(gridcolor="#1E2740"),
        )
        st.plotly_chart(fig_z, use_container_width=True)

    # ── Kelly gauge ────────────────────────────────────────────────────────────
    sig_col, kelly_col = st.columns(2)

    with sig_col:
        st.markdown("#### Sentiment Channels")
        ch_fig = go.Figure(go.Bar(
            x=["S̃_soc (LPA-weighted)", "S_off (News avg)", "ΔP (price chg)", "∇σ_IV (IV grad)"],
            y=[S_soc_agg, S_off_avg, delta_P, grad_iv],
            marker_color=[sentiment_color(S_soc_agg), sentiment_color(S_off_avg),
                          "#00C853" if delta_P >= 0 else "#D50000", "#9E9E9E"],
            text=[f"{v:+.3f}" for v in [S_soc_agg, S_off_avg, delta_P, grad_iv]],
            textposition="outside",
        ))
        ch_fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117", height=260,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(gridcolor="#1E2740", title="Value"),
            xaxis=dict(gridcolor="#0E1117"),
            showlegend=False,
        )
        ch_fig.add_hline(y=0, line=dict(color="#5A6478", width=1))
        st.plotly_chart(ch_fig, use_container_width=True)

    with kelly_col:
        st.markdown("#### Kelly f*(t)")
        kelly_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(f_star * 100, 2),
            number={"suffix": "%", "font": {"size": 36, "color": "#00D4FF"}},
            title={"text": f"Fractional Kelly (c={kelly_c})",
                   "font": {"size": 12, "color": "#5A6478"}},
            gauge={
                "axis": {"range": [0, kelly_c * 100], "tickcolor": "#5A6478",
                         "ticksuffix": "%"},
                "bar":  {"color": "#00D4FF" if not kelly.circuit_open else "#D50000"},
                "bgcolor": "#161C2D",
                "bordercolor": "#1E2740",
                "steps": [
                    {"range": [0, kelly_c * 33],  "color": "rgba(0,200,83,0.10)"},
                    {"range": [kelly_c*33, kelly_c*66], "color": "rgba(255,179,0,0.10)"},
                    {"range": [kelly_c*66, kelly_c*100], "color": "rgba(213,0,0,0.10)"},
                ],
            },
        ))
        if kelly.circuit_open:
            kelly_gauge.add_annotation(
                text="CIRCUIT BREAKER OPEN",
                xref="paper", yref="paper", x=0.5, y=0.05,
                showarrow=False, font=dict(color="#D50000", size=12),
            )
        kelly_gauge.update_layout(
            template="plotly_dark", paper_bgcolor="#0E1117",
            height=260, margin=dict(l=30, r=30, t=30, b=10),
        )
        st.plotly_chart(kelly_gauge, use_container_width=True)

    # ── Optimal stopping summary ───────────────────────────────────────────────
    st.markdown("#### Optimal Stopping Summary")
    st.markdown(f"""
    <div style='background:#161C2D;border:1px solid #1E2740;border-radius:10px;padding:18px;'>
      <table style='width:100%;color:#E8EDF5;font-size:0.88rem;'>
        <tr><td style='color:#5A6478;'>Stopping time τ*</td>
            <td>inf{{t : Z_short(t) > {gamma_thresh}}}</td></tr>
        <tr><td style='color:#5A6478;'>Current Z_short(t)</td>
            <td style='color:{"#D50000" if firing else "#E8EDF5"};font-weight:700;'>{z_current:.4f}</td></tr>
        <tr><td style='color:#5A6478;'>Threshold Γ</td>
            <td>{gamma_thresh:.3f}</td></tr>
        <tr><td style='color:#5A6478;'>Signal firing?</td>
            <td style='color:{"#D50000" if firing else "#00C853"};font-weight:700;'>{"YES — Enter position" if firing else "NO — Continue monitoring"}</td></tr>
        <tr><td style='color:#5A6478;'>Position size f*(t)</td>
            <td><b>{f_star:.1%}</b> of portfolio notional</td></tr>
        <tr><td style='color:#5A6478;'>Circuit breaker</td>
            <td style='color:{"#D50000" if kelly.circuit_open else "#00C853"};'>{"OPEN — trading halted" if kelly.circuit_open else "CLOSED — normal"}</td></tr>
        <tr><td style='color:#5A6478;'>Branching ratio n</td>
            <td>{mu_h:.2f} / {beta_h:.2f} = {alpha_h/beta_h:.3f}
            {"[Near-critical]" if alpha_h/beta_h > 0.8 else "[Stationary]"}</td></tr>
      </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:12px;color:#5A6478;font-size:0.75rem;'>
    <b>Research purposes only.</b> This tool validates the ERCA signal detection mechanism
    (Spearman ρ=0.4773, p&lt;0.0001 on 500 S&P 500 events). Full monetisation requires
    historical IV data from a paid options feed. Not investment advice.
    </div>""", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#5A6478;font-size:0.75rem;'>"
    "ERCA Live &nbsp;·&nbsp; Alejandro Herraiz Sen &nbsp;·&nbsp; Penn State 2026 &nbsp;·&nbsp; "
    "<a href='https://github.com/Alejandro-HerraizSen/erca-live' "
    "style='color:#00D4FF;'>GitHub</a> &nbsp;·&nbsp; "
    "Data: Yahoo Finance · Reddit · StockTwits · SEC EDGAR (all free/public)"
    "</div>",
    unsafe_allow_html=True,
)
