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
from erca import (
    HawkesProcess, LatentProfileAnalysis, DivergenceDetector, FractionalKelly,
    SentimentJumpDiffusion, ERCAEnsemble, MODEL_NAMES,
)

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Dashboard",
    "Options Chain",
    "IV Surface",
    "Social Sentiment",
    "News & Filings",
    "ERCA Signal",
    "Model Explorer",
    "Backtest",
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
# TAB 6 — ERCA SIGNAL  (full 4-phase pipeline)
# ══════════════════════════════════════════════════════════════════════════════

with tab6:
    # ── Global parameters ──────────────────────────────────────────────────────
    with st.expander("Pipeline Parameters", expanded=False):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            theta1       = st.slider("θ₁ (price weight)",      0.0, 3.0, 1.0, 0.1, key="s_t1")
            theta2       = st.slider("θ₂ (IV grad weight)",    0.0, 3.0, 0.5, 0.1, key="s_t2")
        with pc2:
            gamma_thresh = st.slider("Γ_thresh",                0.001, 0.10, 0.008, 0.001, key="s_gam")
            kelly_c      = st.slider("Kelly fraction c",        0.05, 0.50, 0.25, 0.05, key="s_kc")
        with pc3:
            mu_h         = st.slider("Hawkes μ",               0.01, 1.0, 0.10, 0.01, key="s_mu")
            alpha_h      = st.slider("Hawkes α",               0.0,  2.0, 0.50, 0.05, key="s_al")
            beta_h       = st.slider("Hawkes β",               0.1,  5.0, 1.00, 0.10, key="s_be")

    # ── Pre-compute shared inputs ──────────────────────────────────────────────
    all_social  = score_batch(wsb_posts + st_posts, text_key="text")
    all_news_sc = score_batch(news_items,           text_key="text")
    S_off_avg   = float(np.mean([n["sentiment"] for n in all_news_sc])) if all_news_sc else 0.0

    grad_iv = 0.0
    if not calls.empty and "iv" in calls.columns:
        atm_opts = calls[abs(calls["strike"] - price) < price * 0.05]["iv"].dropna()
        if len(atm_opts) >= 2:
            grad_iv = float(atm_opts.iloc[-1] - atm_opts.iloc[0])

    delta_P = 0.0
    if not price_hist.empty:
        _c = price_hist["Close"].squeeze()
        if len(_c) >= 2:
            delta_P = float((_c.iloc[-1] - _c.iloc[-2]) / (_c.iloc[-2] + 1e-9))

    # ══ PHASE A — DATA INGESTION & NLP ════════════════════════════════════════
    st.markdown("""<div style='border-left:3px solid #00D4FF;padding-left:12px;
        margin:18px 0 8px 0;font-size:0.95rem;font-weight:700;color:#E8EDF5;
        letter-spacing:0.5px;'>PHASE A &nbsp;·&nbsp; DATA INGESTION &amp; NLP</div>""",
        unsafe_allow_html=True)

    pa1, pa2, pa3 = st.columns(3)
    with pa1:
        st.markdown(f"""<div class='metric-card'>
          <div class='metric-label'>OPRA / Options feed</div>
          <div class='metric-value'>{len(calls) + len(puts)}</div>
          <div style='color:#5A6478;font-size:0.72rem;'>contracts loaded</div>
        </div>""", unsafe_allow_html=True)
    with pa2:
        st.markdown(f"""<div class='metric-card'>
          <div class='metric-label'>Earnings transcript (SEC 8-K)</div>
          <div class='metric-value'>{len(filings)}</div>
          <div style='color:{sentiment_color(S_off_avg)};font-size:0.72rem;'>
            S_off = {S_off_avg:+.3f} &nbsp; (VADER compound)
          </div>
        </div>""", unsafe_allow_html=True)
    with pa3:
        st.markdown(f"""<div class='metric-card'>
          <div class='metric-label'>Social media firehose</div>
          <div class='metric-value'>{len(all_social)}</div>
          <div style='color:#5A6478;font-size:0.72rem;'>
            Reddit WSB · StockTwits · VADER+LPA
          </div>
        </div>""", unsafe_allow_html=True)

    # FinBERT-style scoring breakdown (VADER as proxy, w=[1,0,−1]^T applied)
    if all_social or all_news_sc:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        fa1, fa2 = st.columns(2)
        with fa1:
            st.markdown("**Official channel  —  FinBERT scores  (s_j^off)**")
            if all_news_sc:
                news_scores = [n["sentiment"] for n in all_news_sc]
                # apply w=[pos,neu,neg] weighting: compound already ≈ pos−neg
                fb_pos = sum(1 for s in news_scores if s > 0.05)
                fb_neu = sum(1 for s in news_scores if -0.05 <= s <= 0.05)
                fb_neg = sum(1 for s in news_scores if s < -0.05)
                fig_fb = go.Figure(go.Bar(
                    x=["Positive", "Neutral", "Negative"],
                    y=[fb_pos, fb_neu, fb_neg],
                    marker_color=["#00C853", "#FFB300", "#D50000"],
                    text=[fb_pos, fb_neu, fb_neg], textposition="outside",
                ))
                fig_fb.update_layout(template="plotly_dark", paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117", height=160, showlegend=False,
                    margin=dict(l=0,r=0,t=10,b=0),
                    yaxis=dict(gridcolor="#1E2740"), xaxis=dict(gridcolor="#0E1117"))
                st.plotly_chart(fig_fb, use_container_width=True)
            else:
                st.info("No news data.")
        with fa2:
            st.markdown("**Social channel  —  FinBERT+LPA  (s_u^soc)**")
            if all_social:
                soc_scores = [p["sentiment"] for p in all_social]
                sb_pos = sum(1 for s in soc_scores if s > 0.05)
                sb_neu = sum(1 for s in soc_scores if -0.05 <= s <= 0.05)
                sb_neg = sum(1 for s in soc_scores if s < -0.05)
                fig_sb = go.Figure(go.Bar(
                    x=["Positive", "Neutral", "Negative"],
                    y=[sb_pos, sb_neu, sb_neg],
                    marker_color=["#00C853", "#FFB300", "#D50000"],
                    text=[sb_pos, sb_neu, sb_neg], textposition="outside",
                ))
                fig_sb.update_layout(template="plotly_dark", paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117", height=160, showlegend=False,
                    margin=dict(l=0,r=0,t=10,b=0),
                    yaxis=dict(gridcolor="#1E2740"), xaxis=dict(gridcolor="#0E1117"))
                st.plotly_chart(fig_sb, use_container_width=True)
            else:
                st.info("No social data.")

    # ══ PHASE B — STOCHASTIC STATE VARIABLES ══════════════════════════════════
    st.markdown("""<div style='border-left:3px solid #7B61FF;padding-left:12px;
        margin:18px 0 8px 0;font-size:0.95rem;font-weight:700;color:#E8EDF5;
        letter-spacing:0.5px;'>PHASE B &nbsp;·&nbsp; STOCHASTIC STATE VARIABLES</div>""",
        unsafe_allow_html=True)

    # Run Hawkes + LPA sequentially on all posts
    lpa_sig    = LatentProfileAnalysis(K=8)
    detector   = DivergenceDetector(theta1=theta1, theta2=theta2, gamma_thresh=gamma_thresh)
    kelly      = FractionalKelly(c=kelly_c, window=20)
    hawkes_sig = HawkesProcess(mu=mu_h, alpha=alpha_h, beta=beta_h)

    posts_to_run = all_social[:60] if all_social else []
    S_soc_agg    = 0.0
    lam_history  = []
    ssoc_history = []

    for i, post in enumerate(posts_to_run):
        t_now = float(i * 5)         # 5-min inter-arrival (minutes)
        lam   = hawkes_sig.update(t_now)
        lam_history.append(lam)
        lpa_sig.update(post.get("sentiment", 0.0))
        S_soc_agg = lpa_sig.aggregate()
        ssoc_history.append(S_soc_agg)
        z = detector.compute(S_soc=S_soc_agg, t=t_now,
                             delta_P=delta_P, grad_iv=grad_iv)
        kelly.update(z=z)

    # Build σ(t) path via SDE using Hawkes λ and LPA S̃_soc
    n_sde   = max(len(lam_history), 30)
    atm_iv  = 0.30
    if not calls.empty and "iv" in calls.columns:
        _atm = calls[abs(calls["strike"] - price) < price * 0.05]["iv"].dropna()
        if len(_atm) > 0:
            atm_iv = float(_atm.mean())

    lam_arr  = np.array(lam_history) if lam_history else np.full(n_sde, mu_h)
    ssoc_arr = np.array(ssoc_history) if ssoc_history else np.zeros(n_sde)
    sde_obj  = SentimentJumpDiffusion(sigma_base=atm_iv, kappa=0.5, r_f=0.05)
    sigma_path = sde_obj.iv_crush_path(lam_arr, ssoc_arr)
    girsanov   = np.array([sde_obj.girsanov_drift(sde_obj.mu_t(S_off_avg), s) for s in sigma_path])
    t_min      = np.arange(len(sigma_path)) * 5

    pb1, pb2 = st.columns(2)
    with pb1:
        st.markdown("**Hawkes λ_soc(t)  +  S̃_soc(t)  (Market filtration)**")
        fig_pb = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.55, 0.45], vertical_spacing=0.05)
        fig_pb.add_trace(go.Scatter(x=t_min, y=lam_arr, mode="lines",
            name="λ_soc(t)", line=dict(color="#00D4FF", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.07)"), row=1, col=1)
        fig_pb.add_hline(y=mu_h, line=dict(color="#5A6478", dash="dash", width=1),
                         row=1, col=1)
        fig_pb.add_trace(go.Scatter(x=t_min, y=ssoc_arr, mode="lines",
            name="S̃_soc(t)", line=dict(color="#7B61FF", width=2),
            fill="tozeroy", fillcolor="rgba(123,97,255,0.07)"), row=2, col=1)
        fig_pb.add_hline(y=0, line=dict(color="#5A6478", width=1), row=2, col=1)
        fig_pb.update_layout(template="plotly_dark", paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117", height=270, margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(orientation="h", y=1.12, font=dict(size=10)))
        fig_pb.update_yaxes(title_text="λ_soc", row=1, col=1, gridcolor="#1E2740")
        fig_pb.update_yaxes(title_text="S̃_soc", row=2, col=1, gridcolor="#1E2740")
        fig_pb.update_xaxes(title_text="Time (min)", row=2, col=1, gridcolor="#1E2740")
        st.plotly_chart(fig_pb, use_container_width=True)
        st.caption(f"Virality ratio n = α/β = {alpha_h/beta_h:.3f}  ·  "
                   f"{'[Near-critical]' if alpha_h/beta_h > 0.8 else '[Stationary]'}")

    with pb2:
        st.markdown("**Sentiment-Coupled Jump-Diffusion σ(t)**")
        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(x=t_min, y=sigma_path * 100, mode="lines",
            name="σ(t) = σ_base + κ·ln(1+λ)·|S̃|",
            line=dict(color="#FFB300", width=2),
            fill="tozeroy", fillcolor="rgba(255,179,0,0.07)"))
        fig_sig.add_hline(y=atm_iv * 100, line=dict(color="#5A6478", dash="dash", width=1),
                          annotation_text=f"σ_base={atm_iv*100:.1f}%  (IV crush target)",
                          annotation_font_color="#5A6478", annotation_position="top right")
        fig_sig.add_trace(go.Scatter(x=t_min, y=girsanov, mode="lines",
            name="Girsanov θ(t) = (μ−r_f)/σ",
            line=dict(color="#00C853", width=1.5, dash="dot")))
        fig_sig.update_layout(template="plotly_dark", paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117", height=270, margin=dict(l=0,r=0,t=10,b=0),
            xaxis_title="Time (min)", yaxis_title="σ(t) %  /  θ(t)",
            legend=dict(orientation="h", y=1.12, font=dict(size=10)),
            xaxis=dict(gridcolor="#1E2740"), yaxis=dict(gridcolor="#1E2740"))
        st.plotly_chart(fig_sig, use_container_width=True)
        st.caption(f"σ_base={atm_iv*100:.1f}%  κ=0.5  ·  "
                   f"σ_peak={sigma_path.max()*100:.1f}%  →  crush Δσ={abs(sigma_path.max()-atm_iv)*100:.1f}pp")

    # ══ PHASE C — DIVERGENCE DETECTION ════════════════════════════════════════
    st.markdown("""<div style='border-left:3px solid #FFB300;padding-left:12px;
        margin:18px 0 8px 0;font-size:0.95rem;font-weight:700;color:#E8EDF5;
        letter-spacing:0.5px;'>PHASE C &nbsp;·&nbsp; DIVERGENCE DETECTION</div>""",
        unsafe_allow_html=True)

    z_current = detector.current_z
    z_max     = detector.max_z
    n_signals = detector.n_signals
    firing    = detector.is_firing
    f_star    = kelly.compute()

    sig_html = (
        '<span class="signal-fire">SIGNAL ACTIVE — Proceed to Phase D</span>'
        if firing else
        '<span class="signal-quiet">● Monitoring — No signal · back to Phase A</span>'
    )
    st.markdown(sig_html, unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    cm1, cm2, cm3, cm4, cm5 = st.columns(5)
    _metric(cm1, "Z_short (now)",  f"{z_current:.4f}")
    _metric(cm2, "Z_short (max)",  f"{z_max:.4f}")
    _metric(cm3, "Signals fired",  str(n_signals))
    _metric(cm4, "Γ threshold",    f"{gamma_thresh:.3f}")
    _metric(cm5, "S̃_soc",         f"{S_soc_agg:.3f}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    t_arr, z_arr = detector.history_arrays()
    if len(z_arr) > 1:
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(x=t_arr, y=z_arr, mode="lines", name="Z_short(t)",
            line=dict(color="#00D4FF", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.07)"))
        fig_z.add_trace(go.Scatter(x=t_arr,
            y=[sde_obj.sigma_t(float(hawkes_sig.intensity_at(t)), S_soc_agg) - atm_iv
               for t in t_arr],
            mode="lines", name="∇σ_IV proxy",
            line=dict(color="#FFB300", width=1.5, dash="dot")))
        fig_z.add_hline(y=gamma_thresh, line=dict(color="#D50000", dash="dash", width=1.5),
                        annotation_text=f"Γ = {gamma_thresh}",
                        annotation_font_color="#D50000", annotation_position="top left")
        sig_mask = z_arr > gamma_thresh
        if sig_mask.any():
            fig_z.add_trace(go.Scatter(x=t_arr[sig_mask], y=z_arr[sig_mask],
                mode="markers", name="Signal fired",
                marker=dict(color="#D50000", size=10, symbol="circle")))
        fig_z.update_layout(template="plotly_dark", paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117", height=240, margin=dict(l=0,r=0,t=10,b=0),
            xaxis_title="Time (min)", yaxis_title="Z_short(t)",
            legend=dict(orientation="h", y=1.12, font=dict(size=10)),
            xaxis=dict(gridcolor="#1E2740"), yaxis=dict(gridcolor="#1E2740"))
        st.plotly_chart(fig_z, use_container_width=True)

    # ══ PHASE D — DQN ENSEMBLE & EXECUTION ════════════════════════════════════
    st.markdown("""<div style='border-left:3px solid #D50000;padding-left:12px;
        margin:18px 0 8px 0;font-size:0.95rem;font-weight:700;color:#E8EDF5;
        letter-spacing:0.5px;'>PHASE D &nbsp;·&nbsp; DQN ENSEMBLE &amp; EXECUTION</div>""",
        unsafe_allow_html=True)

    # Build IV series for ensemble: ATM IV across all loaded expiries
    iv_series_ens   = np.array([])
    lam_series_ens  = np.array([])

    if not all_opts.empty and "iv" in all_opts.columns:
        atm_band = all_opts[(all_opts["strike"] >= price * 0.97) &
                            (all_opts["strike"] <= price * 1.03)]
        if len(atm_band) >= 4:
            iv_by_dte = atm_band.groupby("dte")["iv"].mean().sort_index()
            iv_series_ens  = iv_by_dte.values.astype(float)
            lam_series_ens = np.linspace(float(lam_arr[-1]) if len(lam_arr) else mu_h,
                                         mu_h, len(iv_series_ens))

    if len(iv_series_ens) < 4:
        # Fallback: use synthetic IV from σ(t) path
        iv_series_ens  = sigma_path if len(sigma_path) >= 4 else np.full(10, atm_iv)
        lam_series_ens = lam_arr[:len(iv_series_ens)] if len(lam_arr) >= len(iv_series_ens) \
                         else np.full(len(iv_series_ens), mu_h)

    ens = ERCAEnsemble(seed=42)
    ens_result = ens.run_on_series(iv_series_ens, lam_series_ens)

    preds_arr   = ens_result["preds_arr"]         # (T, 4)
    selections  = ens_result["selections"]        # (T,)
    ens_pred    = ens_result["ensemble_pred"]     # (T,)
    dqn_dist    = ens_result["dqn_dist"]          # (4,)
    dqn_losses  = ens_result["dqn_losses"]

    MODEL_COLORS = ["#00D4FF", "#7B61FF", "#FFB300", "#00C853"]

    pd1, pd2 = st.columns([2, 1])
    with pd1:
        st.markdown("**Four model IV predictions vs realised  (DQN selects)**")
        fig_ens = go.Figure()
        fig_ens.add_trace(go.Scatter(
            x=np.arange(len(iv_series_ens)), y=iv_series_ens * 100,
            mode="lines+markers", name="Realised IV",
            line=dict(color="#E8EDF5", width=2.5),
            marker=dict(size=5)))
        for m, (mname, mcol) in enumerate(zip(MODEL_NAMES, MODEL_COLORS)):
            fig_ens.add_trace(go.Scatter(
                x=np.arange(len(preds_arr)), y=preds_arr[:, m] * 100,
                mode="lines", name=mname,
                line=dict(color=mcol, width=1.5, dash="dot")))
        # Highlight DQN-selected predictions
        fig_ens.add_trace(go.Scatter(
            x=np.arange(len(ens_pred)), y=ens_pred * 100,
            mode="markers", name="DQN selected",
            marker=dict(
                color=[MODEL_COLORS[s] for s in selections],
                size=9, symbol="diamond",
                line=dict(color="#0E1117", width=0.5))))
        fig_ens.update_layout(template="plotly_dark", paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117", height=300, margin=dict(l=0,r=0,t=10,b=0),
            xaxis_title="Expiry index (by DTE)", yaxis_title="IV (%)",
            legend=dict(orientation="h", y=1.12, font=dict(size=9)),
            xaxis=dict(gridcolor="#1E2740"), yaxis=dict(gridcolor="#1E2740"))
        st.plotly_chart(fig_ens, use_container_width=True)

        # DQN TD loss convergence
        if len(dqn_losses) > 2:
            fig_loss = go.Figure(go.Scatter(
                x=np.arange(len(dqn_losses)), y=dqn_losses,
                mode="lines", line=dict(color="#D50000", width=1.5),
                fill="tozeroy", fillcolor="rgba(213,0,0,0.07)"))
            fig_loss.update_layout(template="plotly_dark", paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117", height=120, margin=dict(l=0,r=0,t=20,b=0),
                title=dict(text="DQN TD-loss (replay buffer)", font=dict(size=11, color="#5A6478")),
                xaxis=dict(gridcolor="#1E2740", showticklabels=False),
                yaxis=dict(gridcolor="#1E2740", title="Loss"))
            st.plotly_chart(fig_loss, use_container_width=True)

    with pd2:
        st.markdown("**DQN model selection distribution**")
        fig_dqn = go.Figure(go.Bar(
            x=MODEL_NAMES, y=dqn_dist * 100,
            marker_color=MODEL_COLORS,
            text=[f"{d:.0%}" for d in dqn_dist],
            textposition="outside"))
        fig_dqn.update_layout(template="plotly_dark", paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117", height=200, margin=dict(l=0,r=0,t=10,b=40),
            yaxis=dict(title="%", gridcolor="#1E2740", range=[0, 100]),
            xaxis=dict(tickangle=-25, gridcolor="#0E1117"), showlegend=False)
        st.plotly_chart(fig_dqn, use_container_width=True)

        # Optimal stopping + Kelly execution
        st.markdown(f"""<div style='background:#161C2D;border:1px solid #1E2740;
            border-radius:10px;padding:14px;font-size:0.82rem;margin-top:8px;'>
          <table style='width:100%;color:#E8EDF5;'>
            <tr><td style='color:#5A6478;padding:3px 0;'>τ* condition</td>
                <td style='text-align:right;'>Z_short &gt; {gamma_thresh:.3f}</td></tr>
            <tr><td style='color:#5A6478;padding:3px 0;'>Signal</td>
                <td style='text-align:right;color:{"#D50000" if firing else "#00C853"};
                font-weight:700;'>{"FIRE" if firing else "HOLD"}</td></tr>
            <tr><td style='color:#5A6478;padding:3px 0;'>f*(t) Kelly</td>
                <td style='text-align:right;'><b>{f_star:.1%}</b></td></tr>
            <tr><td style='color:#5A6478;padding:3px 0;'>Circuit breaker</td>
                <td style='text-align:right;color:{"#D50000" if kelly.circuit_open else "#00C853"};'>
                {"OPEN" if kelly.circuit_open else "closed"}</td></tr>
            <tr><td style='color:#5A6478;padding:3px 0;'>DQN best model</td>
                <td style='text-align:right;'><b>{MODEL_NAMES[int(np.argmax(dqn_dist))]}</b></td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

        # MC straddle price
        straddle = sde_obj.price_straddle(
            S0=price, K=price, T=1/252,
            lambda_soc=float(lam_arr[-1]) if len(lam_arr) else mu_h,
            S_soc=S_soc_agg, S_off=S_off_avg, n_paths=2000, seed=0)
        st.markdown(f"""<div style='background:#161C2D;border:1px solid #1E2740;
            border-radius:10px;padding:14px;font-size:0.82rem;margin-top:8px;'>
          <div style='color:#5A6478;font-size:0.72rem;margin-bottom:6px;'>
            0DTE STRADDLE — MC pricing under Q (2 000 paths)</div>
          <table style='width:100%;color:#E8EDF5;'>
            <tr><td style='color:#5A6478;'>Short call premium</td>
                <td style='text-align:right;'>${straddle["call"]:.3f}</td></tr>
            <tr><td style='color:#5A6478;'>Short put premium</td>
                <td style='text-align:right;'>${straddle["put"]:.3f}</td></tr>
            <tr><td style='color:#5A6478;font-weight:700;'>Short straddle credit</td>
                <td style='text-align:right;color:#00C853;font-weight:700;'>
                ${straddle["straddle"]:.3f}</td></tr>
            <tr><td style='color:#5A6478;'>σ used</td>
                <td style='text-align:right;'>{straddle["sigma_used"]*100:.1f}%</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div style='margin-top:14px;color:#5A6478;font-size:0.73rem;'>
    <b>Research purposes only.</b> Spearman ρ=0.4773, p&lt;0.0001 on 500 S&amp;P 500 events.
    Full production deployment requires OPRA tick feed + Bloomberg BLPAPI.
    Not investment advice.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — MODEL EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

with tab7:
    st.markdown("#### ERCA Model Explorer")
    st.markdown(
        "<div style='color:#5A6478;font-size:0.82rem;margin-bottom:20px;'>"
        "Interactively explore each component of the ERCA pipeline. "
        "Adjust parameters and observe how the mathematics responds in real time.</div>",
        unsafe_allow_html=True,
    )

    me1, me2 = st.columns(2, gap="large")

    # ── Left column: Hawkes + LPA ──────────────────────────────────────────────
    with me1:
        # ── Hawkes Explorer ───────────────────────────────────────────────────
        st.markdown("##### Hawkes Self-Exciting Process")
        st.markdown(
            "<div style='color:#5A6478;font-size:0.78rem;margin-bottom:10px;'>"
            "λ(tₖ) = μ + e<sup>−β Δt</sup>[λ(tₖ₋₁) − μ] + α &nbsp;·&nbsp; "
            "Stationary iff α/β &lt; 1</div>",
            unsafe_allow_html=True,
        )
        hc1, hc2, hc3 = st.columns(3)
        me_mu    = hc1.slider("μ (baseline)",   0.01, 1.0,  0.10, 0.01, key="me_mu")
        me_alpha = hc2.slider("α (excitation)", 0.05, 2.0,  0.50, 0.05, key="me_alpha")
        me_beta  = hc3.slider("β (decay)",      0.10, 5.0,  1.00, 0.10, key="me_beta")

        br = me_alpha / me_beta
        br_color = "#D50000" if br >= 1.0 else "#FFB300" if br > 0.8 else "#00C853"
        st.markdown(
            f"<div style='font-size:0.80rem;margin-bottom:8px;'>"
            f"Branching ratio n = α/β = <span style='color:{br_color};font-weight:700;'>"
            f"{br:.3f}</span> &nbsp;—&nbsp; "
            f"{'<span style=\"color:#D50000;\">Supercritical (explosive)</span>' if br >= 1.0 else '<span style=\"color:#FFB300;\">[Near-critical]</span>' if br > 0.8 else '<span style=\"color:#00C853;\">[Stationary]</span>'}"
            f"</div>",
            unsafe_allow_html=True,
        )

        T_sim = 24.0  # 24 hours
        hp = HawkesProcess(mu=me_mu, alpha=me_alpha, beta=min(me_beta, 4.9))
        t_grid, lam_grid = hp.simulate_path(T_sim, n_points=600, seed=7)
        events = hp.simulate(T_sim, seed=7)

        fig_me_h = go.Figure()
        fig_me_h.add_trace(go.Scatter(
            x=t_grid, y=lam_grid, mode="lines", name="λ(t)",
            line=dict(color="#00D4FF", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
        ))
        for ev in events[:200]:
            fig_me_h.add_vline(x=ev, line=dict(color="#FFB300", width=0.5, dash="dot"))
        fig_me_h.add_hline(y=me_mu, line=dict(color="#5A6478", dash="dash", width=1),
                           annotation_text="μ", annotation_font_color="#5A6478",
                           annotation_position="top right")
        fig_me_h.update_layout(
            template="plotly_dark", paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117", height=240,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Time (hours)", yaxis_title="λ(t)",
            showlegend=False,
            xaxis=dict(gridcolor="#1E2740"), yaxis=dict(gridcolor="#1E2740"),
        )
        st.plotly_chart(fig_me_h, use_container_width=True)
        st.caption(f"Simulated {len(events)} events in {T_sim:.0f}h window  ·  "
                   f"Expected rate: μ/(1−n) = {me_mu / max(1 - br, 0.01):.3f} /h")

        st.markdown("---")

        # ── LPA Explorer ──────────────────────────────────────────────────────
        st.markdown("##### Latent Profile Analysis  (K = 8)")
        st.markdown(
            "<div style='color:#5A6478;font-size:0.78rem;margin-bottom:10px;'>"
            "S̃_soc(t) = Σ π_k(t) · β_k · S_k(t) &nbsp;·&nbsp; "
            "Posterior updated per post via Bayes</div>",
            unsafe_allow_html=True,
        )

        lpa_sigma = st.slider("Profile spread σ", 0.05, 0.80, 0.30, 0.05, key="me_sigma")
        n_demo_posts = st.slider("Demo posts", 5, 80, 30, 5, key="me_posts")
        demo_seed    = st.slider("Seed (scenario)", 0, 99, 0, 1, key="me_seed")

        rng_lpa = np.random.default_rng(demo_seed)
        # Mix: mostly mildly bullish with some contrarians
        demo_sentiments = np.clip(
            rng_lpa.normal(0.20, 0.40, n_demo_posts), -1, 1
        ).tolist()

        lpa_demo = LatentProfileAnalysis(K=8, sigma=lpa_sigma)
        weight_history = []
        for s in demo_sentiments:
            lpa_demo.update(s)
            weight_history.append(lpa_demo.weights.copy())

        final_weights = lpa_demo.weights
        s_agg = lpa_demo.aggregate()

        fig_lpa_me = go.Figure(go.Bar(
            x=lpa_demo.names, y=final_weights,
            marker_color=lpa_demo.colors,
            text=[f"{w:.1%}" for w in final_weights],
            textposition="outside",
        ))
        fig_lpa_me.update_layout(
            template="plotly_dark", paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117", height=240,
            margin=dict(l=0, r=0, t=10, b=40),
            yaxis=dict(tickformat=".0%", gridcolor="#1E2740", title="π_k"),
            xaxis=dict(tickangle=-30, gridcolor="#0E1117"),
            showlegend=False,
        )
        st.plotly_chart(fig_lpa_me, use_container_width=True)
        st.caption(
            f"S̃_soc = {s_agg:+.4f}  ·  "
            f"Dominant profile: {lpa_demo.names[lpa_demo.dominant_profile]}  ·  "
            f"Mean raw sentiment: {np.mean(demo_sentiments):+.3f}"
        )

    # ── Right column: Z_short + Kelly ─────────────────────────────────────────
    with me2:
        # ── Z_short Component Breakdown ───────────────────────────────────────
        st.markdown("##### Z_short Divergence Indicator")
        st.markdown(
            "<div style='color:#5A6478;font-size:0.78rem;margin-bottom:10px;'>"
            "Z_short(t) = V[S̃_soc](t) − θ₁·ΔP_t − θ₂·∇σ_IV(t) &nbsp;·&nbsp; "
            "Signal fires when Z_short(t) > Γ</div>",
            unsafe_allow_html=True,
        )

        zc1, zc2, zc3 = st.columns(3)
        me_t1  = zc1.slider("θ₁", 0.0, 3.0, 1.0, 0.1, key="me_t1")
        me_t2  = zc2.slider("θ₂", 0.0, 3.0, 0.5, 0.1, key="me_t2")
        me_gam = zc3.slider("Γ_thresh", 0.05, 2.0, 0.50, 0.05, key="me_gam")

        # Synthetic scenario: trending S_soc, small adverse ΔP
        n_z = 80
        rng_z = np.random.default_rng(demo_seed + 1)
        t_z = np.arange(n_z) * 120.0
        s_soc_stream = np.cumsum(rng_z.normal(0.005, 0.025, n_z)).clip(-1, 1)
        dp_stream    = rng_z.normal(0.001, 0.003, n_z)
        div_stream   = rng_z.normal(0.000, 0.005, n_z)

        lpa_z = LatentProfileAnalysis(K=8)
        det_z = DivergenceDetector(theta1=me_t1, theta2=me_t2, gamma_thresh=me_gam)
        z_vals = []
        v_vals = []
        for i in range(n_z):
            lpa_z.update(float(s_soc_stream[i]))
            s_agg_z = lpa_z.aggregate()
            z = det_z.compute(S_soc=s_agg_z, t=float(t_z[i]),
                              delta_P=float(dp_stream[i]),
                              grad_iv=float(div_stream[i]))
            z_vals.append(z)
            v_vals.append(det_z._V_soc.value)

        sig_idx = [i for i, z in enumerate(z_vals) if z > me_gam]

        fig_z_me = go.Figure()
        fig_z_me.add_trace(go.Scatter(
            x=t_z / 60, y=z_vals, mode="lines", name="Z_short(t)",
            line=dict(color="#00D4FF", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
        ))
        fig_z_me.add_trace(go.Scatter(
            x=t_z / 60, y=v_vals, mode="lines", name="V[S̃_soc]",
            line=dict(color="#7B61FF", width=1.5, dash="dot"),
        ))
        if sig_idx:
            fig_z_me.add_trace(go.Scatter(
                x=t_z[sig_idx] / 60, y=[z_vals[i] for i in sig_idx],
                mode="markers", name="Signal",
                marker=dict(color="#D50000", size=9, symbol="circle"),
            ))
        fig_z_me.add_hline(y=me_gam, line=dict(color="#FFB300", dash="dash", width=1.5),
                           annotation_text=f"Γ={me_gam}",
                           annotation_font_color="#FFB300",
                           annotation_position="top left")
        fig_z_me.update_layout(
            template="plotly_dark", paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117", height=240,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Time (min)", yaxis_title="Value",
            legend=dict(orientation="h", y=1.12, font=dict(size=10)),
            xaxis=dict(gridcolor="#1E2740"), yaxis=dict(gridcolor="#1E2740"),
        )
        st.plotly_chart(fig_z_me, use_container_width=True)
        st.caption(
            f"Signals fired: {len(sig_idx)}  ·  "
            f"Max Z_short: {max(z_vals):.4f}  ·  "
            f"Final Z_short: {z_vals[-1]:.4f}"
        )

        st.markdown("---")

        # ── Kelly Sizing ──────────────────────────────────────────────────────
        st.markdown("##### Fractional Kelly Position Sizing")
        st.markdown(
            "<div style='color:#5A6478;font-size:0.78rem;margin-bottom:10px;'>"
            "f*(t) = c · μ̂_Z / σ̂²_Z &nbsp;·&nbsp; c = quarter-Kelly fraction &nbsp;·&nbsp; "
            "Circuit breaker opens at DD > δ_max</div>",
            unsafe_allow_html=True,
        )

        kc1, kc2, kc3 = st.columns(3)
        me_kc    = kc1.slider("c (fraction)", 0.05, 0.50, 0.25, 0.05, key="me_kc")
        me_win   = kc2.slider("Window", 5, 60, 20, 5, key="me_win")
        me_ddmax = kc3.slider("δ_max (DD)", 0.05, 0.40, 0.15, 0.01, key="me_ddmax")

        kelly_me = FractionalKelly(c=me_kc, window=me_win, delta_max=me_ddmax)
        f_history, dd_history = [], []
        for z in z_vals:
            kelly_me.update(z=z, pnl=z * 0.01)  # toy P&L proportional to signal
            f_history.append(kelly_me.compute())
            dd_history.append(kelly_me.drawdown)

        fig_k_me = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 row_heights=[0.65, 0.35],
                                 vertical_spacing=0.06)
        fig_k_me.add_trace(go.Scatter(
            x=t_z / 60, y=[f * 100 for f in f_history],
            mode="lines", name="f*(t) %",
            line=dict(color="#00D4FF", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
        ), row=1, col=1)
        fig_k_me.add_hline(y=me_kc * 100, line=dict(color="#FFB300", dash="dash", width=1),
                           annotation_text=f"max {me_kc*100:.0f}%",
                           annotation_font_color="#FFB300",
                           row=1, col=1)
        fig_k_me.add_trace(go.Scatter(
            x=t_z / 60, y=[d * 100 for d in dd_history],
            mode="lines", name="Drawdown %",
            line=dict(color="#D50000", width=1.5),
            fill="tozeroy", fillcolor="rgba(213,0,0,0.07)",
        ), row=2, col=1)
        fig_k_me.add_hline(y=me_ddmax * 100, line=dict(color="#D50000", dash="dash", width=1),
                           row=2, col=1)
        fig_k_me.update_layout(
            template="plotly_dark", paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117", height=240,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.12, font=dict(size=10)),
            showlegend=True,
        )
        fig_k_me.update_yaxes(title_text="f*(t) %", row=1, col=1, gridcolor="#1E2740")
        fig_k_me.update_yaxes(title_text="DD %",    row=2, col=1, gridcolor="#1E2740")
        fig_k_me.update_xaxes(title_text="Time (min)", row=2, col=1, gridcolor="#1E2740")
        st.plotly_chart(fig_k_me, use_container_width=True)
        st.caption(
            f"μ̂_Z = {kelly_me.mu_z:.4f}  ·  "
            f"σ̂²_Z = {kelly_me.sigma2_z:.6f}  ·  "
            f"Circuit breaker: {'OPEN' if kelly_me.circuit_open else 'closed'}"
        )

    # ── Equation reference panel ───────────────────────────────────────────────
    with st.expander("ERCA Equation Reference"):
        eq1, eq2 = st.columns(2)
        with eq1:
            st.markdown("""
**Hawkes intensity** (Eq. 7):
`λ(tₖ) = μ + exp(−β Δt)[λ(tₖ₋₁) − μ] + α`

**LPA aggregate sentiment** (Eq. 6):
`S̃_soc(t) = Σ π_k(t) · β_k · S_k(t)`

**Velocity operator** (Eq. 10–11):
`V_k = exp(−γ Δt)·V_{k-1} + (X_k − X_{k-1}) / Δt`
""")
        with eq2:
            st.markdown("""
**Divergence indicator** (Eq. 13):
`Z_short(t) = V[S̃_soc](t) − θ₁ΔP_t − θ₂∇σ_IV(t)`

**Optimal stopping** (Theorem 6.4):
`τ* = inf{t ≥ t₀ : Z_short(t) > Γ}`

**Fractional Kelly** (Eq. 18):
`f*(t) = c · μ̂_Z(t) / σ̂²_Z(t),  c = 0.25`
""")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

with tab8:
    st.markdown("#### Historical Backtest — ERCA Algorithm 1")
    st.markdown(
        "<div style='color:#5A6478;font-size:0.82rem;margin-bottom:16px;'>"
        "Replays the full ERCA pipeline on one year of real price data. "
        "Social events are synthesised via a Hawkes process seeded from actual weekly "
        "volatility — larger price moves generate more posting activity. "
        "Sentiment scores are correlated with price direction plus noise, "
        "matching empirical social media behaviour around earnings events.</div>",
        unsafe_allow_html=True,
    )

    if price_hist.empty:
        st.warning("Price history unavailable — cannot run backtest.")
    else:
        # ── Backtest parameters ────────────────────────────────────────────────
        with st.expander("Backtest Parameters", expanded=False):
            bc1, bc2, bc3 = st.columns(3)
            bt_theta1  = bc1.slider("θ₁ (price)",     0.0, 3.0, 1.0, 0.1, key="bt_t1")
            bt_theta2  = bc2.slider("θ₂ (IV grad)",   0.0, 3.0, 0.5, 0.1, key="bt_t2")
            bt_gamma   = bc3.slider("Γ threshold",     0.001, 0.10, 0.008, 0.001, key="bt_gam")
            bc4, bc5, bc6 = st.columns(3)
            bt_kc      = bc4.slider("Kelly c",          0.05, 0.50, 0.25, 0.05, key="bt_kc")
            bt_window  = bc5.slider("Window weeks",     4, 26, 8, 1, key="bt_win")
            bt_posts   = bc6.slider("Posts / week",     5, 60, 20, 5, key="bt_posts")

        # ── Build weekly windows from price history ────────────────────────────
        close = price_hist["Close"].squeeze().dropna()
        close.index = pd.to_datetime(close.index)
        weekly = close.resample("W").last().dropna()
        weekly_ret  = weekly.pct_change().dropna()
        weekly_vol  = close.resample("W").std().reindex(weekly_ret.index).fillna(0.01)

        bt_results = []
        rng_bt = np.random.default_rng(42)

        # Shared objects reset each week
        lpa_bt  = LatentProfileAnalysis(K=8)
        det_bt  = DivergenceDetector(theta1=bt_theta1, theta2=bt_theta2,
                                     gamma_thresh=bt_gamma)
        kelly_bt = FractionalKelly(c=bt_kc, window=bt_window)
        hawkes_bt = HawkesProcess(mu=0.10, alpha=0.50, beta=1.00)

        for i, (dt_idx, ret) in enumerate(weekly_ret.items()):
            # Reset per-window state
            lpa_bt.reset()
            det_bt.reset()
            hawkes_bt.reset()

            vol = float(weekly_vol.iloc[i]) if i < len(weekly_vol) else 0.01
            n_posts = max(
                int(rng_bt.poisson(bt_posts * (1 + 4 * abs(ret)))), 3
            )
            # Calibrate Hawkes to vol regime
            hawkes_bt.mu    = max(vol * 5, 0.05)
            hawkes_bt.alpha = min(hawkes_bt.mu * 4, hawkes_bt.beta * 0.85)

            # Synthesise social sentiment: correlated with return direction + noise
            sentiments = np.clip(
                np.sign(ret) * abs(rng_bt.normal(0.15, 0.30, n_posts))
                + rng_bt.normal(0.0, 0.20, n_posts),
                -1, 1,
            )

            # IV gradient proxy: vol shock around earnings-like moves
            grad_iv = float(vol * np.sign(-ret))   # vol up when price down
            delta_P = float(ret)

            z_week = []
            for j, s in enumerate(sentiments):
                t_now = float(j * 5)  # 5-minute inter-arrival (time in minutes)
                hawkes_bt.update(t_now)
                lpa_bt.update(float(s))
                s_agg = lpa_bt.aggregate()
                z = det_bt.compute(S_soc=s_agg, t=t_now,
                                   delta_P=delta_P, grad_iv=grad_iv)
                kelly_bt.update(z=z)
                z_week.append(z)

            max_z   = det_bt.max_z
            n_sig   = det_bt.n_signals
            f_star  = kelly_bt.compute()
            firing  = n_sig > 0

            # Next-week return for validation
            next_ret = float(weekly_ret.iloc[i + 1]) if i + 1 < len(weekly_ret) else 0.0

            bt_results.append({
                "date":       dt_idx,
                "weekly_ret": ret,
                "next_ret":   next_ret,
                "vol":        vol,
                "n_posts":    n_posts,
                "max_z":      max_z,
                "n_signals":  n_sig,
                "firing":     firing,
                "f_star":     f_star,
                "s_agg_final": lpa_bt.aggregate(),
            })

        bt_df = pd.DataFrame(bt_results).set_index("date")

        # ── Summary metrics ────────────────────────────────────────────────────
        n_signal_weeks = int(bt_df["firing"].sum())
        n_total        = len(bt_df)
        signal_rets    = bt_df[bt_df["firing"]]["next_ret"]
        no_signal_rets = bt_df[~bt_df["firing"]]["next_ret"]
        avg_sig_ret    = float(signal_rets.mean()) if len(signal_rets) > 0 else 0.0
        avg_no_ret     = float(no_signal_rets.mean()) if len(no_signal_rets) > 0 else 0.0
        spearman_r     = float(bt_df["max_z"].corr(bt_df["next_ret"].abs(), method="spearman"))

        sm1, sm2, sm3, sm4, sm5 = st.columns(5)
        _metric(sm1, "Weeks tested",    str(n_total))
        _metric(sm2, "Signal weeks",    str(n_signal_weeks))
        _metric(sm3, "Avg ret | signal", f"{avg_sig_ret:+.2%}")
        _metric(sm4, "Avg ret | quiet",  f"{avg_no_ret:+.2%}")
        _metric(sm5, "Spearman ρ",       f"{spearman_r:.4f}")

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── Signal timeline overlaid on price chart ────────────────────────────
        st.markdown("##### Signal Timeline vs Price")
        fig_bt = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.50, 0.25, 0.25],
            vertical_spacing=0.04,
            subplot_titles=["Price + Signals", "Z_short (weekly max)", "Kelly f*(t)"],
        )

        # Price
        fig_bt.add_trace(go.Scatter(
            x=close.index, y=close.values,
            mode="lines", name="Close",
            line=dict(color=color, width=1.8),
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.06)",
        ), row=1, col=1)

        # Signal markers on price chart
        signal_rows = bt_df[bt_df["firing"]]
        if not signal_rows.empty:
            # Map each signal week to the close price on that date
            sig_prices = []
            for d in signal_rows.index:
                nearby = close.index.searchsorted(d)
                idx = min(nearby, len(close) - 1)
                sig_prices.append(float(close.iloc[idx]))

            fig_bt.add_trace(go.Scatter(
                x=signal_rows.index, y=sig_prices,
                mode="markers", name="ERCA Signal",
                marker=dict(
                    color="#D50000", size=10,
                    symbol="triangle-down",
                    line=dict(color="#FF6D00", width=1),
                ),
            ), row=1, col=1)

        # Z_short weekly max
        fig_bt.add_trace(go.Bar(
            x=bt_df.index, y=bt_df["max_z"],
            name="Z_short max",
            marker_color=[
                "#D50000" if f else "#00D4FF"
                for f in bt_df["firing"]
            ],
            opacity=0.8,
        ), row=2, col=1)
        fig_bt.add_hline(y=bt_gamma, line=dict(color="#FFB300", dash="dash", width=1),
                         row=2, col=1)

        # Kelly f*
        fig_bt.add_trace(go.Scatter(
            x=bt_df.index, y=bt_df["f_star"] * 100,
            mode="lines", name="f*(t) %",
            line=dict(color="#7B61FF", width=1.5),
            fill="tozeroy", fillcolor="rgba(123,97,255,0.07)",
        ), row=3, col=1)

        fig_bt.update_layout(
            template="plotly_dark", paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117", height=580,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", y=1.03, font=dict(size=10)),
            hovermode="x unified",
        )
        for r in [1, 2, 3]:
            fig_bt.update_xaxes(gridcolor="#1E2740", row=r, col=1)
            fig_bt.update_yaxes(gridcolor="#1E2740", row=r, col=1)
        fig_bt.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig_bt.update_yaxes(title_text="Z_short",   row=2, col=1)
        fig_bt.update_yaxes(title_text="f*(t) %",   row=3, col=1)
        st.plotly_chart(fig_bt, use_container_width=True)

        # ── Scatter: Z_short vs subsequent return ──────────────────────────────
        st.markdown("##### Z_short Signal Strength vs Subsequent Weekly Return")
        scatter_col, stats_col = st.columns([2, 1])
        with scatter_col:
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=bt_df["max_z"], y=bt_df["next_ret"] * 100,
                mode="markers",
                marker=dict(
                    color=[abs(z) for z in bt_df["max_z"]],
                    colorscale="Plasma", size=7, opacity=0.8,
                    colorbar=dict(title="Z_short", thickness=12),
                    line=dict(color="#1E2740", width=0.5),
                ),
                text=[str(d)[:10] for d in bt_df.index],
                hovertemplate="Z_short=%{x:.3f}<br>Next return=%{y:.2f}%<br>%{text}",
                name="Weekly obs.",
            ))
            # Signal firing weeks highlighted
            if not signal_rows.empty:
                fig_sc.add_trace(go.Scatter(
                    x=signal_rows["max_z"],
                    y=signal_rows["next_ret"] * 100,
                    mode="markers",
                    marker=dict(color="#D50000", size=11, symbol="star",
                                line=dict(color="#FF6D00", width=1)),
                    name="Signal fired",
                ))
            fig_sc.add_vline(x=bt_gamma, line=dict(color="#FFB300", dash="dash", width=1),
                             annotation_text=f"Γ={bt_gamma}", annotation_font_color="#FFB300")
            fig_sc.add_hline(y=0, line=dict(color="#5A6478", width=1))
            fig_sc.update_layout(
                template="plotly_dark", paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117", height=320,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Max Z_short (week t)",
                yaxis_title="Subsequent weekly return %",
                legend=dict(font=dict(size=10)),
                xaxis=dict(gridcolor="#1E2740"),
                yaxis=dict(gridcolor="#1E2740"),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        with stats_col:
            # Signal vs no-signal return distribution
            st.markdown("**Signal accuracy**")
            st.markdown(f"""
            <div style='background:#161C2D;border:1px solid #1E2740;
                        border-radius:10px;padding:16px;font-size:0.85rem;'>
              <table style='width:100%;color:#E8EDF5;'>
                <tr><td style='color:#5A6478;padding:4px 0;'>Total weeks</td>
                    <td style='text-align:right;'><b>{n_total}</b></td></tr>
                <tr><td style='color:#5A6478;padding:4px 0;'>Signal weeks</td>
                    <td style='text-align:right;color:#D50000;'><b>{n_signal_weeks}</b></td></tr>
                <tr><td style='color:#5A6478;padding:4px 0;'>Signal rate</td>
                    <td style='text-align:right;'><b>{n_signal_weeks/max(n_total,1):.1%}</b></td></tr>
                <tr><td colspan='2' style='border-top:1px solid #1E2740;padding:6px 0 2px 0;'></td></tr>
                <tr><td style='color:#5A6478;padding:4px 0;'>Avg ret | signal</td>
                    <td style='text-align:right;color:{"#00C853" if avg_sig_ret>0 else "#D50000"};'>
                    <b>{avg_sig_ret:+.2%}</b></td></tr>
                <tr><td style='color:#5A6478;padding:4px 0;'>Avg ret | quiet</td>
                    <td style='text-align:right;color:{"#00C853" if avg_no_ret>0 else "#D50000"};'>
                    <b>{avg_no_ret:+.2%}</b></td></tr>
                <tr><td colspan='2' style='border-top:1px solid #1E2740;padding:6px 0 2px 0;'></td></tr>
                <tr><td style='color:#5A6478;padding:4px 0;'>Spearman ρ</td>
                    <td style='text-align:right;'><b>{spearman_r:.4f}</b></td></tr>
                <tr><td style='color:#5A6478;padding:4px 0;'>Signal threshold</td>
                    <td style='text-align:right;'><b>Γ = {bt_gamma:.2f}</b></td></tr>
              </table>
            </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            # Return distribution histogram
            fig_dist = go.Figure()
            if len(signal_rets) > 1:
                fig_dist.add_trace(go.Histogram(
                    x=signal_rets * 100, name="Signal weeks",
                    marker_color="#D50000", opacity=0.7,
                    xbins=dict(size=1),
                ))
            if len(no_signal_rets) > 1:
                fig_dist.add_trace(go.Histogram(
                    x=no_signal_rets * 100, name="Quiet weeks",
                    marker_color="#00D4FF", opacity=0.5,
                    xbins=dict(size=1),
                ))
            fig_dist.add_vline(x=0, line=dict(color="#5A6478", width=1))
            fig_dist.update_layout(
                template="plotly_dark", paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117", height=200, barmode="overlay",
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Next-week return %",
                yaxis_title="Count",
                legend=dict(font=dict(size=10), x=0, y=1),
                xaxis=dict(gridcolor="#1E2740"),
                yaxis=dict(gridcolor="#1E2740"),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        # ── Weekly results table ───────────────────────────────────────────────
        with st.expander("Weekly Results Table"):
            display_df = bt_df[["weekly_ret","next_ret","vol","max_z","n_signals","f_star"]].copy()
            display_df.columns = ["Weekly Ret", "Next Ret", "Vol", "Max Z_short", "Signals", "f*(t)"]
            display_df.index = display_df.index.strftime("%Y-%m-%d")
            st.dataframe(
                display_df.style
                .format({
                    "Weekly Ret": "{:+.2%}",
                    "Next Ret":   "{:+.2%}",
                    "Vol":        "{:.4f}",
                    "Max Z_short":"{:.4f}",
                    "Signals":    "{:.0f}",
                    "f*(t)":      "{:.1%}",
                })
                .background_gradient(subset=["Max Z_short"], cmap="plasma")
                .map(
                    lambda v: "color:#D50000;font-weight:700;" if v > 0 else "color:#00C853;",
                    subset=["Signals"],
                ),
                use_container_width=True,
            )

        st.markdown(
            "<div style='margin-top:12px;color:#5A6478;font-size:0.75rem;'>"
            "<b>Methodology note:</b> Social events are synthesised from real price volatility "
            "using a calibrated Hawkes process. Sentiment scores are correlated with weekly "
            "price direction (correlation ≈ 0.6 + noise). This matches the paper's empirical "
            "validation approach (Spearman ρ=0.4773, p&lt;0.0001 on 500 real S&amp;P 500 events). "
            "Not investment advice."
            "</div>",
            unsafe_allow_html=True,
        )


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
