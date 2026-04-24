# ⚡ ERCA Live

**Earnings Call Risk & Confidence Analyzer — Live Streamlit Dashboard**

> Alejandro Herraiz Sen · The Pennsylvania State University · 2026  
> Based on the ERCA working paper: *A Stochastic Framework for Multi-Modal Sentiment Divergence, Latent Profile Dynamics, and 0DTE Options Mispricing*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What it does

ERCA Live runs the full paper pipeline on **5 high-engagement tickers** (AAPL · TSLA · NVDA · AMZN · COIN) using entirely free, public data sources:

| Source | Data | Library |
|---|---|---|
| Yahoo Finance | Prices, options chains, earnings dates, news | `yfinance` |
| Reddit | r/wallstreetbets, r/options, r/stocks posts | `requests` (public JSON API) |
| StockTwits | Symbol stream messages | `requests` (public API) |
| SEC EDGAR | 8-K filings (official disclosures) | `requests` (ATOM/RSS feed) |
| VADER | Sentiment scoring (NLP) | `vaderSentiment` |

---

## Dashboard tabs

| Tab | What you see |
|---|---|
| 🎯 **Dashboard** | Price chart, IV smile, earnings countdown, P/C ratio, top OI strikes |
| 📊 **Options Chain** | Full calls/puts table with OI heatmap, strike filter, ATM highlighting |
| 🌋 **IV Surface** | Interactive 3D implied volatility surface + flat heatmap |
| 📡 **Social Radar** | Hawkes λ(t) intensity chart, LPA profile weights, Reddit/StockTwits feed |
| 📰 **News & EDGAR** | Yahoo Finance news + SEC 8-K filings, all VADER-scored |
| ⚡ **ERCA Signal** | Live Z_short(t), Hawkes, LPA, Fractional Kelly gauge, optimal stopping |

---

## Quick start

```bash
git clone https://github.com/Alejandro-HerraizSen/erca-live.git
cd erca-live
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## The math (from the paper)

**Hawkes intensity** — $O(1)$ recursive update:
$$\lambda_{\text{soc}}(t_k) = \mu + e^{-\beta \Delta t_k}[\lambda_{\text{soc}}(t_{k-1}) - \mu] + \alpha$$

**LPA aggregate sentiment** — $K=8$ investor archetypes:
$$\tilde{S}_{\text{soc}}(t) = \sum_{k=1}^{8} \pi_k(t) \cdot \beta_k \cdot S_k(t)$$

**Divergence indicator** — optimal stopping signal:
$$Z_{\text{short}}(t) = \mathcal{V}[\tilde{S}_{\text{soc}}](t) - \theta_1 \Delta P_t - \theta_2 \nabla\sigma_{IV}(t; K_0, T_0)$$

**Fractional Kelly** — quarter-Kelly position sizing:
$$f^*(t) = c \cdot \frac{\hat{\mu}_Z(t)}{\hat{\sigma}^2_Z(t)}, \quad c = 0.25$$

**Empirical result** (500 real S&P 500 earnings events):  
Spearman $\rho = 0.4773$ ($p < 0.0001$), $t = 10.976$ ($p = 2.99 \times 10^{-25}$)

---

## License

MIT — see [LICENSE](LICENSE).  
Not investment advice.
