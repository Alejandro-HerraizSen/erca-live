"""
Sentiment-Coupled Jump-Diffusion SDE  (ERCA paper §5)
Girsanov transform → risk-neutral measure Q  (ERCA paper §7)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SentimentJumpDiffusion:
    """
    Price SDE:
        dS_t/S_{t-} = μ(t,S_off) dt + σ(t,S̃_soc) dW_t + (e^{J_t}−1) dN_t^off

    Vol coupling (IV crush dynamics):
        σ(t) = σ_base + κ·ln(1+λ_soc)·|S̃_soc|
        → σ_base as λ_soc → μ_soc  (post-earnings IV crush)

    Risk-neutral measure (Girsanov):
        dW_t^Q = dW_t + (μ(t,S_off) − r_f) / σ(t,S̃_soc) dt
    """
    sigma_base: float = 0.25    # baseline (post-crush) annualised vol
    kappa: float = 0.50         # sentiment-vol coupling strength
    r_f: float = 0.05           # risk-free rate (annualised)
    jump_intensity: float = 2.0 # λ_N: Poisson jump arrival rate (per year)
    jump_mean: float = -0.03    # mean log-jump size J_t
    jump_std: float = 0.06      # std  log-jump size

    def sigma_t(self, lambda_soc: float, S_soc: float) -> float:
        """σ(t) = σ_base + κ·ln(1+λ_soc)·|S̃_soc|"""
        return self.sigma_base + self.kappa * np.log1p(max(lambda_soc, 0.0)) * abs(S_soc)

    def mu_t(self, S_off: float) -> float:
        """Drift μ(t,S_off) — shifts with official compound-Poisson sentiment."""
        return self.r_f + 0.10 * S_off

    def girsanov_drift(self, mu: float, sigma: float) -> float:
        """Market price of risk θ = (μ − r_f) / σ  (Girsanov kernel)."""
        return (mu - self.r_f) / max(sigma, 1e-9)

    # ── Monte-Carlo path simulation (Euler-Maruyama) ────────────────────────
    def simulate_path(
        self,
        T: float = 30 / 252,     # horizon in years
        S0: float = 100.0,
        n_steps: int = 30,
        S_off_path: Optional[np.ndarray] = None,
        lambda_soc_path: Optional[np.ndarray] = None,
        S_soc_path: Optional[np.ndarray] = None,
        risk_neutral: bool = False,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (times [years], prices, sigmas).
        Under P when risk_neutral=False; under Q when True.
        """
        rng = np.random.default_rng(seed)
        dt  = T / n_steps
        sqrt_dt = np.sqrt(dt)

        if S_off_path    is None: S_off_path    = np.zeros(n_steps)
        if lambda_soc_path is None: lambda_soc_path = np.full(n_steps, 0.10)
        if S_soc_path    is None: S_soc_path    = np.zeros(n_steps)

        times  = np.linspace(0, T, n_steps + 1)
        prices = np.empty(n_steps + 1); prices[0] = S0
        sigmas = np.empty(n_steps + 1)

        for i in range(n_steps):
            sig = self.sigma_t(float(lambda_soc_path[i]), float(S_soc_path[i]))
            mu  = self.mu_t(float(S_off_path[i]))
            drift = self.r_f if risk_neutral else mu

            dW = rng.standard_normal() * sqrt_dt

            # Compound Poisson jumps
            n_jumps = int(rng.poisson(self.jump_intensity * dt))
            jump = np.sum(np.expm1(rng.normal(self.jump_mean, self.jump_std, n_jumps))) if n_jumps else 0.0

            prices[i + 1] = max(prices[i] * (1.0 + drift * dt + sig * dW + jump), 1e-6)
            sigmas[i] = sig

        sigmas[-1] = sigmas[-2]
        return times, prices, sigmas

    # ── IV crush trajectory ─────────────────────────────────────────────────
    def iv_crush_path(
        self,
        lambda_soc_path: np.ndarray,
        S_soc_path: np.ndarray,
    ) -> np.ndarray:
        """σ(t) array along the event window — shows IV crush as λ_soc decays."""
        return np.array([self.sigma_t(l, s) for l, s in zip(lambda_soc_path, S_soc_path)])

    # ── Monte-Carlo option pricing under Q ───────────────────────────────────
    def price_straddle(
        self,
        S0: float,
        K: float,
        T: float,
        lambda_soc: float,
        S_soc: float,
        S_off: float,
        n_paths: int = 2000,
        seed: int = 0,
    ) -> dict:
        """
        Price a short 0DTE straddle under Q via MC.
        Returns {call, put, straddle, sigma_used}.
        """
        sig = self.sigma_t(lambda_soc, S_soc)
        rng = np.random.default_rng(seed)
        dt  = T
        sqrt_dt = np.sqrt(dt)
        n_jumps_arr = rng.poisson(self.jump_intensity * dt, n_paths)
        dW = rng.standard_normal(n_paths) * sqrt_dt
        jumps = np.array([
            np.sum(np.expm1(rng.normal(self.jump_mean, self.jump_std, n))) if n else 0.0
            for n in n_jumps_arr
        ])
        S_T = S0 * np.maximum(1.0 + self.r_f * dt + sig * dW + jumps, 1e-6)
        call_payoffs = np.maximum(S_T - K, 0.0)
        put_payoffs  = np.maximum(K - S_T, 0.0)
        disc = np.exp(-self.r_f * T)
        return {
            "call":       float(disc * call_payoffs.mean()),
            "put":        float(disc * put_payoffs.mean()),
            "straddle":   float(disc * (call_payoffs + put_payoffs).mean()),
            "sigma_used": sig,
        }
