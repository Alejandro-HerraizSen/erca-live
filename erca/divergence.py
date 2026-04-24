"""
Sentiment Velocity Divergence — Z_short(t)
ERCA paper §6, Definitions 6.1–6.3, Theorem 6.4
"""

from __future__ import annotations
import numpy as np
from collections import deque
from typing import List, Tuple


class VelocityOperator:
    """
    Exponentially-weighted velocity operator V[X](t) (ERCA Eq. 10–11):
        V_k = e^{-γ Δt} · V_{k-1} + (X_k − X_{k-1}) / Δt
    """

    def __init__(self, gamma: float = 0.5):
        self.gamma = gamma
        self._V = 0.0
        self._X_prev = 0.0
        self._t_prev = 0.0

    def update(self, X: float, t: float) -> float:
        dt = t - self._t_prev
        if dt <= 1e-9:
            return self._V
        self._V = (
            np.exp(-self.gamma * dt) * self._V
            + (X - self._X_prev) / dt
        )
        self._X_prev = X
        self._t_prev = t
        return self._V

    @property
    def value(self) -> float:
        return self._V

    def reset(self) -> None:
        self._V = 0.0
        self._X_prev = 0.0
        self._t_prev = 0.0


class DivergenceDetector:
    """
    Short-Volatility Divergence Indicator Z_short(t) (ERCA Eq. 13):
        Z_short(t) = V[S̃_soc](t) − θ₁·ΔP_t − θ₂·∇σ_IV(t;K₀,T₀)

    Optimal stopping time (ERCA Theorem 6.4):
        τ* = inf{t ≥ t₀ : Z_short(t) > Γ_thresh}
    """

    def __init__(
        self,
        theta1: float = 1.0,
        theta2: float = 0.5,
        gamma: float = 0.5,
        gamma_thresh: float = 0.50,
        history_len: int = 1000,
    ):
        self.theta1 = theta1
        self.theta2 = theta2
        self.gamma_thresh = gamma_thresh
        self._V_soc = VelocityOperator(gamma)
        self._V_off = VelocityOperator(gamma)
        self._history: deque[Tuple[float, float]] = deque(maxlen=history_len)
        self._signals: List[float] = []

    # ------------------------------------------------------------------
    def compute(
        self,
        S_soc: float,
        t: float,
        delta_P: float = 0.0,
        grad_iv: float = 0.0,
    ) -> float:
        """
        Update velocity and return Z_short(t).

        Parameters
        ----------
        S_soc   : profile-weighted aggregate social sentiment
        t       : event time (seconds from session open)
        delta_P : intraday price return since event onset
        grad_iv : ATM nearest-expiry IV gradient
        """
        vsoc = self._V_soc.update(S_soc, t)
        z = vsoc - self.theta1 * delta_P - self.theta2 * grad_iv
        self._history.append((t, z))
        if z > self.gamma_thresh:
            self._signals.append(t)
        return z

    # ------------------------------------------------------------------
    @property
    def current_z(self) -> float:
        return self._history[-1][1] if self._history else 0.0

    @property
    def max_z(self) -> float:
        return max((z for _, z in self._history), default=0.0)

    @property
    def n_signals(self) -> int:
        return len(self._signals)

    @property
    def is_firing(self) -> bool:
        return self.current_z > self.gamma_thresh

    def history_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (times, z_values) as numpy arrays for plotting."""
        if not self._history:
            return np.array([]), np.array([])
        t, z = zip(*self._history)
        return np.array(t), np.array(z)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._V_soc.reset()
        self._V_off.reset()
        self._history.clear()
        self._signals.clear()
