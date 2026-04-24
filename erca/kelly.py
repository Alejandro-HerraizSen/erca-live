"""
Fractional Kelly Position Sizing + Drawdown Circuit Breaker
ERCA paper §9.1–9.2, Definition 9.1
"""

from __future__ import annotations
import numpy as np
from collections import deque
from typing import Optional


class FractionalKelly:
    """
    Quarter-Kelly allocation (ERCA Eq. 18):
        f*(t) = c · μ̂_Z(t) / σ̂²_Z(t),  c = 0.25

    Drawdown circuit breaker (ERCA Eq. 19):
        f*(t) ← 0  if DD(t) > δ_max
    """

    def __init__(
        self,
        c: float = 0.25,
        window: int = 20,
        delta_max: float = 0.15,
    ):
        self.c = c
        self.window = window
        self.delta_max = delta_max
        self._z_buf: deque[float] = deque(maxlen=window)
        self._pnl_buf: deque[float] = deque(maxlen=500)
        self._peak_pnl: float = 0.0
        self._cumulative_pnl: float = 0.0
        self._circuit_open: bool = False

    # ------------------------------------------------------------------
    def update(self, z: float, pnl: float = 0.0) -> "FractionalKelly":
        self._z_buf.append(z)
        self._cumulative_pnl += pnl
        self._pnl_buf.append(self._cumulative_pnl)
        self._peak_pnl = max(self._peak_pnl, self._cumulative_pnl)
        # Circuit breaker: open if drawdown exceeds δ_max
        dd = self.drawdown
        if dd > self.delta_max:
            self._circuit_open = True
        elif dd < self.delta_max / 2:
            self._circuit_open = False  # reset at half-drawdown
        return self

    def compute(self) -> float:
        """Return current f*(t) in [0, c]."""
        if self._circuit_open:
            return 0.0
        if len(self._z_buf) < 3:
            return 0.0
        mu = float(np.mean(self._z_buf))
        sigma2 = float(np.var(self._z_buf))
        if sigma2 < 1e-10 or mu <= 0:
            return 0.0
        f = self.c * mu / sigma2
        return float(np.clip(f, 0.0, self.c))

    # ------------------------------------------------------------------
    @property
    def drawdown(self) -> float:
        if self._peak_pnl <= 0:
            return 0.0
        return (self._peak_pnl - self._cumulative_pnl) / (self._peak_pnl + 1e-9)

    @property
    def circuit_open(self) -> bool:
        return self._circuit_open

    @property
    def mu_z(self) -> float:
        return float(np.mean(self._z_buf)) if self._z_buf else 0.0

    @property
    def sigma2_z(self) -> float:
        return float(np.var(self._z_buf)) if len(self._z_buf) > 1 else 0.0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._z_buf.clear()
        self._pnl_buf.clear()
        self._peak_pnl = 0.0
        self._cumulative_pnl = 0.0
        self._circuit_open = False
