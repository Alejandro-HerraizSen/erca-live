"""
Hawkes Self-Exciting Process
O(1) recursive update — ERCA paper §4.3, Proposition 4.1
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class HawkesProcess:
    """
    Univariate exponential-kernel Hawkes process.

    Parameters
    ----------
    mu    : baseline intensity (posts / second)
    alpha : excitation amplitude
    beta  : decay rate  (must satisfy beta > alpha for stationarity)
    """
    mu: float = 0.10
    alpha: float = 0.50
    beta: float = 1.00

    _lambda: float = field(init=False, repr=False)
    _last_t: float = field(init=False, repr=False)
    _times: List[float] = field(default_factory=list, init=False, repr=False)
    _lambdas: List[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        self._lambda = self.mu
        self._last_t = 0.0

    # ------------------------------------------------------------------
    @property
    def branching_ratio(self) -> float:
        """n = alpha/beta; process is stationary iff n < 1."""
        return self.alpha / self.beta

    @property
    def event_times(self) -> List[float]:
        return list(self._times)

    @property
    def intensity_history(self) -> List[Tuple[float, float]]:
        return list(zip(self._times, self._lambdas))

    # ------------------------------------------------------------------
    def update(self, t: float) -> float:
        """
        O(1) recursive update (ERCA Eq. 7):
            λ(tₖ) = μ + e^{-β Δt}[λ(tₖ₋₁) − μ] + α
        """
        dt = max(t - self._last_t, 0.0)
        self._lambda = (
            self.mu
            + np.exp(-self.beta * dt) * (self._lambda - self.mu)
            + self.alpha
        )
        self._last_t = t
        self._times.append(t)
        self._lambdas.append(self._lambda)
        return self._lambda

    def intensity_at(self, t: float) -> float:
        """Evaluate λ(t) between events (no new event)."""
        dt = max(t - self._last_t, 0.0)
        return self.mu + np.exp(-self.beta * dt) * (self._lambda - self.mu)

    # ------------------------------------------------------------------
    def simulate(self, T: float, seed: int | None = None) -> List[float]:
        """
        Ogata's thinning algorithm — simulate event times on [0, T].
        Returns list of event times.
        """
        rng = np.random.default_rng(seed)
        times: List[float] = []
        t = 0.0
        lam = self.mu

        while t < T:
            lam_bound = lam + self.alpha  # upper bound after next event
            dt = rng.exponential(1.0 / max(lam_bound, 1e-9))
            t_candidate = t + dt
            if t_candidate >= T:
                break
            lam_at_candidate = self.mu + np.exp(-self.beta * dt) * (lam - self.mu)
            if rng.random() <= lam_at_candidate / lam_bound:
                times.append(t_candidate)
                lam = lam_at_candidate + self.alpha
            else:
                lam = lam_at_candidate
            t = t_candidate

        return times

    def simulate_path(
        self, T: float, n_points: int = 500, seed: int | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (time_grid, lambda_grid) for plotting — runs simulate() first.
        """
        events = self.simulate(T, seed=seed)
        t_grid = np.linspace(0, T, n_points)
        lam_grid = np.full(n_points, self.mu)

        for i, t in enumerate(t_grid):
            lam = self.mu
            for ev in events:
                if ev >= t:
                    break
                lam += self.alpha * np.exp(-self.beta * (t - ev))
            lam_grid[i] = lam

        return t_grid, lam_grid

    # ------------------------------------------------------------------
    def fit_to_timestamps(self, timestamps: List[float]) -> "HawkesProcess":
        """
        Quick moment-matching calibration: set mu from baseline rate,
        alpha/beta from mean cluster size heuristic.
        Returns self for chaining.
        """
        if len(timestamps) < 2:
            return self
        timestamps = sorted(timestamps)
        T = timestamps[-1] - timestamps[0]
        if T <= 0:
            return self
        N = len(timestamps)
        self.mu = max(N / T * 0.3, 0.01)   # ~30% baseline
        self.alpha = min(N / T * 0.5, self.beta * 0.9)
        return self

    def reset(self) -> None:
        self._lambda = self.mu
        self._last_t = 0.0
        self._times.clear()
        self._lambdas.clear()
