"""
Latent Profile Analysis — K=8 investor archetypes
ERCA paper §4.4, Definition 4.2
"""

from __future__ import annotations
import numpy as np
from typing import List

# Fixed profile loadings (β_k) estimated from financial text corpora
# Profile 8 (index 7) is "Irrational Exuberance" with β₈ < 0
PROFILE_LOADINGS = np.array([0.85, 0.62, 0.41, 0.18, -0.05, -0.28, -0.47, -0.67])

PROFILE_NAMES = [
    "Informed Bull",        # β=+0.85
    "Moderate Bull",        # β=+0.62
    "Cautious Bull",        # β=+0.41
    "Uncertain",            # β=+0.18
    "Mild Skeptic",         # β=-0.05
    "Moderate Bear",        # β=-0.28
    "Informed Bear",        # β=-0.47
    "Irrational Exuberance",# β=-0.67  ← inverts raw sentiment
]

PROFILE_COLORS = [
    "#00C853", "#69F0AE", "#B9F6CA", "#FFD740",
    "#FFB300", "#FF6D00", "#DD2C00", "#D50000",
]


class LatentProfileAnalysis:
    """
    Bayesian LPA with K=8 Gaussian profiles.

    At each post arrival, the posterior weights π_k(t) are updated:
        π_k(t) ∝ π_k(t⁻) · p(s_raw | k)

    Profile-weighted aggregate sentiment (ERCA Eq. 6):
        S̃_soc(t) = Σ_k π_k(t) · β_k · S_k(t)
    """

    def __init__(self, K: int = 8, sigma: float = 0.30):
        assert K <= len(PROFILE_LOADINGS), "K must be ≤ 8"
        self.K = K
        self.betas = PROFILE_LOADINGS[:K]
        self.names = PROFILE_NAMES[:K]
        self.colors = PROFILE_COLORS[:K]
        self.sigma = sigma          # profile Gaussian spread
        self._pi = np.ones(K) / K  # uniform prior
        self._S_k = np.zeros(K)    # running mean sentiment per profile
        self._n_k = np.zeros(K)    # counts per profile

    # ------------------------------------------------------------------
    @property
    def weights(self) -> np.ndarray:
        return self._pi.copy()

    @property
    def dominant_profile(self) -> int:
        return int(np.argmax(self._pi))

    # ------------------------------------------------------------------
    def update(self, s_raw: float) -> np.ndarray:
        """
        Update posterior weights given new raw sentiment score s_raw ∈ [-1,1].
        Returns updated π vector.
        """
        likelihoods = np.exp(
            -0.5 * ((s_raw - self.betas) / self.sigma) ** 2
        )
        likelihoods = np.maximum(likelihoods, 1e-12)
        self._pi = self._pi * likelihoods
        total = self._pi.sum()
        if total > 1e-12:
            self._pi /= total
        else:
            self._pi = np.ones(self.K) / self.K  # reset to uniform

        # Update running profile-specific sentiment
        k_star = int(np.argmax(likelihoods))
        self._n_k[k_star] += 1
        alpha = 1.0 / self._n_k[k_star]
        self._S_k[k_star] += alpha * (s_raw - self._S_k[k_star])

        return self._pi.copy()

    def aggregate(self) -> float:
        """
        S̃_soc(t) = Σ_k π_k · β_k · S_k   (ERCA Eq. 6)
        """
        return float(np.dot(self._pi, self.betas * self._S_k))

    def aggregate_batch(self, scores: List[float]) -> float:
        """Process a batch of scores and return final aggregate."""
        self.reset()
        for s in scores:
            self.update(s)
        return self.aggregate()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._pi = np.ones(self.K) / self.K
        self._S_k = np.zeros(self.K)
        self._n_k = np.zeros(self.K)
