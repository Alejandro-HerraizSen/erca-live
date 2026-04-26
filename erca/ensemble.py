"""
Phase D — DQN Ensemble & Adaptive Model Selector  (ERCA paper §8)

Four IV predictors:
  1. NeuralCDE        — Neural Controlled Differential Equation (primary SDE path)
  2. MultiTransformer — softmax attention, volatile regimes
  3. BiTransformer    — bidirectional attention, stable/trending regimes
  4. LinearSVR        — online hinge-loss regression, stable regimes

DQN (linear Q-function, numpy-only, experience replay) selects model each step.
Reward: r_t = −MSE(selected_prediction, realised_IV) × 1000
"""

from __future__ import annotations
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List

MODEL_NAMES = ["Neural CDE", "Multi-Transformer", "Bi-Transformer", "SVR"]
N_MODELS    = len(MODEL_NAMES)


# ── 1. Neural CDE (Euler integration of learned vector field) ─────────────────

class NeuralCDE:
    """
    Approximates a Neural Controlled Differential Equation:
        dh_t = f_θ(h_t, x_t) dt   where x_t = IV observation
    f_θ is a one-hidden-layer network; h evolves via Euler steps.
    Online gradient descent on MSE loss.
    """
    def __init__(self, hidden: int = 12, seed: int = 0):
        rng  = np.random.default_rng(seed)
        d_in = hidden + 1          # [h, iv_obs]
        self.W1 = rng.normal(0, 0.1, (hidden, d_in))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.1, (1, hidden))
        self.b2 = np.zeros(1)
        self.h  = np.zeros(hidden)
        self.hidden = hidden

    def _step(self, iv: float, dt: float = 0.1) -> None:
        x  = np.append(self.h, iv)
        dh = np.tanh(self.W1 @ x + self.b1)
        self.h = self.h + dt * dh

    def predict(self, iv_history: np.ndarray) -> float:
        self.h[:] = 0
        for iv in iv_history[-12:]:
            self._step(iv)
        x   = np.append(self.h, iv_history[-1])
        h1  = np.tanh(self.W1 @ x + self.b1)
        out = float((self.W2 @ h1).item() + self.b2.item())
        return max(out, 1e-4)

    def online_update(self, iv_history: np.ndarray, target: float, lr: float = 0.005) -> None:
        pred  = self.predict(iv_history)
        error = pred - target
        x     = np.append(self.h, iv_history[-1])
        h1    = np.tanh(self.W1 @ x + self.b1)
        # Output layer
        self.W2 -= lr * error * h1[None, :]
        self.b2 -= lr * error
        # Hidden layer
        dh1 = (self.W2.T * error).ravel() * (1 - h1 ** 2)
        self.W1 -= lr * np.outer(dh1, x)
        self.b1 -= lr * dh1


# ── 2. Multi-Transformer (softmax attention, volatile regimes) ────────────────

class MultiTransformer:
    """
    Multi-head softmax attention over IV history.
    Each head has a different temperature → captures short/long memory.
    """
    def __init__(self, n_heads: int = 4):
        self.temps = np.array([0.5 + 0.5 * h for h in range(n_heads)])

    def predict(self, iv_history: np.ndarray) -> float:
        w = iv_history[-min(24, len(iv_history)):]
        if len(w) < 2:
            return float(w[-1]) if len(w) else 0.05
        preds = []
        for T in self.temps:
            scores  = w * w[-1] / T
            weights = np.exp(scores - scores.max())
            weights /= weights.sum()
            preds.append(float(weights @ w))
        return max(float(np.mean(preds)), 1e-4)


# ── 3. Bi-Transformer (bidirectional, stable regimes) ─────────────────────────

class BiTransformer:
    """
    Bidirectional attention: forward + reverse pass averaged.
    Captures both momentum and mean-reversion signals.
    """
    def __init__(self, n_heads: int = 4, temp: float = 1.0):
        self.n_heads = n_heads
        self.temp    = temp

    def predict(self, iv_history: np.ndarray) -> float:
        w = iv_history[-min(24, len(iv_history)):]
        if len(w) < 2:
            return float(w[-1]) if len(w) else 0.05

        def _attn(seq: np.ndarray) -> float:
            scores  = seq * seq[-1] / self.temp
            weights = np.exp(scores - scores.max())
            weights /= weights.sum()
            return float(weights @ seq)

        fwd = _attn(w)
        bwd = _attn(w[::-1])
        return max(0.5 * (fwd + bwd), 1e-4)


# ── 4. Online Linear SVR (stable regimes) ─────────────────────────────────────

class OnlineSVR:
    """
    Online hinge-loss SVR with 5 hand-crafted IV features:
    [mean, std, last, ewma, linear-slope]
    """
    def __init__(self, window: int = 10, eps: float = 0.01, lr: float = 0.02):
        self.window = window
        self.eps    = eps
        self.lr     = lr
        self.w      = np.zeros(5)
        self.b      = 0.05

    def _phi(self, iv_arr: np.ndarray) -> np.ndarray:
        w = iv_arr[-self.window:]
        n = len(w)
        alpha = 2 / (n + 1)
        ewma  = float(w[0])
        for v in w[1:]:
            ewma = alpha * v + (1 - alpha) * ewma
        slope = (float(w[-1]) - float(w[0])) / max(n - 1, 1)
        return np.array([w.mean(), w.std() + 1e-9, float(w[-1]), ewma, slope])

    def predict(self, iv_history: np.ndarray) -> float:
        if len(iv_history) < 2:
            return float(iv_history[-1]) if len(iv_history) else 0.05
        return max(float(self.w @ self._phi(iv_history) + self.b), 1e-4)

    def update(self, iv_history: np.ndarray, target: float) -> None:
        if len(iv_history) < 2:
            return
        phi  = self._phi(iv_history)
        pred = float(self.w @ phi + self.b)
        err  = target - pred
        if abs(err) > self.eps:
            s = np.sign(err)
            self.w += self.lr * s * phi
            self.b += self.lr * s


# ── DQN Selector (linear Q-function, numpy-only, experience replay) ───────────

STATE_DIM = N_MODELS + 1   # [σ̂_CDE, σ̂_Multi, σ̂_Bi, σ̂_SVR, λ_soc]

@dataclass
class DQNSelector:
    """
    s_t = [σ̂_CDE, σ̂_Multi, σ̂_Bi, σ̂_SVR, λ_soc]   (5-dim)
    Q(s,a) = W_a · s   (linear approximation)
    r_t = −MSE(σ̂_selected, σ_IV_realised) × 1000
    TD(0) + mini-batch replay.
    """
    alpha:   float = 0.05
    gamma:   float = 0.95
    epsilon: float = 0.12

    _W:        np.ndarray = field(init=False)
    _replay:   deque      = field(init=False)
    _counts:   np.ndarray = field(init=False)
    _losses:   List[float]= field(init=False, default_factory=list)
    _step:     int        = field(init=False, default=0)
    _last_s:   Optional[np.ndarray] = field(init=False, default=None)
    _last_a:   int        = field(init=False, default=0)

    def __post_init__(self):
        self._W      = np.random.default_rng(7).normal(0, 0.01, (N_MODELS, STATE_DIM))
        self._replay = deque(maxlen=1000)
        self._counts = np.zeros(N_MODELS, dtype=int)

    def _q(self, s: np.ndarray) -> np.ndarray:
        return self._W @ s

    def select(self, state: np.ndarray, rng_seed: Optional[int] = None) -> int:
        rng = np.random.default_rng(rng_seed if rng_seed is not None else self._step)
        self._step += 1
        self._last_s = state.copy()
        a = int(rng.integers(N_MODELS)) if rng.random() < self.epsilon else int(np.argmax(self._q(state)))
        self._last_a = a
        self._counts[a] += 1
        return a

    def update(self, reward: float, next_state: np.ndarray) -> None:
        if self._last_s is None:
            return
        self._replay.append((self._last_s, self._last_a, reward, next_state.copy()))
        if len(self._replay) < 8:
            return
        rng   = np.random.default_rng(self._step)
        batch = [self._replay[i] for i in rng.integers(len(self._replay), size=min(16, len(self._replay)))]
        loss  = 0.0
        for s, a, r, s2 in batch:
            td = r + self.gamma * np.max(self._q(s2)) - float(self._W[a] @ s)
            self._W[a] += self.alpha * td * s
            loss += td ** 2
        self._losses.append(loss / len(batch))

    @property
    def selection_dist(self) -> np.ndarray:
        total = self._counts.sum()
        return self._counts / max(total, 1)

    @property
    def q_snapshot(self) -> np.ndarray:
        """Q-value matrix (N_MODELS × STATE_DIM)."""
        return self._W.copy()


# ── Ensemble orchestrator ─────────────────────────────────────────────────────

class ERCAEnsemble:
    """
    Runs all four IV predictors + DQN selector per step.

    Each step:
      1. All four models predict σ̂_IV_{t+1}
      2. DQN selects winning model from state [preds..., λ_soc]
      3. Reward = −MSE(prev_selected_pred, realised_IV) × 1000
      4. DQN weights updated via TD + replay
      5. Online learning: SVR & NeuralCDE updated with realised IV
    """

    def __init__(self, seed: int = 42):
        self.cde   = NeuralCDE(hidden=12, seed=seed)
        self.multi = MultiTransformer(n_heads=4)
        self.bi    = BiTransformer(n_heads=4, temp=1.0)
        self.svr   = OnlineSVR(window=10)
        self.dqn   = DQNSelector()

        self._iv_buf:    List[float] = []
        self._preds_log: List[np.ndarray] = []
        self._sel_log:   List[int]   = []
        self._rew_log:   List[float] = []
        self._lam_log:   List[float] = []
        self._last_preds: Optional[np.ndarray] = None

    # ── Single step ────────────────────────────────────────────────────────
    def step(self, iv_realised: float, lambda_soc: float) -> dict:
        self._iv_buf.append(iv_realised)
        self._lam_log.append(lambda_soc)
        iv_arr = np.array(self._iv_buf)

        preds = np.array([
            self.cde.predict(iv_arr),
            self.multi.predict(iv_arr),
            self.bi.predict(iv_arr),
            self.svr.predict(iv_arr),
        ])

        # Reward + DQN update from previous step
        if self._last_preds is not None and len(self._iv_buf) >= 2:
            prev_a = self._sel_log[-1] if self._sel_log else 0
            reward = -float((self._last_preds[prev_a] - iv_realised) ** 2) * 1000
            self.dqn.update(reward, np.append(preds, lambda_soc))
            self._rew_log.append(reward)
            # Online model updates
            self.svr.update(iv_arr[:-1], iv_realised)
            self.cde.online_update(iv_arr[:-1], iv_realised)

        state  = np.append(preds, lambda_soc)
        action = self.dqn.select(state)

        self._preds_log.append(preds)
        self._sel_log.append(action)
        self._last_preds = preds.copy()

        return {
            "preds":       preds,
            "selected":    action,
            "model_name":  MODEL_NAMES[action],
            "best_pred":   float(preds[action]),
            "iv_realised": iv_realised,
            "state":       state,
        }

    # ── Full series replay ─────────────────────────────────────────────────
    def run_on_series(
        self,
        iv_series: np.ndarray,
        lambda_soc_series: np.ndarray,
    ) -> dict:
        self._iv_buf.clear(); self._preds_log.clear()
        self._sel_log.clear(); self._rew_log.clear()
        self._lam_log.clear(); self._last_preds = None
        self.dqn._counts[:] = 0

        steps = [self.step(float(iv), float(lam))
                 for iv, lam in zip(iv_series, lambda_soc_series)]

        preds_arr = np.array([s["preds"] for s in steps])   # (T, 4)
        sels      = np.array(self._sel_log)                  # (T,)
        ensemble_pred = preds_arr[np.arange(len(sels)), sels]

        return {
            "steps":          steps,
            "preds_arr":      preds_arr,
            "selections":     sels,
            "ensemble_pred":  ensemble_pred,
            "rewards":        np.array(self._rew_log) if self._rew_log else np.zeros(1),
            "dqn_dist":       self.dqn.selection_dist,
            "dqn_losses":     np.array(self.dqn._losses) if self.dqn._losses else np.zeros(1),
        }

    # ── Multi-epoch training with metric collection ────────────────────────
    def train_and_evaluate(
        self,
        train_iv: np.ndarray,
        test_iv:  np.ndarray,
        train_lam: np.ndarray,
        test_lam:  np.ndarray,
        n_epochs: int = 5,
    ) -> dict:
        """
        Train the DQN ensemble for n_epochs passes over training data.
        Evaluate on held-out test set before and after training.
        Returns comprehensive training metrics for visualisation.
        """
        # ── Pre-training snapshot on test set ──────────────────────────────
        pre_ens = ERCAEnsemble(seed=99)  # fresh, untrained
        pre_res = pre_ens.run_on_series(test_iv, test_lam)
        pre_preds = pre_res["preds_arr"]   # (T_test, 4)
        n_pre     = min(len(pre_preds), len(test_iv) - 1)
        pre_rmse  = np.sqrt(np.mean((pre_preds[:n_pre] - test_iv[1:n_pre+1, None])**2, axis=0))

        # ── Multi-epoch training ────────────────────────────────────────────
        epoch_losses: List[np.ndarray] = []
        epoch_dists:  List[np.ndarray] = []
        epoch_rmse:   List[np.ndarray] = []   # per-model RMSE on training data

        for epoch in range(n_epochs):
            self._iv_buf.clear(); self._preds_log.clear()
            self._sel_log.clear(); self._rew_log.clear()
            self._lam_log.clear(); self._last_preds = None

            step_losses_ep: List[float] = []
            for iv, lam in zip(train_iv, train_lam):
                n_before = len(self.dqn._losses)
                self.step(float(iv), float(lam))
                if len(self.dqn._losses) > n_before:
                    step_losses_ep.append(self.dqn._losses[-1])

            preds_ep = np.array([s["preds"] for s in
                                  [self.step(float(iv), float(lam))
                                   for iv, lam in zip(train_iv, train_lam)]
                                  ]) if epoch == 0 else np.array(self._preds_log)
            # per-model RMSE on training data this epoch
            target_ep = train_iv[1:len(self._preds_log)+1]
            preds_log = np.array(self._preds_log)
            if len(preds_log) > 0 and len(target_ep) > 0:
                n_ = min(len(preds_log), len(target_ep))
                ep_rmse = np.sqrt(np.mean(
                    (preds_log[:n_] - target_ep[:n_, None])**2, axis=0))
                epoch_rmse.append(ep_rmse)

            epoch_losses.append(np.array(step_losses_ep) if step_losses_ep else np.zeros(1))
            epoch_dists.append(self.dqn.selection_dist.copy())

        # ── Post-training evaluation on test set ───────────────────────────
        self._iv_buf.clear(); self._preds_log.clear()
        self._sel_log.clear(); self._rew_log.clear()
        self._lam_log.clear(); self._last_preds = None

        post_steps = [self.step(float(iv), float(lam))
                      for iv, lam in zip(test_iv, test_lam)]
        post_preds = np.array([s["preds"] for s in post_steps])
        post_sels  = np.array(self._sel_log)
        post_ens_pred = post_preds[np.arange(len(post_sels)), post_sels]

        target_test = test_iv[1:len(post_preds)+1]
        n_t = min(len(post_preds), len(target_test))
        post_rmse = np.sqrt(np.mean(
            (post_preds[:n_t] - target_test[:n_t, None])**2, axis=0))
        post_ens_rmse = float(np.sqrt(np.mean(
            (post_ens_pred[:n_t] - target_test[:n_t])**2)))

        # All DQN losses across all epochs
        all_losses = np.concatenate([l for l in epoch_losses if len(l) > 0])

        return {
            # Training dynamics
            "all_dqn_losses":   all_losses,
            "epoch_losses":     epoch_losses,
            "epoch_dists":      epoch_dists,          # (n_epochs, 4)
            "epoch_rmse":       epoch_rmse,           # (n_epochs, 4)
            # Pre/post evaluation on test set
            "pre_rmse":         pre_rmse,             # (4,) before training
            "post_rmse":        post_rmse,            # (4,) after training
            "post_ens_rmse":    post_ens_rmse,
            # Predictions on test set
            "post_preds":       post_preds,
            "post_selections":  post_sels,
            "post_ens_pred":    post_ens_pred,
            "test_iv":          test_iv,
            # DQN internals
            "q_weights":        self.dqn.q_snapshot,  # (4, 5)
            "final_dist":       self.dqn.selection_dist,
            "n_epochs":         n_epochs,
        }
