"""Optional Hidden Markov Model layer. Imports hmmlearn lazily so the
observable model still works if hmmlearn failed to install."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _forward_filter(model, X):
    """Causal HMM state prediction using only the forward pass (no backward).

    At each time t the most-probable state is computed from observations 0..t
    only, so there is no lookahead — safe for walk-forward backtesting.
    """
    from scipy.stats import norm

    K = model.n_components
    n = len(X)

    log_emit = np.zeros((n, K))
    for k in range(K):
        log_emit[:, k] = norm.logpdf(
            X[:, 0], model.means_[k, 0], np.sqrt(model.covars_[k].flatten()[0])
        )

    log_transmat = np.log(model.transmat_ + 1e-300)
    log_alpha = np.log(model.startprob_ + 1e-300) + log_emit[0]
    states = np.empty(n, dtype=int)
    states[0] = int(np.argmax(log_alpha))

    for t in range(1, n):
        log_alpha_new = np.empty(K)
        for k in range(K):
            log_alpha_new[k] = np.logaddexp.reduce(log_alpha + log_transmat[:, k]) + log_emit[t, k]
        log_alpha = log_alpha_new - np.logaddexp.reduce(log_alpha_new)
        states[t] = int(np.argmax(log_alpha))

    return states


def expanding_window_hmm_states(
    returns: pd.Series,
    n_components: int = 3,
    n_seeds: int = 5,
    n_iter: int = 100,
    min_train: int = 252,
    refit_every: int = 60,
) -> pd.Series | None:
    """Compute HMM states using expanding-window refit (no parameter leakage).

    Refits the HMM every `refit_every` days on all data up to that point.
    Between refits, uses the forward filter with the current model.
    Returns a Series of sorted states (0=Bear, 1=Sideways, 2=Bull).
    """
    try:
        from hmmlearn import hmm
    except ImportError:
        return None

    X = returns.dropna().to_numpy().reshape(-1, 1)
    dates = returns.dropna().index
    n = len(X)

    if n < min_train + 30:
        return None

    states = np.full(n, 1, dtype=int)  # default Sideways
    current_model = None
    current_order = None

    for t in range(min_train, n):
        if current_model is None or (t - min_train) % refit_every == 0:
            X_train = X[:t]
            best_model = None
            best_score = -np.inf
            for seed in range(n_seeds):
                model = hmm.GaussianHMM(
                    n_components=n_components, covariance_type="diag",
                    n_iter=n_iter, random_state=seed,
                )
                try:
                    model.fit(X_train)
                    score = float(model.score(X_train))
                except Exception:
                    score = -np.inf
                if score > best_score:
                    best_score = score
                    best_model = model

            if best_model is None:
                continue

            current_model = best_model
            means = np.array([best_model.means_[k][0] for k in range(n_components)])
            current_order = np.argsort(means)

        # Forward filter state for time t using current model
        fwd_states = _forward_filter(current_model, X[:t + 1])
        state_map = {old: new for new, old in enumerate(current_order)}
        states[t] = state_map[fwd_states[-1]]

    return pd.Series(states, index=dates, dtype=int)
