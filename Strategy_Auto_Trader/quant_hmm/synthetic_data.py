"""Synthetic OHLCV path generator for Monte Carlo stress-testing.

Fit a Gaussian HMM on real log-returns, then sample many synthetic price
paths from it. Each path runs through the existing backtest/live_sim engine
unchanged — the stress test exercises outcome variance, not code.

Return generation: block bootstrap (default block_size=24) draws contiguous
blocks of real returns per HMM state, preserving within-state autocorrelation
so RSI/SMA momentum signals can build. Set block_size=1 for legacy iid
Gaussian draws.

Volume and intrabar range are extracted from the *same* block positions as
returns (when block_size > 1), preserving their joint within-bar distribution
instead of sampling them independently.

Limitations:
- Gaussian per-state emissions underestimate fat tails / jump risk (affects
  the HMM state sequence only; block bootstrap draws real returns so tails
  are preserved there).
- Generating HMM is a single static fit on full real history per ticker;
  regime-parameter uncertainty is not stressed by default (use transmat_noise
  > 0 to perturb transition probabilities per path).
- Track B: each ticker's path uses its own independent HMM/seed — no
  cross-ticker correlation. Simultaneous co-crashes are not modelled, so
  capital-contention stress in monte_carlo_live_sim underestimates
  simultaneous-drawdown severity. Flagged as the largest fidelity gap; out
  of scope for v1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .quant_engine import _HOURS_PER_YEAR, fit_hmm_expanding


def fit_generating_hmm(
    close: pd.Series,
    n_components: int = 3,
    n_seeds: int = 5,
    n_iter: int = 100,
):
    """Fit a Gaussian HMM on log-returns of `close`.

    Returns (model, order) where order[j] is the HMM state index whose mean
    is the j-th smallest — i.e. order[0]=Bear, order[1]=Sideways,
    order[2]=Bull for a 3-state fit. Same convention as fit_hmm_expanding.

    Raises ValueError if fit fails for all seeds (no silent fallback).
    """
    log_returns = np.log(close.values[1:] / close.values[:-1])
    result = fit_hmm_expanding(log_returns, n_components=n_components,
                               n_seeds=n_seeds, n_iter=n_iter)
    if result is None:
        raise ValueError(
            f"HMM fit failed for all {n_seeds} seeds on {len(log_returns)} bars"
        )
    return result


def label_hidden_states(model, order: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Viterbi-decode `returns`, remapped to 0=Bear/1=Sideways/2=Bull.

    order is the state_order array from fit_generating_hmm / fit_hmm_expanding.
    labels[i] is the regime state that generated returns[i].
    """
    X = returns.reshape(-1, 1)
    raw_labels = model.predict(X)
    inv = np.argsort(order)
    return inv[raw_labels]


def _perturb_transmat(
    transmat: np.ndarray,
    noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a Dirichlet-perturbed copy of transmat.

    Each row is resampled from Dirichlet(row * (1/noise) + eps).
    Higher noise = lower concentration = more deviation from the original.
    Rows always sum to 1. Never mutates the input (the fitted model is shared
    across paths/processes).
    """
    K = transmat.shape[0]
    perturbed = np.empty_like(transmat)
    concentration = 1.0 / noise
    for i in range(K):
        alpha = transmat[i] * concentration + 1e-8
        perturbed[i] = rng.dirichlet(alpha)
    return perturbed


def sample_coupled_state_sequence(
    n: int,
    startprob: np.ndarray,
    transmat: np.ndarray,
    raw_market_states: np.ndarray,
    coupling: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a length-n Markov chain state sequence coupled to a market state sequence.

    At each step the ticker's effective transition distribution blends its own
    fitted dynamics with a pull toward the current market state:

        blended_row[m, s] = (1 - coupling) * transmat[s]
        blended_row[m, s][m] += coupling          # shift mass toward market state m
        (renormalized so rows sum to 1)

    All three arrays (startprob, transmat, raw_market_states) must be in the
    same raw HMM state space — callers convert semantic market states to raw via
    order[market_states] before calling.

    coupling=0.0: identical to _sample_state_sequence (pure ticker dynamics).
    coupling=1.0: states[t] == raw_market_states[t-1] for all t > 0.

    Blended cumulative transmats are pre-computed once outside the loop (O(K³),
    K=3 → trivial). Loop is O(n) with np.searchsorted, matching _sample_state_sequence.
    """
    K = transmat.shape[0]
    blended_cumtrans = np.empty((K, K, K))
    for m in range(K):
        for s in range(K):
            row = (1.0 - coupling) * transmat[s].copy()
            row[m] += coupling
            row /= row.sum()  # renormalize against float drift
            blended_cumtrans[m, s] = np.cumsum(row)
            blended_cumtrans[m, s, -1] = 1.0  # guard against float rounding

    states = np.empty(n, dtype=np.intp)
    states[0] = int(rng.choice(K, p=startprob))
    u = rng.random(n - 1)
    for t in range(1, n):
        m = int(raw_market_states[t - 1])
        states[t] = np.searchsorted(blended_cumtrans[m, states[t - 1]], u[t - 1])
    return states


def _sample_state_sequence(
    n: int,
    startprob: np.ndarray,
    transmat: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a length-n Markov chain state sequence.

    Replaces hmmlearn's model.sample() Python loop. Uses one bulk rng.random(n-1)
    call + np.searchsorted for transitions, and a single vectorized rng.normal()
    per state for emissions — ~20-50× faster than the hmmlearn implementation on
    paths of 5000+ bars.

    Clamps cumtrans[:, -1] = 1.0 so searchsorted never returns K (out of bounds).
    """
    cumtrans = np.cumsum(transmat, axis=1)
    cumtrans[:, -1] = 1.0  # guard against float rounding
    states = np.empty(n, dtype=np.intp)
    states[0] = int(rng.choice(len(startprob), p=startprob))
    u = rng.random(n - 1)
    for t in range(1, n):
        states[t] = np.searchsorted(cumtrans[states[t - 1]], u[t - 1])
    return states


def sample_synthetic_path(
    model,
    order: np.ndarray,
    n_returns: int,
    seed: int,
    transmat_noise: float = 0.0,
    market_states: np.ndarray | None = None,
    market_coupling: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample `n_returns` bars from `model`.

    Returns (log_returns, state_labels) both length n_returns, with
    state_labels remapped to the Bear/Sideways/Bull convention via order.

    Uses a vectorized NumPy Markov chain sampler for all cases — never calls
    model.sample() (hmmlearn's Python-loop implementation is ~20-50× slower).

    transmat_noise=0.0 (default): uses model.transmat_ unchanged.
    transmat_noise>0: Dirichlet-perturbs a copy of model.transmat_ per call.
    The fitted model is never mutated — it is shared across paths/processes.

    market_states: optional length-n_returns array of semantic market states
    (Bear=0/Sideways=1/Bull=2) sampled from a market HMM. When provided with
    market_coupling>0, each ticker's state sequence is biased toward the market
    state at each bar via sample_coupled_state_sequence. Converted from semantic
    to this ticker's raw HMM state space via order before use.

    transmat_noise and market_coupling compose: noise perturbs the base transmat
    first, then coupling blends the perturbed transmat with the market state.
    """
    rng = np.random.default_rng(seed)
    if transmat_noise > 0.0:
        transmat = _perturb_transmat(model.transmat_, transmat_noise, rng)
    else:
        transmat = model.transmat_

    if market_states is not None and market_coupling > 0.0:
        # Convert semantic market states to this ticker's raw HMM state space.
        # order[j] = raw HMM state index for semantic state j (Bear=0/Sideways=1/Bull=2).
        raw_market = order[market_states[:n_returns]]
        raw_labels = sample_coupled_state_sequence(
            n_returns, model.startprob_, transmat, raw_market, market_coupling, rng,
        )
    else:
        raw_labels = _sample_state_sequence(n_returns, model.startprob_, transmat, rng)

    means = model.means_.ravel()
    stds = np.sqrt(model.covars_.ravel())
    iid_returns = np.empty(n_returns, dtype=float)
    for s in range(model.n_components):
        mask = raw_labels == s
        count = int(mask.sum())
        if count > 0:
            iid_returns[mask] = rng.normal(means[s], stds[s], count)

    inv = np.argsort(order)
    state_labels = inv[raw_labels]
    return iid_returns, state_labels


def bootstrap_by_state(
    historical_values: np.ndarray,
    historical_state_labels: np.ndarray,
    synthetic_state_labels: np.ndarray,
    seed: int,
) -> np.ndarray:
    """For each bar in synthetic_state_labels, draw with replacement from
    historical_values[historical_state_labels == that_state].

    Empty donor pool (regime never visited historically) falls back to the
    full historical_values array — not an error.

    Alignment: callers must pass historical_values[1:] (length N-1) paired
    with log_returns[i] = log(close[i+1]/close[i]), NOT [:-1]. An off-by-one
    silently bootstraps from the wrong bar's regime with no crash.
    """
    rng = np.random.default_rng(seed)
    result = np.empty(len(synthetic_state_labels), dtype=float)
    for state in np.unique(synthetic_state_labels):
        mask = synthetic_state_labels == state
        donor_mask = historical_state_labels == state
        donor_pool = historical_values[donor_mask] if donor_mask.any() else historical_values
        result[mask] = rng.choice(donor_pool, size=int(mask.sum()), replace=True)
    return result


def _choose_block_starts(
    historical_state_labels: np.ndarray,
    synthetic_state_labels: np.ndarray,
    block_size: int,
    seed: int,
) -> np.ndarray:
    """Choose block start indices for a block bootstrap over synthetic_state_labels.

    Returns an integer array of length ceil(n / block_size), where each entry
    is a start position in historical_state_labels / historical_values. The
    k-th entry covers synthetic positions [k*block_size : (k+1)*block_size].

    Separating index selection from value extraction lets callers reuse the
    same starts for multiple arrays (e.g. log_returns, volume, range_ratio)
    so all three come from the same historical bars.
    """
    rng = np.random.default_rng(seed)
    n = len(synthetic_state_labels)
    n_hist = len(historical_state_labels)
    starts: list[int] = []
    i = 0
    while i < n:
        state = int(synthetic_state_labels[i])
        donor_mask = historical_state_labels == state
        if not donor_mask.any():
            donor_mask = np.ones(n_hist, dtype=bool)
        donor_indices = np.where(donor_mask)[0]
        valid_starts = donor_indices[donor_indices + block_size <= n_hist]
        if len(valid_starts) == 0:
            valid_starts = donor_indices
        start_idx = int(rng.choice(valid_starts))
        starts.append(start_idx)
        # Advance by the actual fill length _extract_blocks will use for this block.
        # In the normal case (valid start found) this equals block_size. In the fallback
        # (block_size > n_hist) a block starting near the end is shorter, so we advance
        # by n_hist - start_idx to match the real fill and avoid generating too few blocks.
        i += min(n_hist - start_idx, block_size)
    return np.array(starts, dtype=np.intp)


def _extract_blocks(
    historical_values: np.ndarray,
    block_starts: np.ndarray,
    block_size: int,
    n: int,
) -> np.ndarray:
    """Extract n values from historical_values using pre-chosen block starts."""
    result = np.empty(n, dtype=float)
    i = 0
    for start_idx in block_starts:
        if i >= n:
            break
        block = historical_values[start_idx : start_idx + block_size]
        take = min(len(block), n - i)
        result[i : i + take] = block[:take]
        i += take
    return result


def bootstrap_blocks_by_state(
    historical_values: np.ndarray,
    historical_state_labels: np.ndarray,
    synthetic_state_labels: np.ndarray,
    block_size: int = 24,
    seed: int = 0,
) -> np.ndarray:
    """Draw contiguous blocks of historical_values per HMM state.

    For each position in synthetic_state_labels, draws a block of `block_size`
    consecutive values from historical_values where historical_state_labels
    matched that state. Preserves within-state autocorrelation (RSI/SMA
    momentum buildup) that iid per-bar sampling destroys.

    Falls back to the full historical array for states with no valid block start
    (i.e. when historical_values is shorter than block_size for a given state).
    """
    starts = _choose_block_starts(historical_state_labels, synthetic_state_labels,
                                   block_size, seed)
    return _extract_blocks(historical_values, starts, block_size,
                           len(synthetic_state_labels))


def assemble_synthetic_ohlcv(
    real_df: pd.DataFrame,
    log_returns: np.ndarray,
    volume: np.ndarray,
    range_ratio: np.ndarray,
    start_price: float | None = None,
) -> pd.DataFrame:
    """Assemble a synthetic OHLCV DataFrame from sampled components.

    Close = start_price * exp(cumsum(log_returns))
    Open  = prior bar's Close  (first Open = start_price)
    High  = max(Open, Close) * (1 + range_ratio / 2)
    Low   = min(Open, Close) * (1 - range_ratio / 2)

    Index is built by tiling real_df.index so that paths longer than real
    history produce strictly unique, monotonically increasing timestamps
    instead of duplicate DatetimeIndex entries (which would corrupt
    _simulate_portfolio_value's interest accrual and
    _compute_seasonal_volume_ratio's hour-of-day grouping).
    """
    n = len(log_returns)
    if start_price is None:
        start_price = float(real_df["Close"].iloc[0])

    closes = start_price * np.exp(np.cumsum(log_returns))
    opens = np.empty(n, dtype=float)
    opens[0] = start_price
    opens[1:] = closes[:-1]

    highs = np.maximum(opens, closes) * (1.0 + range_ratio / 2.0)
    lows = np.minimum(opens, closes) * (1.0 - range_ratio / 2.0)

    real_idx = real_df.index
    if len(real_idx) > 1:
        one_bar = (real_idx[-1] - real_idx[0]) / max(len(real_idx) - 1, 1)
    else:
        one_bar = pd.Timedelta(hours=1)
    span = real_idx[-1] - real_idx[0] + one_bar

    needed_tiles = (n // len(real_idx)) + 2
    tiled = pd.DatetimeIndex(
        [ts + i * span for i in range(needed_tiles) for ts in real_idx]
    )[:n]

    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes,
         "Volume": volume.astype(float)},
        index=tiled,
    )


def generate_synthetic_df(
    real_df: pd.DataFrame,
    model,
    order: np.ndarray,
    historical_log_returns: np.ndarray,
    historical_state_labels: np.ndarray,
    n_bars: int,
    seed: int,
    block_size: int = 24,
    transmat_noise: float = 0.0,
    market_states: np.ndarray | None = None,
    market_coupling: float = 0.0,
) -> pd.DataFrame:
    """Convenience wrapper: sample one synthetic path and assemble OHLCV.

    historical_log_returns and historical_state_labels must both be length
    len(real_df) - 1, aligned as bootstrap_by_state's docstring requires.

    block_size: contiguous-block length for return sampling (default 24 = one
    trading day). Block bootstrap draws real returns per HMM state so RSI/SMA
    signals build naturally. block_size=1 reverts to iid Gaussian draws (the
    legacy behaviour, which produces 0 trades in Track A because Gaussian
    emissions have no autocorrelation).

    When block_size > 1, volume and range_ratio are extracted from the *same*
    historical block positions as log_returns (not independently sampled), so
    all three components share their within-bar joint distribution.

    transmat_noise: Dirichlet noise applied to a copy of model.transmat_ per
    path (default 0.0 = off). Stresses HMM parameter uncertainty. Try 0.05-0.2.

    market_states/market_coupling: passed through to sample_synthetic_path.
    See that function's docstring. Default None/0.0 = no cross-ticker coupling.
    """
    iid_log_returns, state_labels = sample_synthetic_path(
        model, order, n_bars, seed, transmat_noise=transmat_noise,
        market_states=market_states, market_coupling=market_coupling,
    )

    hist_volume = real_df["Volume"].values[1:]
    hist_range = ((real_df["High"] - real_df["Low"]) / real_df["Close"]).values[1:]

    if block_size > 1:
        block_starts = _choose_block_starts(
            historical_state_labels, state_labels, block_size, seed,
        )
        log_returns = _extract_blocks(
            historical_log_returns, block_starts, block_size, n_bars,
        )
        volume = _extract_blocks(hist_volume, block_starts, block_size, n_bars)
        range_ratio = _extract_blocks(hist_range, block_starts, block_size, n_bars)
    else:
        log_returns = iid_log_returns
        volume = bootstrap_by_state(
            hist_volume, historical_state_labels, state_labels, seed=seed ^ 0xA1,
        )
        range_ratio = bootstrap_by_state(
            hist_range, historical_state_labels, state_labels, seed=seed ^ 0xB2,
        )

    return assemble_synthetic_ohlcv(real_df, log_returns, volume, range_ratio)
