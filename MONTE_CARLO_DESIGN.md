# Monte Carlo Synthetic-Data Stress Test — Design Document

## Purpose

The Monte Carlo module stress-tests the trading strategy by running backtests on many synthetic price paths rather than a single real history. The goal is to measure the *distribution* of strategy outcomes — Sharpe, drawdown, total return — across markets that look statistically similar to real history but differ in specific event sequences. This separates luck-from-skill and surfaces tail risks that a single backtest masks.

There are two independent tracks:

- **Track A** (`markov_cli/monte_carlo.py`) — single-ticker, isolated capital. How does the strategy perform if this ticker's market had gone differently?
- **Track B** (`markov_cli/monte_carlo_live_sim.py`) — full portfolio, capital-arbitrated. How does the portfolio's capital allocation behave across many tickers simultaneously when all market histories are synthetic?

---

## Generating Model: Gaussian HMM

### What it is

A 3-state Gaussian Hidden Markov Model is fitted on the real log-returns of each ticker. Each hidden state corresponds to a market regime:

- **State 0 — Bear**: lowest-mean log-return state
- **State 1 — Sideways**: middle-mean
- **State 2 — Bull**: highest-mean

The convention (Bear=0, Sideways=1, Bull=2) is enforced by sorting the state means after fitting and building an `order` array that maps raw hmmlearn state indices to this semantic ordering. All downstream code uses remapped indices.

### Fitting

```
fit_generating_hmm(close_series)  →  (model, order)
```

Internally calls `fit_hmm_expanding` from `quant_engine.py`:

- `hmmlearn.hmm.GaussianHMM`, `n_components=3`, `covariance_type="diag"`
- 5 random seeds; keeps the seed with the highest log-likelihood
- `n_iter=100` EM iterations
- Returns `None` only if all 5 seeds fail (not masked — raises `ValueError` in `fit_generating_hmm`)

This is the **same HMM class** used by the live trading engine, but fitted here on the full real history as a *generating* model rather than used incrementally for regime probabilities.

### State sequence generation

```python
X, raw_labels = model.sample(n_returns, random_state=seed)
```

`hmmlearn.GaussianHMM.sample` jointly samples hidden states and Gaussian emissions using the fitted transition matrix and per-state means/variances. The hidden-state sequence follows the Markov transition structure (regime persistence, transition probabilities) learned from real data. The emissions are iid Gaussian draws *within each state*.

**Key design decision**: we use the HMM-generated state sequence (Markov structure) but discard the iid Gaussian emission values for returns. Instead, we replace them with block-bootstrapped real returns (see below).

---

## Synthetic Return Generation: Block Bootstrap

### Problem with iid Gaussian emissions

The HMM's Gaussian per-state emissions sample returns independently at each bar. This destroys within-regime autocorrelation. Without autocorrelation:

- RSI stays near 50 (no directional drift to push it above 70 or below 30)
- SMA crossovers never build (no sustained trend to separate fast/slow moving averages)
- The composite entry score never reaches `buy_threshold=3.0`
- Result: 0 trades on every synthetic path

### Block bootstrap fix

```
bootstrap_blocks_by_state(historical_values, historical_state_labels,
                           synthetic_state_labels, block_size=24, seed)
```

For each position `i` in the synthetic state sequence:

1. Look up `synthetic_state_labels[i]` (e.g. Bull)
2. Find all indices in real history where `historical_state_labels == Bull`
3. From those indices, find valid block start positions: indices where `start + block_size <= len(history)` (so a full block fits)
4. Sample one start index uniformly at random
5. Extract the contiguous block `historical_values[start : start + block_size]`
6. Fill `result[i : i + block_size]` with those values
7. Advance `i` by `block_size`

Default `block_size=24` = one trading day (hourly bars). This preserves intra-day autocorrelation: within a block, the returns are real consecutive hourly returns from a real Bull (or Bear, Sideways) period, so RSI and SMA have the directional momentum to build signals. Inter-block correlation is not preserved — each block is independently drawn.

**Fallback**: if no valid block start exists for a state (e.g. the real history has fewer bars in that state than `block_size`), falls back to any index in that state without the length constraint. If the state was never seen historically, falls back to the full array.

### Volume and range_ratio: iid bootstrap per state

Volume and intrabar range are still sampled iid per bar (not in blocks), grouped by HMM state:

```
bootstrap_by_state(historical_values, historical_state_labels,
                   synthetic_state_labels, seed)
```

Draws with replacement from the historical pool for each state. This preserves the per-state distribution (e.g. Bull periods tend to have higher volume) without requiring block structure — volume is not autocorrelated in the way returns are.

---

## OHLCV Assembly

```
assemble_synthetic_ohlcv(real_df, log_returns, volume, range_ratio)
```

Given `n` log-returns:

```
Close[0] = real_df["Close"][0]  # anchor to real starting price
Close[i] = Close[0] * exp(sum(log_returns[0..i]))

Open[0]  = Close[0]
Open[i]  = Close[i-1]  # prev bar's close = this bar's open

High[i]  = max(Open[i], Close[i]) * (1 + range_ratio[i] / 2)
Low[i]   = min(Open[i], Close[i]) * (1 - range_ratio[i] / 2)
```

This guarantees `High >= max(Open, Close)` and `Low <= min(Open, Close)` by construction — the OHLCV is internally consistent.

### Index tiling

Synthetic paths may be longer than the real `real_df` (e.g. a 3-year synthetic path from 2 years of real data). To avoid duplicate DatetimeIndex entries (which would corrupt hour-of-day seasonality calculations and interest accrual), the index is built by tiling the real index with a uniform time offset:

```python
span = real_idx[-1] - real_idx[0] + one_bar_width
tiled = [ts + i * span for i in range(needed_tiles) for ts in real_idx][:n]
```

Result: strictly monotonically increasing, unique DatetimeIndex.

---

## Convenience Wrapper

```python
generate_synthetic_df(real_df, model, order,
                       historical_log_returns, historical_state_labels,
                       n_bars, seed, block_size=24)
```

1. `sample_synthetic_path(model, order, n_bars, seed)` → `(iid_log_returns, state_labels)`
2. If `block_size > 1`: replace `iid_log_returns` with `bootstrap_blocks_by_state(historical_log_returns, ..., state_labels, block_size, seed)`
3. `bootstrap_by_state` for volume and range_ratio using the same `state_labels`
4. `assemble_synthetic_ohlcv` → synthetic OHLCV DataFrame

`block_size=1` reverts to pure iid Gaussian (legacy, produces 0 trades — not useful for stress testing).

---

## Track A: Single-Ticker (`monte_carlo.py`)

### Setup (once per run)

1. Fetch real hourly OHLCV for the ticker (yfinance or IBKR)
2. `fit_generating_hmm(real_df["Close"])` → `(model, order)`
3. Compute `historical_log_returns` and `historical_state_labels` from real history
4. `volatility_profile(ticker)` → `real_trend_quality`; printed as context info
5. Resolve `entry_strategy`, `exit_strategy` from `resolve_strategy(strategy_name, vol_filter_ok=True)`

**Note on `vol_filter_ok`**: the real ticker's `vol_filter_ok` from the vol-screen is printed but **not used** for synthetic paths. All synthetic paths use `vol_filter_ok=True`. Rationale: the vol-screen is a live-trading admission gate (is this ticker trending well *right now*?). For stress-testing, we want to examine what the strategy would do if it were trading — permanently gating to 0 trades via a real-data filter defeats the stress test entirely.

### Per-path (parallelised via ProcessPoolExecutor)

```
_run_one_path(synth_df, strategy_name, vol_filter_ok=True, ..., backtest_kwargs)
```

1. `resolve_strategy(strategy_name, vol_filter_ok=True, entry_overrides, exit_overrides)` — fresh Entry/Exit objects, not shared
2. `make_cost_model(cost_model_name, ticker, trade_cost)` — fresh per path, ensures picklability
3. `consolidated_backtest(synth_df, regime_model=None, position_sizer=None, entry_strategy=..., exit_strategy=..., **backtest_kwargs)`

**`regime_model=None` invariant**: passing a `PersistentHMMRegimeModel` would read/write the real ticker's on-disk HMM cache (`state/hmm/<ticker>.pkl`), corrupting it with synthetic-data fits. `None` forces the engine to fit a fresh in-memory HMM on the synthetic data itself, which is correct — each synthetic path gets its own HMM regime model fitted on that path's bars.

**`position_sizer=None` invariant**: the engine builds a fresh `KellySizer` from the path's own trade history. No trade history leaks between paths.

### Output

`data/monte_carlo/<ticker>_<strategy>_<timestamp>/`
- `mc_summary.json` — percentile bands (p5/p25/p50/p75/p95 + mean) for Sharpe, Sortino, max_drawdown, total_return, final_portfolio; plus `prob_of_loss`
- `mc_paths.csv` — one row per path with all raw metrics
- `sample_paths/` — optional raw OHLCV CSVs (first N paths, for inspection)

---

## Track B: Portfolio (`monte_carlo_live_sim.py`)

Track B runs the full `generate_candidates → vol-gate → top-K → arbitrate` pipeline on synthetic data, exercising capital contention across many tickers simultaneously.

### Setup (once per strategy)

1. `generate_candidates(tickers=all_universe, strategy_name=..., workers=1, source=...)` on **real data** — generates real candidate trades and trend quality scores
2. `_filter_candidates_by_daily_trend_quality` — apply real vol-screen to real candidates
3. `filter_candidates_by_top_tickers(..., top_k=70)` — select top-70 tickers by score; this is the **fixed ticker universe** used for all synthetic paths
4. `fetch_hourly_cached` for each fixed ticker — full OHLCV
5. `fit_generating_hmm` per ticker — `(model, order)` per ticker stored in `hmm_models` dict
6. Compute `historical_returns` and `historical_labels` per ticker

### Per-path (parallelised via ProcessPoolExecutor)

`_make_df_by_ticker(path_idx)` constructs `df_by_ticker: dict[str, pd.DataFrame]`:

```python
for ticker in final_tickers:
    ticker_seed = (seed + path_idx) ^ hash(ticker) & 0xFFFFFFFF
    df_by_ticker[ticker] = generate_synthetic_df(
        real_dfs[ticker], hmm_models[ticker][0], hmm_models[ticker][1],
        historical_returns[ticker], historical_labels[ticker],
        n_bars=len(real_dfs[ticker]), seed=ticker_seed, block_size=...
    )
```

Each ticker gets a different seed derived from `(path_idx, ticker)` for reproducibility. Tickers are independent — no cross-ticker correlation.

`_run_one_path(df_by_ticker, fixed_tickers, strategy_name, pot_sizes, ...)`:

1. `generate_candidates(tickers=fixed_tickers, df_by_ticker=df_by_ticker, use_persistent_cache=False, workers=1)` — runs full backtest per ticker on the synthetic data; `workers=1` to avoid nested process pools
2. `_filter_candidates_by_daily_trend_quality` — re-evaluated per path from the synthetic data's trend quality scores
3. `filter_candidates_by_top_tickers` — top-K filter dynamically re-applied within the fixed universe
4. `arbitrate(candidates, initial_cash=pot_size, ...)` — walk forward from `start_date`, admit entries against shared capital pool

**`use_persistent_cache=False` invariant**: prevents reads/writes to the HMM disk cache (`state/hmm/`) and the OHLCV IBKR cache for synthetic data.

**`workers=1` inside `generate_candidates`**: path-level parallelism already uses a `ProcessPoolExecutor`; nested pools are not supported by `concurrent.futures`.

**`regime_model=None` in the DI seam**: `generate_candidates` calls `run_ticker_backtest` with `df=synth_df` and `use_persistent_cache=False`, which forces `regime_model` to `None` inside the engine — same invariant as Track A.

**Fixed universe**: the top-K tickers are selected **once** from real data and reused across all synthetic paths unchanged. Re-deriving the universe per path from the full 603-ticker universe would be computationally infeasible (~93s/path already at top-70 scale).

### Output

`data/monte_carlo/<strategy>_portfolio_<timestamp>/`
- `mc_summary.json` — per pot-size: percentile bands for total_return, final_portfolio, max_drawdown, n_admitted; plus `prob_of_loss`
- `mc_paths.csv` — one row per (path, pot_size) with raw metrics

---

## Key Invariants (both tracks)

| Invariant | Why |
|-----------|-----|
| `regime_model=None` on every `consolidated_backtest` call | Prevents synthetic HMM fits from corrupting real ticker's on-disk cache |
| `use_persistent_cache=False` in `generate_candidates` for synthetic paths | Same — no cache reads/writes for synthetic data |
| `position_sizer=None` | Engine builds fresh `KellySizer` per call; no trade history leaks between paths |
| `vol_filter_ok=True` in Track A | Vol-screen is a live-admission gate, not a stress-test gate; forcing True lets signals drive entries |
| Fixed ticker universe in Track B | Selected once from real data; computationally infeasible to re-derive per path |
| `workers=1` inside `generate_candidates` per path | No nested `ProcessPoolExecutor` pools |
| Per-ticker seeds: `(seed + path_idx) ^ hash(ticker)` | Cross-ticker independence; reproducible per (run, path, ticker) triple |

---

## Known Limitations

### No cross-ticker correlation
Each ticker's synthetic path is drawn from its own independent HMM and seed. Simultaneous market crashes (all tickers down together) are not modelled. Track B's capital-contention stress test therefore underestimates simultaneous-drawdown severity. This is the largest fidelity gap.

### Single static generating HMM
One HMM is fitted on full real history per ticker. Regime-parameter uncertainty (the uncertainty in the HMM's own fitted parameters) is not stressed. The HMM's transition matrix and state means/variances are treated as ground truth.

### Gaussian state-sequence emissions
The HMM state *sequence* is generated by sampling Markov transitions from Gaussian emissions. Block bootstrap replaces the return values, but the state *timing* (how long each regime lasts, when transitions happen) is still governed by Gaussian iid sampling from the state-sequence model. Real regimes may have heavier-tailed durations.

### Inter-block return gaps
Block bootstrap preserves intra-block (within 24-bar) autocorrelation. It does not preserve inter-block autocorrelation — the junction between two consecutive blocks introduces an artificial break in return continuity. Volume and range_ratio are sampled iid (no blocks), so their cross-correlation with return magnitude is approximate.

### Track B vol-gate uses synthetic trend quality
`_filter_candidates_by_daily_trend_quality` is re-evaluated per synthetic path using trend quality derived from the synthetic data. A Bear-dominated synthetic path may exclude tickers the real vol-screen would approve, and vice versa. This is intentional (per-path gating is more realistic) but amplifies variance.

---

## CLI Reference

```bash
# Track A: single ticker, 300 paths, workers=4, block_size=24 (default)
uv run python -m Strategy_Auto_Trader.markov_cli.monte_carlo \
    --ticker SPY --strategy default --n-paths 300 --workers 4

# Track B: full universe, top-70, 50 paths
uv run python -m Strategy_Auto_Trader.markov_cli.monte_carlo_live_sim \
    --universe --strategies optimised_new --n-paths 50 \
    --workers 4 --pot-sizes 25000 --top-k 70

# Track A with legacy iid (reproduces 0-trades result, for comparison)
... --block-size 1
```

---

## Source Files

| File | Role |
|------|------|
| `quant_hmm/synthetic_data.py` | HMM fitting, block bootstrap, OHLCV assembly |
| `markov_cli/monte_carlo.py` | Track A CLI |
| `markov_cli/monte_carlo_live_sim.py` | Track B CLI |
| `quant_hmm/ticker_ranking.py` | DI seam: `df`, `df_by_ticker`, `use_persistent_cache` params on `run_ticker_backtest` / `generate_candidates` |
| `quant_hmm/consolidated_engine.py` | Backtest engine; `regime_model=None` path |
| `quant_hmm/quant_engine.py` | `fit_hmm_expanding`, `fetch_hourly` |
| `tests/quant_hmm/test_synthetic_data.py` | Unit + integration tests for synthetic data module |
| `tests/markov_cli/test_monte_carlo.py` | Track A tests |
| `tests/markov_cli/test_monte_carlo_live_sim.py` | Track B tests |
