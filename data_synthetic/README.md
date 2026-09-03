# Synthetic hourly backtest data

Everything under this directory is **fabricated**, generated locally by
`Strategy_Auto_Trader/synthetic_backtest_data/`. It is never mixed with real
data on disk (real caches live under `data/cache/`) and must never be mixed
with real data in a backtest run — see the HMM-cache warning below.

## Why this exists

IBKR's raw hourly history is short (confirmed live: ~1998 for UK names,
further back for old US names but still bounded). Real daily-close history
goes back decades further. This pipeline stretches daily history into
synthetic hourly bars so a strategy can be stress-tested against periods
(e.g. the 2008 financial crisis) that real hourly data can't reach.

## How it's built

1. **Daily closes** — real data, not fabricated. Source priority:
   [Stooq's local bulk EOD dump](../data/cache/stooq_daily/) first (fast,
   deeper UK history — e.g. HSBA back to 1992 vs IBKR's 1998), falling back
   to `IBKRDataClient.fetch_daily` for any ticker missing from the Stooq
   snapshot. Never yfinance.
2. **Daily volatility** — a rolling standard deviation of daily log returns
   (`vol.py`, default 21-day window).
3. **Intraday synthesis** — for each consecutive pair of real daily closes,
   a **Brownian bridge** (`bridge.py`) generates `bars_per_day` (default 7,
   matching the ~6.75 hourly bars/trading-day the rest of the codebase
   assumes) synthetic hourly closes. The path is random, scaled by that
   day's volatility, but is mathematically forced to land exactly on the
   next real daily close — so day-to-day price levels are always real, only
   the shape *within* a day is fabricated.
4. **Volume** — the day's real total volume is distributed across that
   day's synthetic bars proportional to each bar's `|Close - Open|`, so a
   bigger synthetic intraday move gets a bigger share. This is a proxy, not
   real intraday volume (real volume has its own shape, e.g. a U-curve
   around the open/close auctions, unrelated to price-move size). Falls
   back to a flat placeholder when no real daily volume is available.

## Known limitations

- **Survivorship bias**: this only covers tickers still resolvable today
  (via Stooq's current-listing snapshot or IBKR). Delisted/acquired
  companies are not covered — parked as a follow-up, not solved here.
- **No fat tails / jump risk**: the bridge is Gaussian: real crashes have
  fat-tailed intraday jumps this doesn't reproduce.
- **No intraday microstructure**: no open/close auction volume spike, no
  realistic bid-ask/spread pattern — just a random walk conditioned on the
  two real endpoints.
- Not stress-tested against a ticker with a recent split/rights issue.

## Regenerating

```
uv run python -m Strategy_Auto_Trader.synthetic_backtest_data.generate \
    --tickers AAPL MSFT HSBA.L --workers 4
```

`--tickers` accepts any list; `--workers N` parallelizes one ticker per
process. Output: one CSV per ticker in `hourly/`, columns
`Open,High,Low,Close,Volume`, tz-aware hourly index — same shape as
`IBKRDataClient.fetch_hourly`'s return contract.

## Using this data in a backtest

The output is deliberately **not** wired into `live_sim.py` automatically
yet. To use it, pass it via `generate_candidates`'s `df_by_ticker` override
(the same mechanism `monte_carlo_live_sim.py` already uses for synthetic
paths) — **and** either:

- `use_persistent_cache=False`, or
- `hmm_cache_dir=synthetic_backtest_data.generate.SYNTHETIC_HMM_CACHE_DIR`

The persistent HMM cache (`data/cache/hmm_cache/<ticker>.pkl`) is keyed by
ticker name alone, real or synthetic — skipping one of the two options above
would silently corrupt the real ticker's on-disk HMM state.
