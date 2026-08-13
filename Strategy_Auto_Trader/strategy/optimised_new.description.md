# optimised_new — entry/exit reference

Ratchet-exit variant of `optimised`: identical composite entry signal, but
winners are closed solely by a profit-narrowing trailing stop instead of a
fixed take-profit ceiling. Configured live-daemon strategy for FTSE100 +
S&P500 (`config/overnight_strategy.json`). Single-ticker validated only —
not universe-tested.

Source-grounded, read directly from the working tree, 2026-08-12.

## 01 — Entry signal

Weighted vote across six indicators, computed fresh every hourly bar by
`core/momentum.py::composite_signal()`. Each indicator casts `+1 / 0 / -1`;
the vote is multiplied by its weight; the weighted sum is compared to a buy
and a sell threshold. Weights are `optimised_new`'s own class defaults —
every one overridable via its matching CLI flag.

| Indicator | Vote rule | Weight | Max contribution |
|---|---|---|---|
| `markov` | >0.20 → +1, <-0.20 → -1, else 0 — but input is hardcoded `0.0` in `optimised_new`, so the vote is always 0 | 0.0 | 0 |
| `rsi` | RSI ≥ 50 *or* crossed above 50 in the last 4 bars → +1; RSI < 40 *or* crossed below 40 in the last 4 bars → -1; else 0 | 1.0 | ±1 |
| `trend` | merged SMA20+SMA50: price above both → +1, below both → -1, mixed → 0 | 2.0 | ±2 |
| `sma200` | price above 200-bar SMA → +1, below → -1 (vote omitted entirely, from both score and max-score, until 200 bars of history exist) | 3.0 | ±3 |
| `volume` | volume ÷ 20-bar average > 1.2 → +1, < 0.7 → -1, else 0 | 1.0 | ±1 |
| `hmm` | discretized regime vote: Bear → -1, Sideways → 0, Bull → +1 (see §07) | 2.0 | ±2 |

`score = Σ(vote × weight)`. With `markov` structurally inert, max possible
score is **±9** (rsi 1 + trend 2 + sma200 3 + volume 1 + hmm 2).

```
-9 ────────────── -4.5 ──────────────── 6.0 ────────────── +9
        SELL                HOLD                  BUY
```

Default weights are all whole numbers, so `score` always lands on an
integer — the -4.5 sell threshold has the same practical effect as ≤-5.

## 02 — Entry vetoes

Checked only when `currently_in=False` and the gated decision is already
BUY. Either condition downgrades the decision to HOLD — these run *after*
the quality gate (§03), so they apply regardless of whether the gate is on.

| Condition | Logged reason string |
|---|---|
| RSI > 70 (overbought) | `optimised_new veto: RSI > 70 (overbought entries lose)` |
| `regime_signal` ≤ 0 (no bull-regime confirmation) | `optimised_new veto: regime_signal <= 0 (no bull-regime confirmation)` |

## 03 — Quality gate (off by default)

`quality_gate_enabled = False` for `optimised_new` — the one deliberate
divergence from `optimised` (which defaults it on). Investigation found the
gate's adverse-exit escalation fired on 100% of exits in testing, so the
ratchet trailing stop this strategy exists to test never got a chance to
bind. Still overridable back on via `--plugin-gate quality`.

**If re-enabled — weak-buy veto** (`currently_in=False`): ≥2 of 5
conditions (`gate_sensitivity=2`) downgrades a BUY to HOLD:
- `regime_signal` < 0.25
- not (above SMA20 *and* above SMA50)
- volume ratio < 1.0
- below SMA200
- RSI < 50 and no recent cross above 50

**If re-enabled — adverse-exit escalation** (`currently_in=True`): ≥2 of 5
conditions forces an immediate SELL, bypassing `min_hold_bars` entirely:
- `regime_signal` < -0.20
- RSI < 40 or recent cross below 40
- not (above SMA20 *and* above SMA50)
- volume ratio < 0.8
- below SMA200

## 04 — Admission gates (engine-level, outside the weighted score)

Four checks the composite score never sees. Any one of these can block a
bar that would otherwise score a clean BUY.

**Volatility-character pre-screen.** Computed once per ticker (not per
bar) via `quant_hmm/vol_screen.py::volatility_profile()` from 2 years of
daily bars, passed into the strategy as `vol_filter_ok`. When false,
`evaluate()` returns permanent HOLD — `vol_filter: unsuitable
(choppy/mean-reverting)` — every bar, for the life of the ticker.
`optimised_new` additionally sets `skip_overnight_vol_screen=True` to skip
the nightly pipeline's stage-1 copy of this check (see §08) — the per-bar
gate here is the authoritative enforcement; stage-1 is redundant when
top_k's hybrid score already weights trend_quality at 0.7.

```
trend_quality = 1.5·(efficiency_ratio-0.07)/0.05 + 1.5·(autocorr)/0.04
                - 1.0·(ann_vol-0.25)/0.05 - 1.0·(sign_change_freq-0.52)/0.03
```

Gate: `trend_quality ≥ min_trend_quality` (config default **0.0**).
Efficiency ratio (Kaufman ER) and autocorrelation reward trending/momentum
tape; annualised vol and daily sign-flip frequency penalise choppy/
mean-reverting tape — weights set from empirical correlation with realised
P&L (see module docstring).

**Hard volume floor.** `volume_min_ratio = 1.0` (strategy-owned class
attribute on `OptimisedNewEntry`). A BUY is only even evaluated when `volume ÷
20-bar average ≥ 1.0` — stricter than the composite signal's own volume
vote (which only turns bearish below 0.7). Below the floor, the bar can't
produce a BUY regardless of score.

**Flip-entry transition guard.** `require_flip_entry = True` (default, not
overridden). A BUY is only admitted on a HOLD/SELL → BUY transition, not on
consecutive BUY bars — prevents re-entering the same signal bar after bar
while it stays elevated.

**Minimum hold.** `min_hold_bars = 48` hourly bars (~2 trading days —
strategy-owned class attribute on `OptimisedNewExit`). Gates
*signal-based* exits only — a plain composite SELL or a quality-gate
adverse SELL must wait for `bars_held ≥ 48`. The hard stop-loss and
trailing stop in §05 are exempt and can fire on any bar.

## 05 — Exit mechanics (priority order, every bar while in a position)

This is the whole point of the `_new` variant: no hard take-profit
ceiling. Winners are released only by a trailing stop that tightens as
profit grows.

| Param | Value |
|---|---|
| `stop_loss_pct` | 0.08 |
| `take_profit_pct` | 999 (disabled) |
| `vol_stop_mult` | 2.0 |
| `vol_stop_window` | 20 |
| `profit_stop_scale` | 0.30 |
| `min_stop_pct` | 0.03 |
| `max_hold_days` | 0 (disabled) |

**Trailing-distance formula:**

```
effective_stop = vol_stop_mult × realised_vol × √vol_stop_window
  (falls back to a flat 0.0 — i.e. no trailing stop at all — if vol
  isn't yet available)

if unrealised gain > 0:
  effective_stop = max(min_stop_pct, effective_stop - unrealised_pct × profit_stop_scale)
```

The further into profit, the tighter the trail — down to a 3% floor it can
never narrow past. A `trailing_stop` fixed-distance override exists in the
plumbing but is a permanent no-op here: the vol-scaled branch always wins
whenever `vol_stop_mult > 0`.

**Priority order (first hit wins):**
1. Hard stop-loss — `close ≤ entry × (1 - 0.08)`
2. Hard take-profit — effectively never (`target = entry × 1000`)
3. Trailing/vol-scaled stop — only once the trade has been profitable;
   drop-from-peak ≥ `effective_stop`
4. Max-hold-bars — disabled (`max_hold_days=0`)
5. Parabolic SAR stop — disabled
6. MACD cross / RSI reversal / consolidation exits — all disabled
7. Quality-gate adverse SELL — n/a while the gate is off (§03)
8. Plain composite SELL (score ≤ -4.5) — gated by `min_hold_bars ≥ 48`;
   logged reason is the literal string `signal` since the gate's reason
   field is blank while disabled

## 06 — Position sizing (Kelly)

`use_kelly=True`, `kelly_lookback=20`. Sizing is stateful across a
ticker's trade history, not a per-bar input.

- Before 20 closed trades exist: flat **10%** of the pot (sizer's warm-up
  default).
- From the 20th closed trade on, recomputed from the trailing 20 trades:

```
b      = |avg_win ÷ avg_loss|
kelly  = (win_rate × b - (1-win_rate)) ÷ b
fraction = clip(kelly, 0.0, 0.25)  →  floored at 2% as the live position fraction
```

**Kelly ≤ 0 rejects the trade outright** downstream, live and in backtest
alike — no flat-fraction fallback. A candidate with no edge in its
trailing 20 trades simply isn't sized (see §09).

## 07 — Regime model

A 3-state Gaussian HMM (`plugins/hmm_regime.py`) fit on an expanding
window of log returns, stepped forward incrementally one bar at a time.

| Param | Value |
|---|---|
| `min_train_bars` | 500 |
| `refit_bars` | 500 |
| `regime_smooth` | 24 bars |
| `bull_edge` | 0.65 |
| `bear_edge` | 0.40 |

`p_bull_smooth` is the trailing mean of `p_bull` over the last 24 hourly
bars (~1 trading day). `hmm_vote`: `p_bull_smooth ≥ 0.65` → Bull (+1 vote),
`≤ 0.40` → Bear (-1 vote), between → Sideways (0). `regime_signal =
p_bull_smooth - p_bear` — the value both entry vetoes and the (disabled)
quality gate read.

A sentiment/VIX **context adjuster** can nudge `bull_edge`/`bear_edge`/stop
width before the HMM runs, but its two inputs (`sentiment_score`,
`vix_signal`) are both hardcoded to neutral (`0.0` / `0`) on the live
path — the options-derived sentiment screen was removed from the daemon
2026-08-11. The plumbing is live; the signal feeding it is not.

## 08 — Overnight pipeline (02:00 Europe/London, before markets open)

Ticker scope is decided once a night, outside the strategy entirely — a
ticker excluded here never reaches `optimised_new`'s `evaluate()` at all
that trading day. Runs per `markov_cli/overnight_scope.py` against
`config/overnight_strategy.json`.

**Stage 0 — cross-market pre-screen.** Vol-screens both watchlists
combined (trend_quality ≥ 0.0) to build the candidate pool for stage 2, so
ranking slots aren't spent on tickers stage 1 would veto anyway.

**Stage 1 — per-market volatility screen.** Same `trend_quality` gate as
§04, run per market (FTSE, S&P500). Open positions are exempt from
exclusion (`exempt_if_open_position=true`) even if they'd now fail the
screen. `optimised_new` opts out entirely (`skip_overnight_vol_screen=True`
on `OptimisedNewEntry`, read by `registry.wants_vol_screen_disabled()`) — the
per-bar gate in §04 is the authoritative check; stage-1 is redundant given
top_k's trend_quality weighting. Stage-2 (top-K) still applies.

**Stage 2 — global top-K ranking.** A standalone subprocess
(`rank_universe_cli.py`, 5h timeout, isolated from the daemon's IBKR
connection) runs a full-history `optimised_new` backtest per surviving
ticker and scores it:

```
score = 0.7 × trend_quality[0,1] + 0.3 × win_rate_60d
```

Win-rate is that ticker's own share of profitable trades in its trailing
60 days of backtested candidates (0.5 if none). Keeps the top **k=70** per
market, plus every open position unconditionally. On subprocess failure,
timeout, or malformed output: falls back to the previous night's saved
top-K set; if none exists, degrades to vol-screen-only scope. A market is
never left with zero tickers.

Output: `state/in_scope_<market>.json` (kept / excluded+reason / open
positions / orphaned). A position whose ticker has dropped out of the
watchlist file entirely is force-kept and logged as **orphaned**, so a
live exit is never silently stranded.

## 09 — Order placement (every daytime poll cycle)

`markov_cli/execute.py::execute_signals()` — reads each ticker's
precomputed signal and arbitrates against shared cash. Nothing here
touches the composite score again; it only decides *whether* and *how
much* of an already-decided BUY gets filled.

1. Read latest signal per ticker; HOLD is skipped immediately.
2. BUY signals sorted by `entry_score` descending — same ordering as
   `live_sim.py`'s `arbitrate()`, so highest-conviction signals claim cash
   first.
3. `allow_new_entries` — false during a reconciliation halt or user pause
   blocks every BUY this cycle.
4. `portfolio.can_open(ticker)` — an existing open position on that
   ticker blocks re-entry. One position per ticker; **not** a cap on total
   open positions.
5. `qty = floor(available_cash × kelly_fraction ÷ price)`, minimum 1
   share. Zero if `kelly_fraction ≤ 0`, price ≤ 0, or cash < one share's
   price — skipped as `(qty=0)`, never rounded up.
6. Order placed; stop/target recomputed off the actual fill price, not
   the signal's close.
7. Severe-slippage check — if the fill already breached the original
   signal-time stop, records an immediate same-bar stop-out instead of
   holding a broken position.
8. Optional protective stop (`protective_stops`, off by default) — a
   resting GTC stop at `fill × (1 - stop_buffer_pct)`, buffer default
   **1.5%**.

SELL side exits the full recorded position size (no partial exits).
`trades_today` is counted for monitoring/`app_status` but enforces no
daily cap — the old daily buy/sell limit was removed 2026-08-11.

## 10 — Live config snapshot

From `config/overnight_strategy.json` — reflects what's configured to
run, not a guarantee of the daemon's process state at any given moment.

| Key | Value |
|---|---|
| Markets | ftse, sp500 |
| Default strategy | optimised_new (both) |
| Capital pot | 100,000 |
| Dry run | false |
| Overnight run | 02:00 Europe/London |
| Reconciliation | 21:30 |
| Poll interval | 60s |
| Broker port | 4002 |
| Top-K | k=70 |

## Sources

```
Strategy_Auto_Trader/strategy/optimised_new.py
Strategy_Auto_Trader/core/momentum.py
Strategy_Auto_Trader/core/quality_gate.py
Strategy_Auto_Trader/core/exits.py
Strategy_Auto_Trader/plugins/exit_rules.py
Strategy_Auto_Trader/plugins/kelly_sizer.py
Strategy_Auto_Trader/plugins/hmm_regime.py
Strategy_Auto_Trader/plugins/types.py
Strategy_Auto_Trader/quant_hmm/quant_engine.py
Strategy_Auto_Trader/quant_hmm/consolidated_engine.py
Strategy_Auto_Trader/quant_hmm/vol_screen.py
Strategy_Auto_Trader/quant_hmm/ticker_ranking.py
Strategy_Auto_Trader/markov_cli/overnight_scope.py
Strategy_Auto_Trader/markov_cli/rank_universe_cli.py
Strategy_Auto_Trader/markov_cli/execute.py
Strategy_Auto_Trader/broker/portfolio.py
Strategy_Auto_Trader/strategy/base/registry.py
config/overnight_strategy.json
```
