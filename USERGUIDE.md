# Strategy Auto-Trader — User Guide

## What It Does

Runs a single consolidated walk-forward engine on hourly OHLCV data that combines a Gaussian Hidden Markov Model regime filter with a momentum vote system, quality gate, and a full exit set — no lookahead.

**Signal sources (weighted votes):**

| Vote | Bullish | Neutral | Bearish | Weight |
|------|---------|---------|---------|--------|
| HMM regime | P(Bull) ≥ entry_prob | between thresholds | P(Bull) ≤ exit_prob | 1.5 |
| RSI | ≥ 50 or recently crossed above 50 | 40–50, no recent cross | < 40 or recently crossed below 40 | 1.5 |
| SMA20/50 trend | above both MAs | — | below either MA | 1.0 |
| SMA200 | above | — | below | 2.0 |
| Volume | ≥ 100-bar average | — | below average (blocks entry) | 1.0 |

A **quality gate** vetoes weak BUY signals before they reach a trade event, and forces an early SELL in adverse in-trade conditions.

Weighted score ≥ buy_threshold → BUY. ≤ sell_threshold → SELL. Otherwise HOLD. A transition guard ensures BUY only fires on a non-BUY → BUY flip.

**Position sizing:** Kelly criterion from trailing realised trade P&L (capped 25%, floored 2%). Disable with `--no-kelly` for a fixed 10% allocation.

---

## Running the Skill

### Single ticker

```
cd ~/.claude/skills/Strategy_Auto_Trader
uv run python -m Strategy_Auto_Trader.markov_cli.run --ticker <SYMBOL> [options]
```

**Quick examples:**
```bash
# ASTS, default settings (hourly data, 730-day lookback)
uv run python -m Strategy_Auto_Trader.markov_cli.run --ticker ASTS

# TSLA, tighter stops, no Kelly (fixed 10% allocation)
uv run python -m Strategy_Auto_Trader.markov_cli.run --ticker TSLA --stop-loss-pct 0.03 --take-profit-pct 0.09 --no-kelly

# SPY, RSI reversal exits enabled, longer min-hold
uv run python -m Strategy_Auto_Trader.markov_cli.run --ticker SPY --exit-rsi-reversal --min-hold-bars 72
```

### Batch (multiple tickers from watchlist)

```bash
# Run all tickers in the watchlist (with email alerts if SMTP_PASSWORD is set)
uv run python -m Strategy_Auto_Trader.markov_cli.batch

# FTSE only / S&P only
uv run python -m Strategy_Auto_Trader.markov_cli.batch --watchlist config/watchlist_ftse.json
uv run python -m Strategy_Auto_Trader.markov_cli.batch --watchlist config/watchlist_sp500.json

# Send daily roundup and portfolio status emails explicitly
uv run python -m Strategy_Auto_Trader.markov_cli.batch --roundup --portfolio-status

# Skip emails completely
uv run python -m Strategy_Auto_Trader.markov_cli.batch --no-email

> Note: trade alert emails are still only sent for `BUY` and `SELL` events. `HOLD` signals do not generate direct alerts, and SELL alerts are only sent when a prior BUY has been recorded for that ticker since the reference date.
```

### Screener (scan a large universe quickly)

```bash
# Screen all S&P 500 + FTSE 100 tickers (no HMM, ~3 minutes)
uv run python -m Strategy_Auto_Trader.markov_cli.screen
```

Outputs `screen_winners.json` with tickers where the strategy is profitable or outperforms buy-and-hold.

### Two-stage batch (fast screen → full engine)

```bash
# Stage 1: lightweight backtest on every ticker (no HMM, no chart).
# Stage 2: full engine (HMM, chart, email) only for tickers that passed Stage 1.
uv run python -m Strategy_Auto_Trader.markov_cli.batch --fast-screen
```

Without `--fast-screen` the batch runs the full engine on every ticker.
With `--fast-screen`, a ticker is only promoted to Stage 2 if its lightweight backtest
is profitable or beats buy-and-hold, cutting wasted runtime on weak candidates.

---

## Exit Strategy Comparison

`compare_exits.py` is a fixed-console tool for comparing alternate exit rules across a small test universe.

```bash
uv run python -m Strategy_Auto_Trader.markov_cli.compare_exits
```

What it does:
- Uses a hard-coded `TEST_TICKERS` list in `markov_cli/compare_exits.py`.
- Tests a baseline strategy plus variants differing in exit flags (`--exit-rsi-reversal`, `--exit-macd-cross`, `--exit-consolidation`) and `max_hold_days`.
- Each variant is run through `consolidated_backtest` on hourly data with the same entry parameters.
- Prints per-ticker results and an aggregate comparison table with P&L, return, Sharpe, win rate, and exit-reason counts.

Note: there are no CLI arguments for `compare_exits.py` today; change the strategy definitions directly in the source if you want to add or modify variants.

---

## Trade Report

`trade_report.py` generates an Excel trade report from a watchlist JSON and is useful for end-of-run analysis without email alerting.

```bash
uv run python -m Strategy_Auto_Trader.markov_cli.trade_report
```

Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `--tickers` | `config/scan_tickers.json` | JSON file containing ticker lists |
| `--index` | `ftse100` | Key in the JSON file to select the ticker list |
| `--start-date` | `2026-01-12` | Only include trades entered on or after this date |
| `--max-vol` | `0.35` | Skip tickers whose annualised volatility exceeds this |
| `--buy-threshold` | `3` | Minimum composite score required to trigger BUY |
| `--lot-size` | `100` | GBP invested per trade |
| `--trade-cost` | `1` | Cost per trade event in GBP |
| `--output` | `reports/trade_report.xlsx` | Output Excel file |

Behavior:
- Fetches each ticker as hourly OHLCV via yfinance and skips symbols with insufficient history or excessive volatility.
- Runs `consolidated_backtest` with the same HMM + vote engine used by the main batch runner.
- Writes an Excel file with sheets: `Summary`, `All Trades`, `Winners`, `Losers`, `Open Positions`, and `Stats`.
- Adds ticker metadata (company name, sector) from yfinance when available.

This report is separate from the batch/email workflow and does not modify `state/trade_state.json`.

---

## All CLI Options

### Data

| Option | Default | Description |
|--------|---------|-------------|
| `--ticker` | `SPY` | Ticker symbol (anything yfinance understands; hourly data fetched for 730 days) |

### HMM + Signal Thresholds

| Option | Default | Description |
|--------|---------|-------------|
| `--entry-prob` | `0.65` | P(Bull) threshold — HMM must be at least this bullish to consider entry |
| `--exit-prob` | `0.40` | P(Bull) threshold for regime exit (after min-hold has elapsed) |
| `--buy-threshold` | `3.0` | Weighted composite score required to trigger BUY |
| `--sell-threshold` | `-3.0` | Weighted composite score that triggers SELL |
| `--volume-min-ratio` | `0.8` | Volume / 100-bar average minimum for entry |
| `--regime-smooth` | `24` | P(Bull) smoothing window in bars |
| `--min-hold-bars` | `48` | Minimum bars held before regime-exit or signal-SELL can fire |

### Capital and Sizing

| Option | Default | Description |
|--------|---------|-------------|
| `--initial-cash` | `20000` | Starting portfolio value |
| `--transaction-cost` | `10` | Cost per trade event (buy or sell) |
| `--no-kelly` | off | Use fixed 10% allocation instead of Kelly sizing |

### Trailing Stop

The trailing stop is a **profit-protection mechanism only**. It does not fire on losing trades — if the price has never risen above entry, the stop stays silent and losing exits rely on the SELL signal.

Two base stop methods — vol-scaled takes precedence if `--vol-stop-mult > 0`.

**Vol-scaled (recommended):**

| Option | Default | Description |
|--------|---------|-------------|
| `--vol-stop-mult` | `2.0` | Stop = mult × (daily_vol × √window). 0 = off |
| `--vol-stop-window` | `20` | Lookback days for realised daily vol |

The stop self-calibrates to the stock's volatility:
- ASTS (~90% annualised vol): stop ≈ 80% (with mult=2.0) or ≈20% (with mult=0.5)
- TSLA (~55% annualised vol): stop ≈ 49% (with mult=2.0) or ≈13% (with mult=0.5)
- SPY (~15% annualised vol): stop ≈ 13% (with mult=2.0)

Set `--vol-stop-mult 0` to disable.

**Fixed stop:**

| Option | Default | Description |
|--------|---------|-------------|
| `--trailing-stop` | `0.0` | Fixed fraction drop from peak, e.g. `0.20` = 20%. 0 = off. Ignored if vol-stop-mult > 0. |

The trailing stop resets its peak on each new entry. After a trailing stop exit, re-entry happens on the next genuine BUY signal (non-BUY → BUY transition).

**Profit-scaled stop tightening:**

| Option | Default | Description |
|--------|---------|-------------|
| `--profit-stop-scale` | `0.0` | For every 1% of unrealised profit, the stop distance narrows by this many percentage points. 0 = off. |
| `--min-stop` | `0.05` | Floor on the profit-adjusted stop (5%). Prevents stop collapsing to zero at extreme profit. Only active when profit-stop-scale > 0. |

A fresh trade gets a wide stop to breathe; as profit builds the stop tightens to lock in gains.

**Example** — base stop 20%, scale=0.5, floor=5%:

| Unrealised profit | Stop distance from peak |
|-------------------|------------------------|
| 0% (just entered) | 20% (full base stop) |
| 10% | 15% |
| 20% | 10% |
| 40% | 5% (floor) |
| 60%+ | 5% (floor, stays here) |

### P&L Simulation

| Option | Default | Description |
|--------|---------|-------------|
| `--initial-cash` | `20000` | Starting portfolio value |
| `--transaction-cost` | `10` | Cost per trade event (BUY or SELL) |

The simulation starts with the initial cash, deducts the transaction cost on each trade event, and compounds through daily strategy returns. The `portfolio_value` column in `compositeBacktest.csv` tracks the cash value over time.

### Entry Transition Guard

The strategy only enters on a signal **transition**: the previous day's raw signal must not have been BUY. This prevents:
- Starting the backtest already in if the first signal is BUY
- Re-entering during a brief pause inside a continuous BUY zone

The guard is always on. Trailing stop exits are treated as SELL events for guard purposes, so re-entry happens on the next BUY after the stop fires.

---

## Hidden Markov Model (HMM)

When HMM is enabled (default), the system fits a 3-state Gaussian HMM on daily returns using **10 random seeds** and keeps the best model by log-likelihood.

**Regime mapping:**
- State 0 = **Bear** (lowest mean return) → vote −1
- State 1 = **Sideways** → vote 0
- State 2 = **Bull** (highest mean return) → vote +1

**Forward filter (causal):** For the walk-forward backtest, HMM states are computed using the forward algorithm only (no backward pass). This means the state at time t uses only observations 0..t — no lookahead. The model parameters are fitted on all data (some information leakage), but the state assignment is strictly causal.

**Output:**
- Current HMM state printed in the signal summary
- `hmmStates.csv` includes both Viterbi states and forward-filter states per day
- HMM regime details (means, vols, transition matrix, stationary distribution) in the daily summary HTML

---

## Chart Panels

The output chart (`backtest_chart.png`) has four panels sharing the same time axis.

**Panel 1 — Price + moving averages + signals**
- Grey line: close price
- Blue line: SMA20 (short-term trend)
- Orange line: SMA50 (medium-term trend)
- Red line: SMA200 (long-term trend gate)
- Green background: BUY zone
- Red background: SELL zone
- Grey background: HOLD zone
- Vertical orange lines: trailing stop exits

**Panel 2 — Strategy equity vs Buy & Hold (rebased to 1.0)**
- Green line: strategy
- Blue line: buy-and-hold
- Legend shows Sharpe ratio and total return for both

**Panel 3 — In market**
- Shows £20,000 when in a position, £0 when flat
- "Currently: IN / OUT" annotation at right

**Panel 4 — Composite score bars**
- Green bars: net positive score (more bullish than bearish votes)
- Red bars: net negative score
- Score must reach ±threshold to flip flag; colours show raw score, not flag

---

## Output Files

Each run creates a timestamped directory: `data/<TICKER>_<UTC-timestamp>/`

| File | Contents |
|------|----------|
| `inputData.csv` | Raw OHLCV data from yfinance |
| `bullBearSideways.csv` | Close, rolling return, and Bull/Bear/Sideways label per day |
| `transitionMatrix.csv` | 3×3 Markov transition matrix (rows = from, cols = to) |
| `stationaryDistribution.csv` | Long-run steady-state regime probabilities |
| `walkForwardBacktest.csv` | Per-day signal, position, trade_event, and equity from the basic Markov-only backtest |
| `momentumIndicators.csv` | Full time series of RSI, SMAs, and crossover flags |
| `compositeBacktest.csv` | Per-day close, signal, position, trade_event, returns, equity, sell_reason, effective_stop, portfolio_value |
| `backtest_chart.png` | Four-panel chart |
| `daily_summary.html` | Full HTML report with all analysis and embedded chart |
| `hmmStates.csv` | HMM hidden state per day — both Viterbi and forward-filter (only when HMM runs) |

**Key columns in `compositeBacktest.csv`:**

| Column | Description |
|--------|-------------|
| `close` | Share price on that date |
| `trade_event` | `BUY` on entry date, `SELL` on exit date, blank otherwise |
| `sell_reason` | `signal`, `trailing_stop(X% vs stop Y%)`, `ignored(flat)`, `ignored(entry-only)` |
| `effective_stop` | The trailing stop distance for that day (after profit-scaling if active) |
| `portfolio_value` | Simulated portfolio value (£20k initial, £10/trade costs) |

---

## Daily Summary HTML

Each run generates `daily_summary.html` — a self-contained dark-themed HTML report including:

- Signal banner (BUY / HOLD / SELL) with composite score and IN/OUT status
- Signal votes table — each indicator's bull/bear/neutral contribution
- Momentum indicators — RSI with label, price vs SMA20/50/200
- Trailing stop — current effective stop % with vol/profit-scale detail
- P&L simulation — trades count, costs, strategy vs buy-and-hold final portfolio
- Backtest summary — Sharpe, total return, max drawdown with date range
- Backtest chart (embedded)
- Observable Markov regime — current state, transition matrix, stationary distribution
- Hidden Markov Model — current HMM state, regime means/vols, HMM transition matrix
- Recent trades — last 10 BUY/SELL events with price, score, exit reason, portfolio value

---

## Batch Runner & Watchlists

### Watchlist format

`config/watchlist.json` (or `config/watchlist_ftse.json`, `config/watchlist_sp500.json`):

```json
{
  "defaults": {
    "years": 2,
    "position_mode": "state",
    "vol_stop_mult": 0.5,
    "profit_stop_scale": 0.5,
    "min_stop": 0.05,
    "reference_date": "2026-06-19",
    ...
  },
  "tickers": [
    {"ticker": "ASTS"},
    {"ticker": "TSLA", "in_sell_threshold": -2},
    {"ticker": "BP.L"}
  ]
}
```

Per-ticker overrides merge onto the defaults. Any CLI option can be set in either section.

The `reference_date` is used by the email system — SELL alerts are only sent for tickers that have had a BUY alert since this date.

### Scheduling

Two Windows Task Scheduler tasks run the batch automatically:

| Task | Runs at (UK) | Watchlist | Why this time |
|------|-------------|-----------|---------------|
| `StrategyAutoTrader-FTSE` | 5:00 PM | `config/watchlist_ftse.json` | After LSE close (4:30 PM) |
| `StrategyAutoTrader-SP500` | 9:30 PM | `config/watchlist_sp500.json` | After NYSE close (9:00 PM UK) |

Logs are written to `logs/ftse_YYYYMMDD.log` and `logs/sp500_YYYYMMDD.log`.

Manage via Task Scheduler (`taskschd.msc`) — look for `StrategyAutoTrader-FTSE` and `StrategyAutoTrader-SP500`.

---

## Email Alerts

Requires `SMTP_USER` (your email address) and `SMTP_PASSWORD` (an app password) environment variables. Defaults to Yahoo Mail's SMTP server; override with `SMTP_HOST` / `SMTP_PORT` for other providers. Alerts are sent to `SMTP_TO` if set, otherwise to `SMTP_USER`.

**Setup (Yahoo Mail):**
1. Enable 2-Step Verification on your Yahoo account
2. Generate an App Password at https://login.yahoo.com/account/security/app-passwords
3. Set permanently:
   ```powershell
   [System.Environment]::SetEnvironmentVariable("SMTP_USER", "you@yahoo.com", "User")
   [System.Environment]::SetEnvironmentVariable("SMTP_PASSWORD", "your-app-password", "User")
   ```

**What gets emailed:**

| Email | When | Content |
|-------|------|---------|
| **Trade alert** | BUY event + strategy profitable or outperforming B&H | Full daily summary HTML with embedded chart |
| **Trade alert** | SELL event + prior BUY was sent since reference date | Full daily summary HTML with embedded chart |
| **Daily roundup** | End of every batch run | Summary table of all tickers, signal counts, P&L, trade events, failures |

SELL alerts are **only sent for stocks the system previously told you to BUY**. This is tracked in `state/trade_state.json` — a BUY alert records the ticker; a SELL alert removes it. Reset the reference date by editing `config/watchlist.json` and clearing `state/trade_state.json`.

---

## IBKR Paper Trading (Execution Engine)

The execution engine (`markov_cli/execute.py`) reads the latest signal for each ticker and submits orders to Interactive Brokers. It runs **independently** of the batch runner — signals first, then execute separately so each can be tested and restarted on its own.

### Prerequisites

1. **Install ib_insync** (not installed by default):
   ```bash
   uv add ib_insync
   ```

2. **Open TWS** (Trader Workstation) and log in to your **paper trading account**.  
   Paper and live accounts are on separate logins — make sure you use the paper one.

3. **Enable the API in TWS:**  
   `Edit → Global Configuration → API → Settings`
   - Check **Enable ActiveX and Socket Clients**
   - Set **Socket port** to `7497` (paper default)
   - Add `127.0.0.1` to **Trusted IP Addresses**

4. **Configure capital in `config/watchlist.json`** (under `"defaults"`):
   ```json
   "capital_pot": 20000,
   "max_positions": 5
   ```
   `capital_pot` is the total pot across all positions. Each position slot gets `capital_pot / max_positions`, scaled by the Kelly fraction.

### Running

```bash
# Safe dry run — NullBroker, no real orders, no state changes
uv run python -m Strategy_Auto_Trader.markov_cli.execute --dry-run

# Paper account — TWS must be open on localhost:7497
uv run python -m Strategy_Auto_Trader.markov_cli.execute

# Custom watchlist or data directory
uv run python -m Strategy_Auto_Trader.markov_cli.execute --watchlist config/watchlist_sp500.json
```

Always run `--dry-run` first to confirm signals look right before submitting real (paper) orders.

### How it works

For each ticker in the watchlist, the engine reads the latest `qualityGate.json` and `compositeBacktest.csv` from `data/<TICKER>_*/`. If the run is older than 24 hours it is treated as stale and skipped.

| Signal | Action |
|--------|--------|
| `BUY` | Check capacity (not already in, open count < `max_positions`); compute quantity = `(capital_pot / max_positions) × kelly_fraction / price`; submit market BUY |
| `SELL` | If an open position exists, submit market SELL for the held quantity |
| `HOLD` or stale | Skip |

### Position sizing example

`capital_pot = £20,000`, `max_positions = 5`, `kelly_fraction = 0.15`, `price = £200`

```
slot_value = 20000 / 5 = £4,000
quantity   = floor(4000 × 0.15 / 200) = floor(3.0) = 3 shares
```

### State

- **`state/execution_state.json`** — open positions with fill price, quantity, Kelly fraction, stop/target levels, and a full trade log with realised P&L. Managed exclusively by the execution engine.
- **`state/trade_state.json`** — BUY alert dates for the email system. The execution engine does **not** touch this file.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dry-run` | off | Use NullBroker (no real orders, no state writes) |
| `--watchlist` | `config/watchlist.json` | Watchlist JSON to read tickers from |
| `--data-dir` | `data/` | Directory containing per-ticker run subdirectories |
| `--state-dir` | `state/` | Directory for `execution_state.json` |

---

## Strategy Guidance

**For momentum stocks (ASTS, high-vol)**
- Use `--position-mode state` — stays in through HOLD, exits only on real signal or stop
- Default `--in-sell-threshold -1` exits on any net-negative score — responsive but may exit early
- For ASTS specifically, `--in-sell-threshold -2` may reduce whipsaw
- Use `--vol-stop-mult 0.5` (≈20% stop for ASTS) with `--profit-stop-scale 0.5` to tighten as profit grows

**For lower-vol stocks (SPY, large-caps)**
- Vol-scaled stop at 0.5× is meaningful (~7% for SPY)
- `--in-sell-threshold -2` or `-3` suits calmer price action

**For entry-only mode**
- Disable SELL signals entirely with `--position-mode entry-only`
- Must pair with a trailing stop: `--vol-stop-mult 0.5` or `--trailing-stop 0.20`
- Good for stocks where signals are noisy but trend is strong

**On backtests vs live trading**
- All signals use only data available on the signal day (no lookahead)
- Position is applied to the *next* day's return (realistic 1-day execution lag)
- The strategy typically underperforms buy-and-hold on momentum/catalyst stocks because any exit risks missing explosive moves
- Use the backtest to understand *when* you'd be in and out, not to predict future returns

---

## Reading the Current State

At the end of every run the script prints the **current day's signal**:

```
Composite signal  (score +3 / 6  |  sell threshold=-3):
  Markov (Bull state, signal=+0.45): +1 (bull)
  RSI:    +1 (bull)
  SMA20:  +1 (bull)
  SMA50:  +1 (bull)
  SMA200: +1 (bull)
  HMM (Sideways):  0 (neutral)

  >>> BUY  (position: +1.00) <<<

  Vol-scaled trailing stop (today): 20.1%  [0.5 × 9.00% daily vol × sqrt(20)]
```

This is what the model would signal *today* if you were running it live.

---

## Consolidated Engine Architecture

The system runs a single engine that combines the HMM regime model with the momentum vote system on hourly data. All production workflows — batch runner, screener, trade report, exit comparison — use `consolidated_backtest` from `quant_hmm/consolidated_engine.py`.

| Component | Description |
|-----------|-------------|
| Regime filter | 3-state Gaussian HMM on hourly log returns; P(Bull) smoothed over `--regime-smooth` bars |
| Vote system | RSI, SMA20/50 trend, SMA200 gate, volume ratio; weights tuned for hourly bars |
| Quality gate | Vetoes weak BUY contexts; forces early SELL in adverse in-trade conditions |
| Exit set (priority order) | Hard stop-loss → take-profit → trailing/vol-stop → max-hold → SAR → MACD/RSI/consolidation exits → composite-signal SELL (after min-hold) |
| Position sizing | Kelly fraction from trailing trade P&L; capped at 25%, floored at 2% |
| Email / state | Yes — `state/trade_state.json`, SMTP alerts from the batch runner |

### Workflow

```
markov_cli.batch  →  consolidated_backtest (each ticker)  →  qualityGate.json  →  email alerts
                                                          →  compositeBacktest.csv
                                                                    ↓
                                              markov_cli.execute  →  IBKR paper/live orders
                                                                  →  execution_state.json
```

Run `batch` first (signals), then `execute` separately (orders). The two processes share no locks — `execute` reads completed output files.

The `quant_hmm.quant_run` CLI and `quant_trade_report` still exist for on-demand single-ticker analysis and Excel trade reporting — they call the same consolidated engine.

---

## Quant HMM Engine (Single-Ticker CLI)

`quant_hmm/quant_run.py` provides a single-ticker CLI for detailed analysis and the Excel trade report (`quant_trade_report`). Both call the consolidated engine and produce the same signals as the batch runner. Use these when you want more detail than the batch runner outputs, or when running outside the daily watchlist cadence.

### How it decides entries and exits

The HMM is fit on hourly log returns (3-state Gaussian, expanding window, refit every `hmm_refit_bars` bars) and produces a forward-filtered (causal, no lookahead) `P(Bull)` at every bar. Because raw hourly probabilities are noisy, a rolling mean (`--regime-smooth` bars) is used for exit decisions.

- **Entry**: `P(Bull)_smoothed > entry_prob` **and** `volume_ratio >= volume_min_ratio` (current / 100-bar average volume)
- **Exit**, checked in this order, every bar:
  1. **Stop-loss** — `close <= entry_price × (1 - stop_loss_pct)` — fires immediately, no hold restriction
  2. **Take-profit** — `close >= entry_price × (1 + take_profit_pct)` — fires immediately
  3. **Regime exit** — `P(Bull)_smoothed < exit_prob`, but only once `--min-hold` bars have elapsed since entry (prevents being shaken out by the first noisy dip)

### Single ticker

```bash
uv run python -m Strategy_Auto_Trader.quant_hmm.quant_run --ticker <SYMBOL> [options]
```

```bash
# HSBA.L with default settings
uv run python -m Strategy_Auto_Trader.quant_hmm.quant_run --ticker HSBA.L

# Tighter entry, no Kelly sizing, no sentiment
uv run python -m Strategy_Auto_Trader.quant_hmm.quant_run --ticker CSCO --entry-prob 0.70 --no-kelly --no-sentiment
```

### Trade report (multiple tickers from watchlist, Excel output)

```bash
uv run python -m Strategy_Auto_Trader.quant_hmm.quant_trade_report [options]
```

```bash
# FTSE watchlist (default), trades from 2026-01-12 onward
uv run python -m Strategy_Auto_Trader.quant_hmm.quant_trade_report

# S&P watchlist, skip the volatility pre-screen
uv run python -m Strategy_Auto_Trader.quant_hmm.quant_trade_report --watchlist config/watchlist_sp500.json --no-vol-screen
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--ticker` | `SPY` | Ticker symbol (`quant_run` only) |
| `--watchlist` | `config/watchlist_ftse.json` | Watchlist JSON (`quant_trade_report` only) |
| `--period` | `730d` | yfinance period for hourly data (yfinance limits hourly history to ~2 years) |
| `--start-date` | `2026-01-12` | Only include trades entered on/after this date (`quant_trade_report` only) |
| `--entry-prob` | `0.65` | P(Bull) threshold to enter |
| `--exit-prob` | `0.40` | P(Bull) threshold to exit (only checked after `--min-hold`) |
| `--stop-loss` | `0.05` | Hard stop-loss from entry (5%), fires immediately |
| `--take-profit` | `0.15` | Take-profit from entry (15%), fires immediately |
| `--volume-min` | `1.0` | Minimum volume ratio (current / 100-bar avg) required to enter |
| `--regime-smooth` | `24` | Rolling window in bars for smoothing P(Bull) on exit decisions (24 bars ≈ 1 trading day) |
| `--min-hold` | `48` | Minimum bars before a regime exit can fire (48 bars ≈ 2 trading days) |
| `--no-kelly` | off | Disable Kelly sizing — use a flat 10% position instead |
| `--no-sentiment` | off | Disable sentiment/VIX threshold adjustment |
| `--no-vol-screen` | off | Disable pre-screening out choppy/mean-reverting tickers (`quant_trade_report` only) |
| `--min-trend-quality` | `0.0` | Minimum trend-quality score to keep a ticker when vol-screening (`quant_trade_report` only) |
| `--initial-cash` / `--lot-size` | `20000` / `100` | Starting portfolio value (`quant_run`) / GBP invested per trade (`quant_trade_report`) |
| `--trade-cost` | `10` / `1` | Cost per trade event (defaults differ between the two tools) |
| `--output` | `reports/quant_trade_report.xlsx` | Output Excel file (`quant_trade_report` only) |

### Kelly position sizing

When `--no-kelly` is not set, position size is the Kelly fraction computed from the last 20 completed trades' win rate and average win/loss, capped at 25% of capital and floored at 2% (never zero, so the strategy keeps probing). Starts at a conservative 10% before 20 trades have accumulated.

### Sentiment & VIX adjustment

Unless `--no-sentiment` is set, `sentiment.py` combines options put/call ratio, IV rank, options skew, insider buying/selling (90d), and short interest into a single `sentiment_score` and `sentiment_label`. This adjusts the **entry threshold**: bullish sentiment lowers it, bearish raises it (capped at ±0.10). Separately, the VIX term structure (`vix_regime()`) sets `vix_signal`: in a high-vol regime it tightens the exit threshold (+0.05, capped at 0.50) and widens the stop-loss (+0.02, capped at 0.08) to avoid getting stopped out by noise.

### Volatility-character pre-screen

`quant_trade_report` (not `quant_run`) screens the watchlist through `vol_screen.py` before running backtests, unless `--no-vol-screen` is set. It computes the Kaufman Efficiency Ratio, return autocorrelation, and a Choppiness Index per ticker into a composite `trend_quality` score, and drops tickers below `--min-trend-quality`. This engine's edge comes from trending price action — choppy/mean-reverting tickers are excluded before spending time on the (expensive) HMM backtest.

### Output

- `quant_run`: prints the report to console and saves `data/<TICKER>_<UTC-timestamp>/quant_backtest.csv`.
- `quant_trade_report`: writes `reports/quant_trade_report.xlsx` with sheets — `Summary` (per-ticker P&L/Sharpe), `All Trades`, `Winners`, `Losers`, `Open Positions`, `Sentiment` (per-ticker sentiment detail), `Vol Screen` (kept/excluded tickers and their scores), `Exit Breakdown` (win rate by stop-loss/take-profit/regime-exit), `Stats` (run-level summary).

---

## Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run a specific test module pattern
uv run pytest tests/ -v -k "test_regime"

# Run with short output
uv run pytest tests/
```

Tests cover: exits module (_effective_stop_for_bar, _check_exit_conditions with all 7 exit types), RSI/SMA correctness, composite signal thresholds and weights, quality gate (weak-buy veto, adverse-exit), HMM forward filter and discretisation, consolidated engine (entry/exit/sizing logic), trade state persistence, email chart embedding, batch config mapping, sentiment/VIX adjustment, volatility pre-screen, momentum-to-signal integration, and broker layer (NullBroker protocol conformance, PortfolioManager sizing/capacity/state, signal_reader staleness guard, execute.py BUY/SELL/dry-run/capacity scenarios).

---

## Project Structure

```
Strategy_Auto_Trader/
  Strategy_Auto_Trader/
    core/                    # Shared signal/exit logic
      momentum.py                 # RSI, SMA, composite signal, Parabolic SAR
      quality_gate.py             # Veto layer (weak-buy veto, adverse-exit)
      exits.py                    # _effective_stop_for_bar, _check_exit_conditions
    broker/                   # IBKR execution layer
      types.py                      # OrderRequest, FillResult, PositionRecord dataclasses
      protocols.py                  # BrokerAdapterProtocol (runtime-checkable)
      ibkr_adapter.py               # IBKRAdapter via ib_insync
      null_adapter.py               # NullBroker for tests and --dry-run
      portfolio.py                  # PortfolioManager (capital pot, sizing, execution_state I/O)
      signal_reader.py              # Reads latest qualityGate.json + compositeBacktest.csv per ticker
    markov_cli/              # CLI tools (all use consolidated_engine)
      run.py                       # Entry point (single ticker)
      batch.py                      # Batch runner (reads watchlist JSON)
      execute.py                    # IBKR execution engine (reads signals, places orders)
      screen.py                     # Fast universe screener
      compare_exits.py              # Compare exit-strategy variants
      trade_report.py               # Trade-by-trade Excel report
    quant_hmm/                # HMM engine + consolidated backtest
      quant_engine.py               # HMM fit/inference, helper functions
      consolidated_engine.py        # Walk-forward consolidated backtest
      quant_run.py                  # CLI entry point (single ticker, detailed output)
      quant_trade_report.py         # Trade-by-trade Excel report
      sentiment.py                   # Options/VIX/insider/short-interest signals
      vol_screen.py                  # Volatility-character pre-screen
    extensions/
      hmm_extension.py              # Hidden Markov Model (multi-seed + forward filter)
    output/                   # Report/notification generation
      report.py                     # Daily summary HTML report
      charting.py                   # Four-panel backtest chart
      emailer.py                    # SMTP email alerts (trade + roundup)
      trade_state.py                # Tracks BUY alerts for SELL filtering
  tests/
    test_all.py              # Unit tests (pytest)
  config/                   # User-curated input lists
    watchlist.json               # Combined watchlist (all tickers)
    watchlist_ftse.json          # FTSE 100 tickers only
    watchlist_sp500.json         # S&P 500 tickers only
    scan_tickers.json            # Universe for screener
  state/                    # Persistent runtime state
    trade_state.json             # BUY alert dates (email system)
    execution_state.json         # Open positions + trade log (execution engine)
  reports/                  # Generated Excel/JSON reports (re-creatable, not source)
  data/                     # Per-run output directories (re-creatable)
  logs/                     # Batch run logs
  run_ftse.bat              # Scheduled task script (5 PM UK)
  run_sp500.bat             # Scheduled task script (9:30 PM UK)
  pyproject.toml            # Dependencies
  SKILL.md                  # Claude Code skill definition
  USERGUIDE.md              # This file
```
