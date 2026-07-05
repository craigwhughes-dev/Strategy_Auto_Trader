# Strategy Auto-Trader

An hourly-bar algorithmic trading system built as a [Claude Code skill](https://code.claude.com/docs/en/skills). It combines a Gaussian Hidden Markov Model regime filter with a weighted momentum vote system, a quality gate, and a layered exit stack — then backtests, live-simulates, emails trade alerts, and (optionally) executes on Interactive Brokers paper trading.

> ⚠️ **Not financial advice.** This is a research and paper-trading project. Backtests use walk-forward, no-lookahead signals, but past performance predicts nothing. Use the backtests to understand *when* the system would be in or out of the market, not to forecast returns.

## What it does

- **Regime detection** — 3-state Gaussian HMM on hourly log returns, forward-filtered (strictly causal, no lookahead), with `P(Bull)` smoothing.
- **Composite entry signal** — weighted votes from the HMM regime, RSI, SMA20/50 trend, SMA200 gate, and volume ratio, with a quality gate that vetoes weak BUYs.
- **Layered exits** — hard stop-loss → take-profit → vol-scaled/trailing stop (with profit-scaled tightening) → max-hold → Parabolic SAR → MACD/RSI/consolidation exits → composite SELL.
- **Pluggable strategies** — `default`, `conservative`, `trend`, and `optimised` entry/exit pairs, selected via `--strategy` and registered in [registry.py](Strategy_Auto_Trader/strategy/base/registry.py).
- **Kelly position sizing** from trailing realised trade P&L (capped 25%, floored 2%).
- **Email alerts** — BUY/SELL trade alerts with an embedded chart and a daily roundup, via SMTP.
- **IBKR execution** — a separate execution engine reads the latest signals and places paper-trading orders through TWS, plus a continuously running daemon with overnight screening and daily trade limits.

Full detail on every option, output file, and email rule lives in **[USERGUIDE.md](USERGUIDE.md)**.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Market data comes from yfinance — no API key needed for backtesting.

```bash
git clone https://github.com/craigwhughes-dev/Strategy_Auto_Trader.git
cd Strategy_Auto_Trader
uv sync

# Backtest SPY with default settings (hourly bars, ~2 years of history)
uv run python -m Strategy_Auto_Trader.markov_cli.run --ticker SPY

# Try a different strategy and tighter stops
uv run python -m Strategy_Auto_Trader.markov_cli.run --ticker TSLA --strategy conservative --stop-loss-pct 0.03 --no-kelly
```

Each run writes a timestamped directory under `data/` containing the input data, per-bar backtest CSV, a four-panel chart, and a self-contained HTML daily summary — see [USERGUIDE.md](USERGUIDE.md#output-files) for the full list.

### Other entry points

| Command | Purpose |
|---------|---------|
| `... markov_cli.run --ticker SPY` | Single-ticker backtest + daily summary |
| `... markov_cli.batch` | Run the whole watchlist, send email alerts |
| `... markov_cli.screen` | Fast screen of the S&P 500 + FTSE 100 universe |
| `... markov_cli.live_sim --tickers SPY,TSLA` | Multi-ticker live simulation, compare strategies side by side |
| `... markov_cli.execute --dry-run` | Read latest signals, print the orders that *would* be placed |
| `... markov_cli.execute` | Place paper orders via IBKR TWS (port 7497) |
| `... markov_cli.live_daemon` | Continuous hourly trading loop with overnight screening |

(All invoked as `uv run python -m Strategy_Auto_Trader.<module>`.) IBKR setup — TWS configuration, capital pot, position limits — is covered in [USERGUIDE.md](USERGUIDE.md#ibkr-paper-trading-execution-engine); daemon deployment (Windows Task Scheduler) in [README_DAEMON.md](README_DAEMON.md).

## Driving it with Claude Code

This repo doubles as a Claude Code skill: open it in Claude Code and drive the whole workflow conversationally. Some prompts that work well:

**Backtesting and analysis**

- *"Backtest NVDA with the trend strategy and show me the exit-reason breakdown. Why did most trades close?"*
- *"Run the backtest on SPY and on QQQ with identical settings and tell me which one the strategy handles better, and why."*
- *"Compare the default, conservative, and optimised strategies on TSLA over the last two years — which had the best Sharpe and fewest whipsaws?"*
- *"Run the screener and then a full backtest on the top three winners."*
- *"ASTS keeps getting stopped out early. Suggest and test better vol-stop and min-hold settings for a ~90% annualised vol stock."*

**Understanding the system**

- *"Walk me through exactly why the strategy sold ASTS on the last SELL event — which exit fired and what were the votes that day?"*
- *"Explain how the quality gate vetoed BUYs in my last SPY run and whether it helped or hurt P&L."*
- *"What would change in the last backtest if Kelly sizing were disabled? Run both and diff the equity curves."*

**Extending it**

- *"Add a new strategy called 'aggressive' to the registry: looser entry threshold, no SMA200 gate, tighter take-profit. Backtest it against 'default' on my watchlist and write tests for it."*
- *"Add a maximum-drawdown circuit breaker to the exit stack and unit-test its boundary conditions."*
- *"Add three FTSE tickers to the watchlist with a per-ticker sell threshold of -2."*

**Operations**

- *"Run the batch for the FTSE watchlist but skip emails, then summarise the roundup for me here."*
- *"Do a dry-run of the execution engine and tell me what orders it would place and why."*
- *"Check the daemon log and execution state — did it trade today, and are the open positions consistent with the signals?"*

**Auto trading daemon**
- *"start the daemon for ftse100. use the optimised strategy. unlimited buy and sell. 10k original stake. record all results in a clean journal. use ibkr for paper trades"*

## Testing

```bash
uv run pytest tests/ -q
```

The suite covers the exit stack, indicators, composite signal weights, the quality gate, HMM forward filtering, the consolidated engine, the broker layer (sizing, capacity, daily limits, staleness guards), email/state handling, and the daemon's screening and scheduling logic.

## Project layout

```
Strategy_Auto_Trader/
  core/          # Indicators, quality gate, exit logic
  strategy/      # Strategy registry + default/conservative/trend/optimised
  plugins/       # Pluggable sizers, gates, adjusters, exit rules
  quant_hmm/     # HMM engine, consolidated walk-forward backtest, sentiment, vol screen
  markov_cli/    # CLI entry points (run, batch, screen, live_sim, execute, live_daemon)
  broker/        # IBKR adapter, NullBroker, portfolio manager, signal reader
  extensions/    # Multi-seed HMM with forward filter
  output/        # HTML report, chart, emailer, trade state, journal
config/          # Watchlists and daemon config
tests/           # pytest suite (mirrors the package structure)
```

## Documentation

- **[USERGUIDE.md](USERGUIDE.md)** — every CLI option, signal weighting, exit rule, output file, watchlist format, email setup, and IBKR configuration
- **[README_DAEMON.md](README_DAEMON.md)** — deploying the continuous trading daemon on Windows Task Scheduler
- **[TASK_SCHEDULER_SETUP.md](TASK_SCHEDULER_SETUP.md)** — scheduled batch runs after LSE/NYSE close

## Example email

Trade alerts arrive as a self-contained HTML email per ticker: the signal (BUY/SELL/HOLD) with the composite score, the individual signal votes behind it, momentum indicators, trailing-stop status, and the exit-indicator panel (MACD and RSI-reversal state). A BUY alert looks like this:

![Example BUY alert email for GOOGL](docs/example_email.png)

The daily roundup email is a companion summary: one row per watchlist ticker showing its current signal (with any trade event highlighted), close price, composite score, P&L, and strategy return versus buy-and-hold.

## IBKR paper trading

When the daemon runs with dry-run disabled, orders route to Interactive Brokers TWS (paper account) and appear in the TWS Mosaic view:

![IBKR TWS paper trading view](docs/ibkr_trader_view.png)
