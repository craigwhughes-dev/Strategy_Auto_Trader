# S&P 500 Full Backtest — In Progress

**Status**: Running (started 2026-07-03)

## Configuration

**Data**: 
- 325 S&P 500 tickers from watchlist_sp500.json
- Hourly data: Jan 1 2025 — present (via yfinance)

**Strategies** (independent, separate state):
1. **optimised** — Entry: HMM 2.0, volume 1.0, SMA200 3.0; strict buy 6.0/9.0; vetoes: RSI>70, regime≤0; wide exits
2. **trend_follow** — Entry: SMA200 3.0, trend 2.0; wide 8% stop/30% target, vol-scaled trailing stop

**Execution** (per strategy):
- Capital: $10,000
- Max positions: 50 concurrent
- Daily buy/sell limits: Unlimited
- Lot sizing: Kelly fraction, default $100 if undefined
- Transaction cost: $10 per trade

## Monitoring

**Log file**: `logs/sp500_backtest.log`

View progress:
```bash
tail -f logs/sp500_backtest.log
```

**Process**: 
- PID tracked in background
- Expected duration: 6-12+ hours (325 tickers × 2-year hourly HMM fitting)

## Expected Output

**Journal**: `data/journals/live.csv`
- All BUY/SELL trades from both strategies combined
- Columns: date, ticker, action, entry/exit price, quantity, P&L (if exit)

**Data directories**: `data/<TICKER>_optimised_*` and `data/<TICKER>_trend_*`
- One dir per ticker per strategy
- Contains: compositeBacktest.csv (daily OHLC + signals), qualityGate.json

**State**: None persisted (fresh run, starting from zero positions)

## Results Will Show

- Win rate, profit factor, max drawdown per strategy
- Which tickers profitable vs unprofitable
- Whether optimised or trend_follow performs better on hourly data
- Actual trading frequency and position turnover

---

Do not interrupt. Let run to completion overnight.
