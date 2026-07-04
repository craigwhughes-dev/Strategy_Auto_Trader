# Automated Paper Trading — Implementation Complete

**Status**: ✅ All code written, tested (489 tests passing), ready for deployment

**Date**: 2026-07-03

## What You Now Have

A fully automated, continuously-running paper trading system that:

1. **Overnight** (02:00 UK) — Screens all tickers by volatility character and sentiment, excluding choppy/bearish ones (unless they have open positions)
2. **During market hours** — Continuously cycles through in-scope tickers, prioritizing open positions (checked every hour) and round-robining through candidates with a smart time budget
3. **Enforces limits** — One shared IBKR paper account across FTSE+S&P500, respecting daily buy/sell limits and position capacity
4. **Survives failures** — Task Scheduler restarts the daemon if it crashes

## Files Ready to Deploy

| File | Purpose |
|------|---------|
| `config/overnight_strategy.json` | All config (vol thresholds, sentiment, trading hours, limits) |
| `Strategy_Auto_Trader/markov_cli/overnight_scope.py` | Overnight ticker screening |
| `Strategy_Auto_Trader/markov_cli/live_daemon.py` | Main persistent daemon loop |
| `run_daemon.bat` | Entry point for Task Scheduler |
| `CREATE_SCHEDULED_TASK.ps1` | PowerShell setup script (run as Admin) |
| `CREATE_SCHEDULED_TASK.bat` | Batch setup script (run as Admin) |
| `TASK_SCHEDULER_SETUP.md` | Step-by-step task creation guide |

## Quick Start (3 Steps)

### Step 1: Create the Scheduled Task

Choose one method (easiest is Method 2):

**Method 1 — PowerShell (Admin)**:
```powershell
cd C:\Users\Craig\.claude\skills\Strategy_Auto_Trader
.\CREATE_SCHEDULED_TASK.ps1
```

**Method 2 — Command Prompt (Admin)**:
```cmd
cd C:\Users\Craig\.claude\skills\Strategy_Auto_Trader
CREATE_SCHEDULED_TASK.bat
```

**Method 3 — Task Scheduler GUI**: See `TASK_SCHEDULER_SETUP.md`

### Step 2: Test in Dry-Run Mode

1. Start the task manually in Task Scheduler (right-click → Run)
2. Monitor logs:
   ```
   type logs\daemon_<today>.log
   ```
3. Verify it processes tickers and logs trading activity

### Step 3: Go Live

Once dry-run looks good:

1. Edit `config/overnight_strategy.json`:
   ```json
   "execution": {
     "dry_run": false
   }
   ```
2. Start TWS or IB Gateway on paper port (7497)
3. Let the daemon run — it will place real paper orders starting next market open

## Config Overview

`config/overnight_strategy.json` controls everything:

```json
{
  "markets": {
    "ftse": { "timezone": "Europe/London", "trading_start": "08:00", "trading_end": "16:30" },
    "sp500": { "timezone": "America/New_York", "trading_start": "09:30", "trading_end": "16:00" }
  },
  "overnight_run_time": "02:00",
  "vol_screen": { "enabled": true, "min_trend_quality": 0.0 },
  "sentiment_screen": { "enabled": true, "min_sentiment_score": -0.3, "exclude_labels": ["bearish"] },
  "exempt_if_open_position": true,
  "daytime": { "max_seconds_per_cycle": 1500, "poll_interval_seconds": 60 },
  "execution": {
    "capital_pot": 20000,
    "max_positions": 5,
    "daily_buy_limit": 2,
    "dry_run": true
  }
}
```

Key parameters:
- **vol_screen**: Kaufman Efficiency Ratio, autocorrelation, choppiness — tickers below min_trend_quality are excluded
- **sentiment_screen**: Put/call ratio, IV rank, VIX regime, insider activity, short interest — bearish tickers excluded unless open position
- **daytime.max_seconds_per_cycle**: Time budget per hourly market cycle (1500s = 25 min, plenty for 300+ tickers with must-run first)
- **execution.dry_run**: Set to `false` to place real IBKR paper orders

## Verification Checklist

Before going live:

- [ ] Task created successfully (check Task Scheduler)
- [ ] Dry-run started and logs show "Live daemon starting"
- [ ] Overnight scope ran at 02:00 and created `state/in_scope_ftse.json` and `state/in_scope_sp500.json`
- [ ] Tickers were screened (some excluded for vol/sentiment, open positions always kept)
- [ ] Generated watchlists exist at `config/generated/watchlist_ftse_scoped.json`, etc.
- [ ] During market hours, daemon logs show "Starting cycle" every hour
- [ ] Must-run positions processed first, then round-robin candidates
- [ ] `state/execution_state.json` tracks open positions and daily trade counts
- [ ] No real orders placed while `dry_run: true`

## What's Logged

`logs/daemon_<date>.log` shows (per hour per market):

```
================================================================
[ftse] Starting cycle
================================================================
  [ftse] Must-run (2 positions):
    Processing OPEN_TICKER_1
    Processing OPEN_TICKER_2
  [ftse] Round-robin (198 candidates, 1400s budget):
    Processing TICKER_A
    Processing TICKER_B
    ... (until budget exhausted)
  [ftse] Executing signals for 200 processed tickers...
    BUY: TICKER_X x100 @ 123.45
    SELL: TICKER_Y x50 @ 234.56
    Skipped: 10
  [ftse] Cycle done: 200 tickers processed, 45 skipped (budget), 123s elapsed
```

## Known v1 Limitations

1. **No market holidays** — Daemon only checks weekday + trading hours. Christmas/Easter still have the daemon running (harmlessly, because no data is fetched, but it's wasted CPU). Can add a holiday calendar later.
2. **Single shared capital** — Both markets share one pot. If you later want separate books, that's a config change.
3. **No persistent supervision** — Daemon can crash and won't restart until Task Scheduler's next retry. A proper supervisor service would help, but Task Scheduler's restart-on-failure is good enough for v1.

## Troubleshooting

### Task won't create ("Access Denied")
- Run Command Prompt/PowerShell as Administrator (right-click → Run as administrator)

### Daemon starts but closes immediately
- Check `logs/daemon_<date>.log` for error messages
- Verify `overnight_strategy.json` exists and is valid JSON
- Verify `config/watchlist_ftse.json` and `config/watchlist_sp500.json` exist

### No tickers in overnight screening
- Run `overnight_scope` manually: `uv run python -m Strategy_Auto_Trader.markov_cli.overnight_scope`
- Check console output for vol-screen/sentiment errors
- Verify yfinance can reach the internet

### No trades happening during market hours
- Check logs for "cycle done" messages (verify it's processing tickers)
- Check that overnight scope ran and created scoped watchlist
- Verify `execution.dry_run: true` in config (if true, no real orders will be placed)
- If `dry_run: false`, ensure TWS/IB Gateway is running on paper port 7497

## Next Steps (Future Work)

Not in scope for v1, but flagged for later:

1. Market holidays calendar integration
2. Per-market capital allocation
3. Metrics dashboard (daemon health, position P&L, cycle duration)
4. Backtest validation of new `optimised` strategy from 2026-07-03 forward
5. Supervisor service for better crash recovery

## Support

All code follows existing conventions in the repo:
- No comments unless WHY is non-obvious
- DRY across daemon and batch.py (both use `process_ticker()`)
- Comprehensive unit tests (12 new + 477 existing, all passing)
- Logging via Python's `logging` module (rotates daily)

For issues, check:
1. `logs/daemon_<date>.log` for runtime errors
2. `state/daemon_state.json` for cursor/overnight state
3. `state/execution_state.json` for position tracking
4. `state/in_scope_*.json` for overnight screening results
