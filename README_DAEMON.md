# Strategy Auto-Trader Daemon — Deployment Guide

## Overview

You now have a fully-automated, continuously-running paper trading daemon that:

✅ Screens tickers overnight by volatility character and sentiment  
✅ Runs hourly market cycles during FTSE (08:00-16:30) and S&P500 (09:30-16:00) hours  
✅ Prioritizes open positions (checked every hour) and round-robins through remaining candidates  
✅ Enforces daily buy/sell limits and position capacity across one shared IBKR paper account  
✅ Survives crashes with automatic Task Scheduler restart  

## Files You Need

**Entry points**:
- `run_daemon.bat` — Task Scheduler launcher
- `CREATE_SCHEDULED_TASK.ps1` — PowerShell setup (easiest)
- `CREATE_SCHEDULED_TASK.bat` — Command Prompt setup (alternative)

**Config**:
- `config/overnight_strategy.json` — All daemon config (vol thresholds, sentiment, hours, limits)

**Daemon code**:
- `Strategy_Auto_Trader/markov_cli/live_daemon.py` — Main loop (imports overnight_scope)
- `Strategy_Auto_Trader/markov_cli/overnight_scope.py` — Overnight screening

**Refactored for reuse**:
- `Strategy_Auto_Trader/markov_cli/batch.py` — Extracted `process_ticker()`
- `Strategy_Auto_Trader/markov_cli/execute.py` — Extracted `execute_signals()`

**Tests** (all passing, 489 total):
- `tests/markov_cli/test_overnight_scope.py`
- `tests/markov_cli/test_live_daemon.py`

## Deployment Steps

### 1. Create the Scheduled Task (5 minutes)

Open Command Prompt **as Administrator** and run:

```cmd
cd C:\Users\Craig\.claude\skills\Strategy_Auto_Trader
CREATE_SCHEDULED_TASK.bat
```

Or in PowerShell **as Administrator**:

```powershell
cd C:\Users\Craig\.claude\skills\Strategy_Auto_Trader
.\CREATE_SCHEDULED_TASK.ps1
```

Verify in Task Scheduler:
- Task name: **Strategy Auto-Trader Daemon**
- Status: **Ready**
- Trigger: **At logon**

### 2. Test in Dry-Run Mode (5 minutes)

1. In Task Scheduler, right-click the task → **Run**
2. Open a terminal and check the log:
   ```cmd
   type logs\daemon_2026-07-03.log
   ```
   (Replace `2026-07-03` with today's date)

3. Look for:
   ```
   ====================================================================
   Live daemon starting
   ====================================================================
   Using NullBroker (dry run mode)
   Broker connected
   Entering main loop
   ```

4. Check the generated files:
   ```
   state/daemon_state.json  ← overnight state tracking
   state/in_scope_ftse.json ← FTSE tickers screened (if it's 02:00+ UK time)
   state/in_scope_sp500.json ← S&P500 tickers screened
   config/generated/watchlist_ftse_scoped.json ← filtered watchlist
   config/generated/watchlist_sp500_scoped.json
   ```

5. Verify it doesn't crash and logs show processing activity

### 3. Go Live (2 minutes)

Once dry-run looks good:

1. Edit `config/overnight_strategy.json`:
   - Find: `"dry_run": true`
   - Change to: `"dry_run": false`

2. Start TWS or IB Gateway on the paper trading port:
   - **TWS**: Log in to paper account, leave it running
   - **IB Gateway**: Similar, paper port is 7497

3. Restart the daemon (or reboot):
   ```cmd
   taskkill /tn "Strategy Auto-Trader Daemon" /f
   ```
   Task Scheduler will restart it automatically.

4. Monitor logs for real orders:
   ```
   [ftse] Executing signals...
     BUY: TICKER_A x100 @ 123.45
     SELL: TICKER_B x50 @ 234.56
   ```

## Configuration

All settings in `config/overnight_strategy.json`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `vol_screen.enabled` | true | Filter choppy tickers (Kaufman ER, autocorr, choppiness) |
| `vol_screen.min_trend_quality` | 0.0 | Min trend-quality score (higher = stricter) |
| `sentiment_screen.enabled` | true | Filter bearish tickers (P/C ratio, IV rank, VIX, insider, short interest) |
| `sentiment_screen.exclude_labels` | ["bearish"] | Exclude tickers with these labels unless open |
| `overnight_run_time` | "02:00" | When to screen (UTC) |
| `daytime.max_seconds_per_cycle` | 1500 | Time budget per hourly cycle (1500s = 25 min) |
| `daytime.poll_interval_seconds` | 60 | How often to check for new cycles |
| `execution.capital_pot` | 20000 | Total paper account capital |
| `execution.max_positions` | 5 | Max open positions at once |
| `execution.daily_buy_limit` | 2 | Max BUY orders per day (both markets) |
| `execution.daily_sell_limit` | null | Max SELL orders per day (null = unlimited) |
| `execution.dry_run` | true | Safe default (no real orders until you flip to false) |

## How It Works

### Overnight (02:00 UK time)
```
1. Load watchlist tickers for both FTSE and S&P500
2. Vol-screen each: Kaufman ER, autocorr, choppiness index
   → Keep trending tickers, exclude choppy
3. Sentiment-screen each: put/call, IV rank, VIX, insider, short interest
   → Exclude bearish labels, keep open positions regardless
4. Write audit trail: state/in_scope_ftse.json, state/in_scope_sp500.json
5. Generate scoped watchlists: config/generated/watchlist_*_scoped.json
   (with merged defaults including daily limits)
```

### Each Market Hour
```
1. Check if market is in trading hours (FTSE 08:00-16:30, S&P500 09:30-16:00)
2. If yes and this hour hasn't run yet:
   a. Load in-scope tickers from overnight screening
   b. Identify open positions (must-run every hour)
   c. Run open positions first (before time budget starts)
   d. Round-robin through remaining in-scope tickers
      (advancing cursor daily, wrapping to start each day)
   e. Stop taking new tickers when time budget (1500s) is spent
   f. Call execute_signals() once for all processed tickers
      → BUY highest-Kelly tickers first, respecting daily limit
      → SELL any SELL signals not at capacity
      → Record fills in execution_state.json
3. Sleep 60 seconds, loop
```

## Logs

Daily rotating logs in `logs/daemon_<date>.log` show:

```
[ftse] Must-run (2 positions):
  Processing OPEN_TICKER_1
  Processing OPEN_TICKER_2
[ftse] Round-robin (250 candidates, 1400s budget):
  Processing TICKER_A
  Processing TICKER_B
  ... (until budget exhausted or all done)
[ftse] Executing signals for 252 processed tickers...
  BUY: TICKER_X x100 @ 123.45
  SELL: TICKER_Y x50 @ 234.56
  Skipped: 5(daily limit reached), 10(at capacity), 3(qty=0)
[ftse] Cycle done: 252 tickers processed, 18 skipped (budget), 123s elapsed
```

## State Files

| File | Purpose |
|------|---------|
| `state/daemon_state.json` | Overnight date, round-robin cursors per market per day |
| `state/execution_state.json` | Open positions, trade log, daily trade counts |
| `state/in_scope_ftse.json` | FTSE screening results (kept, excluded+reason) |
| `state/in_scope_sp500.json` | S&P500 screening results |

## Monitoring

**Daily**:
- Check `logs/daemon_<today>.log` for errors or unusual activity
- Verify `state/execution_state.json` shows expected positions

**Weekly**:
- Review `state/execution_state.json` trade_log for P&L
- Check overnight screening exclusions (vol-screened tickers, sentiment-filtered)

**Monthly**:
- Analyze trading performance (trade_log, journal entries)
- Adjust `overnight_strategy.json` if needed (vol thresholds, sentiment labels, limits)

## Troubleshooting

### Task won't create
**Error**: Access denied  
**Fix**: Right-click Command Prompt/PowerShell → "Run as administrator"

### Daemon stops immediately
**Check**:
1. `logs/daemon_<date>.log` — does it exist? What's the error?
2. `config/overnight_strategy.json` — valid JSON? All keys present?
3. `config/watchlist_ftse.json`, `config/watchlist_sp500.json` — exist?

**Fix**: Fix the error and restart the task

### No tickers in overnight screening
**Check**: Run manually:
```cmd
uv run python -m Strategy_Auto_Trader.markov_cli.overnight_scope
```
Look for errors about yfinance, vol_screen, or sentiment.

**Fix**: Check internet connection, verify yfinance works

### No orders despite `dry_run: false`
**Check**:
1. Logs show tickers being processed (not quiet)?
2. `state/in_scope_*.json` has tickers (not empty)?
3. TWS/IB Gateway running on paper port 7497?
4. Correct time — is it during market hours?

**Fix**: Verify each item above, restart daemon

## Uninstall (if needed)

Delete the Task Scheduler task:

**Command Prompt (Admin)**:
```cmd
schtasks /delete /tn "Strategy Auto-Trader Daemon" /f
```

**PowerShell (Admin)**:
```powershell
Unregister-ScheduledTask -TaskName "Strategy Auto-Trader Daemon" -Confirm:$false
```

You can delete the code anytime — just the task deletion removes the scheduled run.

## Next Steps

After deploying and running for a week:

1. Review trading performance in `state/execution_state.json`
2. Check overnight screening exclusions in `state/in_scope_*.json`
3. Adjust `overnight_strategy.json` if:
   - Too many tickers excluded → lower vol_screen.min_trend_quality
   - Too many false negatives → raise sentiment_screen.min_sentiment_score
   - Running out of time budget → increase daytime.max_seconds_per_cycle or reduce tickers
4. Later: Add market holiday calendar, per-market capital allocation, metrics dashboard

## Support

All code tested (489 tests passing) and documented. Check:

- **Code docs**: Each module has a docstring explaining its purpose
- **Test examples**: `tests/markov_cli/test_overnight_scope.py` and `test_live_daemon.py` show expected behaviour
- **Config examples**: `config/overnight_strategy.json` is fully commented with defaults
- **Logs**: Daemon logs are verbose and timestamped

Questions? Check `IMPLEMENTATION_COMPLETE.md` or `TASK_SCHEDULER_SETUP.md` for more details.
