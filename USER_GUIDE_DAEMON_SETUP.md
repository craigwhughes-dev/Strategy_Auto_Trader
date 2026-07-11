# Strategy Auto Trader - Daemon Setup Guide

## Overview

This guide explains how to set up and run the Strategy Auto Trader daemon for automated 24/7 live trading.

**The daemon:**
- Runs continuously, processing market cycles hourly
- Automatically restarts if it crashes
- Generates trading signals based on HMM + composite strategy
- Executes trades via Interactive Brokers (paper or live)

---

## Prerequisites

### Software Requirements
- **Python 3.12+** (via `uv` package manager)
- **Interactive Brokers TWS** or **IB Gateway** (running and logged in)
- **Windows 10/11** or Linux/macOS with `systemd` or `supervisord`

### Account Setup
- Interactive Brokers account (paper or live)
- TWS/IB Gateway API enabled on port **7497** (default)
- Account connected and authenticated

---

## Installation

### 1. Clone the Project
```bash
git clone <repository-url>
cd Strategy_Auto_Trader
```

### 2. Install Dependencies
```bash
uv sync
```

This installs all required packages including:
- `pandas`, `numpy`, `yfinance` (data stack)
- `hmmlearn` (HMM modeling)
- `ib_insync` (Interactive Brokers)
- `psutil` (process monitoring)

### 3. Verify Installation
```bash
uv run pytest tests/test_all.py -q --tb=short
```

Should complete with 0 errors.

---

## Configuration

### 1. Create Watchlists

**Location:** `config/watchlist_<market>.json`

**Example:** `config/watchlist_uk.json`
```json
{
  "defaults": {
    "strategy": "optimised",
    "initial_cash": 10000,
    "transaction_cost": 1
  },
  "tickers": [
    {"ticker": "AUTO.L"},
    {"ticker": "SHEL.L"},
    {"ticker": "GSK.L"}
  ]
}
```

**Repeat for other markets** (US, EU, etc.)

### 2. Configure Daemon Settings

**File:** `config/overnight_strategy.json`

```json
{
  "markets": {
    "uk": {
      "watchlist": "config/watchlist_uk.json",
      "timezone": "Europe/London",
      "trading_start": "08:00",
      "trading_end": "16:30",
      "defaults": {"strategy": "optimised"}
    },
    "us": {
      "watchlist": "config/watchlist_us.json",
      "timezone": "America/New_York",
      "trading_start": "09:30",
      "trading_end": "16:00",
      "defaults": {"strategy": "optimised"}
    }
  },
  "overnight_run_time": "02:00",
  "overnight_timezone": "Europe/London",
  "reconciliation_run_time": "21:30",
  "daytime": {
    "cycle_buffer_minutes": 5,
    "max_seconds_per_cycle": 1500,
    "poll_interval_seconds": 60
  },
  "execution": {
    "capital_pot": 10000,
    "max_positions": 5,
    "daily_buy_limit": 2,
    "daily_sell_limit": null,
    "dry_run": false
  }
}
```

**Key settings:**
- `markets` — Define which markets to trade (timezone, hours, tickers)
- `capital_pot` — Total capital per market
- `max_positions` — Max concurrent positions
- `dry_run` — `false` for real trading, `true` for dry-run (NullBroker)

---

## Starting the Daemon

### Option A: Manual (for testing)

```bash
cd /path/to/Strategy_Auto_Trader
uv run python -m Strategy_Auto_Trader.markov_cli.live_daemon
```

**Expected output:**
```
[INFO] ================================================================
[INFO] Live daemon starting
[INFO] ================================================================
[INFO] Startup environment validation OK
[INFO] Process lock acquired (PID 12345)
[INFO] Self-check OK [data stack]: numpy ..., pandas ..., yfinance ...
[INFO] Self-check OK [HMM fit]: hmmlearn ... test fit OK
[INFO] Using IBKRAdapter (live paper trading)
[INFO] Broker connection deferred (will connect on first trade)
[INFO] Entering main loop
[DEBUG]   Market <market>: Starting cycle
```

Press `Ctrl+C` to stop.

### Option B: Windows Task Scheduler (recommended for production)

#### Step 1: Create batch file
**File:** `run_daemon.bat`

```batch
@echo off
cd /d "%~dp0"
uv run python -m Strategy_Auto_Trader.markov_cli.live_daemon >> logs\daemon.log 2>&1
```

#### Step 2: Open Task Scheduler
- Press `Win + R`, type `taskschd.msc`, press Enter

#### Step 3: Create New Task
1. Right-click **Task Scheduler Library** → **Create Task**
2. **Name:** `StrategyAutoTraderDaemon`
3. **Description:** `Automated live trading with auto-restart`

#### Step 4: Configure Trigger
1. Go to **Triggers** tab
2. Click **New**
3. Set: **At logon** (runs when you log in)
4. Check **Enabled** → **OK**

#### Step 5: Configure Action
1. Go to **Actions** tab
2. Click **New**
3. **Action:** Start a program
4. **Program/script:** `cmd.exe`
5. **Add arguments:** `/c cd /d "<full-path-to-project>" && uv run python -m Strategy_Auto_Trader.markov_cli.live_daemon`
6. **Start in:** `<full-path-to-project>`
7. Click **OK**

#### Step 6: Configure Settings
1. Go to **Settings** tab
2. ✓ Allow task to be run on demand
3. ✓ If task fails, restart every: **1 minute**
4. Attempt to restart up to: **999** times
5. If task is already running: **Do not start a new instance**
6. Stop task if it runs longer than: **24 hours**
7. Click **OK** to save

#### Step 7: Test
1. Right-click task → **Run**
2. Wait 10 seconds
3. Verify process started: `Get-Process python`
4. Check logs: `tail -f logs/daemon_*.log`

---

## Monitoring

### Check if Daemon is Running

**Windows:**
```powershell
Get-Process python | Measure-Object
# Should show 2-3 Python processes
```

**Linux/macOS:**
```bash
ps aux | grep live_daemon
```

### View Live Logs

**Latest log file:**
```bash
tail -f logs/daemon_2026-07-11.log
```

**Expected output during market hours:**
```
2026-07-14 09:30:15 [INFO] [us] Starting cycle
2026-07-14 09:30:15 [INFO] [us] Must-run (0 positions):
2026-07-14 09:30:15 [INFO] [us] Round-robin (325 candidates, 1200s budget):
2026-07-14 09:30:15 [DEBUG]   Processing AAPL
2026-07-14 09:31:42 [DEBUG]   Processing MSFT
...
2026-07-14 09:45:00 [INFO] [us] Executing signals for 50 processed tickers...
2026-07-14 09:45:01 [INFO]   BUY:  2, SELL: 1, Skipped: 47
2026-07-14 09:45:01 [INFO]     BUY: AAPL (score=0.75, kelly=0.10)
2026-07-14 09:45:01 [INFO]     SELL: MSFT (reason: quality_gate: adverse exit context)
```

### Check Lock File

```bash
cat state/daemon.lock
# Output: 12345|2026-07-14T09:30:15.123456
# Format: PID|timestamp
```

If lock is stale (PID no longer running), daemon will remove it on next restart.

---

## Troubleshooting

### Daemon Won't Start

**Error:** `Daemon already running (lock file exists)`
- **Cause:** Previous instance crashed, lock not cleaned
- **Fix:** 
  ```bash
  rm state/daemon.lock
  uv run python -m Strategy_Auto_Trader.markov_cli.live_daemon
  ```

**Error:** `Cannot connect to TWS/Gateway`
- **Cause:** TWS not running or API not enabled
- **Fix:**
  1. Start TWS or IB Gateway
  2. Login to account
  3. Enable API (TWS → Config → API → Settings → ✓ Enable ActiveX and Socket Clients)
  4. Restart daemon

**Error:** `Startup environment validation OK` then hangs
- **Cause:** Broker connection timeout
- **Fix:** Daemon defers broker connection; it will retry on first trade. Safe to ignore.

### No Trades Executing

**Check:**
1. Is it market hours?
   - UK: Mon-Fri 08:00-16:30 London time
   - US: Mon-Fri 09:30-16:00 ET

2. Are tickers in watchlist?
   ```bash
   cat config/watchlist_uk.json | grep ticker
   ```

3. Did backtests complete?
   ```bash
   ls -ltr data/ | tail -10
   # Should show recent run directories
   ```

4. Are signals being generated?
   ```bash
   grep "BUY\|SELL" logs/daemon_*.log
   ```

### Daemon Crashes Repeatedly

**Check logs for errors:**
```bash
grep ERROR logs/daemon_*.log | tail -20
grep CRITICAL logs/daemon_*.log | tail -20
```

**Common issues:**
- **HMM fitting timeout** — Reduce tickers or increase timeout
- **Broker disconnection** — Check TWS connection
- **Disk full** — Check available space: `df -h`

**Recovery:** Task Scheduler will restart automatically within 1 minute.

---

## Daily Operations

### Morning (Before Market Open)
1. Verify daemon is running: `Get-Process python`
2. Check logs: `tail -f logs/daemon_*.log`
3. Verify TWS/IB Gateway is connected

### During Market Hours
- Daemon processes tickers hourly
- Generates signals automatically
- Executes trades based on configuration
- Monitor logs for BUY/SELL activity

### Evening (After Market Close)
- Daemon reconciles positions with broker (default 21:30 London time)
- If mismatch detected, email alert sent
- New entries blocked until resolved

### Overnight
- Daemon runs overnight screening (default 02:00 London time)
- Updates in_scope ticker lists for tomorrow
- No trading during off-hours

---

## Production Checklist

Before running in production:

- [ ] Interactive Brokers account set up (paper or live)
- [ ] TWS/IB Gateway running and logged in
- [ ] API enabled on port 7497
- [ ] Watchlist JSON files created and tested
- [ ] `overnight_strategy.json` configured
- [ ] Backtests run for all tickers (verify signals generated)
- [ ] Dry-run tested (with `dry_run: true`)
- [ ] Task Scheduler task created and tested
- [ ] Manual daemon start/stop tested
- [ ] Logs directory accessible
- [ ] Disk space available (at least 10GB recommended)
- [ ] Email alerts configured (optional)

---

## Advanced: Linux/macOS Setup

For Linux/macOS, use `systemd` or `supervisord` instead of Task Scheduler:

### systemd Service File

**File:** `/etc/systemd/system/strategy-trader-daemon.service`

```ini
[Unit]
Description=Strategy Auto Trader Daemon
After=network.target

[Service]
Type=simple
User=<username>
WorkingDirectory=/path/to/Strategy_Auto_Trader
ExecStart=/path/to/uv run python -m Strategy_Auto_Trader.markov_cli.live_daemon
Restart=always
RestartSec=60
StandardOutput=append:/path/to/Strategy_Auto_Trader/logs/daemon.log
StandardError=append:/path/to/Strategy_Auto_Trader/logs/daemon.log

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl enable strategy-trader-daemon
sudo systemctl start strategy-trader-daemon
```

**Monitor:**
```bash
sudo systemctl status strategy-trader-daemon
sudo journalctl -u strategy-trader-daemon -f
```

---

## Support & Logs

### Log Locations
- **Daemon logs:** `logs/daemon_YYYY-MM-DD.log` (daily rotation)
- **Backtest data:** `data/<TICKER>_YYYY-MM-DDTHH:MM:SSZ/`
- **State files:** `state/execution_state.json`, `state/daemon.lock`
- **Configuration:** `config/overnight_strategy.json`, `config/watchlist_*.json`

### Common Log Patterns

**Normal startup:**
```
[INFO] Process lock acquired (PID XXXXX)
[INFO] Self-check OK [data stack]: ...
[INFO] Self-check OK [HMM fit]: ...
[INFO] Using IBKRAdapter
[INFO] Entering main loop
```

**Processing tickers:**
```
[INFO] [<market>] Starting cycle
[DEBUG]   Processing <TICKER>
[INFO] [<market>] Executing signals for X processed tickers...
[INFO]   BUY:  X, SELL: Y, Skipped: Z
```

**Error conditions:**
```
[ERROR] Daemon already running (lock file exists)
[CRITICAL] startup self-checks failed, refusing to run
[ERROR] Reconciliation mismatch: ... — halting new entries
```

---

## Next Steps

1. **Initial setup:** Follow Steps 1-3 under Configuration
2. **Test locally:** Run `uv run python -m Strategy_Auto_Trader.markov_cli.live_daemon` manually
3. **Verify signals:** Check that backtests generate BUY/SELL signals
4. **Set up Task Scheduler:** Follow Option B for production
5. **Monitor first week:** Watch logs for issues, fine-tune settings
6. **Production trade:** Start with `dry_run: false` on paper account
7. **Scale up:** Once confidence is high, increase tickers or capital

---

## Version History

- **v1.0** (2026-07-11): Initial daemon with PID-based lock, Task Scheduler auto-restart, fail-fast validation
- Reliability: Auto-restart every 1 minute on crash (up to 999 times)
- Status: Production-ready

---

**Questions?** Check logs, verify Task Scheduler task is enabled, ensure TWS is connected.
