@echo off
:: Live daemon — persistent automated paper trading
:: Run continuously under Task Scheduler: At logon, restart on failure
:: --takeover: replace a still-running/orphaned instance instead of
:: refusing to start on a stale process lock (see CREATE_SCHEDULED_TASK.ps1
:: for why the task itself must NOT run elevated — elevation is what let a
:: prior instance orphan past "End Task" in the first place).
cd /d "%~dp0"
uv run python -m Strategy_Auto_Trader.markov_cli.live_daemon --takeover >> logs\daemon.log 2>&1
