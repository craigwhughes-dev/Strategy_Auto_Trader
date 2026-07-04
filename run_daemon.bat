@echo off
:: Live daemon — persistent automated paper trading
:: Run continuously under Task Scheduler: At logon, restart on failure
cd /d "%~dp0"
uv run python -m Strategy_Auto_Trader.markov_cli.live_daemon >> logs\daemon.log 2>&1
