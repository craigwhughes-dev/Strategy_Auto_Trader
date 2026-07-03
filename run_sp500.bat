@echo off
:: S&P 500 daily batch — schedule at 9:30 PM UK time
:: Requires SMTP_USER / SMTP_PASSWORD environment variables for email alerts
cd /d "%~dp0"
uv run python -m Strategy_Auto_Trader.markov_cli.batch --watchlist config\watchlist_sp500.json >> logs\sp500_%date:~-4%%date:~3,2%%date:~0,2%.log 2>&1
