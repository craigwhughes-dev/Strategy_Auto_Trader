@echo off
:: Live daemon — persistent automated paper trading
:: Run continuously under Task Scheduler: At logon, restart on failure
:: --takeover: replace a still-running/orphaned instance instead of
:: refusing to start on a stale process lock.
::
:: Calls .venv\Scripts\python.exe directly, NOT "uv run" — even non-
:: elevated, "uv run" still spawns the target interpreter as a child that
:: survived Task Scheduler's "End Task" in testing (uv isn't built for
:: Windows Job Object containment; something in its process hand-off lets
:: the child escape). A directly-launched process has no child to escape
:: from. See CREATE_SCHEDULED_TASK.ps1/.bat for the matching elevation fix
:: (RunLevel must also be non-elevated, or Task Scheduler launches the
:: process outside its own Job Object regardless of this).
cd /d "%~dp0"
".venv\Scripts\python.exe" -m Strategy_Auto_Trader.markov_cli.live_daemon --takeover >> logs\daemon.log 2>&1
