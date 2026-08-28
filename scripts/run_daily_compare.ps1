# run_daily_compare.ps1 — Task Scheduler entry point for daily backtest comparison.
# Runs after US market close (scheduled 22:30 London time).

$Root = "C:\Users\Craig\.claude\skills\Strategy_Auto_Trader"
$LogDir = "$Root\logs"
$Date = (Get-Date -Format "yyyy-MM-dd")
$LogFile = "$LogDir\daily_compare_taskrunner_$Date.log"

Set-Location $Root

"[$Date $(Get-Date -Format 'HH:mm:ss')] Starting daily backtest compare..." | Tee-Object -FilePath $LogFile -Append

uv run python scripts/daily_backtest_compare.py 2>&1 | Tee-Object -FilePath $LogFile -Append

$exit = $LASTEXITCODE
"[$Date $(Get-Date -Format 'HH:mm:ss')] Finished (exit $exit)." | Tee-Object -FilePath $LogFile -Append

exit $exit
