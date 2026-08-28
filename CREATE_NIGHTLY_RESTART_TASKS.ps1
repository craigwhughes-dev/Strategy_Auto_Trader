# Nightly restart for StrategyAutoTraderDaemon (BACKTEST_LIVE_PARITY_PLAN.md Step 1c)
# Run this in PowerShell (admin not required).
#
# Uses schtasks.exe, not Register-ScheduledTask: the cmdlet's underlying
# WMI/CIM registration (Root/Microsoft/.../PS_ScheduledTask) returned
# "Access is denied" in this environment even non-elevated, while schtasks.exe
# (the older Task Scheduler v1-compatible CLI) succeeded against the same
# account with the same rights — a different API path into the same service.
#
# Why two tasks instead of one: an End action and a Run action are the two
# verbs Task Scheduler actually exposes (schtasks /end, schtasks /run — same
# as the GUI's "End" and "Run" buttons). Two triggers 2 minutes apart, rather
# than one task doing both, keeps each half independently visible in Task
# Scheduler's last-run-result column if one fails.
#
# Times are 02:00 / 02:02 local (this machine's local time already tracks
# UK clock incl. DST, so no separate "London time" conversion is needed) —
# well clear of FTSE 08:00-16:30 and S&P 14:30-21:00 UK-clock sessions, and
# of the 21:30 reconciliation run and nightly roundup email.
#
# The main task's own --takeover flag makes the restart a no-op if nothing
# changed and safely picks up whatever code is on disk if something did —
# same as a manual bounce, just scheduled instead of intraday.

$mainTask = "StrategyAutoTraderDaemon"

schtasks /Create /TN "${mainTask}_NightlyStop" `
    /TR "C:\Windows\System32\schtasks.exe /end /tn $mainTask" `
    /SC DAILY /ST 02:00 /F

schtasks /Create /TN "${mainTask}_NightlyStart" `
    /TR "C:\Windows\System32\schtasks.exe /run /tn $mainTask" `
    /SC DAILY /ST 02:02 /F

Write-Output ""
Write-Output "Verify: check logs/daemon_*.log for a 'Live daemon starting' banner"
Write-Output "around 02:02 local, and that startup reconciliation comes back clean."
