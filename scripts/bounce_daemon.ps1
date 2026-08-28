# Manually bounce StrategyAutoTraderDaemon by running the same stop/start
# tasks that the nightly scheduler fires, with a 10-second gap.

$mainTask  = "StrategyAutoTraderDaemon"
$stopTask  = "${mainTask}_NightlyStop"
$startTask = "${mainTask}_NightlyStart"

Write-Output "Stopping $mainTask..."
schtasks /run /tn $stopTask

Write-Output "Waiting 10 seconds..."
Start-Sleep -Seconds 10

Write-Output "Starting $mainTask..."
schtasks /run /tn $startTask

Write-Output "Done. Check logs\daemon_*.log for 'Live daemon starting' banner."
