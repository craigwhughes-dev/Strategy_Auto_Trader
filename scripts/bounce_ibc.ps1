# Manually bounce IBGatewayIBC — ends the task (kills Gateway/IBC process),
# waits for it to die, then re-runs the task so IBC logs back in.
# Must run elevated (S4U task processes require admin to kill).

$task = "IBGatewayIBC"

Write-Output "Stopping $task..."
schtasks /end /tn $task

Write-Output "Waiting 15 seconds for Gateway to close..."
Start-Sleep -Seconds 15

Write-Output "Starting $task..."
schtasks /run /tn $task

Write-Output "Done. IBC should log in within ~30 seconds. Check daemon log for reconciliation success."
