# Manually bounce IBGatewayIBC — ends the task (kills Gateway/IBC process),
# waits for it to die, then re-runs the task so IBC logs back in.
# Must run elevated (S4U task processes require admin to kill).

$task = "IBGatewayIBC"

Write-Output "Stopping $task..."
schtasks /end /tn $task




