# Runs at 03:10 daily after IBC's own ColdRestartTime (07:15 is the Sunday
# cold-restart time; this daily check is for ordinary daily auto-restarts).
# Only starts IBGatewayIBC_Live if port 4001 is not listening - avoids
# double-starting IBC when the nightly restart succeeded normally.

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$listening = netstat -an | Select-String "0\.0\.0\.0:4001\s+.*LISTENING"

if (-not $listening) {
    Write-Output "$timestamp IB Gateway (live) not on port 4001 - starting IBGatewayIBC_Live"
    schtasks /run /tn IBGatewayIBC_Live
} else {
    Write-Output "$timestamp IB Gateway (live) already running on port 4001 - no action"
}
