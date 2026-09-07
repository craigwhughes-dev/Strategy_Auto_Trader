# Runs at 03:00 daily after IBC's own AutoRestartTime (02:30).
# Only starts IBGatewayIBC if port 4002 is not listening - avoids
# double-starting IBC when the nightly restart succeeded normally.

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$listening = netstat -an | Select-String "0\.0\.0\.0:4002\s+.*LISTENING"

if (-not $listening) {
    Write-Output "$timestamp IB Gateway not on port 4002 - starting IBGatewayIBC"
    schtasks /run /tn IBGatewayIBC
} else {
    Write-Output "$timestamp IB Gateway already running on port 4002 - no action"
}
