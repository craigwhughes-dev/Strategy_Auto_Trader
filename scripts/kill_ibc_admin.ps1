# Kill IB Gateway/IBC and all related processes
# Must run as administrator

Write-Output "=== IBC/Gateway Kill Script (Admin Mode) ==="
Write-Output ""

# End scheduled tasks first
Write-Output "[1/4] Stopping scheduled tasks..."
$tasks = @("IBGatewayIBC", "EnsureIBCRunning")
foreach ($task in $tasks) {
    try {
        $result = schtasks /end /tn $task 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Output "  [OK] Stopped: $task"
        } else {
            Write-Output "  - Task not running or not found: $task"
        }
    } catch {
        Write-Output "  - Error stopping $task`:" $_
    }
}
Write-Output ""

# Kill processes
Write-Output "[2/4] Killing processes..."
$processes = @("java", "IBGateway", "IBC")
foreach ($proc in $processes) {
    try {
        $running = Get-Process -Name $proc -ErrorAction SilentlyContinue
        if ($running) {
            Stop-Process -Name $proc -Force -ErrorAction SilentlyContinue
            Write-Output "  [OK] Killed: $proc ($(($running | Measure-Object).Count) process(es))"
        }
    } catch {
        Write-Output "  - Error killing $proc`:" $_
    }
}
Write-Output ""

# Wait and verify
Write-Output "[3/4] Waiting 5 seconds for processes to exit..."
Start-Sleep -Seconds 5
Write-Output ""

Write-Output "[4/4] Verification:"
$stillRunning = $false
foreach ($proc in $processes) {
    $check = Get-Process -Name $proc -ErrorAction SilentlyContinue
    if ($check) {
        Write-Output "  [FAIL] Still running: $proc ($(($check | Measure-Object).Count) process(es))"
        $stillRunning = $true
    } else {
        Write-Output "  [OK] $proc killed successfully"
    }
}

Write-Output ""
if ($stillRunning) {
    Write-Output "WARNING: Some processes still running. They may not exist or require force-kill."
} else {
    Write-Output "SUCCESS: All IBC/Gateway processes killed."
}
