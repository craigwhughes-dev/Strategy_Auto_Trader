# Create Strategy Auto-Trader Daemon scheduled task
# Run this in PowerShell as Administrator

$taskName = "Strategy Auto-Trader Daemon"
$scriptPath = "C:\Users\Craig\.claude\skills\Strategy_Auto_Trader\run_daemon.bat"

Write-Output "Creating scheduled task: $taskName"
Write-Output "Script: $scriptPath"
Write-Output ""

# Check if task already exists
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Output "Task already exists. Removing old version..."
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Start-Sleep -Seconds 1
}

# Create task trigger (at logon)
$trigger = New-ScheduledTaskTrigger -AtLogon

# Create task action
$action = New-ScheduledTaskAction -Execute $scriptPath -WorkingDirectory "C:\Users\Craig\.claude\skills\Strategy_Auto_Trader"

# Create task settings with restart on failure
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0) `
    -AllowStartIfOnBatteries:$true `
    -DontStopIfGoingOnBatteries:$true `
    -StartWhenAvailable:$true `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew

# Register the task
Register-ScheduledTask `
    -TaskName $taskName `
    -Trigger $trigger `
    -Action $action `
    -Settings $settings `
    -RunLevel Highest `
    -Force

Write-Output ""
Write-Output "✓ Task created successfully"
Write-Output ""
Write-Output "Task details:"
Get-ScheduledTask -TaskName $taskName | Format-List TaskName, State
Write-Output ""
Write-Output "The task will:"
Write-Output "  - Start at logon"
Write-Output "  - Run with elevated privileges"
Write-Output "  - Restart on failure (up to 3 times, 1 minute apart)"
Write-Output "  - Not start multiple instances"
