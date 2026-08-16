# Create Strategy Auto-Trader Daemon scheduled task
# Run this in PowerShell (admin not required — see note below)
#
# Not elevated, and points straight at uv.exe (no cmd.exe /c "... && ..."
# wrapper): both matter for clean shutdown. The daemon needs no admin
# rights (all file I/O is under this repo, IBKR connection is a plain
# localhost socket on a non-privileged port) — RunLevel=Highest was
# previously set for no functional reason, and on Windows it makes Task
# Scheduler launch the elevated process outside the Job Object it uses to
# track/kill the task's descendants. Combined with the old cmd.exe wrapper
# (cmd.exe -> uv.exe -> python.exe, 3 process generations), ending the task
# only killed the outer cmd.exe — uv.exe and the actual daemon python.exe
# survived as orphans, still holding open file handles (e.g. the daemon's
# own log file). Non-elevated + a direct 2-hop action (uv.exe -> python.exe)
# stays inside Task Scheduler's normal Job Object, so "End Task" now
# terminates the whole tree.

$taskName = "StrategyAutoTraderDaemon"
$uvPath = "$env:USERPROFILE\.local\bin\uv.exe"
$workDir = "C:\Users\Craig\.claude\skills\Strategy_Auto_Trader"

Write-Output "Creating scheduled task: $taskName"
Write-Output "Action: $uvPath run python -m Strategy_Auto_Trader.markov_cli.live_daemon --takeover"
Write-Output ""

# Check if task already exists
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Output "Task already exists. Removing old version..."
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Start-Sleep -Seconds 1
}

# Registered via raw XML, not the New-ScheduledTask* cmdlets: the
# ScheduledTasks module's -MultipleInstances parameter only exposes
# Parallel/Queue/IgnoreNew — StopExisting exists in the underlying schema
# (and is what the previously-working task used) but isn't reachable
# through that cmdlet's enum. XML has no such gap.
$userId = "$env:COMPUTERNAME\$env:USERNAME"
[xml]$taskXml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Automated live trading daemon with auto-restart</Description>
  </RegistrationInfo>
  <Principals>
    <Principal id="Author">
      <UserId>$userId</UserId>
      <LogonType>S4U</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <MultipleInstancesPolicy>StopExisting</MultipleInstancesPolicy>
    <RestartOnFailure>
      <Count>999</Count>
      <Interval>PT1M</Interval>
    </RestartOnFailure>
    <StartWhenAvailable>true</StartWhenAvailable>
  </Settings>
  <Triggers>
    <LogonTrigger />
  </Triggers>
  <Actions Context="Author">
    <Exec>
      <Command>$uvPath</Command>
      <Arguments>run python -m Strategy_Auto_Trader.markov_cli.live_daemon --takeover</Arguments>
      <WorkingDirectory>$workDir</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@

Register-ScheduledTask -TaskName $taskName -Xml $taskXml.OuterXml -Force

Write-Output ""
Write-Output "Task created successfully"
Write-Output ""
Write-Output "Task details:"
Get-ScheduledTask -TaskName $taskName | Format-List TaskName, State
Write-Output ""
Write-Output "The task will:"
Write-Output "  - Start at logon"
Write-Output "  - Run non-elevated (no admin rights needed for this daemon)"
Write-Output "  - Restart on failure (up to 999 times, 1 minute apart)"
Write-Output "  - Replace any still-running instance instead of stacking (--takeover)"
