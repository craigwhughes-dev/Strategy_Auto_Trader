# Task Scheduler Setup for Strategy Auto-Trader Daemon

The daemon must be registered as a Windows scheduled task to run automatically at logon with crash recovery.

## Method 1: PowerShell (Recommended)

**Prerequisites**: PowerShell 5.0+ running as Administrator

1. Right-click PowerShell and select "Run as administrator"
2. Navigate to the project root:
   ```powershell
   cd C:\Users\Craig\.claude\skills\Strategy_Auto_Trader
   ```
3. Run the setup script:
   ```powershell
   .\CREATE_SCHEDULED_TASK.ps1
   ```

## Method 2: Batch Script (Command Prompt)

**Prerequisites**: Command Prompt running as Administrator

1. Right-click Command Prompt and select "Run as administrator"
2. Navigate to the project root:
   ```cmd
   cd C:\Users\Craig\.claude\skills\Strategy_Auto_Trader
   ```
3. Run the setup script:
   ```cmd
   CREATE_SCHEDULED_TASK.bat
   ```

## Method 3: Manual Task Scheduler GUI

1. Open **Task Scheduler** (search in Start menu)
2. Click **Create Basic Task** in the right panel
3. Name: `Strategy Auto-Trader Daemon`
4. Description: `Continuous automated paper trading daemon`
5. Trigger: **At logon**
6. Action: **Start a program**
   - Program: `C:\Users\Craig\.claude\skills\Strategy_Auto_Trader\run_daemon.bat`
   - Start in: `C:\Users\Craig\.claude\skills\Strategy_Auto_Trader`
7. When finished, edit the task to add advanced settings:
   - Right-click the task → **Properties**
   - **General** tab:
     - Check "Run with highest privileges"
     - Select your user account from the dropdown
   - **Triggers** tab:
     - Edit the "At logon" trigger
     - Check "Enabled"
     - Check "Repeat task every: 1 minute" for 3 repetitions if it fails (optional but recommended)
   - **Settings** tab:
     - Check "Allow task to be run on demand"
     - Check "Run task as soon as possible after scheduled time if missed"
     - Check "If the running task does not end when requested, force it to stop"
     - Set "If the task is already running on a schedule, do not start a new instance"

## Verify the Task

Check that the task was created:

**PowerShell**:
```powershell
Get-ScheduledTask -TaskName "Strategy Auto-Trader Daemon"
```

**Command Prompt**:
```cmd
schtasks /query /tn "Strategy Auto-Trader Daemon" /fo table /v
```

**Expected output**: State = Ready, Next Run Time = At logon

## Testing

After the task is created:

1. **Dry-run test** — Start the task manually in Task Scheduler:
   - Right-click **Strategy Auto-Trader Daemon** → **Run**
   - Check that `logs/daemon_<date>.log` is created and shows activity
   - Verify `state/daemon_state.json` exists and is updated

2. **Verify logs** — Open `logs/daemon_<date>.log` and look for:
   ```
   ====================================================================
   Live daemon starting
   ====================================================================
   Using NullBroker (dry run mode)
   Broker connected
   Entering main loop
   ```

3. **Monitor overnight screening** — At 02:00 UK time, check logs for:
   ```
   Running overnight scope screening...
   Overnight scope screening complete
   ```

## Troubleshooting

### Task fails with "Access Denied"
- Run Task Scheduler as Administrator
- Ensure "Run with highest privileges" is checked in task Properties

### Task doesn't start at logon
- Verify "Enabled" is checked on the trigger
- Restart your computer to test

### Daemon logs are empty
- Check `logs/daemon_<date>.log` exists (date is today's ISO format, e.g., `daemon_2026-07-03.log`)
- Verify `run_daemon.bat` can execute manually:
  ```cmd
  C:\Users\Craig\.claude\skills\Strategy_Auto_Trader\run_daemon.bat
  ```

### Daemon starts multiple instances
- In Task Scheduler, ensure "If the task is already running on a schedule, do not start a new instance" is checked

## Uninstall (if needed)

**PowerShell** (as Administrator):
```powershell
Unregister-ScheduledTask -TaskName "Strategy Auto-Trader Daemon" -Confirm:$false
```

**Command Prompt** (as Administrator):
```cmd
schtasks /delete /tn "Strategy Auto-Trader Daemon" /f
```

## Next Steps

After the task is created and tested in dry-run mode:

1. Flip `execution.dry_run` to `false` in `config/overnight_strategy.json`
2. Start TWS or IB Gateway on the paper trading port (7497)
3. Let the daemon run — it will place real paper orders during trading hours
4. Monitor logs and `state/execution_state.json` for position tracking
