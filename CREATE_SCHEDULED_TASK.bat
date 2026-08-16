@echo off
REM Create Strategy Auto-Trader Daemon scheduled task
REM Admin not required — see CREATE_SCHEDULED_TASK.ps1 for why this task
REM must NOT run elevated (elevation makes Task Scheduler launch the real
REM daemon process outside the Job Object it uses to kill descendants, so
REM "End Task" only kills the outer shell and orphans the daemon underneath).

setlocal enabledelayedexpansion

set TASK_NAME=StrategyAutoTraderDaemon
set SCRIPT_PATH=C:\Users\Craig\.claude\skills\Strategy_Auto_Trader\run_daemon.bat
set WORK_DIR=C:\Users\Craig\.claude\skills\Strategy_Auto_Trader

echo Creating scheduled task: %TASK_NAME%
echo Script: %SCRIPT_PATH%
echo.

REM Delete old task if it exists
schtasks /query /tn "%TASK_NAME%" >nul 2>&1
if %errorlevel% equ 0 (
    echo Task already exists. Removing old version...
    schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1
    timeout /t 1 >nul
)

REM Create new task
REM Trigger: At logon
REM Run level: Limited (non-elevated — not needed and breaks clean shutdown)
REM Restart: matches the live daemon's actual auto-restart behaviour
schtasks /create ^
    /tn "%TASK_NAME%" ^
    /tr "%SCRIPT_PATH%" ^
    /sc onlogon ^
    /rl limited ^
    /f

if %errorlevel% neq 0 (
    echo ERROR: Failed to create task
    exit /b 1
)

REM Set additional options: restart up to 999 times, 1 minute apart
schtasks /change /tn "%TASK_NAME%" /ri 1 /rp 999 >nul 2>&1

echo.
echo Task created successfully!
echo.
echo Task details:
schtasks /query /tn "%TASK_NAME%" /fo table /v

echo.
echo The task will:
echo   - Start at logon
echo   - Run non-elevated (no admin rights needed for this daemon)
echo   - Restart on failure (up to 999 times, 1 minute apart)
echo.
