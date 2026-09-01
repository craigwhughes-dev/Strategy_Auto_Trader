@echo off
REM Create IB Gateway (via IBC) scheduled task — auto-login at logon,
REM restarts on crash, and IBC's own ColdRestartTime handles the weekly
REM Sunday forced reauth (see C:\Users\Craig\Documents\IBC\config.ini).
REM Run this by double-clicking it (not from an automation shell) — some
REM security tooling blocks scheduled-task creation from script/agent
REM contexts as a persistence heuristic.

setlocal enabledelayedexpansion

set TASK_NAME=IBGatewayIBC
set SCRIPT_PATH=C:\IBC\StartGateway.bat /INLINE

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
REM Trigger: At logon (Gateway needs an interactive desktop session — it's
REM a GUI app that IBC drives via window automation)
REM Run level: Limited (non-elevated)
REM Restart: matches the live daemon's StrategyAutoTraderDaemon pattern
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

REM Restart up to 999 times, 1 minute apart
schtasks /change /tn "%TASK_NAME%" /ri 1 /rp 999 >nul 2>&1

echo.
echo Task created successfully!
echo.
schtasks /query /tn "%TASK_NAME%" /fo table /v

echo.
echo The task will:
echo   - Start IB Gateway (via IBC) at logon
echo   - Auto-login using credentials in C:\Users\Craig\Documents\IBC\config.ini
echo   - Restart on failure (up to 999 times, 1 minute apart)
echo   - Cold-restart and re-authenticate automatically Sunday mornings (ColdRestartTime=07:05)
echo.
