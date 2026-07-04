@echo off
REM Create Strategy Auto-Trader Daemon scheduled task
REM Run this in Command Prompt as Administrator

setlocal enabledelayedexpansion

set TASK_NAME=Strategy Auto-Trader Daemon
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
REM Run level: Highest (elevated)
REM Restart: If task fails, restart up to 3 times with 1 minute interval
schtasks /create ^
    /tn "%TASK_NAME%" ^
    /tr "%SCRIPT_PATH%" ^
    /sc onlogon ^
    /rl highest ^
    /f

if %errorlevel% neq 0 (
    echo ERROR: Failed to create task
    exit /b 1
)

REM Set additional options
schtasks /change /tn "%TASK_NAME%" /ri 1 /rp 3 >nul 2>&1

echo.
echo Task created successfully!
echo.
echo Task details:
schtasks /query /tn "%TASK_NAME%" /fo table /v

echo.
echo The task will:
echo   - Start at logon
echo   - Run with elevated privileges
echo   - Restart on failure (up to 3 times, 1 minute apart)
echo   - Not start multiple instances
echo.
