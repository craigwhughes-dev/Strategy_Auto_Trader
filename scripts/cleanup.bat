@echo off
cd /d "%~dp0\.."
echo Cleanup: logs, old data dirs (>7d), caches
echo.
uv run python scripts/cleanup.py --execute
pause
