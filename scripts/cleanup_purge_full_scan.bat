@echo off
cd /d "%~dp0\.."
echo Cleanup: logs, old data dirs (>7d), caches, AND reports/full_scan (32GB)
echo.
uv run python scripts/cleanup.py --execute --purge-full-scan
pause
