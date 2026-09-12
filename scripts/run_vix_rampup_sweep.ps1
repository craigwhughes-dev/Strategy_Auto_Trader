#!/usr/bin/env pwsh
# Sweep vix_recovery_window_days x vix_recovery_kelly_mult on crash + real windows.
# Delegates to run_vix_rampup_sweep.py which generates candidates once per window
# (not once per combo), making the sweep ~10x faster.
# Usage: powershell -File scripts/run_vix_rampup_sweep.ps1

Set-Location (Split-Path $PSScriptRoot)
uv run python scripts/run_vix_rampup_sweep.py @args
