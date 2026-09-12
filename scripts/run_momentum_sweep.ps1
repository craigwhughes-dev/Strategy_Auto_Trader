#!/usr/bin/env pwsh
# Sweep momentum_weight x momentum_lookback_days for top-K ticker scoring.
# Generates candidates ONCE per window, re-runs filter+arbitrate per combo.
# Usage: powershell -File scripts/run_momentum_sweep.ps1

Set-Location (Split-Path $PSScriptRoot)
uv run python scripts/run_momentum_sweep.py @args
