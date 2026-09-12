#!/usr/bin/env pwsh
# Sweep score_lookback_days for top-K ticker scoring on 2023 + real windows.
# Generates candidates ONCE per window, re-runs arbitrate() per lookback value.
# Usage: powershell -File scripts/run_score_lookback_sweep.ps1

Set-Location (Split-Path $PSScriptRoot)
uv run python scripts/run_score_lookback_sweep.py @args
