#!/usr/bin/env pwsh
# Sweep --vol-window across {126, 252, 378, 504, 756} on crash + real windows.
# Usage: powershell -File scripts/run_vol_window_sweep.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot)

$windows = @(126, 252, 378, 504, 756)
$journalDir = "data\journals"

# --- Crash window (synthetic 2008) ---
Write-Host "`n=== CRASH WINDOW (synthetic 2008-01-01 to 2009-07-31) ===" -ForegroundColor Cyan

foreach ($w in $windows) {
    $label = "vol_window_${w}_crash"
    Write-Host "  Running vol_window=$w (crash)..." -ForegroundColor Yellow
    uv run python -m Strategy_Auto_Trader.markov_cli.live_sim `
        --universe --strategies optimised_new `
        --initial-cash 100000 --top-k 70 --workers 4 `
        --vol-window $w `
        --start-date 2008-01-01 `
        --synthetic-data-dir data_synthetic/hourly `
        --synthetic-end-date 2009-07-31 `
        --journal "$journalDir\${label}.csv" `
        --position-summary "$journalDir\${label}_equity.csv"
    if ($LASTEXITCODE -ne 0) { Write-Warning "  vol_window=$w crash run FAILED (exit $LASTEXITCODE)" }
    else { Write-Host "  Done: $label" -ForegroundColor Green }
}

# --- Real/normal window ---
Write-Host "`n=== REAL WINDOW (ibkr, start-date 2024-01-01) ===" -ForegroundColor Cyan

foreach ($w in $windows) {
    $label = "vol_window_${w}_real"
    Write-Host "  Running vol_window=$w (real)..." -ForegroundColor Yellow
    uv run python -m Strategy_Auto_Trader.markov_cli.live_sim `
        --universe --strategies optimised_new `
        --initial-cash 100000 --top-k 70 --workers 4 `
        --vol-window $w `
        --start-date 2024-01-01 `
        --source ibkr `
        --journal "$journalDir\${label}.csv" `
        --position-summary "$journalDir\${label}_equity.csv"
    if ($LASTEXITCODE -ne 0) { Write-Warning "  vol_window=$w real run FAILED (exit $LASTEXITCODE)" }
    else { Write-Host "  Done: $label" -ForegroundColor Green }
}

Write-Host "`nAll done. Run: uv run python scripts/analyze_vol_window_sweep.py" -ForegroundColor Cyan
