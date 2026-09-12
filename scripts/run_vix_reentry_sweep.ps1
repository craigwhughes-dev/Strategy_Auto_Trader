#!/usr/bin/env pwsh
# Sweep vix_gate_allow_reentry {False, True} on crash + real windows.
# Edits optimised_new.py between runs, restores at end.
# Usage: powershell -File scripts/run_vix_reentry_sweep.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot)

$strategyFile = "Strategy_Auto_Trader\strategy\optimised_new.py"
$originalContent = Get-Content $strategyFile -Raw
$journalDir = "data\journals"

function Restore-Strategy {
    Set-Content $strategyFile $originalContent -NoNewline
}

try {
    foreach ($window in @("crash", "real")) {
        if ($window -eq "crash") {
            Write-Host "`n=== CRASH WINDOW (synthetic 2008-01-01 to 2009-07-31) ===" -ForegroundColor Cyan
            $extraArgs = @("--start-date", "2008-01-01", "--synthetic-data-dir", "data_synthetic/hourly", "--synthetic-end-date", "2009-07-31")
        } else {
            Write-Host "`n=== REAL WINDOW (ibkr, start-date 2024-01-01) ===" -ForegroundColor Cyan
            $extraArgs = @("--start-date", "2024-01-01", "--source", "ibkr")
        }

        foreach ($reentry in @("False", "True")) {
            $label = "vix_reentry_${reentry}_${window}"
            Write-Host "  Running vix_gate_allow_reentry=$reentry ($window)..." -ForegroundColor Yellow

            $content = Get-Content $strategyFile -Raw
            $content = $content -replace "vix_gate_allow_reentry: bool = \w+", "vix_gate_allow_reentry: bool = $reentry"
            Set-Content $strategyFile $content -NoNewline

            $cmdArgs = @(
                "--universe", "--strategies", "optimised_new",
                "--initial-cash", "100000", "--top-k", "70", "--workers", "4",
                "--journal", "$journalDir\${label}.csv",
                "--position-summary", "$journalDir\${label}_equity.csv"
            ) + $extraArgs

            uv run python -m Strategy_Auto_Trader.markov_cli.live_sim @cmdArgs
            if ($LASTEXITCODE -ne 0) { Write-Warning "  $label FAILED (exit $LASTEXITCODE)" }
            else { Write-Host "  Done: $label" -ForegroundColor Green }
        }
    }
} finally {
    Restore-Strategy
    Write-Host "`nStrategy file restored." -ForegroundColor Cyan
    Write-Host "Run: uv run python scripts/analyze_vix_reentry_sweep.py" -ForegroundColor Cyan
}
